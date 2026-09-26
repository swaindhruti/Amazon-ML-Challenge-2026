"""
Per-country candidate generation + feature building, shared by training,
validation and inference.

Before this module existed the same ~60 lines (clean text -> build inverted
index -> optional embeddings -> batch -> merge candidates -> features) were
copy-pasted into the inference loop and the validation loop, and training used
a third, different way of producing pairs (random negatives). That last
difference is what this refactor removes: a model trained on pairs produced by
ANY other procedure than the one it will later score is being evaluated on a
distribution it never saw. Now all three call the same functions, so the
classifier is trained on exactly the kind of hard, blocking-surviving pairs
it must separate at inference time.
"""
import gc
import time
from typing import Iterator, Optional, Tuple

import numpy as np
import pandas as pd

from src.preprocessing import prepare_text_chunk
from src.candidate_generation import (
    CompactInvertedIndex, merge_candidate_pairs, rerank_score, top_k_per_query,
)
from src.features import build_batch_features, _sparse_pair_cosine
from src.embeddings import encode_texts, EmbeddingCandidateIndex, EmbeddingRows
from src.parallel import map_chunks, split_ranges, resolve_n_jobs


def _prep_task(task):
    return prepare_text_chunk(*task)


def prepare_text_arrays(df: pd.DataFrame) -> dict:
    """
    Cleans a dataframe's raw name/address columns into every derived text
    column, split across the process pool (see src/parallel.py). Returns a
    dict of parallel lists keyed by column name.
    """
    names = df['business_name'].values
    addrs = df['business_address'].values
    countries = df['country'].values
    ranges = split_ranges(len(df), min_chunk=100_000)
    tasks = [(names[s:e], addrs[s:e], countries[s:e]) for s, e in ranges] or [(names, addrs, countries)]
    parts = map_chunks(_prep_task, tasks)
    keys = ['clean', 'norm', 'stripped', 'compact', 'addr_clean', 'addr_norm', 'nums', 'native', 'state']
    return {k: [x for p in parts for x in p[i]] for i, k in enumerate(keys)}


def _proc_frame(df_raw: pd.DataFrame, t: dict) -> pd.DataFrame:
    return pd.DataFrame({
        'entity_id': df_raw['entity_id'].values,
        'business_name_norm': t['norm'],
        'business_name_stripped': t['stripped'],
        'business_name_compact': t['compact'],
        'business_address_norm': t['addr_norm'],
        'address_numbers': t['nums'],
        'state': t['state'],
    })


def _fit_tfidf(target_texts, query_texts):
    """
    Word-level TF-IDF fitted on the TARGET pool (that is the population whose
    word frequencies decide what counts as 'common'), applied to the queries.
    Returns (Xq, Xt) L2-normalised sparse matrices, or None when the pool has
    no usable vocabulary.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(token_pattern=r"\S+", lowercase=False, sublinear_tf=True,
                          dtype=np.float32, min_df=1)
    try:
        Xt = vec.fit_transform(target_texts)
    except ValueError:  # "empty vocabulary" -- every target text blank
        return None
    return vec.transform(query_texts), Xt


class CountryData:
    """Everything built once per country and reused across that country's batches."""

    def __init__(self):
        self.country = None
        self.df_target_proc = None
        self.df_s1_proc = None
        self.t_ids = None
        self.index = None
        self.q_clean = self.q_addr_clean = self.q_nums = None
        self.tfidf_name = None   # (Xq, Xt) or None
        self.tfidf_addr = None
        self.embeddings = None   # EmbeddingRows or None
        self.emb_index = None
        self.s1_raw_names = None

    def free(self):
        for k in list(self.__dict__):
            setattr(self, k, None)
        gc.collect()


def build_country_data(country, s1_country: pd.DataFrame, s2_s3_country: pd.DataFrame,
                       args, embed_model=None) -> CountryData:
    cd = CountryData()
    cd.country = country

    # -- target pool: text arrays, lexical index, TF-IDF --
    t = prepare_text_arrays(s2_s3_country)
    cd.df_target_proc = _proc_frame(s2_s3_country, t)
    cd.t_ids = s2_s3_country['entity_id'].values

    t0 = time.time()
    cd.index = CompactInvertedIndex(max_block_size=10000, max_candidates=args.top_k)
    cd.index.build(t['clean'], t['addr_clean'], t['nums'])
    print(f"Built inverted index for '{country}' in {time.time() - t0:.2f} s | Keys: {len(cd.index.index)}")

    # -- S1 side --
    q = prepare_text_arrays(s1_country)
    cd.df_s1_proc = _proc_frame(s1_country, q)
    cd.q_clean, cd.q_addr_clean, cd.q_nums = q['clean'], q['addr_clean'], q['nums']
    cd.s1_raw_names = s1_country['business_name'].values

    t0 = time.time()
    cd.tfidf_name = _fit_tfidf(t['stripped'], q['stripped'])
    cd.tfidf_addr = _fit_tfidf(t['addr_norm'], q['addr_norm'])
    print(f"Fitted TF-IDF for '{country}' in {time.time() - t0:.2f} s")

    # -- optional embeddings (native-script targets only) --
    wanted = getattr(args, 'embed_countries', 'india')
    embed_here = embed_model is not None and (wanted == 'all' or country in
                                              [c.strip() for c in wanted.split(',')])
    if embed_here:
        native_rows = np.nonzero(np.asarray(t['native']))[0]
        if len(native_rows) > 0:
            t0 = time.time()
            raw_targets = s2_s3_country['business_name'].values[native_rows]
            emb = encode_texts(embed_model, raw_targets)
            cd.embeddings = EmbeddingRows(len(s2_s3_country), native_rows, emb)
            cd.emb_index = EmbeddingCandidateIndex(emb, row_ids=native_rows)
            print(f"Encoded + indexed {len(native_rows)} native-script targets for '{country}' "
                  f"in {time.time() - t0:.2f} s")
            del emb, raw_targets

    del t, q
    gc.collect()
    return cd


def iter_candidate_batches(cd: CountryData, args, embed_model=None) -> Iterator[Tuple]:
    """
    Yields (b_start, b_end, sub_s1_df, q_idx, t_idx, batch_query_embeddings)
    per S1 batch. q_idx indexes into the batch, t_idx into the country's
    target pool. Lexical candidates are unioned with (optional) semantic ones.
    """
    n = len(cd.df_s1_proc)
    for b_start in range(0, n, args.batch_size):
        b_end = min(b_start + args.batch_size, n)
        pool_k = max(getattr(args, 'pool_k', args.top_k), args.top_k)
        q_idx, t_idx, key_w = cd.index.query_candidates(
            cd.q_clean[b_start:b_end], cd.q_addr_clean[b_start:b_end], cd.q_nums[b_start:b_end],
            max_candidates=pool_k, return_scores=True,
        )
        if pool_k > args.top_k and len(q_idx) > 0:
            # Larger pool by key overlap, then the top_k by cheap TF-IDF similarity
            # (see candidate_generation.rerank_score for why).
            zeros = np.zeros(len(q_idx), dtype=np.float32)
            name_cos = zeros if cd.tfidf_name is None else _sparse_pair_cosine(
                cd.tfidf_name[0][b_start:b_end], cd.tfidf_name[1], q_idx, t_idx)
            addr_cos = zeros if cd.tfidf_addr is None else _sparse_pair_cosine(
                cd.tfidf_addr[0][b_start:b_end], cd.tfidf_addr[1], q_idx, t_idx)
            keep = top_k_per_query(q_idx, rerank_score(name_cos, addr_cos, key_w), args.top_k)
            q_idx, t_idx = q_idx[keep], t_idx[keep]
        batch_emb = None
        if cd.emb_index is not None:
            batch_emb = encode_texts(embed_model, cd.s1_raw_names[b_start:b_end])
            eq, et = cd.emb_index.query(batch_emb, k=args.emb_top_k)
            q_idx, t_idx = merge_candidate_pairs(q_idx, t_idx, eq, et)
        sub = cd.df_s1_proc.iloc[b_start:b_end].reset_index(drop=True)
        yield b_start, b_end, sub, q_idx, t_idx, batch_emb


def batch_features(cd: CountryData, b_start: int, b_end: int, sub_s1_df: pd.DataFrame,
                   q_idx, t_idx, batch_emb, n_jobs=None) -> pd.DataFrame:
    tfidf = {}
    if cd.tfidf_name is not None:
        tfidf['name'] = (cd.tfidf_name[0][b_start:b_end], cd.tfidf_name[1])
    if cd.tfidf_addr is not None:
        tfidf['addr'] = (cd.tfidf_addr[0][b_start:b_end], cd.tfidf_addr[1])
    return build_batch_features(
        sub_s1_df, cd.df_target_proc, q_idx, t_idx,
        query_embeddings=batch_emb,
        target_embeddings=cd.embeddings if batch_emb is not None else None,
        tfidf=tfidf or None,
        n_jobs=resolve_n_jobs(n_jobs),
    )
