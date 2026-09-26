import pandas as pd
import numpy as np
from rapidfuzz import fuzz, distance
from typing import Sequence

# Hand-engineered string/structural similarity features rather than learned
# embeddings: the competition prohibits external data/API lookups, and
# RapidFuzz's C implementation is fast enough (500k+ pairs/sec) to run over
# every candidate pair within the RAM/time budget without needing a GPU or a
# separate embedding index.


def compute_jaccard(s1: str, s2: str) -> float:
    if not s1 or not s2:
        return 0.0
    set1 = set(s1.split())
    set2 = set(s2.split())
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return float(intersection / union) if union > 0 else 0.0


def get_ngram_cos_sim(s1: str, s2: str, n: int = 3) -> float:
    if not s1 or not s2:
        return 0.0

    len1 = len(s1)
    len2 = len(s2)
    if len1 < n or len2 < n:
        return float(s1 == s2)

    set1 = {s1[i:i+n] for i in range(len1 - n + 1)}
    set2 = {s2[i:i+n] for i in range(len2 - n + 1)}

    if not set1 or not set2:
        return 0.0

    intersection = len(set1.intersection(set2))
    norm = np.sqrt(len(set1)) * np.sqrt(len(set2))
    return float(intersection / norm) if norm > 0 else 0.0


def _common_token_fraction(s1: str, s2: str) -> float:
    """Fraction of tokens in the shorter string that appear in the longer."""
    if not s1 or not s2:
        return 0.0
    t1 = set(s1.split())
    t2 = set(s2.split())
    if not t1 or not t2:
        return 0.0
    shorter, longer = (t1, t2) if len(t1) <= len(t2) else (t2, t1)
    overlap = len(shorter.intersection(longer))
    return overlap / len(shorter) if len(shorter) > 0 else 0.0


def _first_token_match(s1: str, s2: str) -> float:
    """1.0 if the first token matches, 0.0 otherwise."""
    if not s1 or not s2:
        return 0.0
    t1 = s1.split()
    t2 = s2.split()
    if not t1 or not t2:
        return 0.0
    return 1.0 if t1[0] == t2[0] else 0.0


def _char_length_ratio(s1: str, s2: str) -> float:
    """Ratio of shorter to longer string length (character-level)."""
    l1 = len(s1)
    l2 = len(s2)
    if l1 == 0 and l2 == 0:
        return 1.0
    if l1 == 0 or l2 == 0:
        return 0.0
    return min(l1, l2) / max(l1, l2)


def _nums_overlap_score(nums1: str, nums2: str) -> float:
    """
    Computes overlap score between two sets of numerical tokens.
    Returns fraction of matching numbers (Jaccard-style).
    """
    if not nums1 or not nums2:
        return 0.0
    s1 = set(nums1.split())
    s2 = set(nums2.split())
    if not s1 or not s2:
        return 0.0
    intersection = len(s1.intersection(s2))
    union = len(s1.union(s2))
    return intersection / union if union > 0 else 0.0


# --- Multi-core string scoring -------------------------------------------
# RapidFuzz's process.cpdist scores two aligned lists pair-by-pair inside its
# own C++ thread pool (workers=N), so the five fuzzy-name and two fuzzy-address
# scorers run on every core with no Python-level loop and no pickling. Falls
# back to a plain loop on old RapidFuzz builds (identical numbers, one core).
try:
    from rapidfuzz import process as _rf_process
    _HAS_CPDIST = hasattr(_rf_process, 'cpdist')
except Exception:  # pragma: no cover
    _HAS_CPDIST = False


def _pairwise(scorer, a, b, workers: int, scale: float = 1.0) -> np.ndarray:
    if len(a) == 0:
        return np.zeros(0, dtype=np.float32)
    if _HAS_CPDIST:
        out = _rf_process.cpdist(a, b, scorer=scorer, dtype=np.float32, workers=workers)
        return out * scale if scale != 1.0 else out
    return np.array([scorer(x, y) * scale for x, y in zip(a, b)], dtype=np.float32)


def _python_feature_chunk(task):
    """
    The pure-Python per-pair features (set/token maths RapidFuzz doesn't
    cover) for one slice of pairs. Module-level and self-contained so
    src.parallel can run slices on separate cores.
    """
    q_names, t_names, q_stripped, t_stripped, q_addrs, t_addrs, q_nums, t_nums = task
    n = len(q_names)
    out = {k: np.zeros(n, dtype=np.float32) for k in (
        'stripped_exact', 'first_token_match', 'common_token_frac', 'name_length_ratio',
        'name_char_ratio', 'name_token_overlap', 'name_prefix_match', 'addr_jaccard',
        'addr_ngram_sim', 'addr_missing', 'zip_exact', 'nums_overlap', 'street_num_match')}
    for i in range(n):
        n1, n2, s1, s2 = q_names[i], t_names[i], q_stripped[i], t_stripped[i]
        a1, a2, z1, z2 = q_addrs[i], t_addrs[i], q_nums[i], t_nums[i]

        out['stripped_exact'][i] = 1.0 if (s1 and s2 and s1 == s2) else 0.0
        out['first_token_match'][i] = _first_token_match(s1, s2)
        out['common_token_frac'][i] = _common_token_fraction(s1, s2)
        t1, t2 = n1.split(), n2.split()
        l1, l2 = len(t1), len(t2)
        out['name_length_ratio'][i] = min(l1, l2) / max(l1, l2) if max(l1, l2) > 0 else 0.0
        out['name_char_ratio'][i] = _char_length_ratio(n1, n2)
        out['name_token_overlap'][i] = float(len(set(t1).intersection(t2)))

        # Truncation noise ("WINTERS MUNICIPALS OF" vs "winters municipals of
        # sacramento"): one stripped name is a whole-token prefix of the other.
        p1, p2 = s1.split(), s2.split()
        if p1 and p2 and p1 != p2:
            short, long_ = (p1, p2) if len(p1) <= len(p2) else (p2, p1)
            if long_[:len(short)] == short and len(' '.join(short)) >= 4:
                out['name_prefix_match'][i] = 1.0

        out['addr_jaccard'][i] = compute_jaccard(a1, a2) * 100
        out['addr_ngram_sim'][i] = get_ngram_cos_sim(a1, a2) * 100
        # Missing address (some sources leave it blank): flagged explicitly so
        # a trained model can tell "no address evidence" from "addresses differ".
        out['addr_missing'][i] = 1.0 if (not a1 or not a2) else 0.0
        out['zip_exact'][i] = 1.0 if (z1 and z1 == z2) else 0.0
        out['nums_overlap'][i] = _nums_overlap_score(z1, z2)
        f1, f2 = (z1.split()[:1] if z1 else []), (z2.split()[:1] if z2 else [])
        out['street_num_match'][i] = 1.0 if (f1 and f1 == f2) else 0.0
    return out


def _sparse_pair_cosine(mat_q, mat_t, q_idx, t_idx, chunk: int = 400_000) -> np.ndarray:
    """Row-wise dot product of L2-normalised sparse rows for aligned (q, t) index pairs."""
    n = len(q_idx)
    out = np.zeros(n, dtype=np.float32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        prod = mat_q[q_idx[s:e]].multiply(mat_t[t_idx[s:e]])
        out[s:e] = np.asarray(prod.sum(axis=1)).ravel()
    return out


def _resolve_cols(df: pd.DataFrame, candidates, default):
    return next((c for c in candidates if c in df.columns), default)


def compute_feature_arrays(
    df_query: pd.DataFrame,
    df_target: pd.DataFrame,
    query_indices: np.ndarray,
    target_indices: np.ndarray,
    query_embeddings: np.ndarray = None,
    target_embeddings: np.ndarray = None,
    tfidf: dict = None,
    n_jobs: int = 1,
) -> dict:
    """
    Numeric feature columns (no ID columns) for aligned (query, target) index
    pairs -- the shared core of build_batch_features. Fuzzy scorers run on
    RapidFuzz's C++ thread pool (n_jobs threads); the pure-Python token
    features run on the src.parallel process pool when one exists.

    tfidf (optional): {'name': (Xq, Xt), 'addr': (Xq, Xt)} L2-normalised
    sparse TF-IDF matrices aligned with df_query / df_target rows. Their
    row-wise cosine (name_idf_cos / addr_idf_cos) down-weights words that
    are common across the whole pool ("pizza", "road", "services") and
    up-weights rare identifying ones, which plain edit-distance features
    treat all alike -- that's why "Joe's Pizza" vs "Tony's Pizza" can look
    like a match to fuzz.ratio but not here.
    """
    from src.parallel import map_chunks, split_ranges

    q_name_col = _resolve_cols(df_query, ['business_name_norm', 'business_name_clean', 'business_name'], 'business_name')
    t_name_col = _resolve_cols(df_target, ['business_name_norm', 'business_name_clean', 'business_name'], 'business_name')
    q_addr_col = _resolve_cols(df_query, ['business_address_norm', 'business_address_clean', 'business_address'], 'business_address')
    t_addr_col = _resolve_cols(df_target, ['business_address_norm', 'business_address_clean', 'business_address'], 'business_address')

    def gather(df, col, idx, fallback=None):
        if col in df.columns:
            return df[col].values[idx]
        return fallback if fallback is not None else np.array([''] * len(idx), dtype=object)

    q_names = gather(df_query, q_name_col, query_indices)
    t_names = gather(df_target, t_name_col, target_indices)
    q_addrs = gather(df_query, q_addr_col, query_indices)
    t_addrs = gather(df_target, t_addr_col, target_indices)
    q_nums = gather(df_query, 'address_numbers', query_indices)
    t_nums = gather(df_target, 'address_numbers', target_indices)

    if 'business_name_stripped' in df_query.columns and 'business_name_stripped' in df_target.columns:
        q_stripped = df_query['business_name_stripped'].values[query_indices]
        t_stripped = df_target['business_name_stripped'].values[target_indices]
    else:
        q_stripped, t_stripped = q_names, t_names

    if 'business_name_compact' in df_query.columns and 'business_name_compact' in df_target.columns:
        q_compact = df_query['business_name_compact'].values[query_indices]
        t_compact = df_target['business_name_compact'].values[target_indices]
    else:
        from src.preprocessing import compact_name
        q_compact = np.array([compact_name(s) for s in q_stripped], dtype=object)
        t_compact = np.array([compact_name(s) for s in t_stripped], dtype=object)

    workers = max(1, n_jobs)
    feats = {}

    # -- Fuzzy name features: multi-threaded C++ (normalized names) --
    feats['name_lev_ratio'] = _pairwise(fuzz.ratio, q_names, t_names, workers)
    feats['name_jw_dist'] = _pairwise(distance.JaroWinkler.normalized_similarity, q_names, t_names, workers, 100.0)
    feats['name_token_sort'] = _pairwise(fuzz.token_sort_ratio, q_names, t_names, workers)
    feats['name_token_set'] = _pairwise(fuzz.token_set_ratio, q_names, t_names, workers)
    feats['name_partial_ratio'] = _pairwise(fuzz.partial_ratio, q_names, t_names, workers)
    # -- Fuzzy name features on stripped names (legal suffix removed) --
    feats['stripped_jw'] = _pairwise(distance.JaroWinkler.normalized_similarity, q_stripped, t_stripped, workers, 100.0)
    feats['stripped_tsr'] = _pairwise(fuzz.token_sort_ratio, q_stripped, t_stripped, workers)
    # -- Glued/domain-style names: compare with spaces removed --
    feats['name_compact_ratio'] = _pairwise(fuzz.ratio, q_compact, t_compact, workers)
    partial = _pairwise(fuzz.partial_ratio, q_compact, t_compact, workers)
    min_len = np.minimum(np.fromiter((len(s) for s in q_compact), dtype=np.int32, count=len(q_compact)),
                         np.fromiter((len(s) for s in t_compact), dtype=np.int32, count=len(t_compact)))
    # partial_ratio scores 100 for any short string contained in a longer one
    # ("abc" inside "abcdefgh"); only trust it once the shorter side is long enough.
    feats['name_compact_partial'] = np.where(min_len >= 5, partial, 0.0).astype(np.float32)
    # -- Fuzzy address features --
    feats['addr_token_set'] = _pairwise(fuzz.token_set_ratio, q_addrs, t_addrs, workers)
    feats['addr_ratio'] = _pairwise(fuzz.ratio, q_addrs, t_addrs, workers)

    # -- Pure-Python token/structure features: process pool --
    ranges = split_ranges(len(q_names), min_chunk=25_000)
    tasks = [(q_names[s:e], t_names[s:e], q_stripped[s:e], t_stripped[s:e],
              q_addrs[s:e], t_addrs[s:e], q_nums[s:e], t_nums[s:e]) for s, e in ranges]
    for part in _merge_parts(map_chunks(_python_feature_chunk, tasks)).items():
        feats[part[0]] = part[1]

    # -- IDF-weighted cosine --
    if tfidf is not None:
        if 'name' in tfidf:
            feats['name_idf_cos'] = _sparse_pair_cosine(tfidf['name'][0], tfidf['name'][1], query_indices, target_indices)
        if 'addr' in tfidf:
            feats['addr_idf_cos'] = _sparse_pair_cosine(tfidf['addr'][0], tfidf['addr'][1], query_indices, target_indices)

    # -- Semantic (embedding) similarity --
    if query_embeddings is not None and target_embeddings is not None:
        feats['semantic_sim'] = _semantic_sim(query_embeddings, target_embeddings, query_indices, target_indices)

    return feats


def _merge_parts(parts: list) -> dict:
    """Concatenates the per-slice dicts returned by _python_feature_chunk back into full-length arrays."""
    if not parts:
        return {}
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def _semantic_sim(query_embeddings, target_embeddings, query_indices, target_indices,
                  chunk: int = 200_000) -> np.ndarray:
    """
    Cosine similarity of aligned rows (embeddings are L2-normalised, so a dot
    product). target_embeddings is either a dense (N, d) matrix or an
    EmbeddingRows (src/embeddings.py) holding only the encoded subset; pairs
    whose target was never encoded get NaN ("missing" to XGBoost) rather than
    a misleading 0.0. Chunked so a 1.5M-pair batch never materialises two
    multi-GB (pairs x d) gathers at once.
    """
    if hasattr(target_embeddings, 'pair_sim'):
        return target_embeddings.pair_sim(query_embeddings, query_indices, target_indices)
    n = len(query_indices)
    out = np.zeros(n, dtype=np.float32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        out[s:e] = np.einsum('ij,ij->i', query_embeddings[query_indices[s:e]],
                             target_embeddings[target_indices[s:e]])
    return out


def build_batch_features(
    df_query: pd.DataFrame,
    df_target: pd.DataFrame,
    query_indices: np.ndarray,
    target_indices: np.ndarray,
    query_embeddings: np.ndarray = None,
    target_embeddings: np.ndarray = None,
    tfidf: dict = None,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """
    Computes pairwise similarity features between df_query and df_target using
    integer index arrays (see compute_feature_arrays for the feature families
    and the multi-core strategy). Adds the two ID columns the scoring/output
    code keys on. Optional inputs (embeddings, tfidf) only add columns when
    provided; nothing is zero-filled, so a downstream model can tell "not
    computed" from "computed and low".
    """
    if len(query_indices) == 0:
        return pd.DataFrame()

    feats = compute_feature_arrays(
        df_query, df_target, query_indices, target_indices,
        query_embeddings=query_embeddings, target_embeddings=target_embeddings,
        tfidf=tfidf, n_jobs=n_jobs,
    )
    result = {
        'source1_entity_id': df_query['entity_id'].values[query_indices],
        'candidate_entity_id': df_target['entity_id'].values[target_indices],
    }
    result.update(feats)
    return pd.DataFrame(result)


# Feature column list used by model -- single source of truth.
# The first 17 are the original set (kept in order so older checkpoints that
# were trained on exactly those still load: model.py selects by the
# checkpoint's own feature names); the rest are the address-aware /
# noise-robust additions.
FEATURE_COLS = [
    'name_lev_ratio', 'name_jw_dist', 'name_token_sort', 'name_token_set',
    'name_partial_ratio',
    'stripped_jw', 'stripped_tsr', 'stripped_exact',
    'first_token_match', 'common_token_frac',
    'name_length_ratio', 'name_char_ratio', 'name_token_overlap',
    'addr_jaccard', 'addr_ngram_sim', 'zip_exact', 'nums_overlap',
    'addr_token_set', 'addr_ratio', 'addr_missing', 'street_num_match',
    'name_compact_ratio', 'name_compact_partial', 'name_prefix_match',
]

# Only present when the corresponding input was supplied (see
# compute_feature_arrays); used when available.
OPTIONAL_FEATURE_COLS = ['name_idf_cos', 'addr_idf_cos', 'semantic_sim']


def build_feature_dataset(
    df_candidates: pd.DataFrame,
    df_s1: pd.DataFrame,
    df_s2_s3: pd.DataFrame
) -> pd.DataFrame:
    """
    Backward-compatible entry point that converts candidate pairs into features.
    """
    if df_candidates.empty:
        return pd.DataFrame()

    s1_id_to_idx = {eid: idx for idx, eid in enumerate(df_s1['entity_id'].values)}
    t_id_to_idx = {eid: idx for idx, eid in enumerate(df_s2_s3['entity_id'].values)}

    q_indices = []
    t_indices = []

    # iterrows() here is fine: this path parses a candidate_pairs.tsv-style
    # DataFrame (comma-joined ID strings needing per-row splitting), not the
    # hot per-pair scoring loop -- that one is build_batch_features() above,
    # which stays vectorized/array-indexed for the 10M+ pair scale.
    for _, row in df_candidates.iterrows():
        s1_id = row['source1_entity_id']
        cands = row['candidate_entity_ids'].split(',') if row['candidate_entity_ids'] else []
        if s1_id in s1_id_to_idx:
            q_i = s1_id_to_idx[s1_id]
            for cand in cands:
                cand = cand.strip()
                if cand and cand in t_id_to_idx:
                    q_indices.append(q_i)
                    t_indices.append(t_id_to_idx[cand])

    if not q_indices:
        return pd.DataFrame()

    return build_batch_features(
        df_s1, df_s2_s3,
        np.array(q_indices, dtype=np.int32),
        np.array(t_indices, dtype=np.uint32)
    )
