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


def build_batch_features(
    df_query: pd.DataFrame,
    df_target: pd.DataFrame,
    query_indices: np.ndarray,
    target_indices: np.ndarray,
    query_embeddings: np.ndarray = None,
    target_embeddings: np.ndarray = None,
) -> pd.DataFrame:
    """
    Computes pairwise similarity features between df_query and df_target using integer index arrays.
    Enhanced with 16 features for high-precision entity resolution.

    query_embeddings/target_embeddings (optional): L2-normalized sentence
    embeddings (see src/embeddings.py), aligned 1:1 with df_query/df_target
    rows (NOT with query_indices/target_indices -- those still index into
    the full arrays the same way they index into df_query/df_target). When
    both are provided, adds a semantic_sim column (cosine similarity, i.e.
    a plain dot product on normalized vectors) -- the one feature here that
    can catch a true match across scripts/languages the other 17 lexical
    features structurally cannot, since they all operate on literal
    characters. Omitted (not just zero-filled) when not provided, so
    model.py's EntityMatchingModel only uses it when it actually has a
    real value to offer, and prepare_X gates it on use_transformer
    separately.
    """
    if len(query_indices) == 0:
        return pd.DataFrame()

    # --- Resolve column names ---
    # Prefers the most-normalized column available but falls back gracefully
    # (_norm -> _clean -> raw) since this function is called from several
    # places with dataframes prepared to different degrees -- training builds
    # a fully-processed frame, but a caller could pass raw source data.
    q_name_col = next(
        (c for c in ['business_name_norm', 'business_name_clean', 'business_name'] if c in df_query.columns),
        'business_name'
    )
    t_name_col = next(
        (c for c in ['business_name_norm', 'business_name_clean', 'business_name'] if c in df_target.columns),
        'business_name'
    )
    q_addr_col = next(
        (c for c in ['business_address_norm', 'business_address_clean', 'business_address'] if c in df_query.columns),
        'business_address'
    )
    t_addr_col = next(
        (c for c in ['business_address_norm', 'business_address_clean', 'business_address'] if c in df_target.columns),
        'business_address'
    )

    has_q_stripped = 'business_name_stripped' in df_query.columns
    has_t_stripped = 'business_name_stripped' in df_target.columns
    has_q_nums = 'address_numbers' in df_query.columns
    has_t_nums = 'address_numbers' in df_target.columns

    # --- Extract arrays ---
    q_names = df_query[q_name_col].values[query_indices]
    t_names = df_target[t_name_col].values[target_indices]
    q_addrs = df_query[q_addr_col].values[query_indices]
    t_addrs = df_target[t_addr_col].values[target_indices]
    q_nums = df_query['address_numbers'].values[query_indices] if has_q_nums else [""] * len(query_indices)
    t_nums = df_target['address_numbers'].values[target_indices] if has_t_nums else [""] * len(target_indices)
    q_ids = df_query['entity_id'].values[query_indices]
    t_ids = df_target['entity_id'].values[target_indices]

    # Stripped names (legal suffix removed) — use norm if stripped unavailable
    if has_q_stripped and has_t_stripped:
        q_stripped = df_query['business_name_stripped'].values[query_indices]
        t_stripped = df_target['business_name_stripped'].values[target_indices]
    else:
        q_stripped = q_names
        t_stripped = t_names

    # ── NAME FEATURES (on normalized names) ──
    name_lev_ratios = [fuzz.ratio(n1, n2) for n1, n2 in zip(q_names, t_names)]
    name_jw_dists = [distance.JaroWinkler.normalized_similarity(n1, n2) * 100 for n1, n2 in zip(q_names, t_names)]
    name_token_sorts = [fuzz.token_sort_ratio(n1, n2) for n1, n2 in zip(q_names, t_names)]
    name_token_sets = [fuzz.token_set_ratio(n1, n2) for n1, n2 in zip(q_names, t_names)]
    name_partial = [fuzz.partial_ratio(n1, n2) for n1, n2 in zip(q_names, t_names)]

    # ── NAME FEATURES (on stripped names — legal suffix removed) ──
    stripped_jw = [distance.JaroWinkler.normalized_similarity(n1, n2) * 100 for n1, n2 in zip(q_stripped, t_stripped)]
    stripped_tsr = [fuzz.token_sort_ratio(n1, n2) for n1, n2 in zip(q_stripped, t_stripped)]
    stripped_exact = [1.0 if (n1 and n2 and n1 == n2) else 0.0 for n1, n2 in zip(q_stripped, t_stripped)]

    # ── STRUCTURAL NAME FEATURES ──
    first_token_matches = [_first_token_match(n1, n2) for n1, n2 in zip(q_stripped, t_stripped)]
    common_token_fracs = [_common_token_fraction(n1, n2) for n1, n2 in zip(q_stripped, t_stripped)]

    token_lens_1 = [len(n.split()) for n in q_names]
    token_lens_2 = [len(n.split()) for n in t_names]
    length_ratios = [
        min(t1, t2) / max(t1, t2) if max(t1, t2) > 0 else 0.0
        for t1, t2 in zip(token_lens_1, token_lens_2)
    ]
    name_char_ratios = [_char_length_ratio(n1, n2) for n1, n2 in zip(q_names, t_names)]

    name_overlaps = [
        float(len(set(n1.split()).intersection(set(n2.split()))))
        for n1, n2 in zip(q_names, t_names)
    ]

    # ── ADDRESS FEATURES ──
    addr_jaccards = [compute_jaccard(a1, a2) * 100 for a1, a2 in zip(q_addrs, t_addrs)]
    addr_ngrams = [get_ngram_cos_sim(a1, a2) * 100 for a1, a2 in zip(q_addrs, t_addrs)]
    nums_exact = [1.0 if (z1 and z1 == z2) else 0.0 for z1, z2 in zip(q_nums, t_nums)]
    nums_overlap = [_nums_overlap_score(z1, z2) for z1, z2 in zip(q_nums, t_nums)]

    result = {
        'source1_entity_id': q_ids,
        'candidate_entity_id': t_ids,
        # -- Name (normalized) features --
        'name_lev_ratio': np.array(name_lev_ratios, dtype=np.float32),
        'name_jw_dist': np.array(name_jw_dists, dtype=np.float32),
        'name_token_sort': np.array(name_token_sorts, dtype=np.float32),
        'name_token_set': np.array(name_token_sets, dtype=np.float32),
        'name_partial_ratio': np.array(name_partial, dtype=np.float32),
        # -- Name (stripped) features --
        'stripped_jw': np.array(stripped_jw, dtype=np.float32),
        'stripped_tsr': np.array(stripped_tsr, dtype=np.float32),
        'stripped_exact': np.array(stripped_exact, dtype=np.float32),
        # -- Structural name features --
        'first_token_match': np.array(first_token_matches, dtype=np.float32),
        'common_token_frac': np.array(common_token_fracs, dtype=np.float32),
        'name_length_ratio': np.array(length_ratios, dtype=np.float32),
        'name_char_ratio': np.array(name_char_ratios, dtype=np.float32),
        'name_token_overlap': np.array(name_overlaps, dtype=np.float32),
        # -- Address features --
        'addr_jaccard': np.array(addr_jaccards, dtype=np.float32),
        'addr_ngram_sim': np.array(addr_ngrams, dtype=np.float32),
        'zip_exact': np.array(nums_exact, dtype=np.float32),
        'nums_overlap': np.array(nums_overlap, dtype=np.float32),
    }

    if query_embeddings is not None and target_embeddings is not None:
        q_emb = query_embeddings[query_indices]
        t_emb = target_embeddings[target_indices]
        result['semantic_sim'] = np.sum(q_emb * t_emb, axis=1).astype(np.float32)

    return pd.DataFrame(result)


# Feature column list used by model — single source of truth
FEATURE_COLS = [
    'name_lev_ratio', 'name_jw_dist', 'name_token_sort', 'name_token_set',
    'name_partial_ratio',
    'stripped_jw', 'stripped_tsr', 'stripped_exact',
    'first_token_match', 'common_token_frac',
    'name_length_ratio', 'name_char_ratio', 'name_token_overlap',
    'addr_jaccard', 'addr_ngram_sim', 'zip_exact', 'nums_overlap',
]


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
