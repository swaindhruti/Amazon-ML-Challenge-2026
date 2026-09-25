#!/usr/bin/env python3
"""
optimize_submission.py — Fast Post-Processing Script for Entity Resolution

Takes an existing candidate_pairs.tsv (or re-generates candidates), re-scores
all pairs using enhanced 17-feature representations, applies precision-tuned
thresholds with per-source empirical capping, and outputs a clean matching_results.tsv.

Can operate in three modes:
  1. RESCORE mode (default): Load candidate_pairs.tsv + source data, compute
     features, score, threshold, and cap.
  2. VALIDATE mode (--validate): Same as rescore but also evaluates against
     ground truth and sweeps for optimal threshold.
  3. TRAIN mode (--train): Trains a new model, tunes threshold, and then
     runs inference.

Usage:
  # Rescore existing test candidates (default threshold 0.62):
  python optimize_submission.py \
      --data_dir student_resource/dataset \
      --candidates output/candidate_pairs.tsv \
      --output output/matching_results.tsv

  # Validate on training set (tune threshold with coarse + fine sweep):
  python optimize_submission.py \
      --data_dir student_resource/dataset \
      --is_train --validate \
      --subset 1000

  # Full train + tune + validate:
  python optimize_submission.py \
      --data_dir student_resource/dataset \
      --train --validate --is_train \
      --subset 5000
"""

import os
import sys
import argparse
import time
import gc
from collections import defaultdict
import numpy as np
import pandas as pd

# Add source directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(
    SCRIPT_DIR, 'aads_submission', 'business_entity_resolution', 'code'
)
for p in [CODE_DIR, SCRIPT_DIR, os.path.join(SCRIPT_DIR, 'src'), os.path.join(CODE_DIR, 'src')]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

try:
    from src.preprocessing import (
        load_data, clean_text, normalize_abbreviations,
        extract_numerical_tokens, strip_legal_suffixes, strip_landmark_words,
    )
    from src.features import build_batch_features
    from src.model import EntityMatchingModel
    from src.evaluate import (
        evaluate_detailed, optimize_threshold, _apply_threshold_with_capping,
    )
    from src.candidate_generation import generate_candidates
    from src.splits import stratified_sample
except ImportError:
    from aads_submission.business_entity_resolution.code.src.preprocessing import (
        load_data, clean_text, normalize_abbreviations,
        extract_numerical_tokens, strip_legal_suffixes, strip_landmark_words,
    )
    from aads_submission.business_entity_resolution.code.src.features import build_batch_features
    from aads_submission.business_entity_resolution.code.src.model import EntityMatchingModel
    from aads_submission.business_entity_resolution.code.src.evaluate import (
        evaluate_detailed, optimize_threshold, _apply_threshold_with_capping,
    )
    from aads_submission.business_entity_resolution.code.src.candidate_generation import (
        generate_candidates,
    )
    from aads_submission.business_entity_resolution.code.src.splits import stratified_sample


def prepare_text_arrays(df: pd.DataFrame):
    """Extracts cleaned, normalized, and suffix-stripped arrays."""
    names_clean = [clean_text(t) for t in df['business_name'].values]
    names_norm = [normalize_abbreviations(t) for t in names_clean]
    names_stripped = [strip_legal_suffixes(t) for t in names_clean]
    addrs_clean = [clean_text(t) for t in df['business_address'].values]
    addrs_norm = [strip_landmark_words(normalize_abbreviations(t)) for t in addrs_clean]
    nums = [extract_numerical_tokens(t) for t in addrs_clean]
    return names_clean, names_norm, names_stripped, addrs_clean, addrs_norm, nums


def build_y_true_dict(df_gt: pd.DataFrame) -> dict:
    y_true = {}
    for _, row in df_gt.iterrows():
        s1_id = row['source1_entity_id']
        matches_str = str(row['matched_entity_ids']).strip()
        if not matches_str or matches_str.lower() == 'nan':
            y_true[s1_id] = set()
        else:
            y_true[s1_id] = set(matches_str.split(','))
    return y_true


def rescore_candidates(
    df_s1: pd.DataFrame,
    df_s2_s3: pd.DataFrame,
    df_candidates: pd.DataFrame,
    model: EntityMatchingModel,
    batch_size: int = 100000,
) -> pd.DataFrame:
    """
    Re-scores candidate pairs using enhanced features.

    Optimized:
    - Filters target pool to only unique candidate IDs appearing in df_candidates
      BEFORE doing heavy text preprocessing, avoiding processing 10M+ rows unnecessarily.
    - Only preprocesses S1 entities present in candidates.
    - Yields massive speedup and eliminates memory bottlenecks.
    """
    if df_candidates.empty or len(df_s1) == 0:
        return pd.DataFrame(columns=['source1_entity_id', 'candidate_entity_id', 'score'])

    s1_id_set = set(df_s1['entity_id'].values)
    target_id_set = set(df_s2_s3['entity_id'].values)

    # 1. Parse candidate pairs (deduplicating per S1-target pair)
    pairs_s1 = []
    pairs_cand = []
    unique_cand_ids = set()
    unique_s1_ids = set()
    seen_pairs = set()

    for _, row in df_candidates.iterrows():
        s1_id = str(row['source1_entity_id']).strip()
        cands_str = str(row.get('candidate_entity_ids', '')).strip()
        if not cands_str or cands_str.lower() == 'nan' or s1_id not in s1_id_set:
            continue

        cands = [c.strip() for c in cands_str.split(',') if c.strip()]
        for cand_id in cands:
            if cand_id in target_id_set:
                pair = (s1_id, cand_id)
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    pairs_s1.append(s1_id)
                    pairs_cand.append(cand_id)
                    unique_cand_ids.add(cand_id)
                    unique_s1_ids.add(s1_id)

    del seen_pairs

    total_pairs = len(pairs_s1)
    print(
        f"  Valid candidate pairs to score: {total_pairs:,} "
        f"(across {len(unique_s1_ids):,} S1 and {len(unique_cand_ids):,} targets)"
    )

    if total_pairs == 0:
        return pd.DataFrame(columns=['source1_entity_id', 'candidate_entity_id', 'score'])

    # 2. Filter dataframes to only entities actually needed for candidate scoring
    df_s1_needed = df_s1[df_s1['entity_id'].isin(unique_s1_ids)].reset_index(drop=True)
    df_t_needed = df_s2_s3[df_s2_s3['entity_id'].isin(unique_cand_ids)].reset_index(drop=True)

    s1_id_to_idx = {eid: i for i, eid in enumerate(df_s1_needed['entity_id'].values)}
    t_id_to_idx = {eid: i for i, eid in enumerate(df_t_needed['entity_id'].values)}

    # Map string IDs to row indices in the filtered arrays
    all_q_idx = np.array([s1_id_to_idx[sid] for sid in pairs_s1], dtype=np.int32)
    all_t_idx = np.array([t_id_to_idx[tid] for tid in pairs_cand], dtype=np.uint32)
    del pairs_s1, pairs_cand
    gc.collect()

    # 3. Preprocess ONLY the filtered records
    print(
        f"  Preprocessing text: {len(df_s1_needed):,} S1 queries, "
        f"{len(df_t_needed):,} unique target candidates..."
    )
    _, s1_norm, s1_stripped, _, s1_addrs_norm, s1_nums = prepare_text_arrays(df_s1_needed)
    df_s1_proc = pd.DataFrame({
        'entity_id': df_s1_needed['entity_id'].values,
        'business_name_norm': s1_norm,
        'business_name_stripped': s1_stripped,
        'business_address_norm': s1_addrs_norm,
        'address_numbers': s1_nums,
    })
    del s1_norm, s1_stripped, s1_addrs_norm, s1_nums

    _, t_norm, t_stripped, _, t_addrs_norm, t_nums = prepare_text_arrays(df_t_needed)
    df_t_proc = pd.DataFrame({
        'entity_id': df_t_needed['entity_id'].values,
        'business_name_norm': t_norm,
        'business_name_stripped': t_stripped,
        'business_address_norm': t_addrs_norm,
        'address_numbers': t_nums,
    })
    del t_norm, t_stripped, t_addrs_norm, t_nums
    gc.collect()

    # 4. Process in batches
    all_results = []
    num_batches = int(np.ceil(total_pairs / batch_size))

    for b in range(num_batches):
        b_start = b * batch_size
        b_end = min((b + 1) * batch_size, total_pairs)
        b_q = all_q_idx[b_start:b_end]
        b_t = all_t_idx[b_start:b_end]

        df_feat = build_batch_features(df_s1_proc, df_t_proc, b_q, b_t)
        scores = model.predict_proba(df_feat)

        all_results.append(pd.DataFrame({
            'source1_entity_id': df_feat['source1_entity_id'].values,
            'candidate_entity_id': df_feat['candidate_entity_id'].values,
            'score': scores.astype(np.float32),
        }))

        if (b + 1) % 10 == 0 or b == num_batches - 1:
            print(f"  Scored batch {b+1}/{num_batches} ({b_end:,}/{total_pairs:,} pairs)")

        del df_feat, scores
        if (b + 1) % 5 == 0:
            gc.collect()

    del all_q_idx, all_t_idx, df_s1_proc, df_t_proc
    gc.collect()

    return pd.concat(all_results, ignore_index=True)


def apply_threshold_and_cap(
    df_scores: pd.DataFrame,
    all_s1_ids: np.ndarray,
    threshold: float,
    max_s2: int = 5,
    max_s3: int = 6,
) -> pd.DataFrame:
    """
    Applies threshold and per-source capping to produce final matching results.
    Returns DataFrame with ['source1_entity_id', 'matched_entity_ids'].
    Fast single-pass implementation.
    """
    if df_scores.empty:
        return pd.DataFrame({
            'source1_entity_id': all_s1_ids,
            'matched_entity_ids': [''] * len(all_s1_ids)
        })

    # Filter above threshold and sort by score descending
    df_above = df_scores[df_scores['score'] >= threshold].sort_values('score', ascending=False)

    s2_counts = defaultdict(int)
    s3_counts = defaultdict(int)
    match_map = defaultdict(list)
    seen_matches = defaultdict(set)

    s1_arr = df_above['source1_entity_id'].values
    cand_arr = df_above['candidate_entity_id'].values

    for s1_id, cand_id in zip(s1_arr, cand_arr):
        if cand_id in seen_matches[s1_id]:
            continue
        if cand_id.startswith('S2-'):
            if s2_counts[s1_id] < max_s2:
                match_map[s1_id].append(cand_id)
                s2_counts[s1_id] += 1
                seen_matches[s1_id].add(cand_id)
        elif cand_id.startswith('S3-'):
            if s3_counts[s1_id] < max_s3:
                match_map[s1_id].append(cand_id)
                s3_counts[s1_id] += 1
                seen_matches[s1_id].add(cand_id)

    # Format output for every S1 entity
    rows = [
        {
            'source1_entity_id': s1_id,
            'matched_entity_ids': ','.join(match_map[s1_id]) if s1_id in match_map else ''
        }
        for s1_id in all_s1_ids
    ]

    return pd.DataFrame(rows)


def print_summary(df_results: pd.DataFrame, threshold: float):
    """Print submission summary statistics."""
    total = len(df_results)
    if total == 0:
        print("  Empty results dataframe.")
        return

    empty = (df_results['matched_entity_ids'] == '').sum()
    non_empty = total - empty

    match_counts = df_results['matched_entity_ids'].apply(
        lambda x: len(x.split(',')) if x else 0
    )

    s2_counts = df_results['matched_entity_ids'].apply(
        lambda x: sum(1 for m in x.split(',') if m.startswith('S2-')) if x else 0
    )
    s3_counts = df_results['matched_entity_ids'].apply(
        lambda x: sum(1 for m in x.split(',') if m.startswith('S3-')) if x else 0
    )

    print(f"\n{'='*60}")
    print(f"SUBMISSION SUMMARY (threshold={threshold:.2f})")
    print(f"{'='*60}")
    print(f"  Total S1 entities:        {total:>10,}")
    print(f"  Predicted singletons:     {empty:>10,} ({empty/total*100:.1f}%)")
    print(f"  Entities with matches:    {non_empty:>10,} ({non_empty/total*100:.1f}%)")
    print(f"  Total predicted matches:  {match_counts.sum():>10,}")
    print(f"  Avg matches per entity:   {match_counts.mean():>10.2f}")
    print(f"  Max matches per entity:   {match_counts.max():>10}")
    print("  ── S2 matches ──")
    print(
        f"    Entities with S2 match: {(s2_counts > 0).sum():>10,} "
        f"({(s2_counts > 0).sum()/total*100:.1f}%)"
    )
    print(f"    Avg S2 per entity:      {s2_counts.mean():>10.2f}")
    print(f"    Max S2 per entity:      {s2_counts.max():>10}")
    print("  ── S3 matches ──")
    print(
        f"    Entities with S3 match: {(s3_counts > 0).sum():>10,} "
        f"({(s3_counts > 0).sum()/total*100:.1f}%)"
    )
    print(f"    Avg S3 per entity:      {s3_counts.mean():>10.2f}")
    print(f"    Max S3 per entity:      {s3_counts.max():>10}")
    print(f"{'='*60}")

    print("\n  Match count distribution:")
    for n_matches in sorted(match_counts.unique()):
        cnt = (match_counts == n_matches).sum()
        print(f"    {n_matches:>2} matches: {cnt:>8,} entities ({cnt/total*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description='Optimize Entity Resolution submission for max F0.5'
    )
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to dataset directory (containing train/ and/or test/)')
    parser.add_argument('--candidates', type=str, default=None,
                        help='Path to candidate_pairs.tsv (input)')
    parser.add_argument('--output', type=str, default='output/matching_results.tsv',
                        help='Path to output matching_results.tsv')
    parser.add_argument('--model_path', type=str, default='models/entity_model.json',
                        help='Path to trained model (optional, uses heuristic if missing)')
    parser.add_argument('--is_train', action='store_true',
                        help='Use training data instead of test data')
    parser.add_argument('--validate', action='store_true',
                        help='Evaluate against ground truth and tune threshold')
    parser.add_argument('--train', action='store_true',
                        help='Train a new model before scoring')
    parser.add_argument('--threshold', type=float, default=0.62,
                        help='Score decision threshold. 0.62 is a placeholder, not a verified '
                             'optimum -- run pipeline.py --is_train --validate to get the real '
                             'value via the held-out test-fold harness in src/splits.py, then pass '
                             'that value here. Kept in sync with pipeline.py\'s default.')
    parser.add_argument('--max_s2', type=int, default=5,
                        help='Max S2 matches per S1 entity (default: 5)')
    parser.add_argument('--max_s3', type=int, default=6,
                        help='Max S3 matches per S1 entity (default: 6)')
    parser.add_argument('--subset', type=int, default=0,
                        help='Process only first N S1 entities (0 = all)')
    parser.add_argument('--batch_size', type=int, default=100000,
                        help='Scoring batch size')
    parser.add_argument('--no_candidates', action='store_true',
                        help='Generate candidates from scratch instead of loading candidate file')
    args = parser.parse_args()

    start_time = time.time()
    print(f"{'='*60}")
    print("Entity Resolution Submission Optimizer")
    print(f"{'='*60}")

    # Set default candidates path if not provided
    if args.candidates is None:
        if args.is_train:
            args.candidates = 'output/candidate_pairs_train.tsv'
        else:
            args.candidates = 'output/candidate_pairs.tsv'

    # ── Load source data ──────────────────────────────────────────────────
    print("\n[1/5] Loading source data...")
    if args.is_train:
        s1_path = os.path.join(args.data_dir, 'train', 'train_source1.tsv')
        s2_path = os.path.join(args.data_dir, 'train', 'train_source2.tsv')
        s3_path = os.path.join(args.data_dir, 'train', 'train_source3.tsv')
    else:
        s1_path = os.path.join(args.data_dir, 'test', 'test_source1.tsv')
        s2_path = os.path.join(args.data_dir, 'test', 'test_source2.tsv')
        s3_path = os.path.join(args.data_dir, 'test', 'test_source3.tsv')

    df_s1 = load_data(s1_path)
    if args.subset > 0:
        df_s1 = df_s1.head(args.subset).copy()
        print(f"  Using subset: {args.subset} S1 entities")

    # Load S2 and S3
    df_s2 = load_data(s2_path)
    df_s3 = load_data(s3_path)
    df_s2_s3 = pd.concat([df_s2, df_s3], ignore_index=True)
    del df_s2, df_s3
    gc.collect()

    print(f"  S1: {len(df_s1):,} | S2+S3: {len(df_s2_s3):,}")

    # ── Load / setup model ────────────────────────────────────────────────
    print("\n[2/5] Loading model...")
    model = EntityMatchingModel()

    if args.train:
        print("  Training new model with country-partitioned negatives...")
        gt_path = os.path.join(args.data_dir, 'train', 'train_ground_truth.tsv')
        if os.path.exists(gt_path):
            df_gt = load_data(gt_path)
            gt_dict = build_y_true_dict(df_gt)
            del df_gt
            gc.collect()

            df_s1['country_clean'] = [clean_text(c) for c in df_s1['country'].values]
            df_s2_s3['country_clean'] = [clean_text(c) for c in df_s2_s3['country'].values]

            s23_map = {eid: i for i, eid in enumerate(df_s2_s3['entity_id'].values)}
            country_to_t_idx = defaultdict(list)
            for idx, c in enumerate(df_s2_s3['country_clean'].values):
                country_to_t_idx[c].append(idx)
            for c in country_to_t_idx:
                country_to_t_idx[c] = np.array(country_to_t_idx[c], dtype=np.uint32)

            pos_q, pos_t, neg_q, neg_t = [], [], [], []
            rng = np.random.RandomState(42)
            n_sample = min(50000, len(df_s1))
            # Sampled proportionally by country rather than df_s1.iloc[:n_sample] --
            # the first N rows give no guarantee of country balance.
            s1_sample = stratified_sample(df_s1, n_sample, key='country_clean', seed=42)

            for i in range(len(s1_sample)):
                s1_id = s1_sample['entity_id'].iloc[i]
                s1_c = s1_sample['country_clean'].iloc[i]
                avail_t = country_to_t_idx.get(s1_c, np.array([], dtype=np.uint32))

                if s1_id in gt_dict:
                    for m in gt_dict[s1_id]:
                        if m in s23_map:
                            pos_q.append(i)
                            pos_t.append(s23_map[m])
                            # Negatives drawn from the SAME country as the
                            # positive, not the whole target pool: a random
                            # cross-country negative is nearly always trivially
                            # easy (blocking would never even propose it as a
                            # candidate), so it teaches the model little. A
                            # same-country negative is closer to what the model
                            # will actually have to discriminate at inference
                            # time, once blocking has already narrowed things
                            # down to same-country candidates.
                            if len(avail_t) > 0:
                                n_pick = min(5, len(avail_t))
                                for ti in rng.choice(avail_t, size=n_pick, replace=False):
                                    neg_q.append(i)
                                    neg_t.append(ti)

            if pos_q:
                q_idx = np.array(pos_q + neg_q, dtype=np.int32)
                t_idx = np.array(pos_t + neg_t, dtype=np.uint32)
                labels = np.array([1]*len(pos_q) + [0]*len(neg_q), dtype=np.int32)

                s1_sub = s1_sample.reset_index(drop=True)
                _, s1_n, s1_st, _, s1_an, s1_nu = prepare_text_arrays(s1_sub)
                s1_proc = pd.DataFrame({
                    'entity_id': s1_sub['entity_id'].values,
                    'business_name_norm': s1_n,
                    'business_name_stripped': s1_st,
                    'business_address_norm': s1_an,
                    'address_numbers': s1_nu,
                })

                t_uniq = np.unique(t_idx)
                t_map = {o: n for n, o in enumerate(t_uniq)}
                t_sub = df_s2_s3.iloc[t_uniq].reset_index(drop=True)
                _, tn, ts, _, ta, tnu = prepare_text_arrays(t_sub)
                t_proc = pd.DataFrame({
                    'entity_id': t_sub['entity_id'].values,
                    'business_name_norm': tn,
                    'business_name_stripped': ts,
                    'business_address_norm': ta,
                    'address_numbers': tnu,
                })

                t_idx_remap = np.array([t_map[ti] for ti in t_idx], dtype=np.uint32)
                df_feat = build_batch_features(s1_proc, t_proc, q_idx, t_idx_remap)
                df_feat['label'] = labels

                print(f"  Training on {len(df_feat):,} pairs ({len(pos_q):,} pos, {len(neg_q):,} neg)")
                model.fit(df_feat, df_feat['label'])
                model.save(args.model_path)
                print(f"  Model saved to {args.model_path}")
                del s1_proc, t_proc, df_feat, gt_dict, s23_map, country_to_t_idx
                gc.collect()
        else:
            print(f"  Ground truth file not found at {gt_path}. Skipping training.")
    elif os.path.exists(args.model_path):
        if model.load(args.model_path):
            print(f"  Loaded model from {args.model_path}")
        else:
            print("  Failed to load model, using heuristic scoring")
    else:
        print(f"  No model found at {args.model_path}, using heuristic scoring")

    # ── Load or generate candidates ───────────────────────────────────────
    print("\n[3/5] Loading candidates...")
    cand_loaded = False
    df_cand = pd.DataFrame()

    if not args.no_candidates and os.path.exists(args.candidates):
        df_cand_raw = load_data(args.candidates)
        s1_id_set = set(df_s1['entity_id'].values)
        df_cand = df_cand_raw[df_cand_raw['source1_entity_id'].isin(s1_id_set)].reset_index(drop=True)
        del df_cand_raw

        if len(df_cand) > 0:
            print(f"  Loaded {len(df_cand):,} candidate rows from {args.candidates}")
            cand_loaded = True
        else:
            print(f"  Notice: '{args.candidates}' has 0 matches for selected S1 entities.")

    if not cand_loaded:
        print("  Generating candidates from scratch (country-partitioned inverted index)...")
        # Ensure country columns are cleaned
        df_s1['country_clean'] = [clean_text(c) for c in df_s1['country'].values]
        df_s2_s3['country_clean'] = [clean_text(c) for c in df_s2_s3['country'].values]

        # Ensure normalized text arrays exist for inverted index
        _, s1_n, _, _, s1_an, s1_nu = prepare_text_arrays(df_s1)
        df_s1['business_name_norm'] = s1_n
        df_s1['business_address_norm'] = s1_an
        df_s1['address_numbers'] = s1_nu

        # Only process target entities that match countries in S1
        matching_countries = set(df_s1['country_clean'].values)
        t_country_mask = df_s2_s3['country_clean'].isin(matching_countries)
        df_t_subset = df_s2_s3[t_country_mask].copy().reset_index(drop=True)

        _, tn, _, _, ta, tnu = prepare_text_arrays(df_t_subset)
        df_t_subset['business_name_norm'] = tn
        df_t_subset['business_address_norm'] = ta
        df_t_subset['address_numbers'] = tnu

        df_cand = generate_candidates(df_s1, df_t_subset, top_k=30, batch_size=50000)
        print(f"  Generated {len(df_cand):,} candidate rows")
        del df_t_subset
        gc.collect()

    # ── Score candidate pairs ─────────────────────────────────────────────
    print("\n[4/5] Scoring candidate pairs...")
    df_scores = rescore_candidates(
        df_s1, df_s2_s3, df_cand, model, batch_size=args.batch_size
    )
    print(f"  Scored {len(df_scores):,} pairs")

    if not df_scores.empty:
        print(f"  Score distribution: min={df_scores['score'].min():.3f} "
              f"median={df_scores['score'].median():.3f} "
              f"mean={df_scores['score'].mean():.3f} "
              f"max={df_scores['score'].max():.3f}")
        for pct in [0.40, 0.50, 0.60, 0.62, 0.70, 0.75, 0.80, 0.85, 0.90]:
            above = (df_scores['score'] >= pct).sum()
            print(f"    >= {pct:.2f}: {above:>10,} pairs ({above/len(df_scores)*100:.1f}%)")

    # ── Validation (optional) ─────────────────────────────────────────────
    best_threshold = args.threshold

    if args.validate and args.is_train:
        print("\n[V] Validation: Tuning threshold on ground truth...")
        gt_path = os.path.join(args.data_dir, 'train', 'train_ground_truth.tsv')
        if os.path.exists(gt_path):
            df_gt = load_data(gt_path)
            gt_dict = build_y_true_dict(df_gt)
            del df_gt

            val_s1_ids = set(df_s1['entity_id'].values)
            gt_sub = {k: v for k, v in gt_dict.items() if k in val_s1_ids}
            print(f"  Validation entities: {len(gt_sub):,}")

            # Coarse sweep: 0.40 to 0.95
            print("\n  Coarse threshold sweep:")
            ths_coarse = [round(x, 2) for x in np.arange(0.40, 0.96, 0.04)]
            best_th_c, _ = optimize_threshold(
                df_scores, gt_sub, ths_coarse,
                max_s2=args.max_s2, max_s3=args.max_s3
            )

            # Fine sweep around best
            print(f"\n  Fine threshold sweep around {best_th_c:.2f}:")
            ths_fine = [
                round(x, 2)
                for x in np.arange(max(0.30, best_th_c - 0.06), min(0.98, best_th_c + 0.07), 0.01)
            ]
            best_threshold, best_f05 = optimize_threshold(
                df_scores, gt_sub, ths_fine,
                max_s2=args.max_s2, max_s3=args.max_s3
            )

            # Detailed eval at best threshold
            y_pred = _apply_threshold_with_capping(
                df_scores, gt_sub, best_threshold, args.max_s2, args.max_s3
            )
            details = evaluate_detailed(gt_sub, y_pred)

            print("\n  ╔══════════════════════════════════════════╗")
            print(f"  ║  OPTIMAL THRESHOLD: {best_threshold:.2f}                  ║")
            print(f"  ║  Macro F0.5: {best_f05:.4f}                      ║")
            print("  ╚══════════════════════════════════════════╝")
            print("  Detailed metrics:")
            for k, v in details.items():
                if isinstance(v, float):
                    print(f"    {k:35s}: {v:.4f}")
                else:
                    print(f"    {k:35s}: {v}")

            del gt_dict, gt_sub
            gc.collect()

    # ── Apply threshold and cap, write output ─────────────────────────────
    print(f"\n[5/5] Generating output with threshold={best_threshold:.2f}...")
    all_s1_ids = df_s1['entity_id'].values

    df_results = apply_threshold_and_cap(
        df_scores, all_s1_ids,
        threshold=best_threshold,
        max_s2=args.max_s2,
        max_s3=args.max_s3,
    )

    # Write output
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df_results.to_csv(args.output, sep='\t', index=False)
    print(f"  Written {len(df_results):,} rows to {args.output}")

    # Print summary statistics
    print_summary(df_results, best_threshold)

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")


if __name__ == '__main__':
    main()
