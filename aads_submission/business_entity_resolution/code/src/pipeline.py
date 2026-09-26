import os
import sys

# Ensure src can be imported whether PYTHONPATH is . or code directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(CURRENT_DIR)
for p in [CODE_DIR, os.path.join(os.getcwd(), 'aads_submission', 'business_entity_resolution', 'code')]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

import argparse
import pandas as pd
import numpy as np
import time
import gc

from src.preprocessing import load_data, clean_text
from src.parallel import init_pool, close_pool, resolve_n_jobs
from src.country_pipeline import build_country_data, iter_candidate_batches, batch_features
from src.model import EntityMatchingModel
from src.evaluate import (
    optimize_threshold, evaluate_macro_f05, evaluate_detailed,
    evaluate_candidate_quality, evaluate_detailed_by_country,
    _apply_threshold_with_capping, cap_per_source,
)
from src.splits import (
    stratified_country_split, y_true_dict_from_split, s1_id_to_country_map,
)
from src.memlog import log_memory
from src.embeddings import load_embedding_model
from src.training import train_model, load_calibration, save_calibration

# Used only when there is neither an explicit --threshold nor a calibration
# file: the hand-tuned heuristic scorer's historical operating point.
HEURISTIC_DEFAULT_THRESHOLD = 0.62


def build_y_true_dict(df_gt: pd.DataFrame) -> dict:
    """Converts train_ground_truth.tsv into {s1_id: set(matched_ids)} for scoring."""
    y_true = {}
    for s1_id, matches in zip(df_gt['source1_entity_id'].values, df_gt['matched_entity_ids'].values):
        matches_str = str(matches).strip()
        if not matches_str or matches_str.lower() == 'nan':
            y_true[s1_id] = set()
        else:
            y_true[s1_id] = set(matches_str.split(','))
    return y_true


def add_country_column(df: pd.DataFrame) -> pd.DataFrame:
    # Categorical, not plain str/object: measured on a 100k-row proxy, this
    # column alone drops from ~757MB to ~13MB at the real ~12.5M-row scale.
    df['country_clean'] = pd.Categorical([clean_text(c) for c in df['country'].values])
    return df


def load_train_frames(args):
    """Loads the TRAIN split's S1 / S2+S3 / ground truth from data_dir/train."""
    d = os.path.join(args.data_dir, 'train')
    df_s1 = add_country_column(load_data(os.path.join(d, 'train_source1.tsv')))
    df_s2_s3 = pd.concat([load_data(os.path.join(d, 'train_source2.tsv')),
                          load_data(os.path.join(d, 'train_source3.tsv'))], ignore_index=True)
    add_country_column(df_s2_s3)
    return df_s1, df_s2_s3


def train_or_load_model(args, df_s1, df_s2_s3, model_path, embed_model, exclude_ids=None):
    """
    Returns (model, threshold_or_None).

    - Not training and a checkpoint exists: load it and its saved calibration.
    - Otherwise, if ground truth is available: train (see src/training.py).
      In --is_train mode the frames already in memory ARE the train split; in
      inference mode they are the TEST split, so the train split is loaded
      explicitly -- the old code reused the in-memory (test) frames there,
      which contain none of the training ground-truth IDs, so it silently
      trained on nothing and fell back to the heuristic.
    - No ground truth: unfitted model (heuristic scorer).
    """
    model = EntityMatchingModel(use_transformer=args.use_embeddings, n_jobs=args.n_jobs)

    if not args.is_train and model.load(model_path):
        cal = load_calibration(model_path)
        th = cal.get('threshold')
        print(f"Loaded trained model from {model_path}"
              + (f" (calibrated threshold {th:.3f}, val macro F0.5 {cal.get('val_macro_f05', float('nan')):.4f})"
                 if th is not None else " (no calibration file found)"))
        if cal and bool(cal.get('use_embeddings')) != bool(args.use_embeddings):
            print("  WARNING: this model was trained with use_embeddings="
                  f"{cal.get('use_embeddings')} but this run has --use_embeddings={args.use_embeddings}; "
                  "semantic_sim will be missing/NaN for the model. Retrain or match the flag.")
        return model, th

    gt_path = os.path.join(args.data_dir, 'train', 'train_ground_truth.tsv')
    if os.path.exists(gt_path):
        print("Training pair classifier on ground truth (blocking-derived pairs)...")
        gt_dict = build_y_true_dict(load_data(gt_path))
        if args.is_train:
            tr_s1, tr_s23 = df_s1, df_s2_s3
        else:
            print("  (no trained checkpoint found: loading the train split to train one now)")
            tr_s1, tr_s23 = load_train_frames(args)
        trained, th, _ = train_model(args, tr_s1, tr_s23, gt_dict, model_path, embed_model,
                                     exclude_ids=exclude_ids)
        if not args.is_train:
            del tr_s1, tr_s23
        del gt_dict
        gc.collect()
        if trained is not None:
            return trained, th

    print("Using high-precision composite heuristic (no trained model available).")
    return model, None


def score_pairs(df_s1, df_s2_s3, model, args, embed_model) -> pd.DataFrame:
    """Scores every blocking-surviving pair for df_s1; returns [s1 id, candidate id, score]."""
    frames = []
    countries = sorted(set(df_s1['country_clean'].unique()) & set(df_s2_s3['country_clean'].unique()))
    for country in countries:
        s1_c = df_s1[df_s1['country_clean'] == country].reset_index(drop=True)
        s23_c = df_s2_s3[df_s2_s3['country_clean'] == country].reset_index(drop=True)
        if s1_c.empty or s23_c.empty:
            continue
        print(f"\n--- Scoring country: {country} | S1: {len(s1_c)} | S2+S3: {len(s23_c)} ---")
        cd = build_country_data(country, s1_c, s23_c, args, embed_model)
        for b_start, b_end, sub, q_idx, t_idx, emb in iter_candidate_batches(cd, args, embed_model):
            if len(q_idx) == 0:
                continue
            feats = batch_features(cd, b_start, b_end, sub, q_idx, t_idx, emb, args.n_jobs)
            feats['score'] = model.predict_proba(feats)
            frames.append(feats[['source1_entity_id', 'candidate_entity_id', 'score']])
            del feats
        cd.free()
        del s1_c, s23_c
        gc.collect()
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def run_validation(args, model, df_s1, df_s2_s3, splits, embed_model, threshold_hint):
    """
    Threshold is tuned on the VAL fold only; the reported score comes from the
    TEST fold, which the tuning never sees. Neither fold was used to train the
    model (see main(): they're excluded from the training sample).
    """
    df_train_s1, df_val_s1, df_test_s1 = splits
    if args.eval_sample and args.eval_sample > 0:
        df_val_s1 = df_val_s1.sample(n=min(args.eval_sample, len(df_val_s1)), random_state=1)
        df_test_s1 = df_test_s1.sample(n=min(args.eval_sample, len(df_test_s1)), random_state=2)
    print(f"Evaluating on val={len(df_val_s1)} | test={len(df_test_s1)} S1 entities")

    val_ids = set(df_val_s1['entity_id'].values)
    test_ids = set(df_test_s1['entity_id'].values)
    y_true_val = y_true_dict_from_split(df_val_s1)
    y_true_test = y_true_dict_from_split(df_test_s1)
    id_to_country = {**s1_id_to_country_map(df_val_s1), **s1_id_to_country_map(df_test_s1)}

    eval_s1 = pd.concat([df_val_s1, df_test_s1], ignore_index=True)
    all_scored = score_pairs(eval_s1, df_s2_s3, model, args, embed_model)
    del eval_s1
    gc.collect()
    if all_scored.empty:
        print("No candidate pairs were scored; nothing to validate.")
        return threshold_hint

    val_scores = all_scored[all_scored['source1_entity_id'].isin(val_ids)]
    test_scores = all_scored[all_scored['source1_entity_id'].isin(test_ids)]

    cand_quality = evaluate_candidate_quality(
        all_scored[['source1_entity_id', 'candidate_entity_id']], {**y_true_val, **y_true_test})
    print("\nCandidate/blocking quality (val+test combined):")
    for k, v in cand_quality.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\nThreshold sweep (val fold only):")
    best_th, _ = optimize_threshold(val_scores, y_true_val, np.arange(0.20, 0.96, 0.05).tolist(),
                                    max_s2=args.max_s2, max_s3=args.max_s3)
    fine = np.arange(max(0.05, best_th - 0.05), min(0.99, best_th + 0.06), 0.01).tolist()
    print("\nFine-tuning (val fold only):")
    best_th, best_f05_val = optimize_threshold(val_scores, y_true_val, fine,
                                               max_s2=args.max_s2, max_s3=args.max_s3)
    print(f"\n★ Threshold chosen on val fold: {best_th:.2f} (val Macro F0.5 = {best_f05_val:.4f})")

    y_pred_test = _apply_threshold_with_capping(test_scores, y_true_test, best_th, args.max_s2, args.max_s3)
    test_f05 = evaluate_macro_f05(y_true_test, y_pred_test)
    print(f"\n★★★ HELD-OUT TEST FOLD Macro F0.5 = {test_f05:.4f} "
          f"(threshold never tuned against this fold) ★★★")

    details = evaluate_detailed(y_true_test, y_pred_test)
    print(f"\nDetailed test-fold evaluation at threshold={best_th:.2f}:")
    for k, v in details.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\nPer-country breakdown (test fold):")
    for country, c in evaluate_detailed_by_country(y_true_test, y_pred_test, id_to_country).items():
        print(f"  [{country}] macro_f05={c['macro_f05']:.4f} precision={c['mean_precision']:.4f} "
              f"recall={c['mean_recall']:.4f} n={c['total_entities']}")

    if model.is_fitted:
        cal = load_calibration(args.model_path)
        cal.update({'threshold': float(best_th), 'val_macro_f05': float(best_f05_val),
                    'test_macro_f05': float(test_f05)})
        save_calibration(args.model_path, cal)
        print(f"\nSaved validated threshold {best_th:.2f} to the model's calibration file.")

    del all_scored, val_scores, test_scores
    gc.collect()
    return float(best_th)


def _write_tsv(df: pd.DataFrame, path: str, first: bool):
    df.to_csv(path, sep='\t', index=False, header=first, mode='a')


def _join_by_query(q_idx, values, n_queries) -> np.ndarray:
    """Comma-joins `values` per query index (order preserved); '' for queries with none."""
    out = np.full(n_queries, '', dtype=object)
    if len(q_idx):
        joined = pd.DataFrame({'q': q_idx, 'v': values}).groupby('q')['v'].agg(','.join)
        out[joined.index.values] = joined.values
    return out


def main(args):
    start_time = time.time()

    # Worker pool FIRST, while the process is still small: forking after the
    # multi-GB dataframes / torch / xgboost exist is slow and can hang (see
    # src/parallel.py). Sizes both this pool and the native thread pools.
    args.n_jobs = init_pool(args.n_jobs)
    print(f"Using {args.n_jobs} worker processes/threads "
          f"(CPUs visible: {resolve_n_jobs(0)}). Override with --n_jobs.")

    # 1. Load Data
    print("Loading data...")
    if args.is_train:
        s1_path = os.path.join(args.data_dir, 'train', 'train_source1.tsv')
        s2_path = os.path.join(args.data_dir, 'train', 'train_source2.tsv')
        s3_path = os.path.join(args.data_dir, 'train', 'train_source3.tsv')
    else:
        s1_path = os.path.join(args.data_dir, 'test', 'test_source1.tsv')
        s2_test = os.path.join(args.data_dir, 'test', 'test_source2.tsv')
        s3_test = os.path.join(args.data_dir, 'test', 'test_source3.tsv')
        # Falls back to the train-split S2/S3 files only if the test-split ones
        # aren't present -- lets --subset smoke tests run before the full test
        # dataset has been downloaded.
        s2_path = s2_test if os.path.exists(s2_test) else os.path.join(args.data_dir, 'train', 'train_source2.tsv')
        s3_path = s3_test if os.path.exists(s3_test) else os.path.join(args.data_dir, 'train', 'train_source3.tsv')

    df_s1 = load_data(s1_path)
    df_s2_s3 = pd.concat([load_data(s2_path), load_data(s3_path)], ignore_index=True)
    gc.collect()

    if args.subset > 0:
        df_s1 = df_s1.head(args.subset)
        print(f"Using subset of {args.subset} S1 records for execution.")

    add_country_column(df_s1)
    add_country_column(df_s2_s3)
    log_memory("after loading + country partitioning")

    # Loaded once (cached in src.embeddings), after the worker pool exists.
    embed_model = load_embedding_model(args.embedding_model, args.n_jobs) if args.use_embeddings else None

    # 2. Held-out folds are carved out BEFORE training so they can be excluded
    #    from it -- otherwise the "held-out" score would be measured on
    #    entities the model was trained on.
    splits = None
    exclude_ids = None
    if args.validate and args.is_train:
        gt_path = os.path.join(args.data_dir, 'train', 'train_ground_truth.tsv')
        if os.path.exists(gt_path):
            splits = stratified_country_split(df_s1, load_data(gt_path), val_frac=0.15, test_frac=0.15, seed=42)
            print("Split (stratified by country): train={} | val={} | test={}".format(*[len(x) for x in splits]))
            exclude_ids = set(splits[1]['entity_id'].values) | set(splits[2]['entity_id'].values)

    # 3. Model setup / training
    model, cal_th = train_or_load_model(args, df_s1, df_s2_s3, args.model_path, embed_model, exclude_ids)
    del exclude_ids

    # Threshold precedence: explicit --threshold > calibrated > heuristic default.
    best_th = args.threshold if args.threshold is not None else (
        cal_th if cal_th is not None else HEURISTIC_DEFAULT_THRESHOLD)

    if splits is not None:
        print("\n═══ VALIDATION MODE: held-out val/test split, stratified by country ═══")
        tuned_th = run_validation(args, model, df_s1, df_s2_s3, splits, embed_model, best_th)
        if args.threshold is None:  # an explicit --threshold still wins for the output run
            best_th = tuned_th
        del splits
        gc.collect()

    print(f"\nUsing decision threshold: {best_th:.4f}")
    print(f"Per-source caps: max_s2={args.max_s2}, max_s3={args.max_s3}")

    # ─── INFERENCE: Write output files ────────────────────────────────────────
    print(f"\n═══ INFERENCE: Writing outputs with threshold={best_th:.4f} ═══")
    os.makedirs(os.path.dirname(os.path.abspath(args.matching_out)), exist_ok=True)
    if os.path.exists(args.matching_out):
        os.remove(args.matching_out)

    save_candidates = bool(args.candidate_out)
    if save_candidates:
        os.makedirs(os.path.dirname(os.path.abspath(args.candidate_out)), exist_ok=True)
        if os.path.exists(args.candidate_out):
            os.remove(args.candidate_out)

    first_match = first_cand = True
    total_processed = total_matched = total_singletons = 0

    for country in sorted(set(df_s1['country_clean'].unique()) | set(df_s2_s3['country_clean'].unique())):
        s1_c = df_s1[df_s1['country_clean'] == country].reset_index(drop=True)
        s23_c = df_s2_s3[df_s2_s3['country_clean'] == country].reset_index(drop=True)
        if s1_c.empty:
            continue

        if s23_c.empty:
            # No target records in this country: every S1 entity is a singleton.
            empty = np.full(len(s1_c), '', dtype=object)
            _write_tsv(pd.DataFrame({'source1_entity_id': s1_c['entity_id'].values,
                                     'matched_entity_ids': empty}), args.matching_out, first_match)
            first_match = False
            if save_candidates:
                _write_tsv(pd.DataFrame({'source1_entity_id': s1_c['entity_id'].values,
                                         'candidate_entity_ids': empty}), args.candidate_out, first_cand)
                first_cand = False
            total_processed += len(s1_c)
            total_singletons += len(s1_c)
            continue

        print(f"\n--- Country: {country} | S1: {len(s1_c)} | S2+S3: {len(s23_c)} ---")
        cd = build_country_data(country, s1_c, s23_c, args, embed_model)
        log_memory(f"after building index for '{country}'")
        t_ids = cd.t_ids
        n_batches = int(np.ceil(len(s1_c) / args.batch_size))
        print(f"Processing S1 in {n_batches} batches (batch_size={args.batch_size})...")

        for b, (b_start, b_end, sub, q_idx, t_idx, emb) in enumerate(iter_candidate_batches(cd, args, embed_model)):
            b_t0 = time.time()
            ids = sub['entity_id'].values

            # candidate_pairs.tsv: every candidate that reaches scoring
            if save_candidates:
                cand_str = _join_by_query(q_idx, t_ids[t_idx], len(ids))
                _write_tsv(pd.DataFrame({'source1_entity_id': ids, 'candidate_entity_ids': cand_str}),
                           args.candidate_out, first_cand)
                first_cand = False

            match_str = np.full(len(ids), '', dtype=object)
            if len(q_idx) > 0:
                feats = batch_features(cd, b_start, b_end, sub, q_idx, t_idx, emb, args.n_jobs)
                scores = model.predict_proba(feats)
                keep = scores >= best_th
                surv = feats.loc[keep, ['source1_entity_id', 'candidate_entity_id']].assign(score=scores[keep])
                del feats, scores
                kept = cap_per_source(surv, args.max_s2, args.max_s3)
                if not kept.empty:
                    joined = kept.groupby('source1_entity_id', sort=False)['candidate_entity_id'].agg(','.join)
                    match_str = pd.Series(ids).map(joined).fillna('').values
                del surv, kept

            _write_tsv(pd.DataFrame({'source1_entity_id': ids, 'matched_entity_ids': match_str}),
                       args.matching_out, first_match)
            first_match = False

            n_matched = int((match_str != '').sum())
            total_matched += n_matched
            total_singletons += len(ids) - n_matched
            total_processed += len(ids)
            print(f"  Batch {b+1}/{n_batches} ({len(ids)} records) | Candidates: {len(q_idx)} | "
                  f"Matched entities: {n_matched} | Time: {time.time() - b_t0:.2f} s")
            del q_idx, t_idx, sub, match_str
            gc.collect()
            if (b + 1) % 5 == 0 or b == n_batches - 1:
                log_memory(f"'{country}' batch {b+1}/{n_batches}")

        cd.free()
        del s1_c, s23_c
        gc.collect()

    close_pool()
    print("\n==========================================")
    print(f"Pipeline finished successfully in {time.time() - start_time:.1f} seconds.")
    print(f"Total S1 entities processed: {total_processed}")
    print(f"Entities with predicted matches: {total_matched} ({total_matched/max(1,total_processed)*100:.1f}%)")
    print(f"Predicted singletons: {total_singletons} ({total_singletons/max(1,total_processed)*100:.1f}%)")
    print(f"Matching results saved to: {args.matching_out}")
    if save_candidates:
        print(f"Candidate pairs saved to: {args.candidate_out}")
    print("==========================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--is_train', action='store_true', help='Run on the TRAIN split and (re)train the model')
    parser.add_argument('--validate', action='store_true',
                        help='With --is_train: hold out val/test folds, exclude them from training, '
                             'tune the threshold on val and report macro F0.5 on test')
    parser.add_argument('--candidate_out', type=str, default='output/candidate_pairs.tsv', help='Candidate pairs TSV output path')
    parser.add_argument('--matching_out', type=str, default='output/matching_results.tsv', help='Matching results TSV output path')
    parser.add_argument('--model_path', type=str, default='models/entity_model.json', help='Model checkpoint path')
    parser.add_argument('--batch_size', type=int, default=50000, help='Batch size for S1 chunking')
    parser.add_argument('--top_k', type=int, default=30, help='Max lexical candidates per query record')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Score decision threshold. Default: the value calibrated during training '
                             '(saved beside the model as *_calibration.json); only if no calibration '
                             f'exists, {HEURISTIC_DEFAULT_THRESHOLD} for the heuristic scorer.')
    parser.add_argument('--max_s2', type=int, default=5, help='Max S2 matches per S1 entity')
    parser.add_argument('--max_s3', type=int, default=6, help='Max S3 matches per S1 entity')
    parser.add_argument('--subset', type=int, default=0, help='Subset size for fast baseline execution (0 for full)')
    parser.add_argument('--n_jobs', type=int, default=0,
                        help='Cores to use for feature building, text cleaning, XGBoost and the '
                             'embedding model. 0 (default) = every CPU visible to the process. '
                             'In Docker, also pass --cpus to `docker run` if you want a hard cap.')
    parser.add_argument('--train_sample', type=int, default=80000,
                        help='S1 entities (country-stratified) used to train the classifier. More = '
                             'better model but a longer blocking pass over the training data.')
    parser.add_argument('--eval_sample', type=int, default=60000,
                        help='Max S1 entities per validation fold (val and test each); 0 = use all.')
    parser.add_argument('--use_embeddings', action='store_true',
                        help='Add multilingual sentence-embedding candidates + a semantic_sim feature '
                             '(src/embeddings.py). Off by default: encoding is the most expensive step.')
    parser.add_argument('--embedding_model', type=str, default=None,
                        help="Preset ('labse' [default], 'bge-m3', 'minilm'), a Hub id, or a LOCAL "
                             "directory of a sentence-transformers model.")
    parser.add_argument('--embed_countries', type=str, default='india',
                        help="Comma-separated cleaned country names to run embeddings for, or 'all'. "
                             "Default 'india': the only country with a large native-script share.")
    parser.add_argument('--emb_top_k', type=int, default=5,
                        help='Max semantic-neighbour candidates per query (kept small: candidate-set '
                             'size is graded separately).')
    args = parser.parse_args()
    main(args)
