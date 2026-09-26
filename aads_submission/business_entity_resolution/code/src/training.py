"""
Trains the XGBoost pair classifier and calibrates its decision threshold.

Three things here differ from the earlier in-pipeline trainer, each fixing a
concrete reason a model scoring well locally can collapse on the real test:

1. Training pairs come from the SAME blocking + feature code used at inference
   (country_pipeline.py), so negatives are the genuinely confusable
   near-misses blocking surfaces -- not randomly drawn records that are
   trivially dissimilar. A model trained against random negatives learns
   "is this pair remotely similar?" and then floods the output with false
   positives once it faces hard near-misses; F0.5 punishes exactly that.
2. Train/validation are split by S1 ENTITY (all of an entity's candidate
   pairs land on one side). Splitting individual pairs 80/20, as before, put
   the same entity's pairs in both halves and made validation loss/threshold
   look better than reality.
3. The decision threshold is tuned against the actual competition metric
   (entity-level macro F0.5, with the per-source caps) on those held-out
   entities and saved next to the model, so inference no longer depends on a
   hand-guessed --threshold default.
"""
import gc
import json
import os
import time

import numpy as np
import pandas as pd

from src.splits import stratified_sample
from src.evaluate import optimize_threshold, evaluate_candidate_quality
from src.country_pipeline import build_country_data, iter_candidate_batches, batch_features
from src.model import EntityMatchingModel
from src.features import FEATURE_COLS, OPTIONAL_FEATURE_COLS

_EMPTY = frozenset()


def calibration_path(model_path: str) -> str:
    return os.path.splitext(model_path)[0] + '_calibration.json'


def save_calibration(model_path: str, info: dict):
    path = calibration_path(model_path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(info, f, indent=2)


def load_calibration(model_path: str) -> dict:
    path = calibration_path(model_path)
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}
    return {}


def collect_training_pairs(df_sample: pd.DataFrame, df_s2_s3: pd.DataFrame, gt_dict: dict,
                           args, embed_model=None) -> pd.DataFrame:
    """
    Runs blocking + feature extraction for the sampled S1 entities, country by
    country, and labels every surviving (S1, candidate) pair from the ground
    truth. Returns one DataFrame of features + 'label'.
    """
    frames = []
    for country in sorted(df_sample['country_clean'].unique()):
        s1_c = df_sample[df_sample['country_clean'] == country].reset_index(drop=True)
        s23_c = df_s2_s3[df_s2_s3['country_clean'] == country].reset_index(drop=True)
        if s1_c.empty or s23_c.empty:
            continue
        print(f"\n[train] {country}: {len(s1_c)} S1 entities vs {len(s23_c)} targets")
        cd = build_country_data(country, s1_c, s23_c, args, embed_model)
        for b_start, b_end, sub, q_idx, t_idx, emb in iter_candidate_batches(cd, args, embed_model):
            if len(q_idx) == 0:
                continue
            feats = batch_features(cd, b_start, b_end, sub, q_idx, t_idx, emb, args.n_jobs)
            feats['label'] = np.fromiter(
                (c in gt_dict.get(s, _EMPTY) for s, c in zip(feats['source1_entity_id'].values,
                                                              feats['candidate_entity_id'].values)),
                dtype=np.int8, count=len(feats))
            frames.append(feats)
            del feats
        cd.free()
        del s1_c, s23_c
        gc.collect()
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def train_model(args, df_s1: pd.DataFrame, df_s2_s3: pd.DataFrame, gt_dict: dict,
                model_path: str, embed_model=None, exclude_ids=None):
    """
    Trains on a country-stratified sample of S1 entities (excluding
    exclude_ids -- the held-out validation/test entities when --validate is
    on, so the reported score is never measured on training data), tunes the
    threshold on held-out entities, saves model + calibration. Returns
    (model, threshold, val_macro_f05) or (None, None, None) if there was
    nothing to train on.
    """
    t0 = time.time()
    pool = df_s1
    if exclude_ids:
        pool = df_s1[~df_s1['entity_id'].isin(exclude_ids)]
    sample = stratified_sample(pool, min(args.train_sample, len(pool)), key='country_clean', seed=42)

    rng = np.random.RandomState(7)
    is_val = rng.rand(len(sample)) < 0.2
    val_ids = set(sample['entity_id'].values[is_val])
    print(f"[train] {len(sample)} S1 entities sampled "
          f"({len(sample) - len(val_ids)} train / {len(val_ids)} held-out for threshold tuning)")

    df_pairs = collect_training_pairs(sample, df_s2_s3, gt_dict, args, embed_model)
    if df_pairs.empty or df_pairs['label'].sum() == 0:
        print("[train] no positive pairs surfaced from blocking -- cannot train a classifier.")
        return None, None, None

    y_true_all = {s: set(gt_dict.get(s, _EMPTY)) for s in sample['entity_id'].values}
    quality = evaluate_candidate_quality(df_pairs[['source1_entity_id', 'candidate_entity_id']], y_true_all)
    print("[train] blocking quality on the training sample (upper bound for any model):")
    for k, v in quality.items():
        print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    in_val = df_pairs['source1_entity_id'].isin(val_ids).values
    df_tr, df_va = df_pairs[~in_val], df_pairs[in_val]
    print(f"[train] pairs: train={len(df_tr)} ({int(df_tr['label'].sum())} pos) | "
          f"val={len(df_va)} ({int(df_va['label'].sum())} pos)")

    model = EntityMatchingModel(use_transformer=bool(args.use_embeddings), n_jobs=args.n_jobs)
    if len(df_va) and df_va['label'].sum() > 0:
        model.fit(df_tr, df_tr['label'], df_va, df_va['label'])
    else:
        model.fit(df_tr, df_tr['label'])

    # -- threshold on held-out entities, against the real metric --
    val_scores = df_va[['source1_entity_id', 'candidate_entity_id']].copy()
    val_scores['score'] = model.predict_proba(df_va)
    y_true_val = {s: y_true_all[s] for s in val_ids}
    thresholds = np.arange(0.20, 0.96, 0.025).tolist()
    best_th, best_f05 = optimize_threshold(val_scores, y_true_val, thresholds,
                                           max_s2=args.max_s2, max_s3=args.max_s3, verbose=False)
    print(f"[train] threshold tuned on held-out entities: {best_th:.3f} (macro F0.5 = {best_f05:.4f})")

    try:
        booster = model.model.get_booster()
        imp = sorted(booster.get_score(importance_type='gain').items(), key=lambda kv: -kv[1])
        print("[train] top features by gain: " + ", ".join(f"{k}={v:.0f}" for k, v in imp[:10]))
    except Exception:
        pass

    model.save(model_path)
    save_calibration(model_path, {
        'threshold': float(best_th),
        'val_macro_f05': float(best_f05),
        'max_s2': args.max_s2,
        'max_s3': args.max_s3,
        'top_k': args.top_k,
        'use_embeddings': bool(args.use_embeddings),
        'train_entities': int(len(sample) - len(val_ids)),
        'feature_cols': [c for c in FEATURE_COLS + OPTIONAL_FEATURE_COLS if c in df_tr.columns],
    })
    print(f"[train] saved {model_path} + calibration in {time.time() - t0:.1f} s")

    del df_pairs, df_tr, df_va, val_scores
    gc.collect()
    return model, float(best_th), float(best_f05)
