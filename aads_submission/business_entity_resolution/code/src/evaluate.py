import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


def evaluate_macro_f05(y_true: Dict[str, set], y_pred: Dict[str, set]) -> float:
    """
    Computes macro-averaged F_0.5 score across all Source 1 entities.
    y_true: dict mapping Source 1 entity_id to a set of true matching entity_ids.
    y_pred: dict mapping Source 1 entity_id to a set of predicted matching entity_ids.
    """
    f05_scores = []

    for s1_id in y_true.keys():
        true_matches = y_true[s1_id]
        pred_matches = y_pred.get(s1_id, set())

        # Singleton logic
        if len(true_matches) == 0:
            if len(pred_matches) == 0:
                f05_scores.append(1.0)
            else:
                f05_scores.append(0.0)
            continue

        if len(pred_matches) == 0:
            f05_scores.append(0.0)
            continue

        true_positives = len(true_matches.intersection(pred_matches))

        precision = true_positives / len(pred_matches) if len(pred_matches) > 0 else 0.0
        recall = true_positives / len(true_matches) if len(true_matches) > 0 else 0.0

        if precision == 0 and recall == 0:
            f05 = 0.0
        else:
            f05 = (1.25 * precision * recall) / (0.25 * precision + recall)

        f05_scores.append(f05)

    return np.mean(f05_scores) if f05_scores else 0.0


def evaluate_detailed(y_true: Dict[str, set], y_pred: Dict[str, set]) -> dict:
    """
    Returns detailed evaluation breakdown: F0.5, precision, recall, singleton accuracy,
    and per-source statistics.
    """
    f05_scores = []
    precision_scores = []
    recall_scores = []
    singleton_correct = 0
    singleton_total = 0
    non_singleton_f05 = []

    for s1_id in y_true.keys():
        true_matches = y_true[s1_id]
        pred_matches = y_pred.get(s1_id, set())

        if len(true_matches) == 0:
            singleton_total += 1
            if len(pred_matches) == 0:
                f05_scores.append(1.0)
                singleton_correct += 1
            else:
                f05_scores.append(0.0)
            continue

        if len(pred_matches) == 0:
            f05_scores.append(0.0)
            precision_scores.append(0.0)
            recall_scores.append(0.0)
            non_singleton_f05.append(0.0)
            continue

        tp = len(true_matches.intersection(pred_matches))
        precision = tp / len(pred_matches) if len(pred_matches) > 0 else 0.0
        recall = tp / len(true_matches) if len(true_matches) > 0 else 0.0

        if precision == 0 and recall == 0:
            f05 = 0.0
        else:
            f05 = (1.25 * precision * recall) / (0.25 * precision + recall)

        f05_scores.append(f05)
        precision_scores.append(precision)
        recall_scores.append(recall)
        non_singleton_f05.append(f05)

    # Prediction statistics
    total_predicted = sum(len(v) for v in y_pred.values())
    entities_with_preds = sum(1 for v in y_pred.values() if len(v) > 0)
    avg_preds = total_predicted / len(y_pred) if y_pred else 0

    return {
        'macro_f05': np.mean(f05_scores) if f05_scores else 0.0,
        'mean_precision': np.mean(precision_scores) if precision_scores else 0.0,
        'mean_recall': np.mean(recall_scores) if recall_scores else 0.0,
        'singleton_accuracy': singleton_correct / singleton_total if singleton_total > 0 else 1.0,
        'singleton_total': singleton_total,
        'singleton_correct': singleton_correct,
        'non_singleton_f05': np.mean(non_singleton_f05) if non_singleton_f05 else 0.0,
        'total_entities': len(y_true),
        'entities_with_predictions': entities_with_preds,
        'total_predicted_matches': total_predicted,
        'avg_predictions_per_entity': avg_preds,
    }


def evaluate_candidate_quality(df_candidates: pd.DataFrame, y_true: Dict[str, set]) -> dict:
    """
    Measures blocking/candidate-generation quality independently of any
    threshold or model -- exactly what the competition says it reviews
    separately from the leaderboard score ("recall ceiling, reduction ratio").

    df_candidates must have ['source1_entity_id', 'candidate_entity_id'], one
    row per (S1, candidate) pair that reached the scoring stage -- i.e. the
    same set of pairs that ends up in candidate_pairs.tsv. y_true is the
    {s1_id: set(true matches)} dict for the same entities.

    recall_ceiling: fraction of ground-truth positive pairs that appear
    ANYWHERE in df_candidates, regardless of score. This is the hard upper
    bound on recall -- no threshold or model can recover a true match that
    blocking never proposed as a candidate.
    """
    total_s1 = len(y_true)
    if df_candidates.empty:
        return {
            'avg_candidates_per_entity': 0.0,
            'median_candidates_per_entity': 0.0,
            'total_candidate_pairs': 0,
            'recall_ceiling': 1.0 if all(len(v) == 0 for v in y_true.values()) else 0.0,
            'entities_with_zero_candidates': total_s1,
        }

    cand_sets = df_candidates.groupby('source1_entity_id')['candidate_entity_id'].apply(set)

    found_true = 0
    total_true_pairs = 0
    for s1_id, true_matches in y_true.items():
        if not true_matches:
            continue
        total_true_pairs += len(true_matches)
        cands = cand_sets.get(s1_id, set())
        found_true += len(true_matches.intersection(cands))

    counts_per_entity = cand_sets.apply(len)
    return {
        'avg_candidates_per_entity': float(counts_per_entity.mean()) if len(counts_per_entity) else 0.0,
        'median_candidates_per_entity': float(counts_per_entity.median()) if len(counts_per_entity) else 0.0,
        'total_candidate_pairs': int(len(df_candidates)),
        'recall_ceiling': (found_true / total_true_pairs) if total_true_pairs > 0 else 1.0,
        'entities_with_zero_candidates': int(total_s1 - cand_sets.shape[0]),
    }


def evaluate_detailed_by_country(
    y_true: Dict[str, set],
    y_pred: Dict[str, set],
    s1_id_to_country: Dict[str, str],
) -> Dict[str, dict]:
    """
    Runs evaluate_detailed() separately per country so a regression in one
    country (e.g. France, which has zero training coverage) isn't averaged
    away by volume from the others.
    """
    by_country: Dict[str, dict] = {}
    countries = {s1_id_to_country.get(k, 'unknown') for k in y_true}
    for country in sorted(countries):
        ids = [k for k in y_true if s1_id_to_country.get(k, 'unknown') == country]
        y_true_c = {k: y_true[k] for k in ids}
        y_pred_c = {k: y_pred.get(k, set()) for k in ids}
        by_country[country] = evaluate_detailed(y_true_c, y_pred_c)
    return by_country


def optimize_threshold(
    df_scores: pd.DataFrame,
    y_true: Dict[str, set],
    thresholds: List[float] = None,
    max_s2: int = 5,
    max_s3: int = 6,
    verbose: bool = True
) -> Tuple[float, float]:
    """
    Finds the optimal probability threshold to maximize macro F_0.5.
    Applies per-source capping (top-K S2 and top-K S3) at each threshold.

    df_scores should have: ['source1_entity_id', 'candidate_entity_id', 'score']
    y_true: true matches dict mapping s1_id -> set(matches)
    """
    if thresholds is None:
        thresholds = np.arange(0.50, 0.96, 0.02).tolist()

    best_threshold = 0.5
    best_f05 = -1.0
    results = []

    for th in thresholds:
        y_pred = _apply_threshold_with_capping(df_scores, y_true, th, max_s2, max_s3)
        f05 = evaluate_macro_f05(y_true, y_pred)

        if verbose:
            details = evaluate_detailed(y_true, y_pred)
            print(f"  th={th:.2f} | F0.5={f05:.4f} | P={details['mean_precision']:.4f} "
                  f"| R={details['mean_recall']:.4f} | Sng={details['singleton_accuracy']:.4f} "
                  f"| AvgPreds={details['avg_predictions_per_entity']:.2f}")

        results.append((th, f05))
        if f05 > best_f05:
            best_f05 = f05
            best_threshold = th

    if verbose:
        print(f"\n  ★ Best threshold: {best_threshold:.2f} → Macro F0.5 = {best_f05:.4f}")

    return best_threshold, best_f05


def _apply_threshold_with_capping(
    df_scores: pd.DataFrame,
    y_true: Dict[str, set],
    threshold: float,
    max_s2: int,
    max_s3: int,
) -> Dict[str, set]:
    """
    Generate predictions at a given threshold with per-source capping.
    """
    df_pred = df_scores[df_scores['score'] >= threshold].copy()

    y_pred = {s1_id: set() for s1_id in y_true.keys()}

    if df_pred.empty:
        return y_pred

    for s1_id, group in df_pred.groupby('source1_entity_id'):
        if s1_id not in y_pred:
            continue

        # Separate S2 and S3 candidates
        s2_cands = group[group['candidate_entity_id'].str.startswith('S2-')].nlargest(max_s2, 'score')
        s3_cands = group[group['candidate_entity_id'].str.startswith('S3-')].nlargest(max_s3, 'score')

        matches = set(s2_cands['candidate_entity_id'].tolist() + s3_cands['candidate_entity_id'].tolist())
        y_pred[s1_id] = matches

    return y_pred
