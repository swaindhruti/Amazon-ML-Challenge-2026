import numpy as np
import pandas as pd
from typing import Dict, Tuple


def stratified_country_split(
    df_s1: pd.DataFrame,
    df_gt: pd.DataFrame,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits Source 1 training entities into train/val/test folds, stratified by
    country_clean, so threshold tuning (val) and the final reported score
    (test) never touch the same rows -- a threshold picked on val and then
    reported on val is an optimistic, not a generalizing, number.

    df_s1 must already have a 'country_clean' column. Ground truth travels
    with the split via a left-merge on entity_id, so callers get a
    'matched_entity_ids' column directly (empty string for singletons).

    Returns (df_train, df_val, df_test), each a subset of df_s1's columns
    plus 'matched_entity_ids'.
    """
    if 'country_clean' not in df_s1.columns:
        raise ValueError("df_s1 must have a 'country_clean' column before splitting")

    merged = df_s1.merge(
        df_gt[['source1_entity_id', 'matched_entity_ids']],
        left_on='entity_id', right_on='source1_entity_id', how='left'
    )
    merged['matched_entity_ids'] = merged['matched_entity_ids'].fillna('')
    merged = merged.drop(columns=['source1_entity_id'])

    rng = np.random.RandomState(seed)
    train_parts, val_parts, test_parts = [], [], []

    for _, group in merged.groupby('country_clean'):
        idx = group.index.values.copy()
        rng.shuffle(idx)
        n = len(idx)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        test_idx = idx[:n_test]
        val_idx = idx[n_test:n_test + n_val]
        train_idx = idx[n_test + n_val:]
        train_parts.append(group.loc[train_idx])
        val_parts.append(group.loc[val_idx])
        test_parts.append(group.loc[test_idx])

    df_train = pd.concat(train_parts, ignore_index=True) if train_parts else merged.iloc[0:0]
    df_val = pd.concat(val_parts, ignore_index=True) if val_parts else merged.iloc[0:0]
    df_test = pd.concat(test_parts, ignore_index=True) if test_parts else merged.iloc[0:0]
    return df_train, df_val, df_test


def stratified_sample(
    df: pd.DataFrame,
    n: int,
    key: str = 'country_clean',
    seed: int = 42,
) -> pd.DataFrame:
    """
    Draws a random sample of ~n rows from df, proportional to each country's
    share of the whole -- instead of just taking df.iloc[:n], which silently
    trains/calibrates on whatever country happens to sort first in the file
    (a real risk: train_or_load_model previously used df_s1.iloc[:50000],
    with no guarantee the file isn't grouped by country).

    Returns a new DataFrame with its own reset RangeIndex, so callers can
    keep using positional .iloc[i] access exactly as before.
    """
    if len(df) <= n:
        return df.reset_index(drop=True).copy()

    rng = np.random.RandomState(seed)
    total = len(df)
    parts = []
    for _, group in df.groupby(key):
        group_n = max(1, int(round(n * len(group) / total)))
        idx = group.index.values.copy()
        rng.shuffle(idx)
        parts.append(group.loc[idx[:group_n]])

    result = pd.concat(parts, ignore_index=True)
    return result


def y_true_dict_from_split(df_split: pd.DataFrame) -> Dict[str, set]:
    """Builds the {s1_id: set(matched_ids)} dict evaluate.py expects, from a split DataFrame."""
    y_true = {}
    for eid, matches_str in zip(df_split['entity_id'].values, df_split['matched_entity_ids'].values):
        matches_str = str(matches_str).strip()
        y_true[eid] = set(matches_str.split(',')) if matches_str and matches_str.lower() != 'nan' else set()
    return y_true


def s1_id_to_country_map(df_split: pd.DataFrame) -> Dict[str, str]:
    """Builds the {s1_id: country_clean} lookup used for per-country evaluation breakdowns."""
    return dict(zip(df_split['entity_id'].values, df_split['country_clean'].values))
