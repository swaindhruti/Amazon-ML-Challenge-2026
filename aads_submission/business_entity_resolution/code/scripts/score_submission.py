"""
Scores a matching_results.tsv against a ground-truth TSV with the competition's
macro F0.5 (singletons included), and prints the same detailed breakdown the
pipeline's --validate mode prints. Works for any GT in the
`source1_entity_id / matched_entity_ids` format, e.g. the test_ground_truth.tsv
that scripts/make_subworld.py writes.

    python scripts/score_submission.py --pred output/matching_results.tsv \
        --truth subworld/test/test_ground_truth.tsv
"""
import argparse
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from src.evaluate import evaluate_macro_f05, evaluate_detailed  # noqa: E402


def read_lists(path):
    df = pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False)
    out = {}
    for a, b in zip(df.iloc[:, 0].values, df.iloc[:, 1].values):
        out[a] = set(x for x in b.split(',') if x)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True)
    ap.add_argument('--truth', required=True)
    args = ap.parse_args()
    truth, pred = read_lists(args.truth), read_lists(args.pred)
    missing = [k for k in truth if k not in pred]
    if missing:
        print(f"WARNING: {len(missing)} truth entities missing from predictions (scored as empty)")
    print(f"Macro F0.5 = {evaluate_macro_f05(truth, pred):.4f}")
    for k, v in evaluate_detailed(truth, pred).items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == '__main__':
    main()
