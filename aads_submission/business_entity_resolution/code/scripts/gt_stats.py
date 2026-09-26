"""
Prints the ground-truth facts the pipeline's design relies on, so anyone can re-verify them:

  * singleton rate, matches per non-singleton, per-source match-count distributions
  * how many matched target records are claimed by more than one S1 entity
    (the one-owner rule in src/decision.py assumes this is 0)
  * what fraction of target records is owned by some S1 entity
  * what fraction of matched pairs cross a country boundary
    (strict per-country blocking assumes this is ~0; any cross-country ground-truth
    pair is a recall loss no per-country pipeline can recover)

Needs a few GB of RAM (it joins ~7.6M matched pairs against the ~10M target records),
so run it on the SageMaker instance, not a laptop:

    python scripts/gt_stats.py --data_dir student_resource/dataset
"""
import argparse
import os

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    args = ap.parse_args()
    d = os.path.join(args.data_dir, 'train')
    rd = dict(sep='\t', dtype=str, keep_default_na=False)

    gt = pd.read_csv(os.path.join(d, 'train_ground_truth.tsv'), **rd)
    s1 = pd.read_csv(os.path.join(d, 'train_source1.tsv'), usecols=['entity_id', 'country'], **rd)
    targets = pd.concat([
        pd.read_csv(os.path.join(d, 'train_source2.tsv'), usecols=['entity_id', 'country'], **rd),
        pd.read_csv(os.path.join(d, 'train_source3.tsv'), usecols=['entity_id', 'country'], **rd),
    ], ignore_index=True)

    lists = gt['matched_entity_ids'].str.split(',')
    n_match = lists.map(lambda l: 0 if l == [''] else len(l))
    non_single = n_match > 0
    print(f"S1 entities: {len(gt):,} | singletons: {(~non_single).mean():.2%}")
    print(f"matches per non-singleton: mean {n_match[non_single].mean():.2f}, "
          f"median {n_match[non_single].median():.0f}, max {n_match.max()}")
    s2 = lists.map(lambda l: sum(1 for x in l if x.startswith('S2-')))
    s3 = lists.map(lambda l: sum(1 for x in l if x.startswith('S3-')))
    print("S2 matches / non-singleton:", s2[non_single].value_counts(normalize=True).sort_index().round(3).to_dict())
    print("S3 matches / non-singleton:", s3[non_single].value_counts(normalize=True).sort_index().round(3).to_dict())
    print(f"non-singletons matched in both S2 and S3: {((s2 > 0) & (s3 > 0))[non_single].mean():.2%}")

    pairs = gt.assign(t=lists).explode('t')
    pairs = pairs[pairs['t'].fillna('') != ''][['source1_entity_id', 't']]
    counts = pairs['t'].value_counts()
    print(f"\nmatched (S1, target) pairs: {len(pairs):,} | distinct targets: {len(counts):,} | "
          f"targets claimed by more than one S1 entity: {(counts > 1).sum():,}")
    print(f"share of the {len(targets):,} target records owned by some S1 entity: {len(counts) / len(targets):.2%}")

    m = (pairs.merge(s1.rename(columns={'entity_id': 'source1_entity_id', 'country': 'c_s1'}),
                     on='source1_entity_id', how='left')
              .merge(targets.rename(columns={'entity_id': 't', 'country': 'c_t'}), on='t', how='left'))
    known = m['c_s1'].notna() & m['c_t'].notna()
    cross = (m.loc[known, 'c_s1'].str.strip().str.lower() != m.loc[known, 'c_t'].str.strip().str.lower())
    print(f"\nmatched pairs whose S1 and target country differ: {int(cross.sum()):,} of {int(known.sum()):,} "
          f"({cross.mean():.3%})")
    if cross.any():
        print(m.loc[known][cross.values].groupby(['c_s1', 'c_t']).size().sort_values(ascending=False).head(10).to_string())


if __name__ == '__main__':
    main()
