"""
Builds a small, CLOSED sub-world from the real training data, laid out exactly
like the real dataset (train/ and test/ folders), so the whole pipeline can be
run end-to-end in minutes instead of hours -- and so a train->test run has
ground truth to score against (the real test set has none).

Why "closed": in the real data every matched Source 2/3 record is owned by
exactly ONE Source 1 entity (verified: 7,638,365 matched targets, 7,638,365
distinct, 0 shared) and ~74% of all target records are owned. A random sample
of S1 rows breaks that structure (most sampled entities' rivals are missing).
Instead this keeps a whole REGION of S1 entities -- every entity whose name
starts with one of --letters (default rare letters z,y,q,x, ~2% of S1) --
together with every target those entities own, plus all the un-owned
("distractor") targets whose cleaned name starts with the same letters. Within
the region, the density of similar-looking names, the ownership rate, the
singleton rate and the noise are the real ones; what's missing is cross-region
confusers (e.g. a same-street business whose name starts with another
letter), so absolute scores here are OPTIMISTIC vs the full dataset. Use it to
compare approaches and to smoke-test, not to predict the leaderboard number.

The region is split into a train world and a test world (--test_frac); each
gets its own pool. The test world also gets test_ground_truth.tsv (not part of
the real dataset) so scripts/score_submission.py can score a real inference
run against it.

Usage:
    python scripts/make_subworld.py --data_dir student_resource/dataset --out_dir subworld
    python -m src.pipeline --data_dir subworld --is_train --validate
    python -m src.pipeline --data_dir subworld --matching_out output/sub_match.tsv ...
    python scripts/score_submission.py --pred output/sub_match.tsv --truth subworld/test/test_ground_truth.tsv
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from src.preprocessing import clean_text  # noqa: E402

_LETTER = re.compile(r'[a-z]')


def id_to_int(ids: pd.Series) -> np.ndarray:
    """'S2-123' -> 2*10**10 + 123, so S2 and S3 ids with equal numbers never collide."""
    num = ids.str.slice(3).astype(np.int64).values
    src = ids.str.slice(1, 2).astype(np.int64).values
    return src * 10**10 + num


def first_letter_clean(name: str) -> str:
    m = _LETTER.search(clean_text(name))
    return m.group(0) if m else '_'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True, help='dataset dir containing train/')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--letters', default='zyqx', help='S1 name first letters defining the region')
    ap.add_argument('--test_frac', type=float, default=0.35)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    src = os.path.join(args.data_dir, 'train')
    letters = set(args.letters.lower())
    rng = np.random.RandomState(args.seed)
    rd = dict(sep='\t', dtype=str, keep_default_na=False)

    print("Reading S1 + ground truth...")
    s1 = pd.read_csv(os.path.join(src, 'train_source1.tsv'), **rd)
    gt = pd.read_csv(os.path.join(src, 'train_ground_truth.tsv'), **rd)

    fl = s1['business_name'].str.lower().str.extract(r'([a-z])', expand=False).fillna('_')
    region = s1[fl.isin(letters)].copy()
    print(f"Region '{args.letters}': {len(region)} of {len(s1)} S1 entities ({len(region)/len(s1):.2%})")

    region = region.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    n_test = int(round(len(region) * args.test_frac))
    worlds = {'test': region.iloc[:n_test], 'train': region.iloc[n_test:]}

    gt = gt.set_index('source1_entity_id')
    exploded = gt['matched_entity_ids'].str.split(',').explode()
    exploded = exploded[exploded != '']
    all_owned = np.sort(id_to_int(exploded.reset_index(drop=True)))
    print(f"Owned targets overall: {len(all_owned)}")

    owned_by_world, gt_rows = {}, {}
    for name, w in worlds.items():
        sub = gt.reindex(w['entity_id'].values)
        gt_rows[name] = sub.reset_index()
        ex = sub['matched_entity_ids'].str.split(',').explode()
        ex = ex[ex.notna() & (ex != '')]
        owned_by_world[name] = np.sort(id_to_int(ex.reset_index(drop=True)))
        print(f"  {name}: {len(w)} S1, {len(owned_by_world[name])} owned targets, "
              f"singleton rate {(sub['matched_entity_ids'] == '').mean():.3f}")

    def member(sorted_arr, x):
        pos = np.searchsorted(sorted_arr, x)
        pos[pos == len(sorted_arr)] = 0
        return sorted_arr[pos] == x

    out = {name: {'S2': [], 'S3': []} for name in worlds}
    for tag, fname in (('S2', 'train_source2.tsv'), ('S3', 'train_source3.tsv')):
        print(f"Streaming {fname}...")
        for chunk in pd.read_csv(os.path.join(src, fname), chunksize=500_000, **rd):
            ints = id_to_int(chunk['entity_id'])
            owned_any = member(all_owned, ints)
            chosen = np.full(len(chunk), '', dtype=object)
            for name in worlds:
                chosen[member(owned_by_world[name], ints)] = name
            # un-owned distractors from the same region, split randomly between worlds
            cand = np.nonzero(~owned_any)[0]
            if len(cand):
                names = chunk['business_name'].values[cand]
                in_region = np.fromiter((first_letter_clean(n) in letters for n in names),
                                        dtype=bool, count=len(cand))
                pick = cand[in_region]
                to_test = rng.rand(len(pick)) < args.test_frac
                chosen[pick[to_test]] = 'test'
                chosen[pick[~to_test]] = 'train'
            for name in worlds:
                sel = chosen == name
                if sel.any():
                    out[name][tag].append(chunk[sel])

    for name, w in worlds.items():
        d = os.path.join(args.out_dir, name)
        os.makedirs(d, exist_ok=True)
        w[['entity_id', 'business_name', 'business_address', 'country']].to_csv(
            os.path.join(d, f'{name}_source1.tsv'), sep='\t', index=False)
        for tag, n in (('S2', 2), ('S3', 3)):
            df = pd.concat(out[name][tag], ignore_index=True)
            df.to_csv(os.path.join(d, f'{name}_source{n}.tsv'), sep='\t', index=False)
            print(f"  {name} source{n}: {len(df)} rows")
        gt_out = gt_rows[name].rename(columns={'index': 'source1_entity_id'})
        gt_out.to_csv(os.path.join(d, 'train_ground_truth.tsv' if name == 'train' else 'test_ground_truth.tsv'),
                      sep='\t', index=False)
    print(f"Done -> {args.out_dir}")


if __name__ == '__main__':
    main()
