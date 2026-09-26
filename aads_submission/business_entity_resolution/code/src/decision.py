"""
Turning per-pair match probabilities into the final matches.

The pair classifier scores each (S1, candidate) pair on its own. But the data
has a hard structural fact that no pairwise score can see: in the real ground
truth every matched Source 2/3 record belongs to exactly ONE Source 1 entity
(7,638,365 matched targets, 7,638,365 distinct, 0 shared; ~74% of all target
records are owned by some S1 entity). So when two S1 entities both look like a
plausible owner of the same target, at most one can be right, and the pairwise
scores alone would happily give it to both -- on a real-data proxy, ~40% of the
highest-scoring false positives were targets owned by a different S1 entity
whose own match was simply better.

owner_normalize() applies that constraint softly: a Luce-style choice model in
which a target is owned by candidate i with probability w_i / (1 + sum_j w_j),
w = odds(p)^alpha, and the "1" is the option "owned by none of them" (26% of
targets are un-owned distractors). For a target with a single candidate it
reduces to the original p (alpha = 1); when several candidates compete, the
clear winner keeps most of its probability and the rest are pushed down.
Measured on the closed sub-world (train-world tuned, held-out test world):
0.9650 -> 0.9713 macro F0.5, stable for alpha in [0.5, 2] and with or without
probability calibration, and better than a hard "best owner only" filter
(0.9673). A learned second stage over per-entity context features did NOT beat
it on the held-out world (0.9631), so it isn't used.
"""
import numpy as np

P_FLOOR = 0.02  # pairs scored below this are dropped before decisions (memory only)


def owner_normalize(t_idx: np.ndarray, p: np.ndarray, n_targets: int,
                    alpha: float = 1.0, eps: float = 1e-6) -> np.ndarray:
    """
    Competition-aware match probability. t_idx are target indices (into one
    country's target pool) aligned with p; ALL scored pairs of the country must
    be present (in any order), since the competitors for a target come from
    different S1 batches.
    """
    if len(p) == 0:
        return np.zeros(0, dtype=np.float32)
    p = np.clip(p.astype(np.float64), eps, 1.0 - eps)
    w = (p / (1.0 - p)) ** alpha
    sums = np.bincount(t_idx, weights=w, minlength=n_targets)
    return (w / (1.0 + sums[t_idx])).astype(np.float32)


def cap_and_threshold(q_idx: np.ndarray, t_idx: np.ndarray, score: np.ndarray,
                      target_is_s2: np.ndarray, threshold: float,
                      max_s2: int = 5, max_s3: int = 6):
    """
    Keeps pairs with score >= threshold, then at most the top max_s2 Source 2 and
    max_s3 Source 3 candidates per query (the caps come from the ground-truth
    cardinalities: S2 <= 5, S3 <= 6 per S1 entity). Returns (q, t, score) sorted
    by query then best score first.
    """
    keep = score >= threshold
    q, t, s = q_idx[keep], t_idx[keep], score[keep]
    if len(q) == 0:
        return q, t, s
    src = target_is_s2[t]
    order = np.lexsort((-s, src, q))
    q, t, s, src = q[order], t[order], s[order], src[order]
    new_group = np.r_[True, (q[1:] != q[:-1]) | (src[1:] != src[:-1])]
    starts = np.flatnonzero(new_group)
    group_start = np.repeat(starts, np.diff(np.r_[starts, len(q)]))
    rank = np.arange(len(q)) - group_start
    ok = rank < np.where(src, max_s2, max_s3)
    q, t, s = q[ok], t[ok], s[ok]
    order = np.lexsort((-s, q))
    return q[order], t[order], s[order]
