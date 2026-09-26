"""Plain-assert tests (no pytest needed):  python -m tests.test_decision   (from the code/ dir, PYTHONPATH=.)"""
import numpy as np
from src.decision import owner_normalize, cap_and_threshold


def test_single_candidate_unchanged():
    p = np.array([0.9, 0.3, 0.05], dtype=np.float32)
    out = owner_normalize(np.array([0, 1, 2]), p, 3)
    assert np.allclose(out, p, atol=1e-4), out


def test_competition_pushes_loser_down():
    # target 0 has two competing owners (0.9 and 0.6); target 1 has one (0.6)
    t = np.array([0, 0, 1])
    p = np.array([0.9, 0.6, 0.6], dtype=np.float32)
    out = owner_normalize(t, p, 2)
    assert out[0] > out[1], out
    assert out[1] < 0.2 and out[0] > 0.7, out          # 9/(1+9+1.5)=0.78, 1.5/11.5=0.13
    assert abs(out[2] - 0.6) < 1e-4                        # uncontested target untouched


def test_alpha_zero_is_uniform_competition():
    out = owner_normalize(np.array([0, 0]), np.array([0.99, 0.01], dtype=np.float32), 1, alpha=0.0)
    assert np.allclose(out, [1 / 3, 1 / 3], atol=1e-5)


def test_caps_and_order():
    # entity 0: five S2 candidates + two S3 above threshold, caps (2, 1); entity 1: one S3
    q = np.array([0, 0, 0, 0, 0, 0, 0, 1])
    t = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    is_s2 = np.array([1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)
    s = np.array([.9, .8, .7, .95, .6, .85, .99, .5], dtype=np.float32)
    kq, kt, ks = cap_and_threshold(q, t, s, is_s2, 0.55, max_s2=2, max_s3=1)
    got = sorted(zip(kq.tolist(), kt.tolist()))
    # S2 top-2 by score: t3 (.95), t0 (.9); S3 top-1: t6 (.99); entity 1: t7 (.5) is below threshold
    assert got == [(0, 0), (0, 3), (0, 6)], got
    assert ks[0] == np.float32(.99)                        # best first within the entity


def test_empty():
    e = np.array([], dtype=np.int32)
    kq, kt, ks = cap_and_threshold(e, e.astype(np.uint32), np.array([], dtype=np.float32),
                                   np.array([True]), 0.5)
    assert len(kq) == 0
    assert len(owner_normalize(e, np.array([], dtype=np.float32), 0)) == 0


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
