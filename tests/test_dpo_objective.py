"""Validate Task 1's DPO objective against the assignment equation.

Run: python tests/test_dpo_objective.py
"""
from __future__ import annotations

import math

import numpy as np
import torch

from task1_dpo.dpo import dpo_loss


def reference_loss(pc, pr, rc, rr, beta):
    z = beta * ((np.asarray(pc) - np.asarray(rc)) - (np.asarray(pr) - np.asarray(rr)))
    return float(np.mean(np.logaddexp(0.0, -z)))


def buggy_loss(pc, pr, rc, rr, beta):
    z = beta * ((np.asarray(pc) - np.asarray(pr)) + (np.asarray(rc) - np.asarray(rr)))
    return float(np.mean(np.logaddexp(0.0, -z)))


def student_loss(pc, pr, rc, rr, beta):
    tensors = [torch.as_tensor(value, dtype=torch.float64) for value in (pc, pr, rc, rr)]
    loss, _ = dpo_loss(*tensors, beta)
    return float(loss.item())


LOSS = student_loss
RNG = np.random.default_rng(6304)


def _batch(n=16):
    rc, rr = RNG.normal(-250, 60, n), RNG.normal(-250, 60, n)
    return rc, rr


def test_step0_loss_is_ln2_for_any_reference_margin():
    for beta in (0.03, 0.1, 0.3):
        rc, rr = _batch()
        assert abs(LOSS(rc, rr, rc, rr, beta) - math.log(2)) < 1e-12


def test_loss_decreases_when_policy_margin_grows():
    rc, rr = _batch()
    base = LOSS(rc, rr, rc, rr, 0.1)
    better = LOSS(rc + 1.0, rr - 1.0, rc, rr, 0.1)
    assert better < base


def test_reference_margin_enters_with_opposite_sign():
    rc, rr = _batch()
    pc, pr = rc + 0.5, rr - 0.5
    easy = LOSS(pc, pr, rc, rr, 0.1)
    hard = LOSS(pc, pr, rc + 5.0, rr, 0.1)
    assert hard > easy


def test_gradient_signs_and_magnitude():
    n, beta, eps = 8, 0.1, 1e-4
    rc, rr = _batch(n)
    pc, pr = rc + RNG.normal(0, 1, n), rr + RNG.normal(0, 1, n)
    z = beta * ((pc - rc) - (pr - rr))
    sigma_neg_z = 1.0 / (1.0 + np.exp(z))
    for i in range(n):
        up, dn = pc.copy(), pc.copy()
        up[i] += eps
        dn[i] -= eps
        grad = (LOSS(up, pr, rc, rr, beta) - LOSS(dn, pr, rc, rr, beta)) / (2 * eps)
        assert abs(grad - (-beta * sigma_neg_z[i] / n)) < 1e-8
        up, dn = pr.copy(), pr.copy()
        up[i] += eps
        dn[i] -= eps
        grad = (LOSS(pc, up, rc, rr, beta) - LOSS(pc, dn, rc, rr, beta)) / (2 * eps)
        assert abs(grad - (beta * sigma_neg_z[i] / n)) < 1e-8


def test_matches_closed_form():
    rc, rr = _batch()
    pc, pr = rc + RNG.normal(0, 2, 16), rr + RNG.normal(0, 2, 16)
    beta = 0.3
    assert abs(LOSS(pc, pr, rc, rr, beta) - reference_loss(pc, pr, rc, rr, beta)) < 1e-12


def test_numerically_stable_for_extreme_margins():
    rc, rr = np.array([-100.0]), np.array([-100.0])
    for shift in (1e4, -1e4):
        assert math.isfinite(LOSS(rc + shift, rr - shift, rc, rr, 0.3))


def test_defect_is_detected():
    rc, rr = _batch()
    assert abs(buggy_loss(rc, rr, rc, rr, 0.1) - math.log(2)) > 1e-3
    pc, pr = rc + 0.5, rr - 0.5
    assert buggy_loss(pc, pr, rc + 5.0, rr, 0.1) < buggy_loss(pc, pr, rc, rr, 0.1)


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print("PASS", test.__name__)
    print(f"{len(tests)} tests passed (objective under test: {LOSS.__name__})")
