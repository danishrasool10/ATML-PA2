"""Validate Task 3's GRPO objective (per-group advantages, clipped surrogate, length normalization).

Run: python -m pytest tests/test_grpo_objective.py   or   python -m task3_grpo.grpo
"""
from __future__ import annotations

import torch

from task3_grpo.grpo import group_relative_advantages, validate_grpo_objective


def test_validate_grpo_objective():
    validate_grpo_objective(verbose=False)


def test_identical_rewards_give_zero_advantage_not_nan():
    r = torch.full((4,), 0.1)
    for scale in (True, False):
        a = group_relative_advantages(r, torch.zeros(4, dtype=torch.long), scale_by_std=scale)
        assert torch.all(a == 0) and torch.isfinite(a).all()


def test_advantages_are_within_prompt():
    r = torch.tensor([0.0, 1.0, 10.0, 11.0])
    a = group_relative_advantages(r, torch.tensor([0, 0, 1, 1]))
    assert torch.allclose(a, torch.tensor([-1.0, 1.0, -1.0, 1.0]), atol=1e-5)


if __name__ == "__main__":
    test_validate_grpo_objective()
    test_identical_rewards_give_zero_advantage_not_nan()
    test_advantages_are_within_prompt()
    print("ok")
