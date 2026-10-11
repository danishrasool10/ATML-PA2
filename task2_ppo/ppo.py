from __future__ import annotations

import torch

from common.metrics import masked_mean


def compute_gae(rewards, values, mask, gamma=1.0, lam=0.95):
    """Token-level GAE over response positions.

    rewards, values, mask: [batch, response_steps]. Padding positions must have mask=0.
    The final valid response position bootstraps with zero.
    """
    batch, steps = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_adv = torch.zeros(batch, device=rewards.device, dtype=rewards.dtype)

    for t in reversed(range(steps)):
        current_valid = mask[:, t]
        if t + 1 < steps:
            next_valid = mask[:, t + 1]
            next_value = values[:, t + 1] * next_valid
        else:
            next_valid = torch.zeros_like(current_valid)
            next_value = torch.zeros_like(last_adv)

        delta = rewards[:, t] + gamma * next_value - values[:, t]
        last_adv = delta + gamma * lam * next_valid * last_adv
        last_adv = last_adv * current_valid
        advantages[:, t] = last_adv

    returns = advantages + values
    return advantages, returns


def shaped_rewards(task_reward, policy_logp, ref_logp, response_mask, beta_kl):
    """Sampled-action KL shaping plus terminal learned reward."""
    rewards = -float(beta_kl) * (policy_logp - ref_logp) * response_mask
    for b in range(rewards.shape[0]):
        valid = int(response_mask[b].sum().item())
        if valid > 0:
            rewards[b, valid - 1] += task_reward[b]
    return rewards


def ppo_policy_loss(new_logp, old_logp, advantage, mask, eps=0.2):
    """Return PPO clipped policy loss and diagnostics.

    Validate this starter implementation against the clipped surrogate in the assignment manual.
    """
    ratio = torch.exp(new_logp - old_logp)
    surr1 = ratio * advantage
    surr2 = ratio.clamp(1.0 - eps, 1.0 + eps) * advantage

    # >>> DEFECT FIX (the one deliberate bug in this file) <<<
    # Starter used torch.maximum(surr1, surr2). The PPO objective is the PESSIMISTIC bound
    #   L_clip = E[min(rho*A, clip(rho, 1-eps, 1+eps)*A)],
    # so the elementwise op must be torch.minimum. With maximum, a token with A>0 and rho>1+eps (or
    # A<0 and rho<1-eps) keeps its UNclipped, ever-growing gradient, i.e. the trust region is inverted
    # and clipping rewards moving further out of it instead of stopping.
    objective = torch.minimum(surr1, surr2)

    loss = -masked_mean(objective, mask)
    affected = ((ratio < (1.0 - eps)) | (ratio > (1.0 + eps))).float()
    clip_fraction = masked_mean(affected, mask)
    return loss, ratio.detach(), clip_fraction.detach()


def value_mse_loss(predicted_values, returns, mask):
    return masked_mean((predicted_values - returns) ** 2, mask)


def normalize_advantages(advantages, mask, eps=1e-6):
    valid = advantages[mask.bool()]
    if valid.numel() <= 1:
        return advantages
    mean = valid.mean()
    std = valid.std(unbiased=False).clamp_min(eps)
    return ((advantages - mean) / std) * mask


def clipping_diagnostics(ratio, advantage, mask, eps=0.2):
    """Extra clipping geometry for the clipping study (does not change ppo_policy_loss' return signature).

    clip_fraction_outside : fraction of tokens with rho outside [1-eps, 1+eps]  (== ppo_policy_loss' clip_fraction)
    clip_fraction_active  : fraction of tokens where the clip branch is the active minimum, i.e. the gradient is
                            actually zeroed (A>0 & rho>1+eps) | (A<0 & rho<1-eps)
    max_abs_ratio_dev     : max |rho - 1| over valid tokens
    """
    ratio = ratio.detach()
    advantage = advantage.detach()
    m = mask.bool()
    outside = ((ratio < 1.0 - eps) | (ratio > 1.0 + eps)).float()
    active = (((advantage > 0) & (ratio > 1.0 + eps)) | ((advantage < 0) & (ratio < 1.0 - eps))).float()
    dev = (ratio - 1.0).abs()
    return {
        "clip_fraction_outside": float(masked_mean(outside, mask)),
        "clip_fraction_active": float(masked_mean(active, mask)),
        "max_abs_ratio_dev": float(dev[m].max()) if m.any() else 0.0,
    }


def validate_ppo_objective(verbose: bool = True) -> None:
    """CPU self-test of the objective against a brute-force scalar reference and the clipping invariants.
    Fails loudly (AssertionError) if the min/max defect (or any similar sign/geometry bug) is present."""
    g = torch.Generator().manual_seed(0)
    B, T = 4, 17
    for eps in (0.05, 0.20, 0.50):
        old = torch.randn(B, T, generator=g) * 0.5 - 2.0
        new = (old + torch.randn(B, T, generator=g) * 0.4).requires_grad_(True)
        adv = torch.randn(B, T, generator=g)
        mask = (torch.rand(B, T, generator=g) > 0.2).float()
        mask[:, 0] = 1.0

        loss, ratio, clipfrac = ppo_policy_loss(new, old, adv, mask, eps)

        # 1) brute-force scalar reference of -E[min(rho*A, clip(rho,1-eps,1+eps)*A)]
        tot, n = 0.0, 0.0
        for b in range(B):
            for t in range(T):
                if mask[b, t] == 0:
                    continue
                r = float(torch.exp(new[b, t] - old[b, t]))
                a = float(adv[b, t])
                tot += min(r * a, min(max(r, 1 - eps), 1 + eps) * a)
                n += 1
        assert abs(float(loss) + tot / n) < 1e-4, f"eps={eps}: loss {float(loss)} != reference {-tot / n}"

        # 2) pessimism: surrogate must never exceed the unclipped objective (this is what maximum() violates)
        unclipped = float(((ratio * adv) * mask).sum() / mask.sum())
        assert -float(loss) <= unclipped + 1e-6, f"eps={eps}: clipped surrogate above unclipped (max instead of min?)"

        # 3) gradient is exactly zero on tokens where the clip branch is active
        loss.backward()
        active = (((adv > 0) & (ratio > 1 + eps)) | ((adv < 0) & (ratio < 1 - eps))) & mask.bool()
        assert torch.all(new.grad[active] == 0), f"eps={eps}: non-zero gradient on clipped tokens"
        assert torch.all(new.grad[(~active) & mask.bool()] != 0)
        assert torch.all(new.grad[~mask.bool()] == 0), "padding tokens must not receive gradient"

        # 4) diagnostics agree with ppo_policy_loss
        d = clipping_diagnostics(ratio, adv, mask, eps)
        assert abs(d["clip_fraction_outside"] - float(clipfrac)) < 1e-6
        assert d["clip_fraction_active"] <= d["clip_fraction_outside"] + 1e-9

    # GAE: lambda=1, gamma=1, zero values -> advantage == reward-to-go; masked tail is zero
    r = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    v = torch.zeros(1, 4)
    m = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    adv, ret = compute_gae(r, v, m, gamma=1.0, lam=1.0)
    assert torch.allclose(adv, torch.tensor([[1.0, 1.0, 1.0, 0.0]])) and torch.allclose(ret, adv)
    if verbose:
        print("[validate_ppo_objective] clipped surrogate (min), gradient masking, GAE: OK")


if __name__ == "__main__":
    validate_ppo_objective()
