from __future__ import annotations

import torch

from common.metrics import masked_mean, sample_entropy


def group_statistics(rewards: torch.Tensor, group_ids: torch.Tensor, eps: float = 1e-6, *,
                     scale_by_std: bool = True, std_gate: float = 1e-6) -> dict:
    """Per-prompt reward statistics and group-relative advantages.

    A_k = (r_k - mu_g) / (sigma_g + eps)   (scale_by_std=True,  canonical GRPO)
    A_k =  r_k - mu_g                      (scale_by_std=False, Dr. GRPO)
    sigma_g is the population std of the K rewards of prompt g. Groups with sigma_g < std_gate are uninformative and
    receive A = 0 exactly.
    """
    # >>> DEFECT FIX <<< Starter normalized with the whole-batch mean/std and ignored group_ids. The GRPO baseline is
    # the mean/std of the K completions of the SAME prompt (Shao et al. 2024, Eq. 3, Sec. 4.1.2); batch statistics
    # make A encode prompt difficulty and never map a zero-variance group to A = 0.
    r = rewards.detach().float()
    _, inv, counts = torch.unique(group_ids, return_inverse=True, return_counts=True)
    n = counts.to(r)
    mean = torch.zeros_like(n).index_add_(0, inv, r) / n
    centered = r - mean[inv]
    var = torch.zeros_like(n).index_add_(0, inv, centered.square()) / n
    std = var.sqrt()
    informative = std >= std_gate
    adv = centered / (std[inv] + eps) if scale_by_std else centered
    adv = torch.where(informative[inv], adv, torch.zeros_like(adv))
    return {
        "advantages": adv.to(rewards.dtype),
        "group_index": inv,
        "group_mean": mean,
        "group_std": std,
        "group_std_unbiased": (var * n / (n - 1.0).clamp_min(1.0)).sqrt(),
        "informative": informative,
    }


def group_relative_advantages(rewards: torch.Tensor, group_ids: torch.Tensor, eps: float = 1e-6, *,
                              scale_by_std: bool = True, std_gate: float = 1e-6) -> torch.Tensor:
    """One scalar advantage per sampled completion; `group_ids[i]` identifies the prompt that produced `rewards[i]`."""
    return group_statistics(rewards, group_ids, eps, scale_by_std=scale_by_std, std_gate=std_gate)["advantages"]


def grpo_policy_loss(
    new_logp,
    old_logp,
    seq_adv,
    token_mask,
    ref_logp,
    eps,
    beta,
    loss_type="grpo",
    max_completion_length: int | None = None,
    *,
    n_seq: float | torch.Tensor | None = None,
    n_tok: float | torch.Tensor | None = None,
):
    """PPO-style clipped GRPO loss for already-sampled completions.

    `token_mask` is all-zero for a completion masked because it hit the generation cap. `n_seq` / `n_tok` are the
    number of unmasked sequences / tokens in the *full* update batch; passing them makes the sum of micro-batch losses
    equal the full-batch loss. Defaults: this batch's own counts.
    """
    ratio = torch.exp(new_logp - old_logp)
    adv = seq_adv[:, None]
    objective = torch.minimum(ratio * adv, ratio.clamp(1.0 - eps, 1.0 + eps) * adv)

    lengths = token_mask.sum(-1)
    token_sum = (objective * token_mask).sum(-1)
    if n_seq is None:
        n_seq = (lengths > 0).sum().clamp_min(1).to(objective.dtype)
    if n_tok is None:
        n_tok = lengths.sum().clamp_min(1.0)

    if loss_type == "grpo":
        per_sequence = token_sum / lengths.clamp_min(1.0)  # 1/T_k, Shao et al. 2024 Eq. 3
    elif loss_type == "dr_grpo":
        if max_completion_length is None:
            raise ValueError("dr_grpo requires max_completion_length")
        per_sequence = token_sum / float(max_completion_length)  # Liu et al. 2025 (Dr. GRPO): constant, not |o_i|
    else:
        raise ValueError(f"Unknown loss_type={loss_type!r}")
    # E_k over unmasked completions only (starter averaged over all B rows, so masked rows diluted the step).
    policy_term = -per_sequence.sum() / n_seq

    log_ratio_ref_over_policy = ref_logp - new_logp
    per_token_kl = torch.exp(log_ratio_ref_over_policy) - log_ratio_ref_over_policy - 1.0  # Shao et al. 2024 Eq. 4
    kl = (per_token_kl * token_mask).sum() / n_tok
    loss = policy_term + float(beta) * kl
    affected = ((ratio < (1.0 - eps)) | (ratio > (1.0 + eps))).float()
    return loss, {
        "policy_term": policy_term.detach(),
        "sampled_kl": kl.detach(),
        "clip_fraction": masked_mean(affected, token_mask).detach(),
        "ratio_mean": masked_mean(ratio.detach(), token_mask),
        "sample_entropy": sample_entropy(new_logp.detach(), token_mask),
        "valid_sequences": (lengths > 0).sum().detach(),
    }


def mask_truncated_sequences(token_mask: torch.Tensor, truncated: list[bool] | torch.Tensor):
    truncated = torch.as_tensor(truncated, device=token_mask.device, dtype=torch.bool)
    keep = (~truncated).to(token_mask.dtype)[:, None]
    return token_mask * keep


# ===================================================================================================== validation
def _brute_advantages(r, g, eps, scale, gate):
    out = []
    for i in range(len(r)):
        grp = [r[j] for j in range(len(r)) if g[j] == g[i]]
        mu = sum(grp) / len(grp)
        sd = (sum((x - mu) ** 2 for x in grp) / len(grp)) ** 0.5
        out.append(0.0 if sd < gate else ((r[i] - mu) / (sd + eps) if scale else r[i] - mu))
    return out


def _brute_loss(new, old, adv, mask, ref, eps, beta, loss_type, L):
    B, T = mask.shape
    pol, n_seq, kl, n_tok = 0.0, 0, 0.0, 0.0
    for b in range(B):
        Tb = float(mask[b].sum())
        if Tb == 0:
            continue
        n_seq += 1
        s = 0.0
        for t in range(T):
            if mask[b, t] == 0:
                continue
            rho, a = float(torch.exp(new[b, t] - old[b, t])), float(adv[b])
            s += min(rho * a, min(max(rho, 1 - eps), 1 + eps) * a)
            x = float(ref[b, t] - new[b, t])
            kl += float(torch.exp(torch.tensor(x))) - x - 1.0
            n_tok += 1
        pol += s / (Tb if loss_type == "grpo" else L)
    return -pol / max(n_seq, 1) + beta * kl / max(n_tok, 1.0)


@torch.no_grad()
def _starter_batch_advantages(rewards, eps=1e-6):
    return (rewards - rewards.mean()) / rewards.std(unbiased=False).clamp_min(eps)


def validate_grpo_objective(verbose: bool = True) -> None:
    """CPU self-test against brute-force scalar references. Raises AssertionError on any deviation."""
    g = torch.Generator().manual_seed(0)

    # ---- advantages: interleaved groups with different offsets, a zero-variance group and a float-residual tie group
    K, G = 4, 6
    gids = torch.arange(G).repeat_interleave(K)[torch.randperm(G * K, generator=g)]
    r = torch.randn(G * K, generator=g) + (gids.float() - 2.5) * 3.0
    r[gids == 4] = 1.8359375
    r[gids == 5] = 0.1
    for scale in (True, False):
        st = group_statistics(r, gids, 1e-6, scale_by_std=scale, std_gate=1e-6)
        ref = _brute_advantages(r.tolist(), gids.tolist(), 1e-6, scale, 1e-6)
        assert torch.allclose(st["advantages"], torch.tensor(ref), atol=1e-5), "advantages != per-group reference"
        assert torch.all(st["advantages"][(gids == 4) | (gids == 5)] == 0), "zero-variance group must give A == 0"
        assert st["informative"].tolist() == [True] * 4 + [False] * 2
        shift = torch.randn(G, generator=g)[gids] * 10.0
        assert torch.allclose(group_relative_advantages(r + shift, gids, scale_by_std=scale), st["advantages"],
                              atol=1e-4), "advantages must be invariant to a per-prompt reward offset"
    adv_std = group_relative_advantages(r, gids)
    for k in range(4):
        ak = adv_std[gids == k]
        assert abs(float(ak.mean())) < 1e-5 and abs(float(ak.std(unbiased=False)) - 1.0) < 1e-4
    assert not torch.allclose(_starter_batch_advantages(r), adv_std, atol=1e-3), \
        "self-test lacks power: starter batch normalization would pass"

    # ---- loss: brute force, pessimism, clipped-token gradients, truncated rows, padding
    B, T, L = 6, 13, 16
    for loss_type in ("grpo", "dr_grpo"):
        for eps in (0.05, 0.2, 0.5):
            for beta in (0.0, 0.1):
                old = torch.randn(B, T, generator=g) * 0.5 - 2.0
                new = (old + torch.randn(B, T, generator=g) * 0.4).requires_grad_(True)
                ref = old + torch.randn(B, T, generator=g) * 0.3
                adv = torch.randn(B, generator=g)
                lens = torch.randint(1, T + 1, (B,), generator=g)
                mask = (torch.arange(T)[None] < lens[:, None]).float()
                mask = mask_truncated_sequences(mask, [False, False, True, False, False, False])
                assert mask[2].sum() == 0 and mask[[0, 1, 3, 4, 5]].sum() == lens[[0, 1, 3, 4, 5]].sum()

                loss, d = grpo_policy_loss(new, old, adv, mask, ref, eps, beta, loss_type, L)
                bf = _brute_loss(new.detach(), old, adv, mask, ref, eps, beta, loss_type, L)
                assert abs(float(loss.detach()) - bf) < 1e-5, f"{loss_type} eps={eps} beta={beta}: {float(loss.detach())} != {bf}"

                rho = torch.exp(new.detach() - old)
                D = mask.sum(-1).clamp_min(1.0) if loss_type == "grpo" else torch.full((B,), float(L))
                unclipped = float(((rho * adv[:, None] * mask).sum(-1) / D).sum() / (mask.sum(-1) > 0).sum())
                assert -float(d["policy_term"]) <= unclipped + 1e-6, "clipped surrogate exceeds unclipped (max?)"

                loss.backward()
                active = (((adv[:, None] > 0) & (rho > 1 + eps)) | ((adv[:, None] < 0) & (rho < 1 - eps))) & mask.bool()
                assert torch.all(new.grad[~mask.bool()] == 0), "masked / padded tokens must not receive gradient"
                if beta == 0.0:
                    assert torch.all(new.grad[active] == 0), "non-zero gradient on clip-active tokens"

    # ---- normalization: at rho == 1, beta == 0, dL/dlogp[b,t] = -A_b / (D_b * N_valid), D_b = T_b or L_max
    old = torch.randn(B, T, generator=g)
    adv = torch.randn(B, generator=g)
    lens = torch.tensor([1, 3, 5, 8, 13, 13])
    mask = mask_truncated_sequences((torch.arange(T)[None] < lens[:, None]).float(), [0, 0, 0, 0, 0, 1])
    for loss_type in ("grpo", "dr_grpo"):
        new = old.clone().requires_grad_(True)
        grpo_policy_loss(new, old, adv, mask, old, 0.2, 0.0, loss_type, L)[0].backward()
        D = lens.float() if loss_type == "grpo" else torch.full((B,), float(L))
        expect = (-adv / (D * 5.0))[:, None] * mask
        assert torch.allclose(new.grad, expect, atol=1e-7), f"{loss_type}: token weights != -A/(D*N)"
        mass = (new.grad.abs().sum(-1) / adv.abs())[:5]
        if loss_type == "grpo":
            assert torch.allclose(mass, torch.full_like(mass, 0.2)), "grpo sequence mass must be length-independent"
        else:
            assert torch.allclose(mass, lens[:5].float() / (L * 5.0)), "dr_grpo sequence mass must scale with T_k"

    # ---- micro-batch accumulation with global normalizers reproduces the full-batch gradient
    new = (old + 0.1 * torch.randn(B, T, generator=g)).requires_grad_(True)
    ref = old + 0.2 * torch.randn(B, T, generator=g)
    for loss_type in ("grpo", "dr_grpo"):
        full = torch.autograd.grad(grpo_policy_loss(new, old, adv, mask, ref, 0.2, 0.1, loss_type, L)[0], new)[0]
        n_seq, n_tok = (mask.sum(-1) > 0).sum().float(), mask.sum()
        acc = torch.zeros_like(new)
        for b in range(B):
            sl = slice(b, b + 1)
            lb, _ = grpo_policy_loss(new[sl], old[sl], adv[sl], mask[sl], ref[sl], 0.2, 0.1, loss_type, L,
                                     n_seq=n_seq, n_tok=n_tok)
            acc += torch.autograd.grad(lb, new)[0]
        assert torch.allclose(full, acc, atol=1e-7), f"{loss_type}: micro-batch accumulation != full batch"

    # ---- fully truncated group: finite zero loss, zero gradient; KL estimator is >= 0 and 0 at new == ref
    new = old.clone().requires_grad_(True)
    zero = torch.zeros_like(mask)
    lz, _ = grpo_policy_loss(new, old, adv, zero, ref, 0.2, 0.1, "grpo", L)
    lz.backward()
    assert float(lz.detach()) == 0.0 and torch.all(new.grad == 0)
    _, dk = grpo_policy_loss(old, old, adv, mask, old, 0.2, 0.1, "grpo", L)
    _, dk2 = grpo_policy_loss(old, old, adv, mask, ref, 0.2, 0.1, "grpo", L)
    assert float(dk["sampled_kl"]) == 0.0 and float(dk2["sampled_kl"]) > 0.0

    if verbose:
        print("[validate_grpo_objective] per-group advantages + zero-variance gate, clipped surrogate (min), "
              "1/T_k vs 1/L_max token weights, truncation masking, micro-batch accumulation, k3 KL: OK")


if __name__ == "__main__":
    validate_grpo_objective()
