from __future__ import annotations

import argparse

import pandas as pd
import torch

from common.data import load_yaml, prompt_messages, repo_path
from common.generation import score_reward_pairs
from common.logging_utils import save_json
from common.metrics import masked_mean
from task2_ppo.continue_train import (
    advantage_pipeline,
    midpoint_baseline,
    ppo_update,
    prepare_ppo_continuation,
    reset_to_midpoint,
    run_fork,
)
from task2_ppo.ppo import clipping_diagnostics, ppo_policy_loss
from task2_ppo.utils import policy_forward_stats

_REWARD_KEYS = ["reward", "task_reward", "rm_reward", "reward_score", "score"]


def load_cached_rollouts(path):
    rows = torch.load(repo_path(path), map_location="cpu", weights_only=False)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty list in the supplied PPO rollout cache")

    # Instructor iterations used two equivalent names for these fields. Normalize once here so
    # the student analysis code sees one stable interface.
    normalized = []
    for row in rows:
        row = dict(row)
        if "old_logprobs" not in row and "old_policy_logprobs" in row:
            row["old_logprobs"] = row["old_policy_logprobs"]
        if "ref_logprobs" not in row and "reference_logprobs" in row:
            row["ref_logprobs"] = row["reference_logprobs"]
        normalized.append(row)

    required = {"source_index", "response", "old_logprobs", "ref_logprobs"}
    if not required.issubset(normalized[0]):
        raise ValueError(f"Unexpected PPO cache schema; need at least {sorted(required)}")
    return normalized


def _first(row: dict, keys):
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return None


def _describe(v) -> str:
    if torch.is_tensor(v):
        return f"tensor{tuple(v.shape)} {v.dtype}"
    if isinstance(v, (list, tuple)):
        return f"{type(v).__name__}[{len(v)}]"
    s = repr(v)
    return f"{type(v).__name__}: {s[:70]}"


# =========================================================================================== cached batch
@torch.no_grad()
def reconstruct_cached_batch(bundle: dict, rows: list[dict], max_rows: int | None = None) -> dict:
    """Rebuild the fixed cached batch as padded tensors and run the SAME advantage pipeline as the live loop.
      prompt   : row['messages'|'prompt_messages'] if present, else rl_prompt_train[source_index]
      response : row['response_ids'|...] if present, else re-tokenized row['response'] (+EOS when the cached
                 log-prob vector is exactly one longer, i.e. the EOS token was decoded away)
      old/ref  : cached log-probs (the supplied behavior policy / frozen reference)
      reward   : re-scored with the frozen RM (cached value, if any, is only used as a reconstruction check)
    Rows whose token count cannot be aligned with the cached log-probs are dropped and reported, never padded over."""
    cfg, tok, policy = bundle["cfg"], bundle["tokenizer"], bundle["policy"]
    pool = bundle["prompt_rows"]
    device = next(policy.parameters()).device
    eos, pad = tok.eos_token_id, tok.pad_token_id

    kept, skipped = [], []
    for k, row in enumerate(rows[:max_rows] if max_rows else rows):
        msgs = _first(row, ["messages", "prompt_messages"])
        if msgs is None:
            msgs = prompt_messages(pool[int(row["source_index"])])
        rendered = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        p_ids = tok(rendered, truncation=True, max_length=int(cfg["max_prompt_length"]))["input_ids"]

        r_ids = _first(row, ["response_ids", "response_token_ids", "completion_ids"])
        if r_ids is None:
            r_ids = tok(str(row["response"]), add_special_tokens=False)["input_ids"]
        r_ids = [int(x) for x in (r_ids.tolist() if torch.is_tensor(r_ids) else list(r_ids))]
        old = torch.as_tensor(row["old_logprobs"], dtype=torch.float32).flatten()
        ref = torch.as_tensor(row["ref_logprobs"], dtype=torch.float32).flatten()
        n = old.numel()
        if len(r_ids) + 1 == n and eos is not None:
            r_ids.append(eos)
        if len(r_ids) != n or ref.numel() != n:
            skipped.append({"row": k, "source_index": row.get("source_index"), "n_tokens": len(r_ids),
                            "n_old": n, "n_ref": int(ref.numel())})
            continue
        cached_r = _first(row, _REWARD_KEYS)
        kept.append({
            "msgs": msgs, "p": p_ids, "r": r_ids, "old": old, "ref": ref,
            "text": str(row["response"]),
            "terminated": bool(row["terminated_with_eos"]) if "terminated_with_eos" in row
            else (eos is not None and r_ids[-1] == eos),
            "cached_reward": float(cached_r) if (isinstance(cached_r, (int, float)) or
                                                 (torch.is_tensor(cached_r) and cached_r.numel() == 1)) else None,
        })
    if not kept:
        raise RuntimeError(f"No cached row could be aligned with its log-probs; skipped={skipped[:5]}. "
                           "Run with --inspect and check the schema.")

    N = len(kept)
    pw = max(len(x["p"]) for x in kept)
    R = max(len(x["r"]) for x in kept)
    seq = torch.full((N, pw + R), pad, dtype=torch.long)
    attn = torch.zeros((N, pw + R), dtype=torch.long)
    resp = torch.full((N, R), pad, dtype=torch.long)
    rmask = torch.zeros((N, R))
    old_lp, ref_lp = torch.zeros((N, R)), torch.zeros((N, R))
    for i, x in enumerate(kept):
        lp_, n = len(x["p"]), len(x["r"])
        seq[i, pw - lp_:pw] = torch.tensor(x["p"])
        attn[i, pw - lp_:pw] = 1
        seq[i, pw:pw + n] = torch.tensor(x["r"])
        attn[i, pw:pw + n] = 1
        resp[i, :n] = torch.tensor(x["r"])
        rmask[i, :n] = 1.0
        old_lp[i, :n], ref_lp[i, :n] = x["old"], x["ref"]
    seq, attn, resp, rmask, old_lp, ref_lp = (t.to(device) for t in (seq, attn, resp, rmask, old_lp, ref_lp))

    rm = score_reward_pairs(bundle["reward_model"], bundle["reward_tokenizer"],
                            [x["msgs"] for x in kept], [x["text"] for x in kept],
                            max_length=int(cfg["reward_max_length"])).to(device)
    term = torch.tensor([x["terminated"] for x in kept], device=device)
    task_reward = rm - float(cfg["missing_eos_penalty"]) * (~term).float()

    core = {"seq": seq, "attn": attn, "prompt_width": pw, "resp": resp, "rmask": rmask,
            "old_lp": old_lp, "ref_lp": ref_lp}
    advantage_pipeline(bundle, core, task_reward, float(cfg["kl_beta"]))

    chunk = max(1, min(int(cfg.get("micro_batch_size", 2)), N))
    new_lp, _ = policy_forward_stats(policy, seq, attn, pw, resp, chunk, want_entropy=False)
    core["new_lp"] = new_lp

    check = [(a, b) for a, b in zip(rm.tolist(), [x["cached_reward"] for x in kept]) if b is not None]
    core["meta"] = {
        "n_rows_cache": len(rows), "n_rows_used": N, "n_tokens": int(rmask.sum()), "skipped": skipped,
        "reward_recompute_max_abs_diff": max((abs(a - b) for a, b in check), default=None),
    }
    return core


def ratio_spread(batch: dict) -> dict:
    m = batch["rmask"].bool()
    d = (batch["new_lp"] - batch["old_lp"])[m]
    return {"mean_abs_log_ratio": float(d.abs().mean()), "max_abs_log_ratio": float(d.abs().max()),
            "p99_abs_log_ratio": float(torch.quantile(d.abs().float(), 0.99)),
            "frac_tokens_log_ratio_gt_1e-3": float((d.abs() > 1e-3).float().mean())}


def static_clip_table(batch: dict, eps_values) -> list[dict]:
    """Clipped surrogate + affected-token fraction at rho = pi_midpoint / pi_old(cached), per epsilon."""
    out = []
    for eps in eps_values:
        loss, ratio, frac = ppo_policy_loss(batch["new_lp"], batch["old_lp"], batch["adv"], batch["rmask"], float(eps))
        d = clipping_diagnostics(ratio, batch["adv"], batch["rmask"], float(eps))
        out.append({
            "epsilon": float(eps),
            "clipped_surrogate": -float(loss),                                   # L_clip (higher = more objective)
            "unclipped_surrogate": float(masked_mean(ratio * batch["adv"], batch["rmask"])),
            "affected_token_fraction": float(frac),                              # rho outside [1-eps, 1+eps]
            "active_clip_fraction": d["clip_fraction_active"],                   # gradient actually zeroed
            "max_abs_ratio_dev": d["max_abs_ratio_dev"],
        })
    return out


def probe_table(bundle: dict, batch: dict, eps_values, epochs: int) -> list[dict]:
    """If the cache was produced by the midpoint itself then rho == 1 and the static table is trivially 0.
    Probe instead: from the exact midpoint, take `epochs` PPO optimizer steps on the cached batch per epsilon and
    record how much of the batch the clip region touches after each step (exposes the *immediate* geometry)."""
    out = []
    for eps in eps_values:
        reset_to_midpoint(bundle)
        upd = ppo_update(bundle, batch, float(eps), epochs=epochs)
        for e, rec in enumerate(upd["per_epoch"]):
            out.append({"epsilon": float(eps), "epoch": e, **{k: rec[k] for k in (
                "clip_fraction", "clip_fraction_active", "max_abs_ratio_dev", "policy_loss", "approx_kl_step")}})
        reset_to_midpoint(bundle)
    return out


# =========================================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--max-rows", type=int, default=None, help="limit cached rows (default: all)")
    ap.add_argument("--probe-epochs", type=int, default=4)
    ap.add_argument("--probe", choices=["auto", "always", "never"], default="auto")
    ap.add_argument("--skip-forks", action="store_true")
    ap.add_argument("--force", action="store_true", help="recompute forks even if cached on disk")
    ap.add_argument("--eval-prompts", type=int, default=None, help="subsample held-out prompts (default: all)")
    ap.add_argument("--inspect", action="store_true", help="print the cache schema and exit (no model loading)")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    rows = load_cached_rollouts(cfg["cached_rollouts"])
    print("Cached PPO rollouts:", len(rows))
    print("Required epsilon values:", cfg["clip_values"])
    print("Cache keys:", sorted(rows[0].keys()))
    if args.inspect:
        for k, v in sorted(rows[0].items()):
            print(f"  {k:24s} {_describe(v)}")
        return

    bundle = prepare_ppo_continuation(args.config)
    cfg = bundle["cfg"]
    res_dir = repo_path(cfg["results_dir"])
    reset_to_midpoint(bundle)

    # ---------------------------------------------------------------- (1) fixed cached batch
    batch = reconstruct_cached_batch(bundle, rows, args.max_rows)
    spread = ratio_spread(batch)
    static = static_clip_table(batch, cfg["clip_values"])
    degenerate = spread["max_abs_log_ratio"] < 1e-3
    print("\nCached-batch reconstruction:", batch["meta"])
    print("rho spread vs cached old log-probs:", spread)
    if batch["meta"]["reward_recompute_max_abs_diff"] is not None and batch["meta"]["reward_recompute_max_abs_diff"] > 0.05:
        print("WARNING: recomputed RM rewards differ from cached rewards -> prompt/response reconstruction is suspect")
    if degenerate:
        print("NOTE: rho ~= 1 everywhere (cache generated by this very checkpoint); static clip fractions are ~0 "
              "by construction -> the probe table below carries the epsilon geometry.")
    print(pd.DataFrame(static).to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    probe = []
    if args.probe == "always" or (args.probe == "auto" and degenerate):
        probe = probe_table(bundle, batch, cfg["clip_values"], args.probe_epochs)
        print("\nProbe (optimizer steps on the cached batch from the exact midpoint):")
        print(pd.DataFrame(probe).to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    save_json(res_dir / "clipping_cached_batch.json",
              {"meta": batch["meta"], "ratio_spread": spread, "static": static, "probe": probe})
    if args.skip_forks:
        return

    # ---------------------------------------------------------------- (2) matched short forks
    base = midpoint_baseline(bundle, force=args.force, eval_prompts=args.eval_prompts)["summary"]
    forks = {}
    # ============== ABLATION LOOP 1/2 -- CLIPPING STUDY: epsilon in {0.05, 0.20, 0.50}, KL beta fixed ==============
    # Every fork restarts from the identical midpoint snapshot, sees the identical prompt stream / per-update seeds,
    # uses the same fork_updates budget and the same held-out protocol. Only clip_epsilon differs.
    for eps in cfg["clip_values"]:
        forks[float(eps)] = run_fork(bundle, args.config, float(eps), float(cfg["kl_beta"]),
                                     force=args.force, eval_prompts=args.eval_prompts)
    # ===============================================================================================================

    table = []
    static_by_eps = {r["epsilon"]: r for r in static}
    for eps, f in forks.items():
        h, s = f["heldout"], f["stability"]
        table.append({
            "eps": eps,
            "cached_affected_frac": static_by_eps[eps]["affected_token_fraction"],
            "cached_L_clip": static_by_eps[eps]["clipped_surrogate"],
            "heldout_reward": h["reward_mean"], "d_reward_vs_mid": h["reward_mean"] - base["reward_mean"],
            "heldout_kl_tok": h["kl_per_token"], "heldout_len": h["response_length"], "heldout_entropy": h["entropy"],
            "fork_clip_frac": s["clip_fraction_mean"], "fork_active_clip": s["clip_fraction_active_mean"],
            "max_abs_ratio_dev": s["max_abs_ratio_dev"], "grad_norm_cv": s["grad_norm_cv"],
            "approx_kl_step_max": s["approx_kl_step_max"], "skipped": s["skipped_steps"],
            "train_reward_last": f["last_window"]["reward"], "wall_s": f["train_wall_s"],
        })
    print("\nBaseline (midpoint, held-out):", {k: round(v, 4) for k, v in base.items() if isinstance(v, float)})
    print(pd.DataFrame(table).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    save_json(res_dir / "clipping_study.json", {
        "baseline_heldout": base, "table": table,
        "forks": {str(k): {kk: vv for kk, vv in v.items() if kk != "logs"} for k, v in forks.items()},
        "definitions": {
            "cached_affected_frac": "fraction of cached-batch tokens with rho outside [1-eps,1+eps], rho=pi_mid/pi_old",
            "fork_clip_frac": "mean over updates of in-training clip fraction (epoch mean; epoch 0 has rho==1)",
            "max_abs_ratio_dev": "PRIMARY stability stat: max over tokens/epochs/updates of |rho-1|",
            "grad_norm_cv": "std/mean of pre-clip policy gradient norm across updates",
        },
    })


if __name__ == "__main__":
    main()
