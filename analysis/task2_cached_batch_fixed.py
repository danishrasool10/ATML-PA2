"""Corrected cached-batch clipping measurement for Task 2 step (2). Wrapper only: task2_ppo/ is NOT modified.

Why this exists
  task2_ppo.analyze_clipping.reconstruct_cached_batch rebuilds each cached prompt as
  rl_prompt_pool_train[row["source_index"]]. In cached/ppo_rollout.pt the 32 rows are rows 0-31 of
  rl_prompt_pool_eval (every prompt_id is found there, none in the train pool) and `source_index` is the
  upstream-dataset index (it equals the eval row's own `source_index` field). Every cached response was
  therefore scored under an unrelated prompt, which produced the mean |log rho| = 0.57 and the
  0.55/0.34/0.18 affected fractions of the Kaggle rerun. reconstruct_cached_batch already prefers
  row["messages"] when present, so this wrapper injects the correct messages (matched by prompt_id) plus a
  `reward` field (= raw_terminal_reward, which enables the existing reward-reconstruction check) and then
  calls the unmodified task2_ppo functions.

Extra checks (not in analyze_clipping):
  * ref-logprob numerical floor: the frozen reference is identical in the staff and student pipelines, so
    |log pi_ref(recomputed) - log pi_ref(cached)| isolates cross-environment numerics (dtype, kernels, padding).
  * critic check: recomputed critic values vs the cached staff values (same critic checkpoint).
  * noise-only affected fraction per eps (rho_floor = pi_ref,recomputed / pi_ref,cached): the TA's "zero baseline".

Run on the GPU machine from the repository root (no forks; a few minutes):
    python -m analysis.task2_cached_batch_fixed --config configs/ppo.yaml
Output: results/task2_ppo/clipping_cached_batch_fixed.json  (then re-run analysis.task2_report_stats)
"""
from __future__ import annotations

import argparse
import math

import torch

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.logging_utils import save_json
from common.models import reference_mode
from task2_ppo.analyze_clipping import (
    load_cached_rollouts,
    probe_table,
    ratio_spread,
    reconstruct_cached_batch,
    static_clip_table,
)
from task2_ppo.continue_train import prepare_ppo_continuation, reset_to_midpoint
from task2_ppo.utils import policy_forward_stats


def _affected(log_ratio: torch.Tensor, eps: float) -> float:
    return float(((log_ratio < math.log(1 - eps)) | (log_ratio > math.log(1 + eps))).float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--probe-epochs", type=int, default=4)
    ap.add_argument("--no-probe", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    rows = load_cached_rollouts(cfg["cached_rollouts"])
    eval_by_pid = {r["prompt_id"]: r for r in read_jsonl(cfg["paths"]["rl_prompt_eval"])}
    missing = [r["prompt_id"] for r in rows if r["prompt_id"] not in eval_by_pid]
    if missing:
        raise SystemExit(f"{len(missing)} cached prompt_ids are not in rl_prompt_pool_eval; inspect the cache")
    for r in rows:
        r["messages"] = prompt_messages(eval_by_pid[r["prompt_id"]])
        r["reward"] = float(r["raw_terminal_reward"])

    bundle = prepare_ppo_continuation(args.config)
    cfg = bundle["cfg"]
    reset_to_midpoint(bundle)
    batch = reconstruct_cached_batch(bundle, rows)
    skipped = {s["row"] for s in batch["meta"]["skipped"]}
    kept = [k for k in range(len(rows)) if k not in skipped]
    m = batch["rmask"].bool()

    # numerical floor from the frozen reference
    policy = bundle["policy"]
    chunk = max(1, min(int(cfg.get("micro_batch_size", 2)), batch["seq"].shape[0]))
    with reference_mode(policy):
        ref_re, _ = policy_forward_stats(policy, batch["seq"], batch["attn"], batch["prompt_width"], batch["resp"],
                                         chunk, want_entropy=False)
    floor = (ref_re - batch["ref_lp"])[m]

    # critic check against cached staff values
    cached_v = torch.zeros_like(batch["values"])
    for i, k in enumerate(kept):
        v = torch.as_tensor(rows[k]["values"], dtype=torch.float32).flatten()
        cached_v[i, : v.numel()] = v.to(cached_v.device)
    vdiff = (batch["values"] - cached_v)[m].abs()

    log_rho = (batch["new_lp"] - batch["old_lp"])[m]
    spread = ratio_spread(batch)
    static = static_clip_table(batch, cfg["clip_values"])
    for row in static:
        row["noise_only_affected_fraction"] = _affected(floor, row["epsilon"])
    probe = [] if args.no_probe else probe_table(bundle, batch, cfg["clip_values"], args.probe_epochs)

    out = {
        "fix": "prompts matched by prompt_id in rl_prompt_pool_eval (cache rows = eval rows 0-31)",
        "meta": batch["meta"], "ratio_spread_mid_vs_cached_old": spread,
        "ref_numerical_floor": {"mean_abs": float(floor.abs().mean()), "p99_abs": float(torch.quantile(floor.abs(), 0.99)),
                                "max_abs": float(floor.abs().max())},
        "critic_values_recomputed_vs_cached": {"mean_abs": float(vdiff.mean()), "max_abs": float(vdiff.max())},
        "log_rho_minus_floor_mean_abs": float((log_rho.abs() - floor.abs()).mean()),
        "static": static, "probe": probe,
    }
    path = repo_path(cfg["results_dir"]) / "clipping_cached_batch_fixed.json"
    save_json(path, out)
    print({k: v for k, v in out.items() if k not in ("static", "probe")})
    for r in static:
        print(r)
    for r in probe:
        print(r)
    print("saved ->", path)


if __name__ == "__main__":
    main()
