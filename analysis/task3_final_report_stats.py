"""Numbers for the Task 3 LaTeX section (results/task3_grpo/analysis/final_report/).

Reads raw artifacts only and writes final_report/numbers.json; every number in section5_final.tex comes from there.
Conventions match analysis/task2_report_stats.py: std over updates with ddof=1; held-out deltas vs the midpoint use a
paired bootstrap over prompts (2000 resamples, seed 6304, 95% percentile CI).

Usage (repository root):  python -m analysis.task3_final_report_stats
"""
from __future__ import annotations

import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

R = Path("results/task3_grpo")
OUT = R / "analysis/final_report"
SEED, N_BOOT, L_MAX = 6304, 2000, 512


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def sd(x):
    x = np.asarray(x, float)
    return float(np.std(x, ddof=1)) if x.size > 1 else float("nan")


def ms(logs, key, scale=1.0):
    v = [r[key] * scale for r in logs if r.get(key) is not None]
    return {"mean": float(np.mean(v)), "std": sd(v), "n": len(v)}


def boot(d):
    d = np.asarray(d, float)
    rng = np.random.default_rng(SEED)
    m = d[rng.integers(0, d.size, size=(N_BOOT, d.size))].mean(axis=1)
    return {"mean": float(d.mean()), "lo": float(np.quantile(m, 0.025)), "hi": float(np.quantile(m, 0.975)), "n": int(d.size)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    N = {}
    runs = {"standard": R / "standard", "grpo": R / "forks/fork_grpo", "dr_grpo": R / "forks/fork_dr_grpo"}
    ho = {k: json.load(open(p / "heldout.json")) for k, p in runs.items()}
    ho["midpoint"] = json.load(open(R / "forks/midpoint_baseline_heldout.json"))
    mid = ho["midpoint"]["samples"]

    # ---------------------------------------------------------------- table rows
    tab = {}
    for k, p in runs.items():
        logs = jl(p / "metrics.jsonl")
        summ = json.load(open(p / "summary.json"))
        s = ho[k]["samples"]
        assert [x["idx"] for x in s] == [x["idx"] for x in mid]
        tab[k] = {
            "updates": len(logs),
            "no_step_updates": int(sum(not r["stepped"] for r in logs)),
            "masked_completions": int(round(sum(r["truncation_rate"] * 4 for r in logs))),
            "completions": 4 * len(logs),
            "reward_std_within": ms(logs, "reward_std_within"),
            "grad_norm": ms(logs, "grad_norm"),
            "kl_1e4": ms(logs, "kl", 1e4),
            "reward_train": ms(logs, "reward"),
            "entropy": ms(logs, "entropy"),
            "response_length": ms(logs, "response_length"),
            "frac_uninformative_groups_sum": float(sum(r["frac_uninformative_groups"] for r in logs)),
            "generated_tokens": summ["generated_tokens"], "trained_tokens": summ["trained_tokens"],
            "train_wall_s": summ["train_wall_s"], "sec_per_update": summ["sec_per_update"],
            "peak_vram_gib": summ["peak_vram_gib"],
            "clip_fraction_max": max(r["clip_fraction"] for r in logs if r["clip_fraction"] is not None),
            "heldout_reward": ho[k]["summary"]["reward_mean"],
            "heldout_length": ho[k]["summary"]["response_length"],
            "heldout_kl_seq": ho[k]["summary"]["kl_per_sequence"],
            "heldout_trunc": ho[k]["summary"]["truncation_rate"],
            "heldout_d_reward_vs_mid": boot([a["reward"] - b["reward"] for a, b in zip(s, mid)]),
            "heldout_d_len_vs_mid": boot([a["length"] - b["length"] for a, b in zip(s, mid)]),
        }
    tab["midpoint"] = {"heldout_reward": ho["midpoint"]["summary"]["reward_mean"],
                       "heldout_length": ho["midpoint"]["summary"]["response_length"],
                       "heldout_kl_seq": ho["midpoint"]["summary"]["kl_per_sequence"],
                       "heldout_trunc": ho["midpoint"]["summary"]["truncation_rate"]}
    g, d = ho["grpo"]["samples"], ho["dr_grpo"]["samples"]
    N["table"] = tab
    N["dr_minus_grpo"] = {"reward": boot([b["reward"] - a["reward"] for a, b in zip(g, d)]),
                          "length": boot([b["length"] - a["length"] for a, b in zip(g, d)])}
    std_logs, grpo_logs = jl(runs["standard"] / "metrics.jsonl"), jl(runs["grpo"] / "metrics.jsonl")
    keys = ["reward", "reward_std_within", "kl", "entropy", "response_length", "grad_norm"]
    N["standard_first8_equals_fork_grpo"] = all(a[k] == b[k] for a, b in zip(std_logs[:8], grpo_logs) for k in keys)
    N["fork_grpo_vs_dr_identical_rows"] = {
        k: [r[k] for r in jl(runs["dr_grpo"] / "metrics.jsonl")] == [r[k] for r in grpo_logs]
        for k in ("response_length", "generated_tokens", "trained_tokens", "stepped")}

    # ---------------------------------------------------------------- baseline mechanics (standard run)
    dead = [r for r in std_logs if not r["stepped"]]
    part = [r for r in std_logs if r["stepped"] and r["truncation_rate"] > 0]
    live = [r for r in std_logs if r["stepped"]]
    N["standard"] = {
        "dead_updates": [r["update"] for r in dead],
        "dead_tokens": float(sum(r["generated_tokens"] for r in dead)),
        "dead_wall_s": float(sum(r["update_wall_s"] for r in dead)),
        "zero_policy_loss_updates": int(sum(r["policy_loss"] == 0 for r in live)),
        "stepped_updates": len(live),
        "partial_mask_survivor_mean_adv": {r["update"]: -r["policy_loss"] for r in part},
        "grad_clip_hits": [r["update"] for r in live if r["grad_norm"] > 1.0],
        "u8": {k: std_logs[7][k] for k in ("reward_std_within", "grad_norm", "response_length_trained")},
        "u15": {k: std_logs[14][k] for k in ("grad_norm", "trained_sequences", "response_length_trained", "policy_loss")},
        "u8_rewards": [c["reward"] for c in std_logs[7]["samples"][0]["completions"]],
        "u8_adv": [c["adv"] for c in std_logs[7]["samples"][0]["completions"]],
        "u8_texts": [c["text"] for c in std_logs[7]["samples"][0]["completions"]],
        "u15_completions": [{k: c[k] for k in ("reward", "adv", "length", "truncated")} for c in std_logs[14]["samples"][0]["completions"]],
    }

    # ---------------------------------------------------------------- RQ1 (group-size study + exact enumeration on cache)
    gs = json.load(open(R / "group_size_study.json"))
    N["group_size"] = {"budget": gs["budget"], "replicates": gs["replicates"], "difficulty_bins": gs["difficulty_bins"],
                       "table": [{k: r[k] for k in ("K", "prompts", "informative_rate", "mean_group_std", "adv_var_raw",
                                                    "trainable_rate", "informative_prompts_per_budget")} for r in gs["table"]],
                       "tau_sweep": gs["tau_sweep"]}
    byp = defaultdict(list)
    for r in jl("cached/grpo_k_cache.jsonl"):
        byp[str(r["source_index"])].append(r)
    order = sorted(byp, key=lambda p: np.mean([c["reward"] for c in byp[p]]))
    bins = {p: min(i * 3 // len(order), 2) for i, p in enumerate(order)}
    ex = {}
    for K in (2, 8):
        for b, name in enumerate(("hard", "medium", "easy")):
            v = [np.mean([np.std(np.array([c["reward"] for c in byp[p]])[list(s)]) >= 0.25
                          for s in itertools.combinations(range(8), K)]) for p in byp if bins[p] == b]
            ex[f"K{K}_{name}_tau0.25"] = float(np.mean(v))
    N["group_size"]["exact_tau025"] = ex
    surv = lambda c: bool(c["terminated_with_eos"]) and int(c["completion_tokens"]) <= L_MAX
    N["group_size"]["cache"] = {
        "prompts": len(byp), "completions": sum(len(v) for v in byp.values()),
        "fail_cap_frac": float(np.mean([not surv(c) for v in byp.values() for c in v])),
        "fully_masked_prompts": int(sum(all(not surv(c) for c in v) for v in byp.values())),
        "zero_var_prompts": [{"source_index": p, "rewards": sorted({c["reward"] for c in v}),
                              "texts": sorted({c["completion"] for c in v})} for p, v in byp.items()
                             if np.std([c["reward"] for c in v]) < 1e-6]}

    # ---------------------------------------------------------------- RQ2 (allocation)
    ns = json.load(open(R / "normalization_study.json"))
    N["allocation"] = {lt: {"loglog_slope": ns["gradient_allocation"][lt]["loglog_slope"],
                            "n": ns["gradient_allocation"][lt]["n"],
                            "share_by_bin": [b["grad_mass_share"] for b in ns["gradient_allocation"][lt]["bins"]],
                            "mean_len_by_bin": [b["length"] for b in ns["gradient_allocation"][lt]["bins"]]}
                       for lt in ("grpo", "dr_grpo")}
    st = json.load(open(R / "analysis/stats.json"))
    N["allocation"]["slope_se"] = {lt: st["normalization"]["loglog_fit"][lt]["slope_se"] for lt in ("grpo", "dr_grpo")}
    N["allocation"]["per_update_grad_norm"] = st["normalization"]["per_update_grad_norm"]
    N["allocation"]["train_length_edges"] = ns["train_length_edges"]

    # ---------------------------------------------------------------- qualitative held-out examples
    idn = [i for i in range(len(g)) if g[i]["response"] == d[i]["response"] and g[i]["length"] == d[i]["length"]]
    N["heldout_identical_grpo_dr"] = len(idn)
    N["dr_minus_grpo"]["reward_identical_text_only"] = boot([d[i]["reward"] - g[i]["reward"] for i in idn])
    N["rm_rescoring_median_abs_diff_identical_text"] = {
        k: v["identical_short_abs_dR_median"] for k, v in st["normalization"]["paired_noise"].items()}
    N["standard"]["max_abs_policy_loss_unmasked"] = max(
        abs(r["policy_loss"]) for r in std_logs if r["stepped"] and r["truncation_rate"] == 0)
    N["idx165"] = {k: {"length": ho[k]["samples"][165]["length"], "reward": ho[k]["samples"][165]["reward"],
                       "prompt": ho[k]["samples"][165]["prompt"][:120], "response": ho[k]["samples"][165]["response"][:160]}
                   for k in ("midpoint", "standard", "grpo", "dr_grpo")}
    json.dump(N, open(OUT / "numbers.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(N, indent=1, ensure_ascii=False)[:20000])


if __name__ == "__main__":
    main()
