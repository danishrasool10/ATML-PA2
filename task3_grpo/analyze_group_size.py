from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.logging_utils import save_json
from task3_grpo.grpo import group_statistics

METRICS = ["informative_rate", "trainable_rate", "mean_group_std", "adv_var_grpo", "adv_var_raw"]
DEFINITIONS = {
    "budget": "total generations per condition; each K uses budget/K DISTINCT prompts (sampled without replacement) "
              "and K distinct cached completions per prompt, so no prompt contributes twice to one draw",
    "difficulty_bin": "prompt difficulty = mean reward of its 8 cached completions; bins = equal-count quantiles of "
                      "that mean over the cached prompts, bin 0 = hardest (lowest mean reward)",
    "informative_rate": "fraction of K-groups with population reward std >= std_gate (training-loop gate; A = 0 below)",
    "trainable_rate": "informative AND >= 1 member that ends with EOS within max_completion_length (survives the "
                      "max-length mask at the training cap)",
    "mean_group_std": "mean over groups of the population std of the K rewards (the GRPO normalizer sigma_g)",
    "adv_var_grpo": "variance, over every completion in the budget, of A = (r - mu_g)/(sigma_g + eps), gate applied",
    "adv_var_raw": "variance, over every completion in the budget, of r - mu_g (the Dr. GRPO signal), gate applied",
    "interval": "2.5/97.5 percentiles across random draws = spread of what ONE rollout of this budget would show; "
                "it is not divided by the number of draws",
}


def load_k8_cache(path):
    rows = read_jsonl(path)
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[str(row["source_index"])].append(row)
    # Instructor cache has 8 rows per prompt, one row per completion.
    bad = {pid: len(group) for pid, group in by_prompt.items() if len(group) < 8}
    if bad:
        raise ValueError(f"Expected at least K=8 cached completions per prompt; short groups: {bad}")
    for group in by_prompt.values():
        group.sort(key=lambda x: int(x.get("generation_index", 0)))
    return by_prompt


def difficulty_bins(by_prompt, n_bins: int) -> dict[str, int]:
    order = sorted(by_prompt, key=lambda p: np.mean([c["reward"] for c in by_prompt[p]]))
    return {p: min(i * n_bins // len(order), n_bins - 1) for i, p in enumerate(order)}


def regroup_equal_generation_budget(by_prompt, k: int, budget: int | None = None, rng: np.random.Generator | None = None):
    """One equal-generation draw: `budget // k` distinct prompts, each contributing `k` distinct cached completions.

    The cache holds P prompts x 8 completions, so the largest budget that every K in {2, 4, 8} can fill without
    reusing a prompt is P * min(K) (default: P * 2). Comparing K at this fixed budget, rather than at a fixed prompt
    count, is what charges larger groups for the generations they spend on each prompt.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    pids = sorted(by_prompt)
    budget = budget or 2 * len(pids)
    n_prompts = budget // k
    if budget % k or n_prompts > len(pids) or any(len(by_prompt[p]) < k for p in pids):
        raise ValueError(f"budget={budget} cannot be split into K={k} groups over {len(pids)} distinct prompts")
    chosen = rng.choice(len(pids), size=n_prompts, replace=False)
    return [[by_prompt[pids[i]][j] for j in rng.choice(len(by_prompt[pids[i]]), size=k, replace=False)] for i in chosen]


def draw_metrics(groups, bins: dict[str, int], n_bins: int, cfg: dict) -> dict:
    K, gate, eps = len(groups[0]), float(cfg["std_gate"]), float(cfg["adv_eps"])
    L = int(cfg["max_completion_length"])
    r = torch.tensor([[c["reward"] for c in g] for g in groups], dtype=torch.float64).flatten()
    gids = torch.arange(len(groups)).repeat_interleave(K)
    st = group_statistics(r, gids, eps, scale_by_std=True, std_gate=gate)
    a_raw = group_statistics(r, gids, eps, scale_by_std=False, std_gate=gate)["advantages"].view(-1, K).numpy()
    a_std = st["advantages"].view(-1, K).numpy()
    sigma, inf = st["group_std"].double().numpy(), st["informative"].numpy()
    survives = np.array([[bool(c["terminated_with_eos"]) and int(c["completion_tokens"]) <= L for c in g] for g in groups])
    trainable = inf & survives.any(1)
    b = np.array([bins[str(g[0]["source_index"])] for g in groups])

    def agg(m):
        if not m.any():
            return {k: np.nan for k in METRICS} | {"n_groups": 0}
        return {"informative_rate": inf[m].mean(), "trainable_rate": trainable[m].mean(),
                "mean_group_std": sigma[m].mean(), "adv_var_grpo": a_std[m].var(), "adv_var_raw": a_raw[m].var(),
                "n_groups": int(m.sum())}

    out = {"all": agg(np.ones(len(groups), bool)), **{f"bin{i}": agg(b == i) for i in range(n_bins)}}
    out["tau"] = {f"{t:g}": float((sigma >= t).mean()) for t in cfg["group_study"]["std_thresholds"]}
    return out


def summarize(draws: list[dict], key: str) -> dict:
    out = {}
    for m in METRICS + ["n_groups"]:
        v = np.array([d[key][m] for d in draws], dtype=float)
        out[m] = float(np.nanmean(v))
        out[f"{m}_lo"], out[f"{m}_hi"] = (float(x) for x in np.nanpercentile(v, [2.5, 97.5]))
    return out


def examples(by_prompt, bins, prompt_text, cfg) -> dict:
    gate = float(cfg["std_gate"])
    stats = {p: (np.mean([c["reward"] for c in g]), np.std([c["reward"] for c in g])) for p, g in by_prompt.items()}
    flat = [p for p, (_, s) in stats.items() if s < gate]
    spread = max(stats, key=lambda p: stats[p][1])

    def show(p):
        g = sorted(by_prompt[p], key=lambda c: c["reward"])
        return {"source_index": p, "difficulty_bin": bins[p], "prompt": prompt_text.get(p, "")[:400],
                "mean_reward": stats[p][0], "std_reward": stats[p][1],
                "rewards": [c["reward"] for c in g], "tokens": [c["completion_tokens"] for c in g],
                "lowest": g[0]["completion"][:500], "highest": g[-1]["completion"][:500]}

    return {"uninformative_at_k8": [show(p) for p in flat], "largest_spread": show(spread)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--replicates", type=int, default=None)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    gs = cfg["group_study"]
    by_prompt = load_k8_cache(cfg["group_cache"])
    sizes = [int(k) for k in cfg["group_sizes"]]
    n_bins = int(gs["difficulty_bins"])
    budget = int(args.budget or gs.get("budget") or len(by_prompt) * min(sizes))
    R = int(args.replicates or gs["replicates"])
    bins = difficulty_bins(by_prompt, n_bins)
    prompt_text = {str(r["source_index"]): str(prompt_messages(r)[-1].get("content", ""))
                   for r in read_jsonl(cfg["paths"]["rl_prompt_eval"]) + read_jsonl(cfg["paths"]["rl_prompt_train"])}
    n_cached = sum(len(g) for g in by_prompt.values())
    print(f"cache: {len(by_prompt)} prompts x 8 = {n_cached} completions | budget {budget} generations/condition | "
          f"{R} draws | prompts per condition: " + ", ".join(f"K={k}: {budget // k}" for k in sizes))

    rng = np.random.default_rng(int(cfg["seed"]))
    table, by_bin, tau = [], [], []
    for k in sizes:
        draws = [draw_metrics(regroup_equal_generation_budget(by_prompt, k, budget, rng), bins, n_bins, cfg)
                 for _ in range(R)]
        s = summarize(draws, "all")
        table.append({"K": k, "prompts": budget // k, **s,
                      "informative_prompts_per_budget": s["informative_rate"] * budget / k,
                      "trainable_prompts_per_budget": s["trainable_rate"] * budget / k,
                      "informative_generations_per_budget": s["informative_rate"] * budget})
        for i in range(n_bins):
            by_bin.append({"K": k, "bin": i, **summarize(draws, f"bin{i}")})
        tau.append({"K": k, **{t: float(np.mean([d["tau"][t] for d in draws])) for t in draws[0]["tau"]}})

    bin_means = defaultdict(list)
    for p, i in bins.items():
        bin_means[i].append(float(np.mean([c["reward"] for c in by_prompt[p]])))
    bin_info = {i: {"n_prompts": len(v), "mean_reward_range": [min(v), max(v)]} for i, v in sorted(bin_means.items())}

    pd.set_option("display.width", 200)
    cols = ["K", "prompts", "informative_rate", "trainable_rate", "mean_group_std", "adv_var_grpo", "adv_var_raw",
            "informative_prompts_per_budget", "trainable_prompts_per_budget"]
    print("\nEqual-generation comparison (mean over draws):")
    print(pd.DataFrame(table)[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nBy prompt difficulty (bin 0 = hardest):", bin_info)
    print(pd.DataFrame(by_bin)[["K", "bin", "n_groups"] + METRICS].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nInformative rate vs std threshold tau (reward units):")
    print(pd.DataFrame(tau).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    ex = examples(by_prompt, bins, prompt_text, cfg)
    out = repo_path(cfg["results_dir"]) / "group_size_study.json"
    save_json(out, {"definitions": DEFINITIONS, "budget": budget, "replicates": R, "cached_prompts": len(by_prompt),
                    "cached_completions": n_cached, "std_gate": float(cfg["std_gate"]),
                    "max_completion_length": int(cfg["max_completion_length"]), "difficulty_bins": bin_info,
                    "table": table, "by_bin": by_bin, "tau_sweep": tau, "examples": ex})
    print("\nsaved ->", out)


if __name__ == "__main__":
    main()
