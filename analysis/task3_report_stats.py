"""Post-hoc statistics, tables and figures for Task 3 (GRPO continuation, group-size study, normalization study).

Reads saved artifacts only; never loads a model or trains.

Inputs:
  results/task3_grpo/standard/{metrics.jsonl,summary.json,heldout.json}
  results/task3_grpo/forks/{fork_grpo,fork_dr_grpo}/{metrics.jsonl,summary.json,seq_grads.jsonl,heldout.json}
  results/task3_grpo/forks/midpoint_baseline_heldout.json
  results/task3_grpo/{group_size_study,normalization_study}.json
  cached/grpo_k_cache.jsonl, configs/grpo.yaml

Outputs (under --out, default results/task3_grpo/analysis): stats.json, tables/*.csv, figures/*.png|pdf

Usage (repository root):  python -m analysis.task3_report_stats
"""
from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps

from common.data import load_yaml, read_jsonl

GRPO_C, DR_C, MID_C = "#2a78d6", "#eb6834", "#8a8984"
K_C = {2: "#86b6ef", 4: "#2a78d6", 8: "#104281"}
BIN_NAMES = {0: "hard", 1: "medium", 2: "easy"}
INK, INK2, GRID, SHADE = "#0b0b0b", "#52514e", "#e4e3df", "#efeeea"


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.axisbelow": True, "lines.linewidth": 1.6, "legend.frameon": False,
        "legend.fontsize": 7.5, "figure.dpi": 150, "savefig.bbox": "tight", "savefig.dpi": 300,
    })


def save(fig, out: Path, name: str):
    for ext in ("png", "pdf"):
        fig.savefig(out / "figures" / f"{name}.{ext}")
    plt.close(fig)


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def paired(a, b, key):
    x = np.array([s[key] for s in b], float) - np.array([s[key] for s in a], float)
    return {"mean": float(x.mean()), "se": float(x.std(ddof=1) / np.sqrt(len(x))), "n": len(x)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/task3_grpo")
    ap.add_argument("--out", default="results/task3_grpo/analysis")
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    R, out = Path(args.results), Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    cfg = load_yaml(args.config)
    style()
    S = {}

    # ================================================================= standard continuation
    m = jl(R / "standard/metrics.jsonl")
    summ = json.load(open(R / "standard/summary.json"))
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("samples", "peak_vram_gib", "prompt_ids")}
                       | {"prompt_id": r["prompt_ids"][0]} for r in m])
    df.to_csv(out / "tables/table1_standard_run.csv", index=False)
    dead = df[~df["stepped"]]
    live = df[df["stepped"]]
    partial = df[(df["truncation_rate"] > 0) & df["stepped"]]
    surv_adv = {int(r.update): float(-r.policy_loss) for r in partial.itertuples()}
    gn = live["grad_norm"].to_numpy()
    S["standard"] = {
        "train_wall_s": summ["train_wall_s"], "sec_per_update": summ["sec_per_update"],
        "peak_vram_gib": summ["peak_vram_gib"], "generated_tokens": summ["generated_tokens"],
        "trained_tokens": summ["trained_tokens"], "trained_token_frac": summ["trained_tokens"] / summ["generated_tokens"],
        "updates_without_step": [int(u) for u in dead["update"]],
        "dead_tokens": float(dead["generated_tokens"].sum()),
        "dead_token_frac": float(dead["generated_tokens"].sum() / summ["generated_tokens"]),
        "dead_wall_s": float(dead["update_wall_s"].sum()),
        "dead_wall_frac": float(dead["update_wall_s"].sum() / summ["train_wall_s"]),
        "masked_completions": int(round((df["truncation_rate"] * 4).sum())), "completions": 4 * len(df),
        "uninformative_updates": int((df["frac_uninformative_groups"] > 0).sum()),
        "reward_std_within_mean": float(df["reward_std_within"].mean()),
        "reward_std_within_range": [float(df["reward_std_within"].min()), float(df["reward_std_within"].max())],
        "kl_range": [float(df["kl"].min()), float(df["kl"].max())],
        "kl_k3_range": [float(live["kl_k3"].min()), float(live["kl_k3"].max())],
        "entropy_range": [float(df["entropy"].min()), float(df["entropy"].max())],
        "grad_norm_mean": float(gn.mean()), "grad_norm_cv": float(gn.std() / gn.mean()),
        "grad_norm_max": float(gn.max()), "grad_clip_hits": [int(u) for u in live.loc[live.grad_norm > 1, "update"]],
        "clip_fraction_max": float(live["clip_fraction"].max()), "ratio_mean_range": [float(live.ratio_mean.min()), float(live.ratio_mean.max())],
        "survivor_mean_adv_partial_updates": surv_adv,
        "spearman_gn_vs_sigma": float(sps.spearmanr(live["grad_norm"], live["reward_std_within"])[0]),
        "spearman_gn_vs_trained_len": float(sps.spearmanr(live["grad_norm"], live["response_length_trained"])[0]),
        "spearman_reward_vs_update": [float(x) for x in sps.spearmanr(df["update"], df["reward"])],
        "spearman_len_vs_update": [float(x) for x in sps.spearmanr(df["update"], df["response_length"])],
        "first_window": summ["first_window"], "last_window": summ["last_window"], "overall": summ["overall"],
    }
    fg = jl(R / "forks/fork_grpo/metrics.jsonl")
    keys = ["reward", "reward_std_within", "kl", "entropy", "response_length", "grad_norm", "policy_loss"]
    S["standard"]["determinism_first8_vs_fork_grpo_max_absdiff"] = float(max(
        abs((a[k] or 0) - (b[k] or 0)) for a, b in zip(m[:8], fg) for k in keys))

    no = dead["update"].to_numpy()
    fig, axs = plt.subplots(2, 4, figsize=(10.5, 4.4), sharex=True)
    panels = [("reward", "Reward (group mean)", None), ("kl", r"KL$(\pi\,\|\,\pi_{ref})$ / token ($\times10^{-4}$)", 1e4),
              ("reward_std_within", r"Within-group reward std $\sigma_g$", None),
              ("frac_uninformative_groups", "Group fraction", None),
              ("policy_loss", "Policy loss", None), ("grad_norm", "Grad norm (pre-clip)", None),
              ("entropy", "Entropy (nats / token)", None), ("response_length", "Response length (tokens)", None)]
    u = df["update"].to_numpy()
    for ax, (k, lab, sc) in zip(axs.flat, panels):
        for x in no:
            ax.axvspan(x - 0.5, x + 0.5, color=SHADE, lw=0, zorder=0)
        y = df[k].to_numpy(dtype=float) * (sc or 1)
        if k == "frac_uninformative_groups":
            ax.plot(u, y, "-o", color=GRPO_C, ms=3.5, label=r"uninformative ($\sigma_g<10^{-6}$)")
            ax.plot(u, df["frac_groups_all_truncated"], "s", color=DR_C, ms=4, label="all K truncated")
            ax.set_ylim(-0.08, 1.12)
            ax.legend(loc="center right", fontsize=6.5)
        elif k == "response_length":
            ax.plot(u, y, "-o", color=GRPO_C, ms=3.5, label="generated")
            ax.plot(u, df["response_length_trained"], "o", mfc="white", color=GRPO_C, ms=4, label="trained (unmasked)")
            ax.axhline(cfg["max_completion_length"], color=INK2, lw=0.8, ls="--")
            ax.set_ylim(-10, 640)
            ax.legend(loc="upper center", fontsize=6.5, ncol=2, bbox_to_anchor=(0.5, 1.0))
        else:
            ax.plot(u, y, "-o", color=GRPO_C, ms=3.5)
        if k == "grad_norm":
            ax.axhline(cfg["max_grad_norm"], color=INK2, lw=0.8, ls="--")
            ax.text(0.6, cfg["max_grad_norm"] + 0.03, "max_grad_norm", va="bottom", ha="left", fontsize=6.5, color=INK2)
        ax.set_title(lab, loc="left")
    for ax in axs[1]:
        ax.set_xlabel("Update")
        ax.set_xticks([1, 5, 10, 15, 20])
    fig.tight_layout()
    save(fig, out, "fig1_standard_run")

    fig, ax = plt.subplots(figsize=(6.2, 2.4))
    ax.bar(u, df["trained_tokens"], color=GRPO_C, width=0.8, label="entered the loss")
    ax.bar(u, df["generated_tokens"] - df["trained_tokens"], bottom=df["trained_tokens"], color="#c9c8c2",
           width=0.8, label="masked (completion hit the 512-token cap)", edgecolor="white", linewidth=0.6)
    ax.set_xlabel("Update")
    ax.set_ylabel("Generated tokens")
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.legend(loc="upper right", fontsize=7, ncol=2, bbox_to_anchor=(1, 1.18))
    fig.tight_layout()
    save(fig, out, "fig2_token_budget")

    # ================================================================= group-size study
    gs = json.load(open(R / "group_size_study.json"))
    rows = read_jsonl(cfg["group_cache"])
    byp = defaultdict(list)
    for r in rows:
        byp[str(r["source_index"])].append(r)
    order = sorted(byp, key=lambda p: np.mean([c["reward"] for c in byp[p]]))
    bins = {p: min(i * 3 // len(order), 2) for i, p in enumerate(order)}
    taus = [1e-6, 0.05, 0.10, 0.25]
    L = int(cfg["max_completion_length"])
    exact = []
    for K in (2, 4, 8):
        for b in range(3):
            ps = [p for p in byp if bins[p] == b]
            rec = {"K": K, "bin": BIN_NAMES[b]}
            for t in taus:
                v = []
                for p in ps:
                    r = np.array([c["reward"] for c in byp[p]])
                    v.append(np.mean([np.std(r[list(s)]) >= t for s in itertools.combinations(range(8), K)]))
                rec[f"tau={t:g}"] = float(np.mean(v))
            exact.append(rec)
    pd.DataFrame(exact).to_csv(out / "tables/table2b_informative_by_bin_tau_exact.csv", index=False)
    surv = lambda c: bool(c["terminated_with_eos"]) and int(c["completion_tokens"]) <= L
    cache_facts = {}
    for b in range(3):
        ps = [p for p in byp if bins[p] == b]
        fail = [sum(not surv(c) for c in byp[p]) for p in ps]
        rr = [c["reward"] for p in ps for c in byp[p]]
        cache_facts[BIN_NAMES[b]] = {
            "n_prompts": len(ps), "mean_reward_range": [float(min(np.mean([c["reward"] for c in byp[p]]) for p in ps)),
                                                        float(max(np.mean([c["reward"] for c in byp[p]]) for p in ps))],
            "completion_fail_512_frac": float(sum(fail) / (8 * len(ps))), "fully_masked_prompts": int(sum(f == 8 for f in fail)),
            "pairwise_tie_rate": float(np.mean([a["reward"] == c["reward"] for p in ps for a, c in itertools.combinations(byp[p], 2)])),
            "mean_tokens": float(np.mean([c["completion_tokens"] for p in ps for c in byp[p]])),
            "mean_reward": float(np.mean(rr))}
    allc = [c for p in byp for c in byp[p]]
    S["group_size"] = {
        "budget": gs["budget"], "replicates": gs["replicates"], "table": gs["table"], "by_bin": gs["by_bin"],
        "tau_sweep": gs["tau_sweep"], "exact_by_bin": exact, "cache_by_bin": cache_facts,
        "cache_fail_512_frac": float(np.mean([not surv(c) for c in allc])),
        "cache_fully_masked_prompts": int(sum(all(not surv(c) for c in byp[p]) for p in byp)),
        "cache_pairwise_tie_rate": float(np.mean([a["reward"] == c["reward"] for p in byp for a, c in itertools.combinations(byp[p], 2)])),
        "reward_survivors_vs_masked": [float(np.mean([c["reward"] for c in allc if surv(c)])),
                                       float(np.mean([c["reward"] for c in allc if not surv(c)]))],
        "spearman_tokens_reward": float(sps.spearmanr([c["completion_tokens"] for c in allc], [c["reward"] for c in allc])[0]),
    }
    t = pd.DataFrame(gs["table"])
    t.to_csv(out / "tables/table2_group_size.csv", index=False)
    pd.DataFrame(gs["by_bin"]).to_csv(out / "tables/table2a_group_size_by_bin.csv", index=False)

    fig, axs = plt.subplots(1, 3, figsize=(10.5, 2.9))
    ax = axs[0]
    tx = [1e-6, 0.05, 0.10, 0.25]
    for r in gs["tau_sweep"]:
        ys = [r[f"{x:g}"] for x in tx]
        ax.plot(range(4), ys, "-o", color=K_C[r["K"]], ms=4.5, label=f"K={r['K']}")
        ax.text(3.08, ys[-1], f"K={r['K']}", va="center", fontsize=7, color=INK)
    ax.set_xticks(range(4), [r"$10^{-6}$", "0.05", "0.10", "0.25"])
    ax.set_xlim(-0.2, 3.6)
    ax.set_xlabel(r"Informativeness threshold $\tau$ on $\sigma_g$ (reward units)")
    ax.set_ylabel("Informative-group rate")
    ax.set_title("(a) All prompts, budget = 48 generations", loc="left")
    for ax, col, title in ((axs[1], "tau=1e-06", r"(b) Strict gate $\tau=10^{-6}$ (exact)"),
                           (axs[2], "tau=0.25", r"(c) Resolution-aware $\tau=0.25$ (exact)")):
        w = 0.26
        for j, K in enumerate((2, 4, 8)):
            ys = [next(e[col] for e in exact if e["K"] == K and e["bin"] == BIN_NAMES[b]) for b in range(3)]
            ax.bar(np.arange(3) + (j - 1) * w, ys, w * 0.92, color=K_C[K], label=f"K={K}")
        ax.set_xticks(range(3), ["hard", "medium", "easy"])
        ax.set_ylim(0, 1.2)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_title(title, loc="left")
        ax.set_xlabel("Prompt-difficulty tertile")
    for ax in axs[1:]:
        ax.legend(loc="upper center", ncol=3, fontsize=6.8)
    fig.tight_layout()
    save(fig, out, "fig3_group_size")

    # ================================================================= normalization study
    ns = json.load(open(R / "normalization_study.json"))
    seq = {lt: jl(R / f"forks/fork_{lt}/seq_grads.jsonl") for lt in ("grpo", "dr_grpo")}
    logs = {lt: jl(R / f"forks/fork_{lt}/metrics.jsonl") for lt in ("grpo", "dr_grpo")}
    ho = {"midpoint": json.load(open(R / "forks/midpoint_baseline_heldout.json")),
          "grpo": json.load(open(R / "forks/fork_grpo/heldout.json")),
          "dr_grpo": json.load(open(R / "forks/fork_dr_grpo/heldout.json")),
          "standard": json.load(open(R / "standard/heldout.json"))}
    smp = {k: v["samples"] for k, v in ho.items()}
    rowpairs = list(zip(seq["grpo"], seq["dr_grpo"]))
    same_rollouts = all(a["length"] == b["length"] and a["reward"] == b["reward"] for a, b in rowpairs)
    ratio = [(b["grad_norm_seq"] / abs(b["adv"])) / (a["grad_norm_seq"] / abs(a["adv"])) for a, b in rowpairs]
    tl = [a["length"] / L for a, _ in rowpairs]
    fit = {}
    for lt in seq:
        d = pd.DataFrame(seq[lt])
        x, y = np.log(d["length"]), np.log(d["grad_norm_seq"] / d["adv"].abs())
        res = sps.linregress(x, y)
        fit[lt] = {"slope": float(res.slope), "slope_se": float(res.stderr), "intercept": float(res.intercept),
                   "r": float(res.rvalue), "n": len(d), "spearman_T_g": float(sps.spearmanr(d["length"], d["grad_norm_seq"])[0])}
    gl = {lt: [r["grad_norm"] for r in logs[lt] if r["stepped"]] for lt in logs}
    sig = [r["reward_std_within"] for r in logs["grpo"] if r["stepped"]]
    lens_tr = [r["response_length_trained"] for r in logs["grpo"] if r["stepped"]]
    upd = [r["update"] for r in logs["grpo"] if r["stepped"]]

    def ident(a, b, short=False):
        return [i for i in range(len(smp[a])) if smp[a][i]["response"] == smp[b][i]["response"]
                and smp[a][i]["length"] == smp[b][i]["length"] and (not short or len(smp[a][i]["response"]) < 1100)]

    noise = {}
    for a, b in (("midpoint", "grpo"), ("midpoint", "dr_grpo"), ("midpoint", "standard"), ("grpo", "dr_grpo")):
        idn = ident(a, b, short=True)
        d = np.abs([smp[b][i]["reward"] - smp[a][i]["reward"] for i in idn])
        allid = ident(a, b)
        div = [i for i in range(200) if i not in set(allid)]
        dd = np.array([smp[b][i]["reward"] - smp[a][i]["reward"] for i in div])
        dl = np.array([smp[b][i]["length"] - smp[a][i]["length"] for i in div])
        noise[f"{a}->{b}"] = {
            "identical_responses": len(allid), "identical_short_n": len(idn),
            "identical_short_abs_dR_median": float(np.median(d)), "identical_short_abs_dR_mean": float(d.mean()),
            "identical_short_frac_dR_gt_0.01": float(np.mean(d > 0.01)), "identical_short_abs_dR_max": float(d.max()),
            "diverged_n": len(div), "diverged_dR_mean": float(dd.mean()), "diverged_dR_se": float(dd.std(ddof=1) / np.sqrt(len(dd))),
            "diverged_dLen_mean": float(dl.mean()), "diverged_dLen_se": float(dl.std(ddof=1) / np.sqrt(len(dl))),
            "paired_dR": paired(smp[a], smp[b], "reward"), "paired_dLen": paired(smp[a], smp[b], "length"),
            "paired_dKLseq": paired(smp[a], smp[b], "kl_seq")}
    S["normalization"] = {
        "heldout": {k: v["summary"] for k, v in ho.items()}, "paired_noise": noise,
        "train_rollouts_identical_across_conditions": same_rollouts,
        "per_seq_ratio_vs_T_over_L": {"ratio": ratio, "T_over_L": tl,
                                      "max_abs_log_dev": float(np.max(np.abs(np.log(np.array(ratio) / np.array(tl))))),
                                      "updates_1_2_max_abs_log_dev": float(np.max(np.abs(np.log(np.array(ratio[:5]) / np.array(tl[:5])))))},
        "loglog_fit": fit, "allocation": ns["gradient_allocation"], "train_length_edges": ns["train_length_edges"],
        "heldout_length_edges": ns["heldout_length_edges"], "heldout_by_length": ns["heldout_by_length"],
        "per_update_grad_norm": {"update": upd, "sigma_g": sig, "trained_len": lens_tr, "grpo": gl["grpo"], "dr_grpo": gl["dr_grpo"],
                                 "ratio_dr_over_grpo": [d / g for d, g in zip(gl["dr_grpo"], gl["grpo"])]},
        "spearman_ratio_vs_sigma": float(sps.spearmanr(sig, [d / g for d, g in zip(gl["dr_grpo"], gl["grpo"])])[0]),
        "train": ns["train"],
    }
    pd.DataFrame([{"condition": k, **{kk: vv for kk, vv in v["summary"].items()}} for k, v in ho.items()]).to_csv(
        out / "tables/table3_heldout.csv", index=False)

    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.0))
    ax = axs[0]
    for lt, c, name in (("grpo", GRPO_C, "GRPO"), ("dr_grpo", DR_C, "Dr. GRPO")):
        d = pd.DataFrame(seq[lt])
        x, y = d["length"].to_numpy(), (d["grad_norm_seq"] / d["adv"].abs()).to_numpy()
        ax.scatter(x, y, s=22, color=c, edgecolor="white", linewidth=0.8, zorder=3, label=name)
        xs = np.geomspace(x.min(), x.max(), 50)
        ax.plot(xs, np.exp(fit[lt]["intercept"]) * xs ** fit[lt]["slope"], color=c, lw=1.4)
        x0 = xs[0] if lt == "grpo" else xs[-1]
        ax.annotate(f"{name}: slope {fit[lt]['slope']:+.2f} ± {fit[lt]['slope_se']:.2f}",
                    (x0, np.exp(fit[lt]["intercept"]) * x0 ** fit[lt]["slope"]),
                    xytext=(4, 12) if lt == "grpo" else (-4, -18), textcoords="offset points",
                    ha="left" if lt == "grpo" else "right", fontsize=7, color=INK)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([10, 20, 50, 100, 200, 500], ["10", "20", "50", "100", "200", "500"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel(r"Completion length $T_k$ (tokens, log)")
    ax.set_ylabel(r"$\|\nabla_\theta \ell_k\| \,/\, |A_k|$ (log)")
    ax.set_title("(a) Per-sequence gradient per unit advantage", loc="left")
    ax = axs[1]
    w = 0.36
    for j, (lt, c, name) in enumerate((("grpo", GRPO_C, "GRPO"), ("dr_grpo", DR_C, "Dr. GRPO"))):
        b = ns["gradient_allocation"][lt]["bins"]
        ys = [x["grad_mass_share"] for x in b]
        ax.bar(np.arange(3) + (j - 0.5) * w, ys, w * 0.92, color=c, label=name)
        for i, yv in enumerate(ys):
            ax.text(i + (j - 0.5) * w, yv + 0.015, f"{yv:.2f}", ha="center", fontsize=6.8, color=INK)
    e = ns["train_length_edges"]
    ax.set_xticks(range(3), [f"short\n{e[0]:.0f}–{e[1]:.0f}", f"medium\n{e[1]:.0f}–{e[2]:.0f}", f"long\n{e[2]:.0f}–{e[3]:.0f}"])
    ax.set_ylabel("Share of summed per-seq. grad norm")
    ax.set_ylim(0, 0.85)
    ax.legend(loc="upper left", ncol=1)
    ax.set_title("(b) Gradient mass by length tertile", loc="left")
    ax = axs[2]
    r = S["normalization"]["per_update_grad_norm"]
    ax.scatter(r["sigma_g"], r["ratio_dr_over_grpo"], s=34, color=DR_C, edgecolor="white", linewidth=0.8, zorder=3)
    for s_, q, uu, ln in zip(r["sigma_g"], r["ratio_dr_over_grpo"], r["update"], r["trained_len"]):
        off = (5, -9) if uu == 3 else (5, 2)
        ax.annotate(f"u{uu} (T̄={ln:.0f})", (s_, q), xytext=off, textcoords="offset points", fontsize=6.5, color=INK2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Within-group reward std $\sigma_g$ (log)")
    ax.set_ylabel("Update grad norm, Dr. GRPO / GRPO (log)")
    ax.set_title("(c) Step-scale ratio tracks $\\sigma_g$", loc="left")
    fig.tight_layout()
    save(fig, out, "fig4_length_normalization")

    fig, axs = plt.subplots(1, 2, figsize=(8.6, 2.8))
    ax = axs[0]
    w = 0.27
    for j, (k, c, name) in enumerate((("midpoint", MID_C, "midpoint"), ("grpo", GRPO_C, "GRPO (8 upd.)"),
                                      ("dr_grpo", DR_C, "Dr. GRPO (8 upd.)"))):
        edges = np.array(ns["heldout_length_edges"])
        d = pd.DataFrame(smp[k])
        d["bin"] = np.clip(np.searchsorted(edges[1:-1], d["length"], side="left"), 0, 2)
        g = d.groupby("bin")["reward"].agg(["mean", "sem"])
        ax.bar(np.arange(3) + (j - 1) * w, g["mean"], w * 0.92, yerr=g["sem"], color=c, label=name,
               error_kw={"elinewidth": 0.8, "ecolor": INK2, "capsize": 0})
    e = ns["heldout_length_edges"]
    ax.set_xticks(range(3), [f"short\n≤{e[1]:.0f}", f"medium\n{e[1]:.0f}–{e[2]:.0f}", f"long\n>{e[2]:.0f}"])
    ax.set_ylabel("Held-out reward (mean ± s.e.)")
    ax.legend(loc="upper left", fontsize=6.8)
    ax.set_title("(a) Held-out reward by length tertile", loc="left")
    ax = axs[1]
    a, b = smp["grpo"], smp["dr_grpo"]
    idn = set(ident("grpo", "dr_grpo"))
    d_id = [b[i]["reward"] - a[i]["reward"] for i in range(200) if i in idn]
    d_dv = [b[i]["reward"] - a[i]["reward"] for i in range(200) if i not in idn]
    bins_h = np.linspace(-3.2, 3.2, 33)
    ax.hist(d_id, bins=bins_h, color="#c9c8c2", label=f"identical response (n={len(d_id)})")
    ax.hist(d_dv, bins=bins_h, color=DR_C, alpha=0.85, label=f"diverged response (n={len(d_dv)})")
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel(r"Paired held-out $\Delta$reward, Dr. GRPO − GRPO")
    ax.set_ylabel("Prompts")
    ax.legend(loc="upper left", fontsize=6.8)
    ax.set_title("(b) Paired differences: RM noise vs. divergence", loc="left")
    fig.tight_layout()
    save(fig, out, "fig5_heldout_normalization")

    json.dump(S, open(out / "stats.json", "w"), indent=1, default=float)
    print(json.dumps({k: (v if k != "group_size" else {kk: vv for kk, vv in v.items() if kk not in ("table", "by_bin")})
                      for k, v in S.items() if k != "normalization"}, indent=1, default=float)[:6000])
    nn = S["normalization"]
    print(json.dumps({k: nn[k] for k in ("train_rollouts_identical_across_conditions", "loglog_fit", "spearman_ratio_vs_sigma",
                                         "paired_noise")}, indent=1)[:6000])
    print(json.dumps(nn["per_seq_ratio_vs_T_over_L"]["max_abs_log_dev"]), nn["per_seq_ratio_vs_T_over_L"]["updates_1_2_max_abs_log_dev"])
    print(json.dumps(nn["per_update_grad_norm"], indent=0))


if __name__ == "__main__":
    main()
