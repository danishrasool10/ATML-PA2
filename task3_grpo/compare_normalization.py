from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import load_json, save_json
from common.evaluate import evaluate_policy
from task3_grpo.continue_train import prepare_grpo_continuation, reset_to_midpoint, run_grpo

CONDITIONS = ("grpo", "dr_grpo")
TRAJ = ["reward", "kl", "response_length", "response_length_trained", "truncation_rate", "entropy", "policy_loss",
        "grad_norm", "frac_uninformative_groups"]
HELDOUT = ["reward_mean", "reward_stderr", "kl_per_token", "kl_per_sequence", "response_length", "truncation_rate",
           "entropy", "distinct2"]
DEFINITIONS = {
    "grad_norm_seq": "||grad_theta (1/D_k) sum_t min(rho A_k, clip(rho) A_k)|| for one completion at rho = 1 "
                     "(D_k = T_k for grpo, L_max for dr_grpo); measured per sequence before the optimizer step",
    "g_per_adv": "grad_norm_seq / |A_k|: gradient magnitude per unit advantage, isolates the length normalization",
    "loglog_slope": "OLS slope of log(g_per_adv) on log(T_k). 1/T_k normalization pushes it down by ~1 relative to "
                    "the constant 1/L_max of Dr. GRPO on the same data",
    "length_bins": "equal-count tertiles of T_k over the trained completions of BOTH forks pooled (one rule, shared "
                   "edges); held-out length bins use tertiles of the untouched midpoint's held-out lengths",
    "paired": "held-out prompts, order and per-batch seeds are identical across conditions, so deltas are paired",
}


class LazyBundle:
    def __init__(self, config_path: str):
        self.config_path, self._bundle = config_path, None

    def __call__(self) -> dict:
        if self._bundle is None:
            self._bundle = prepare_grpo_continuation(self.config_path)
        return self._bundle


def fork_dir(cfg: dict, loss_type: str) -> Path:
    return repo_path(cfg["results_dir"]) / "forks" / f"fork_{loss_type}"


def load_fork(d: Path) -> dict | None:
    if not all((d / f).exists() for f in ("summary.json", "metrics.jsonl", "seq_grads.jsonl", "heldout.json")):
        return None
    return {"summary": load_json(d / "summary.json"), "logs": read_jsonl(d / "metrics.jsonl"),
            "seq": read_jsonl(d / "seq_grads.jsonl"), "heldout": load_json(d / "heldout.json")}


def run_condition(bundle: LazyBundle, cfg: dict, config_path: str, loss_type: str, *, force: bool,
                  eval_prompts: int | None) -> dict:
    """Exact midpoint -> fork_updates updates with `loss_type` -> held-out evaluation. Everything except the loss
    accumulator and the advantage std division (both keyed on loss_type) is shared: reward model, beta, epsilon,
    prompt schedule, per-update sampling seeds, K, max_completion_length, learning rate, evaluation protocol."""
    d = fork_dir(cfg, loss_type)
    if not force and (f := load_fork(d)) is not None:
        print(f"[fork] reusing cached {d.name}")
        return f
    run_grpo(config_path, None, int(cfg["fork_updates"]), loss_type, d.name, bundle=bundle(), evaluate_after=True,
             save_adapter=False, results_subdir=f"forks/{d.name}", eval_prompts=eval_prompts, track_seq_grads=True)
    return load_fork(d)


def midpoint_baseline(bundle: LazyBundle, cfg: dict, *, force: bool, eval_prompts: int | None) -> dict:
    path = repo_path(cfg["results_dir"]) / "forks" / "midpoint_baseline_heldout.json"
    if path.exists() and not force:
        return load_json(path)
    b = bundle()
    reset_to_midpoint(b)
    res = evaluate_policy(b["policy"], b["tokenizer"], b["reward_model"], b["reward_tokenizer"], b["eval_rows"],
                          b["cfg"], max_prompts=eval_prompts)
    save_json(path, res)
    return res


def _bin(x, edges):
    return np.clip(np.searchsorted(edges[1:-1], x, side="left"), 0, len(edges) - 2)


def _spearman(a, b) -> float:
    return float(pd.Series(a).corr(pd.Series(b), method="spearman")) if len(a) > 2 else float("nan")


def gradient_allocation(seq: list[dict], edges: np.ndarray) -> dict:
    df = pd.DataFrame(seq)
    df = df[(df["adv"].abs() > 0) & (df["grad_norm_seq"] > 0)].copy() if len(df) else df
    if len(df) < 3:
        return {"n": int(len(df)), "loglog_slope": float("nan"), "bins": []}
    df["g_per_adv"] = df["grad_norm_seq"] / df["adv"].abs()
    df["bin"] = _bin(df["length"].to_numpy(), edges)
    slope = float(np.polyfit(np.log(df["length"]), np.log(df["g_per_adv"]), 1)[0])
    total = df["grad_norm_seq"].sum()
    bins = (df.groupby("bin")
              .agg(n=("length", "size"), length=("length", "mean"), adv_abs=("adv", lambda a: a.abs().mean()),
                   grad_norm_seq=("grad_norm_seq", "mean"), g_per_adv=("g_per_adv", "mean"),
                   grad_mass=("grad_norm_seq", "sum"))
              .reset_index())
    bins["grad_mass_share"] = bins["grad_mass"] / total
    return {"n": int(len(df)), "loglog_slope": slope,
            "spearman_length_grad_norm": _spearman(df["length"], df["grad_norm_seq"]),
            "spearman_length_g_per_adv": _spearman(df["length"], df["g_per_adv"]),
            "bins": bins.drop(columns="grad_mass").to_dict("records")}


def heldout_by_length(samples: list[dict], edges: np.ndarray) -> list[dict]:
    df = pd.DataFrame(samples)
    df["bin"] = _bin(df["length"].to_numpy(), edges)
    g = df.groupby("bin").agg(n=("reward", "size"), reward=("reward", "mean"), length=("length", "mean"))
    g["share"] = g["n"] / len(df)
    return g.reset_index().to_dict("records")


def paired_delta(a: list[dict], b: list[dict], key: str) -> dict:
    x = np.array([s[key] for s in b], float) - np.array([s[key] for s in a], float)
    return {"mean": float(x.mean()), "stderr": float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")}


def qualitative(a: list[dict], b: list[dict], topk: int = 2) -> dict:
    pairs = [{"idx": s["idx"], "prompt": s["prompt"], "len_grpo": s["length"], "len_dr": t["length"],
              "reward_grpo": s["reward"], "reward_dr": t["reward"], "d_len": t["length"] - s["length"],
              "d_reward": t["reward"] - s["reward"], "response_grpo": s["response"][:500],
              "response_dr": t["response"][:500]} for s, t in zip(a, b)]
    pairs.sort(key=lambda p: p["d_len"])
    return {"dr_much_shorter": pairs[:topk], "dr_much_longer": pairs[::-1][:topk],
            "note": "selected by length change only; read before citing"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--force", action="store_true", help="re-run forks even if their results exist on disk")
    ap.add_argument("--eval-prompts", type=int, default=None)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    bundle = LazyBundle(args.config)
    print("Fork updates:", cfg["fork_updates"], "| conditions:", CONDITIONS)

    base_full = midpoint_baseline(bundle, cfg, force=args.force, eval_prompts=args.eval_prompts)
    forks = {lt: run_condition(bundle, cfg, args.config, lt, force=args.force, eval_prompts=args.eval_prompts)
             for lt in CONDITIONS}
    a, b = (forks[lt]["heldout"]["samples"] for lt in CONDITIONS)
    n = {len(base_full["samples"]), len(a), len(b)}
    if len(n) != 1 or [s["idx"] for s in a] != [s["idx"] for s in b]:
        raise RuntimeError(f"held-out sets differ across conditions ({n}); re-run with --force and one --eval-prompts")

    u1 = [[c["reward"] for s in forks[lt]["logs"][0]["samples"] for c in s["completions"]] for lt in CONDITIONS]
    matched = bool(np.allclose(*u1, atol=1e-3))
    print(f"update-1 rollouts identical across conditions (same midpoint, prompts, seeds): {matched}")

    base = base_full["summary"]
    rows = [{"condition": "midpoint", **{k: base[k] for k in HELDOUT}, "d_reward_vs_mid": 0.0}]
    for lt in CONDITIONS:
        h = forks[lt]["heldout"]["summary"]
        rows.append({"condition": lt, **{k: h[k] for k in HELDOUT}, "d_reward_vs_mid": h["reward_mean"] - base["reward_mean"]})
    paired = {k: paired_delta(a, b, k) for k in ("reward", "length", "kl_seq")}

    seq_all = [s for lt in CONDITIONS for s in forks[lt]["seq"]]
    tr_edges = np.quantile([s["length"] for s in seq_all], [0, 1 / 3, 2 / 3, 1]) if seq_all else np.array([0, 1, 2, 3.])
    alloc = {lt: gradient_allocation(forks[lt]["seq"], tr_edges) for lt in CONDITIONS}
    ho_edges = np.quantile([s["length"] for s in base_full["samples"]], [0, 1 / 3, 2 / 3, 1])
    ho_len = {name: heldout_by_length(s, ho_edges) for name, s in
              (("midpoint", base_full["samples"]), ("grpo", a), ("dr_grpo", b))}
    traj = {lt: {m: [r.get(m) for r in forks[lt]["logs"]] for m in TRAJ} for lt in CONDITIONS}
    train = {lt: {k: forks[lt]["summary"][k] for k in ("first_window", "last_window", "signal", "generated_tokens",
                                                         "trained_tokens", "train_wall_s", "peak_vram_gib")}
             for lt in CONDITIONS}

    fmt = lambda x: f"{x:.4f}"
    pd.set_option("display.width", 200)
    print("\nHeld-out (common protocol of common.evaluate):")
    print(pd.DataFrame(rows).to_string(index=False, float_format=fmt))
    print("\nPaired held-out deltas (dr_grpo - grpo):", {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in paired.items()})
    print("\nPer-update trajectories:")
    for m in ("reward", "kl", "response_length", "grad_norm"):
        print(f"  {m:16s}", " | ".join(f"{lt}: " + " ".join("   n/a" if v is None else f"{v:6.3f}" for v in traj[lt][m])
                                       for lt in CONDITIONS))
    print(f"\nLength-conditioned gradient allocation (train length tertile edges {np.round(tr_edges, 1).tolist()}):")
    for lt in CONDITIONS:
        al = alloc[lt]
        print(f"  {lt:8s} n={al['n']} loglog slope(g/|A| vs T)={al['loglog_slope']:+.3f} "
              f"spearman(T, g)={al.get('spearman_length_grad_norm', float('nan')):+.3f}")
        if al["bins"]:
            print(pd.DataFrame(al["bins"]).to_string(index=False, float_format=fmt))
    print(f"\nHeld-out reward by length tertile (midpoint edges {np.round(ho_edges, 1).tolist()}):")
    for name, v in ho_len.items():
        print(f"  {name:8s}", "  ".join(f"bin{r['bin']}: n={r['n']} R={r['reward']:+.3f}" for r in v))

    examples = qualitative(a, b)
    out = repo_path(cfg["results_dir"])
    save_json(out / "normalization_study.json", {
        "definitions": DEFINITIONS, "update1_rollouts_matched": matched, "heldout": rows, "paired_dr_minus_grpo": paired,
        "gradient_allocation": alloc, "train_length_edges": tr_edges.tolist(), "heldout_length_edges": ho_edges.tolist(),
        "heldout_by_length": ho_len, "trajectories": traj, "train": train,
    })
    save_json(out / "normalization_examples.json", examples)
    print("\nsaved ->", out / "normalization_study.json")


if __name__ == "__main__":
    main()
