from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common.data import repo_path
from common.logging_utils import load_json, save_json
from common.metrics import safe_corr
from task2_ppo.continue_train import fork_name, midpoint_baseline, prepare_ppo_continuation, run_fork

# per-update trajectory metrics compared across beta (all logged by continue_train.run_ppo)
TRAJ_METRICS = ["reward", "kl", "entropy", "response_length", "distinct2", "truncation_rate"]


def series(fork: dict, key: str) -> np.ndarray:
    return np.array([np.nan if r.get(key) is None else r[key] for r in fork["logs"]], dtype=float)


def first_divergence(ref: np.ndarray, other: np.ndarray, tol_std: float) -> int | None:
    """First 1-indexed update where |other - ref| > tol_std * std(ref trajectory). Because every fork replays the same
    prompts/seeds from the same midpoint, update 1 is identical across conditions by construction and any later
    difference is attributable to beta (paired comparison)."""
    tol = tol_std * max(float(np.nanstd(ref)), 1e-8)
    for u, (a, b) in enumerate(zip(ref, other)):
        if abs(a - b) > tol:
            return u + 1
    return None


def mine_examples(base_samples, weak_samples, topk: int = 2) -> dict:
    """Heuristic candidate retrieval for the qualitative section (weakest-KL fork vs untouched midpoint, per prompt).
    These are CANDIDATES: reward up + no degeneration signals vs reward up + length blow-up / new truncation /
    repetition. Read them before citing them."""
    cands = []
    for b, w in zip(base_samples, weak_samples):
        cands.append({
            "idx": b["idx"], "prompt": b["prompt"],
            "reward_base": b["reward"], "reward_fork": w["reward"], "d_reward": w["reward"] - b["reward"],
            "len_base": b["length"], "len_fork": w["length"], "len_ratio": w["length"] / max(b["length"], 1),
            "d_distinct2": w["distinct2"] - b["distinct2"],
            "newly_truncated": bool(w["truncated"] and not b["truncated"]),
            "response_base": b["response"][:600], "response_fork": w["response"][:600],
        })
    up = [c for c in cands if c["d_reward"] > 0]
    disagree = [c for c in up if c["len_ratio"] >= 1.5 or c["newly_truncated"] or c["d_distinct2"] <= -0.10]
    agree = [c for c in up if 0.75 <= c["len_ratio"] <= 1.25 and c["d_distinct2"] > -0.02 and not c["newly_truncated"]]
    key = lambda c: -c["d_reward"]
    return {"reward_and_quality_agree": sorted(agree, key=key)[:topk],
            "reward_up_quality_not": sorted(disagree, key=key)[:topk],
            "note": "heuristic candidates only; verify by reading before using as evidence"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--force", action="store_true", help="recompute forks even if cached on disk")
    ap.add_argument("--eval-prompts", type=int, default=None, help="subsample held-out prompts (default: all)")
    ap.add_argument("--div-tol", type=float, default=0.25,
                    help="divergence threshold in units of the reference trajectory's std")
    args = ap.parse_args()

    bundle = prepare_ppo_continuation(args.config)
    cfg = bundle["cfg"]
    eps, ref_beta = float(cfg["clip_epsilon"]), float(cfg["kl_beta"])
    print("KL beta conditions:", cfg["kl_values"])
    print("Fork update budget:", cfg["fork_updates"])

    base_full = midpoint_baseline(bundle, force=args.force, eval_prompts=args.eval_prompts)
    base = base_full["summary"]

    forks = {}
    # ============== ABLATION LOOP 2/2 -- KL-PRESSURE STUDY: beta_KL in {0, 0.10, 0.20}, epsilon fixed ==============
    # Same midpoint snapshot, prompt stream, per-update seeds, update budget and held-out protocol for every beta.
    # (beta=0.10, eps=0.20) is the same fork as in the clipping study and is read from the on-disk cache if present.
    for beta in cfg["kl_values"]:
        forks[float(beta)] = run_fork(bundle, args.config, eps, float(beta),
                                      force=args.force, eval_prompts=args.eval_prompts)
    # ===============================================================================================================

    # ------------------------------------------------------------------ held-out + trajectory table
    rows = []
    for beta, f in forks.items():
        h = f["heldout"]
        r, kl = series(f, "reward"), series(f, "kl")
        k = f["window"]
        rows.append({
            "beta": beta,
            "heldout_reward": h["reward_mean"], "d_vs_mid": h["reward_mean"] - base["reward_mean"],
            "heldout_kl_tok": h["kl_per_token"], "heldout_kl_seq": h["kl_per_sequence"],
            "heldout_entropy": h["entropy"], "heldout_len": h["response_length"],
            "heldout_trunc": h["truncation_rate"], "heldout_distinct2": h["distinct2"],
            "train_reward_gain": f["last_window"]["reward"] - f["first_window"]["reward"],
            "train_kl_gain": f["last_window"]["kl"] - f["first_window"]["kl"],
            "corr_reward_kl": safe_corr(r, kl),
            "window": k,
        })
    print("\nBaseline (midpoint, held-out):", {k: round(v, 4) for k, v in base.items() if isinstance(v, float)})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ------------------------------------------------------------------ RQ2: what moves first when beta is lowered?
    ref = forks[ref_beta]
    div, effect = {}, {}
    for beta, f in forks.items():
        if beta == ref_beta:
            continue
        div[str(beta)] = {m: first_divergence(series(ref, m), series(f, m), args.div_tol) for m in TRAJ_METRICS}
        effect[str(beta)] = {
            m: float(np.nanmean(np.abs(series(f, m) - series(ref, m))) / max(np.nanstd(series(ref, m)), 1e-8))
            for m in TRAJ_METRICS
        }
        effect[str(beta)]["per_update_delta"] = {m: (series(f, m) - series(ref, m)).tolist() for m in TRAJ_METRICS}
    print(f"\nFirst update where |metric(beta) - metric(beta={ref_beta:g})| > {args.div_tol} x std(ref trajectory) "
          "(None = no divergence within the fork budget):")
    print(pd.DataFrame(div).T.to_string())
    print("\nMean |delta| in units of ref-trajectory std (which observable moved most):")
    print(pd.DataFrame({b: {m: v for m, v in e.items() if m != "per_update_delta"} for b, e in effect.items()})
          .T.to_string(float_format=lambda x: f"{x:.3f}"))

    # ------------------------------------------------------------------ RQ3: overoptimization evidence + examples
    # weakest-KL-pressure fork vs the untouched midpoint, matched per held-out prompt (saved by run_ppo)
    examples = mine_examples(base_full["samples"], _load_samples(cfg, min(forks), eps))
    for tag, items in examples.items():
        if isinstance(items, list):
            print(f"\n[{tag}]")
            for c in items:
                print(f"  prompt#{c['idx']} dR={c['d_reward']:+.2f} len x{c['len_ratio']:.2f} "
                      f"dDistinct2={c['d_distinct2']:+.2f}\n    Q: {c['prompt'][:140]!r}\n"
                      f"    mid : {c['response_base'][:200]!r}\n    beta={min(forks):g}: {c['response_fork'][:200]!r}")

    out = repo_path(cfg["results_dir"])
    save_json(out / "kl_study.json", {
        "baseline_heldout": base, "table": rows, "first_divergence": div, "effect_size": effect,
        "forks": {str(k): {kk: vv for kk, vv in v.items() if kk != "logs"} for k, v in forks.items()},
        "definitions": {
            "train_reward_gain": "last-window minus first-window mean learned reward over the fork's updates",
            "corr_reward_kl": "Pearson corr of per-update (reward, kl) trajectories; high positive => reward bought with drift",
            "heldout_*": "common held-out protocol of evaluate.evaluate_policy, eval_max_response_length cap",
        },
    })
    save_json(out / "qualitative_candidates.json", examples)


def _load_samples(cfg: dict, beta: float, eps: float):
    path = repo_path(cfg["results_dir"]) / "forks" / fork_name(eps, beta) / "heldout.json"
    return load_json(path)["samples"]


if __name__ == "__main__":
    main()
