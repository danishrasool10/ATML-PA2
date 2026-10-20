"""Task 5 final RLVR-vs-RLAIF comparison: in-domain GSM8K, controlled reward diagnostics, SVAMP transfer.

Torch-free. Reads results/task5_feedback/{gsm,transfer,diagnostic}_summary.json; --recompute rebuilds them from the
per-example records under outputs/task5_feedback/ first.
"""
from __future__ import annotations

import argparse
import math

import pandas as pd

from common.data import load_yaml
from task5_feedback.analysis import GRADERS, aggregate_diagnostics, aggregate_math
from task5_feedback.artifacts import (POLICIES, REFERENCE, Z95, load_json, results_dir, save_json, summary_path,
                                      to_markdown)

# (reported metric, pooled scope, rate key)
SIGNAL_ROWS = [
    ("better_preference_rate", "all_pairs", "better"),
    ("tie_rate", "all_pairs", "tie"),
    ("wrong_preference_rate", "all_pairs", "wrong"),
    ("better_preference_rate (dominance pairs)", "dominance_pairs", "better"),
    ("wrong_preference_rate (dominance pairs)", "dominance_pairs", "wrong"),
    ("S_reason", "S_reason", "better"),
    ("S_reason tie rate", "S_reason", "tie"),
    ("S_reason (incl. filler)", "S_reason_incl_filler", "better"),
    ("S_outcome", "S_outcome", "better"),
    ("S_outcome tie rate", "S_outcome", "tie"),
    ("S_outcome (minimal: final line only)", "S_outcome_minimal", "better"),
    ("filler_susceptibility", "filler", "wrong"),
    ("distractor_robustness", "distractor", "better"),
]


def _summary(cfg, name: str) -> dict:
    p = summary_path(cfg, name)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing; run the corresponding Task 5 step first")
    return load_json(p)


def _frame(records: dict) -> pd.DataFrame:
    return pd.DataFrame.from_dict(records, orient="index").apply(pd.to_numeric, errors="coerce")


def _ci(v, lo, hi, fmt="{:.3f}") -> str:
    if v is None or not math.isfinite(v):
        return "—"
    return fmt.format(v) if lo is None or not math.isfinite(lo) else f"{fmt.format(v)} [{fmt.format(lo)}, {fmt.format(hi)}]"


def policy_table(gsm: dict, tr: dict) -> pd.DataFrame:
    G, T = _frame(gsm["summary"]), _frame(tr["summary"])
    ng, nt = gsm["n_examples"], tr["n_examples"]
    nan = math.nan
    rows = {}
    for p in POLICIES:
        g, t = G.loc[p], T.loc[p]
        # independent samples: Wald interval on the accuracy gap
        drop = g["accuracy"] - t["accuracy"]
        se = math.sqrt(g["accuracy"] * (1 - g["accuracy"]) / ng + t["accuracy"] * (1 - t["accuracy"]) / nt)
        rows[p] = {
            **{f"gsm_{k}": g.get(k, nan) for k in (
                "accuracy", "accuracy_lo", "accuracy_hi", "format_compliance", "accuracy_given_format",
                "win_rate_vs_sft", "win_rate_lo", "win_rate_hi", "win_rate_vs_sft_differing", "identical_rate",
                "len_mean", "len_std", "len_median", "len_q25", "len_q75", "len_iqr", "truncation_rate",
                "verifier_judge_agreement", "verifier_judge_kappa", "verifier_judge_decisive_agreement",
                "judge_reverses_verifier_rate", "judge_slot_A_rate", "judge_prefers_longer_rate",
                "delta_accuracy_vs_sft", "mcnemar_p")},
            **{f"svamp_{k}": t.get(k, nan) for k in (
                "accuracy", "accuracy_lo", "accuracy_hi", "format_compliance", "win_rate_vs_sft", "win_rate_lo",
                "win_rate_hi", "len_mean", "len_std", "truncation_rate", "verifier_judge_decisive_agreement",
                "delta_accuracy_vs_sft", "mcnemar_p")},
            "transfer_drop": drop, "transfer_drop_lo": drop - Z95 * se, "transfer_drop_hi": drop + Z95 * se,
            "transfer_drop_rel": drop / g["accuracy"] if g["accuracy"] > 0 else nan,
            "rl_gain_change_vs_sft": (t.get("delta_accuracy_vs_sft", nan) - g.get("delta_accuracy_vs_sft", nan))
            if p != REFERENCE else nan,
        }
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "policy"
    return out


def signal_table(diag: dict) -> pd.DataFrame:
    pooled = pd.DataFrame(diag["pooled"]).set_index(["scope", "grader"])
    rows = {}
    for name, scope, key in SIGNAL_ROWS:
        rows[name] = {f"{g}{s}": pooled.loc[(scope, g), f"{key}{s}"] for g in GRADERS for s in ("", "_lo", "_hi")}
        rows[name]["n_pairs"] = pooled.loc[(scope, GRADERS[0]), "n"]
    out = pd.DataFrame.from_dict(rows, orient="index").apply(pd.to_numeric, errors="coerce")
    out.index.name = "metric"
    return out


def _md_tables(pt: pd.DataFrame, st: pd.DataFrame, gsm: dict, tr: dict, diag: dict) -> str:
    c = lambda r, k: _ci(r[k], r.get(f"{k}_lo"), r.get(f"{k}_hi"))
    win = lambda r, pre: _ci(r[f"{pre}_win_rate_vs_sft"], r[f"{pre}_win_rate_lo"], r[f"{pre}_win_rate_hi"])
    dec = gsm["decoding"]
    step1 = pd.DataFrame({p: {
        "accuracy": c(r, "gsm_accuracy"), "format": f"{r['gsm_format_compliance']:.3f}",
        "win vs SFT": win(r, "gsm"), "len mean ± std": f"{r['gsm_len_mean']:.1f} ± {r['gsm_len_std']:.1f}",
        "len IQR [q25, q75]": f"{r['gsm_len_iqr']:.0f} [{r['gsm_len_q25']:.0f}, {r['gsm_len_q75']:.0f}]",
        "trunc": f"{r['gsm_truncation_rate']:.3f}",
        "V-J agree": _ci(r["gsm_verifier_judge_agreement"], None, None),
        "V-J agree (decisive)": _ci(r["gsm_verifier_judge_decisive_agreement"], None, None),
        "Δacc vs SFT (McNemar p)": "—" if p == REFERENCE else
        f"{r['gsm_delta_accuracy_vs_sft']:+.3f} (p={r['gsm_mcnemar_p']:.3g})",
    } for p, r in pt.iterrows()}).T.rename_axis("policy")
    step3 = pd.DataFrame({p: {
        "GSM8K acc": f"{r['gsm_accuracy']:.3f}", "SVAMP acc": c(r, "svamp_accuracy"),
        "SVAMP win vs SFT": win(r, "svamp"), "SVAMP len mean": f"{r['svamp_len_mean']:.1f}",
        "drop (GSM − SVAMP)": _ci(r["transfer_drop"], r["transfer_drop_lo"], r["transfer_drop_hi"], "{:+.3f}"),
        "relative drop": f"{r['transfer_drop_rel']:+.1%}" if math.isfinite(r["transfer_drop_rel"]) else "—",
        "Δacc vs SFT: GSM → SVAMP": "—" if p == REFERENCE else
        f"{r['gsm_delta_accuracy_vs_sft']:+.3f} → {r['svamp_delta_accuracy_vs_sft']:+.3f}",
    } for p, r in pt.iterrows()}).T.rename_axis("policy")
    signals = pd.DataFrame({m: {g: _ci(r[g], r[f"{g}_lo"], r[f"{g}_hi"]) for g in GRADERS} | {"n pairs": int(r["n_pairs"])}
                            for m, r in st.iterrows()}).T.rename_axis("metric")
    bt = pd.DataFrame(diag["by_type"])
    types = (bt.pivot_table(index=["pair_type", "relation", "reference_better"], columns="grader",
                            values=["better", "tie", "wrong"], sort=False))
    types.columns = [f"{g} {k}" for k, g in types.columns]
    types = types[[f"{g} {k}" for g in GRADERS for k in ("better", "tie", "wrong")]]
    var = pd.DataFrame(diag["variants"]).set_index("short")[
        ["reasoning_quality", "expected_exact_reward", "verifier_reward", "judge_borda_mean", "judge_borda_lo",
         "judge_borda_hi", "judge_top1_rate", "words_mean"]]
    cont = pd.DataFrame(gsm["contingency"])
    jb = diag["judge_bias"]
    bias = (f"diagnostics: P(slot A | decisive) = {jb['slot_A_rate']:.3f}, P(better shown first) = "
            f"{jb['better_shown_first_rate']:.3f}, P(prefers longer | decisive) = {jb['prefers_longer_rate']:.3f}, "
            f"judge tie rate = {jb['tie_rate']:.3f}")
    return "\n\n".join([
        f"## Step 1: in-domain GSM8K (n={gsm['n_examples']}, greedy, max_new_tokens={dec['max_new_tokens']}, "
        f"repetition_penalty={dec['repetition_penalty']}, batch={dec['batch_size']})",
        to_markdown(step1),
        "### GSM8K: judge preference vs verifier outcome (differing policy/SFT pairs)", to_markdown(cont),
        f"## Step 2: controlled reward diagnostics ({diag['n_problems']} problems x {diag['n_pairs'] // diag['n_problems']} "
        "pairs; 95% problem-cluster bootstrap CIs)",
        to_markdown(signals),
        "### By pair type (reference-better listed in `reference_better`)", to_markdown(types),
        "### By variant (judge_borda = RLAIF group reward over the 5 variants)", to_markdown(var),
        "### Judge biases", bias,
        f"## Step 3: SVAMP transfer (n={tr['n_examples']}); drop CI is Wald on independent samples",
        to_markdown(step3),
    ]) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--recompute", action="store_true", help="rebuild step summaries from outputs/ records")
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    if args.recompute:
        aggregate_math(cfg, "gsm", n_boot=args.bootstrap)
        aggregate_math(cfg, "transfer", n_boot=args.bootstrap)
        aggregate_diagnostics(cfg, n_boot=args.bootstrap)
    gsm, tr, diag = _summary(cfg, "gsm"), _summary(cfg, "transfer"), _summary(cfg, "diagnostic")

    pt, st = policy_table(gsm, tr), signal_table(diag)
    rd = results_dir(cfg)
    pt.to_csv(rd / "feedback_policy_comparison.csv")
    st.to_csv(rd / "feedback_signal_comparison.csv")
    pt[[c for c in pt.columns if c.startswith("transfer_") or c in ("gsm_accuracy", "svamp_accuracy")]].to_csv(
        rd / "transfer_drop.csv")
    save_json(rd / "feedback_comparison.json", {
        "policies": pt.to_dict(orient="index"),
        "reward_signals": st.to_dict(orient="index"),
        "diagnostic_headline": diag["headline"],
        "judge_bias": {"diagnostics": diag["judge_bias"],
                       **{d: {p: {k: s["summary"][p].get(k) for k in ("judge_slot_A_rate", "judge_prefers_longer_rate")}
                              for p in POLICIES if p != REFERENCE} for d, s in (("gsm", gsm), ("transfer", tr))}},
        "decoding": gsm["decoding"],
        "definitions": {"math": gsm["definitions"], "diagnostics": diag["definitions"],
                        "transfer_drop": "GSM8K accuracy - SVAMP accuracy (positive = drop); Wald 95% CI",
                        "transfer_drop_rel": "transfer_drop / GSM8K accuracy",
                        "rl_gain_change_vs_sft": "(SVAMP acc - SFT SVAMP acc) - (GSM8K acc - SFT GSM8K acc)"},
    })
    md = _md_tables(pt, st, gsm, tr, diag)
    (rd / "tables.md").write_text(md, encoding="utf-8")
    print(md)
    print("saved ->", rd)


if __name__ == "__main__":
    main()
