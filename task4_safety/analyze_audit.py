"""Judge-vs-manual agreement on the fixed blind audit (torch-free). Human labels are the reference throughout.

Disagreement taxonomy over LABEL_AXES = (implied prompt class, behavior):
  behavior  - judge and human disagree on comply vs refuse
  intent    - same behavior, different reading of whether the prompt was harmful
  both      - both axes differ
  ambiguous - exactly one side is AMBIGUOUS
"""
from __future__ import annotations

import argparse
import math
import re

import numpy as np
import pandas as pd

from common.data import load_yaml
from task4_safety.artifacts import (LABEL_AXES, LABELS, POLICIES, audit_dir, load_rows, read_worksheet, save_json,
                                    to_markdown)
from task4_safety.evaluate_safety import HEADLINE, class_rates, wilson

ALIASES = {"SA": "SAFE_ANSWER", "OR": "OVER_REFUSAL", "JR": "JUSTIFIED_REFUSAL", "UC": "UNSAFE_COMPLIANCE",
           "AMB": "AMBIGUOUS"}
JUDGE_COLS = ["policy", "xstest_id", "prompt", "response", "response_tokens", "judge_label", "judge_confidence",
              "judge_rationale_tag"]
_IMPLIED = {k: v[0] for k, v in LABEL_AXES.items()}


def normalize_label(x) -> str:
    t = re.sub(r"[\s\-]+", "_", str(x).strip().upper())
    return ALIASES.get(t, t)


def load_audit(cfg, worksheet=None, allow_partial: bool = False) -> pd.DataFrame:
    """Joins on audit_id only; Prompt/Response text in the worksheet is never trusted (spreadsheet edits)."""
    ad = audit_dir(cfg)
    ws = read_worksheet(worksheet or ad / "audit_worksheet.csv")
    if "Human_Notes" not in ws:
        ws["Human_Notes"] = ""
    key = pd.read_csv(ad / "audit_key.csv", dtype={"audit_id": str})
    policies = tuple(p for p in POLICIES if p in set(key["policy"]))
    judged = load_rows(cfg, "judged", policies)[JUDGE_COLS]
    df = (key.merge(ws[["audit_id", "Human_Label", "Human_Notes"]], on="audit_id", how="left", validate="1:1")
             .merge(judged, on=["policy", "xstest_id"], how="left", validate="1:1"))
    if df["Human_Label"].isna().any():
        raise ValueError(f"{int(df['Human_Label'].isna().sum())} key rows missing from worksheet (audit_id edited?)")
    if df["judge_label"].isna().any():
        raise ValueError("audit rows without judge labels; run task4_safety.judge_responses first")
    df["human_label"] = df["Human_Label"].map(normalize_label)
    blank = df["human_label"].eq("")
    invalid = ~blank & ~df["human_label"].isin(LABELS)
    if invalid.any():
        raise ValueError(f"invalid Human_Label: {df.loc[invalid, ['audit_id', 'Human_Label']].values.tolist()[:10]}")
    if blank.any() and not allow_partial:
        raise ValueError(f"{int(blank.sum())}/{len(df)} rows unlabelled; --allow-partial analyses the labelled subset")
    df = df.loc[~blank].drop(columns="Human_Label").rename(columns={"Human_Notes": "human_notes"})
    df["policy"] = pd.Categorical(df["policy"], list(policies), ordered=True)
    df["error_type"] = [error_type(h, j) for h, j in zip(df["human_label"], df["judge_label"])]
    return df.sort_values(["xstest_id", "policy"]).reset_index(drop=True)


def error_type(h: str, j: str) -> str:
    if h == j:
        return "agree"
    if "AMBIGUOUS" in (h, j):
        return "ambiguous"
    (hc, hb), (jc, jb) = LABEL_AXES[h], LABEL_AXES[j]
    return {(True, False): "intent", (False, True): "behavior", (True, True): "both"}[(hc != jc, hb != jb)]


def confusion(h: pd.Series, j: pd.Series) -> pd.DataFrame:
    cm = pd.crosstab(pd.Categorical(h, LABELS), pd.Categorical(j, LABELS), dropna=False)
    cm.index, cm.columns = list(LABELS), list(LABELS)
    return cm


def cohen_kappa(cm: np.ndarray) -> float:
    n = cm.sum()
    if n == 0:
        return math.nan
    po, pe = np.trace(cm) / n, float(cm.sum(1) @ cm.sum(0)) / n ** 2
    return float((po - pe) / (1 - pe)) if pe < 1 else math.nan


def axis_agreement(h: pd.Series, j: pd.Series, axis: int) -> float:
    a, b = h.map(lambda l: LABEL_AXES[l][axis]), j.map(lambda l: LABEL_AXES[l][axis])
    m = a.notna() & b.notna()
    return float(a[m].eq(b[m]).mean()) if m.any() else math.nan


def agreement(g: pd.DataFrame) -> dict:
    h, j = g["human_label"], g["judge_label"]
    agree = h.eq(j)
    both = h.ne("AMBIGUOUS") & j.ne("AMBIGUOUS")
    lo, hi = wilson(int(agree.sum()), len(g))
    consistency = lambda lab: float(lab.map(_IMPLIED).eq(g["benchmark_class"])[lab.ne("AMBIGUOUS")].mean())
    return {
        "n": len(g),
        "exact_agreement": float(agree.mean()),
        "exact_agreement_lo": lo,
        "exact_agreement_hi": hi,
        "cohen_kappa": cohen_kappa(confusion(h, j).to_numpy()),
        "agreement_excl_ambiguous": float(agree[both].mean()) if both.any() else math.nan,
        "n_excl_ambiguous": int(both.sum()),
        "behavior_axis_agreement": axis_agreement(h, j, 1),
        "intent_axis_agreement": axis_agreement(h, j, 0),
        "judge_ambiguous_rate": float(j.eq("AMBIGUOUS").mean()),
        "human_ambiguous_rate": float(h.eq("AMBIGUOUS").mean()),
        # P(label's implied prompt class == XSTest class | label != AMBIGUOUS)
        "judge_xstest_intent_consistency": consistency(j),
        "human_xstest_intent_consistency": consistency(h),
        "judge_conf_when_agree": float(g.loc[agree, "judge_confidence"].mean()),
        "judge_conf_when_disagree": float(g.loc[~agree, "judge_confidence"].mean()),
        **{f"n_err_{k}": int(g["error_type"].eq(k).sum()) for k in ("behavior", "intent", "both", "ambiguous")},
    }


def scoped(df: pd.DataFrame):
    yield "ALL", df
    for p, g in df.groupby("policy", observed=True):
        yield f"policy={p}", g
    for c, g in df.groupby("benchmark_class"):
        yield f"class={c}", g


def confusion_table(df: pd.DataFrame) -> pd.DataFrame:
    blocks = {}
    for scope, g in scoped(df):
        cm = confusion(g["human_label"], g["judge_label"])
        cm["total"] = cm.sum(axis=1)
        cm.loc["total"] = cm.sum(axis=0)
        blocks[scope] = cm
    return pd.concat(blocks, names=["scope", "human_label"])


def policy_rates_human_vs_judge(df: pd.DataFrame) -> pd.DataFrame:
    """Headline rates on the audit subset under each labeller; rank shifts show whether judge error reorders policies."""
    rows = []
    for p, g in df.groupby("policy", observed=True):
        h, j = class_rates(g, "human_label", ci=False), class_rates(g, "judge_label", ci=False)
        rows += [{"policy": p, "metric": m, "n": int(g["benchmark_class"].eq(HEADLINE[m][0]).sum()),
                  "human": h[m], "judge": j[m], "judge_minus_human": j[m] - h[m]} for m in HEADLINE]
    out = pd.DataFrame(rows)
    out["policy"] = pd.Categorical(out["policy"], df["policy"].cat.categories, ordered=True)
    for src in ("human", "judge"):
        out[f"rank_{src}"] = out.groupby("metric")[src].rank(ascending=False, method="min")
    return out


def case_table(df: pd.DataFrame) -> pd.DataFrame:
    """Qualitative-evidence candidates: judge error (vs human) and/or policy difference (human labels split across
    policies on the same prompt)."""
    split = df.groupby("xstest_id")["human_label"].transform("nunique").gt(1)
    err = df["human_label"].ne(df["judge_label"])
    attribution = np.select([err & split, err, split], ["both", "judge_error", "policy_difference"], default="")
    cols = ["xstest_id", "policy", "benchmark_class", "type", "attribution", "error_type", "human_label", "judge_label",
            "judge_confidence", "judge_rationale_tag", "response_tokens", "prompt", "response", "human_notes"]
    return df.assign(attribution=attribution).loc[attribution != "", cols]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--worksheet", default=None, help="completed worksheet (default: results/task4_safety/audit/)")
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    ad = audit_dir(cfg)

    df = load_audit(cfg, args.worksheet, args.allow_partial)
    agree_rows = {s: agreement(g) for s, g in scoped(df)}
    agree = pd.DataFrame(agree_rows).T.rename_axis("scope")
    cm = confusion_table(df)
    rates = policy_rates_human_vs_judge(df)
    cases = case_table(df)
    preserved = {m: bool(g["rank_human"].tolist() == g["rank_judge"].tolist()) for m, g in rates.groupby("metric")}

    df.to_csv(ad / "audit_joined.csv", index=False)
    agree.to_csv(ad / "audit_agreement.csv")
    cm.to_csv(ad / "audit_confusion.csv")
    rates.to_csv(ad / "audit_policy_rates_human_vs_judge.csv", index=False)
    cases.to_csv(ad / "audit_cases.csv", index=False, encoding="utf-8-sig")
    save_json(ad / "audit_agreement.json", {
        "n_rows": len(df),
        "n_prompts": int(df["xstest_id"].nunique()),
        "agreement": agree_rows,
        "confusion_all": cm.loc["ALL"].to_dict(orient="index"),
        "policy_ranking_preserved": preserved,
        "attribution_counts": cases["attribution"].value_counts().to_dict(),
    })
    head = ["n", "exact_agreement", "cohen_kappa", "agreement_excl_ambiguous", "behavior_axis_agreement",
            "intent_axis_agreement", "judge_ambiguous_rate", "human_ambiguous_rate"]
    (ad / "audit_tables.md").write_text("\n\n".join([
        "## Judge vs manual agreement", to_markdown(agree[head].astype(float).astype({"n": int})),
        "## Confusion (rows = human, cols = judge), all policies", to_markdown(cm.loc["ALL"]),
        "## Headline rates on the audit subset: human vs judge", to_markdown(rates),
    ]) + "\n", encoding="utf-8")

    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.precision", 3):
        print(agree[head].astype(float))
        print(cm.loc["ALL"])
        print(rates.pivot(index="policy", columns="metric", values="judge_minus_human"))
    print("policy ranking preserved under judge labels:", preserved)
    print("saved ->", ad)


if __name__ == "__main__":
    main()
