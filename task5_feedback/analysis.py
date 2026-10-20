"""Torch-free Task 5 aggregation.

Math eval (GSM8K in-domain, SVAMP transfer): per-policy exact-verifier accuracy, format compliance and length
dispersion, the fixed AI judge's pairwise preference against SFT, and verifier-judge concordance on those same pairs.

Controlled diagnostics: every unordered variant pair of each problem is scored by the verifier (scalar exact reward,
compared) and the judge (pairwise), and both are signed against a reference order derived from the released variant
metadata, so "better / tie / wrong preference" is defined per pair rather than per response.
"""
from __future__ import annotations

import json
import math
import re
from itertools import combinations

import numpy as np
import pandas as pd

from common.data import read_jsonl
from task5_feedback.artifacts import (DATASET_LABEL, POLICIES, REFERENCE, cohen_kappa, diagnostic_pairs_path,
                                      diagnostic_responses_path, dispersion, generation_manifest_path,
                                      generations_path, judge_manifest_path, judged_path, load_json, mcnemar_exact,
                                      pair_digest, ratio_bootstrap, results_dir, save_json, summary_path, wilson)
from task5_feedback.rlvr import exact_reward

# ======================================================================================= math evaluation
# Judge preference on the policy-vs-SFT axis. WIN_SCORE is the task's 1 / 0.5 / 0 convention.
PREF_SIGN = {"policy": 1, "tie": 0, REFERENCE: -1}
WIN_SCORE = {"policy": 1.0, "tie": 0.5, REFERENCE: 0.0}
COUNT_COLUMNS = ("n", "n_gained", "n_lost", "wins", "ties", "losses", "n_verifier_decisive", "n_judge_decisive")
VERIFIER_OUTCOME = {(1, 0): "policy_only_correct", (0, 1): "sft_only_correct",
                    (1, 1): "both_correct", (0, 0): "both_wrong"}

MATH_DEFINITIONS = {
    "accuracy": "P(exact_reward = 1): last designated '#### <number>' numerically equals gold_final",
    "accuracy_lo/hi": "Wilson 95% interval",
    "format_compliance": "P(a designated '#### <number>' final is extractable), regardless of correctness",
    "accuracy_given_format": "P(correct | format_ok)",
    "len_*": "response length in policy-tokenizer tokens excluding the terminating EOS (cap = math_max_new_tokens); "
             "std is the sample std, iqr = q75 - q25",
    "truncation_rate": "P(reached math_max_new_tokens without EOS)",
    "delta_accuracy_vs_sft / mcnemar_p": "paired accuracy difference vs SFT; exact two-sided McNemar on per-prompt flips",
    "win_rate_vs_sft": "mean over prompts of 1 (judge prefers policy) / 0.5 (TIE) / 0 (prefers SFT); byte-identical "
                       "policy/SFT responses are a TIE by construction and are never sent to the judge",
    "win_rate_lo/hi": "percentile bootstrap over prompts (95%)",
    "win_rate_vs_sft_differing": "win rate over prompts whose policy and SFT responses differ",
    "verifier_judge_agreement": "P(sign(correct_policy - correct_sft) == judge preference sign; TIE = 0), differing pairs",
    "verifier_judge_agreement_all": "same, including identical pairs (trivially concordant)",
    "verifier_judge_kappa": "Cohen's kappa between the two 3-way preference signs, differing pairs",
    "verifier_judge_decisive_agreement": "P(judge prefers the verifier-correct response | exactly one is correct)",
    "judge_reverses_verifier_rate": "P(judge prefers the verifier-incorrect response | exactly one is correct)",
    "judge_slot_A_rate": "P(judge picks the physically first candidate | decisive); 0.5 = no position bias",
    "judge_prefers_longer_rate": "P(judge picks the longer response | decisive, lengths differ)",
}


def check_paired(gen: pd.DataFrame) -> None:
    dup = gen.duplicated(["policy", "example_id"])
    if dup.any():
        raise ValueError(f"duplicated (policy, example_id): {gen.loc[dup, ['policy', 'example_id']].values[:3].tolist()}")
    ids = gen.groupby("policy")["example_id"].agg(frozenset)
    for p, s in ids.items():
        if s != ids[REFERENCE]:
            raise ValueError(f"{p}: example set differs from {REFERENCE} ({len(s ^ ids[REFERENCE])} ids)")


def paired_frame(gen: pd.DataFrame, judged: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (non-reference policy, example): policy vs SFT correctness, length, and judge outcome."""
    keep = ["example_id", "response", "correct", "response_tokens"]
    ref = gen.loc[gen["policy"] == REFERENCE, keep].rename(columns={c: f"sft_{c}" for c in keep[1:]})
    m = gen.loc[gen["policy"] != REFERENCE, ["policy", *keep]].merge(ref, on="example_id", validate="many_to_one")
    m["verifier_sign"] = m["correct"] - m["sft_correct"]
    if judged is None or judged.empty:
        return m
    j = judged[["policy", "example_id", "judge_pref", "identical", "policy_shown_first", "pair_digest"]]
    m = m.merge(j, on=["policy", "example_id"], how="left", validate="one_to_one")
    if m["judge_pref"].isna().any():
        raise ValueError(f"{int(m['judge_pref'].isna().sum())} policy/SFT pairs lack a judgment; rerun --stage judge")
    stale = m["pair_digest"] != [pair_digest(a, b) for a, b in zip(m["response"], m["sft_response"])]
    if stale.any():
        raise RuntimeError(f"{int(stale.sum())} judgments predate the current generations; rerun --stage judge")
    m["judge_sign"] = m["judge_pref"].map(PREF_SIGN)
    m["win_score"] = m["judge_pref"].map(WIN_SCORE)
    return m


def _policy_metrics(g: pd.DataFrame) -> dict:
    n, k = len(g), int(g["correct"].sum())
    fmt = g["format_ok"].astype(bool)
    tok = g["response_tokens"].astype(float)
    lo, hi = wilson(k, n)
    return {
        "n": n, "accuracy": k / n, "accuracy_lo": lo, "accuracy_hi": hi,
        "format_compliance": fmt.mean(),
        "accuracy_given_format": g.loc[fmt, "correct"].mean() if fmt.any() else math.nan,
        "truncation_rate": g["truncated"].astype(float).mean(),
        **{f"len_{s}": v for s, v in dispersion(tok).items()},
        "len_mean_correct": tok[g["correct"] == 1].mean(),
        "len_mean_incorrect": tok[g["correct"] == 0].mean(),
    }


def _pair_metrics(m: pd.DataFrame, n_boot: int, seed: int) -> dict:
    gained = int(((m["correct"] == 1) & (m["sft_correct"] == 0)).sum())
    lost = int(((m["correct"] == 0) & (m["sft_correct"] == 1)).sum())
    out = {"delta_accuracy_vs_sft": m["correct"].mean() - m["sft_correct"].mean(),
           "n_gained": gained, "n_lost": lost, "mcnemar_p": mcnemar_exact(gained, lost)}
    if "judge_sign" not in m:
        return out

    w = m["win_score"].to_numpy(float)
    lo, hi = ratio_bootstrap(w, np.ones_like(w), n_boot, seed)
    ident = m["identical"].astype(bool)
    d = m[~ident]                                  # identical pairs carry no information about either signal
    dec = d[d["verifier_sign"] != 0]
    jd = d[d["judge_sign"] != 0]
    first = jd["policy_shown_first"].astype(bool)
    picked_a = np.where(jd["judge_sign"] > 0, first, ~first)
    dl = jd[jd["response_tokens"] != jd["sft_response_tokens"]]
    longer = np.sign(dl["response_tokens"] - dl["sft_response_tokens"]) == dl["judge_sign"]
    out.update({
        "win_rate_vs_sft": w.mean(), "win_rate_lo": lo, "win_rate_hi": hi,
        "wins": int((m["judge_sign"] > 0).sum()), "ties": int((m["judge_sign"] == 0).sum()),
        "losses": int((m["judge_sign"] < 0).sum()),
        "identical_rate": ident.mean(),
        "win_rate_vs_sft_differing": d["win_score"].mean(),
        "verifier_judge_agreement": (d["verifier_sign"] == d["judge_sign"]).mean(),
        "verifier_judge_agreement_all": (m["verifier_sign"] == m["judge_sign"]).mean(),
        "verifier_judge_kappa": cohen_kappa(d["verifier_sign"], d["judge_sign"]),
        "n_verifier_decisive": len(dec),
        "verifier_judge_decisive_agreement": (dec["judge_sign"] == dec["verifier_sign"]).mean(),
        "judge_reverses_verifier_rate": (dec["judge_sign"] == -dec["verifier_sign"]).mean(),
        "n_judge_decisive": len(jd),
        "judge_slot_A_rate": float(picked_a.mean()) if len(jd) else math.nan,
        "judge_prefers_longer_rate": float(longer.mean()) if len(dl) else math.nan,
    })
    return out


def verifier_judge_contingency(m: pd.DataFrame) -> pd.DataFrame:
    """Judge preference counts conditioned on the verifier outcome of the (policy, SFT) pair; differing pairs only."""
    if "judge_pref" not in m:
        return pd.DataFrame()
    d = m[~m["identical"].astype(bool)].copy()
    d["verifier_outcome"] = [VERIFIER_OUTCOME[(int(a), int(b))] for a, b in zip(d["correct"], d["sft_correct"])]
    idx = pd.MultiIndex.from_product([[p for p in POLICIES if p != REFERENCE and p in set(m["policy"])],
                                      list(VERIFIER_OUTCOME.values())], names=["policy", "verifier_outcome"])
    t = (d.groupby(["policy", "verifier_outcome"])["judge_pref"].value_counts().unstack(fill_value=0)
          .reindex(index=idx, columns=["policy", "tie", REFERENCE], fill_value=0))
    t.columns = ["judge_prefers_policy", "judge_tie", "judge_prefers_sft"]
    t["n"] = t.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        t["judge_win_rate"] = (t["judge_prefers_policy"] + 0.5 * t["judge_tie"]) / t["n"]
    return t.reset_index()


def summarize_math(gen: pd.DataFrame, judged: pd.DataFrame | None = None, n_boot: int = 2000, seed: int = 0) -> dict:
    check_paired(gen)
    m = paired_frame(gen, judged)
    rows = {}
    for p in POLICIES:
        g = gen[gen["policy"] == p]
        if len(g):
            rows[p] = {**_policy_metrics(g),
                       **(_pair_metrics(m[m["policy"] == p], n_boot, seed) if p != REFERENCE else {})}
    summary = pd.DataFrame.from_dict(rows, orient="index")
    counts = [c for c in COUNT_COLUMNS if c in summary]
    summary[counts] = summary[counts].astype("Int64")
    summary.index.name = "policy"
    return {"summary": summary, "contingency": verifier_judge_contingency(m), "pairs": m}


def load_generations(cfg, dataset: str) -> tuple[pd.DataFrame, dict]:
    man_path = generation_manifest_path(cfg, dataset)
    if not man_path.exists():
        raise FileNotFoundError(f"{man_path} missing; run task5_feedback.evaluate_math --dataset {dataset}")
    man = load_json(man_path)
    missing = [p for p in POLICIES if p not in man["policies"] or not generations_path(cfg, dataset, p).exists()]
    if missing:
        raise FileNotFoundError(f"{dataset}: no generations for {missing}; run evaluate_math --stage generate")
    if len({json.dumps(man["policies"][p]["generation_config"], sort_keys=True) for p in POLICIES}) > 1:
        raise RuntimeError(f"{dataset}: effective generation_config differs across policies; decoding is not common")
    gen = pd.DataFrame([r for p in POLICIES for r in read_jsonl(generations_path(cfg, dataset, p))])
    check_paired(gen)
    counts = gen.groupby("policy").size()
    if (counts != man["common"]["n_examples"]).any():
        raise RuntimeError(f"{dataset}: expected {man['common']['n_examples']} rows/policy, got {counts.to_dict()}")
    return gen, man


def aggregate_math(cfg, dataset: str, n_boot: int = 2000) -> dict:
    gen, man = load_generations(cfg, dataset)
    jp = judged_path(cfg, dataset)
    judged = pd.DataFrame(read_jsonl(jp)) if jp.exists() else None
    res = summarize_math(gen, judged, n_boot=n_boot, seed=int(cfg["seed"]))
    rd, jm = results_dir(cfg), judge_manifest_path(cfg, dataset)
    res["summary"].to_csv(rd / f"{dataset}_summary.csv")
    res["contingency"].to_csv(rd / f"{dataset}_verifier_judge_contingency.csv", index=False)
    save_json(summary_path(cfg, dataset), {
        "dataset": DATASET_LABEL[dataset],
        "n_examples": man["common"]["n_examples"],
        "decoding": man["common"]["decoding"],
        "judge": load_json(jm) if jm.exists() else None,
        "definitions": MATH_DEFINITIONS,
        "summary": res["summary"].to_dict(orient="index"),
        "contingency": res["contingency"].to_dict(orient="records"),
    })
    return res


# ======================================================================================= controlled diagnostics
VARIANTS = ("clean_correct", "persuasive_filler_correct", "corrupt_reasoning_correct_final",
            "good_reasoning_wrong_final", "gold_distractor_wrong_final")
SHORT = dict(zip(VARIANTS, ("clean", "filler", "corrupt", "wrong_final", "distractor")))
REASONING_RANK = {"correct": 3, "mostly_correct": 2, "corrupted": 1, "incorrect": 0}
JUDGE_SIGN = {"A": 1, "TIE": 0, "B": -1}
GRADERS = ("verifier", "judge")
_WORD = re.compile(r"\b\w+\b")

DIAG_DEFINITIONS = {
    "reference order": "lexicographic on released metadata: expected_exact_reward (designated-final correctness), then "
                       "reasoning_quality (correct > mostly_correct > corrupted > incorrect), then style (clean > "
                       "persuasive filler: the judge rubric asks for relevance and no credit for confidence)",
    "relation": "dominance = no axis disagrees; style_only = clean vs filler (outcome and reasoning tie); contested = "
                "corrupt-reasoning/correct-final vs coherent/wrong-final (outcome and reasoning disagree; outcome wins)",
    "R (verifier)": "exact_reward in {0,1}; a pair is a TIE when both rewards are equal",
    "R (judge)": "fixed pairwise judge on (question, a, b); R(a) > R(b) iff the judge prefers a",
    "better / tie / wrong": "P(grader prefers the reference-better response) / P(TIE) / P(prefers the reference-worse)",
    "S_reason": "Pr[R(clean) > R(corrupt-reasoning)], both with the identical correct designated final",
    "S_reason_incl_filler": "S_reason over {clean, filler} vs corrupt-reasoning",
    "S_outcome": "Pr[R(correct-final) > R(wrong-final)] over every pair whose designated finals differ in correctness "
                 "({clean, filler, corrupt} x {wrong_final, distractor})",
    "S_outcome_minimal": "S_outcome on clean vs coherent-reasoning/wrong-final (only the designated final changes)",
    "filler_susceptibility": "Pr[R(filler) > R(clean)] (wrong preference on clean|filler)",
    "distractor_robustness": "P(better preferred) over pairs containing the gold-distractor response",
    "judge_borda": "per-response RLAIF group reward over the 5 variants: (wins + 0.5 ties) / (K - 1), K = 5",
    "CIs": "pooled rates: percentile bootstrap resampling problems (pairs of a problem are dependent), 95%",
}


def word_count(text: str) -> int:
    return len(_WORD.findall(str(text)))


def reference_preference(a: dict, b: dict) -> tuple[int, str]:
    """+1 if diagnostic row `a` is the reference-better response, -1 if `b`; see DIAG_DEFINITIONS for the order."""
    do = int(a["expected_exact_reward"]) - int(b["expected_exact_reward"])
    dr = int(np.sign(REASONING_RANK[a["reasoning_quality"]] - REASONING_RANK[b["reasoning_quality"]]))
    if do and dr and do != dr:
        return do, "contested"
    if do or dr:
        return do or dr, "dominance"
    ds = int(a["style"] == "clean") - int(b["style"] == "clean")
    if not ds:
        raise ValueError(f"no reference order between {a['variant_type']} and {b['variant_type']}")
    return ds, "style_only"


def _natural(pid: str):
    return (0, int(pid), "") if str(pid).lstrip("-").isdigit() else (1, 0, str(pid))


def score_diagnostic_groups(groups: dict, judge_fn) -> tuple[list[dict], list[dict]]:
    """groups: {problem_id: {variant_type: row}}. judge_fn(question, a, b) -> (raw in {A,B,TIE}, a_shown_first, hit).

    Each unordered pair is judged once with `a` = the earlier variant in VARIANTS; the fixed judge's hash-parity swap
    decides the physical slot, recorded as a_shown_first for the position-bias check."""
    pairs, responses = [], []
    k = len(VARIANTS)
    for pid in sorted(groups, key=_natural):
        v = groups[pid]
        questions, golds = {r["question"] for r in v.values()}, {str(r["gold_final"]) for r in v.values()}
        if len(questions) != 1 or len(golds) != 1:
            raise ValueError(f"problem {pid}: variants disagree on question/gold_final")
        q, gold = questions.pop(), golds.pop()
        reward = {name: exact_reward(v[name]["response"], gold) for name in VARIANTS}
        borda = dict.fromkeys(VARIANTS, 0.0)
        for a, b in combinations(VARIANTS, 2):
            ra, rb = v[a], v[b]
            ref, relation = reference_preference(ra, rb)
            raw, a_first, hit = judge_fn(q, ra["response"], rb["response"])
            js, vs = JUDGE_SIGN[raw], int(np.sign(reward[a] - reward[b]))
            borda[a] += (1 + js) / 2
            borda[b] += (1 - js) / 2
            pairs.append({
                "problem_id": pid, "pair_type": f"{SHORT[a]}|{SHORT[b]}", "a_variant": a, "b_variant": b,
                "relation": relation, "better_variant": a if ref > 0 else b, "reference_sign": ref,
                "a_outcome": int(ra["expected_exact_reward"]), "b_outcome": int(rb["expected_exact_reward"]),
                "a_words": word_count(ra["response"]), "b_words": word_count(rb["response"]),
                "verifier_a": reward[a], "verifier_b": reward[b], "verifier_sign": vs,
                "judge_raw": raw, "judge_sign": js, "a_shown_first": a_first, "judge_cache_hit": hit,
                "verifier_vs_ref": vs * ref, "judge_vs_ref": js * ref,
            })
        for name in VARIANTS:
            r = v[name]
            responses.append({
                "problem_id": pid, "variant": name, "reasoning_quality": r["reasoning_quality"], "style": r["style"],
                "expected_exact_reward": int(r["expected_exact_reward"]), "verifier_reward": reward[name],
                "judge_borda": borda[name] / (k - 1), "words": word_count(r["response"]),
            })
    return pairs, responses


def probe_masks(p: pd.DataFrame) -> dict[str, pd.Series]:
    t = p["pair_type"]
    return {
        "S_reason": t.eq("clean|corrupt"),
        "S_reason_incl_filler": t.isin(["clean|corrupt", "filler|corrupt"]),
        "S_outcome": p["a_outcome"] != p["b_outcome"],
        "S_outcome_minimal": t.eq("clean|wrong_final"),
        "filler": t.eq("clean|filler"),
        "distractor": p["a_variant"].eq(VARIANTS[-1]) | p["b_variant"].eq(VARIANTS[-1]),
    }


def _rates(p: pd.DataFrame, col: str, n_boot: int, seed: int) -> dict:
    keys = ("better", "tie", "wrong")
    if p.empty:
        return {"n": 0, **{f"{k}{s}": math.nan for k in keys for s in ("", "_lo", "_hi")}}
    num = pd.crosstab(p["problem_id"], p[col]).reindex(columns=[1, 0, -1], fill_value=0).to_numpy(float)
    den = np.repeat(num.sum(1, keepdims=True), 3, axis=1)
    lo, hi = ratio_bootstrap(num, den, n_boot, seed)
    r = num.sum(0) / den[:, 0].sum()
    out = {"n": int(den[:, 0].sum())}
    for i, k in enumerate(keys):
        out.update({k: r[i], f"{k}_lo": lo[i], f"{k}_hi": hi[i]})
    return out


def summarize_diagnostics(pairs: pd.DataFrame, responses: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict:
    scopes = {"all_pairs": pd.Series(True, index=pairs.index),
              "dominance_pairs": pairs["relation"].eq("dominance"), **probe_masks(pairs)}
    pooled = pd.DataFrame([{"scope": s, "grader": g, **_rates(pairs[m], f"{g}_vs_ref", n_boot, seed)}
                           for s, m in scopes.items() for g in GRADERS])
    by_type = pd.DataFrame([
        {"pair_type": t, "relation": grp["relation"].iat[0], "reference_better": SHORT[grp["better_variant"].iat[0]],
         "grader": g, **_rates(grp, f"{g}_vs_ref", n_boot, seed)}
        for t, grp in pairs.groupby("pair_type", sort=False) for g in GRADERS])

    resp = responses.copy()
    top = resp.groupby("problem_id")["judge_borda"].transform("max").eq(resp["judge_borda"])
    resp["judge_top1"] = top / top.groupby(resp["problem_id"]).transform("sum")   # shared credit on Borda ties
    variants = []
    for v in VARIANTS:
        g = resp[resp["variant"] == v]
        b = g["judge_borda"].to_numpy(float)
        lo, hi = ratio_bootstrap(b, np.ones_like(b), n_boot, seed)
        variants.append({
            "variant": v, "short": SHORT[v], "n": len(g), "reasoning_quality": g["reasoning_quality"].iat[0],
            "expected_exact_reward": g["expected_exact_reward"].mean(), "verifier_reward": g["verifier_reward"].mean(),
            "verifier_matches_expected": (g["verifier_reward"] == g["expected_exact_reward"]).mean(),
            "judge_borda_mean": b.mean(), "judge_borda_lo": lo, "judge_borda_hi": hi,
            "judge_borda_std": b.std(ddof=1) if len(b) > 1 else math.nan,
            "judge_top1_rate": g["judge_top1"].mean(), "words_mean": g["words"].mean(),
        })
    variants = pd.DataFrame(variants)

    jd = pairs[pairs["judge_sign"] != 0]
    first = jd["a_shown_first"].astype(bool)
    picked_a = np.where(jd["judge_sign"] > 0, first, ~first)
    shown = pairs[pairs["a_shown_first"].notna()]
    better_first = np.where(shown["reference_sign"] > 0, shown["a_shown_first"].astype(bool),
                            ~shown["a_shown_first"].astype(bool))
    dl = jd[jd["a_words"] != jd["b_words"]]
    judge_bias = {
        "n_pairs": len(pairs), "n_decisive": len(jd), "tie_rate": float((pairs["judge_sign"] == 0).mean()),
        "slot_A_rate": float(picked_a.mean()) if len(jd) else math.nan,
        "better_shown_first_rate": float(better_first.mean()) if len(shown) else math.nan,
        "prefers_longer_rate": float((np.sign(dl["a_words"] - dl["b_words"]) == dl["judge_sign"]).mean())
        if len(dl) else math.nan,
        "verifier_judge_agreement": float((pairs["verifier_sign"] == pairs["judge_sign"]).mean()),
    }

    def at(scope, grader, key):
        return pooled.loc[pooled["scope"].eq(scope) & pooled["grader"].eq(grader), key].iat[0]

    headline = {g: {
        "better_preference_rate": at("all_pairs", g, "better"), "tie_rate": at("all_pairs", g, "tie"),
        "wrong_preference_rate": at("all_pairs", g, "wrong"),
        "better_preference_rate_dominance": at("dominance_pairs", g, "better"),
        "wrong_preference_rate_dominance": at("dominance_pairs", g, "wrong"),
        "S_reason": at("S_reason", g, "better"), "S_reason_tie": at("S_reason", g, "tie"),
        "S_reason_incl_filler": at("S_reason_incl_filler", g, "better"),
        "S_outcome": at("S_outcome", g, "better"), "S_outcome_tie": at("S_outcome", g, "tie"),
        "S_outcome_minimal": at("S_outcome_minimal", g, "better"),
        "filler_susceptibility": at("filler", g, "wrong"), "filler_tie": at("filler", g, "tie"),
        "distractor_robustness": at("distractor", g, "better"), "distractor_fooled": at("distractor", g, "wrong"),
    } for g in GRADERS}
    return {"headline": headline, "pooled": pooled, "by_type": by_type, "variants": variants,
            "judge_bias": judge_bias}


def aggregate_diagnostics(cfg, n_boot: int = 2000) -> dict:
    pairs = pd.DataFrame(read_jsonl(diagnostic_pairs_path(cfg)))
    responses = pd.DataFrame(read_jsonl(diagnostic_responses_path(cfg)))
    res = summarize_diagnostics(pairs, responses, n_boot=n_boot, seed=int(cfg["seed"]))
    rd, jm = results_dir(cfg), judge_manifest_path(cfg, "diagnostics")
    res["pooled"].to_csv(rd / "diagnostic_pooled.csv", index=False)
    res["by_type"].to_csv(rd / "diagnostic_pair_types.csv", index=False)
    res["variants"].to_csv(rd / "diagnostic_variants.csv", index=False)
    mismatch = responses[responses["verifier_reward"] != responses["expected_exact_reward"]]
    save_json(summary_path(cfg, "diagnostic"), {
        "n_problems": int(responses["problem_id"].nunique()), "n_responses": len(responses), "n_pairs": len(pairs),
        "judge": load_json(jm) if jm.exists() else None,
        "definitions": DIAG_DEFINITIONS,
        "headline": res["headline"], "judge_bias": res["judge_bias"],
        "verifier_expected_mismatches": mismatch[["problem_id", "variant"]].to_dict(orient="records"),
        "pooled": res["pooled"].to_dict(orient="records"),
        "by_type": res["by_type"].to_dict(orient="records"),
        "variants": res["variants"].to_dict(orient="records"),
    })
    return res
