"""Hand-checked cases for Task 5 aggregation (torch-free).

Run: python -m pytest tests/test_task5_analysis.py
"""
from __future__ import annotations

import math

import pandas as pd

from task5_feedback.analysis import (VARIANTS, reference_preference, score_diagnostic_groups, summarize_diagnostics,
                                     summarize_math)
from task5_feedback.artifacts import pair_digest
from task5_feedback.rlvr import exact_reward

SFT_OK = [1, 1, 0, 0, 1, 0, 1, 0, 0, 0]
POL_OK = [1, 1, 1, 0, 1, 1, 1, 0, 0, 0]
# judge preference per prompt; prompt 0 has byte-identical responses
PREF = ["tie", "policy", "policy", "sft", "tie", "sft", "policy", "policy", "sft", "tie"]


def _math_frames():
    gen, judged = [], []
    for i, (s, p, pref) in enumerate(zip(SFT_OK, POL_OK, PREF)):
        sft_text = f"sft {i} #### {s}"
        pol_text = sft_text if i == 0 else f"policy answer {i} " * (1 + i % 3) + f"#### {p}"
        for name, text, ok, n in (("sft", sft_text, s, 10), ("rlvr", pol_text, p, 10 + i)):
            gen.append({"example_id": str(i), "policy": name, "response": text, "correct": ok, "response_tokens": n,
                        "format_ok": i != 9 or name == "sft", "truncated": False, "terminated_with_eos": True})
        judged.append({"example_id": str(i), "policy": "rlvr", "judge_pref": pref, "identical": i == 0,
                       "policy_shown_first": i % 2 == 0, "pair_digest": pair_digest(pol_text, sft_text)})
    return pd.DataFrame(gen), pd.DataFrame(judged)


def test_math_summary_hand_values():
    gen, judged = _math_frames()
    s = summarize_math(gen, judged, n_boot=200, seed=0)["summary"]
    r = s.loc["rlvr"]
    assert math.isclose(s.loc["sft", "accuracy"], 0.4) and math.isclose(r["accuracy"], 0.6)
    assert (r["n_gained"], r["n_lost"]) == (2, 0) and math.isclose(r["mcnemar_p"], 0.5)
    assert math.isclose(r["win_rate_vs_sft"], 0.55)                    # (0.5+1+1+0+0.5+0+1+1+0+0.5)/10
    assert math.isclose(r["verifier_judge_agreement"], 3 / 9)          # differing pairs only
    assert math.isclose(r["verifier_judge_agreement_all"], 4 / 10)
    assert math.isclose(r["verifier_judge_decisive_agreement"], 0.5)   # prompt 2 agrees, prompt 5 reverses
    assert math.isclose(r["format_compliance"], 0.9) and math.isclose(r["identical_rate"], 0.1)
    # decisive judgments: prompts 1,2,6,7 -> policy (always longer), 3,5,8 -> SFT
    assert math.isclose(r["judge_prefers_longer_rate"], 4 / 7)
    assert math.isclose(r["judge_slot_A_rate"], 4 / 7)                 # policy shown first on even prompts
    assert r["win_rate_lo"] <= r["win_rate_vs_sft"] <= r["win_rate_hi"]


def test_stale_judgments_are_rejected():
    gen, judged = _math_frames()
    judged.loc[3, "pair_digest"] = "0" * 16
    try:
        summarize_math(gen, judged)
    except RuntimeError as e:
        assert "predate" in str(e)
    else:
        raise AssertionError("stale judgment accepted")


def _variant(name, outcome, rq, style, text):
    return {"variant_type": name, "expected_exact_reward": outcome, "reasoning_quality": rq, "style": style,
            "response": text, "question": "q", "gold_final": 9}


GROUP = {
    "clean_correct": _variant("clean_correct", 1, "correct", "clean", "2+7 = 9\n#### 9"),
    "persuasive_filler_correct": _variant("persuasive_filler_correct", 1, "correct", "persuasive_filler",
                                          "2+7 = 9\nUnquestionably correct.\n#### 9"),
    "corrupt_reasoning_correct_final": _variant("corrupt_reasoning_correct_final", 1, "corrupted", "clean",
                                                "2+7 = 8\n#### 9"),
    "good_reasoning_wrong_final": _variant("good_reasoning_wrong_final", 0, "mostly_correct", "clean",
                                           "2+7 = 9\n#### 10"),
    "gold_distractor_wrong_final": _variant("gold_distractor_wrong_final", 0, "incorrect", "distractor",
                                            "9 is tempting\n#### 10"),
}


def test_reference_order_is_total_and_matches_variant_order():
    rels = {}
    for i, a in enumerate(VARIANTS):
        for b in VARIANTS[i + 1:]:
            sign, rel = reference_preference(GROUP[a], GROUP[b])
            assert sign == 1, (a, b)
            rels[(a, b)] = rel
    assert rels[("clean_correct", "persuasive_filler_correct")] == "style_only"
    assert rels[("corrupt_reasoning_correct_final", "good_reasoning_wrong_final")] == "contested"
    assert sum(r == "dominance" for r in rels.values()) == 8


def _slot_a_judge(q, a, b):
    # pure position bias: always the physical first slot; `a` shown first iff len(a) is even
    a_first = len(a) % 2 == 0
    return ("A" if a_first else "B"), a_first, False


def test_diagnostic_signs_and_position_bias():
    assert all(exact_reward(GROUP[v]["response"], "9") == GROUP[v]["expected_exact_reward"] for v in VARIANTS)
    pairs, resp = score_diagnostic_groups({"1": GROUP, "2": GROUP}, _slot_a_judge)
    res = summarize_diagnostics(pd.DataFrame(pairs), pd.DataFrame(resp), n_boot=100)
    v, j = res["headline"]["verifier"], res["headline"]["judge"]
    assert (v["S_outcome"], v["S_reason"], v["S_reason_tie"], v["filler_tie"]) == (1.0, 0.0, 1.0, 1.0)
    assert math.isclose(v["better_preference_rate"], 6 / 10) and v["wrong_preference_rate"] == 0.0
    assert res["judge_bias"]["slot_A_rate"] == 1.0 and j["tie_rate"] == 0.0
    borda = pd.DataFrame(resp).groupby("variant")["judge_borda"].mean()
    assert math.isclose(borda.sum(), len(VARIANTS) / 2)                 # each pair hands out exactly one point


if __name__ == "__main__":
    for f in (test_math_summary_hand_values, test_stale_judgments_are_rejected,
              test_reference_order_is_total_and_matches_variant_order, test_diagnostic_signs_and_position_bias):
        f()
    print("ok")
