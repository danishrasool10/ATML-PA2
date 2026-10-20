"""Task 5, Step 2: reward-sensitivity / robustness study on the 100-response controlled diagnostic set.

All C(5,2) = 10 variant pairs of each of the 20 problems are scored by the exact verifier and the fixed pairwise judge.
Per-pair records -> outputs/task5_feedback/diagnostic_{pairs,responses}.jsonl; metrics -> results/task5_feedback/.
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict

import pandas as pd

from common.data import load_yaml, read_jsonl, write_jsonl
from common.logging_utils import set_seed
from task5_feedback.analysis import VARIANTS, aggregate_diagnostics, score_diagnostic_groups
from task5_feedback.artifacts import (diagnostic_pairs_path, diagnostic_responses_path, judge_cache_path,
                                      judge_manifest_path, results_dir, save_json, to_markdown)
from task5_feedback.judging import judge_pair, judge_spec, open_judge, release_judge

EXPECTED_VARIANTS = {
    "clean_correct",
    "corrupt_reasoning_correct_final",
    "good_reasoning_wrong_final",
    "persuasive_filler_correct",
    "gold_distractor_wrong_final",
}
assert EXPECTED_VARIANTS == set(VARIANTS)


def load_diagnostic_groups(path):
    rows = read_jsonl(path)
    by_problem = defaultdict(dict)
    for row in rows:
        pid, vt = str(row["problem_id"]), row["variant_type"]
        if vt in by_problem[pid]:
            raise ValueError(f"Problem {pid} has duplicate variant {vt}")
        by_problem[pid][vt] = row
    for pid, variants in by_problem.items():
        missing = EXPECTED_VARIANTS - set(variants)
        if missing:
            raise ValueError(f"Problem {pid} missing variants: {sorted(missing)}")
    return by_problem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--aggregate-only", action="store_true", help="recompute metrics from saved pair records")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    set_seed(int(cfg["seed"]))

    if not args.aggregate_only:
        groups = load_diagnostic_groups(cfg["paths"]["task5_diagnostics"])
        print(f"Diagnostic problems: {len(groups)} | pairs/problem: {len(VARIANTS) * (len(VARIANTS) - 1) // 2}")
        judge = open_judge(cfg, judge_cache_path(cfg, "diagnostics"))
        spec, t0 = judge_spec(cfg, judge), time.perf_counter()
        try:
            pairs, responses = score_diagnostic_groups(groups, lambda q, a, b: judge_pair(judge, q, a, b))
        finally:
            release_judge(judge)
        write_jsonl(diagnostic_pairs_path(cfg), pairs)
        write_jsonl(diagnostic_responses_path(cfg), responses)
        save_json(judge_manifest_path(cfg, "diagnostics"), {
            **spec, "n_pairs": len(pairs), "n_cache_hits": sum(p["judge_cache_hit"] for p in pairs),
            "wall_s_this_run": time.perf_counter() - t0,
        })

    res = aggregate_diagnostics(cfg, n_boot=args.bootstrap)
    head = {g: {k: v for k, v in h.items() if not k.endswith("_dominance")} for g, h in res["headline"].items()}
    print(to_markdown(pd.DataFrame(head).rename_axis("metric")))
    print("judge bias:", {k: round(v, 3) if isinstance(v, float) else v for k, v in res["judge_bias"].items()})
    print("saved ->", results_dir(cfg))


if __name__ == "__main__":
    main()
