"""Orchestration around the fixed PairwiseAIJudge. Rubric, decoding, parsing and the orientation rule are untouched.

Calls stay unbatched on purpose: left-padding a batch would perturb the 4-bit judge's logits and break the
determinism (and cache validity) that the RLAIF reward relied on.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from common.models import clear_gpu
from task5_feedback.rlaif import PairwiseAIJudge


def open_judge(cfg: dict, cache_path: Path) -> PairwiseAIJudge:
    cache_path = Path(cache_path)
    if cache_path.exists():
        try:
            json.loads(cache_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # the judge rewrites its full cache on every call; a session killed mid-write leaves a torn file
            bad = cache_path.with_name(f"{cache_path.stem}.corrupt-{int(time.time())}.json")
            cache_path.rename(bad)
            print(f"[judge] unreadable cache moved to {bad.name}; affected pairs will be re-judged", flush=True)
    return PairwiseAIJudge(cfg, cache_path)


def judge_pair(judge: PairwiseAIJudge, problem: str, a: str, b: str) -> tuple[str, bool | None, bool]:
    """-> (raw in {A, B, TIE} w.r.t. the argument order, a_shown_first, cache_hit).

    Byte-identical candidates are a TIE by construction: the fixed judge would otherwise settle a self-comparison
    by position bias alone."""
    if a == b:
        return "TIE", None, False
    key = judge._key(problem, a, b)
    hit = key in judge.cache
    # mirrors PairwiseAIJudge.compare: odd leading hash word -> (b, a) presented in slots (A, B)
    return judge.compare(problem, a, b), int(key[:8], 16) % 2 == 0, hit


def judge_spec(cfg: dict, judge: PairwiseAIJudge) -> dict:
    q = getattr(judge.model, "is_loaded_in_4bit", False)
    return {"judge_model": cfg["ai_judge_model"], "quantized_4bit": bool(q), "max_new_tokens": 4, "do_sample": False,
            "cache": judge.cache_path.name}


def release_judge(judge: PairwiseAIJudge | None) -> None:
    if judge is not None:
        judge.model = None
    clear_gpu()
