"""Small evaluation helpers for recording Task 1 DPO results."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable, Sequence


def _softplus_negative(value: float) -> float:
    return math.log1p(math.exp(-value)) if value >= 0 else -value + math.log1p(math.exp(value))


def _sigmoid_negative(value: float) -> float:
    if value >= 0:
        exp_neg = math.exp(-value)
        return exp_neg / (1.0 + exp_neg)
    exp_pos = math.exp(value)
    return 1.0 / (1.0 + exp_pos)


def dpo_loss_from_margins(margins: Sequence[float]) -> dict[str, float]:
    """Summarize held-out per-example implicit reward margins."""
    z = [float(value) for value in margins]
    if not z:
        raise ValueError("Cannot calculate held-out DPO loss for an empty margin list")
    losses = [_softplus_negative(value) for value in z]
    n = len(losses)
    mean = sum(losses) / n
    standard_error = (
        math.sqrt(sum((value - mean) ** 2 for value in losses) / (n - 1) / n)
        if n > 1 else 0.0
    )
    return {
        "heldout_dpo_loss": mean,
        "heldout_dpo_loss_se": standard_error,
        "mean_sigma_neg_z": sum(_sigmoid_negative(value) for value in z) / n,
    }


def rescore_jointly(
    score_one: Callable[[object, str], float],
    prompts: Sequence[object],
    ref_responses: Sequence[str],
    pol_responses: Sequence[str],
) -> tuple[list[float], list[float]]:
    """Score both conditions with the same single-example reward-model call."""
    if not (len(prompts) == len(ref_responses) == len(pol_responses)):
        raise ValueError(
            "prompts, reference responses, and policy responses must have equal lengths"
        )
    ref_scores, pol_scores = [], []
    for prompt, ref_response, policy_response in zip(prompts, ref_responses, pol_responses):
        ref_score = float(score_one(prompt, ref_response))
        ref_scores.append(ref_score)
        pol_scores.append(
            ref_score if policy_response == ref_response
            else float(score_one(prompt, policy_response))
        )
    return ref_scores, pol_scores


def dump_policy_generation_records(
    path: str | Path,
    *,
    prompt_ids: Sequence[object],
    prompts: Sequence[str],
    responses: Sequence[str],
    lengths: Sequence[int],
    truncated: Sequence[bool],
    terminated: Sequence[bool],
    seq_kl: Sequence[float],
    kl_tok_sum: Sequence[float],
    kl_tok_n: Sequence[float],
    rm_scores: Sequence[float],
    rm_ref_rescored: Sequence[float] | None = None,
    meta: dict | None = None,
) -> None:
    """Write per-prompt policy generations and diagnostics in JSON format."""
    n = len(responses)
    columns = {
        "prompt_ids": prompt_ids,
        "prompts": prompts,
        "responses": responses,
        "lengths": lengths,
        "truncated": truncated,
        "terminated": terminated,
        "seq_kl": seq_kl,
        "kl_tok_sum": kl_tok_sum,
        "kl_tok_n": kl_tok_n,
        "rm_scores": rm_scores,
    }
    if rm_ref_rescored is not None:
        columns["rm_ref_rescored"] = rm_ref_rescored
    for key, values in columns.items():
        if len(values) != n:
            raise ValueError(f"{key} has length {len(values)} but there are {n} responses")

    output = {
        key: [value.item() if hasattr(value, "item") else value for value in values]
        for key, values in columns.items()
    }
    output["meta"] = meta or {}
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as stream:
        json.dump(output, stream, ensure_ascii=False)


if __name__ == "__main__":
    import tempfile

    score_calls = []

    def score(prompt, response):
        score_calls.append((prompt, response))
        return float(len(response))

    ref, policy = rescore_jointly(score, ["a", "b"], ["xx", "yyy"], ["xx", "zzzz"])
    assert ref == [2.0, 3.0] and policy == [2.0, 4.0] and len(score_calls) == 3
    assert dpo_loss_from_margins([0.0])["heldout_dpo_loss"] == math.log(2.0)
    with tempfile.TemporaryDirectory() as directory:
        output_path = Path(directory) / "policy_generations.json"
        dump_policy_generation_records(
            output_path,
            prompt_ids=[1],
            prompts=["a"],
            responses=["xx"],
            lengths=[2],
            truncated=[False],
            terminated=[True],
            seq_kl=[0.1],
            kl_tok_sum=[0.1],
            kl_tok_n=[2],
            rm_scores=[2.0],
            rm_ref_rescored=[2.0],
        )
        assert json.loads(output_path.read_text(encoding="utf-8"))["lengths"] == [2]
    print("record_hooks smoke test passed")
