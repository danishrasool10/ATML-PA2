"""Common held-out protocol for the RL continuation tasks (identical for every condition so results compare)."""
from __future__ import annotations

import math
import time

import torch

from common.data import prompt_messages
from common.generation import batch_generate, score_reward_pairs
from common.models import reference_mode
from common.utils import cuda_sync, distinct2, generation_mode, policy_forward_stats


def _last_user_text(msgs) -> str:
    for m in reversed(msgs):
        if isinstance(m, dict) and m.get("role") == "user":
            return str(m.get("content", ""))
    return str(msgs)


def eval_cap(cfg: dict) -> int:
    for k in ("eval_max_response_length", "max_response_length", "max_completion_length"):
        if cfg.get(k) is not None:
            return int(cfg[k])
    raise KeyError("config needs eval_max_response_length / max_response_length / max_completion_length")


@torch.no_grad()
def evaluate_policy(policy, tokenizer, reward_model, reward_tokenizer, rows, cfg, *,
                    max_prompts: int | None = None, batch_size: int | None = None,
                    logprob_chunk: int | None = None, seed: int | None = None) -> dict:
    """Same prompts/order, generation config and per-batch seed (common random numbers) for every condition.
    Metrics: learned reward, KL(pi||ref) per token/sequence (ref = adapter disabled), full-vocab entropy, length,
    truncation rate, distinct-2. Returns {"summary": {...}, "samples": [...]}."""
    rows = rows[:max_prompts] if max_prompts else rows
    bs = int(batch_size or cfg.get("eval_batch_size", 8))
    chunk = int(logprob_chunk or cfg.get("eval_logprob_chunk", 2))
    gen_cfg = cfg.get("eval_generation", cfg["generation"])
    max_new = eval_cap(cfg)
    base_seed = int(cfg["seed"] if seed is None else seed) + 17
    t0 = time.perf_counter()

    samples = []
    for bi, start in enumerate(range(0, len(rows), bs)):
        batch_rows = rows[start:start + bs]
        msgs = [prompt_messages(r) for r in batch_rows]
        torch.manual_seed(base_seed + bi)
        with generation_mode(policy):
            gen = batch_generate(
                policy, tokenizer, msgs, int(cfg["max_prompt_length"]), max_new,
                temperature=gen_cfg["temperature"], top_p=gen_cfg["top_p"], do_sample=gen_cfg["do_sample"],
            )
        pw = gen["prompt_width"]
        seq, resp = gen["sequences"].clone(), gen["response_ids"].clone()
        attn, rmask = gen["attention_mask"].clone(), gen["response_mask"].clone()

        lp, ent = policy_forward_stats(policy, seq, attn, pw, resp, chunk)
        with reference_mode(policy):
            ref_lp, _ = policy_forward_stats(policy, seq, attn, pw, resp, chunk, want_entropy=False)
        rm = score_reward_pairs(reward_model, reward_tokenizer, msgs, gen["responses"],
                                max_length=int(cfg.get("reward_max_length", 1280)))

        kl_sum = ((lp - ref_lp) * rmask).sum(-1).tolist()
        ent_sum = (ent * rmask).sum(-1).tolist()
        rewards = rm.tolist()
        for i, text in enumerate(gen["responses"]):
            samples.append({
                "idx": start + i,
                "prompt": _last_user_text(msgs[i])[:400],
                "response": text[:1200],
                "reward": rewards[i],
                "length": int(gen["response_lengths"][i]),
                "kl_seq": kl_sum[i],
                "entropy_sum": ent_sum[i],
                "truncated": bool(gen["truncated"][i]),
                "distinct2": distinct2(text),
            })
        del seq, resp, lp, ent, ref_lp, gen

    n = len(samples)
    toks = sum(s["length"] for s in samples)
    r = [s["reward"] for s in samples]
    mean_r = sum(r) / n
    sd_r = math.sqrt(sum((x - mean_r) ** 2 for x in r) / max(n - 1, 1))
    cuda_sync()
    summary = {
        "n_prompts": n,
        "reward_mean": mean_r,
        "reward_stderr": sd_r / math.sqrt(n),
        "kl_per_token": sum(s["kl_seq"] for s in samples) / max(toks, 1),
        "kl_per_sequence": sum(s["kl_seq"] for s in samples) / n,
        "entropy": sum(s["entropy_sum"] for s in samples) / max(toks, 1),
        "response_length": toks / n,
        "truncation_rate": sum(s["truncated"] for s in samples) / n,
        "distinct2": sum(s["distinct2"] for s in samples) / n,
        "eval_max_new_tokens": max_new,
        "eval_wall_s": time.perf_counter() - t0,
    }
    return {"summary": summary, "samples": samples}
