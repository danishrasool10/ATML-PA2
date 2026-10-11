from __future__ import annotations

import argparse
import math
import time

import torch

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import batch_generate, score_reward_pairs
from common.logging_utils import save_json
from common.models import load_policy, load_reward_model, load_tokenizer, reference_mode
from task2_ppo.utils import cuda_sync, distinct2, generation_mode, policy_forward_stats


def load_evaluation_bundle(config_path: str, adapter: str):
    cfg = load_yaml(config_path)
    reward_model, reward_tokenizer = load_reward_model(cfg)
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["rl_prompt_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": (reward_model, reward_tokenizer),
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
    }


def _last_user_text(msgs) -> str:
    for m in reversed(msgs):
        if isinstance(m, dict) and m.get("role") == "user":
            return str(m.get("content", ""))
    return str(msgs)


@torch.no_grad()
def evaluate_policy(policy, tokenizer, reward_model, reward_tokenizer, rows, cfg, *,
                    max_prompts: int | None = None, batch_size: int | None = None,
                    logprob_chunk: int | None = None, seed: int | None = None) -> dict:
    """Common held-out protocol (identical for every condition so results are comparable):
      * same prompts in the same order, same generation config, same per-batch RNG seed (common random numbers)
      * eval_max_response_length (768) cap -- the midpoint has a long-tail termination problem at 512
      * metrics: learned reward, KL(pi||ref) per token and per sequence, full-vocab entropy, length, truncation rate
    Returns {"summary": {...}, "samples": [...]}."""
    rows = rows[:max_prompts] if max_prompts else rows
    bs = int(batch_size or cfg.get("eval_batch_size", 8))
    chunk = int(logprob_chunk or cfg.get("eval_logprob_chunk", 2))
    gen_cfg = cfg.get("eval_generation", cfg["generation"])
    max_new = int(cfg.get("eval_max_response_length", cfg["max_response_length"]))
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
        attn, rmask = gen["attention_mask"], gen["response_mask"]

        lp, ent = policy_forward_stats(policy, seq, attn, pw, resp, chunk)
        with reference_mode(policy):
            ref_lp, _ = policy_forward_stats(policy, seq, attn, pw, resp, chunk, want_entropy=False)
        rm = score_reward_pairs(reward_model, reward_tokenizer, msgs, gen["responses"],
                                max_length=int(cfg["reward_max_length"]))

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
        "eval_wall_s": time.perf_counter() - t0,
    }
    return {"summary": summary, "samples": samples}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    ap.add_argument("--max-prompts", type=int, default=None)
    args = ap.parse_args()
    b = load_evaluation_bundle(args.config, args.adapter)
    cfg = b["cfg"]
    res = evaluate_policy(b["policy"], b["tokenizer"], b["reward_model"], b["reward_tokenizer"],
                          b["rows"], cfg, max_prompts=args.max_prompts)
    out = repo_path(cfg["results_dir"]) / f"{args.name}_heldout.json"
    save_json(out, {"adapter": args.adapter, **res})
    print(f"[evaluate:{args.name}]", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res["summary"].items()})
    print("saved ->", out)


if __name__ == "__main__":
    main()
