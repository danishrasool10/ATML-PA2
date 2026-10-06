from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import nullcontext

import numpy as np
import torch

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
    write_jsonl,
)
from common.generation import batch_generate, response_sequence_logprobs, response_token_logprobs, score_reward_pairs
from common.logging_utils import load_json, save_json, set_seed
from common.metrics import safe_corr, word_count
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer, reference_mode
from task1_dpo.train import row_prompt_id


def _mean(x):
    return float(np.mean(x)) if len(x) else float("nan")


def load_evaluation_bundle(config_path: str, adapter: str, eval_key: str = "dpo_standard_eval", reward=None):
    cfg = load_yaml(config_path)
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"][eval_key]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": reward if reward is not None else load_reward_model(cfg),  # (model, tokenizer)
    }


def stratum_of(row: dict) -> str:
    """Uses the eval file's own stratum label if present (key names assumed); else derives from word counts."""
    for k in ("stratum", "length_stratum", "length_bucket", "bucket"):
        if row.get(k) is not None:
            return str(row[k])
    yc, yr = preference_responses(row)
    d = word_count(yc) - word_count(yr)
    return "chosen_longer" if d > 0 else ("rejected_longer" if d < 0 else "equal_length")


@torch.no_grad()
def preference_records(policy, tokenizer, rows, cfg, beta, batch_size=4):
    max_len = int(cfg["max_sequence_length"])
    device = next(policy.parameters()).device
    out = []
    for s in range(0, len(rows), batch_size):
        chunk = rows[s : s + batch_size]
        enc_c, enc_r = [], []
        for row in chunk:
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            enc_c.append(encode_prompt_response(tokenizer, prompt, yc, max_len))
            enc_r.append(encode_prompt_response(tokenizer, prompt, yr, max_len))
        bc = {k: v.to(device) for k, v in pad_batch(tokenizer, enc_c).items()}
        br = {k: v.to(device) for k, v in pad_batch(tokenizer, enc_r).items()}

        pc, _, _ = response_sequence_logprobs(policy, bc)
        pr, _, _ = response_sequence_logprobs(policy, br)
        with reference_mode(policy):
            rc, _, _ = response_sequence_logprobs(policy, bc)
            rr, _, _ = response_sequence_logprobs(policy, br)
        pc, pr, rc, rr = (t.float().cpu().tolist() for t in (pc, pr, rc, rr))
        ntc, ntr = bc["response_mask"].sum(-1).tolist(), br["response_mask"].sum(-1).tolist()

        for j, row in enumerate(chunk):
            margin = (pc[j] - pr[j]) - (rc[j] - rr[j])
            out.append({
                "data_index": s + j,
                "prompt_id": row_prompt_id(row),
                "stratum": stratum_of(row),
                "policy_chosen_logp": pc[j], "policy_rejected_logp": pr[j],
                "ref_chosen_logp": rc[j], "ref_rejected_logp": rr[j],
                "chosen_tokens": int(ntc[j]), "rejected_tokens": int(ntr[j]),
                "implicit_reward_margin": beta * margin,
                "pref_correct": float(margin > 0),
                "policy_raw_correct": float(pc[j] > pr[j]),
                "ref_raw_correct": float(rc[j] > rr[j]),
            })
    return out


def summarize_preferences(records):
    return {
        "n": len(records),
        "pref_accuracy": _mean([r["pref_correct"] for r in records]),          # implicit-reward accuracy
        "policy_raw_accuracy": _mean([r["policy_raw_correct"] for r in records]),
        "ref_raw_accuracy": _mean([r["ref_raw_correct"] for r in records]),
        "reward_margin_mean": _mean([r["implicit_reward_margin"] for r in records]),
        "chosen_logratio_mean": _mean([r["policy_chosen_logp"] - r["ref_chosen_logp"] for r in records]),
        "rejected_logratio_mean": _mean([r["policy_rejected_logp"] - r["ref_rejected_logp"] for r in records]),
    }


def summarize_by_stratum(records):
    groups = defaultdict(list)
    for r in records:
        groups[r["stratum"]].append(r)
    return {k: summarize_preferences(v) for k, v in sorted(groups.items())}


@torch.no_grad()
def generate_responses(policy, tokenizer, prompts, cfg, batch_size=4, use_reference=False, compute_kl=False, seed=None):
    gen = cfg["generation"]
    max_new = int(cfg["max_generation_tokens"])
    max_prompt = int(cfg["max_sequence_length"]) - max_new
    if seed is not None:
        set_seed(seed)
    res = {"responses": [], "lengths": [], "truncated": [], "terminated": [], "seq_kl": [], "kl_tok_sum": 0.0, "kl_tok_n": 0.0}
    for s in range(0, len(prompts), batch_size):
        chunk = prompts[s : s + batch_size]
        with (reference_mode(policy) if use_reference else nullcontext()):
            out = batch_generate(
                policy, tokenizer, chunk, max_prompt, max_new,
                temperature=float(gen["temperature"]), top_p=float(gen["top_p"]), do_sample=bool(gen["do_sample"]),
            )
        res["responses"] += out["responses"]
        res["lengths"] += out["response_lengths"]
        res["truncated"] += out["truncated"]
        res["terminated"] += out["terminated_with_eos"]
        if compute_kl and not use_reference:
            args = (out["sequences"], out["attention_mask"], out["prompt_width"], out["response_ids"])
            pol_lp, _ = response_token_logprobs(policy, *args)
            with reference_mode(policy):
                ref_lp, _ = response_token_logprobs(policy, *args)
            kl_tok = (pol_lp - ref_lp) * out["response_mask"]
            res["seq_kl"] += kl_tok.sum(-1).float().cpu().tolist()
            res["kl_tok_sum"] += float(kl_tok.sum().item())
            res["kl_tok_n"] += float(out["response_mask"].sum().item())
    return res


def score_in_batches(rm, rm_tok, prompts, responses, batch_size=8):
    out = []
    for s in range(0, len(prompts), batch_size):
        out += score_reward_pairs(rm, rm_tok, prompts[s : s + batch_size], responses[s : s + batch_size]).cpu().tolist()
    return out


def evaluate_adapter(config_path, adapter, name="standard", eval_key="dpo_standard_eval", beta=None, reward=None, n_gen=None):
    bundle = load_evaluation_bundle(config_path, adapter, eval_key, reward)
    cfg, rows, tok, policy = bundle["cfg"], bundle["rows"], bundle["tokenizer"], bundle["policy"]
    rm, rm_tok = bundle["reward"]
    beta = float(cfg["beta"] if beta is None else beta)
    seed = int(cfg["seed"])
    n_gen = int(n_gen or cfg.get("eval_generation_prompts", 200))  # optional dpo.yaml key
    results_dir = repo_path(cfg["results_dir"])

    # 1) held-out preference metrics
    records = preference_records(policy, tok, rows, cfg, beta)
    write_jsonl(results_dir / f"{name}_preference_records.jsonl", records)

    # 2) generations: policy (with sampled KL) vs. reference (adapter off, cached across runs)
    prompts = [prompt_messages_from_preference(r) for r in rows[:n_gen]]
    pol = generate_responses(policy, tok, prompts, cfg, compute_kl=True, seed=seed)
    pol_rm = score_in_batches(rm, rm_tok, prompts, pol["responses"])

    cache = results_dir / f"ref_generations_{eval_key}_{n_gen}.json"  # delete if config/seed changes
    if cache.exists():
        ref = load_json(cache)
    else:
        ref = generate_responses(policy, tok, prompts, cfg, use_reference=True, seed=seed)
        ref["rm_scores"] = score_in_batches(rm, rm_tok, prompts, ref["responses"])
        save_json(cache, ref)

    metrics = {
        "name": name,
        "adapter": str(adapter),
        "beta": beta,
        "seed": seed,
        "eval_file": cfg["paths"][eval_key],
        "preference": summarize_preferences(records),
        "preference_by_stratum": summarize_by_stratum(records),
        "generation": {
            "n_prompts": len(prompts),
            "kl_per_token": pol["kl_tok_sum"] / max(pol["kl_tok_n"], 1.0),
            "kl_per_sequence": _mean(pol["seq_kl"]),
            "rm_score_policy": _mean(pol_rm),
            "rm_score_ref": _mean(ref["rm_scores"]),
            "rm_win_rate_vs_ref": _mean([float(a > b) for a, b in zip(pol_rm, ref["rm_scores"])]),
            "mean_len_tokens_policy": _mean(pol["lengths"]),
            "median_len_tokens_policy": float(np.median(pol["lengths"])),
            "mean_len_tokens_ref": _mean(ref["lengths"]),
            "mean_words_policy": _mean([word_count(t) for t in pol["responses"]]),
            "truncation_rate_policy": _mean([float(t) for t in pol["truncated"]]),
            "eos_rate_policy": _mean([float(t) for t in pol["terminated"]]),
            "length_rm_corr_policy": safe_corr(pol["lengths"], pol_rm),
        },
        "qualitative_examples": [
            {
                "prompt": prompts[i][-1]["content"],
                "reference_response": ref["responses"][i],
                "policy_response": pol["responses"][i],
                "rm_ref": ref["rm_scores"][i],
                "rm_policy": pol_rm[i],
            }
            for i in range(min(5, len(prompts)))
        ],
    }
    save_json(results_dir / f"{name}_eval.json", metrics)

    bundle.clear()
    del policy, rm
    clear_gpu()
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    ap.add_argument("--eval-key", default="dpo_standard_eval")
    ap.add_argument("--beta", type=float)
    ap.add_argument("--n-gen", type=int)
    args = ap.parse_args()
    m = evaluate_adapter(args.config, args.adapter, args.name, args.eval_key, args.beta, None, args.n_gen)
    print(m["preference"])
    print(m["generation"])


if __name__ == "__main__":
    main()