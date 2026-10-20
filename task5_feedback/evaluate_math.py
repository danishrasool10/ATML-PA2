"""Task 5, Steps 1 & 3: matched greedy SFT / RLVR / RLAIF generations on GSM8K (in-domain) or SVAMP (transfer),
exact-verifier scoring, the fixed pairwise AI judge against SFT, and per-policy aggregates.

Stages are resumable: generation per policy (keyed on the full data/decoding spec in the manifest), judging through
the judge's persistent cache. Metrics/manifests -> results/task5_feedback/, samples/judgments -> outputs/task5_feedback/.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch

from common.data import load_yaml, prompt_messages, read_jsonl, render_prompt, write_jsonl
from common.generation import batch_generate
from common.logging_utils import set_seed
from common.models import clear_gpu, load_policy, load_tokenizer
from task5_feedback.analysis import aggregate_math, load_generations
from task5_feedback.artifacts import (DATASET_LABEL, POLICIES, REFERENCE, generation_manifest_path, generations_path,
                                      judge_cache_path, judge_manifest_path, judged_path, load_json, pair_digest,
                                      save_json, to_markdown, verified_sha256)
from task5_feedback.judging import judge_pair, judge_spec, open_judge, release_judge
from task5_feedback.rlvr import exact_reward, extract_designated_final

MAX_PROMPT_LENGTH = 512
GEN_KEYS = ("do_sample", "num_beams", "repetition_penalty", "no_repeat_ngram_size", "temperature", "top_p", "top_k")
SHOW = ["accuracy", "format_compliance", "win_rate_vs_sft", "len_mean", "len_std", "len_iqr", "truncation_rate",
        "verifier_judge_agreement", "verifier_judge_decisive_agreement"]


def policy_specs(cfg):
    return {
        "sft": None,
        "rlvr": cfg["policies"]["rlvr"],
        "rlaif": cfg["policies"]["rlaif"],
    }


def dataset_path(cfg, dataset: str):
    if dataset == "gsm":
        return cfg["paths"]["gsm_eval"]
    if dataset == "transfer":
        return cfg["paths"]["math_transfer_eval"]
    raise ValueError(dataset)


def load_math_evaluation(config_path: str, dataset: str, limit: int | None = None):
    cfg = load_yaml(config_path)
    rows = read_jsonl(dataset_path(cfg, dataset))[:limit]
    tokenizer = load_tokenizer(cfg["base_model"])
    return cfg, rows, tokenizer


def load_frozen_policy(cfg, name: str, device=None):
    specs = policy_specs(cfg)
    if name not in specs:
        raise KeyError(name)
    return load_policy(cfg, adapter_path=specs[name], trainable=False, device=device)


def example_id(row) -> str:
    return str(row.get("prompt_id", row["source_index"]))


def decoding_spec(cfg, batch_size: int, repetition_penalty: float) -> dict:
    # Batch size is part of the spec: left-pad width perturbs fp16 logits, so batches must match across policies.
    # repetition_penalty is pinned instead of inherited from the hub generation_config (1.1 for Qwen2.5-Instruct):
    # under greedy decoding it still fires and penalises re-emitting numbers that already occur in the problem.
    return {"do_sample": False, "num_beams": 1, "max_new_tokens": int(cfg["math_max_new_tokens"]),
            "max_prompt_length": MAX_PROMPT_LENGTH, "batch_size": int(batch_size),
            "repetition_penalty": float(repetition_penalty)}


def pin_decoding_(model, spec: dict) -> dict:
    gc = model.generation_config
    gc.do_sample, gc.num_beams, gc.repetition_penalty = False, 1, spec["repetition_penalty"]
    gc.temperature = gc.top_p = gc.top_k = None   # sampling-only knobs; unset so greedy validation stays silent
    return {k: getattr(gc, k, None) for k in GEN_KEYS}


def assert_prompts_fit(tokenizer, rows) -> int:
    # batch_generate right-truncates; an over-long prompt would silently lose its assistant header
    n = max(len(tokenizer.apply_chat_template(prompt_messages(r), tokenize=True, add_generation_prompt=True))
            for r in rows)
    if n > MAX_PROMPT_LENGTH:
        raise ValueError(f"longest rendered prompt {n} > {MAX_PROMPT_LENGTH}")
    return n


def release_prompt_match(tokenizer, rows) -> float | None:
    """GSM rows carry the exact rendered prompt used for the RLVR/RLAIF rollouts; check our rendering reproduces it."""
    pairs = [(render_prompt(tokenizer, prompt_messages(r)), r["prompt"]) for r in rows if isinstance(r.get("prompt"), str)]
    return sum(a == b for a, b in pairs) / len(pairs) if pairs else None


@torch.no_grad()
def generate_policy(cfg, name: str, tokenizer, rows, spec: dict, device=None, log_every: int = 2):
    model = load_frozen_policy(cfg, name, device=device)
    gen_cfg = pin_decoding_(model, spec)
    bs = spec["batch_size"]
    n_batches = -(-len(rows) // bs)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    records, t0 = [], time.perf_counter()
    try:
        for bi, start in enumerate(range(0, len(rows), bs)):
            chunk = rows[start:start + bs]
            gen = batch_generate(model, tokenizer, [prompt_messages(r) for r in chunk],
                                 spec["max_prompt_length"], spec["max_new_tokens"], do_sample=False)
            for r, text, n, trunc, eos in zip(chunk, gen["responses"], gen["response_lengths"], gen["truncated"],
                                              gen["terminated_with_eos"]):
                gold, pred = str(r["gold_final"]), extract_designated_final(text)
                records.append({
                    "example_id": example_id(r), "policy": name, "question": r["question"], "gold_final": gold,
                    "response": text,
                    "response_tokens": int(n) - int(eos),   # response mask includes the first EOS position
                    "truncated": bool(trunc), "terminated_with_eos": bool(eos),
                    "pred_final": pred, "format_ok": pred is not None, "correct": int(exact_reward(text, gold)),
                })
            del gen
            if (bi + 1) % log_every == 0 or bi + 1 == n_batches:
                acc = np.mean([r["correct"] for r in records])
                print(f"[gen:{name}] {bi + 1}/{n_batches} batches  {(time.perf_counter() - t0) / (bi + 1):.1f}s/batch"
                      f"  running acc={acc:.3f}", flush=True)
        peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None
    finally:
        del model
        clear_gpu()
    return records, {"generation_config": gen_cfg, "wall_s": time.perf_counter() - t0, "peak_vram_gib": peak}


def generate_all(cfg, dataset: str, rows, tokenizer, policies, spec: dict, device=None, overwrite: bool = False):
    data_rel = dataset_path(cfg, dataset)
    common = {
        "base_model": cfg["base_model"], "dataset": DATASET_LABEL[dataset], "data_path": data_rel,
        "data_sha256": verified_sha256(data_rel), "n_examples": len(rows),
        "max_rendered_prompt_tokens": assert_prompts_fit(tokenizer, rows),
        "release_prompt_match_rate": release_prompt_match(tokenizer, rows),
        "decoding": spec,
    }
    if common["release_prompt_match_rate"] not in (None, 1.0):
        print(f"[warn] rendered prompts reproduce the released RL prompt for only "
              f"{common['release_prompt_match_rate']:.1%} of rows (chat-template drift?)", flush=True)

    man_path = generation_manifest_path(cfg, dataset)
    manifest = load_json(man_path) if man_path.exists() else {}
    if manifest.get("common", common) != common:
        if set(policies) != set(POLICIES):
            raise RuntimeError(f"data/decoding spec differs from {man_path}; regenerate all three policies together")
        print(f"[gen] spec changed vs {man_path.name}; regenerating {list(POLICIES)}", flush=True)
        manifest = {}
    save_json(man_path, {"common": common, "policies": manifest.get("policies", {})})

    for name in policies:
        dst = generations_path(cfg, dataset, name)
        if dst.exists() and name in manifest.get("policies", {}) and not overwrite:
            print(f"[skip] {name}: {dst.name} matches the current spec", flush=True)
            continue
        adapter = policy_specs(cfg)[name]
        sha = verified_sha256(f"{adapter}/adapter_model.safetensors") if adapter else None
        records, info = generate_policy(cfg, name, tokenizer, rows, spec, device=device)
        write_jsonl(dst, records)
        entry = {
            "adapter": adapter, "adapter_sha256": sha, "n": len(records),
            "accuracy": float(np.mean([r["correct"] for r in records])),
            "format_compliance": float(np.mean([r["format_ok"] for r in records])),
            "mean_response_tokens": float(np.mean([r["response_tokens"] for r in records])),
            "truncation_rate": float(np.mean([r["truncated"] for r in records])),
            **info,
        }
        cur = load_json(man_path)   # re-read: a sibling process may have added another policy
        if cur["common"] != common:
            raise RuntimeError(f"{man_path} spec changed underneath this run")
        cur["policies"][name] = entry
        save_json(man_path, cur)
        print(f"[{dataset}:{name}] acc={entry['accuracy']:.3f} fmt={entry['format_compliance']:.3f} "
              f"tok={entry['mean_response_tokens']:.1f} trunc={entry['truncation_rate']:.3f} "
              f"wall={entry['wall_s']:.0f}s -> {dst.name}", flush=True)


def judge_vs_reference(cfg, dataset: str, gen: pd.DataFrame, log_every: int = 50) -> list[dict]:
    ref = gen[gen["policy"] == REFERENCE].set_index("example_id")["response"]
    todo = gen[gen["policy"] != REFERENCE]
    judge = open_judge(cfg, judge_cache_path(cfg, dataset))
    spec = judge_spec(cfg, judge)
    out, calls, hits, t0 = [], 0, 0, time.perf_counter()
    try:
        for i, r in enumerate(todo.itertuples(index=False), 1):
            sft = ref.loc[r.example_id]
            # policy always enters as candidate `a`; the judge's parity swap randomises the physical slot
            raw, a_first, hit = judge_pair(judge, r.question, r.response, sft)
            calls += r.response != sft
            hits += hit
            out.append({
                "example_id": r.example_id, "policy": r.policy, "judge_raw": raw,
                "judge_pref": {"A": "policy", "B": REFERENCE, "TIE": "tie"}[raw],
                "identical": r.response == sft, "policy_shown_first": a_first,
                "pair_digest": pair_digest(r.response, sft),
            })
            if i % log_every == 0 or i == len(todo):
                fresh = max(calls - hits, 1)
                print(f"[judge:{dataset}] {i}/{len(todo)}  cache_hits={hits}  "
                      f"{(time.perf_counter() - t0) / fresh:.2f}s/new-call", flush=True)
    finally:
        release_judge(judge)
    save_json(judge_manifest_path(cfg, dataset), {
        **spec, "n_pairs": len(out), "n_identical_pairs": len(out) - calls, "n_judge_calls": calls,
        "n_cache_hits": hits, "wall_s_this_run": time.perf_counter() - t0,
    })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--dataset", choices=["gsm", "transfer"], default="gsm")
    ap.add_argument("--stage", choices=["all", "generate", "judge"], default="all")
    ap.add_argument("--policies", nargs="+", default=list(POLICIES), choices=POLICIES,
                    help="generation subset (e.g. split across GPUs); judging always needs all three")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=None, help="first N examples only (smoke test); part of the spec")
    ap.add_argument("--device", default=None)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg, rows, tokenizer = load_math_evaluation(args.config, args.dataset, args.limit)
    set_seed(int(cfg["seed"]))
    print(f"{DATASET_LABEL[args.dataset]}: {len(rows)} prompts | policies {list(policy_specs(cfg))}", flush=True)

    if args.stage in ("all", "generate"):
        spec = decoding_spec(cfg, args.batch_size, args.repetition_penalty)
        generate_all(cfg, args.dataset, rows, tokenizer, args.policies, spec, device=args.device,
                     overwrite=args.overwrite)
    if args.stage == "generate" or (args.stage == "all" and set(args.policies) != set(POLICIES)):
        print("Generation only; run --stage judge once all three policies are generated.")
        return

    gen, _ = load_generations(cfg, args.dataset)
    write_jsonl(judged_path(cfg, args.dataset), judge_vs_reference(cfg, args.dataset, gen))
    res = aggregate_math(cfg, args.dataset, n_boot=args.bootstrap)
    print(to_markdown(res["summary"][[c for c in SHOW if c in res["summary"]]].astype(float)))
    print("saved ->", generation_manifest_path(cfg, args.dataset).parent)


if __name__ == "__main__":
    main()
