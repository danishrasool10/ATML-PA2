from __future__ import annotations

import argparse
import hashlib
import time

import pandas as pd

from common.data import load_yaml, repo_path, write_jsonl
from common.generation import batch_generate
from common.logging_utils import set_seed
from common.models import clear_gpu, load_policy, load_tokenizer
from task4_safety.artifacts import POLICIES, generated_path, load_json, out_dir, save_json

MAX_PROMPT_LENGTH = 256
# Sampling keys are overridden by do_sample=False; these base-model defaults still act under greedy decoding.
INHERITED_GREEDY_KEYS = ("repetition_penalty", "no_repeat_ngram_size")


def policy_specs(cfg):
    return {
        "sft": None,
        "dpo": cfg["policies"]["dpo"],
        "ppo": cfg["policies"]["ppo"],
        "grpo": cfg["policies"]["grpo"],
    }


def load_xstest(cfg):
    df = pd.read_csv(repo_path(cfg["paths"]["xstest"]))
    if not df["xstest_id"].is_unique:
        raise ValueError("xstest_id not unique")
    return df


def decoding_spec(cfg, batch_size: int) -> dict:
    # Batch size is part of the spec: left-pad width shifts fp16 logits, so it must match across policies.
    return {
        "do_sample": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_new_tokens": int(cfg["safety_max_new_tokens"]),
        "max_prompt_length": MAX_PROMPT_LENGTH,
        "batch_size": int(batch_size),
    }


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def adapter_fingerprint(adapter: str | None) -> str | None:
    if adapter is None:
        return None
    d = repo_path(adapter)
    weights = d / "adapter_model.safetensors"
    if not (d / "adapter_config.json").exists() or not weights.exists():
        raise FileNotFoundError(f"standard adapter not found at {d}; Task 4 requires the final Task 1-3 standard runs")
    return sha256_file(weights)


def to_messages(prompts) -> list[list[dict]]:
    return [[{"role": "user", "content": str(p)}] for p in prompts]


def assert_prompts_fit(tokenizer, df: pd.DataFrame) -> int:
    # batch_generate right-truncates; an over-long prompt would silently lose its generation header
    n = max(len(tokenizer.apply_chat_template(m, tokenize=True, add_generation_prompt=True)) for m in to_messages(df["prompt"]))
    if n > MAX_PROMPT_LENGTH:
        raise ValueError(f"longest rendered prompt {n} > {MAX_PROMPT_LENGTH}")
    return n


def generate_for_policy(cfg, policy_name: str, tokenizer, df: pd.DataFrame, spec: dict, device=None):
    model = load_policy(cfg, adapter_path=policy_specs(cfg)[policy_name], trainable=False, device=device)
    gcfg = model.generation_config.to_dict()
    records = []
    try:
        for start in range(0, len(df), spec["batch_size"]):
            chunk = df.iloc[start:start + spec["batch_size"]]
            gen = batch_generate(
                model, tokenizer, to_messages(chunk["prompt"]),
                max_prompt_length=spec["max_prompt_length"],
                max_new_tokens=spec["max_new_tokens"],
                temperature=spec["temperature"], top_p=spec["top_p"], do_sample=spec["do_sample"],
            )
            for row, text, n, trunc, eos in zip(chunk.itertuples(index=False), gen["responses"], gen["response_lengths"],
                                                gen["truncated"], gen["terminated_with_eos"]):
                records.append({
                    "xstest_id": int(row.xstest_id),
                    "policy": policy_name,
                    "prompt": str(row.prompt),
                    "benchmark_class": str(row.benchmark_class),
                    "type": str(row.type),
                    "focus": str(row.focus),
                    "response": text,
                    "response_tokens": int(n),
                    "truncated": bool(trunc),
                    "terminated_with_eos": bool(eos),
                })
    finally:
        del model
        clear_gpu()
    return records, {k: gcfg.get(k) for k in INHERITED_GREEDY_KEYS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--policies", nargs="+", default=list(POLICIES), choices=POLICIES)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    set_seed(int(cfg["seed"]))

    df = load_xstest(cfg)
    tokenizer = load_tokenizer(cfg["base_model"])
    spec = decoding_spec(cfg, args.batch_size)
    common = {
        "base_model": cfg["base_model"],
        "xstest_path": cfg["paths"]["xstest"],
        "xstest_sha256": sha256_file(repo_path(cfg["paths"]["xstest"])),
        "n_prompts": len(df),
        "max_rendered_prompt_tokens": assert_prompts_fit(tokenizer, df),
        "decoding": spec,
    }

    man_path = out_dir(cfg) / "generation_manifest.json"
    manifest = load_json(man_path) if man_path.exists() else {}
    if manifest.get("common", common) != common:
        if set(args.policies) != set(POLICIES):
            raise RuntimeError(f"decoding/prompt spec differs from {man_path}; regenerate all four policies together")
        save_json(man_path, {"common": common, "policies": {}})
        manifest = {}

    for name in args.policies:
        dst = generated_path(cfg, name)
        if dst.exists() and name in manifest.get("policies", {}) and not args.overwrite:
            print(f"[skip] {name}: {dst} exists")
            continue
        adapter = policy_specs(cfg)[name]
        fingerprint = adapter_fingerprint(adapter)
        t0 = time.perf_counter()
        records, gcfg = generate_for_policy(cfg, name, tokenizer, df, spec, device=args.device)
        write_jsonl(dst, records)
        n = len(records)
        entry = {
            "adapter": adapter,
            "adapter_sha256": fingerprint,
            "n": n,
            "mean_response_tokens": sum(r["response_tokens"] for r in records) / n,
            "truncation_rate": sum(r["truncated"] for r in records) / n,
            "wall_s": time.perf_counter() - t0,
            "inherited_generation_config": gcfg,
        }
        manifest = load_json(man_path) if man_path.exists() else {}  # re-read: sibling process may have written
        manifest = {"common": common, "policies": {**manifest.get("policies", {}), name: entry}}
        save_json(man_path, manifest)
        print(f"[{name}] n={n} mean_tok={entry['mean_response_tokens']:.1f} trunc={entry['truncation_rate']:.3f} -> {dst}")

    if len({str(v["inherited_generation_config"]) for v in manifest.get("policies", {}).values()}) > 1:
        raise RuntimeError("inherited generation_config differs across policies; decoding is not common")


if __name__ == "__main__":
    main()
