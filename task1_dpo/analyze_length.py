from __future__ import annotations

import argparse

import numpy as np

from common.data import (
    load_yaml,
    preference_responses,
    prompt_messages,
    read_jsonl,
    repo_path,
    write_jsonl,
)
from common.logging_utils import save_json
from common.metrics import parse_word_limit, word_count, word_limit_compliance
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer
from task1_dpo.evaluate import evaluate_adapter, generate_responses
from task1_dpo.train import row_prompt_id, run_training


def dataset_length_stats(rows):
    d = np.asarray([word_count(preference_responses(r)[0]) - word_count(preference_responses(r)[1]) for r in rows])
    return {"n": int(len(d)), "chosen_longer_frac": float((d > 0).mean()), "mean_word_diff": float(d.mean())}


def _last_user_text(messages):
    for m in reversed(messages):
        if m.get("role") == "user":
            return str(m["content"])
    return str(messages[-1]["content"])


def word_limit_analysis(config_path, adapter, name, include_reference=False):
    cfg = load_yaml(config_path)
    rows = read_jsonl(cfg["paths"]["word_limit_prompts"])
    prompts = [prompt_messages(r) for r in rows]
    tok = load_tokenizer(cfg["base_model"])
    policy = load_policy(cfg, adapter_path=adapter, trainable=False)
    seed = int(cfg["seed"])
    results_dir = repo_path(cfg["results_dir"])

    out = {}
    for label, use_ref in [("policy", False)] + ([("reference", True)] if include_reference else []):
        gen = generate_responses(policy, tok, prompts, cfg, use_reference=use_ref, seed=seed)
        recs = []
        for i, (row, p, resp) in enumerate(zip(rows, prompts, gen["responses"])):
            text = _last_user_text(p)
            limit, words = parse_word_limit(text), word_count(resp)
            recs.append({
                "data_index": i,
                "prompt_id": row_prompt_id(row),
                "limit": limit,
                "words": words,
                "prompt": text,
                "response": resp,
                "tokens": gen["lengths"][i],
                "truncated": gen["truncated"][i],
                "compliant": word_limit_compliance(text, resp),
                "overshoot": None if limit is None else max(0, words - limit),
            })
        write_jsonl(results_dir / f"word_limit_{name}_{label}.jsonl", recs)
        scored = [r for r in recs if r["compliant"] is not None]
        out[label] = {
            "n_prompts": len(recs),
            "n_with_parsed_limit": len(scored),
            "compliance": float(np.mean([r["compliant"] for r in scored])) if scored else float("nan"),
            "mean_overshoot_words": float(np.mean([r["overshoot"] for r in scored])) if scored else float("nan"),
            "mean_words": float(np.mean([r["words"] for r in recs])),
        }
    del policy
    clear_gpu()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    std_adapter, bal_adapter = cfg["standard_output"], cfg["length_output"]

    print("Length-balanced train rows:", len(read_jsonl(cfg["paths"]["dpo_length_train"])))
    print("Length-stratified eval rows:", len(read_jsonl(cfg["paths"]["dpo_length_eval"])))

    if not (repo_path(std_adapter) / "adapter_config.json").exists():
        raise FileNotFoundError(f"Run `python -m task1_dpo.train` first; missing {std_adapter}")

    # 1) length-balanced condition: identical hyperparameters/seed, only the training data differs
    study = {"train_data": {
        "standard": dataset_length_stats(read_jsonl(cfg["paths"]["dpo_standard_train"])),
        "length_balanced": dataset_length_stats(read_jsonl(cfg["paths"]["dpo_length_train"])),
    }}
    if not args.skip_train:
        study["length_balanced_training"] = run_training(
            args.config, "length_balanced", cfg["paths"]["dpo_length_train"], bal_adapter
        )
        clear_gpu()

    # 2) per-stratum + word-limit analyses for both conditions
    reward = load_reward_model(cfg)
    for k, (name, adapter) in enumerate([("standard", std_adapter), ("length_balanced", bal_adapter)]):
        ev = evaluate_adapter(args.config, adapter, name=f"{name}_lengthstrat", eval_key="dpo_length_eval", reward=reward)
        wl = word_limit_analysis(args.config, adapter, name, include_reference=(k == 0))
        study[name] = {
            "stratified_eval": {
                "preference": ev["preference"],
                "by_stratum": ev["preference_by_stratum"],
                "generation": ev["generation"],
            },
            "word_limit": wl,
        }

    save_json(repo_path(cfg["results_dir"]) / "length_study.json", study)
    for name in ("standard", "length_balanced"):
        print(name, {s: round(v["pref_accuracy"], 3) for s, v in study[name]["stratified_eval"]["by_stratum"].items()},
              "word-limit compliance:", round(study[name]["word_limit"]["policy"]["compliance"], 3))


if __name__ == "__main__":
    main()