from __future__ import annotations

import argparse

from common.data import load_yaml, read_jsonl, repo_path
from common.evaluate import evaluate_policy
from common.logging_utils import save_json
from common.models import load_policy, load_reward_model, load_tokenizer
from common.utils import placement


def load_evaluation_bundle(config_path: str, adapter: str):
    cfg = load_yaml(config_path)
    dev, frz = placement(cfg)
    reward_model, reward_tokenizer = load_reward_model(cfg, device=frz)
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["rl_prompt_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False, device=dev),
        "reward": (reward_model, reward_tokenizer),
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
    }


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
