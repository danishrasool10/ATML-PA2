from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common.data import load_yaml, repo_path
from common.logging_utils import load_json, save_json
from common.models import clear_gpu, load_reward_model
from task1_dpo.evaluate import evaluate_adapter
from task1_dpo.train import run_training


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    betas = [float(b) for b in cfg["betas"]]
    n_short = int(cfg["short_ablation_examples"])
    base_dir = Path(cfg["standard_output"]).parent

    print("Required beta values:", betas)
    print("Short-run examples per condition:", n_short)

    # ---- train matched forks: same seed -> same LoRA init, same first-n examples, same data order ----
    runs = {}
    for beta in betas:
        name = f"beta_{beta:.2f}"
        out = base_dir / name
        if args.skip_existing and (repo_path(out) / "adapter_config.json").exists():
            runs[beta] = load_json(repo_path(out) / "train_summary.json")
        else:
            runs[beta] = run_training(args.config, name, None, str(out), beta=beta, max_examples=n_short)
            clear_gpu()

    for key in ("init_checksum", "data_order_sha256"):
        vals = {r[key] for r in runs.values()}
        assert len(vals) == 1, f"beta forks are not matched on {key}: {vals}"

    # ---- common evaluation protocol: same held-out set, seed, generation settings, one RM load ----
    reward = load_reward_model(cfg)
    table = []
    for beta in betas:
        name = f"beta_{beta:.2f}"
        m = evaluate_adapter(args.config, str(base_dir / name), name=name, beta=beta, reward=reward)
        table.append({
            "beta": beta,
            "final_train_loss": runs[beta]["final_step_loss"],
            "pref_accuracy": m["preference"]["pref_accuracy"],
            "reward_margin": m["preference"]["reward_margin_mean"],
            "chosen_logratio": m["preference"]["chosen_logratio_mean"],
            "kl_per_token": m["generation"]["kl_per_token"],
            "rm_score": m["generation"]["rm_score_policy"],
            "rm_win_rate_vs_ref": m["generation"]["rm_win_rate_vs_ref"],
            "mean_len_tokens": m["generation"]["mean_len_tokens_policy"],
        })

    results_dir = repo_path(cfg["results_dir"])
    save_json(results_dir / "beta_ablation.json", {"runs": runs, "table": table})
    df = pd.DataFrame(table)
    df.to_csv(results_dir / "beta_ablation.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()