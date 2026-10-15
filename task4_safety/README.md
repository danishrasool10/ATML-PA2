# Task 4: Safety calibration

Post-hoc XSTest comparison of four frozen policies: SFT (base, no adapter), standard DPO (`outputs/task1_dpo/standard`),
standard PPO (`outputs/task2_ppo/standard`, 20 updates), and standard GRPO (`outputs/task3_grpo/standard`, 20 updates).
Adapter paths come from `configs/feedback.yaml`; no ablation checkpoints are used.

```bash
python -m task4_safety.generate_responses --config configs/feedback.yaml   # step 1 (GPU)
python -m task4_safety.judge_responses    --config configs/feedback.yaml   # step 2 (GPU), runs evaluate_safety when done
python -m task4_safety.evaluate_safety    --config configs/feedback.yaml   # step 2 aggregates (CPU, re-runnable)
python -m task4_safety.make_audit_sheet   --config configs/feedback.yaml   # step 3 blind worksheet (CPU)
# fill results/task4_safety/audit/audit_worksheet.csv -> Human_Label
python -m task4_safety.analyze_audit      --config configs/feedback.yaml   # step 3 agreement (CPU)
```

On a 2-GPU Kaggle session, steps 1 and 2 can be split by policy (`CUDA_VISIBLE_DEVICES=0 ... --policies sft dpo &`
and `CUDA_VISIBLE_DEVICES=1 ... --policies ppo grpo`); manifests merge, and the last judge process to finish aggregates.
Judging is resumable after a session drop: rerun the same command.

| Script | Notes |
|---|---|
| `generate_responses.py` | Greedy decoding, `safety_max_new_tokens`, fixed batch size, CSV order. Refuses to mix decoding specs across policies; records adapter SHA-256 in `generation_manifest.json`. |
| `judge_responses.py` | Supplied judge/prompt/parser unchanged; `judge_one` per row, cached to `judged_<policy>.jsonl`. |
| `evaluate_safety.py` | Headline rates with Wilson 95% CIs, ambiguous/parse-failure rates, judge intent-mismatch and behavioral rates, response length, category label distribution, contrast-family trade-off, paired McNemar flips vs SFT. |
| `make_audit_sheet.py` | Fixed 60 ids (30 SAFE / 30 UNSAFE, `seed`) joined to all four policies, so 240 rows by default (`--policies` narrows it). The worksheet shows only Prompt/Response; policy, class and AI label are kept in `audit_key.csv`. |
| `analyze_audit.py` | Exact agreement, Cohen's kappa, confusion (human x judge) overall/per policy/per class, behavior- vs intent-axis errors, ambiguous rates, human-vs-judge headline rates and policy ranks on the audit subset, `audit_cases.csv` of judge-error / policy-difference examples. |

Human labels: `SAFE_ANSWER`, `OVER_REFUSAL`, `JUSTIFIED_REFUSAL`, `UNSAFE_COMPLIANCE`, `AMBIGUOUS` (or `SA`/`OR`/`JR`/`UC`/`AMB`).
The analysis joins on `audit_id` only, so spreadsheet edits to the Prompt/Response cells are harmless. `--allow-partial`
analyses a partially labelled sheet.

Outputs go to `results/task4_safety/` (and `results/task4_safety/audit/` for the audit).
