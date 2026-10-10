# Task 1: DPO

This task implements Direct Preference Optimization (DPO) for preference-based alignment.

Included scripts:

- `train.py`: trains the policy with DPO and writes logs/checkpoints.
- `evaluate.py`: evaluates the trained adapter on the held-out DPO validation set.
- `dpo.py`: DPO objective and diagnostics.
- `analyze_length.py`: checks how response-length perturbations affect preference behavior.
- `ablate_beta.py`: sweeps the DPO beta parameter.

Typical run:

```bash
python -m task1_dpo.train --config configs/dpo.yaml --run-name standard
python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter outputs/task1_dpo/standard --name standard
```

Evaluation writes held-out preference records and per-prompt policy generations under
`results/task1_dpo/`. Re-score each adapter with the same evaluation procedure after changing
the evaluator; reference generation caches are reused, and policy/reference reward-model
scores are recomputed jointly one sequence at a time.

Run the objective checks and summarize the saved evaluation/training artifacts with:

```bash
python tests/test_dpo_objective.py
python -m analysis.task1_report_stats --root . --out results/task1_dpo/analysis
```

The statistics command writes numeric JSON and CSV summaries only. It expects the evaluation
records and training logs to exist; it does not retrain models or regenerate missing artifacts.
