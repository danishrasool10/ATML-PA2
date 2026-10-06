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
