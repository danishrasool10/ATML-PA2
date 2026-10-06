# Configuration files

This directory stores the YAML configuration files used by the five post-training tasks.

Typical usage:

```bash
python -m task1_dpo.train --config configs/dpo.yaml
python -m task2_ppo.continue_train --config configs/ppo.yaml
python -m task3_grpo.continue_train --config configs/grpo.yaml
python -m task4_safety.generate_responses --config configs/feedback.yaml
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset gsm
```

The config files define model IDs, training hyperparameters, dataset paths, output locations, and task-specific checkpoints.
