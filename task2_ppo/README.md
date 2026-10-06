# Task 2: PPO continuation

This task continues a midpoint PPO policy with on-policy optimization and value-function training.

Included scripts:

- `continue_train.py`: runs the PPO continuation loop from a supplied checkpoint.
- `evaluate.py`: evaluates the continuation policy.
- `analyze_clipping.py`: investigates clipping behavior and optimization stability.
- `ablate_kl.py`: compares runs across KL penalty strengths.

Typical run:

```bash
python -m task2_ppo.continue_train --config configs/ppo.yaml --run-name standard
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/standard --name standard
```
