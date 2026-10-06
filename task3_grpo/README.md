# Task 3: GRPO

This task continues a midpoint GRPO policy and studies group-relative optimization dynamics.

Included scripts:

- `continue_train.py`: runs the GRPO continuation loop.
- `evaluate.py`: evaluates the trained adapter.
- `analyze_group_size.py`: studies how group size affects optimization behavior.
- `compare_normalization.py`: compares normalization choices and reward scaling assumptions.

Typical run:

```bash
python -m task3_grpo.continue_train --config configs/grpo.yaml --run-name standard
python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter outputs/task3_grpo/standard --name standard
```
