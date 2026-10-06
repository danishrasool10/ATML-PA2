# Task 4: Safety calibration

This task evaluates model safety and alignment under a fixed judge setup.

Included scripts:

- `generate_responses.py`: creates candidate responses for safety prompts.
- `judge_responses.py`: scores generated outputs with the course safety judge.
- `make_audit_sheet.py`: assembles auditing summaries and diagnostics.
- `evaluate_safety.py`: computes final safety metrics and comparisons.

Typical run:

```bash
python -m task4_safety.generate_responses --config configs/feedback.yaml
python -m task4_safety.judge_responses --config configs/feedback.yaml
python -m task4_safety.evaluate_safety --config configs/feedback.yaml
```
