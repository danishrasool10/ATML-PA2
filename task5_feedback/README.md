# Task 5: RLVR and RLAIF evaluation

This task compares reward-model and feedback-based evaluation strategies for reasoning and alignment tasks.

Included scripts:

- `evaluate_math.py`: runs RLVR and RLAIF evaluation on GSM8K or transfer prompts.
- `score_perturbations.py`: scores response perturbations and feedback sensitivity.
- `compare_feedback.py`: compares feedback-driven reward signals across evaluation settings.

Typical runs:

```bash
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset gsm
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset transfer
python -m task5_feedback.compare_feedback --config configs/feedback.yaml
```
