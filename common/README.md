# Shared utilities

Project-wide helpers used by every training and evaluation task.

- `data.py`: YAML/JSONL loading, prompt formatting, text encoding, padding.
- `generation.py`: sampling, response log-probabilities, reward-model scoring.
- `models.py`: tokenizer, policy/LoRA, reward-model and value-model loading (optional explicit `device`).
- `metrics.py`: masked means, sampled KL/entropy, text metrics.
- `logging_utils.py`: seeding, JSONL appends, JSON I/O, timing.
- `utils.py`: RL continuation helpers shared by PPO and GRPO: device placement (trainable models on
  `devices.policy`, frozen reference/reward models on `devices.frozen`), per-device peak-VRAM tracking, dropout/precision
  fixes, generation mode, exact-midpoint snapshot/restore, chunked log-prob/entropy passes, the deterministic prompt
  schedule shared by every fork, and window statistics.
- `evaluate.py`: the common held-out protocol (`evaluate_policy`) used by every PPO and GRPO condition.
