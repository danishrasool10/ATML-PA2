# Shared utilities

This folder contains the project-wide helpers used by every training and evaluation task.

Key modules:

- `data.py`: YAML/JSONL loading, prompt formatting, text encoding, padding, and dataset helpers.
- `generation.py`: generation and log-probability utilities for policy evaluation and reward scoring.
- `logging_utils.py`: deterministic seeding, JSONL appends, JSON serialization, and timing helpers.
- `models.py`: tokenizer, base-model, LoRA, reward-model, and value-model loading logic.

These utilities keep task-specific code focused on experiment logic rather than duplicated boilerplate.
