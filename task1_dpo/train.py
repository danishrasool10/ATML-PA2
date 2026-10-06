from __future__ import annotations

import argparse
import hashlib
import math
from collections import defaultdict
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
    write_jsonl,
)
from common.generation import response_sequence_logprobs
from common.logging_utils import append_jsonl, save_json, set_seed, wall_timer
from common.models import (
    load_policy,
    load_tokenizer,
    reference_mode,
    resolve_dtype,
    trainable_parameters,
)
from task1_dpo.dpo import dpo_loss


def row_prompt_id(row: dict):
    for k in ("prompt_id", "id", "uid", "example_id"):
        if row.get(k) is not None:
            return row[k]
    return None


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def filter_fitting(tokenizer, rows, max_length, max_examples=None):
    """Drop rows whose prompt alone does not fit (the exact condition encode_prompt_response raises on).

    Returns (kept source indices, kept rows, skipped records). If max_examples is set, stops after
    that many *usable* rows, so "first 600" still means 600 trainable examples.
    """
    kept_idx, kept_rows, skipped = [], [], []
    for i, row in enumerate(rows):
        n = len(tokenizer.apply_chat_template(
            prompt_messages_from_preference(row), tokenize=True, add_generation_prompt=True
        ))
        if n >= max_length:
            skipped.append({"data_index": i, "prompt_id": row_prompt_id(row), "prompt_tokens": n})
            continue
        kept_idx.append(i)
        kept_rows.append(row)
        if max_examples is not None and len(kept_rows) >= int(max_examples):
            break
    return kept_idx, kept_rows, skipped


class IndexedRows(Dataset):
    """Yields (source_file_index, row) so data indices survive filtering and shuffling."""

    def __init__(self, indices, rows):
        self.indices, self.rows = indices, rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.indices[i], self.rows[i]

def make_collate(tokenizer, max_length):
    def collate(rows):
        chosen, rejected, meta = [], [], []
        for data_index, row in rows:
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            chosen.append(encode_prompt_response(tokenizer, prompt, yc, max_length))
            rejected.append(encode_prompt_response(tokenizer, prompt, yr, max_length))
            meta.append({"data_index": data_index, "prompt_id": row_prompt_id(row)})
        return pad_batch(tokenizer, chosen), pad_batch(tokenizer, rejected), meta
    return collate


def prepare_dpo_run(config_path: str, dataset_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    cfg = load_yaml(config_path)
    seed = int(cfg["seed"])
    set_seed(seed)  # before model/LoRA init -> identical adapter init for every fork
    path = dataset_path or cfg["paths"]["dpo_standard_train"]
    max_len = int(cfg["max_sequence_length"])

    tokenizer = load_tokenizer(cfg["base_model"])
    source_indices, rows, skipped = filter_fitting(tokenizer, read_jsonl(path), max_len, max_examples)
    print(f"[data] {path}: kept {len(rows)} examples, skipped {len(skipped)} with prompt >= {max_len} tokens")

    model = load_policy(cfg, trainable=True, fresh_lora=True)
    for p in trainable_parameters(model):  # fp32 master LoRA weights (needed for GradScaler with fp16 base)
        p.data = p.data.float()

    generator = torch.Generator().manual_seed(seed)  # data order independent of global RNG / beta
    loader = DataLoader(
        IndexedRows(source_indices, rows),
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        generator=generator,
        collate_fn=make_collate(tokenizer, max_len),
    )
    optimizer = AdamW(
        trainable_parameters(model),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    return {
        "cfg": cfg,
        "rows": rows,
        "source_indices": source_indices,
        "skipped": skipped,
        "tokenizer": tokenizer,
        "model": model,
        "loader": loader,
        "optimizer": optimizer,
        "beta": float(cfg["beta"] if beta is None else beta),
        "seed": seed,
        "dataset_path": path,
    }


def run_training(config_path: str, run_name: str, dataset_path: str | None = None, output_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    bundle = prepare_dpo_run(config_path, dataset_path, beta, max_examples)
    cfg = bundle["cfg"]
    rows = bundle["rows"]
    model = bundle["model"]
    loader = bundle["loader"]
    optimizer = bundle["optimizer"]
    seed = bundle["seed"]
    beta = bundle["beta"]

    output = repo_path(output_path or cfg["standard_output"])
    output.parent.mkdir(parents=True, exist_ok=True)

    log_path, order_path = output / "train_log.jsonl", output / "data_order.jsonl"
    for p in (log_path, order_path):
        p.unlink(missing_ok=True)

    accum = int(cfg["grad_accum_steps"])
    max_norm = float(cfg["max_grad_norm"])
    epochs = int(cfg["epochs"])
    n_micro = len(loader)
    device = next(model.parameters()).device
    use_scaler = torch.cuda.is_available() and resolve_dtype(cfg.get("dtype", "float16")) == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    init_checksum = round(sum(p.detach().double().sum().item() for p in trainable_parameters(model)), 6)
    ds_file = repo_path(bundle["dataset_path"])
    write_jsonl(
        output / "data_manifest.jsonl",
        ({"data_index": i, "prompt_id": row_prompt_id(r)} for i, r in zip(bundle["source_indices"], rows)),
    )
    write_jsonl(output / "skipped_examples.jsonl", bundle["skipped"])
    save_json(output / "run_config.json", {
        "run_name": run_name,
        "seed": seed,
        "beta": beta,
        "dataset_path": str(ds_file),
        "dataset_sha256": sha256_file(ds_file),
        "num_examples": len(rows),
        "num_skipped_overlength": len(bundle["skipped"]),
        "max_examples": max_examples,
        "effective_batch_size": int(cfg["batch_size"]) * accum,
        "micro_batches_per_epoch": n_micro,
        "init_checksum": init_checksum,
        "fp16_grad_scaler": use_scaler,
        "hyperparameters": cfg,
    })

    set_seed(seed)
    timer = wall_timer()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    step, seen, last = 0, 0, {}

    for epoch in range(epochs):
        sums = defaultdict(float)
        for i, (chosen, rejected, meta) in enumerate(loader):
            group_size = min(accum, n_micro - (i // accum) * accum)
            chosen = {k: v.to(device) for k, v in chosen.items()}
            rejected = {k: v.to(device) for k, v in rejected.items()}

            with torch.no_grad(), reference_mode(model):
                ref_c, _, _ = response_sequence_logprobs(model, chosen)
                ref_r, _, _ = response_sequence_logprobs(model, rejected)
            pol_c, _, _ = response_sequence_logprobs(model, chosen)
            pol_r, _, _ = response_sequence_logprobs(model, rejected)

            loss, diag = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta)
            if epoch == 0 and i == 0:
                print(f"[sanity] step-0 DPO loss = {loss.item():.4f} (expected ~ ln2 = {math.log(2):.4f})")
            scaler.scale(loss / group_size).backward()

            sums["loss"] += loss.item()
            for k, v in diag.items():
                sums[k] += float(v)
            seen += len(meta)
            append_jsonl(order_path, {
                "epoch": epoch,
                "micro_step": i,
                "optimizer_step": step + 1,
                "data_indices": [m["data_index"] for m in meta],
                "prompt_ids": [m["prompt_id"] for m in meta],
            })

            if (i + 1) % accum == 0 or (i + 1) == n_micro:
                scaler.unscale_(optimizer)
                gnorm = torch.nn.utils.clip_grad_norm_(trainable_parameters(model), max_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                last = {k: v / group_size for k, v in sums.items()}
                append_jsonl(log_path, {
                    "epoch": epoch,
                    "step": step,
                    "examples_seen": seen,
                    "lr": optimizer.param_groups[0]["lr"],
                    "grad_norm": float(gnorm),
                    "elapsed_s": timer(),
                    **last,
                })
                if step % 10 == 0:
                    print(f"[{run_name}] step {step} loss {last['loss']:.4f} acc {last['preference_accuracy']:.3f}")
                sums = defaultdict(float)

    model.save_pretrained(str(output))
    summary = {
        "run_name": run_name,
        "beta": beta,
        "optimizer_steps": step,
        "examples_seen": seen,
        "final_step_loss": last.get("loss"),
        "final_step_pref_accuracy": last.get("preference_accuracy"),
        "init_checksum": init_checksum,
        "data_order_sha256": sha256_file(order_path),
        "wall_seconds": timer(),
        "adapter_path": str(output),
    }
    save_json(output / "train_summary.json", summary)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--dataset")
    ap.add_argument("--output")
    ap.add_argument("--beta", type=float)
    ap.add_argument("--max-examples", type=int)
    args = ap.parse_args()
    run_training(args.config, args.run_name, args.dataset, args.output, args.beta, args.max_examples)


if __name__ == "__main__":
    main()
