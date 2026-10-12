"""Training/evaluation helpers shared by the RL continuation tasks (Task 2 PPO, Task 3 GRPO)."""
from __future__ import annotations

import random
import re
from contextlib import contextmanager, nullcontext

import numpy as np
import torch
import torch.nn.functional as F

from common.generation import response_token_logprobs
from common.models import resolve_dtype


# --------------------------------------------------------------------------------------- devices
def placement(cfg: dict) -> tuple[torch.device, torch.device]:
    """(trainable models, frozen reference/reward models) from cfg['devices']; collapses to one device when fewer
    GPUs are visible and to CPU without CUDA."""
    if not torch.cuda.is_available():
        return torch.device("cpu"), torch.device("cpu")
    d = cfg.get("devices") or {}
    train, frozen = torch.device(d.get("policy", "cuda:0")), torch.device(d.get("frozen", "cuda:1"))
    n = torch.cuda.device_count()
    if (train.index or 0) >= n:
        train = torch.device("cuda:0")
    if (frozen.index or 0) >= n:
        frozen = train
    return train, frozen


def _cuda(devs) -> list[torch.device]:
    if not torch.cuda.is_available():
        return []
    devs = devs or (torch.device("cuda", torch.cuda.current_device()),)
    return sorted({torch.device(d) for d in devs if torch.device(d).type == "cuda"}, key=lambda d: d.index or 0)


def cuda_sync() -> None:
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(i)


def reset_peak_vram(*devs) -> None:
    for d in _cuda(devs):
        torch.cuda.reset_peak_memory_stats(d)


def peak_vram_gib(*devs) -> dict:
    """Peak allocated/reserved GiB summed over `devs` (default: current device)."""
    ds = _cuda(devs)
    return {"allocated": sum(torch.cuda.max_memory_allocated(d) for d in ds) / 2**30,
            "reserved": sum(torch.cuda.max_memory_reserved(d) for d in ds) / 2**30}


def peak_vram_by_device(*devs) -> dict[str, float]:
    return {str(d): torch.cuda.max_memory_allocated(d) / 2**30 for d in _cuda(devs)}


# --------------------------------------------------------------------------------------- precision / modes
def amp_context(cfg: dict):
    """Autocast for forwards that mix fp32 trainable heads with an fp16/bf16 backbone (PPO critic)."""
    dt = resolve_dtype(cfg.get("dtype", "float16"))
    if torch.cuda.is_available() and dt in (torch.float16, torch.bfloat16):
        return torch.autocast("cuda", dtype=dt)
    return nullcontext()


def disable_dropout_(model) -> int:
    """Zero every nn.Dropout so the update pass reproduces the rollout log-probs exactly (rho == 1 at epoch 0).
    The model stays in train() so gradient checkpointing remains active."""
    n = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
            n += 1
    return n


def upcast_trainable_(model) -> int:
    """fp16 trainable params underflow under AdamW(eps=1e-8); keep them in fp32."""
    n = 0
    for p in model.parameters():
        if p.requires_grad and p.dtype in (torch.float16, torch.bfloat16):
            p.data = p.data.float()
            n += 1
    return n


@contextmanager
def generation_mode(model):
    """load_policy(trainable=True) sets config.use_cache=False for gradient checkpointing; re-enable for generate()."""
    cfg = model.config
    prev = getattr(cfg, "use_cache", None)
    cfg.use_cache = True
    try:
        yield
    finally:
        cfg.use_cache = prev if prev is not None else False


# --------------------------------------------------------------------------------------- snapshots
def snapshot_trainable(model) -> dict:
    return {n: p.detach().to("cpu", copy=True) for n, p in model.named_parameters() if p.requires_grad}


@torch.no_grad()
def restore_trainable(model, snap: dict) -> None:
    params = dict(model.named_parameters())
    missing = [n for n in snap if n not in params]
    if missing:
        raise KeyError(f"Snapshot/model mismatch, e.g. {missing[:3]}")
    for n, t in snap.items():
        params[n].copy_(t.to(device=params[n].device, dtype=params[n].dtype))


# --------------------------------------------------------------------------------------- forward statistics
def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    lp = F.log_softmax(logits.float(), dim=-1)
    return -(lp.exp() * lp).sum(-1)


@torch.no_grad()
def policy_forward_stats(model, seq, attn, prompt_width, resp, chunk: int = 2, want_entropy: bool = True):
    """No-grad response log-probs [B, R] (+ full-vocab entropy), chunked over the batch. Uses the same function as
    the grad-enabled update pass, so equal micro-batch shapes give bit-identical old/new log-probs."""
    lps, ents = [], []
    for i in range(0, seq.shape[0], chunk):
        sl = slice(i, i + chunk)
        lp, logits = response_token_logprobs(model, seq[sl], attn[sl], prompt_width, resp[sl])
        lps.append(lp)
        if want_entropy:
            ents.append(entropy_from_logits(logits))
        del logits
    return torch.cat(lps), (torch.cat(ents) if want_entropy else None)


# --------------------------------------------------------------------------------------- schedules / logs
def prompt_schedule(n_pool: int, seed: int, n_updates: int, per_update: int) -> list[list[int]]:
    """Deterministic prompt stream shared by every condition: a k-update fork sees the first k updates' prompts."""
    rng = random.Random(seed)
    order = list(range(n_pool))
    rng.shuffle(order)
    flat = [order[i % n_pool] for i in range(n_updates * per_update)]
    return [flat[u * per_update:(u + 1) * per_update] for u in range(n_updates)]


def window_mean(logs: list[dict], keys: list[str], sl: slice) -> dict:
    out = {}
    for k in keys:
        v = [r[k] for r in logs[sl] if r.get(k) is not None and not (isinstance(r[k], float) and np.isnan(r[k]))]
        out[k] = float(np.mean(v)) if v else None
    return out


# --------------------------------------------------------------------------------------- text / stats
def distinct2(text: str) -> float:
    """Fraction of unique word bigrams (1.0 = no repetition)."""
    w = re.findall(r"\b\w+\b", text.lower())
    if len(w) < 3:
        return 1.0
    bg = list(zip(w[:-1], w[1:]))
    return len(set(bg)) / len(bg)


def explained_variance(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    m = mask.bool()
    t, p = target[m].float(), pred[m].float()
    if t.numel() < 2:
        return float("nan")
    var_t = t.var(unbiased=False)
    if var_t < 1e-8:
        return float("nan")
    return float(1.0 - (t - p).var(unbiased=False) / var_t)
