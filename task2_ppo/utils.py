"""Small shared helpers for task2_ppo (kept out of common/ on purpose; nothing here changes common/)."""
from __future__ import annotations

import re
from contextlib import contextmanager, nullcontext

import torch
import torch.nn.functional as F

from common.generation import response_token_logprobs
from common.models import resolve_dtype


# --------------------------------------------------------------------------------------- precision / modes
def amp_context(cfg: dict):
    """Autocast for the CRITIC forward only. PEFT keeps LoRA weights fp32 but a ModulesToSave `score` head can
    stay fp16; an fp16 trainable head under AdamW(eps=1e-8) underflows to NaN. We upcast trainable params to fp32
    (upcast_trainable_) and run the critic under autocast so mixed dtypes are legal."""
    dt = resolve_dtype(cfg.get("dtype", "float16"))
    if torch.cuda.is_available() and dt in (torch.float16, torch.bfloat16):
        return torch.autocast("cuda", dtype=dt)
    return nullcontext()


def disable_dropout_(model) -> int:
    """Zero every nn.Dropout (PEFT lora_dropout=0.05 in the release config). Rationale: old log-probs are computed
    without dropout; with dropout active in the update pass rho != 1 at epoch 0 and the clip fraction would measure
    dropout noise instead of policy movement, corrupting the clipping study. Model stays in train() so gradient
    checkpointing remains active."""
    n = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
            n += 1
    return n


def upcast_trainable_(model) -> int:
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


# --------------------------------------------------------------------------------------- snapshots (exact-midpoint forks)
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


# --------------------------------------------------------------------------------------- cuda bookkeeping
def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_peak_vram() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_vram_gib() -> dict:
    if not torch.cuda.is_available():
        return {"allocated": 0.0, "reserved": 0.0}
    return {
        "allocated": torch.cuda.max_memory_allocated() / 2**30,
        "reserved": torch.cuda.max_memory_reserved() / 2**30,
    }


# --------------------------------------------------------------------------------------- forward statistics
def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Full-vocabulary token entropy [B, R] of the (temperature-1) policy distribution."""
    lp = F.log_softmax(logits.float(), dim=-1)
    return -(lp.exp() * lp).sum(-1)


@torch.no_grad()
def policy_forward_stats(model, seq, attn, prompt_width, resp, chunk: int = 2, want_entropy: bool = True):
    """No-grad response log-probs (+ optional full-vocab entropy), chunked over the batch dim so the
    [chunk, R, vocab] fp32 tensors stay small. Uses common.generation.response_token_logprobs so the numerics match
    the grad-enabled update pass exactly (same function, same micro-batch shape => rho == 1 at epoch 0)."""
    lps, ents = [], []
    for i in range(0, seq.shape[0], chunk):
        sl = slice(i, i + chunk)
        lp, logits = response_token_logprobs(model, seq[sl], attn[sl], prompt_width, resp[sl])
        lps.append(lp)
        if want_entropy:
            ents.append(entropy_from_logits(logits))
        del logits
    return torch.cat(lps), (torch.cat(ents) if want_entropy else None)


# --------------------------------------------------------------------------------------- text / stats
def distinct2(text: str) -> float:
    """Fraction of unique word bigrams (1.0 = no repetition). Cheap degeneration/repetition proxy."""
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
