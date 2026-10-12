from __future__ import annotations

import argparse
import gc
import time

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.evaluate import evaluate_policy
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import append_jsonl, save_json, set_seed, wall_timer
from common.models import load_policy, load_reward_model, load_tokenizer, trainable_parameters
from common.utils import (
    cuda_sync,
    disable_dropout_,
    distinct2,
    generation_mode,
    peak_vram_by_device,
    placement,
    policy_forward_stats,
    prompt_schedule,
    reset_peak_vram,
    restore_trainable,
    snapshot_trainable,
    upcast_trainable_,
    window_mean,
)
from task3_grpo.grpo import group_statistics, grpo_policy_loss, mask_truncated_sequences, validate_grpo_objective


# =========================================================================================== setup
def make_optimizer(policy, cfg: dict):
    return AdamW(trainable_parameters(policy), lr=float(cfg["learning_rate"]))


def prepare_grpo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    dev, frz = placement(cfg)
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(cfg, adapter_path=cfg["paths"]["grpo_midpoint_policy"], trainable=True, device=dev)
    ref_model = load_policy(cfg, adapter_path=None, trainable=False, device=frz).requires_grad_(False)
    reward_model, reward_tokenizer = load_reward_model(cfg, device=frz)
    n_dropout = disable_dropout_(policy) if cfg.get("disable_dropout", True) else 0
    upcast_trainable_(policy)
    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "ref_model": ref_model,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": read_jsonl(cfg["paths"]["rl_prompt_train"]),
        "eval_rows": read_jsonl(cfg["paths"]["rl_prompt_eval"]),
        "optimizer": make_optimizer(policy, cfg),
        "devices": (dev, frz),
        "dropout_modules_zeroed": n_dropout,
        "midpoint": snapshot_trainable(policy),
    }


def reset_to_midpoint(bundle: dict) -> None:
    restore_trainable(bundle["policy"], bundle["midpoint"])
    bundle["policy"].zero_grad(set_to_none=True)
    bundle["optimizer"] = make_optimizer(bundle["policy"], bundle["cfg"])
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =========================================================================================== rollout
@torch.no_grad()
def rollout_group(bundle: dict, row: dict, seed: int) -> dict:
    """K completions of ONE prompt (generation batch = K copies of a single prompt)."""
    cfg, tok, policy = bundle["cfg"], bundle["tokenizer"], bundle["policy"]
    dev, frz = bundle["devices"]
    K, mb = int(cfg["num_generations"]), int(cfg.get("micro_batch_size", 1))
    gcfg = cfg["generation"]
    msgs = [prompt_messages(row)] * K

    torch.manual_seed(seed)
    with generation_mode(policy), torch.inference_mode():
        gen = batch_generate(
            policy, tok, msgs, int(cfg["max_prompt_length"]), int(cfg["max_completion_length"]),
            temperature=gcfg["temperature"], top_p=gcfg["top_p"], do_sample=gcfg["do_sample"],
        )
    pw = gen["prompt_width"]
    seq, resp = gen["sequences"].clone(), gen["response_ids"].clone()
    attn, rmask = gen["attention_mask"].clone(), gen["response_mask"].clone()

    old_lp, ent = policy_forward_stats(policy, seq, attn, pw, resp, mb)
    ref_lp, _ = policy_forward_stats(bundle["ref_model"], seq.to(frz), attn.to(frz), pw, resp.to(frz), mb,
                                     want_entropy=False)
    rewards = score_reward_pairs(bundle["reward_model"], bundle["reward_tokenizer"], msgs, gen["responses"],
                                 max_length=int(cfg["reward_max_length"]))
    truncated = torch.tensor([not e for e in gen["terminated_with_eos"]], device=dev)
    tmask = mask_truncated_sequences(rmask, truncated) if cfg.get("mask_truncated_completions", True) else rmask
    return {
        "seq": seq, "attn": attn, "pw": pw, "resp": resp, "rmask": rmask, "tmask": tmask,
        "old_lp": old_lp, "ref_lp": ref_lp.to(dev), "ent": ent, "rewards": rewards.float().to(dev),
        "truncated": truncated, "texts": gen["responses"], "prompt": str(msgs[0][-1].get("content", "")),
    }


# =========================================================================================== update
def grpo_update(bundle: dict, groups: list[dict], loss_type: str, track_seq_grads: bool = False) -> dict:
    """One optimizer step per policy epoch over every unmasked completion of every group (micro-batch = 1 sequence).

    Groups whose K completions are all truncated contribute no rows; if no row survives the step is skipped (an
    AdamW step on a zero gradient would still move the weights through its moment estimates).
    With `track_seq_grads`, also records ||grad_theta of the per-sequence policy objective|| for every row.
    """
    cfg, policy = bundle["cfg"], bundle["policy"]
    params = trainable_parameters(policy)
    eps, beta, L = float(cfg["clip_epsilon"]), float(cfg["kl_beta"]), int(cfg["max_completion_length"])
    rows = [(gi, i) for gi, g in enumerate(groups) for i in torch.nonzero(g["tmask"].sum(-1) > 0).flatten().tolist()]
    out = {k: None for k in ("policy_loss", "loss", "kl_k3", "clip_fraction", "ratio_mean", "grad_norm")}
    out.update(stepped=False, skipped_steps=0, trained_sequences=len(rows), trained_tokens=0.0, seq_grads=[])
    if not rows:
        return out

    n_seq = float(len(rows))
    n_tok = float(sum(groups[gi]["tmask"][i].sum() for gi, i in rows))
    kw = {"loss_type": loss_type, "max_completion_length": L, "n_seq": n_seq, "n_tok": n_tok}
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    policy.train()
    opt = bundle["optimizer"]
    per_epoch = []
    for epoch in range(int(cfg.get("policy_epochs", 1))):
        opt.zero_grad(set_to_none=True)
        acc = torch.zeros(5, device=groups[0]["old_lp"].device)
        for gi, i in rows:
            g, sl = groups[gi], slice(i, i + 1)
            new_lp, logits = response_token_logprobs(policy, g["seq"][sl], g["attn"][sl], g["pw"], g["resp"][sl])
            del logits
            args = (new_lp, g["old_lp"][sl], g["adv"][sl], g["tmask"][sl], g["ref_lp"][sl], eps)
            loss, d = grpo_policy_loss(*args, beta, **kw)
            if track_seq_grads and epoch == 0:
                pol, _ = grpo_policy_loss(*args, 0.0, **kw)
                gr = torch.autograd.grad(pol, params, retain_graph=True, allow_unused=True)
                gn = torch.sqrt(sum(x.float().square().sum() for x in gr if x is not None)) * n_seq
                out["seq_grads"].append({
                    "group": gi, "row": i, "length": float(g["tmask"][i].sum()), "adv": float(g["adv"][i]),
                    "reward": float(g["rewards"][i]), "grad_norm_seq": float(gn),
                })
                del gr
            loss.backward()
            w = g["tmask"][i].sum() / n_tok
            acc += torch.stack([d["policy_term"], loss.detach(), d["sampled_kl"], d["clip_fraction"] * w,
                                d["ratio_mean"] * w])
            del new_lp, loss
        gn = clip_grad_norm_(params, float(cfg["max_grad_norm"]))
        ok = bool(torch.isfinite(gn))
        if ok:
            opt.step()
        out["skipped_steps"] += int(not ok)
        per_epoch.append(acc.tolist() + [float(gn)])

    m = np.mean(per_epoch, axis=0)
    out.update(policy_loss=m[0], loss=m[1], kl_k3=m[2], clip_fraction=m[3], ratio_mean=m[4], grad_norm=m[5],
               stepped=out["skipped_steps"] < len(per_epoch), trained_tokens=n_tok)
    opt.zero_grad(set_to_none=True)
    return out


# =========================================================================================== logging
@torch.no_grad()
def update_record(groups: list[dict], st: dict, upd: dict) -> dict:
    K = groups[0]["rmask"].shape[0]
    lens = torch.cat([g["rmask"].sum(-1) for g in groups])
    tlens = torch.cat([g["tmask"].sum(-1) for g in groups])
    trunc = torch.cat([g["truncated"] for g in groups])
    tok = lens.sum().clamp_min(1.0)
    kl_seq = torch.cat([((g["old_lp"] - g["ref_lp"]) * g["rmask"]).sum(-1) for g in groups])
    ent = sum((g["ent"] * g["rmask"]).sum() for g in groups) / tok
    nll = -sum((g["old_lp"] * g["rmask"]).sum() for g in groups) / tok
    rewards = torch.cat([g["rewards"] for g in groups])
    adv = st["advantages"]
    uninf = ~st["informative"]
    all_trunc = trunc.view(-1, K).all(-1)
    texts = [t for g in groups for t in g["texts"]]
    return {
        "reward": float(rewards.mean()),
        "reward_std_within": float(st["group_std"].mean()),
        "reward_std_within_unbiased": float(st["group_std_unbiased"].mean()),
        "frac_uninformative_groups": float(uninf.float().mean()),
        "frac_groups_all_truncated": float(all_trunc.float().mean()),
        "frac_groups_no_policy_signal": float((uninf | all_trunc).float().mean()),
        "adv_abs_mean_trained": float(adv[tlens > 0].abs().mean()) if bool((tlens > 0).any()) else 0.0,
        "kl": float(kl_seq.sum() / tok),
        "kl_seq": float(kl_seq.mean()),
        "kl_k3": upd["kl_k3"],
        "entropy": float(ent),
        "entropy_sampled": float(nll),
        "response_length": float(lens.mean()),
        "response_length_trained": float(tlens[tlens > 0].mean()) if bool((tlens > 0).any()) else None,
        "truncation_rate": float(trunc.float().mean()),
        "policy_loss": upd["policy_loss"],
        "loss": upd["loss"],
        "grad_norm": upd["grad_norm"],
        "clip_fraction": upd["clip_fraction"],
        "ratio_mean": upd["ratio_mean"],
        "stepped": upd["stepped"],
        "skipped_steps": upd["skipped_steps"],
        "generated_tokens": float(lens.sum()),
        "trained_tokens": upd["trained_tokens"],
        "trained_sequences": upd["trained_sequences"],
        "distinct2": float(np.mean([distinct2(t) for t in texts])),
        "samples": [
            {"prompt": g["prompt"][:300],
             "completions": [{"reward": float(g["rewards"][i]), "adv": float(adv[gi * K + i]),
                              "length": int(g["rmask"][i].sum()), "truncated": bool(g["truncated"][i]),
                              "text": g["texts"][i][:400]} for i in range(K)]}
            for gi, g in enumerate(groups)
        ],
    }


_WINDOW_KEYS = ["reward", "reward_std_within", "frac_uninformative_groups", "frac_groups_all_truncated", "kl",
                "kl_k3", "entropy", "entropy_sampled", "response_length", "response_length_trained",
                "truncation_rate", "policy_loss", "grad_norm", "clip_fraction", "distinct2"]


# =========================================================================================== continuation
def run_grpo(config_path: str, output: str | None = None, updates: int | None = None, loss_type: str = "grpo",
             run_name: str = "standard", *, bundle: dict | None = None, evaluate_after: bool = False,
             save_adapter: bool | None = None, results_subdir: str | None = None, eval_prompts: int | None = None,
             track_seq_grads: bool = False, quiet: bool = False):
    validate_grpo_objective(verbose=not quiet)

    bundle = bundle if bundle is not None else prepare_grpo_continuation(config_path)
    cfg = dict(bundle["cfg"])
    if updates is not None:
        cfg["updates"] = int(updates)
    bundle = {**bundle, "cfg": cfg}
    set_seed(int(cfg["seed"]))
    reset_to_midpoint(bundle)

    dev, frz = bundle["devices"]
    n_updates, per, K = int(cfg["updates"]), int(cfg["prompts_per_update"]), int(cfg["num_generations"])
    seed = int(cfg["seed"])
    schedule = prompt_schedule(len(bundle["prompt_rows"]), seed, n_updates, per)
    res_dir = repo_path(cfg["results_dir"]) / (results_subdir or run_name)
    res_dir.mkdir(parents=True, exist_ok=True)
    metrics_path, seq_path = res_dir / "metrics.jsonl", res_dir / "seq_grads.jsonl"
    for p in (metrics_path, seq_path):
        p.unlink(missing_ok=True)
    if track_seq_grads:
        seq_path.touch()
    do_save = save_adapter if save_adapter is not None else (output is not None or run_name == "standard")

    reset_peak_vram(dev, frz)
    cuda_sync()
    elapsed = wall_timer()
    for u in range(n_updates):
        t_u = time.perf_counter()
        groups = [rollout_group(bundle, bundle["prompt_rows"][pid], seed * 10_007 + u * per + j)
                  for j, pid in enumerate(schedule[u])]
        rewards = torch.cat([g["rewards"] for g in groups])
        gids = torch.arange(len(groups), device=rewards.device).repeat_interleave(K)
        st = group_statistics(rewards, gids, float(cfg["adv_eps"]), scale_by_std=(loss_type == "grpo"),
                              std_gate=float(cfg["std_gate"]))
        for j, g in enumerate(groups):
            g["adv"] = st["advantages"][j * K:(j + 1) * K]
        upd = grpo_update(bundle, groups, loss_type, track_seq_grads)
        cuda_sync()

        rec = {"update": u + 1, "run": run_name, "loss_type": loss_type, "prompt_ids": schedule[u],
               **update_record(groups, st, upd), "update_wall_s": time.perf_counter() - t_u,
               "elapsed_s": elapsed(), "peak_vram_gib": peak_vram_by_device(dev, frz)}
        append_jsonl(metrics_path, rec)
        for s in upd["seq_grads"]:
            append_jsonl(seq_path, {"update": u + 1, "loss_type": loss_type, **s})
        if not quiet:
            f = lambda x, s="+.4f": "  n/a" if x is None else format(x, s)
            print(f"[{run_name}] {u + 1:02d}/{n_updates} R={rec['reward']:+.3f} sdR={rec['reward_std_within']:.3f} "
                  f"uninf={rec['frac_uninformative_groups']:.2f} trunc={rec['truncation_rate']:.2f} "
                  f"KL={rec['kl']:+.5f} H={rec['entropy']:.3f} len={rec['response_length']:.0f} "
                  f"Lpi={f(rec['policy_loss'])} gn={f(rec['grad_norm'], '.3f')} t={rec['update_wall_s']:.1f}s",
                  flush=True)
        del groups, st, upd

    cuda_sync()
    train_wall = elapsed()
    logs = read_jsonl(metrics_path)
    vram = peak_vram_by_device(dev, frz)
    g = np.array([r["grad_norm"] for r in logs if r["grad_norm"] is not None], dtype=float)
    k = max(1, n_updates // 4)
    summary = {
        "run_name": run_name, "loss_type": loss_type, "updates": n_updates, "prompts_per_update": per,
        "num_generations": K, "policy_epochs": int(cfg.get("policy_epochs", 1)), "seed": seed,
        "learning_rate": float(cfg["learning_rate"]), "clip_epsilon": float(cfg["clip_epsilon"]),
        "kl_beta": float(cfg["kl_beta"]), "max_completion_length": int(cfg["max_completion_length"]),
        "devices": {"policy": str(dev), "frozen": str(frz)},
        "train_wall_s": train_wall, "sec_per_update": train_wall / n_updates,
        "peak_vram_gib": vram, "peak_vram_gib_max": max(vram.values(), default=0.0),
        "generated_tokens": float(sum(r["generated_tokens"] for r in logs)),
        "trained_tokens": float(sum(r["trained_tokens"] for r in logs)),
        "window": k,
        "first_window": window_mean(logs, _WINDOW_KEYS, slice(0, k)),
        "last_window": window_mean(logs, _WINDOW_KEYS, slice(n_updates - k, n_updates)),
        "overall": window_mean(logs, _WINDOW_KEYS, slice(0, n_updates)),
        "signal": {
            "frac_uninformative_groups": float(np.mean([r["frac_uninformative_groups"] for r in logs])),
            "frac_groups_all_truncated": float(np.mean([r["frac_groups_all_truncated"] for r in logs])),
            "frac_groups_no_policy_signal": float(np.mean([r["frac_groups_no_policy_signal"] for r in logs])),
            "updates_without_step": int(sum(not r["stepped"] for r in logs)),
            "nonfinite_steps": int(sum(r["skipped_steps"] for r in logs)),
            "grad_norm_cv": float(g.std() / max(g.mean(), 1e-12)) if g.size else None,
            "grad_clip_hit_rate": float((g > float(cfg["max_grad_norm"])).mean()) if g.size else None,
        },
        "dropout_modules_zeroed": bundle.get("dropout_modules_zeroed", 0),
    }
    if do_save:
        out = repo_path(output or cfg["output"])
        out.mkdir(parents=True, exist_ok=True)
        bundle["policy"].save_pretrained(str(out))
        summary["adapter_dir"] = str(out)
    if evaluate_after:
        res = evaluate_policy(bundle["policy"], bundle["tokenizer"], bundle["reward_model"],
                              bundle["reward_tokenizer"], bundle["eval_rows"], cfg, max_prompts=eval_prompts)
        save_json(res_dir / "heldout.json", res)
        summary["heldout"] = res["summary"]
    save_json(res_dir / "summary.json", summary)
    if not quiet:
        print(f"[{run_name}] done: {train_wall:.1f}s ({train_wall / n_updates:.1f}s/update), peak VRAM "
              + ", ".join(f"{d} {v:.2f} GiB" for d, v in vram.items()) + f" -> {res_dir}")
    return {**summary, "logs": logs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--loss-type", choices=["grpo", "dr_grpo"], default="grpo")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--eval-prompts", type=int, default=None)
    ap.add_argument("--track-seq-grads", action="store_true")
    args = ap.parse_args()
    run_grpo(args.config, args.output, args.updates, args.loss_type, args.run_name, evaluate_after=args.evaluate,
             eval_prompts=args.eval_prompts, track_seq_grads=args.track_seq_grads)


if __name__ == "__main__":
    main()
