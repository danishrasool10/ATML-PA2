from __future__ import annotations

import argparse
import gc
import random
import time

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import append_jsonl, load_json, save_json, set_seed, wall_timer
from common.metrics import masked_mean, sampled_kl
from common.models import (
    load_policy,
    load_reward_model,
    load_tokenizer,
    load_value_model,
    reference_mode,
    token_values,
    trainable_parameters,
    value_parameter_groups,
)
from task2_ppo.evaluate import evaluate_policy
from task2_ppo.ppo import (
    clipping_diagnostics,
    compute_gae,
    normalize_advantages,
    ppo_policy_loss,
    shaped_rewards,
    validate_ppo_objective,
    value_mse_loss,
)
from task2_ppo.utils import (
    amp_context,
    cuda_sync,
    disable_dropout_,
    distinct2,
    explained_variance,
    generation_mode,
    peak_vram_gib,
    policy_forward_stats,
    reset_peak_vram,
    restore_trainable,
    snapshot_trainable,
    upcast_trainable_,
)


# =========================================================================================== setup
def make_optimizers(bundle):
    cfg = bundle["cfg"]
    policy_optimizer = AdamW(
        trainable_parameters(bundle["policy"]),
        lr=float(cfg["policy_learning_rate"]),
    )
    value_optimizer = AdamW(
        value_parameter_groups(
            bundle["value_model"],
            lora_lr=float(cfg["value_lora_learning_rate"]),
            head_lr=float(cfg["value_head_learning_rate"]),
        ),
        weight_decay=0.0,
    )
    return policy_optimizer, value_optimizer


def prepare_ppo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))

    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["ppo_midpoint_policy"],
        trainable=True,
    )
    value_model = load_value_model(
        cfg,
        cfg["paths"]["ppo_midpoint_value"],
        train_mode=cfg.get("value_train_mode", "head_only"),
    )
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])
    eval_rows = read_jsonl(cfg["paths"]["rl_prompt_eval"])

    n_dropout = 0
    if cfg.get("disable_dropout", True):
        n_dropout = disable_dropout_(policy) + disable_dropout_(value_model)
    upcast_trainable_(policy)
    upcast_trainable_(value_model)

    bundle = {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "value_model": value_model,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "eval_rows": eval_rows,
        "dropout_modules_zeroed": n_dropout,
        # Exact midpoint (policy LoRA, critic LoRA + head) kept on CPU so every condition can be re-started from
        # the *identical* state without re-loading the 1.5B policy / 8-bit reward model from disk.
        "midpoint": {
            "policy": snapshot_trainable(policy),
            "value": snapshot_trainable(value_model),
        },
    }
    bundle["policy_optimizer"], bundle["value_optimizer"] = make_optimizers(bundle)
    return bundle


def reset_to_midpoint(bundle):
    """Restore the supplied checkpoint state and re-create fresh optimizers (no Adam state carried across runs)."""
    restore_trainable(bundle["policy"], bundle["midpoint"]["policy"])
    restore_trainable(bundle["value_model"], bundle["midpoint"]["value"])
    bundle["policy"].zero_grad(set_to_none=True)
    bundle["value_model"].zero_grad(set_to_none=True)
    bundle["policy_optimizer"], bundle["value_optimizer"] = make_optimizers(bundle)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def prompt_schedule(n_pool: int, seed: int, n_updates: int, per_update: int):
    """Deterministic prompt stream: identical across conditions, so every fork sees the same prompts at the same
    update index (matched design). Fork k's updates == the first k updates of the standard run's prompts."""
    rng = random.Random(seed)
    order = list(range(n_pool))
    rng.shuffle(order)
    flat = [order[i % n_pool] for i in range(n_updates * per_update)]
    return [flat[u * per_update:(u + 1) * per_update] for u in range(n_updates)]


# =========================================================================================== rollout + advantages
@torch.no_grad()
def advantage_pipeline(bundle, core: dict, task_reward: torch.Tensor, kl_beta: float) -> dict:
    """critic values -> KL-shaped token rewards -> GAE -> (normalized advantages, returns).
    Shared by live rollouts and the cached-batch clipping analysis so both use the identical computation.
    `core` needs: seq, attn, prompt_width, resp, rmask, old_lp, ref_lp."""
    cfg = bundle["cfg"]
    seq, attn, pw, rmask = core["seq"], core["attn"], core["prompt_width"], core["rmask"]
    R = core["resp"].shape[1]
    chunk = max(1, min(int(cfg.get("micro_batch_size", 2)), seq.shape[0]))

    vals = []
    for i in range(0, seq.shape[0], chunk):
        with amp_context(cfg):
            v = token_values(bundle["value_model"], seq[i:i + chunk], attn[i:i + chunk])
        # V(s_t) is read at the position that predicts a_t: index (prompt_width - 1 + t)
        vals.append(v[:, pw - 1: pw - 1 + R].float())
    values = torch.cat(vals)

    rewards = shaped_rewards(task_reward, core["old_lp"], core["ref_lp"], rmask, kl_beta)
    adv_raw, returns = compute_gae(
        rewards, values, rmask, gamma=float(cfg["gamma"]), lam=float(cfg["gae_lambda"])
    )
    core.update(
        values=values,
        rewards=rewards,
        task_reward=task_reward,
        adv_raw=adv_raw,
        adv=normalize_advantages(adv_raw, rmask),  # normalize AFTER returns = adv_raw + values were formed
        ret=returns,
        ev=explained_variance(values, returns, rmask),
    )
    return core


@torch.no_grad()
def collect_rollout(bundle, rows: list[dict], step: int, kl_beta: float) -> dict:
    cfg, tok, policy = bundle["cfg"], bundle["tokenizer"], bundle["policy"]
    gen_cfg = cfg["generation"]
    msgs = [prompt_messages(r) for r in rows]

    # Per-update seed => at update 0 every condition draws the *same* rollouts (policies are identical there).
    torch.manual_seed(int(cfg["seed"]) * 10_007 + step)
    with generation_mode(policy):
        gen = batch_generate(
            policy, tok, msgs,
            int(cfg["max_prompt_length"]), int(cfg["max_response_length"]),
            temperature=gen_cfg["temperature"], top_p=gen_cfg["top_p"], do_sample=gen_cfg["do_sample"],
        )
    pw = gen["prompt_width"]
    # clone(): generate() runs under inference_mode; inference tensors cannot be saved for backward later.
    seq, resp = gen["sequences"].clone(), gen["response_ids"].clone()
    attn, rmask = gen["attention_mask"], gen["response_mask"]
    chunk = max(1, min(int(cfg.get("micro_batch_size", 2)), seq.shape[0]))

    # old policy log-probs + full-vocab entropy in ONE no-grad pass; reference = same model with adapter disabled
    old_lp, ent = policy_forward_stats(policy, seq, attn, pw, resp, chunk)
    with reference_mode(policy):
        ref_lp, _ = policy_forward_stats(policy, seq, attn, pw, resp, chunk, want_entropy=False)

    rm_score = score_reward_pairs(
        bundle["reward_model"], bundle["reward_tokenizer"], msgs, gen["responses"],
        max_length=int(cfg["reward_max_length"]),
    ).to(old_lp.device)
    terminated = torch.tensor(gen["terminated_with_eos"], device=old_lp.device, dtype=torch.bool)
    task_reward = rm_score - float(cfg["missing_eos_penalty"]) * (~terminated).float()

    core = {"seq": seq, "attn": attn, "prompt_width": pw, "resp": resp, "rmask": rmask,
            "old_lp": old_lp, "ref_lp": ref_lp}
    advantage_pipeline(bundle, core, task_reward, kl_beta)

    m = rmask.bool()
    stats = torch.stack([
        rm_score.mean(),
        task_reward.mean(),
        sampled_kl(old_lp, ref_lp, rmask),                       # per-token KL(pi||ref), sampled-token estimator
        ((old_lp - ref_lp) * rmask).sum(-1).mean(),              # per-sequence KL
        masked_mean(ent, rmask),                                 # full-vocab entropy
        -masked_mean(old_lp, rmask),                             # sampled-token NLL ("sample_entropy" of common.metrics)
        rmask.sum(-1).mean(),                                    # response length
        core["values"][m].mean(),
        core["ret"][m].mean(),
        rmask.sum(),                                             # generated tokens this update
    ]).tolist()
    core["stats"] = {
        "reward": stats[0], "reward_penalized": stats[1], "kl": stats[2], "kl_seq": stats[3],
        "entropy": stats[4], "entropy_sampled": stats[5], "response_length": stats[6],
        "value_mean": stats[7], "return_mean": stats[8], "generated_tokens": stats[9],
        "truncation_rate": float(np.mean(gen["truncated"])),
        "eos_rate": float(np.mean(gen["terminated_with_eos"])),
        "explained_variance": core["ev"],
        "distinct2": float(np.mean([distinct2(t) for t in gen["responses"]])),
        "sample_prompt": str(msgs[0][-1].get("content", ""))[:300] if isinstance(msgs[0][-1], dict) else "",
        "sample_response": gen["responses"][0][:500],
        "sample_reward": float(rm_score[0]),
    }
    del gen
    return core


# =========================================================================================== PPO update
def ppo_update(bundle, batch: dict, eps: float, epochs: int | None = None) -> dict:
    """`ppo_epochs` passes over the rollout batch. Policy and critic have separate optimizers and disjoint graphs, so
    each micro-batch does policy fwd->bwd and THEN critic fwd->bwd (never both graphs alive: lower peak VRAM).
    Micro-batch losses are token-weighted so the accumulated gradient equals the full-batch masked mean."""
    cfg = bundle["cfg"]
    policy, value_model = bundle["policy"], bundle["value_model"]
    popt, vopt = bundle["policy_optimizer"], bundle["value_optimizer"]
    epochs = int(epochs or cfg["ppo_epochs"])
    pw, R, B = batch["prompt_width"], batch["resp"].shape[1], batch["seq"].shape[0]
    micro = max(1, min(int(cfg.get("micro_batch_size", 2)), B))
    total_tok = batch["rmask"].sum().clamp_min(1.0)
    p_params, v_params = trainable_parameters(policy), trainable_parameters(value_model)
    max_gn, vcoef = float(cfg["max_grad_norm"]), float(cfg["value_coef"])
    zero = torch.zeros((), device=batch["seq"].device)

    policy.train()
    value_model.train()
    per_epoch, skipped = [], 0
    for _ in range(epochs):
        popt.zero_grad(set_to_none=True)
        vopt.zero_grad(set_to_none=True)
        t_pl = t_vl = t_cf = t_act = t_kl = zero
        max_dev = 0.0
        for s in range(0, B, micro):
            sl = slice(s, s + micro)
            m = batch["rmask"][sl]
            w = m.sum() / total_tok

            # ---- policy: clipped surrogate (fixed objective in ppo.py) ----
            new_lp, logits = response_token_logprobs(policy, batch["seq"][sl], batch["attn"][sl], pw, batch["resp"][sl])
            del logits
            loss_p, ratio, clip_frac = ppo_policy_loss(new_lp, batch["old_lp"][sl], batch["adv"][sl], m, eps)
            (loss_p * w).backward()
            diag = clipping_diagnostics(ratio, batch["adv"][sl], m, eps)
            max_dev = max(max_dev, diag["max_abs_ratio_dev"])
            t_pl = t_pl + loss_p.detach() * w
            t_cf = t_cf + clip_frac * w
            t_act = t_act + diag["clip_fraction_active"] * w
            t_kl = t_kl + masked_mean(batch["old_lp"][sl] - new_lp.detach(), m) * w
            del new_lp, loss_p, ratio

            # ---- critic: MSE to GAE returns ----
            with amp_context(cfg):
                v_all = token_values(value_model, batch["seq"][sl], batch["attn"][sl])
            v_new = v_all[:, pw - 1: pw - 1 + R].float()
            loss_v = value_mse_loss(v_new, batch["ret"][sl], m)
            (vcoef * loss_v * w).backward()
            t_vl = t_vl + loss_v.detach() * w
            del v_all, v_new, loss_v

        gn_p = clip_grad_norm_(p_params, max_gn)   # returns the PRE-clip norm
        gn_v = clip_grad_norm_(v_params, max_gn)
        p_ok, v_ok = bool(torch.isfinite(gn_p)), bool(torch.isfinite(gn_v))
        if p_ok:
            popt.step()
        if v_ok:
            vopt.step()
        skipped += int(not p_ok) + int(not v_ok)   # fp16 overflow guard: never apply a non-finite step

        vals = torch.stack([t_pl, t_vl, t_cf, t_act, t_kl, gn_p.float(), gn_v.float()]).tolist()
        per_epoch.append({
            "policy_loss": vals[0], "value_loss": vals[1], "clip_fraction": vals[2],
            "clip_fraction_active": vals[3], "approx_kl_step": vals[4],
            "grad_norm": vals[5], "value_grad_norm": vals[6], "max_abs_ratio_dev": max_dev,
        })

    keys = ["policy_loss", "value_loss", "clip_fraction", "clip_fraction_active", "approx_kl_step",
            "grad_norm", "value_grad_norm"]
    out = {k: float(np.mean([e[k] for e in per_epoch])) for k in keys}
    out.update({
        "clip_fraction_last_epoch": per_epoch[-1]["clip_fraction"],
        "clip_fraction_active_last_epoch": per_epoch[-1]["clip_fraction_active"],
        "max_abs_ratio_dev": max(e["max_abs_ratio_dev"] for e in per_epoch),
        "skipped_steps": skipped,
        "per_epoch": per_epoch,
    })
    return out


# =========================================================================================== summaries
def stability_stats(logs: list[dict], max_grad_norm: float) -> dict:
    """Stability statistics (all computed from the per-update log of ONE continuation):
      max_abs_ratio_dev       max over every token/epoch/update of |rho-1|            (PRIMARY: peak policy move vs old policy)
      clip_fraction_mean      mean over updates of the fraction of tokens with rho outside [1-eps, 1+eps]
      clip_fraction_active_mean   same but only tokens whose gradient is actually zeroed by the min() (sign-aware)
      grad_norm_cv            std/mean of pre-clip policy grad norm across updates (gradient volatility)
      grad_clip_hit_rate      fraction of updates whose pre-clip grad norm exceeded max_grad_norm
      approx_kl_step_{mean,max}   E[log pi_old - log pi_new] after the update epochs (per-update policy movement)
      kl_max / reward_std     worst-case reference drift / reward volatility across updates
      skipped_steps           optimizer steps dropped for non-finite gradients"""
    g = np.array([r["grad_norm"] for r in logs], dtype=float)
    kl = np.array([r["approx_kl_step"] for r in logs], dtype=float)
    return {
        "max_abs_ratio_dev": float(max(r["max_abs_ratio_dev"] for r in logs)),
        "clip_fraction_mean": float(np.mean([r["clip_fraction"] for r in logs])),
        "clip_fraction_active_mean": float(np.mean([r["clip_fraction_active"] for r in logs])),
        "grad_norm_mean": float(g.mean()),
        "grad_norm_cv": float(g.std() / max(g.mean(), 1e-12)),
        "grad_clip_hit_rate": float((g > max_grad_norm).mean()),
        "approx_kl_step_mean": float(kl.mean()),
        "approx_kl_step_max": float(kl.max()),
        "kl_max": float(max(r["kl"] for r in logs)),
        "reward_std": float(np.std([r["reward"] for r in logs])),
        "skipped_steps": int(sum(r["skipped_steps"] for r in logs)),
    }


_WINDOW_KEYS = ["reward", "reward_penalized", "kl", "entropy", "entropy_sampled", "response_length",
                "policy_loss", "value_loss", "clip_fraction", "grad_norm", "explained_variance",
                "truncation_rate", "distinct2"]


def _window(logs, keys, sl):
    out = {}
    for k in keys:
        v = [r[k] for r in logs[sl] if r.get(k) is not None and not (isinstance(r[k], float) and np.isnan(r[k]))]
        out[k] = float(np.mean(v)) if v else None
    return out


# =========================================================================================== main continuation
def run_ppo(config_path: str, output: str | None = None, updates: int | None = None,
            clip_epsilon: float | None = None, kl_beta: float | None = None, run_name: str = "standard",
            *, bundle: dict | None = None, evaluate_after: bool = False, save_adapter: bool | None = None,
            results_subdir: str | None = None, eval_prompts: int | None = None, quiet: bool = False):
    validate_ppo_objective(verbose=not quiet)   # fails loudly if the clipped-surrogate defect is still present

    bundle = bundle if bundle is not None else prepare_ppo_continuation(config_path)
    cfg = dict(bundle["cfg"])                    # private copy: overrides must not leak into other forks
    if updates is not None:
        cfg["updates"] = int(updates)
    if clip_epsilon is not None:
        cfg["clip_epsilon"] = float(clip_epsilon)
    if kl_beta is not None:
        cfg["kl_beta"] = float(kl_beta)
    bundle = {**bundle, "cfg": cfg}              # shares models/reward model; private cfg + optimizers
    set_seed(int(cfg["seed"]))
    reset_to_midpoint(bundle)                    # identical starting point for every condition

    n_updates, eps, beta = int(cfg["updates"]), float(cfg["clip_epsilon"]), float(cfg["kl_beta"])
    per = int(cfg["prompts_per_update"])
    schedule = prompt_schedule(len(bundle["prompt_rows"]), int(cfg["seed"]), n_updates, per)

    res_dir = repo_path(cfg["results_dir"]) / (results_subdir or run_name)
    res_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = res_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    do_save = save_adapter if save_adapter is not None else (output is not None or run_name == "standard")

    # ------------------------------------------------------------------ continuation loop
    reset_peak_vram()
    cuda_sync()
    elapsed = wall_timer()
    logs: list[dict] = []
    for u in range(n_updates):
        t_u = time.perf_counter()
        rows = [bundle["prompt_rows"][i] for i in schedule[u]]
        batch = collect_rollout(bundle, rows, step=u, kl_beta=beta)
        upd = ppo_update(bundle, batch, eps)
        cuda_sync()

        rec = {"update": u + 1, "run": run_name, "clip_epsilon": eps, "kl_beta": beta,
               "prompt_ids": schedule[u], **batch["stats"],
               **{k: v for k, v in upd.items() if k != "per_epoch"},
               "per_epoch": upd["per_epoch"], "update_wall_s": time.perf_counter() - t_u,
               "vram_peak_gib_so_far": peak_vram_gib()["allocated"]}
        logs.append(rec)
        append_jsonl(metrics_path, rec)
        if not quiet:
            print(f"[{run_name}] {u + 1:02d}/{n_updates} R={rec['reward']:+.3f} KL={rec['kl']:+.4f} "
                  f"H={rec['entropy']:.3f} len={rec['response_length']:.0f} clip={rec['clip_fraction']:.3f} "
                  f"gn={rec['grad_norm']:.3f} Lpi={rec['policy_loss']:+.4f} Lv={rec['value_loss']:.4f} "
                  f"EV={rec['explained_variance']:+.2f} t={rec['update_wall_s']:.1f}s", flush=True)
        del batch

    cuda_sync()
    train_wall = elapsed()
    vram = peak_vram_gib()

    k = max(1, n_updates // 4)
    summary = {
        "run_name": run_name, "clip_epsilon": eps, "kl_beta": beta, "updates": n_updates,
        "prompts_per_update": per, "ppo_epochs": int(cfg["ppo_epochs"]), "seed": int(cfg["seed"]),
        "train_wall_s": train_wall, "sec_per_update": train_wall / n_updates,
        "peak_vram_gib_allocated": vram["allocated"], "peak_vram_gib_reserved": vram["reserved"],
        "generated_tokens": float(sum(r["generated_tokens"] for r in logs)),
        "window": k,
        "first_window": _window(logs, _WINDOW_KEYS, slice(0, k)),
        "last_window": _window(logs, _WINDOW_KEYS, slice(n_updates - k, n_updates)),
        "stability": stability_stats(logs, float(cfg["max_grad_norm"])),
        "dropout_modules_zeroed": bundle.get("dropout_modules_zeroed", 0),
    }

    if do_save:
        out = repo_path(output or cfg["output"])
        out.mkdir(parents=True, exist_ok=True)
        bundle["policy"].save_pretrained(str(out))                         # adapter consumed by evaluate.py / later tasks
        bundle["value_model"].save_pretrained(str(out / "value_state"))
        summary["adapter_dir"] = str(out)

    if evaluate_after:
        res = evaluate_policy(
            bundle["policy"], bundle["tokenizer"], bundle["reward_model"], bundle["reward_tokenizer"],
            bundle["eval_rows"], cfg, max_prompts=eval_prompts,
        )
        summary["heldout"] = res["summary"]
        save_json(res_dir / "heldout.json", res)

    save_json(res_dir / "summary.json", summary)
    if not quiet:
        print(f"[{run_name}] done: {train_wall:.1f}s, peak VRAM {vram['allocated']:.2f} GiB allocated / "
              f"{vram['reserved']:.2f} GiB reserved -> {res_dir}")
    return {**summary, "logs": logs}


# =========================================================================================== fork helpers
def fork_name(eps: float, beta: float) -> str:
    return f"fork_eps{eps:g}_beta{beta:g}"


def load_fork(cfg: dict, name: str) -> dict | None:
    d = repo_path(cfg["results_dir"]) / "forks" / name
    if not (d / "summary.json").exists() or not (d / "metrics.jsonl").exists():
        return None
    return {**load_json(d / "summary.json"), "logs": read_jsonl(d / "metrics.jsonl")}


def run_fork(bundle: dict, config_path: str, eps: float, beta: float, *, force: bool = False,
             eval_prompts: int | None = None) -> dict:
    """One matched short fork: exact midpoint -> cfg['fork_updates'] updates -> held-out evaluation.
    Results are cached on disk by (eps, beta): the (0.20, 0.10) fork is shared by the clipping and KL studies."""
    cfg = bundle["cfg"]
    name = fork_name(eps, beta)
    if not force:
        cached = load_fork(cfg, name)
        if cached is not None:
            print(f"[fork] reusing cached {name}")
            return cached
    return run_ppo(config_path, None, int(cfg["fork_updates"]), eps, beta, name, bundle=bundle,
                   evaluate_after=True, save_adapter=False, results_subdir=f"forks/{name}",
                   eval_prompts=eval_prompts)


def midpoint_baseline(bundle: dict, *, force: bool = False, eval_prompts: int | None = None) -> dict:
    """Held-out metrics of the untouched midpoint (update 0): the zero point for every 'reward went up' claim."""
    cfg = bundle["cfg"]
    path = repo_path(cfg["results_dir"]) / "forks" / "midpoint_baseline_heldout.json"
    if path.exists() and not force:
        return load_json(path)
    reset_to_midpoint(bundle)
    res = evaluate_policy(bundle["policy"], bundle["tokenizer"], bundle["reward_model"],
                          bundle["reward_tokenizer"], bundle["eval_rows"], cfg, max_prompts=eval_prompts)
    save_json(path, res)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--clip-epsilon", type=float)
    ap.add_argument("--kl-beta", type=float)
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--evaluate", action="store_true", help="also run held-out evaluation after the continuation")
    ap.add_argument("--eval-prompts", type=int, default=None)
    args = ap.parse_args()
    run_ppo(args.config, args.output, args.updates, args.clip_epsilon, args.kl_beta, args.run_name,
            evaluate_after=args.evaluate, eval_prompts=args.eval_prompts)


if __name__ == "__main__":
    main()
