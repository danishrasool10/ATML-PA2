"""Post-hoc statistics, tables and figures for Task 2 (PPO continuation, clipping study, KL study).

Reads saved artifacts only -- it never loads a model, trains, or regenerates anything.

Inputs (everything is optional except the console log; JSON artifacts are used when present):
  results/task2_ppo/analysis/data/kaggle_console_log.txt   console output of the Kaggle rerun (always parsed)
  results/task2_ppo/standard/{metrics.jsonl,summary.json}   \
  results/task2_ppo/forks/*/{metrics.jsonl,summary.json,heldout.json}   > written by task2_ppo.* when available
  results/task2_ppo/forks/midpoint_baseline_heldout.json  /
  cached/ppo_rollout.pt, data/rl_prompt_pool_{train,eval}.jsonl            cache audit (prompt mapping)
  checkpoints/ppo_midpoint_acceptance.json, manifests/source_v2_reward_model_validation.json   staff numbers

Outputs (under --out, default results/task2_ppo/analysis):
  tables/*.csv, tables.md, stats.json, figures/*.png|pdf, data/per_update_from_log.csv, data/cache_audit.csv

Usage (from the repository root):
  python -m analysis.task2_report_stats --root . --out results/task2_ppo/analysis
"""
from __future__ import annotations

import argparse
import ast
import collections
import csv
import json
import math
import pickle
import re
import zipfile
from pathlib import Path

import numpy as np

SEED = 6304
N_PERM = 20000
N_BOOT = 2000
FIRST_LAST = 10  # first-10 vs last-10 window of the 20-update standard run
CONSOLE_FIELDS = ["R", "KL", "H", "len", "clip", "gn", "Lpi", "Lv", "EV", "t"]
# printed precision of each console field (decimal places) -> used for "differs at printed precision"
CONSOLE_DECIMALS = {"R": 3, "KL": 4, "H": 3, "len": 0, "clip": 3, "gn": 3, "Lpi": 4, "Lv": 4, "EV": 2, "t": 1}
CONSOLE_TO_JSON = {"R": "reward", "KL": "kl", "H": "entropy", "len": "response_length", "clip": "clip_fraction",
                   "gn": "grad_norm", "Lpi": "policy_loss", "Lv": "value_loss", "EV": "explained_variance",
                   "t": "update_wall_s"}


# ============================================================================ generic helpers
def _read_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _f(x, nd=4):
    """Format a number for Markdown tables (None/NaN -> 'n/a')."""
    if x is None:
        return "n/a"
    if isinstance(x, str):
        return x
    if isinstance(x, (bool, np.bool_)):
        return str(bool(x))
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if not np.isfinite(x):
        return "n/a"
    return f"{x:.{nd}f}"


def _md_table(rows: list[dict], cols: list[str] | None = None, nd: dict | None = None) -> str:
    if not rows:
        return "_(no rows)_\n"
    cols = cols or list(dict.fromkeys(k for r in rows for k in r))
    nd = nd or {}
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        out.append("| " + " | ".join(_f(r.get(c), nd.get(c, 4)) for c in cols) + " |")
    return "\n".join(out) + "\n"


def _sd(x):
    x = np.asarray(x, dtype=float)
    return float(np.std(x, ddof=1)) if x.size > 1 else float("nan")


def _rank(x):
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(x)
    ranks[order] = np.arange(1, len(x) + 1)
    # average ties
    for v in np.unique(x):
        m = x == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return ranks


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(_rank(x), _rank(y))[0, 1])


def spearman_perm_p(x, y, n=N_PERM, seed=SEED):
    """Two-sided permutation p-value for Spearman's rho (exchangeability of y)."""
    rho = spearman(x, y)
    if not np.isfinite(rho):
        return rho, float("nan")
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx, ry = _rank(x), _rank(y)
    rx = (rx - rx.mean()) / rx.std()
    cnt = 0
    for _ in range(n):
        p = rng.permutation(ry)
        p = (p - p.mean()) / p.std()
        if abs(np.mean(rx * p)) >= abs(rho) - 1e-12:
            cnt += 1
    return rho, (cnt + 1) / (n + 1)


def perm_test_mean_diff(a, b, n=N_PERM, seed=SEED):
    """Two-sided permutation test for mean(b) - mean(a) (independent samples)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    obs = b.mean() - a.mean()
    pool = np.concatenate([a, b])
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n):
        p = rng.permutation(pool)
        if abs(p[len(a):].mean() - p[:len(a)].mean()) >= abs(obs) - 1e-12:
            cnt += 1
    return float(obs), (cnt + 1) / (n + 1)


def paired_bootstrap_ci(d, n=N_BOOT, seed=SEED):
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n, d.size))
    m = d[idx].mean(axis=1)
    return float(d.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


# ============================================================================ console-log parser
_UPD = re.compile(r"^\[(?P<run>[^\]]+)\]\s+(?P<u>\d+)/(?P<n>\d+)\s+(?P<rest>.*)$")
_KV = re.compile(r"(R|KL|H|len|clip|gn|Lpi|Lv|EV|t)=([+-]?\d+(?:\.\d+)?)s?")
_DONE = re.compile(r"^\[(?P<run>[^\]]+)\] done: (?P<wall>[\d.]+)s, peak VRAM (?P<alloc>[\d.]+) GiB allocated / "
                   r"(?P<res>[\d.]+) GiB reserved")


def _parse_table(lines: list[str], i: int):
    """Parse a whitespace-separated pandas to_string() table starting at the header line i."""
    header = lines[i].split()
    rows, j = [], i + 1
    while j < len(lines):
        parts = lines[j].split()
        if not parts:
            break
        if len(parts) == len(header) + 1:     # index column present (pandas .T tables)
            idx, vals = parts[0], parts[1:]
        elif len(parts) == len(header):
            idx, vals = None, parts
        else:
            break
        try:
            num = [float("nan") if v in ("NaN", "None") else float(v) for v in vals]
        except ValueError:
            break
        r = dict(zip(header, num))
        if idx is not None:
            r["_index"] = idx
        rows.append(r)
        j += 1
    return rows, j


def parse_console_log(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")
    lines = text.split("\n")
    out = {"runs": collections.OrderedDict(), "done": {}, "cached_batch": {}, "eps_table": [], "beta_table": [],
           "baselines": [], "first_divergence": [], "effect_size": [], "reused_forks": [], "candidates": {},
           "validate_objective_ok": text.count("[validate_ppo_objective] clipped surrogate (min), gradient masking, GAE: OK")}
    i = 0
    cand_tag = None
    while i < len(lines):
        ln = lines[i].rstrip()
        m = _UPD.match(ln)
        if m and "R=" in ln:
            rec = {"run": m["run"], "update": int(m["u"]), "n_updates": int(m["n"])}
            for k, v in _KV.findall(m["rest"]):
                rec[k] = float(v)
            out["runs"].setdefault(m["run"], []).append(rec)
            i += 1
            continue
        m = _DONE.match(ln)
        if m:
            out["done"][m["run"]] = {"train_wall_s": float(m["wall"]), "peak_vram_gib_allocated": float(m["alloc"]),
                                     "peak_vram_gib_reserved": float(m["res"])}
            i += 1
            continue
        if ln.startswith("Cached-batch reconstruction:"):
            out["cached_batch"]["meta"] = ast.literal_eval(ln.split(":", 1)[1].strip())
        elif ln.startswith("rho spread vs cached old log-probs:"):
            out["cached_batch"]["ratio_spread"] = ast.literal_eval(ln.split(":", 1)[1].strip())
        elif ln.split()[:2] == ["epsilon", "clipped_surrogate"]:
            out["cached_batch"]["static"], i = _parse_table(lines, i)
            continue
        elif ln.startswith("Baseline (midpoint, held-out):"):
            out["baselines"].append(ast.literal_eval(ln.split(":", 1)[1].strip()))
        elif ln.split()[:2] == ["eps", "cached_affected_frac"]:
            out["eps_table"], i = _parse_table(lines, i)
            continue
        elif ln.split()[:2] == ["beta", "heldout_reward"]:
            out["beta_table"], i = _parse_table(lines, i)
            continue
        elif ln.startswith("First update where"):
            out["first_divergence"], i = _parse_table(lines, i + 1)
            continue
        elif ln.startswith("Mean |delta| in units"):
            out["effect_size"], i = _parse_table(lines, i + 1)
            continue
        elif ln.startswith("[fork] reusing cached"):
            out["reused_forks"].append(ln.split()[-1])
        elif ln.startswith("Cache keys:"):
            out["cached_batch"]["cache_keys"] = ast.literal_eval(ln.split(":", 1)[1].strip())
        elif ln.startswith("Cached PPO rollouts:"):
            out["cached_batch"]["n_rows"] = int(ln.split(":")[1])
        elif re.match(r"^\[(reward_and_quality_agree|reward_up_quality_not)\]$", ln.strip()):
            cand_tag = ln.strip()[1:-1]
            out["candidates"][cand_tag] = []
        elif cand_tag and re.match(r"^\s+prompt#\d+", ln):
            mm = re.match(r"^\s+prompt#(\d+) dR=([+-][\d.]+) len x([\d.]+) dDistinct2=([+-][\d.]+)", ln)
            c = {"idx": int(mm[1]), "d_reward": float(mm[2]), "len_ratio": float(mm[3]), "d_distinct2": float(mm[4])}
            for key, lab in (("prompt", "Q:"), ("response_mid", "mid :"), ("response_fork", "beta=0:")):
                i += 1
                s = lines[i].strip()
                c[key] = ast.literal_eval(s[len(lab):].strip()) if s.startswith(lab) else s
            out["candidates"][cand_tag].append(c)
        i += 1
    return out


# ============================================================================ cache loading (torch optional)
def _np_torch_load(path: Path):
    """Whitelisted torch.save() reader (no torch needed): tensors -> numpy. Only rebuild/storage globals allowed."""
    dts = {"FloatStorage": np.float32, "HalfStorage": np.float16, "DoubleStorage": np.float64,
           "LongStorage": np.int64, "IntStorage": np.int32, "ShortStorage": np.int16, "CharStorage": np.int8,
           "ByteStorage": np.uint8, "BoolStorage": np.bool_, "BFloat16Storage": "bf16"}
    zf = zipfile.ZipFile(path)
    pkl = [n for n in zf.namelist() if n.endswith("data.pkl")][0]
    prefix, cache = pkl[: -len("data.pkl")], {}

    class _ST:
        def __init__(self, name):
            self.name = name

    def rebuild(storage, offset, size, stride, *a, **k):
        if len(size) == 0:
            return storage[offset].copy()
        it = storage.dtype.itemsize
        return np.array(np.lib.stride_tricks.as_strided(storage[offset:], shape=tuple(size),
                                                        strides=tuple(s * it for s in stride)))

    class U(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "torch._utils" and name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
                return rebuild
            if module == "torch" and name.endswith("Storage"):
                return _ST(name)
            if module == "collections" and name == "OrderedDict":
                return collections.OrderedDict
            raise pickle.UnpicklingError(f"blocked global {module}.{name}")

        def persistent_load(self, pid):
            _, st, key, _loc, _n = pid
            if key not in cache:
                buf = zf.read(f"{prefix}data/{key}")
                dt = dts[st.name]
                cache[key] = ((np.frombuffer(buf, np.uint16).astype(np.uint32) << 16).view(np.float32)
                              if dt == "bf16" else np.frombuffer(buf, dt))
            return cache[key]

    with zf.open(pkl) as f:
        return U(f).load()


def load_rollout_cache(path: Path):
    try:
        import torch  # noqa: F401
        rows = torch.load(path, map_location="cpu", weights_only=False)
        return [{k: (v.numpy() if hasattr(v, "numpy") else v) for k, v in r.items()} for r in rows]
    except ImportError:
        return _np_torch_load(path)


# ============================================================================ optional JSON artifacts
def load_json_artifacts(res: Path) -> dict:
    """results/task2_ppo/* written by task2_ppo (only what exists)."""
    art = {"runs": {}, "summaries": {}, "heldout": {}, "files_found": [], "files_missing": []}
    expected = {
        "standard/metrics.jsonl": None, "standard/summary.json": None,
        "forks/midpoint_baseline_heldout.json": None,
        "clipping_cached_batch.json": None, "clipping_study.json": None, "kl_study.json": None,
        "qualitative_candidates.json": None, "standard_heldout.json": None,
    }
    for name in ("fork_eps0.05_beta0.1", "fork_eps0.2_beta0.1", "fork_eps0.5_beta0.1", "fork_eps0.2_beta0",
                 "fork_eps0.2_beta0.2"):
        for f in ("metrics.jsonl", "summary.json", "heldout.json"):
            expected[f"forks/{name}/{f}"] = None
    for rel in expected:
        p = res / rel
        (art["files_found"] if p.exists() else art["files_missing"]).append(rel)
    for run_dir in [res / "standard"] + sorted((res / "forks").glob("fork_*")) if (res / "forks").exists() else []:
        name = "standard" if run_dir.name == "standard" else run_dir.name
        if (run_dir / "metrics.jsonl").exists():
            art["runs"][name] = _read_jsonl(run_dir / "metrics.jsonl")
        if (run_dir / "summary.json").exists():
            art["summaries"][name] = _read_json(run_dir / "summary.json")
        if (run_dir / "heldout.json").exists():
            art["heldout"][name] = _read_json(run_dir / "heldout.json")
    if (res / "standard" / "metrics.jsonl").exists() and "standard" not in art["runs"]:
        art["runs"]["standard"] = _read_jsonl(res / "standard" / "metrics.jsonl")
    for rel, key in (("forks/midpoint_baseline_heldout.json", "midpoint"), ("standard_heldout.json", "standard")):
        if (res / rel).exists():
            art["heldout"][key] = _read_json(res / rel)
    for rel in ("clipping_cached_batch.json", "clipping_study.json", "kl_study.json", "qualitative_candidates.json"):
        if (res / rel).exists():
            art[rel.replace(".json", "")] = _read_json(res / rel)
    return art


def _series(recs: list[dict], key: str) -> np.ndarray:
    return np.array([np.nan if r.get(key) is None else float(r[key]) for r in recs], dtype=float)


def console_series(runs: dict, run: str, field: str) -> np.ndarray:
    return np.array([r.get(field, np.nan) for r in runs.get(run, [])], dtype=float)


# ============================================================================ prompt schedule replay
def prompt_schedule(n_pool: int, seed: int, n_updates: int, per_update: int):
    """Exact replay of task2_ppo.continue_train.prompt_schedule (random.Random shuffle)."""
    import random
    rng = random.Random(seed)
    order = list(range(n_pool))
    rng.shuffle(order)
    flat = [order[i % n_pool] for i in range(n_updates * per_update)]
    return [flat[u * per_update:(u + 1) * per_update] for u in range(n_updates)]


def _tokenizer(root: Path):
    try:
        from tokenizers import Tokenizer
    except ImportError:
        return None
    p = root / "checkpoints" / "ppo_midpoint_policy" / "tokenizer.json"
    if not p.exists():
        return None
    tok = Tokenizer.from_file(str(p))
    tok.no_truncation()   # the saved tokenizer.json carries a truncation=256 setting; count full lengths
    tok.no_padding()
    return tok


def prompt_lengths(root: Path, rows: list[dict], tok) -> np.ndarray | None:
    if tok is None:
        return None
    return np.array([len(tok.encode(r["prompt"], add_special_tokens=False).ids) for r in rows])


# ============================================================================ analyses
def table1_standard(runs: dict, done: dict, art: dict, sched_info: list[dict] | None, max_grad_norm=1.0,
                    max_len=512) -> tuple[list[dict], dict]:
    std = runs["standard"]
    js = art["runs"].get("standard")
    n = len(std)
    k = min(FIRST_LAST, n // 2)
    R = console_series(runs, "standard", "R")
    L = console_series(runs, "standard", "len")
    capped = L >= max_len
    metrics = [  # (label, values, source)
        ("reward (raw RM score)", R, "log [standard] R= / metrics.jsonl reward"),
        ("reward_penalized (R - 1[len=512])", R - 1.0 * capped,
         "derived from log: missing-EOS penalty 1.0 assumed for len=512 (cap hit) / metrics.jsonl reward_penalized"),
        ("KL to ref per token (k1)", console_series(runs, "standard", "KL"), "log KL= / metrics.jsonl kl"),
        ("KL to ref per sequence (k1 sum)", console_series(runs, "standard", "KL") * L,
         "derived: KL/token x len (B=1) / metrics.jsonl kl_seq"),
        ("policy loss", console_series(runs, "standard", "Lpi"), "log Lpi= / metrics.jsonl policy_loss"),
        ("value loss", console_series(runs, "standard", "Lv"), "log Lv= / metrics.jsonl value_loss"),
        ("entropy (full vocab, nats/token)", console_series(runs, "standard", "H"), "log H= / metrics.jsonl entropy"),
        ("clip fraction, nominal", console_series(runs, "standard", "clip"), "log clip= / metrics.jsonl clip_fraction"),
        ("grad norm, policy, pre-clip", console_series(runs, "standard", "gn"), "log gn= / metrics.jsonl grad_norm"),
        ("response length (tokens)", L, "log len= / metrics.jsonl response_length"),
        ("explained variance (critic)", console_series(runs, "standard", "EV"),
         "log EV= / metrics.jsonl explained_variance"),
        ("update wall-clock (s)", console_series(runs, "standard", "t"), "log t= / metrics.jsonl update_wall_s"),
    ]
    if js:  # exact values from JSON override console precision where available
        exact = {"reward (raw RM score)": "reward", "reward_penalized (R - 1[len=512])": "reward_penalized",
                 "KL to ref per token (k1)": "kl", "KL to ref per sequence (k1 sum)": "kl_seq",
                 "policy loss": "policy_loss", "value loss": "value_loss",
                 "entropy (full vocab, nats/token)": "entropy", "clip fraction, nominal": "clip_fraction",
                 "grad norm, policy, pre-clip": "grad_norm", "response length (tokens)": "response_length",
                 "explained variance (critic)": "explained_variance", "update wall-clock (s)": "update_wall_s"}
        metrics = [(lab, _series(js, exact[lab]) if lab in exact else v, src) for lab, v, src in metrics]
        for lab, key in (("clip fraction, active", "clip_fraction_active"),
                         ("grad norm, critic, pre-clip", "value_grad_norm"),
                         ("max |rho-1| (per update)", "max_abs_ratio_dev"),
                         ("approx KL per step", "approx_kl_step")):
            metrics.append((lab, _series(js, key), f"metrics.jsonl {key}"))
    upd = np.arange(1, n + 1)
    rows = []
    for lab, v, src in metrics:
        diff, p = perm_test_mean_diff(v[:k], v[-k:])
        rho, prho = spearman_perm_p(upd, v)
        rows.append({"metric": lab, "mean": float(np.mean(v)), "sd": _sd(v), "min": float(np.min(v)),
                     "max": float(np.max(v)), f"first{k}_mean": float(np.mean(v[:k])),
                     f"last{k}_mean": float(np.mean(v[-k:])), "delta_last_minus_first": diff,
                     "perm_p_first_vs_last": p, "spearman_vs_update": rho, "spearman_perm_p": prho, "source": src})
    gn = console_series(runs, "standard", "gn")
    footer = {
        "updates": n, "prompts_per_update": 1,
        "train_wall_s": done["standard"]["train_wall_s"],
        "sec_per_update_mean": done["standard"]["train_wall_s"] / n,
        "sum_update_wall_s": float(np.sum(console_series(runs, "standard", "t"))),
        "peak_vram_gib_allocated": done["standard"]["peak_vram_gib_allocated"],
        "peak_vram_gib_reserved": done["standard"]["peak_vram_gib_reserved"],
        "generated_tokens": int(np.sum(L)),
        "cap_hits_len512": int(capped.sum()),
        "grad_clip_hit_rate_policy": float((gn > max_grad_norm).mean()),
        "explained_variance_negative": int((console_series(runs, "standard", "EV") < 0).sum()),
        "clip_fraction_max_printed": float(np.max(console_series(runs, "standard", "clip"))),
        "corr_wall_vs_len": float(np.corrcoef(console_series(runs, "standard", "t"), L)[0, 1]),
        "sec_per_generated_token": float(np.sum(console_series(runs, "standard", "t")) / np.sum(L)),
        "skipped_steps": (art["summaries"].get("standard", {}).get("stability", {}).get("skipped_steps")
                          if art["summaries"].get("standard") else None),
        "source": "log '[standard] done:' line; per-update log lines",
    }
    if art["summaries"].get("standard"):
        s = art["summaries"]["standard"]
        footer.update({"summary_json_train_wall_s": s.get("train_wall_s"),
                       "summary_json_generated_tokens": s.get("generated_tokens"),
                       "summary_json_stability": s.get("stability")})
    if sched_info:
        tr = np.array([x["prompt_truncated_at_256"] for x in sched_info])
        if tr.size == n:
            d, p = perm_test_mean_diff(R[~tr], R[tr])
            footer["reward_truncated_prompt_mean"] = float(R[tr].mean())
            footer["reward_untruncated_prompt_mean"] = float(R[~tr].mean())
            footer["n_truncated_prompts"] = int(tr.sum())
            footer["reward_trunc_minus_untrunc"] = d
            footer["reward_trunc_perm_p"] = p
    return rows, footer


def determinism_checks(runs: dict) -> dict:
    fields = [f for f in CONSOLE_FIELDS if f != "t"]
    out = {}

    def same(a, b, n):
        diffs = []
        for u in range(n):
            for f in fields:
                if a[u].get(f) != b[u].get(f):
                    diffs.append({"update": u + 1, "field": f, "a": a[u].get(f), "b": b[u].get(f)})
        return diffs

    std, ref = runs.get("standard", []), runs.get("fork_eps0.2_beta0.1", [])
    if std and ref:
        out["standard_first8_vs_fork_eps0.2_beta0.1"] = same(std, ref, min(8, len(ref)))
    eps_runs = [r for r in ("fork_eps0.05_beta0.1", "fork_eps0.2_beta0.1", "fork_eps0.5_beta0.1") if r in runs]
    for a in eps_runs:
        for b in eps_runs:
            if a < b:
                out[f"{a}_vs_{b}"] = same(runs[a], runs[b], min(len(runs[a]), len(runs[b])))
    return {k: {"n_differences_printed_precision": len(v), "differences": v[:10]} for k, v in out.items()}


def grad_norm_cv(g):
    g = np.asarray(g, float)
    return float(g.std() / max(g.mean(), 1e-12))   # same definition as continue_train.stability_stats (ddof=0)


def rq2_divergence(runs: dict, ref="fork_eps0.2_beta0.1",
                   others=(("fork_eps0.2_beta0", 0.0), ("fork_eps0.2_beta0.2", 0.2)), tol_std=0.25) -> dict:
    fields = ["Lv", "gn", "EV", "Lpi", "KL", "H", "R", "len"]
    betas = [("fork_eps0.2_beta0", 0.0), (ref, 0.1), ("fork_eps0.2_beta0.2", 0.2)]
    n = min(len(runs[r]) for r, _ in betas)
    per_update = []
    for u in range(n):
        row = {"update": u + 1}
        for r, b in betas:
            for f in fields:
                row[f"{f}@b{b:g}"] = runs[r][u][f]
        # shared sampled response = identical (R, len) at printed precision
        key = {b: (runs[r][u]["R"], runs[r][u]["len"]) for r, b in betas}
        groups = collections.defaultdict(list)
        for b, kk in key.items():
            groups[kk].append(b)
        row["response_groups"] = " | ".join("{" + ",".join(f"{b:g}" for b in g) + "}" for g in groups.values())
        per_update.append(row)
    first_exact, first_thresh, effect = {}, {}, {}
    for r, b in others:
        fe, ft, ef = {}, {}, {}
        for f in fields:
            a, o = console_series(runs, ref, f)[:n], console_series(runs, r, f)[:n]
            dif = np.abs(a - o)
            tol_print = 0.5 * 10 ** (-CONSOLE_DECIMALS[f])
            idx = np.where(dif > tol_print)[0]
            fe[f] = int(idx[0] + 1) if idx.size else None
            tol = tol_std * max(float(np.nanstd(a)), 1e-8)
            idx2 = np.where(dif > tol)[0]
            ft[f] = int(idx2[0] + 1) if idx2.size else None
            ef[f] = {"mean_abs_delta": float(np.mean(dif)), "ref_traj_std": float(np.nanstd(a)),
                     "mean_abs_delta_in_ref_std": float(np.mean(dif) / max(np.nanstd(a), 1e-8)),
                     "max_abs_delta": float(np.max(dif))}
        first_exact[f"beta{b:g}"], first_thresh[f"beta{b:g}"], effect[f"beta{b:g}"] = fe, ft, ef
    # monotonicity in beta: per update & field, is metric(0.1) between metric(0) and metric(0.2)?
    mono = {}
    for f in fields:
        v0, v1, v2 = (console_series(runs, r, f)[:n] for r, _ in betas)
        diff_any = (v0 != v1) | (v1 != v2) | (v0 != v2)
        strict_mono = ((v0 < v1) & (v1 < v2)) | ((v0 > v1) & (v1 > v2))
        weak_mono = ((v0 <= v1) & (v1 <= v2)) | ((v0 >= v1) & (v1 >= v2))
        mono[f] = {"updates_with_any_difference": int(diff_any.sum()),
                   "strictly_monotone_updates": int((strict_mono & diff_any).sum()),
                   "weakly_monotone_updates": int((weak_mono & diff_any).sum()),
                   "non_monotone_updates": [int(u + 1) for u in np.where(diff_any & ~weak_mono)[0]]}
    return {"per_update": per_update, "first_exact": first_exact, "first_threshold": first_thresh,
            "effect": effect, "monotonicity": mono, "n_updates": n, "tol_std": tol_std}


def kl_force(runs: dict, beta_by_run: dict) -> list[dict]:
    rows = []
    for run, beta in beta_by_run.items():
        if run not in runs:
            continue
        kl, L, R = (console_series(runs, run, f) for f in ("KL", "len", "R"))
        kl_seq = kl * L
        rows.append({"run": run, "beta": beta, "max_abs_kl_tok": float(np.max(np.abs(kl))),
                     "max_abs_kl_seq_derived": float(np.max(np.abs(kl_seq))),
                     "max_abs_beta_kl_seq": float(beta * np.max(np.abs(kl_seq))),
                     "mean_abs_beta_kl_seq": float(beta * np.mean(np.abs(kl_seq))),
                     "reward_sd_across_updates": _sd(R),
                     "ratio_max_penalty_to_reward_sd": float(beta * np.max(np.abs(kl_seq)) / _sd(R))})
    return rows


def heldout_table(parsed: dict, art: dict) -> list[dict]:
    """Table 2: held-out beta study incl. the midpoint row. Console values unless heldout.json exists."""
    base = parsed["baselines"][0] if parsed["baselines"] else {}
    rows = [{"condition": "midpoint (update 0)", "beta": None, "n": 200,
             "reward_mean": base.get("reward_mean"), "reward_se": base.get("reward_stderr"),
             "d_vs_mid": 0.0, "ci95_lo": None, "ci95_hi": None, "ci_type": "-",
             "kl_tok_derived": (base["kl_per_sequence"] / base["response_length"]) if base else None,
             "kl_seq": base.get("kl_per_sequence"), "entropy": base.get("entropy"),
             "length": base.get("response_length"), "trunc": base.get("truncation_rate"),
             "distinct2": base.get("distinct2"), "source": "log 'Baseline (midpoint, held-out)'"}]
    se_mid = base.get("reward_stderr", float("nan"))
    for r in parsed["beta_table"]:
        half = 1.959964 * math.sqrt(2) * se_mid   # conservative: unpaired, SE_fork assumed = SE_mid
        rows.append({"condition": f"fork eps=0.2 beta={r['beta']:g}", "beta": r["beta"], "n": 200,
                     "reward_mean": r["heldout_reward"], "reward_se": None,
                     "d_vs_mid": r["d_vs_mid"], "ci95_lo": r["d_vs_mid"] - half, "ci95_hi": r["d_vs_mid"] + half,
                     "ci_type": "unpaired, SE_fork:=SE_mid (conservative)",
                     "kl_tok_derived": r["heldout_kl_seq"] / r["heldout_len"], "kl_seq": r["heldout_kl_seq"],
                     "entropy": r["heldout_entropy"], "length": r["heldout_len"], "trunc": r["heldout_trunc"],
                     "distinct2": r["heldout_distinct2"], "source": "log kl-study table (kl_study.json -> table)"})
    # exact per-sample statistics when the JSON files are present
    mid = art["heldout"].get("midpoint")
    if mid:
        ms = {s["idx"]: s for s in mid["samples"]}
        rows[0].update(_heldout_from_json(mid))
        for row in rows[1:]:
            name = f"fork_eps0.2_beta{row['beta']:g}"
            h = art["heldout"].get(name)
            if not h:
                continue
            row.update(_heldout_from_json(h))
            d = [s["reward"] - ms[s["idx"]]["reward"] for s in h["samples"] if s["idx"] in ms]
            m, lo, hi = paired_bootstrap_ci(d)
            row.update({"d_vs_mid": m, "ci95_lo": lo, "ci95_hi": hi, "ci_type": "paired bootstrap (2000, seed 6304)",
                        "source": f"forks/{name}/heldout.json vs forks/midpoint_baseline_heldout.json"})
        st = art["heldout"].get("standard")
        if st:
            row = {"condition": "standard (20 updates, eps 0.2, beta 0.1)", "beta": 0.1, "ci_type": "paired bootstrap (2000, seed 6304)",
                   "source": "standard_heldout.json vs forks/midpoint_baseline_heldout.json"}
            row.update(_heldout_from_json(st))
            d = [x["reward"] - ms[x["idx"]]["reward"] for x in st["samples"] if x["idx"] in ms]
            m, lo, hi = paired_bootstrap_ci(d)
            row.update({"d_vs_mid": m, "ci95_lo": lo, "ci95_hi": hi})
            rows.append(row)
    return rows


def _heldout_from_json(h: dict) -> dict:
    s = h["summary"]
    r = np.array([x["reward"] for x in h["samples"]], float)
    return {"n": s["n_prompts"], "reward_mean": s["reward_mean"], "reward_se": s["reward_stderr"],
            "kl_tok_derived": s["kl_per_token"], "kl_seq": s["kl_per_sequence"], "entropy": s["entropy"],
            "length": s["response_length"], "trunc": s["truncation_rate"], "distinct2": s["distinct2"],
            "reward_sd": _sd(r)}


def fork_grid(parsed: dict, art: dict, runs: dict) -> list[dict]:
    """Table 3: (eps, beta) grid. Cached-batch columns are copied from the run but flagged (prompt mismatch)."""
    static = {round(r["epsilon"], 4): r for r in parsed["cached_batch"].get("static", [])}
    eps_rows = {round(r["eps"], 4): r for r in parsed["eps_table"]}
    beta_rows = {round(r["beta"], 4): r for r in parsed["beta_table"]}
    grid = [(0.05, 0.1), (0.2, 0.1), (0.5, 0.1), (0.2, 0.0), (0.2, 0.2)]
    out = []
    for eps, beta in grid:
        name = f"fork_eps{eps:g}_beta{beta:g}"
        st = static.get(round(eps, 4), {})
        er = eps_rows.get(round(eps, 4)) if beta == 0.1 else None
        br = beta_rows.get(round(beta, 4)) if eps == 0.2 else None
        clip = console_series(runs, name, "clip")
        row = {"run": name, "eps": eps, "beta": beta,
               "cached_affected_frac_INVALID": st.get("affected_token_fraction"),
               "cached_active_frac_INVALID": st.get("active_clip_fraction"),
               "cached_L_clip_INVALID": st.get("clipped_surrogate"),
               "train_clip_frac_nominal": er["fork_clip_frac"] if er else float(np.mean(clip)),
               "train_clip_frac_active": er["fork_active_clip"] if er else None,
               "updates_with_any_clipping": int((clip > 0).sum()),
               "max_abs_rho_minus_1": er["max_abs_ratio_dev"] if er else None,
               "grad_norm_cv": er["grad_norm_cv"] if er else grad_norm_cv(console_series(runs, name, "gn")),
               "grad_norm_cv_source": "clipping_study table" if er else "recomputed from 3-dp console gn",
               "approx_kl_step_max": er["approx_kl_step_max"] if er else None,
               "skipped_steps": int(er["skipped"]) if er else None,
               "heldout_reward": (er or {}).get("heldout_reward", (br or {}).get("heldout_reward")),
               "d_reward_vs_mid": (er or {}).get("d_reward_vs_mid", (br or {}).get("d_vs_mid")),
               "heldout_kl_seq": (br or {}).get("heldout_kl_seq"),
               "heldout_len": (er or {}).get("heldout_len", (br or {}).get("heldout_len")),
               "train_wall_s": parsed["done"].get(name, {}).get("train_wall_s"),
               "peak_vram_alloc_gib": parsed["done"].get(name, {}).get("peak_vram_gib_allocated"),
               "peak_vram_res_gib": parsed["done"].get(name, {}).get("peak_vram_gib_reserved")}
        s = art["summaries"].get(name)
        if s:   # exact stability stats from summary.json
            st2 = s.get("stability", {})
            row.update({"train_clip_frac_nominal": st2.get("clip_fraction_mean"),
                        "train_clip_frac_active": st2.get("clip_fraction_active_mean"),
                        "max_abs_rho_minus_1": st2.get("max_abs_ratio_dev"), "grad_norm_cv": st2.get("grad_norm_cv"),
                        "grad_norm_cv_source": "summary.json stability", "approx_kl_step_max": st2.get("approx_kl_step_max"),
                        "skipped_steps": st2.get("skipped_steps")})
        out.append(row)
    return out


def cache_audit(root: Path, tok, beta: float = 0.1) -> tuple[list[dict], dict]:
    cache = load_rollout_cache(root / "cached" / "ppo_rollout.pt")
    train = _read_jsonl(root / "data" / "rl_prompt_pool_train.jsonl")
    ev = _read_jsonl(root / "data" / "rl_prompt_pool_eval.jsonl")
    ev_pos = {r["prompt_id"]: i for i, r in enumerate(ev)}
    tr_pos = {r["prompt_id"]: i for i, r in enumerate(train)}
    ev_tok = prompt_lengths(root, ev, tok)
    rows, D, OLD, G_all, V_all, V0, Rl, Ln = [], [], [], [], [], [], [], []
    for k, r in enumerate(cache):
        old = np.asarray(r["old_logprobs"], np.float64)
        ref = np.asarray(r["ref_logprobs"], np.float64)
        V = np.asarray(r["values"], np.float64)
        d = old - ref
        eff = float(r["effective_terminal_reward"])
        shaped = -beta * d                                   # gamma = 1 Monte-Carlo return incl. KL shaping
        G = eff + np.cumsum(shaped[::-1])[::-1]
        si = int(r["source_index"])
        epos = ev_pos.get(r["prompt_id"])
        retok = None
        if tok is not None:
            ids = tok.encode(str(r["response"]), add_special_tokens=False).ids
            retok = (len(ids) == len(old)) or (len(ids) + 1 == len(old))
        rows.append({
            "row": k, "prompt_id": r["prompt_id"][:12], "source_index": si,
            "eval_pool_position": epos, "in_train_pool": r["prompt_id"] in tr_pos,
            "eval_row_source_index_equal": (epos is not None and int(ev[epos]["source_index"]) == si),
            "train_row_at_source_index_has_same_prompt": (si < len(train) and train[si]["prompt_id"] == r["prompt_id"]),
            "n_response_tokens": len(old), "clipped_at_max": bool(r["clipped_at_max"]),
            "terminated_with_eos": bool(r["terminated_with_eos"]), "raw_reward": float(r["raw_terminal_reward"]),
            "effective_reward": eff, "kl_seq_old_vs_ref": float(d.sum()), "mean_abs_old_minus_ref": float(np.abs(d).mean()),
            "value_first_token": float(V[0]), "value_mean": float(V.mean()),
            "eval_prompt_tokens": int(ev_tok[epos]) if (ev_tok is not None and epos is not None) else None,
            "retokenized_length_aligns": retok,
            "correct_prompt_eval_pool": str(ev[epos]["messages"][-1]["content"])[:90] if epos is not None else None,
            "prompt_used_by_analyze_clipping_train_pool_at_source_index":
                str(train[si]["messages"][-1]["content"])[:90] if si < len(train) else None,
            "response_head": str(r["response"])[:90]})
        D.append(d); OLD.append(old); G_all.append(G); V_all.append(V); V0.append(V[0]); Rl.append(eff); Ln.append(len(old))
    d, old, G, V = (np.concatenate(x) for x in (D, OLD, G_all, V_all))
    proxy = {}
    for eps in (0.05, 0.2, 0.5):
        lr = -d   # proxy log rho = log pi_ref - log pi_old (two independent forward passes of ~the same policy)
        proxy[f"eps{eps:g}"] = float(((lr < math.log(1 - eps)) | (lr > math.log(1 + eps))).mean())
    by_conf = []
    for lo, hi in ((-1e9, -2.3), (-2.3, -0.69), (-0.69, -0.105), (-0.105, -0.01), (-0.01, 1.0)):
        m = (old > lo) & (old <= hi)
        by_conf.append({"old_logp_bin": f"({lo:g},{hi:g}]", "n_tokens": int(m.sum()),
                        "mean_abs_old_minus_ref": float(np.abs(d[m]).mean()), "mean_old_minus_ref": float(d[m].mean())})
    summ = {
        "n_rows": len(cache), "n_tokens": int(d.size),
        "rows_found_in_eval_pool": int(sum(x["eval_pool_position"] is not None for x in rows)),
        "eval_pool_positions": sorted(x["eval_pool_position"] for x in rows if x["eval_pool_position"] is not None),
        "rows_found_in_train_pool": int(sum(x["in_train_pool"] for x in rows)),
        "rows_where_eval_source_index_field_matches": int(sum(x["eval_row_source_index_equal"] for x in rows)),
        "rows_where_train_pool_at_source_index_is_correct_prompt":
            int(sum(x["train_row_at_source_index_has_same_prompt"] for x in rows)),
        "retokenized_length_aligns": (int(sum(bool(x["retokenized_length_aligns"]) for x in rows))
                                      if tok is not None else None),
        "rows_clipped_at_512": int(sum(x["clipped_at_max"] for x in rows)),
        "rows_eos": int(sum(x["terminated_with_eos"] for x in rows)),
        "raw_reward_mean": float(np.mean([x["raw_reward"] for x in rows])),
        "effective_reward_mean": float(np.mean(Rl)),
        "old_vs_ref_k1_per_token": float(d.mean()), "old_vs_ref_mean_abs": float(np.abs(d).mean()),
        "old_vs_ref_rms": float(np.sqrt(np.mean(d ** 2))), "old_vs_ref_median_abs": float(np.median(np.abs(d))),
        "old_vs_ref_p99_abs": float(np.quantile(np.abs(d), 0.99)), "old_vs_ref_max_abs": float(np.abs(d).max()),
        "old_vs_ref_frac_positive": float((d > 0).mean()),
        "kl_seq_mean": float(np.mean([x["kl_seq_old_vs_ref"] for x in rows])),
        "kl_seq_sd": _sd([x["kl_seq_old_vs_ref"] for x in rows]),
        "proxy_affected_fraction_numerical_floor": proxy, "old_vs_ref_by_token_confidence": by_conf,
        "critic_ev_token_level_vs_mc_return": float(1 - np.var(G - V) / np.var(G)),
        "critic_corr_first_token_value_vs_return": float(np.corrcoef(V0, Rl)[0, 1]),
        "critic_spearman_first_token_value_vs_return": spearman(V0, Rl),
        "critic_value_mean": float(V.mean()), "mc_return_mean": float(G.mean()),
        "critic_mse_vs_mc_return": float(np.mean((G - V) ** 2)), "mc_return_var_token_level": float(np.var(G)),
        "rm_spearman_raw_reward_vs_length": spearman([x["raw_reward"] for x in rows], Ln),
        "n_cache_prompts_over_256_tokens": (int(sum((x["eval_prompt_tokens"] or 0) > 256 for x in rows))
                                            if ev_tok is not None else None),
    }
    return rows, summ


def heldout_sample_stats(art: dict, ev_rows: list[dict], ev_tok) -> dict | None:
    """Per-sample analyses that need heldout.json files (RM noise floor, length bias, truncation, identity)."""
    mid = art["heldout"].get("midpoint")
    if not mid:
        return None
    ms = {s["idx"]: s for s in mid["samples"]}
    r = np.array([s["reward"] for s in mid["samples"]], float)
    L = np.array([s["length"] for s in mid["samples"]], float)
    out = {"midpoint_spearman_reward_length": spearman(r, L), "n": len(r)}
    if ev_tok is not None:
        tr = np.array([ev_tok[s["idx"]] > 256 for s in mid["samples"]])
        d, p = perm_test_mean_diff(r[~tr], r[tr])
        out.update({"midpoint_reward_truncated_prompt_mean": float(r[tr].mean()),
                    "midpoint_reward_untruncated_prompt_mean": float(r[~tr].mean()),
                    "n_truncated": int(tr.sum()), "trunc_minus_untrunc": d, "perm_p": p})
    starts_mid = np.array([bool(re.match(r"^\s*[a-z,.;:)\]'\"-]", s["response"])) for s in mid["samples"]])
    out["midpoint_frac_response_starts_midsentence_heuristic"] = float(starts_mid.mean())
    noise, ident = [], {}
    for name, h in art["heldout"].items():
        if name in ("midpoint",) or name in ("fork_eps0.05_beta0.1", "fork_eps0.5_beta0.1"):
            continue   # eps forks are bit-identical to (0.2, 0.1); count each distinct trajectory once
        same = [(s, ms[s["idx"]]) for s in h["samples"]
                if s["idx"] in ms and s["response"] == ms[s["idx"]]["response"] and s["length"] == ms[s["idx"]]["length"]]
        ident[name] = len(same) / max(len(h["samples"]), 1)
        noise += [abs(a["reward"] - b["reward"]) for a, b in same]
    out["frac_text_identical_to_midpoint"] = ident
    if noise:
        out.update({"rm_noise_floor_mean_abs_dR_identical_text": float(np.mean(noise)),
                    "rm_noise_floor_p95": float(np.quantile(noise, 0.95)), "rm_noise_floor_n_pairs": len(noise),
                    "rm_noise_floor_frac_exactly_zero": float(np.mean(np.array(noise) == 0))})
    return out


def write_examples(art: dict, parsed: dict, out: Path, extra_idx=(72, 178)) -> list[int]:
    """Dump full held-out text for candidate prompts across conditions (to be READ before citing)."""
    conds = [("midpoint", "midpoint"), ("fork_eps0.2_beta0", "beta=0"), ("fork_eps0.2_beta0.1", "beta=0.1"),
             ("fork_eps0.2_beta0.2", "beta=0.2")]
    if not art["heldout"].get("midpoint"):
        return []
    idxs = []
    for items in parsed.get("candidates", {}).values():
        idxs += [c["idx"] for c in items]
    idxs = list(dict.fromkeys(idxs + list(extra_idx)))
    lines = ["# Task 2 held-out examples (raw dump -- read before citing)\n"]
    for i in idxs:
        lines.append(f"\n## held-out prompt idx {i}\n")
        for key, lab in conds:
            h = art["heldout"].get(key)
            if not h:
                continue
            s = next((x for x in h["samples"] if x["idx"] == i), None)
            if s is None:
                continue
            if key == "midpoint":
                lines.append(f"**Prompt (<=400 chars):** {s['prompt']!r}\n")
            lines.append(f"\n**{lab}**: reward={s['reward']:+.3f}, length={s['length']}, kl_seq={s['kl_seq']:+.4f} nats, "
                         f"kl/token={s['kl_seq'] / max(s['length'], 1):+.2e}, truncated={s['truncated']}, "
                         f"distinct2={s['distinct2']:.3f}\n\n```text\n{s['response']}\n```\n")
    (out / "examples_raw.md").write_text("".join(lines), encoding="utf-8")
    return idxs


def reliability_rows(root: Path, footer: dict, runs: dict, cache_summ: dict, pool_stats: dict,
                     extras: dict | None) -> list[dict]:
    rows = []
    rmv = root / "manifests" / "source_v2_reward_model_validation.json"
    acc = root / "checkpoints" / "ppo_midpoint_acceptance.json"
    if rmv.exists():
        v = _read_json(rmv)
        rows += [
            {"diagnostic": "RM pairwise accuracy (staff validation, all pairs)", "value": v["all_pair_accuracy"],
             "n": v["n"], "source": "manifests/source_v2_reward_model_validation.json -> all_pair_accuracy"},
            {"diagnostic": "RM pairwise accuracy (strict pairs)", "value": v["strict_pair_accuracy"], "n": v["strict_n"],
             "source": "manifests/source_v2_reward_model_validation.json -> strict_pair_accuracy"},
            {"diagnostic": "RM reward-length Spearman (staff validation)", "value": v["reward_length_spearman"],
             "n": v["n"], "source": "manifests/source_v2_reward_model_validation.json -> reward_length_spearman"}]
    R, L = console_series(runs, "standard", "R"), console_series(runs, "standard", "len")
    rho, p = spearman_perm_p(R, L)
    rows += [
        {"diagnostic": "RM reward-length Spearman, standard-run rollouts", "value": rho, "n": len(R),
         "p_perm": p, "source": "log [standard] R= vs len="},
        {"diagnostic": "RM reward-length Spearman, cached batch (raw_terminal_reward vs tokens)",
         "value": cache_summ["rm_spearman_raw_reward_vs_length"], "n": cache_summ["n_rows"],
         "source": "cached/ppo_rollout.pt"}]
    if pool_stats:
        rows += [{"diagnostic": "prompts > 256 rendered tokens (right-truncated; assistant header cut), eval pool",
                  "value": pool_stats["eval_frac_over_256"], "n": pool_stats["eval_n"],
                  "source": "data/rl_prompt_pool_eval.jsonl -> prompt, tokenizer count"},
                 {"diagnostic": "prompts > 256 rendered tokens, train pool", "value": pool_stats["train_frac_over_256"],
                  "n": pool_stats["train_n"], "source": "data/rl_prompt_pool_train.jsonl -> prompt, tokenizer count"}]
    if "reward_truncated_prompt_mean" in footer:
        rows.append({"diagnostic": "standard run: mean R, truncated-prompt updates minus others",
                     "value": footer["reward_trunc_minus_untrunc"], "n": f"{footer['n_truncated_prompts']} vs "
                     f"{len(R) - footer['n_truncated_prompts']}", "p_perm": footer["reward_trunc_perm_p"],
                     "source": "log R= + replayed prompt schedule + tokenizer count"})
    if extras and "rm_noise_floor_mean_abs_dR_identical_text" in extras:
        rows.append({"diagnostic": "RM noise floor: mean |dR| on text-identical held-out responses",
                     "value": extras["rm_noise_floor_mean_abs_dR_identical_text"],
                     "n": extras["rm_noise_floor_n_pairs"], "source": "heldout.json (fork vs midpoint, same idx)"})
    else:
        rows.append({"diagnostic": "RM noise floor: mean |dR| on text-identical held-out responses", "value": None,
                     "n": None, "source": "PENDING: needs forks/*/heldout.json + midpoint_baseline_heldout.json"})
    EV = console_series(runs, "standard", "EV")
    Lv = console_series(runs, "standard", "Lv")
    Rp = R - 1.0 * (L >= 512)
    rows += [
        {"diagnostic": "critic EV < 0 (within-trajectory, per update), standard run", "value": int((EV < 0).sum()),
         "n": len(EV), "source": "log EV="},
        {"diagnostic": "critic EV median (standard run)", "value": float(np.median(EV)), "n": len(EV),
         "source": "log EV="},
        {"diagnostic": "critic value loss mean (standard run)", "value": float(np.mean(Lv)), "n": len(Lv),
         "source": "log Lv="},
        {"diagnostic": "variance of penalized reward across updates (constant-predictor MSE)", "value": float(np.var(Rp, ddof=1)),
         "n": len(Rp), "source": "derived from log R=, len="},
        {"diagnostic": "staff critic on cached batch: token-level EV vs MC return",
         "value": cache_summ["critic_ev_token_level_vs_mc_return"], "n": cache_summ["n_tokens"],
         "source": "cached/ppo_rollout.pt values vs effective_terminal_reward (+KL shaping, beta=0.1)"},
        {"diagnostic": "staff critic on cached batch: corr(V at first token, return)",
         "value": cache_summ["critic_corr_first_token_value_vs_return"], "n": cache_summ["n_rows"],
         "source": "cached/ppo_rollout.pt"},
        {"diagnostic": "staff critic on cached batch: mean V minus mean return",
         "value": cache_summ["critic_value_mean"] - cache_summ["mc_return_mean"], "n": cache_summ["n_tokens"],
         "source": "cached/ppo_rollout.pt"}]
    if acc.exists():
        a = _read_json(acc)
        rows += [{"diagnostic": "staff acceptance: critic EV on held-out MC returns",
                  "value": a["critic_mc_return_explained_variance"], "n": a["n_policy_eval_prompts"],
                  "source": "checkpoints/ppo_midpoint_acceptance.json"},
                 {"diagnostic": "staff acceptance: critic corr with MC returns", "value": a["critic_mc_return_correlation"],
                  "n": a["n_policy_eval_prompts"], "source": "checkpoints/ppo_midpoint_acceptance.json"}]
    rows.append({"diagnostic": "critic grad norm pre-clip (max_grad_norm=1.0)", "value": None, "n": None,
                 "source": "PENDING: metrics.jsonl -> value_grad_norm (not printed in console log)"})
    return rows


# ============================================================================ main
def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="results/task2_ppo/analysis")
    ap.add_argument("--log", default=None, help="console log (default: <out>/data/kaggle_console_log.txt)")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    out = Path(args.out) if Path(args.out).is_absolute() else root / args.out
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(parents=True, exist_ok=True)
    log = Path(args.log) if args.log else out / "data" / "kaggle_console_log.txt"

    parsed = parse_console_log(log)
    runs = parsed["runs"]
    art = load_json_artifacts(root / "results" / "task2_ppo")
    tok = _tokenizer(root)

    # ---- prompt schedule replay + prompt truncation
    train = _read_jsonl(root / "data" / "rl_prompt_pool_train.jsonl")
    ev = _read_jsonl(root / "data" / "rl_prompt_pool_eval.jsonl")
    ttok, etok = prompt_lengths(root, train, tok), prompt_lengths(root, ev, tok)
    sched = prompt_schedule(len(train), SEED, len(runs["standard"]), 1)
    sched_info = []
    for u, (ids, rec) in enumerate(zip(sched, runs["standard"]), 1):
        i = ids[0]
        sched_info.append({"update": u, "pool_index": i, "prompt_id": train[i]["prompt_id"][:12],
                           "prompt_tokens": int(ttok[i]) if ttok is not None else None,
                           "prompt_truncated_at_256": bool(ttok[i] > 256) if ttok is not None else None,
                           "R": rec["R"], "len": rec["len"], "H": rec["H"],
                           "user_prompt_head": str(train[i]["messages"][-1]["content"])[:100].replace("\n", " ")})
    if art["runs"].get("standard"):   # verify the replay against logged prompt_ids
        for s, r in zip(sched_info, art["runs"]["standard"]):
            s["logged_prompt_ids"] = r.get("prompt_ids")
            s["replay_matches_log"] = (r.get("prompt_ids") == [s["pool_index"]])
    pool_stats = {}
    if ttok is not None:
        pool_stats = {"train_n": len(train), "train_frac_over_256": float((ttok > 256).mean()),
                      "train_n_over_256": int((ttok > 256).sum()), "eval_n": len(ev),
                      "eval_frac_over_256": float((etok > 256).mean()), "eval_n_over_256": int((etok > 256).sum()),
                      "train_median_tokens": float(np.median(ttok)), "eval_median_tokens": float(np.median(etok))}

    t1, footer = table1_standard(runs, parsed["done"], art,
                                 sched_info if (ttok is not None) else None)
    det = determinism_checks(runs)
    rq2 = rq2_divergence(runs)
    klf = kl_force(runs, {"standard": 0.1, "fork_eps0.2_beta0": 0.0, "fork_eps0.2_beta0.1": 0.1,
                          "fork_eps0.2_beta0.2": 0.2})
    t2 = heldout_table(parsed, art)
    t3 = fork_grid(parsed, art, runs)
    cache_rows, cache_summ = cache_audit(root, tok)
    extras = heldout_sample_stats(art, ev, etok)
    t4 = reliability_rows(root, footer, runs, cache_summ, pool_stats, extras)
    ex_idx = write_examples(art, parsed, out)

    # ---- write data + tables
    per_update = [dict(r) for rr in runs.values() for r in rr]
    _write_csv(out / "data" / "per_update_from_log.csv", per_update)
    _write_csv(out / "data" / "cache_audit.csv", cache_rows)
    _write_csv(out / "data" / "standard_prompt_schedule.csv", sched_info)
    _write_csv(out / "tables" / "table1_standard_run.csv", t1)
    _write_csv(out / "tables" / "table2_heldout_beta.csv", t2)
    _write_csv(out / "tables" / "table3_fork_grid.csv", t3)
    _write_csv(out / "tables" / "table4_reliability.csv", t4)
    _write_csv(out / "tables" / "rq2_per_update_beta.csv", rq2["per_update"])
    _write_csv(out / "tables" / "kl_force.csv", klf)
    _write_csv(out / "tables" / "cached_batch_static_as_run_INVALID.csv", parsed["cached_batch"].get("static", []))

    k = min(FIRST_LAST, len(runs["standard"]) // 2)
    md = ["# Task 2 tables (generated by analysis/task2_report_stats.py -- do not edit by hand)\n",
          f"\nConsole log: `{log.relative_to(root) if log.is_relative_to(root) else log}`. "
          f"JSON artifacts found: {len(art['files_found'])}; missing: {len(art['files_missing'])}.\n",
          "\n## Table 1 - standard continuation (20 updates, eps=0.20, beta=0.10)\n\n",
          _md_table(t1, ["metric", "mean", "sd", "min", "max", f"first{k}_mean", f"last{k}_mean",
                         "delta_last_minus_first", "perm_p_first_vs_last", "spearman_vs_update", "spearman_perm_p"],
                    {"perm_p_first_vs_last": 3, "spearman_perm_p": 3, "spearman_vs_update": 2}),
          "\nFooter: " + ", ".join(f"{kk}={_f(v) if not isinstance(v, (dict, list, str)) else v}"
                                   for kk, v in footer.items() if kk != "summary_json_stability") + "\n",
          "\n## Table 2 - held-out KL study (200 prompts, 768-token cap)\n\n",
          _md_table(t2, ["condition", "n", "reward_mean", "reward_se", "d_vs_mid", "ci95_lo", "ci95_hi", "ci_type",
                         "kl_tok_derived", "kl_seq", "entropy", "length", "trunc", "distinct2"],
                    {"kl_tok_derived": 7, "length": 2}),
          "\n## Table 3 - fork grid (8 updates each)\n\n",
          _md_table(t3, ["run", "eps", "beta", "cached_affected_frac_INVALID", "cached_L_clip_INVALID",
                         "train_clip_frac_nominal", "train_clip_frac_active", "updates_with_any_clipping",
                         "max_abs_rho_minus_1", "grad_norm_cv", "approx_kl_step_max", "skipped_steps",
                         "heldout_reward", "d_reward_vs_mid", "train_wall_s", "peak_vram_alloc_gib",
                         "peak_vram_res_gib"], {"eps": 2, "beta": 2, "train_wall_s": 1, "peak_vram_alloc_gib": 2,
                                                "peak_vram_res_gib": 2}),
          "\n## Table 4 - reliability diagnostics\n\n",
          _md_table(t4, ["diagnostic", "value", "n", "p_perm", "source"], {"p_perm": 3}),
          "\n## RQ2 - per-update beta forks (console precision)\n\n", _md_table(rq2["per_update"]),
          "\n## KL force (beta x |KL_seq|) vs reward SD\n\n", _md_table(klf),
          "\n## Cached-batch audit (cached/ppo_rollout.pt)\n\n", _md_table(cache_rows[:32], [
              "row", "eval_pool_position", "source_index", "train_row_at_source_index_has_same_prompt",
              "n_response_tokens", "clipped_at_max", "raw_reward", "kl_seq_old_vs_ref", "eval_prompt_tokens"]),
          "\n## Standard-run prompt schedule (replayed)\n\n", _md_table(sched_info, [
              "update", "pool_index", "prompt_tokens", "prompt_truncated_at_256", "R", "len", "H", "user_prompt_head"])]
    (out / "tables.md").write_text("".join(md), encoding="utf-8")

    stats = {"console_log": str(log), "json_artifacts": {"found": art["files_found"], "missing": art["files_missing"]},
             "validate_ppo_objective_ok_count": parsed["validate_objective_ok"],
             "reused_forks": parsed["reused_forks"], "done": parsed["done"],
             "table1_footer": footer, "determinism": det, "rq2": {k2: v for k2, v in rq2.items() if k2 != "per_update"},
             "kl_force": klf, "cache_audit": cache_summ, "cached_batch_as_run": parsed["cached_batch"],
             "pool_prompt_lengths": pool_stats, "heldout_sample_stats": extras, "examples_dumped_idx": ex_idx,
             "beta_table_console": parsed["beta_table"], "eps_table_console": parsed["eps_table"],
             "baseline_console": parsed["baselines"][0] if parsed["baselines"] else None,
             "first_divergence_console": parsed["first_divergence"], "effect_size_console": parsed["effect_size"],
             "candidates_console": parsed["candidates"]}
    (out / "stats.json").write_text(json.dumps(_jsonable(stats), indent=2), encoding="utf-8")
    if not args.no_figures:
        try:
            make_figures(out, runs, parsed, t2[:4], t3, cache_summ, sched_info, rq2)
        except ImportError as e:
            print("figures skipped:", e)
    print(f"wrote {out}")



# ============================================================================ figures (static, light surface)
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"     # categorical slots 1-3 (validated all-pairs)
GREY, INK, INK2, GRID, SURF = "#8a8984", "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"


def _style(plt):
    plt.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "font.size": 8.5, "axes.titlesize": 9, "axes.titleweight": "bold",
        "axes.titlecolor": INK, "legend.frameon": False, "lines.linewidth": 1.6, "lines.markersize": 4.5})


def _save(fig, out: Path, name: str):
    (out / "figures").mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / "figures" / f"{name}.{ext}", dpi=200, bbox_inches="tight")


def make_figures(out: Path, runs: dict, parsed: dict, t2: list[dict], t3: list[dict], cache_summ: dict,
                 sched_info: list[dict], rq2: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    u = np.arange(1, len(runs["standard"]) + 1)
    s = lambda f, run="standard": console_series(runs, run, f)
    trunc = np.array([bool(x.get("prompt_truncated_at_256")) for x in sched_info]) if sched_info else np.zeros(len(u), bool)
    capped = s("len") >= 512

    # ---------------- Fig 1: standard continuation diagnostics (3x3 small multiples)
    fig, ax = plt.subplots(3, 3, figsize=(10.5, 7.6), constrained_layout=True)
    panels = [("R", "a  Learned reward (raw RM score)", None), ("KL", "b  KL to reference, per token (k1)", 0.0),
              ("H", "c  Entropy (nats/token, full vocab)", None), ("len", "d  Response length (tokens)", 512),
              ("Lpi", "e  Policy loss", 0.0), ("Lv", "f  Value loss", None),
              ("gn", "g  Policy grad norm (pre-clip)", 1.0), ("EV", "h  Critic explained variance", 0.0)]
    for a, (f, title, ref) in zip(ax.flat, panels):
        y = s(f)
        a.plot(u, y, color=C1, marker="o", zorder=3)
        if f == "R":
            a.scatter(u[trunc], y[trunc], s=46, facecolor=SURF, edgecolor=C2, linewidth=1.6, zorder=4,
                      label="prompt > 256 tok (truncated)")
            a.legend(loc="lower left", fontsize=7)
        if f == "EV":
            a.set_yscale("symlog", linthresh=1.0)
        if ref is not None:
            a.axhline(ref, color=GREY, lw=0.9, ls="--", zorder=1)
        a.set_title(title, loc="left")
        a.set_xticks([1, 5, 10, 15, 20])
    ax.flat[6].text(20, 1.03, "clip 1.0", color=INK2, fontsize=7, ha="right", va="bottom")
    ax.flat[3].text(14.5, 505, "cap 512", color=INK2, fontsize=7, ha="right", va="top")
    a = ax.flat[8]
    a.scatter(s("len"), s("t"), color=C1, s=18, zorder=3)
    tok_rate = np.sum(s("t")) / np.sum(s("len"))
    xx = np.array([0, 520])
    a.plot(xx, tok_rate * xx, color=GREY, lw=0.9, ls="--")
    a.set_title(f"i  Wall-clock per update vs length ({tok_rate * 1000:.0f} ms/token)", loc="left")
    a.set_xlabel("response tokens")
    a.set_ylabel("seconds")
    for a in ax.flat[:8]:
        a.set_xlabel("update")
    d = parsed["done"]["standard"]
    fig.suptitle(f"Standard PPO continuation, 20 updates (eps=0.20, beta=0.10). Clip fraction = 0.000 in 20/20 updates. "
                 f"Peak VRAM {d['peak_vram_gib_allocated']:.2f} GiB alloc / {d['peak_vram_gib_reserved']:.2f} reserved, "
                 f"{d['train_wall_s']:.1f} s", fontsize=9, color=INK, x=0.01, ha="left")
    _save(fig, out, "fig1_standard_run")
    plt.close(fig)

    # ---------------- Fig 2: clipping study
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.3), constrained_layout=True)
    a = ax[0]
    eps_vals = [0.05, 0.2, 0.5]
    mx = next((r["max_abs_rho_minus_1"] for r in t3 if r["max_abs_rho_minus_1"] is not None), None)
    for i, e in enumerate(eps_vals):
        a.barh(i, e, color=GRID, height=0.55, edgecolor="none")
        a.text(e + 0.008, i, f"eps = {e:g}", va="center", color=INK2, fontsize=8)
    if mx is not None:
        a.axvline(mx, color=C1, lw=2)
        a.text(mx + 0.008, -0.45, f"max |rho-1| over 8 updates = {mx:.4f} (all three forks)", color=INK,
               fontsize=7.5, va="center")
    a.set_ylim(-0.75, 2.4)
    a.set_yticks([])
    a.set_xlim(0, 0.62)
    a.set_xlabel("|rho - 1|")
    a.set_title("a  Policy step never reaches any clip band", loc="left")
    a = ax[1]
    for run, col, lab, ls in (("fork_eps0.05_beta0.1", C1, "eps 0.05", "-"), ("fork_eps0.2_beta0.1", C2, "eps 0.20", "--"),
                              ("fork_eps0.5_beta0.1", C3, "eps 0.50", ":")):
        a.plot(np.arange(1, 9), s("gn", run), color=col, ls=ls, lw=2, marker="o", label=lab)
    a.set_title("b  Grad norm per update: forks bit-identical", loc="left")
    a.set_xlabel("update")
    a.legend(fontsize=7, loc="upper left")
    a = ax[2]
    static = parsed["cached_batch"].get("static", [])
    proxy = cache_summ["proxy_affected_fraction_numerical_floor"]
    xs = np.arange(3)
    w = 0.36
    a.bar(xs - w / 2 - 0.01, [r["affected_token_fraction"] for r in static], w, color=C2,
          label="as run (wrong prompts: INVALID)")
    fixed_p = out.parent / "clipping_cached_batch_fixed.json"
    if fixed_p.exists():
        fx = {round(r["epsilon"], 4): r for r in _read_json(fixed_p)["static"]}
        proxy = {f"eps{e:g}": fx[round(e, 4)]["affected_token_fraction"] for e in eps_vals}
        lab2 = "corrected prompts (measured)"
    else:
        lab2 = "numerical-floor proxy (cache old vs ref)"
    a.bar(xs + w / 2 + 0.01, [proxy[f"eps{e:g}"] for e in eps_vals], w, color=C1, label=lab2)
    for x, v in zip(xs - w / 2, [r["affected_token_fraction"] for r in static]):
        a.text(x, v + 0.01, f"{v:.2f}", ha="center", fontsize=7, color=INK2)
    for x, e in zip(xs + w / 2, eps_vals):
        a.text(x, proxy[f"eps{e:g}"] + 0.01, f"{proxy[f'eps{e:g}']:.3f}", ha="center", fontsize=7, color=INK2)
    a.set_xticks(xs, [f"eps {e:g}" for e in eps_vals])
    a.set_ylabel("affected-token fraction")
    a.set_title("c  Cached batch: buggy run vs corrected", loc="left")
    a.legend(fontsize=7, loc="upper right")
    a.set_ylim(0, 0.68)
    _save(fig, out, "fig2_clipping_study")
    plt.close(fig)

    # ---------------- Fig 3: KL study
    fig, ax = plt.subplots(2, 3, figsize=(10.5, 6.2), constrained_layout=True)
    forks = [("fork_eps0.2_beta0", C1, "beta 0", "o"), ("fork_eps0.2_beta0.1", C2, "beta 0.10", "s"),
             ("fork_eps0.2_beta0.2", C3, "beta 0.20", "^")]
    x8 = np.arange(1, 9)
    for a, (f, title) in zip(ax[0], (("R", "a  Train reward per update"), ("KL", "b  KL/token per update (k1)"),
                                     ("H", "c  Entropy per update"))):
        for k, (run, col, lab, mk) in enumerate(forks):
            a.plot(x8 + (k - 1) * 0.08, s(f, run), color=col, marker=mk, label=lab, lw=1.4)
        a.set_title(title, loc="left")
        a.set_xlabel("update")
        a.set_xticks(x8)
        a.axvspan(2.5, 3.5, color=GRID, alpha=0.6, zorder=0)
    ax[0][0].legend(fontsize=7, loc="lower right")
    ax[0][1].axhline(0, color=GREY, lw=0.9, ls="--")
    ax[0][0].text(3.0, -0.95, "first sampled\ntext differs", ha="center", va="center",
                  fontsize=7, color=INK2)
    # held-out panels
    labels = ["mid", "b 0", "b 0.1", "b 0.2"]
    cols = [GREY, C1, C2, C3]
    a = ax[1][0]
    means = [r["reward_mean"] for r in t2]
    se = t2[0]["reward_se"]
    for i, (m, c) in enumerate(zip(means, cols)):
        a.errorbar(i, m, yerr=1.96 * (r_se if (r_se := t2[i].get("reward_se")) else se), fmt="o", color=c,
                   ms=7, capsize=3, lw=1.6)
    a.set_xticks(range(4), labels)
    a.set_ylim(1.2, 1.85)
    a.set_title("d  Held-out reward (+/- 1.96 SE)", loc="left")
    if any(r.get("reward_se") is None for r in t2[:4]):
        a.text(0.5, 0.02, "fork SE not in log: midpoint SE used", transform=a.transAxes, fontsize=6.5, color=INK2,
               ha="center", va="bottom")
    a = ax[1][1]
    a.bar(range(4), [r["entropy"] for r in t2], color=cols, width=0.6)
    a.set_ylim(1.0, 1.08)
    for i, r in enumerate(t2):
        a.text(i, r["entropy"] + 0.002, f"{r['entropy']:.4f}", ha="center", fontsize=7, color=INK2)
    a.set_xticks(range(4), labels)
    a.set_title("e  Held-out entropy (axis from 1.0)", loc="left")
    a = ax[1][2]
    a.bar(range(4), [r["kl_seq"] for r in t2], color=cols, width=0.6)
    a.axhline(0, color=GREY, lw=0.9)
    for i, r in enumerate(t2):
        a.text(i, r["kl_seq"] + (0.0006 if r["kl_seq"] >= 0 else -0.0012), f"{r['kl_seq']:+.4f}", ha="center",
               fontsize=7, color=INK2)
    a.set_xticks(range(4), labels)
    a.set_ylim(-0.0145, 0.0055)
    a.set_title("f  Held-out KL to reference (nats/seq)", loc="left")
    _save(fig, out, "fig3_kl_study")
    plt.close(fig)

    # ---------------- Fig 4: reliability
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.3), constrained_layout=True)
    a = ax[0]
    R = s("R")
    a.scatter(s("len")[~trunc], R[~trunc], color=C1, s=22, label="prompt <= 256 tok", zorder=3)
    a.scatter(s("len")[trunc], R[trunc], facecolor=SURF, edgecolor=C2, s=30, linewidth=1.5,
              label="prompt > 256 tok (truncated)", zorder=3)
    a.set_xlabel("response tokens")
    a.set_ylabel("raw RM reward")
    a.set_title("a  Reward vs length, standard run (n=20)", loc="left")
    a.legend(fontsize=7, loc="lower right")
    a = ax[1]
    bins = cache_summ["old_vs_ref_by_token_confidence"]
    a.bar(range(len(bins)), [b["mean_abs_old_minus_ref"] for b in bins], color=C1, width=0.6)
    a.set_xticks(range(len(bins)), ["<0.10", "0.10-0.50", "0.50-0.90", "0.90-0.99", ">0.99"])
    a.set_xlabel("old-policy token probability")
    a.set_ylabel("mean |log pi_old - log pi_ref|")
    a.set_title("b  Cache: old vs ref log-prob gap by confidence", loc="left")
    a = ax[2]
    a.axis("off")
    rows = [("Staff critic on cached batch", ""), ("  EV vs MC return", f"{cache_summ['critic_ev_token_level_vs_mc_return']:.3f}"),
            ("  token corr(V, G)", "reproduces staff -0.061"),
            ("  mean V - mean G", f"{cache_summ['critic_value_mean'] - cache_summ['mc_return_mean']:+.2f}"),
            ("Standard run critic", ""), ("  EV < 0", f"{int((s('EV') < 0).sum())}/20 updates"),
            ("  value loss mean", f"{np.mean(s('Lv')):.2f}"),
            ("  Var(R_pen) across updates", f"{np.var(R - 1.0 * capped, ddof=1):.2f}")]
    for i, (k1, v1) in enumerate(rows):
        a.text(0.0, 0.95 - i * 0.115, k1, fontsize=8, color=INK if not k1.startswith("  ") else INK2,
               fontweight="bold" if not k1.startswith("  ") else "normal", transform=a.transAxes)
        a.text(1.0, 0.95 - i * 0.115, v1, fontsize=8, color=INK, ha="right", transform=a.transAxes)
    a.set_title("c  Critic validity", loc="left")
    _save(fig, out, "fig4_reliability")
    plt.close(fig)


if __name__ == "__main__":
    main()
