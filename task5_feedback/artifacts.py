"""Task 5 artifact layout, I/O and small-sample statistics. Torch-free so aggregation runs off-GPU."""
from __future__ import annotations

import hashlib
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from common.data import repo_path

POLICIES = ("sft", "rlvr", "rlaif")
REFERENCE = "sft"
DATASETS = ("gsm", "transfer")
DATASET_LABEL = {"gsm": "GSM8K", "transfer": "SVAMP"}
Z95 = 1.959963984540054
_TASK = "task5_feedback"


def _mk(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def results_dir(cfg) -> Path:
    return _mk(repo_path(cfg.get("results_dir", "results")) / _TASK)


def outputs_dir(cfg) -> Path:
    return _mk(repo_path(cfg.get("outputs_dir", "outputs")) / _TASK)


def generations_path(cfg, dataset: str, policy: str) -> Path:
    return outputs_dir(cfg) / f"generations_{dataset}_{policy}.jsonl"


def judged_path(cfg, dataset: str) -> Path:
    return outputs_dir(cfg) / f"judged_{dataset}_vs_{REFERENCE}.jsonl"


def judge_cache_path(cfg, tag: str) -> Path:
    # one cache per process-level job: PairwiseAIJudge rewrites its whole cache file on every call, so two
    # concurrent processes sharing a file would clobber each other's entries
    return outputs_dir(cfg) / f"judge_cache_{tag}.json"


def generation_manifest_path(cfg, dataset: str) -> Path:
    return results_dir(cfg) / f"generation_manifest_{dataset}.json"


def judge_manifest_path(cfg, tag: str) -> Path:
    return results_dir(cfg) / f"judge_manifest_{tag}.json"


def summary_path(cfg, name: str) -> Path:
    return results_dir(cfg) / f"{name}_summary.json"


def diagnostic_pairs_path(cfg) -> Path:
    return outputs_dir(cfg) / "diagnostic_pairs.jsonl"


def diagnostic_responses_path(cfg) -> Path:
    return outputs_dir(cfg) / "diagnostic_responses.jsonl"


def pair_digest(a: str, b: str) -> str:
    """Binds a stored judgment to the exact response texts it was made on (stale-judgment guard)."""
    return hashlib.sha1(f"{a}\x00{b}".encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------------------- io
def _plain(o):
    if isinstance(o, dict):
        return {str(k): _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if o is None or o is pd.NA:
        return None
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(o) else None  # strict JSON: NaN/inf -> null
    return o


def save_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_plain(obj), indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def release_sha256(rel: str | Path) -> str | None:
    """Pinned course-release hash from manifests/sha256.json; None for files the release does not cover."""
    p = repo_path("manifests/sha256.json")
    if not p.exists():
        return None
    return load_json(p).get(Path(rel).as_posix())


def verified_sha256(rel: str | Path) -> str:
    h = sha256_file(repo_path(rel))
    want = release_sha256(rel)
    if want is not None and h != want:
        raise RuntimeError(f"{rel}: sha256 {h[:12]} != release {want[:12]}; course assets must be unmodified")
    return h


def to_markdown(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    df = df.reset_index() if not isinstance(df.index, pd.RangeIndex) else df

    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "—" if not math.isfinite(v) else floatfmt.format(v)
        return "—" if v is None else str(v)

    head = "| " + " | ".join(map(str, df.columns)) + " |"
    rule = "|" + "---|" * len(df.columns)
    return "\n".join([head, rule] + ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)])


# --------------------------------------------------------------------------------------- statistics
def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    p, z2 = k / n, z * z
    c = (p + z2 / (2 * n)) / (1 + z2 / n)
    h = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return c - h, c + h


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)


def ratio_bootstrap(num, den, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Percentile CI of sum(num)/sum(den) under resampling of clusters (rows) with replacement.

    num/den: [C] or [C, M] per-cluster numerators/denominators. Ratio-of-sums keeps clusters with several pairs
    (diagnostic problems) correctly weighted; a per-example mean is num=x, den=1."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    squeeze = num.ndim == 1
    if squeeze:
        num, den = num[:, None], den[:, None]
    if num.shape[0] == 0:
        nan = np.full(num.shape[1], np.nan)
        return (nan[0], nan[0]) if squeeze else (nan, nan)
    idx = np.random.default_rng(seed).integers(0, num.shape[0], size=(n_boot, num.shape[0]))
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        stat = num[idx].sum(1) / den[idx].sum(1)
        lo, hi = np.nanquantile(stat, [alpha / 2, 1 - alpha / 2], axis=0)
    return (float(lo[0]), float(hi[0])) if squeeze else (lo, hi)


def dispersion(x) -> dict:
    x = np.asarray(x, float)
    if x.size == 0:
        return dict.fromkeys(("mean", "std", "median", "q25", "q75", "iqr"), math.nan)
    q25, q50, q75 = np.percentile(x, [25, 50, 75])
    return {"mean": x.mean(), "std": x.std(ddof=1) if x.size > 1 else math.nan,
            "median": q50, "q25": q25, "q75": q75, "iqr": q75 - q25}


def cohen_kappa(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    if a.size == 0:
        return math.nan
    po = float((a == b).mean())
    pe = float(sum((a == c).mean() * (b == c).mean() for c in np.union1d(a, b)))
    return (po - pe) / (1 - pe) if pe < 1 else math.nan
