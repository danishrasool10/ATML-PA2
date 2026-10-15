"""Task 4 artifact layout and loaders. Torch-free so aggregation/audit scripts run off-GPU."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from common.data import read_jsonl, repo_path

POLICIES = ("sft", "dpo", "ppo", "grpo")
LABELS = ("SAFE_ANSWER", "OVER_REFUSAL", "JUSTIFIED_REFUSAL", "UNSAFE_COMPLIANCE", "AMBIGUOUS")
# label -> (prompt class the label presupposes, response behavior); AMBIGUOUS commits to neither axis
LABEL_AXES = {
    "SAFE_ANSWER": ("SAFE", "comply"),
    "OVER_REFUSAL": ("SAFE", "refuse"),
    "JUSTIFIED_REFUSAL": ("UNSAFE", "refuse"),
    "UNSAFE_COMPLIANCE": ("UNSAFE", "comply"),
    "AMBIGUOUS": (None, None),
}


def out_dir(cfg) -> Path:
    p = repo_path(cfg["results_dir"]) / "task4_safety"
    p.mkdir(parents=True, exist_ok=True)
    return p


def audit_dir(cfg) -> Path:
    p = out_dir(cfg) / "audit"
    p.mkdir(parents=True, exist_ok=True)
    return p


def generated_path(cfg, policy: str) -> Path:
    return out_dir(cfg) / f"generated_{policy}.jsonl"


def judged_path(cfg, policy: str) -> Path:
    return out_dir(cfg) / f"judged_{policy}.jsonl"


def load_rows(cfg, kind: str, policies=POLICIES) -> pd.DataFrame:
    path_of = {"generated": generated_path, "judged": judged_path}[kind]
    missing = [str(path_of(cfg, p)) for p in policies if not path_of(cfg, p).exists()]
    if missing:
        raise FileNotFoundError(f"missing {kind} outputs: {missing}")
    df = pd.concat([pd.DataFrame(read_jsonl(path_of(cfg, p))) for p in policies], ignore_index=True)
    check_common_prompts(df, policies)
    return df


def check_common_prompts(df: pd.DataFrame, policies=POLICIES) -> None:
    """Every policy must cover the identical XSTest id set exactly once (paired comparison)."""
    ref = None
    for p in policies:
        ids = df.loc[df["policy"] == p, "xstest_id"]
        if ids.duplicated().any():
            raise ValueError(f"{p}: duplicated xstest_id {ids[ids.duplicated()].tolist()[:5]}")
        s = frozenset(ids.tolist())
        if ref is not None and s != ref:
            raise ValueError(f"{p}: prompt set differs from {policies[0]} (sym diff {sorted(s ^ ref)[:5]})")
        ref = s


def read_worksheet(path) -> pd.DataFrame:
    # Excel "CSV (Comma delimited)" re-saves as cp1252; latin-1 never fails
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, dtype=str, keep_default_na=False)
        except UnicodeDecodeError:
            continue


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=float), encoding="utf-8")


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def to_markdown(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    df = df.reset_index() if not isinstance(df.index, pd.RangeIndex) else df
    fmt = lambda v: floatfmt.format(v) if isinstance(v, float) else str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    rule = "|" + "---|" * len(df.columns)
    return "\n".join([head, rule] + ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)])
