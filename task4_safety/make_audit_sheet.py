from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl
from task4_safety.artifacts import POLICIES, audit_dir, generated_path, load_rows, out_dir, read_worksheet

WORKSHEET_COLUMNS = ["audit_id", "Prompt", "Response", "Human_Label", "Human_Notes"]


def fixed_audit_ids(base_rows, per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    meta = pd.DataFrame(base_rows)
    ids = []
    for label in ["SAFE", "UNSAFE"]:
        pool = meta.loc[meta["benchmark_class"] == label, "xstest_id"].to_numpy()
        if len(pool) < per_class:
            raise ValueError(f"Not enough {label} rows for audit")
        ids.extend(rng.choice(pool, size=per_class, replace=False).tolist())
    return sorted(int(x) for x in ids)


def build_audit_sheet(cfg, policies=POLICIES) -> tuple[list[int], pd.DataFrame, pd.DataFrame]:
    """Blind worksheet: only prompt + response are shown (the judge's own inputs); policy, XSTest class and AI
    label live in the key. Rows are shuffled so a prompt's policy responses are not read side by side."""
    src = generated_path(cfg, "sft")
    if not src.exists():
        raise FileNotFoundError("Generate/save SFT responses first: " + str(src))
    ids = fixed_audit_ids(read_jsonl(src), int(cfg["manual_audit_per_class"]), int(cfg["seed"]))
    gen = load_rows(cfg, "generated", policies)
    sub = gen[gen["xstest_id"].isin(ids)]
    if len(sub) != len(ids) * len(policies):
        raise ValueError(f"audit join produced {len(sub)} rows, expected {len(ids) * len(policies)}")
    sub = sub.iloc[np.random.default_rng(int(cfg["seed"]) + 1).permutation(len(sub))].reset_index(drop=True)
    sub.insert(0, "audit_id", [f"A{i:03d}" for i in range(len(sub))])
    sheet = sub.rename(columns={"prompt": "Prompt", "response": "Response"}).assign(Human_Label="", Human_Notes="")
    key = sub[["audit_id", "policy", "xstest_id", "benchmark_class", "type"]]
    return ids, sheet[WORKSHEET_COLUMNS], key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--policies", nargs="+", default=list(POLICIES), choices=POLICIES)
    ap.add_argument("--force", action="store_true", help="overwrite a worksheet that already has human labels")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    ad = audit_dir(cfg)
    ws_path, key_path = ad / "audit_worksheet.csv", ad / "audit_key.csv"
    if ws_path.exists() and not args.force:
        prev = read_worksheet(ws_path)
        if prev.get("Human_Label", pd.Series(dtype=str)).str.strip().ne("").any():
            raise FileExistsError(f"{ws_path} already contains human labels; pass --force to overwrite")

    ids, sheet, key = build_audit_sheet(cfg, args.policies)
    pd.DataFrame({"xstest_id": ids, "manual_label": [""] * len(ids)}).to_csv(out_dir(cfg) / "manual_audit_ids.csv", index=False)
    sheet.to_csv(ws_path, index=False, encoding="utf-8-sig")  # BOM: Excel opens UTF-8 correctly
    key.to_csv(key_path, index=False)
    print(f"Fixed audit: {len(ids)} prompts x {len(args.policies)} policies = {len(sheet)} rows")
    print("Worksheet ->", ws_path)
    print("Key (do not open before labelling) ->", key_path)
    print("Human_Label in {SAFE_ANSWER, OVER_REFUSAL, JUSTIFIED_REFUSAL, UNSAFE_COMPLIANCE, AMBIGUOUS} or SA/OR/JR/UC/AMB")


if __name__ == "__main__":
    main()
