"""Create numeric post-hoc summaries for Task 1 DPO evaluation artifacts."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
from pathlib import Path

import numpy as np


BOOTSTRAP_SEED = 6304
BOOTSTRAP_SAMPLES = 2000


def _read_json(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _mean(values):
    values = np.asarray(values, dtype=float)
    return float(np.mean(values)) if values.size else float("nan")


def _std(values):
    values = np.asarray(values, dtype=float)
    return float(np.std(values, ddof=1)) if values.size > 1 else (0.0 if values.size else float("nan"))


def _wilson(successes: int, count: int, z: float = 1.959963984540054):
    if count == 0:
        return float("nan"), float("nan")
    p = successes / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    half_width = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return center - half_width, center + half_width


def _bootstrap_ci(values, rng):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan")
    indices = rng.integers(0, values.size, size=(BOOTSTRAP_SAMPLES, values.size))
    means = values[indices].mean(axis=1)
    return tuple(float(x) for x in np.quantile(means, [0.025, 0.975]))


def _get_beta(name: str, root: Path, eval_json: dict | None):
    if eval_json and eval_json.get("beta") is not None:
        return float(eval_json["beta"])
    match = re.search(r"beta_(\d+(?:\.\d+)?)", name)
    if match:
        return float(match.group(1))
    config = root / "configs" / "dpo.yaml"
    if config.exists():
        match = re.search(r"(?m)^\s*beta:\s*([0-9.eE+-]+)\s*$", config.read_text(encoding="utf-8"))
        if match:
            return float(match.group(1))
    raise ValueError(f"Could not determine beta for evaluation condition {name!r}")


def _heldout_row(name: str, records: list[dict], beta: float, rng):
    if not records:
        raise ValueError(f"Preference record file for {name!r} contains no records")
    margins, losses, accuracies = [], [], []
    chosen_logratios, rejected_logratios = [], []
    for record in records:
        policy_margin = float(record["policy_chosen_logp"]) - float(record["policy_rejected_logp"])
        reference_margin = float(record["ref_chosen_logp"]) - float(record["ref_rejected_logp"])
        unscaled_margin = policy_margin - reference_margin
        margin = float(record["implicit_reward_margin"])
        if not math.isclose(margin, beta * unscaled_margin, rel_tol=1e-6, abs_tol=1e-5):
            raise AssertionError(
                f"{name} data_index={record.get('data_index')}: logged margin {margin} "
                f"does not equal beta * recomputed margin {beta * unscaled_margin}"
            )
        margins.append(unscaled_margin)
        z = beta * unscaled_margin
        losses.append(float(np.logaddexp(0.0, -z)))
        accuracies.append(float(record.get("pref_correct", unscaled_margin > 0)))
        chosen_logratios.append(float(record["policy_chosen_logp"]) - float(record["ref_chosen_logp"]))
        rejected_logratios.append(float(record["policy_rejected_logp"]) - float(record["ref_rejected_logp"]))

    loss_low, loss_high = _bootstrap_ci(losses, rng)
    acc = _mean(accuracies)
    acc_low, acc_high = _wilson(sum(accuracies), len(accuracies))
    return {
        "condition": name,
        "n": len(records),
        "beta": beta,
        "heldout_dpo_loss": _mean(losses),
        "heldout_dpo_loss_se": _std(losses) / math.sqrt(len(losses)) if len(losses) > 1 else 0.0,
        "heldout_dpo_loss_ci_low": loss_low,
        "heldout_dpo_loss_ci_high": loss_high,
        "preference_accuracy": acc,
        "preference_accuracy_ci_low": acc_low,
        "preference_accuracy_ci_high": acc_high,
        "unscaled_margin_mean": _mean(margins),
        "unscaled_margin_std": _std(margins),
        "chosen_logratio_mean": _mean(chosen_logratios),
        "rejected_logratio_mean": _mean(rejected_logratios),
        "mean_sigma_neg_z": _mean([
            math.exp(-z) / (1 + math.exp(-z)) if z >= 0 else 1 / (1 + math.exp(z))
            for z in (beta * margin for margin in margins)
        ]),
    }


def _group_records(records: list[dict], key: str):
    groups = {}
    for record in records:
        groups.setdefault(str(record.get(key, "unknown")), []).append(record)
    return groups


def _paired_mcnemar(left, right, label: str):
    left_by_id = {row.get("data_index"): row for row in left if row.get("data_index") is not None}
    right_by_id = {row.get("data_index"): row for row in right if row.get("data_index") is not None}
    common = left_by_id.keys() & right_by_id.keys()
    left_wins = right_wins = 0
    for index in common:
        a = bool(left_by_id[index].get("pref_correct", False))
        b = bool(right_by_id[index].get("pref_correct", False))
        left_wins += int(a and not b)
        right_wins += int(b and not a)
    discordant = left_wins + right_wins
    if discordant:
        tail = min(left_wins, right_wins)
        p_value = min(
            1.0,
            2 * sum(math.comb(discordant, k) for k in range(tail + 1)) / (2 ** discordant),
        )
    else:
        p_value = 1.0
    return {
        "comparison": label,
        "n_paired": len(common),
        "left_only_correct": left_wins,
        "right_only_correct": right_wins,
        "mcnemar_exact_p": p_value,
    }


def _comparisons(records_by_condition: dict[str, list[dict]]):
    results = []
    names = sorted(records_by_condition)
    for left_name, right_name in itertools.combinations(names, 2):
        beta_pair = left_name.startswith("beta_") and right_name.startswith("beta_")
        balanced_pair = (left_name, right_name) == (
            "length_balanced_lengthstrat", "standard_lengthstrat"
        )
        if not (beta_pair or balanced_pair):
            continue
        left, right = records_by_condition[left_name], records_by_condition[right_name]
        pair_label = f"{left_name} vs {right_name}"
        results.append(_paired_mcnemar(left, right, pair_label))
        left_groups, right_groups = _group_records(left, "stratum"), _group_records(right, "stratum")
        for stratum in sorted(left_groups.keys() & right_groups.keys()):
            results.append(_paired_mcnemar(
                left_groups[stratum],
                right_groups[stratum],
                f"{pair_label} [{stratum}]",
            ))
    return results


def _training_stats(root: Path):
    reports = {}
    for log_path in sorted((root / "outputs" / "task1_dpo").glob("*/train_log.jsonl")):
        run_dir = log_path.parent
        config_path = run_dir / "run_config.json"
        config = _read_json(config_path) if config_path.exists() else {}
        cfg = config.get("hyperparameters", {})
        accum = int(cfg.get("grad_accum_steps", 0))
        entries = _read_jsonl(log_path)
        finite_entries = [
            entry for entry in entries
            if all(math.isfinite(float(entry[key])) for key in ("loss", "grad_norm") if key in entry)
        ]
        nan_steps = [
            {"epoch": entry.get("epoch"), "step": entry.get("step")}
            for entry in entries
            if any(
                not math.isfinite(float(value))
                for value in entry.values()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            )
        ]
        full = [
            entry for entry in finite_entries
            if entry.get("accumulation_group_size") == accum and accum > 0
        ]
        if accum == 0:
            full = finite_entries
        last_ten = full[-10:]
        norms = [float(entry["grad_norm"]) for entry in entries if entry.get("grad_norm") is not None
                 and math.isfinite(float(entry["grad_norm"]))]
        max_norm = float(cfg.get("max_grad_norm", float("nan")))
        partial = [
            int(entry["accumulation_group_size"])
            for entry in entries
            if entry.get("accumulation_group_size") is not None
            and accum > 0
            and int(entry["accumulation_group_size"]) < accum
        ]
        reports[run_dir.name] = {
            "optimizer": config.get("optimizer"),
            "n_optimizer_steps": len(entries),
            "nan_steps": nan_steps,
            "partial_group_sizes": partial,
            "last_10_full_group_steps": len(last_ten),
            "last_10_full_group_mean_loss": _mean([entry["loss"] for entry in last_ten]),
            "last_10_full_group_mean_preference_accuracy": _mean(
                [entry["preference_accuracy"] for entry in last_ten if "preference_accuracy" in entry]
            ),
            "max_grad_norm": max_norm,
            "fraction_steps_gradnorm_gt_clip": (
                sum(norm > max_norm for norm in norms) / len(norms)
                if norms and math.isfinite(max_norm) else float("nan")
            ),
        }
    return reports


def _word_limit_stats(root: Path):
    output = {}
    for path in sorted((root / "results" / "task1_dpo").glob("word_limit_*.jsonl")):
        records = _read_jsonl(path)
        compliant = [int(float(row["compliant"])) for row in records if row.get("compliant") is not None]
        low, high = _wilson(sum(compliant), len(compliant))
        words = [int(row["words"]) for row in records if row.get("words") is not None]
        output[path.stem] = {
            "n_prompts": len(records),
            "n_with_parsed_limit": len(compliant),
            "compliance": _mean(compliant),
            "compliance_ci_low": low,
            "compliance_ci_high": high,
            "mean_words": _mean(words),
            "std_words": _std(words),
        }
    return output


def _generation_stats(root: Path):
    output = {}
    for path in sorted((root / "results" / "task1_dpo").glob("policy_generations_*.json")):
        records = _read_json(path)
        lengths = [float(x) for x in records.get("lengths", [])]
        rm_scores = [float(x) for x in records.get("rm_scores", [])]
        output[path.stem.removeprefix("policy_generations_")] = {
            "n_prompts": len(records.get("responses", [])),
            "mean_length_tokens": _mean(lengths),
            "std_length_tokens": _std(lengths),
            "median_length_tokens": float(np.median(lengths)) if lengths else float("nan"),
            "mean_sequence_kl": _mean(records.get("seq_kl", [])),
            "mean_policy_rm_score": _mean(rm_scores),
            "mean_rescored_reference_rm_score": _mean(records.get("rm_ref_rescored", [])),
        }
    return output


def build_report(root: Path):
    result_dir = root / "results" / "task1_dpo"
    record_paths = sorted(result_dir.glob("*_preference_records.jsonl"))
    if not record_paths:
        raise FileNotFoundError(f"No Task 1 preference records found under {result_dir}")
    eval_jsons = {
        path.stem.removesuffix("_eval"): _read_json(path)
        for path in result_dir.glob("*_eval.json")
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    condition_records, heldout_rows, stratum_rows = {}, [], []
    for path in record_paths:
        condition = path.name.removesuffix("_preference_records.jsonl")
        records = _read_jsonl(path)
        condition_records[condition] = records
        beta = _get_beta(condition, root, eval_jsons.get(condition))
        heldout_rows.append(_heldout_row(condition, records, beta, rng))
        for stratum, stratum_records in sorted(_group_records(records, "stratum").items()):
            row = _heldout_row(f"{condition}:{stratum}", stratum_records, beta, rng)
            row["condition"] = condition
            row["stratum"] = stratum
            stratum_rows.append(row)
    report = {
        "heldout": heldout_rows,
        "paired_mcnemar": _comparisons(condition_records),
        "length_margin_coupling": {
            condition: {
                "n": len(records),
                "pearson_r": float(np.corrcoef(
                    [float(row["chosen_tokens"]) - float(row["rejected_tokens"]) for row in records],
                    [float(row["policy_chosen_logp"]) - float(row["policy_rejected_logp"])
                     - float(row["ref_chosen_logp"]) + float(row["ref_rejected_logp"])
                     for row in records],
                )[0, 1]) if len(records) > 1 else float("nan"),
            }
            for condition, records in condition_records.items()
        },
        "training": _training_stats(root),
        "word_limit": _word_limit_stats(root),
        "generation": _generation_stats(root),
    }
    return report, heldout_rows, stratum_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("results/task1_dpo/analysis"))
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.out if args.out.is_absolute() else root / args.out
    report, heldout_rows, stratum_rows = build_report(root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "task1_report_stats.json").write_text(
        json.dumps(report, indent=2, allow_nan=True), encoding="utf-8"
    )
    _write_csv(out / "heldout_summary.csv", heldout_rows)
    _write_csv(out / "stratum_summary.csv", stratum_rows)


if __name__ == "__main__":
    main()
