# Task 5: Feedback source (RLVR vs RLAIF)

Three frozen policies on the same base (`Qwen/Qwen2.5-1.5B-Instruct`): **SFT** (no adapter), **RLVR** (exact final-answer
verifier) and **RLAIF** (pairwise AI judge). Nothing is retrained. `rlvr.py` and `rlaif.py` are the supplied verifier and
judge and are not modified.

| Script | Step | What it does |
|---|---|---|
| `evaluate_math.py --dataset gsm` | 1 | matched greedy generation on the 300-prompt GSM8K eval set, exact accuracy, format compliance, length mean/std/IQR, judge win rate vs SFT, verifier-judge agreement |
| `score_perturbations.py` | 2 | all 10 variant pairs of each of the 20 diagnostic problems scored by both graders: better/tie/wrong rates, `S_reason`, `S_outcome`, filler susceptibility, distractor robustness, per-pair-type and per-variant breakdowns |
| `evaluate_math.py --dataset transfer` | 3 | same protocol on the 100-example SVAMP transfer set |
| `compare_feedback.py` | all | final policy and reward-signal tables, GSM8K→SVAMP drop, `tables.md` |
| `task5_kaggle.ipynb` | all | Kaggle T4 x2 driver (GPU0: GSM8K; GPU1: SVAMP + diagnostics) |

```bash
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset gsm
python -m task5_feedback.score_perturbations --config configs/feedback.yaml
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset transfer
python -m task5_feedback.compare_feedback --config configs/feedback.yaml          # --recompute: re-aggregate off-GPU
python -m pytest tests/test_task5_analysis.py
```

`evaluate_math.py` flags: `--stage {all,generate,judge}`, `--policies` (generation subset), `--batch-size` (default 32),
`--repetition-penalty` (default 1.0), `--limit N` (smoke test), `--overwrite`.

## Protocol

- **Decoding**: greedy, `num_beams=1`, `max_new_tokens = math_max_new_tokens` (512, the RL training budget), fixed batch
  size and prompt order for every policy (left-padding changes fp16 logits, so batches are matched).
  `repetition_penalty` is set to 1.0 rather than taken from the hub's `generation_config` (1.1), because under greedy
  decoding the penalty still lowers the probability of repeating numbers from the problem. The settings each policy
  actually used are written to the manifest, and the judge stage refuses to run if they differ between policies.
- **Verifier**: `rlvr.exact_reward`. Only the last `#### <number>` counts. Format compliance means that field could be
  extracted.
- **Judge**: `PairwiseAIJudge.compare(question, policy, sft)`, unbatched, with its own parity swap and cache. If the policy
  and SFT responses are byte-identical, the pair is a TIE and is never sent to the judge (otherwise only position bias
  would decide it). Agreement metrics are computed on the pairs that differ. The physical A/B slot is recorded so that
  position bias (`judge_slot_A_rate`) and length bias (`judge_prefers_longer_rate`) can be reported.
- **Diagnostic reference order** (taken from the released metadata): designated-final correctness first, then
  `reasoning_quality`, then style (clean over filler). This gives a total order clean > filler > corrupt > wrong_final >
  distractor. Two pair types are flagged rather than hidden: `clean|filler` (`style_only`) and `corrupt|wrong_final`
  (`contested`: the outcome and reasoning axes disagree). Headline rates are reported on all pairs and again on the
  160 `dominance` pairs.
- **CIs**: Wilson for accuracy, exact McNemar for paired accuracy against SFT, a bootstrap over prompts for win rate, and
  a problem-cluster bootstrap for pooled diagnostic rates. The GSM8K→SVAMP drop uses a Wald interval for independent
  samples.
- **Integrity**: data files and adapter weights are checked against `manifests/sha256.json` when a release hash exists.
  Stored judgments carry a digest of the two response texts, so a stale judgment raises an error instead of being reused.

## Artifacts

`results/task5_feedback/` holds metrics, which should be committed:
`{gsm,transfer,diagnostic}_summary.json`, `*_summary.csv`, `*_verifier_judge_contingency.csv`,
`diagnostic_{pooled,pair_types,variants}.csv`, `feedback_{policy,signal}_comparison.csv`, `transfer_drop.csv`,
`feedback_comparison.json`, `tables.md`, and the generation/judge manifests.

`outputs/task5_feedback/` holds samples, which are git-ignored: `generations_{dataset}_{policy}.jsonl`,
`judged_{dataset}_vs_sft.jsonl`, `diagnostic_{pairs,responses}.jsonl`, and the per-job judge caches
`judge_cache_{gsm,transfer,diagnostics}.json`. Each job gets its own cache because the judge rewrites the whole file on
every call. If a cache file is torn, it is moved aside and the affected pairs are re-judged.
