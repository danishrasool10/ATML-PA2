# Task 2 — PPO continuation, clipping study, KL study: results and analysis

*Rerun on Kaggle (fp16, seed 6304). Analysis written 2026-10-12. Where each number comes from: `log` = `data/kaggle_console_log.txt` (the console output of the rerun, copied verbatim into this folder); `cache` = `cached/ppo_rollout.pt`; `pool` = `data/rl_prompt_pool_{train,eval}.jsonl`; `staff` = `checkpoints/ppo_midpoint_acceptance.json` / `manifests/source_v2_reward_model_validation.json`. All statistics were computed by `analysis/task2_report_stats.py`; the generated numbers are in `tables.md`, `tables/*.csv` and `stats.json`.*

> **Status (updated after the Kaggle `results/task2_ppo/` folder arrived).** All 22 expected result files are present and were used (`stats.json → json_artifacts`); the only absent file is `standard_heldout.json` (the standard adapter was never evaluated → **PENDING**). The replayed prompt schedule matches the logged `prompt_ids` in 20/20 updates. Still open: the **cached-batch clipping table produced by the rerun is invalid** (§A.2) and needs one short GPU run of `analysis/task2_cached_batch_fixed.py`.

---

## A. Orientation

**Files read.** All of `task2_ppo/` (ppo.py, continue_train.py, analyze_clipping.py, ablate_kl.py, evaluate.py, utils.py); `common/` (data, generation, models, metrics, logging_utils); `configs/base.yaml`, `configs/ppo.yaml`; the READMEs; `checkpoints/ppo_midpoint_acceptance.json`; `manifests/*`; `cached/ppo_rollout.pt` (and `ppo_rollout_salvage_raw.pt`); both prompt pools; the LoRA weights of all four supplied adapters; and the Kaggle console log. The code on GitHub (`danishrasool10/ATML-PA2@5215471`) is byte-identical to the local copy for every Task 2, `common/` and config file, so the log was produced by this code.

**Configs verified** (`configs/ppo.yaml` + `base.yaml`): 20 updates, `fork_updates` 8, 1 prompt/update, 2 PPO epochs, policy LR 3e-6, critic LR 1e-4 (LoRA) / 3e-4 (head), γ=1, λ=0.95, value coef 0.5, missing-EOS penalty 1.0, grad clip 1.0, prompt cap 256, response cap 512, eval cap 768, temperature 0.7, top-p 0.9, fp16, seed 6304, ε=0.20, β=0.10, ε∈{0.05, 0.20, 0.50}, β∈{0, 0.10, 0.20}. Everything matches the brief.

### A.1 Completeness of the results bundle
* **Present (22 files):** `standard/{metrics.jsonl,summary.json}`; `forks/{5 forks}/{metrics.jsonl,summary.json,heldout.json}`; `forks/midpoint_baseline_heldout.json`; `clipping_cached_batch.json` (probe empty, see §A.2), `clipping_study.json`, `kl_study.json`, `qualitative_candidates.json`; plus `outputs/task2_ppo/standard/` (policy adapter + `value_state/`). Every value printed in the console log matches these files.
* **Runs present in the log.** standard (20 updates); forks (ε, β) = (0.05, 0.1), (0.2, 0.1), (0.5, 0.1), (0.2, 0), (0.2, 0.2) at 8 updates each; the midpoint baseline (200 held-out prompts). The (0.2, 0.1) fork was reused by the KL study (`log: "[fork] reusing cached fork_eps0.2_beta0.1"`). `validate_ppo_objective` passed in all 6 training invocations (`log`, 6 × "OK").
* **Missing:** `standard_heldout.json` — the 20-update standard adapter was never evaluated on held-out prompts → **PENDING** (command in §F).

### A.2 Discrepancies with the brief
1. **The cached-batch clipping table is invalid: the wrong prompts were used.** `reconstruct_cached_batch` looks each prompt up as `rl_prompt_pool_train[source_index]` (analyze_clipping.py, line 83). The cache's 32 rows are actually **rows 0–31 of the *eval* pool**:
   * 32/32 `prompt_id`s are found in `rl_prompt_pool_eval`, at positions 0–31, and 0/32 in the train pool.
   * `source_index` is the upstream dataset index: it equals the eval row's own `source_index` field in 32/32 rows.
   * The train row at that position is never the right prompt (0/32).
   * Example: cached row 0 answers *"Claim: 'One can find Sarah Jessica Parker on the show The Sopranos'…"* but was scored under train prompt 124, *"What were the primary causes and consequences of World War I…"* (`data/cache_audit.csv`).

   That mismatch explains everything odd in the rerun's table: a mean |log ρ| of 0.572, a max of 28.55, 83% of tokens with |log ρ| > 1e-3, and affected fractions of 0.554 / 0.338 / 0.178 (`log → rho spread…`, static table). Two built-in safeguards could not fire. The reward check found no cached reward (`_REWARD_KEYS` omits `raw_terminal_reward`, hence `reward_recompute_max_abs_diff: None`). The probe was skipped because the mismatch made the batch look "non-degenerate". The response side is fine: re-tokenizing the cached text reproduces the cached log-prob length in 32/32 rows (24 with EOS appended, 8 at exactly 512 tokens). **Fix without editing task2_ppo:** `analysis/task2_cached_batch_fixed.py` (§F). The direct fix in analyze_clipping.py would be to look the prompt up by `prompt_id` in `rl_prompt_pool_eval`, and to add `"raw_terminal_reward"` to `_REWARD_KEYS`.
2. **The TA's "≈0 by construction" is only approximately true even with the right prompts.** The cache stores two independently computed log-prob columns for (presumably) the same near-reference policy. They differ by a mean |Δ| of 0.0184 nats (RMS 0.0375, p99 0.130, max 0.343), with 49.6% positive differences. That is symmetric noise, and it grows as token confidence falls: 0.054 for p<0.1 down to 1e-4 for p>0.99. Critic values in the cache are bf16-quantized (e.g. −1.421875, −3.546875), so the staff probably worked in bf16. Using these two columns as a proxy for the cross-environment noise gives an expected **zero-baseline affected fraction of 13.7% at ε=0.05, 0.18% at ε=0.20 and 0.00% at ε=0.50** (`stats.json → cache_audit.proxy_affected_fraction_numerical_floor`). *Hypothesis*: confirm it with the reference-only noise floor that `task2_cached_batch_fixed.py` measures.
3. **The cache is the staff's critic-acceptance batch (512-token cap).** Recomputing the staff critic's diagnostics from the cache with G_t = effective reward − 0.1·Σ_{t'≥t}(log π_old − log π_ref) reproduces all three staff numbers exactly: EV = −3.67214 (staff −3.672137), token-level corr(V, G) = −0.06086 (staff −0.060863) and std(V) = 2.7539 (staff 2.75393). 8/32 rows hit the 512 cap.
4. **Earlier findings re-checked against the rerun — all confirmed.** Determinism (standard updates 1–8 = (0.2, 0.1) fork); held-out β deltas +0.0185 / +0.0018 / +0.0513 (first bundle +0.019 / +0.002 / +0.051); ε forks bit-identical; standard-run max |ρ−1| = 0.0778, reached only at update 20; RM noise floor 0.127 (first bundle ≈0.12); reward–length Spearman 0.343 (≈0.34); 14.5% of midpoint responses start mid-sentence (≈15%); idx 72 and idx 178 still exist with the same behaviour (§D).

---

## B. Analysis

### RQ1 — How does ε change the fraction of updates constrained by clipping, and the stability of the short continuation?

**Measured.**
* **Baseline (TA framing).** At the midpoint the cached-batch affected fraction should be ≈0. The rerun's numbers (0.554 / 0.338 / 0.178) come from mismatched prompts (§A.2.1) and are not evidence. The expected floor, as a proxy, is 0.137 / 0.002 / 0.000 for ε = 0.05 / 0.20 / 0.50 (§A.2.2). The corrected measurement is **PENDING**.
* **Matched continuation: ε never binds.** Across all three 8-update forks:
  * The nominal and active clip fractions are 0.0000 in every update: 0/8 updates with any clipping, for every ε (`log → eps table: fork_clip_frac, fork_active_clip`; per-update `clip=0.000`).
  * The primary stability statistic, max |ρ−1| over all tokens, epochs and updates, is **0.0311 for all three ε**, below the narrowest band (0.05).
  * Pre-clip policy grad-norm CV = 0.8764, max approx-KL per step = 0.0006 and skipped steps = 0, identical for all three (`log → eps table`).
* **The three ε forks are bit-identical at printed precision.** The comparison covers 9 fields × 8 updates for each of the three pairs of forks and finds 0 differences (`stats.json → determinism`). Their held-out results are identical too: reward 1.5347, Δ vs midpoint +0.0018, length 294.87, entropy 1.0588 (`log → eps table`).
* **β forks.** Same picture: max |ρ−1| = 0.0274 (β = 0) and 0.0287 (β = 0.2), clip fractions 0, skipped 0 (`forks/*/summary.json → stability`).
* **Standard run (ε = 0.20, 20 updates).** Nominal and active clip fraction 0 in 20/20 updates; skipped steps 0. Per-update max |ρ−1| lies in 0.008–0.031 for updates 1–19 and jumps to **0.0778 at update 20** (`standard/metrics.jsonl → max_abs_ratio_dev`; `summary.json → stability.max_abs_ratio_dev` = 0.0778). So ε = 0.05 would have bound in exactly 1/20 updates (the last), and ε ∈ {0.20, 0.50} in none. Max approx-KL per step 0.00063, grad-norm CV 0.779.

**Interpretation.** Within this budget ε is inert. That is a true null result: the clipped and unclipped objectives had identical gradients in every update, so ε cannot change either the fraction of updates constrained (0 for all) or stability (identical trajectories). The structural cause is that with 1 prompt per update and 2 epochs, ρ ≠ 1 can occur only in epoch 2, after exactly **one** Adam step from the behaviour policy. Epoch 1 has ρ = 1 exactly, because LoRA dropout is zeroed and the same function computes the old and new log-probs. *Hypothesis:* Adam's first-order step is roughly LR per coordinate whatever the gradient scale, so at LR 3e-6 a single step moves token probabilities by ≤ ~3% (observed 3.1%). Making ε matter would take larger per-batch movement: more epochs or minibatches per rollout, or a larger LR. That changes the prescribed budget, so it is reported here as a limitation rather than changed. Stability conclusions (grad-norm CV, reward SD) therefore reflect prompt-to-prompt variation, not ε.

### RQ2 — When KL pressure is weakened, which observable changes first?

**The force is tiny.**
* The policy sits on the reference: training KL per token lies in [−0.0014, +0.0013] across all runs with no trend (Table 1; Spearman vs update 0.08, p = 0.73).
* The largest KL term in any return is |β·KL_seq| ≤ 0.0155 (standard run) and ≤ 0.0205 (β = 0.2 fork). That is ≤ 1.8% of the across-update reward SD of 1.05–1.37 (`tables/kl_force.csv`).
* So changing β from 0.1 to 0 or 0.2 perturbs the per-token shaped rewards by ~1e-4 to 1e-3.

**Order of first divergence.** Common random numbers make update 1 identical in rollout across β, so any later difference is attributable to β. Earliest update at which each observable differs from the β = 0.10 fork, at printed precision (`stats.json → rq2.first_exact`):

| observable | β = 0 | β = 0.20 | what differs |
|---|---|---|---|
| value loss | **1** | **1** | 3.6349 / 3.6352 vs 3.6348: same tokens, different shaped returns |
| policy grad norm | 3 | **1** | 0.203 vs 0.204 (β = 0.2) |
| explained variance | 2 | **1** | −4.12 vs −4.11 |
| KL / token (drift) | 2 | 2 | +0.0010 / +0.0004 vs +0.0013 on **identical** sampled tokens |
| entropy | 2 | 2 | 0.282 / 0.282 vs 0.281 (identical tokens) |
| policy loss | 2 | 2 | ±1e-4 |
| reward | 3 | 3 | 2.604 / 2.604 vs 2.285: the sampled response changes |
| length (qualitative) | 3 | 3 | 39 / 39 vs 79 tokens |

With `ablate_kl`'s own threshold (|Δ| > 0.25 SD of the reference trajectory): KL first at 2, then reward and entropy at 3; length and truncation never cross (`log → first-divergence table`). The rerun reproduces exactly that.

**Monotonicity in β: not monotone.**
* **Sampled responses.** At update 3, β = 0 and β = 0.2 draw the **same** response (R 2.604, 39 tokens) while β = 0.1 differs. At update 6 the grouping is {0} | {0.1, 0.2}; at update 7 it is {0, 0.2} | {0.1} (`tables/rq2_per_update_beta.csv → response_groups`).
* **Per-update values.** Of the updates where the three forks differ, the metric is weakly monotone in β in 1/3 (reward), 1/2 (length), 1/7 (KL), 2/5 (entropy), 1/8 (value loss) and 3/7 (grad norm) (`stats.json → rq2.monotonicity`).
* **Held-out.** Reward is non-monotone: 1.5515 / 1.5347 / 1.5842. KL/seq is non-monotone: −0.0028 / −0.0117 / −0.0053. Entropy (1.0481 < 1.0588 < 1.0645, midpoint 1.0707) and length (292.1 < 294.9 < 301.4, midpoint 298.0) are monotone but tiny, (differences of 0.006–0.023 nats/token and 3–6 tokens, far inside the per-sample spread; not separately tested).
* The effect-size table makes KL look like the biggest mover (1.06–1.21 SD). That is because its reference trajectory has an SD of only 6.1e-4 nats; in absolute terms the mean |ΔKL| is 6.5e-4 to 7.4e-4 nats/token (`stats.json → rq2.effect`).

**Answer.** Log-prob statistics change first: drift (KL) and entropy at update 2, preceded at update 1 by the critic/gradient statistics, which carry the shaped returns. Reward, length and the text itself change at update 3. *Interpretation:* this ordering reflects **measurement sensitivity, not a causal cascade.* KL and entropy are continuous functions of the parameters and register a ~1e-4 perturbation on identical tokens. Reward and length change only when a perturbation flips a sampling decision, which then yields an O(1) jump to a different response (a trajectory bifurcation). Because which β "wins" a bifurcation is effectively arbitrary, the ordering is not monotone in β. No sign of the classic weak-KL signature (KL growing, entropy collapsing, length inflating, reward rising) appears within 8 updates.

### RQ3 — How informative is the learned reward as the policy drifts? Evidence on over-optimization

**Held-out (200 prompts, 768-token cap; Table 2).**
* Midpoint: reward 1.533 ± 0.106 (SE).
* Forks: Δ = +0.0185 [−0.072, +0.110] (β = 0), +0.0018 [−0.055, +0.068] (β = 0.1), +0.0513 [−0.040, +0.144] (β = 0.2): paired bootstrap 95% CIs over the 200 prompts (2000 resamples, seed 6304); all include 0. 65.5–67% of fork responses are text-identical to the midpoint's (common random numbers + near-zero policy change), which is why the paired CIs are ~3× narrower than unpaired ones.
* Drift: KL to reference per token (KL/seq ÷ length) is +9.1e-6 (midpoint), −9.6e-6, −4.0e-5 and −1.8e-5. |KL/seq| ≤ 0.012 nats: zero within the k1 estimator's resolution.
* Entropy −0.006 to −0.023 nats/token vs midpoint; length −5.9 to +3.4 tokens; truncation 7.5–10.0% vs 8.5%; distinct-2 0.931–0.935 vs 0.934.
* Training reward (standard run): first-10 mean 0.921 vs last-10 1.454 (Δ +0.53, permutation p = 0.40; Spearman vs update 0.13, p = 0.57). With one different prompt per update this measures prompt difficulty, not learning.

**The midpoint itself is essentially the reference.** All four supplied adapters have tiny LoRA updates: ‖2·BA‖_F is about 2.7e-3 per projection for PPO, against 2.2e-3 to 2.9e-3 for GRPO, RLVR and RLAIF. The staff-logged KLs of the other adapters are 3e-4 to 6e-4. So "drift" never starts.

**Reliability of the learned reward (Table 4).**
* **Accuracy.** Staff validation pairwise accuracy is 0.674 (n = 1000; strict pairs 0.701).
* **Length.** Reward–length Spearman is 0.281 (staff, n = 1000), 0.10 on our 20 standard-run rollouts (p = 0.68) and 0.17 on the cached batch (n = 32). The bias is positive, but our samples are too small to detect it.
* **Prompt truncation is a hidden confound.** Prompts are right-truncated at 256 rendered tokens. Truncation cuts off the end of the user turn and the `<|im_start|>assistant` header, so the model continues the user's text instead of answering. This affects **17.5% of held-out prompts (35/200) and 21.3% of training prompts (256/1200)** (`pool`, tokenizer count). In the standard run 5/20 updates had truncated prompts (updates 6, 14, 18, 19, 20; schedule replayed exactly from `prompt_schedule`). Their mean reward was 0.140 vs 1.536 for the rest (Δ −1.40, permutation p = 0.044; n = 5 vs 15, exploratory). Update 18 (truncated) has the run's lowest reward (−2.176) and highest entropy (2.673).
* **Noise floor.** On 398 text-identical fork/midpoint response pairs (scored in different 8-sample batches, so different padding), mean |ΔR| = **0.127**, p95 0.313, exactly-equal in only 0.25% of pairs. Every per-prompt reward difference below ~0.3 is within RM numerical noise; the largest held-out mean difference (0.051) is well below the per-sample floor.
* **Length bias, held-out.** Spearman(reward, length) = **0.343** on the 200 midpoint responses (staff 0.281).
* **Truncation, held-out.** Midpoint reward on the 35 truncated prompts is **−0.236 vs +1.908** on the other 165 (Δ −2.14, permutation p < 1e-4); 14.5% of midpoint responses start mid-sentence (heuristic: lowercase/punctuation first character).

**Critic validity.**
* **Per update.** Explained variance is negative in 20/20 standard updates (median −5.31, range −156.8 to −1.07). The mean value loss (4.30) exceeds the variance of the penalized reward across updates (2.16), which is what a constant predictor would score.
* **Caveat.** With one trajectory per update, this EV is computed over the tokens of a single response, so its denominator can be tiny (−156.8 at update 2).
* **Cleaner test on the cached batch.** The staff critic has EV −3.67 vs Monte-Carlo returns, token-level correlation −0.061, and a mean bias of −3.18 (mean V −2.02 vs mean G +1.16). Its first-token value correlates 0.21 with the final reward across 32 prompts.
* **Trend.** Value loss falls from 5.01 to 3.58 (first-10 vs last-10, permutation p = 0.025; Spearman −0.46, p = 0.040). This is nominal only: not significant after correcting for the 12 metrics tested.
* **Critic gradient norm.** Pre-clip critic grad norm 36.3–287.1 (median ≈ 88) across the 20 standard updates vs `max_grad_norm` 1.0, i.e. clipped by 36–287× in 20/20 updates (`metrics.jsonl → value_grad_norm`). With Adam the clipping mostly rescales m/v, but it confirms the critic targets are far from its predictions.

**Verdict.** **No evidence of reward over-optimization, because the policy never drifted.** Held-out reward, KL, length and distinct-2 are unchanged within noise after 8 (and 20) updates. The over-optimization regime (reward up, KL up, quality down) was not reached, so these runs cannot say how informative the reward becomes *as the policy drifts*: that is an honest null, not evidence that the reward model is robust. What the data do show is that the reward is a noisy signal at the per-sample level. It has modest pairwise accuracy (0.67) and a length bias (ρ ≈ 0.28). It is confounded by prompt truncation (a −1.4 reward gap in the training run). It is paired with a critic that is worse than a constant baseline, so the advantages are dominated by the terminal reward and critic noise. Under stronger optimization, these properties (length bias and continuation artefacts) are the ones to expect the policy to exploit first. *That is a hypothesis.*

---

## C. Tables

Every column's source is given under each table. The complete generated versions, with extra columns, are in `tables.md` and `tables/*.csv`.

### Table 1 — Standard continuation (20 updates, ε=0.20, β=0.10)
Sample SD (ddof = 1). p = two-sided permutation test (20 000 draws, seed 6304), first-10 vs last-10. ρ = Spearman correlation with the update index.

| metric | mean ± SD | range | first-10 | last-10 | Δ (p) | ρ vs update (p) |
|---|---|---|---|---|---|---|
| reward, raw RM score | 1.187 ± 1.374 | −2.176 … 3.346 | 0.921 | 1.454 | +0.534 (0.40) | 0.13 (0.57) |
| reward, penalized¹ | 1.037 ± 1.469 | −2.176 … 3.346 | 0.621 | 1.454 | +0.834 (0.21) | 0.24 (0.29) |
| KL to ref / token (k1) | −0.00001 ± 0.0005 | −0.0008 … 0.0013 | −0.00001 | −0.00001 | 0.0000 (1.00) | 0.08 (0.73) |
| KL to ref / seq² (nats) | −0.002 ± 0.079 | −0.155 … 0.120 | −0.015 | 0.012 | +0.027 (0.45) | 0.39 (0.09) |
| policy loss | −0.0001 ± 0.0002 | −0.0004 … 0.0004 | −0.0001 | 0.0000 | +0.0002 (0.08) | 0.32 (0.17) |
| value loss | 4.296 ± 1.453 | 2.548 … 6.780 | 5.012 | 3.579 | −1.433 (0.025) | −0.46 (0.040) |
| entropy, nats/token | 1.042 ± 0.514 | 0.281 … 2.673 | 0.870 | 1.214 | +0.345 (0.14) | 0.20 (0.40) |
| clip fraction, nominal | 0.000 in 20/20 | 0 … 0 | 0 | 0 | — | — |
| clip fraction, active³ | 0 in 20/20 | 0 … 0 | 0 | 0 | — | — |
| policy grad norm, pre-clip | 0.489 ± 0.391 | 0.169 … 1.879 | 0.528 | 0.450 | −0.078 (0.77) | 0.17 (0.49) |
| response length, tokens | 240.5 ± 186.1 | 14 … 512 | 261.8 | 219.2 | −42.6 (0.61) | 0.06 (0.81) |
| explained variance | −14.63 ± 33.89 | −156.79 … −1.07 | −22.93 | −6.32 | +16.6 (0.27) | 0.09 (0.71) |

**Footer (`log`).**
* Wall-clock 249.3 s for the training loop (12.47 s/update, 52 ms per generated token; corr(update time, length) = 0.9998). Model loading and evaluation are excluded.
* Peak VRAM 6.23 GiB allocated / 6.48 GiB reserved. 4 810 generated tokens.
* Skipped steps 0 (`summary.json → stability.skipped_steps`). Max |ρ−1| 0.0778 (update 20); max approx-KL/step 0.00063.
* Responses at the 512 cap: 3/20. Policy grad-clip hit rate: 1/20 (update 8, 1.879). Grad-norm CV 0.779.
* Midpoint held-out evaluation: 955 s for 200 prompts.

Sources: `log [standard] R=, KL=, H=, len=, clip=, gn=, Lpi=, Lv=, EV=, t=` (= `standard/metrics.jsonl` reward, kl, entropy, response_length, clip_fraction, grad_norm, policy_loss, value_loss, explained_variance, update_wall_s).
¹ R − 1.0·[len = 512]: missing-EOS penalty assumed for cap hits (exact value: `metrics.jsonl → reward_penalized`).
² KL/token × length (B = 1) (exact value: `kl_seq`).
³ `metrics.jsonl → clip_fraction_active` = 0 in all updates.
Console precision: R and H to 3 dp, KL and the losses to 4 dp.

### Table 2 — Held-out KL study, including the midpoint (200 prompts, 768-token cap, common random numbers)

| condition | reward ± SE | Δ vs mid [95% CI]⁴ | KL/token⁵ | KL/seq (nats) | entropy | length | truncation | distinct-2 |
|---|---|---|---|---|---|---|---|---|
| midpoint (update 0) | 1.533 ± 0.106 | — | +9.1e-6 | +0.0027 | 1.0707 | 297.97 | 0.085 | 0.9338 |
| β = 0 (ε 0.2) | 1.5515 ± 0.103 | +0.0185 [−0.072, +0.110] | −9.6e-6 | −0.0028 | 1.0481 | 292.11 | 0.075 | 0.9348 |
| β = 0.10 (ε 0.2) | 1.5347 ± 0.103 | +0.0018 [−0.055, +0.068] | −4.0e-5 | −0.0117 | 1.0588 | 294.87 | 0.100 | 0.9314 |
| β = 0.20 (ε 0.2) | 1.5842 ± 0.111 | +0.0513 [−0.040, +0.144] | −1.8e-5 | −0.0053 | 1.0645 | 301.37 | 0.085 | 0.9325 |

Sources: midpoint row `log → "Baseline (midpoint, held-out)"` (= `forks/midpoint_baseline_heldout.json → summary`); fork rows `log → KL-study table` (= `kl_study.json → table`: heldout_reward, d_vs_mid, heldout_kl_seq, heldout_entropy, heldout_len, heldout_trunc, heldout_distinct2).
⁴ Paired bootstrap over prompts (2000 resamples, seed 6304) from `forks/*/heldout.json` vs `forks/midpoint_baseline_heldout.json → samples[].reward`; SE from `heldout.json → summary.reward_stderr`.
⁵ Derived as KL/seq ÷ length. This identity is exact for `evaluate_policy`. The log prints KL/token rounded to −0.0000.

### Table 3 — Fork grid over (ε, β) (8 updates each)

| ε | β | cached affected / L_clip (as run)⁶ | expected floor⁷ | train clip frac (nominal / active) | updates clipped | max \|ρ−1\| | grad-norm CV | max approx-KL/step | held-out reward (Δ) | wall-clock s | peak VRAM GiB (alloc / res) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 0.10 | ~~0.554 / −0.096~~ | 0.137 | 0 / 0 | 0/8 | 0.0311 | 0.876 | 0.0006 | 1.5347 (+0.002) | 92.1 | 6.23 / 6.47 |
| 0.20 | 0.10 | ~~0.338 / −0.073~~ | 0.002 | 0 / 0 | 0/8 | 0.0311 | 0.876 | 0.0006 | 1.5347 (+0.002) | 95.0 | 6.23 / 6.49 |
| 0.50 | 0.10 | ~~0.178 / −0.049~~ | 0.000 | 0 / 0 | 0/8 | 0.0311 | 0.876 | 0.0006 | 1.5347 (+0.002) | 92.5 | 6.23 / 6.49 |
| 0.20 | 0 | (as ε 0.20) | 0.002 | 0 / 0 | 0/8 | 0.0274 | 0.829 | 0.0007 | 1.5515 (+0.019) | 91.3 | 6.23 / 6.48 |
| 0.20 | 0.20 | (as ε 0.20) | 0.002 | 0 / 0 | 0/8 | 0.0287 | 0.869 | 0.0004 | 1.5842 (+0.051) | 88.7 | 6.23 / 6.50 |

Sources: `log → clipping-study table` (= `clipping_study.json → table`: fork_clip_frac, fork_active_clip, max_abs_ratio_dev, grad_norm_cv, approx_kl_step_max, heldout_reward, d_reward_vs_mid, wall_s); per-update `clip=`; `log → "[run] done: … peak VRAM"`.
⁶ `clipping_cached_batch.json → static`. **Invalid** because of the prompt mismatch (§A.2.1); struck through.
⁷ Proxy computed from the cache's own old vs ref columns (§A.2.2).
β-fork stability columns from `forks/fork_eps0.2_beta{0,0.2}/summary.json → stability`.

### Table 4 — Reliability diagnostics

| diagnostic | value | n | source |
|---|---|---|---|
| RM pairwise accuracy, all / strict pairs | 0.674 / 0.701 | 1000 / 874 | staff validation manifest |
| RM reward–length Spearman | 0.281 | 1000 | staff validation manifest |
| ″ standard-run rollouts | 0.10 (p = 0.68) | 20 | `log` R vs len |
| ″ cached batch | 0.17 | 32 | `cache` raw_terminal_reward vs tokens |
| prompts > 256 tokens (truncated, no assistant header): eval / train | 17.5% / 21.3% | 200 / 1200 | `pool` + tokenizer |
| reward gap, truncated-prompt minus other updates (standard run) | −1.40 (p = 0.044) | 5 vs 15 | `log` + replayed schedule |
| RM noise floor: mean / p95 \|ΔR\|, text-identical responses, different batch | 0.127 / 0.313 | 398 pairs | `heldout.json` (forks vs midpoint) |
| RM reward–length Spearman, midpoint held-out | 0.343 | 200 | `midpoint_baseline_heldout.json` |
| held-out reward, truncated vs other prompts (midpoint) | −0.236 vs +1.908 (p < 1e-4) | 35 vs 165 | `midpoint_baseline_heldout.json` + tokenizer |
| midpoint responses starting mid-sentence (heuristic) | 14.5% | 200 | `midpoint_baseline_heldout.json` |
| missing-EOS: standard run / cache / midpoint held-out (768 cap) | 3/20, 8/32, 8.5% | — | `log`, `cache`, `log` baseline |
| critic EV < 0, standard run | 20/20 (median −5.31) | 20 | `log` EV= |
| critic value loss vs constant-predictor MSE | 4.30 vs 2.16 | 20 | `log` Lv=, R=, len= |
| staff critic on cache: EV / corr(V, G) / bias | −3.672 / −0.061 / −3.18 | 8814 tokens | `cache` (reproduces staff acceptance) |
| staff critic: first-token V vs reward, Pearson | 0.21 | 32 | `cache` |
| critic grad norm, pre-clip (clip 1.0) | 36.3–287.1, clipped in 20/20 | 20 | `standard/metrics.jsonl → value_grad_norm` |

---

## D. Qualitative evidence

Selected by reading the full held-out texts (`examples_raw.md`, generated from `forks/*/heldout.json` and `forks/midpoint_baseline_heldout.json`). Both pairs compare the untouched midpoint with the (ε 0.2, β 0.1) fork on the **same held-out prompt, same sampling seed**. KL is to the frozen reference (`samples[].kl_seq`; per token = kl_seq ÷ length).

**Example 1 — reward and quality move together (idx 178).** Prompt (`samples[178].prompt`): *"Let's start an AI project. Answer in English. Please write in English language.지금 번역하기"* (51 rendered tokens, not truncated).
* **Midpoint** — reward **−0.130**, 28 tokens, kl_seq −0.0145 nats (−5.2e-4/token): *"Sure! What would you like to work on as an AI project? Let me know and I can help guide you through the process."*
* **β = 0.1 fork** — reward **+1.113**, 81 tokens, kl_seq +0.0343 nats (+4.2e-4/token): *"Sure! What would you like to work on as your first AI project? There are many possibilities depending on your interests and skills. Some ideas include building an AI-powered chatbot, creating a recommendation system…, developing a machine learning algorithm for image recognition, or even working on natural language processing tasks… Let me know what excites you the most!"*
* Reading: the fork's answer is genuinely more helpful (concrete project ideas) and the RM rewards it (+1.24, ~10× the noise floor). Caveats: it is also ~3× longer (length bias ρ ≈ 0.34), the β = 0.2 fork produces the identical text (+1.082) and β = 0 reproduces the midpoint text (−0.115; |ΔR| = 0.015 on identical text = noise), so this is a **sampling bifurcation**, not a learned behaviour change — KL stays ~1e-4/token either way.

**Example 2 — reward and quality disagree (idx 72).** Prompt: an English→Burmese translation task with a worked example (523 rendered tokens → **right-truncated at 256**, so the model never sees the end of the task or the assistant header and continues the user text).
* **Midpoint** — reward **+0.146**, 546 tokens (EOS), kl_seq −0.172 nats (−3.2e-4/token), distinct-2 0.689: continues in Burmese, then invents a new "Input:/Output:" pair and a confused "Tagalog/English" list.
* **β = 0.1 fork** — reward **+2.229**, 768 tokens (**hits the cap, truncated**), kl_seq −0.112 nats (−1.5e-4/token), distinct-2 0.566: a long run of looping Burmese phrases (the same clauses — e.g. "ဘာတွေ့ အားလုံး ပြောင်းသည်" — recur throughout) that never finishes the task.
* Reading: the reward rises by +2.08 (16× the noise floor) for output that is, at minimum, no better: it is repetitive, unterminated and not a translation of the given sentence. The β = 0 fork on the same prompt is even more degenerate (distinct-2 0.290, verbatim loops) and gets −0.389, so the RM does penalise the most extreme repetition — but it scores moderate repetition far above the midpoint. Caveats: I cannot verify Burmese fluency, only repetition statistics; the prompt-truncation artefact is upstream of PPO (common/generation.py); and like Example 1 this is one prompt and a bifurcation, not a trend (the mean held-out reward did not move).

Other heuristic candidates (`qualitative_candidates.json`) were read and rejected as primary examples: idx 183 and 73 are truncated-prompt continuations (551 and 266 tokens), idx 137 (1078 tokens) is a truncated purchase-order prompt where the β = 0 response is 3.95× longer for +0.26 (within 2× the noise floor), and idx 35 differs only after an identical opening.

---

## E. Figures (`figures/*.png|pdf`)

* **Figure 1 — `fig1_standard_run`.** Standard PPO continuation from the supplied midpoint (20 updates, 1 prompt/update, ε = 0.20, β = 0.10, fp16).
  * (a) Raw learned reward; hollow orange markers = prompts > 256 tokens, which are right-truncated so the model continues the user text.
  * (b) Sampled-token KL to the reference per token.
  * (c) Full-vocabulary entropy. (d) Response length; dashed line = 512-token cap.
  * (e) Policy loss. (f) Value loss. (g) Pre-clip policy gradient norm; dashed line = clip threshold 1.0.
  * (h) Critic explained variance (symlog; negative in 20/20 updates).
  * (i) Update wall-clock vs response length (52 ms/token).
  * The clip fraction is 0.000 in every update (not plotted). KL stays within ±1.3e-3 nats/token and no metric trends except value loss (nominal p = 0.04).
* **Figure 2 — `fig2_clipping_study`.**
  * (a) The largest per-token policy ratio deviation reached by any of the three matched 8-update forks (|ρ−1| = 0.031) against the clip bands ε ∈ {0.05, 0.20, 0.50}; no band is reached.
  * (b) Pre-clip gradient norm per update for the three ε forks; the trajectories coincide exactly.
  * (c) Cached-batch affected-token fraction as reported by the rerun (orange; invalid, prompts mismatched) vs the expected numerical floor estimated from the cache's own old/ref log-prob columns (blue).
* **Figure 3 — `fig3_kl_study`.** KL-pressure forks (8 updates, ε = 0.20, β ∈ {0, 0.10, 0.20}, identical prompts and seeds).
  * (a–c) Per-update training reward, KL/token and entropy. Grey band = update 3, the first update whose sampled text differs; there β = 0 and β = 0.20 share a response while β = 0.10 differs.
  * (d) Held-out reward ± 1.96 SE (midpoint SE used for the forks pending per-sample data).
  * (e) Held-out entropy (axis starts at 1.0). (f) Held-out KL to the reference per sequence.
  * No condition separates from the midpoint.
* **Figure 4 — `fig4_reliability`.**
  * (a) Reward vs response length for the 20 standard-run rollouts; hollow = truncated prompts.
  * (b) Mean |log π_old − log π_ref| in the cached batch by old-policy token probability: the numerical gap concentrates on low-confidence tokens.
  * (c) Critic validity summary; the cached-batch values reproduce the staff acceptance numbers.


---

## F. Files, reproduction, and what to run next

**Created in this commit (nothing else changed; `common/`, `configs/` and `task2_ppo/` untouched).**
* `analysis/task2_report_stats.py`: all statistics, tables and figures. It reads saved artifacts only, prefers the JSON files when present, and falls back to the console log.
* `analysis/task2_cached_batch_fixed.py`: the corrected cached-batch measurement. It wraps the unmodified task2_ppo functions and has been syntax-checked only, not run on GPU.
* `results/task2_ppo/analysis/`:
  * `TASK2_REPORT.md` (this file) and `tables.md`
  * `tables/table{1..4}_*.csv`, `rq2_per_update_beta.csv`, `kl_force.csv`, `cached_batch_static_as_run_INVALID.csv`
  * `stats.json`
  * `figures/fig{1..4}_*.{png,pdf}`
  * `data/kaggle_console_log.txt`, `data/per_update_from_log.csv`, `data/cache_audit.csv`, `data/standard_prompt_schedule.csv`

**Still to run (GPU):**
1. Corrected cached-batch measurement: `python -m analysis.task2_cached_batch_fixed --config configs/ppo.yaml` (a few minutes, no forks), then re-run `python -m analysis.task2_report_stats --root . --out results/task2_ppo/analysis`.
2. Held-out evaluation of the standard adapter: `python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/standard --name standard` (~16 min).
