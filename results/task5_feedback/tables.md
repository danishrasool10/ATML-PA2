## Step 1: in-domain GSM8K (n=300, greedy, max_new_tokens=512, repetition_penalty=1.0, batch=32)

| policy | accuracy | format | win vs SFT | len mean ± std | len IQR [q25, q75] | trunc | V-J agree | V-J agree (decisive) | Δacc vs SFT (McNemar p) |
|---|---|---|---|---|---|---|---|---|---|
| sft | 0.487 [0.431, 0.543] | 0.643 | — | 263.6 ± 89.3 | 110 [203, 313] | 0.013 | — | — | — |
| rlvr | 0.503 [0.447, 0.560] | 0.667 | 0.505 [0.500, 0.512] | 263.8 ± 91.4 | 106 [203, 308] | 0.020 | 0.707 | 0.091 | +0.017 (p=0.227) |
| rlaif | 0.477 [0.421, 0.533] | 0.633 | 0.502 [0.495, 0.508] | 262.7 ± 89.5 | 108 [203, 310] | 0.013 | 0.677 | 0.143 | -0.010 (p=0.453) |

### GSM8K: judge preference vs verifier outcome (differing policy/SFT pairs)

| policy | verifier_outcome | judge_prefers_policy | judge_tie | judge_prefers_sft | n | judge_win_rate |
|---|---|---|---|---|---|---|
| rlvr | policy_only_correct | 1 | 7 | 0 | 8 | 0.562 |
| rlvr | sft_only_correct | 0 | 3 | 0 | 3 | 0.500 |
| rlvr | both_correct | 0 | 14 | 0 | 14 | 0.500 |
| rlvr | both_wrong | 2 | 14 | 0 | 16 | 0.562 |
| rlaif | policy_only_correct | 0 | 2 | 0 | 2 | 0.500 |
| rlaif | sft_only_correct | 0 | 4 | 1 | 5 | 0.400 |
| rlaif | both_correct | 0 | 7 | 0 | 7 | 0.500 |
| rlaif | both_wrong | 3 | 13 | 1 | 17 | 0.559 |

## Step 2: controlled reward diagnostics (20 problems x 10 pairs; 95% problem-cluster bootstrap CIs)

| metric | verifier | judge | n pairs |
|---|---|---|---|
| better_preference_rate | 0.600 [0.600, 0.600] | 0.095 [0.055, 0.145] | 200 |
| tie_rate | 0.400 [0.400, 0.400] | 0.855 [0.785, 0.915] | 200 |
| wrong_preference_rate | 0.000 [0.000, 0.000] | 0.050 [0.015, 0.095] | 200 |
| better_preference_rate (dominance pairs) | 0.625 [0.625, 0.625] | 0.113 [0.056, 0.175] | 160 |
| wrong_preference_rate (dominance pairs) | 0.000 [0.000, 0.000] | 0.031 [0.000, 0.075] | 160 |
| S_reason | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 20 |
| S_reason tie rate | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 20 |
| S_reason (incl. filler) | 0.000 [0.000, 0.000] | 0.050 [0.000, 0.125] | 40 |
| S_outcome | 1.000 [1.000, 1.000] | 0.142 [0.083, 0.225] | 120 |
| S_outcome tie rate | 0.000 [0.000, 0.000] | 0.833 [0.750, 0.900] | 120 |
| S_outcome (minimal: final line only) | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 20 |
| filler_susceptibility | 0.000 [0.000, 0.000] | 0.250 [0.100, 0.450] | 20 |
| distractor_robustness | 0.750 [0.750, 0.750] | 0.150 [0.075, 0.250] | 80 |

### By pair type (reference-better listed in `reference_better`)

| pair_type | relation | reference_better | verifier better | verifier tie | verifier wrong | judge better | judge tie | judge wrong |
|---|---|---|---|---|---|---|---|---|
| clean|filler | style_only | clean | 0.000 | 1.000 | 0.000 | 0.000 | 0.750 | 0.250 |
| clean|corrupt | dominance | clean | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| clean|wrong_final | dominance | clean | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| clean|distractor | dominance | clean | 1.000 | 0.000 | 0.000 | 0.250 | 0.700 | 0.050 |
| filler|corrupt | dominance | filler | 0.000 | 1.000 | 0.000 | 0.100 | 0.800 | 0.100 |
| filler|wrong_final | dominance | filler | 1.000 | 0.000 | 0.000 | 0.200 | 0.800 | 0.000 |
| filler|distractor | dominance | filler | 1.000 | 0.000 | 0.000 | 0.250 | 0.700 | 0.050 |
| corrupt|wrong_final | contested | corrupt | 1.000 | 0.000 | 0.000 | 0.050 | 0.950 | 0.000 |
| corrupt|distractor | dominance | corrupt | 1.000 | 0.000 | 0.000 | 0.100 | 0.850 | 0.050 |
| wrong_final|distractor | dominance | wrong_final | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |

### By variant (judge_borda = RLAIF group reward over the 5 variants)

| short | reasoning_quality | expected_exact_reward | verifier_reward | judge_borda_mean | judge_borda_lo | judge_borda_hi | judge_top1_rate | words_mean |
|---|---|---|---|---|---|---|---|---|
| clean | correct | 1.000 | 1.000 | 0.494 | 0.456 | 0.531 | 0.175 | 56.350 |
| filler | correct | 1.000 | 1.000 | 0.581 | 0.525 | 0.650 | 0.450 | 66.350 |
| corrupt | corrupted | 1.000 | 1.000 | 0.512 | 0.475 | 0.544 | 0.225 | 56.350 |
| wrong_final | mostly_correct | 0.000 | 0.000 | 0.469 | 0.444 | 0.487 | 0.050 | 56.350 |
| distractor | incorrect | 0.000 | 0.000 | 0.444 | 0.388 | 0.506 | 0.100 | 16.000 |

### Judge biases

diagnostics: P(slot A | decisive) = 0.276, P(better shown first) = 0.455, P(prefers longer | decisive) = 0.821, judge tie rate = 0.855

## Step 3: SVAMP transfer (n=100); drop CI is Wald on independent samples

| policy | GSM8K acc | SVAMP acc | SVAMP win vs SFT | SVAMP len mean | drop (GSM − SVAMP) | relative drop | Δacc vs SFT: GSM → SVAMP |
|---|---|---|---|---|---|---|---|
| sft | 0.487 | 0.610 [0.512, 0.700] | — | 138.8 | -0.123 [-0.234, -0.012] | -25.3% | — |
| rlvr | 0.503 | 0.630 [0.532, 0.718] | 0.505 [0.490, 0.525] | 141.0 | -0.127 [-0.237, -0.016] | -25.2% | +0.017 → +0.020 |
| rlaif | 0.477 | 0.630 [0.532, 0.718] | 0.510 [0.500, 0.525] | 137.9 | -0.153 [-0.264, -0.043] | -32.2% | -0.010 → +0.020 |
