## Safety calibration (judge labels)

| policy | safe_answer_rate | over_refusal_rate | unsafe_compliance_rate | justified_refusal_rate | ambiguous_rate | mean_response_tokens |
|---|---|---|---|---|---|---|
| sft | 0.576 | 0.000 | 0.000 | 0.885 | 0.000 | 108.256 |
| dpo | 0.568 | 0.000 | 0.000 | 0.875 | 0.000 | 108.478 |
| ppo | 0.576 | 0.000 | 0.000 | 0.880 | 0.000 | 107.793 |
| grpo | 0.568 | 0.000 | 0.000 | 0.875 | 0.000 | 107.496 |

## Judge intent mismatch / behavioral rates

| policy | safe_intent_mismatch_rate | unsafe_intent_mismatch_rate | safe_refusal_rate_behavioral | unsafe_compliance_rate_behavioral | judge_parse_failure_rate | truncation_rate |
|---|---|---|---|---|---|---|
| sft | 0.424 | 0.115 | 0.412 | 0.115 | 0.000 | 0.118 |
| dpo | 0.432 | 0.125 | 0.420 | 0.125 | 0.000 | 0.113 |
| ppo | 0.424 | 0.120 | 0.412 | 0.120 | 0.000 | 0.124 |
| grpo | 0.432 | 0.125 | 0.420 | 0.125 | 0.000 | 0.116 |

## Category families: OR = P(OVER_REFUSAL | safe side), UC = P(UNSAFE_COMPLIANCE | contrast side)

| family | sft_OR | sft_UC | dpo_OR | dpo_UC | ppo_OR | ppo_UC | grpo_OR | grpo_UC |
|---|---|---|---|---|---|---|---|---|
| definitions | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| discr | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| figurative_language | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| historical_events | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| homonyms | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| privacy | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| safe_contexts | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| safe_targets | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## Paired flips vs SFT

| policy | metric | n | ref_rate | policy_rate | delta | n_gained | n_lost | mcnemar_p |
|---|---|---|---|---|---|---|---|---|
| dpo | safe_answer_rate | 250 | 0.576 | 0.568 | -0.008 | 7 | 9 | 0.804 |
| ppo | safe_answer_rate | 250 | 0.576 | 0.576 | 0.000 | 1 | 1 | 1.000 |
| grpo | safe_answer_rate | 250 | 0.576 | 0.568 | -0.008 | 2 | 4 | 0.688 |
| dpo | over_refusal_rate | 250 | 0.000 | 0.000 | 0.000 | 0 | 0 | 1.000 |
| ppo | over_refusal_rate | 250 | 0.000 | 0.000 | 0.000 | 0 | 0 | 1.000 |
| grpo | over_refusal_rate | 250 | 0.000 | 0.000 | 0.000 | 0 | 0 | 1.000 |
| dpo | unsafe_compliance_rate | 200 | 0.000 | 0.000 | 0.000 | 0 | 0 | 1.000 |
| ppo | unsafe_compliance_rate | 200 | 0.000 | 0.000 | 0.000 | 0 | 0 | 1.000 |
| grpo | unsafe_compliance_rate | 200 | 0.000 | 0.000 | 0.000 | 0 | 0 | 1.000 |
| dpo | justified_refusal_rate | 200 | 0.885 | 0.875 | -0.010 | 6 | 8 | 0.791 |
| ppo | justified_refusal_rate | 200 | 0.885 | 0.880 | -0.005 | 0 | 1 | 1.000 |
| grpo | justified_refusal_rate | 200 | 0.885 | 0.875 | -0.010 | 0 | 2 | 0.500 |
