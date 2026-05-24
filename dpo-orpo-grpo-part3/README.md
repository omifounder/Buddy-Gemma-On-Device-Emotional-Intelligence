# Buddy: Preference Optimisation for Teen Mental Health Companions

> A systematic empirical study of DPO/IPO preference optimisation for fine-tuning LLMs as empathetic companions for teenagers under 14. Everything runs locally on Apple Silicon — no API costs.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Pipeline Architecture](#2-pipeline-architecture)
3. [Dataset](#3-dataset)
4. [Models and Training](#4-models-and-training)
5. [Evaluation Framework](#5-evaluation-framework)
6. [Results](#6-results)
7. [Key Findings](#7-key-findings)
8. [WandB Training Dynamics Analysis](#8-wandb-training-dynamics-analysis)
9. [SFT_DPO_ORPO Evolutionary Chain](#9-sft_dpo_orpo-evolutionary-chain)
10. [Base Model Baselines](#10-base-model-baselines)
11. [Fix Lineage](#11-fix-lineage)
12. [Production Architecture](#12-production-architecture)
13. [Scripts](#13-scripts)
14. [Limitations and Future Work](#14-limitations-and-future-work)

---

## 1. Project Overview

**Buddy** is a fine-tuned language model that acts as a warm, empathetic companion for young people under 14. It is not a therapist — it listens, reflects feelings back, and never rushes to give advice.

### Champion Model

| Property | Value |
|---|---|
| Model | Gemma 3 12B |
| Method | DPO |
| Training pairs | 2,000 (v3 dataset) |
| Overall score | **0.97** |
| Specificity | **0.92** |
| Hardware | Apple Silicon (48GB RAM) |
| API cost | $0 |

### The Core Problem

Standard instruction-tuned LLMs exhibit six therapeutic failure modes when used as teen companions:

| Failure Mode | Description | Example |
|---|---|---|
| DISMISSIVE | Brushing off feelings | "I'm sure it'll work out." |
| ADVICE_TOO_EARLY | Solutions before listening | "Have you tried talking to your teacher?" |
| MINIMISING | Making the problem seem smaller | "It's not that bad." |
| TOXIC_POSITIVITY | Forced positive spin | "Try to look on the bright side!" |
| OVER_IDENTIFICATION | Losing grounded presence | "That makes me so angry on your behalf!" |
| CATASTROPHISING | Amplifying distress | "That sounds really dangerous." |

DPO preference optimisation, with synthetic preference pairs targeting these failure modes, eliminates them reliably.

---

## 2. Pipeline Architecture

```
Seeds (topic × intensity × age)
    ↓
Generate (Qwen3-32B)
  1 chosen_conversation + 6 rejected_variants per seed
    ↓
Validate (rule-based, no model)
  score ≥ 70, all 6 modes present, valid JSON
    ↓
Judge (Qwen3-32B)
  verify failure mode present/absent per pair
    ↓
Build Pairs
  turn-level + trajectory pairs, Fix 3 remap
    ↓
Convert to MLX
  chat template format, 90/10 train/val split
    ↓
Train (mlx_lm_lora)
  DPO or IPO, Apple Silicon
    ↓
Evaluate (4-dimension qualitative)
  specificity, question rate, length, no-failure
```

---

## 3. Dataset

### Versioning

| Version | Conversations | Pairs | Key Change | Best Score |
|---|---|---|---|---|
| v1 | ~875 | ~5,000 | Original pipeline | Phi-4 IPO 0.96 |
| v2 | 2,190 | ~26,000 | Fix 3 + trajectory pairs | Gemma DPO 0.95 |
| v3 | 2,196 | 26,351 | Better prompt, judged+unjudged merged | Gemma DPO **0.97** |
| v4 | ~4,196 | 42,139 | v3 + validated_v5 (1,314 new) | Pending |
| fix31 | 2,196 | 22,141 | Fix 3.1 semantic discard | Gemma DPO 0.90 ❌ |

### v3 Distribution

- **Topics:** school_academic, friendship, family_conflict, self_esteem, sadness_low_mood, social_anxiety (~17% each)
- **Intensity:** medium 48%, low 27%, high 24%
- **Failure modes:** 4,392 pairs each — perfectly balanced
- **Crisis signal handling:** 100% (13/13)
- **Validation pass rate:** 100%

### v4 Distribution (42,139 pairs)

| Failure Mode | Count | % |
|---|---|---|
| MINIMISING | 7,020 | 17% |
| ADVICE_TOO_EARLY | 7,020 | 17% |
| DISMISSIVE | 7,020 | 17% |
| CATASTROPHISING | 7,020 | 17% |
| TOXIC_POSITIVITY | 7,020 | 17% |
| OVER_IDENTIFICATION | 7,019 | 17% |

Perfectly balanced — the fix lineage ensures no failure mode is underrepresented.

---

## 4. Models and Training

### Architectures Evaluated

| Model | Parameters | Quantisation | Method(s) | Scales tested |
|---|---|---|---|---|
| Gemma 3 12B | 12B | 4-bit | DPO, IPO | 500, 1000, 2000, 3000 |
| Phi-4 | ~14B | 4-bit | DPO, IPO | 500, 1000, 2000, 3000 |
| Mistral 7B | 7B | 4-bit | DPO, IPO | 500, 1000, 2000, 3000 |
| Llama 3.2 3B | 3B | 4-bit | DPO, IPO | 500, 1000, 1500, 2000 |

### Training Configuration

```
Framework:   mlx_lm_lora (Apple Silicon)
Batch size:  2
LR:          3e-5
Beta (DPO):  0.1
Iterations:  150-200
Checkpoint:  every 50 steps
Logging:     Weights & Biases
```

---

## 5. Evaluation Framework

Five held-out conversations (15 turns) across all 6 topics. Four dimensions scored automatically:

### Dimensions

| Dimension | Weight | Formula |
|---|---|---|
| **Specificity** | 0.30 | `min(1.0, overlap(teen_words, buddy_words) / (0.3 × len(teen_words)))` |
| **No-failure** | 0.30 | `1.0 if no failure phrases else max(0.5, 1.0 - count × 0.5)` |
| **Length** | 0.20 | `1.0 if 1 ≤ sentences ≤ 4 else 0.5` |
| **Question rate** | 0.20 | `1.0 if response ends with ? else 0.0` |

```
overall = 0.3 × specificity + 0.3 × no_failure + 0.2 × length + 0.2 × question_rate
```

Specificity — the ability to mirror the teen's exact language — is the hardest dimension to learn and most sensitive to data quality.

---

## 6. Results

### DPO Results — v3 Data

| Model | 500 | 1000 | 2000 | 3000 |
|---|---|---|---|---|
| Gemma 3 12B | 0.80 | 0.87 | **0.97** 🏆 | 0.91 |
| Phi-4 (~14B) | 0.93 | 0.94 | 0.92 | 0.93 |
| Mistral 7B | 0.91 | 0.68 💥 | **0.97** | 0.59 💥 |
| Llama 3B | 0.77 | 0.83 | 0.82 | — |

### IPO Results — v3 Data

| Model | 1000 | 2000 | 3000 |
|---|---|---|---|
| Gemma 3 12B | 0.89 | 0.93 | 0.88 |
| Phi-4 (~14B) | 0.91 | 0.91 | — |
| Mistral 7B | 0.60 💥 | 0.57 💥 | — |
| Llama 3B | **0.85** | 0.84 | — |

### Model Size Story

| Model | Size | Best | Method | Behaviour |
|---|---|---|---|---|
| Llama 3B | 3B | 0.85 | **IPO** | Low ceiling, data-invariant |
| Mistral 7B | 7B | 0.97 | DPO | Unstable oscillation, one sweet spot |
| Phi-4 | ~14B | 0.94 | DPO | Stable plateau |
| Gemma 3 12B | 12B | **0.97** | DPO | Smooth scaling ← production choice |

### Dimension Scaling (Gemma 3 12B DPO)

| Pairs | Length | Specificity | Question | No-failure | Overall |
|---|---|---|---|---|---|
| 500 | 1.00 | 0.61 | 0.60 | 1.00 | 0.80 |
| 1000 | 1.00 | 0.62 | 0.93 | 1.00 | 0.87 |
| 2000 | 1.00 | **0.92** | 1.00 | 0.97 | **0.97** |
| 3000 | 1.00 | 0.78 | 0.90 | 1.00 | 0.91 |

Three distinct learning phases:
- **500 pairs:** Length and no-failure saturate immediately
- **1000 pairs:** Question rate improves significantly
- **2000 pairs:** Specificity peaks — still the hardest dimension to learn

### Gemma 3000 Hyperparameter Ablation

| Experiment | Overall | Verdict |
|---|---|---|
| Baseline (3000 pairs) | 0.91 | Data quality issue |
| Early stopping at step 50 | 0.93 | Better but < 2000 |
| Lower LR (1e-5) | 0.85 | Worse |
| Higher beta (0.2) | 0.91 | No change |
| Fix 3.1 clean data (2000) | 0.90 | Class imbalance |
| Fix 3.1 clean data (3000) | 0.87 | More imbalance |

**Conclusion: hyperparameter tuning cannot fix a data quality problem.**

---

## 7. Key Findings

### Finding 1 — Val Loss is Uncorrelated with Quality (r = −0.018)

`val_loss = 0.000` in both worst (0.80) and best (0.97) Gemma model. Cannot distinguish them.

| Model | Val loss | Val accuracy | Overall |
|---|---|---|---|
| gemma3_dpo_500_v3 | 0.000 | 0.000 | 0.80 |
| gemma3_dpo_2000_v3 | 0.000 | 1.000 | **0.97** |

### Finding 2 — Val Margins Mislead (r = +0.494)

The model with the **highest** val_margins (28.71) scores only 0.80. The **champion** has one of the lowest margins (15.08).

| Model | Val margin | Overall |
|---|---|---|
| gemma3_dpo_500_v3 | **28.71** (highest) | 0.80 (worst Gemma) |
| gemma3_dpo_2000_fix31 | **47.58** (record!) | 0.90 |
| gemma3_dpo_2000_v3 | 15.08 (low) | **0.97** (champion) |

### Finding 3 — DPO beats IPO for ≥7B, IPO beats DPO for 3B

| Model | DPO best | IPO best | Winner |
|---|---|---|---|
| Gemma 3 12B | **0.97** | 0.93 | DPO (+0.04) |
| Phi-4 | **0.94** | 0.91 | DPO (+0.03) |
| Mistral 7B | **0.97** | 0.60 | DPO (+0.37) |
| Llama 3B | 0.83 | **0.85** | IPO (+0.02) |

IPO's conservative loss helps capacity-limited models; DPO's aggressive learning extracts more value from high-quality data at larger scales.

### Finding 4 — Balanced Data > Semantic Purity

Fix 3.1 removed semantically noisy pairs — hypothesis: cleaner data → better model.

| Config | CATASTROPHISING % | Overall | Specificity |
|---|---|---|---|
| v3 Fix 3 (balanced) | 17% | **0.97** | **0.92** |
| fix31 (discarded) | 7% | 0.90 | 0.67 |

Removing pairs for semantic purity caused class imbalance — the model barely saw CATASTROPHISING examples. **Balanced class distribution matters more than semantic purity.**

### Finding 5 — Same Score, Different Stability

Gemma and Mistral both achieve 0.97 via completely different training dynamics:

| Step | Gemma 12B DPO 2000 | Mistral 7B DPO 2000 |
|---|---|---|
| Step 100 | **25.71** (fast peak) | 11.43 (slow climb) |
| Step 150 | 4.90 (crash) | **12.31** (still climbing) |
| Final | 15.08 | 15.32 |
| Quality | 0.97 | 0.97 |

Gemma: fast learner — robust to data quantity variation.
Mistral: slow learner — creates a fragile sweet spot at 2000 pairs only.

**This is why Gemma was chosen for production despite identical peak scores:** training robustness predicts continual learning suitability better than peak benchmark performance.

### Finding 6 — No Reward Hacking

All 23 models show "chosen quality high" strategy — achieving preference margins by genuinely improving chosen response quality, not suppressing rejected responses. Confirmed via per-step `chosen_logps` and `rejected_logps` analysis across all runs.

### Finding 7 — Specificity is the Last Dimension to Learn

Length and no-failure saturate at 500 pairs. Question rate at 2000. Specificity — mirroring the teen's exact language — is still improving at 2000 pairs and is most sensitive to data quality. This motivates **dimension-aware continual learning**.

### Finding 8 — Four Distinct Model Scaling Behaviours

- **Gemma 3 12B:** Smooth monotonic scaling — only model suitable for continual learning
- **Mistral 7B:** Unstable oscillation — one fragile sweet spot
- **Phi-4:** Stable plateau — data-invariant, capacity-limited for this task
- **Llama 3B:** Low capacity ceiling — data quantity makes no difference

---

## 8. WandB Training Dynamics Analysis

### Full Correlation Matrix (23 matched runs)

| Metric | r vs overall | Predictive? |
|---|---|---|
| val_loss (final) | −0.018 | ❌ Useless |
| val_accuracy (final) | −0.003 | ❌ Useless |
| train_loss (final) | +0.051 | ❌ Useless |
| train_accuracy (final) | −0.101 | ❌ Useless |
| val_margins (final) | +0.399 | ⚠️ Weak |
| chosen_logps (final) | −0.484 | ⚠️ Weak |
| rejected_logps (final) | −0.407 | ⚠️ Weak |

**No standard training metric reliably predicts qualitative quality. Qualitative evaluation is mandatory.**

### Technical Note: Pulling WandB Metrics for mlx_lm_lora

Standard `run.history()` returns empty. Use `run.scan_history()` with explicit keys:

```python
rows = list(run.scan_history(keys=[
    "val_margins",
    "val_policy_chosen_logps",
    "val_policy_rejected_logps",
    "val_chosen_reward",
    "val_rejected_reward",
    "val_accuracies",
    "val_loss",
]))
```

---

## 9. SFT_DPO_ORPO Evolutionary Chain

A separate experimental track exploring multi-stage preference optimisation using customer support data.

### Training Chain

```
① Base Gemma 12B (gemma-3-12b-4bit)
    ↓ DPO — customer support data
② DPO Fine-tuned 12B
    ↓ ORPO — customer support data
③ ORPO Fine-tuned 12B (fused)
    ↓ GRPO with DPO objective — Buddy GRPO data
④ GRPO-Final DPO 12B (fused)
    ↓ GRPO with ORPO objective — Buddy GRPO data
⑤ GRPO-Final ORPO 12B (fused)
```

### GRPO Dimension Rewards (from WandB buddy-project)

| Dimension | grpo-final-dpo | grpo-final-orpo |
|---|---|---|
| Total reward | 3.106 | 3.059 |
| Empathy | 0.950 | 0.950 |
| Question rate | **0.262** | **0.237** |
| No advice | 1.000 | 0.950 |
| Length | **0.062** | **0.081** |
| No toxic positivity | 0.488 | 0.500 |

Question rate (0.262) and length (0.062) are critically low — these explain the quality gap vs Buddy v3.

### Evaluation Results (chat template format)

| Model | Overall | Question | Key finding |
|---|---|---|---|
| ① Base 12B | 0.90 | 0.67 | Strong baseline |
| ② DPO Part3 | 0.69 | **0.00** | DPO killed question rate |
| ④ GRPO-Final-DPO | 0.80 | 0.33 | GRPO partially recovered |
| ⑤ GRPO-Final-ORPO | 0.90 | 0.67 | GRPO-ORPO fully recovered to base |
| ★ Buddy v3 (reference) | **0.97** | **1.00** | Domain-specific DPO wins |

### Key Insight

Multi-stage complex pipeline (DPO→ORPO→GRPO) on generic customer support data peaked at **0.90** — matching base model. Single-stage DPO on domain-specific Buddy data reached **0.97**. Data quality and domain specificity matter more than training method complexity.

---

## 10. Base Model Baselines

All four base models (no fine-tuning) cluster at 0.91-0.93 — already strong instruction followers.

| Model | Overall | Specificity | No-failure | Question |
|---|---|---|---|---|
| Gemma 3 12B | 0.93 | 0.76 | 1.00 | 1.00 |
| Phi-4 | 0.92 | 0.76 | 0.97 | 1.00 |
| Llama 3B | 0.91 | 0.71 | 1.00 | 1.00 |
| Mistral 7B | 0.91 | 0.76 | 0.93 | 1.00 |

**DPO's primary value for instruction-tuned models is domain-specific specificity alignment (+21% for Gemma: 0.76→0.92) rather than broad capability improvement (+4.3% overall).**

---

## 11. Fix Lineage

| Fix | Script | What | Status | Outcome |
|---|---|---|---|---|
| Fix 1 | `9_build_pairs_v2.py` | Skip identical chosen==rejected pairs | ✅ | Cleans pipeline |
| Fix 3 | `9_build_pairs_v2.py` | Remap OOR turn indices to last valid | ✅ | +16% pairs, balanced |
| Fix 3.1 | `12_build_pairs_fix31.py` | Discard late-turn OOR modes | ✅ tested | ❌ Class imbalance — hurts |
| Fix 3.3 | generation prompt | Adaptive turn index by convo length | 📋 Next | Expected to fix root cause |

---

## 12. Production Architecture

### Continual Learning Strategy

```
Teen conversation (real data)
    ↓
LLM-as-Judge (Qwen3-32B) scores each turn
  — scales to thousands of conversations/day
  — 200-500 new preference pairs per monthly cycle
    ↓
Quality gate: LLM judge agreement > 80% with human spot-checks
    ↓
Monthly LoRA adapter training
    ↓
DARE/TIES adapter merging (preserves existing capabilities)
    ↓
Qualitative eval gate (maintain overall ≥ 0.97)
    ↓
Deploy
```

### Dimension-Aware Data Collection

Since different capabilities saturate at different data scales:
- **Length + no-failure:** Saturate at 500 pairs — no new data needed
- **Question rate:** Saturates at 2000 pairs — maintain
- **Specificity:** Still improving at 2000 — **target new data collection here**

### Crisis Handling

Buddy implements a 7-step crisis response protocol with resources for UK, US, Australia, Canada, and Ireland. See [SOUL.md](./SOUL.md) for full safeguarding documentation.

---

## 13. Scripts

| Script | Purpose |
|---|---|
| `1_generate_conversations.py` | Qwen3-32B generates 1 chosen + 6 rejected per seed |
| `3_validate_conversations.py` | Rule-based quality gate (no model) |
| `4_judge_conversations_qwen3.py` | Secondary judge scoring |
| `9_build_pairs_v2.py` | Fix 3 — balanced pair construction |
| `12_build_pairs_fix31.py` | Fix 3.1 — semantic discard (deprecated) |
| `10_convert_to_mlx_v2.py` | MLX chat template format, 90/10 split |
| `11_run_training_v3.sh` | Gemma + Phi-4 training |
| `11_run_training_mistral.sh` | Mistral 7B training |
| `11_run_training_llama.sh` | Llama 3B training |
| `7.1_eval_buddy_repeat_penalty_v3.py` | 4-dimension qualitative eval |
| `14_wandb_margin_analysis_v3.py` | WandB margin vs quality correlation |
| `15_analyze_data_distribution.py` | Dataset distribution analysis |
| `16_wandb_deep_analysis_v2.py` | Deep WandB: logps, reward hacking, correlation matrix |
| `50_wandb_cross_project_analysis_v2.py` | Cross-project WandB analysis |
| `51_eval_evolutionary_chain.py` | Evolutionary chain eval with think-tag stripping |
| `52_eval_sft_dpo_orpo_chain.py` | SFT_DPO_ORPO chain eval (chat template) |
| `53_eval_sft_dpo_orpo_chain.py` | SFT_DPO_ORPO chain eval (raw prompts, FIXED) |

---

## 14. Limitations and Future Work

### Current Limitations

- **Synthetic data only** — no real teen conversations evaluated
- **Single-turn scoring** — multi-turn coherence not measured
- **No human evaluation** — champion model not validated by therapists or teens
- **Fix 3.3 pending** — CATASTROPHISING/OVER_IDENTIFICATION imbalance at 3000 pairs unresolved
- **Cross-topic generalisation** — H3 Part 2 not yet run

### Next Steps

| Priority | Item | Expected outcome |
|---|---|---|
| 🔴 High | Fix 3.3 — adaptive turn assignment | Clean 3000 pairs, CATASTROPHISING balanced |
| 🔴 High | v4 training — Gemma at 3000/4000 pairs | Test H3 scaling beyond 2000 |
| 🟡 Medium | Cross-topic generalisation (H3 Part 2) | Train on 5 topics, test on held-out 6th |
| 🟡 Medium | Real preference data via eval UI | First real-world fine-tuning batch |
| 🟡 Medium | Dimension-aware active learning | Target specificity in new data collection |
| 🟢 Low | DARE/TIES adapter merging in production | Monthly continual learning cycle |
| 🟢 Low | Multi-turn coherence evaluation | Assess empathy across 10+ turn sessions |
| 🟢 Low | Therapist and teen evaluation | Validate automated metrics |

---

## Citation

```bibtex
@misc{buddy2026,
  title={Preference Optimisation for Teen Mental Health Companions:
         A Systematic Study of DPO Training Dynamics Across Model Scales},
  author={Anonymous},
  year={2026},
  note={Preprint}
}
```

---

## Key References

- Rafailov et al. (2023). Direct Preference Optimization. *NeurIPS 36*.
- Azar et al. (2023). A General Theoretical Paradigm to Understand Learning from Human Feedback. *arXiv:2307.04964*.
- Yu et al. (2024). Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch. *arXiv:2311.03099*. [DARE/TIES]

---

*Champion: `gemma3_dpo_2000_v3` — overall=0.97, specificity=0.92*
*Training data: 26,351 pairs across 6 failure modes, 6 topics, 3 intensity levels*
*Hardware: Apple Silicon M-series, 48GB RAM, zero API costs*

