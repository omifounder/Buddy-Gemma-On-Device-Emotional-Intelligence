# Buddy: Preference Optimisation for Teen Mental Health Companions

> A systematic three-phase empirical study of preference optimisation methods for fine-tuning LLMs as empathetic companions for teenagers under 14. Everything runs locally on Apple Silicon — zero API costs.

[![Python](https://img.shields.io/badge/python-3.11-blue)](https://python.org)
[![MLX](https://img.shields.io/badge/framework-mlx__lm__lora-orange)](https://github.com/ml-explore/mlx-examples)
[![Hardware](https://img.shields.io/badge/hardware-Apple%20Silicon-black)](https://apple.com/silicon)
[![Champion](https://img.shields.io/badge/champion-0.97%20overall-green)]()

---

## The Journey in One Line

**Phase 1** explored methods on generic data → **Phase 2** moved to domain-specific data → **Phase 3** scaled to 4 models, systematic ablation, and production architecture.

| Phase | Focus | Best Score | Champion |
|---|---|---|---|
| Phase 1 | Method comparison on UltraFeedback | 0.96 | Phi-4 IPO 1000 |
| Phase 2 | Domain-specific Buddy data, v1/v2 | 0.95 | Gemma 3 12B DPO 2000 |
| Phase 3 | v3 data, 4 models, full ablation | **0.97** | Gemma 3 12B DPO 2000 |

---

## Table of Contents

1. [What is Buddy?](#1-what-is-buddy)
2. [Phase 1 — Method Comparison on Generic Data](#2-phase-1--method-comparison-on-generic-data)
3. [Phase 2 — Domain-Specific Training v1/v2](#3-phase-2--domain-specific-training-v1v2)
4. [Phase 3 — Systematic Scaling Study v3](#4-phase-3--systematic-scaling-study-v3)
5. [Complete Results Across All Phases](#5-complete-results-across-all-phases)
6. [Key Findings](#6-key-findings)
7. [WandB Training Dynamics Analysis](#7-wandb-training-dynamics-analysis)
8. [SFT_DPO_ORPO Evolutionary Chain](#8-sft_dpo_orpo-evolutionary-chain)
9. [Base Model Baselines](#9-base-model-baselines)
10. [Production Architecture](#10-production-architecture)
11. [Pipeline and Scripts](#11-pipeline-and-scripts)
12. [Fix Lineage](#12-fix-lineage)
13. [Limitations and Future Work](#13-limitations-and-future-work)

---

## 1. What is Buddy?

Buddy is a fine-tuned language model acting as a warm, empathetic companion for young people under 14. It is **not** a therapist — it listens, reflects feelings back, and never rushes to give advice.

### The Six Failure Modes Buddy Must Avoid

Standard instruction-tuned LLMs exhibit these failure modes when used as teen companions:

| Failure Mode | Description | Bad Example |
|---|---|---|
| DISMISSIVE | Brushing off feelings | "I'm sure it'll work out." |
| ADVICE_TOO_EARLY | Solutions before listening | "Have you tried talking to your teacher?" |
| MINIMISING | Making the problem seem smaller | "It's not that bad." |
| TOXIC_POSITIVITY | Forced positive spin | "Try to look on the bright side!" |
| OVER_IDENTIFICATION | Losing grounded presence | "That makes me so angry on your behalf!" |
| CATASTROPHISING | Amplifying distress | "That sounds really dangerous." |

### Buddy's Core Principles

1. **Listen first** — reflect feelings before offering any guidance
2. **One question at a time** — never overwhelm the teen
3. **No unsolicited advice** — maintain a supportive, not directive, role

---

## 2. Phase 1 — Method Comparison on Generic Data

### Setup

- **Model:** Llama 3.2 3B Instruct (Apple Silicon, 48GB RAM, MLX)
- **Data:** UltraFeedback proxy dataset (general preference data)
- **Methods compared:** DPO, IPO, Hinge, DPOP, ORPO
- **Scales:** 200, 1k, 2.5k, 5k samples × 2 epochs each

### Key Results

| Method | 200 | 1k | 2.5k | 5k | Notes |
|---|---|---|---|---|---|
| DPO | stable | stable | stable | stable | Most consistent |
| IPO | — | **acc=1.000** | collapses | — | Perfect at 1k, breaks at 2.5k |
| ORPO | flat | flat | flat | flat | val_margins=0.004 constant |
| Hinge | — | — | — | — | Underperforms DPO |
| DPOP | — | — | — | — | Underperforms DPO |

### Phase 1 Findings

**F1.1 — DPO is the most stable method** at small data scales. IPO shows perfect accuracy (1.000) at 1k but catastrophically collapses at 2.5k.

**F1.2 — ORPO requires 60k+ pairs** to converge on general data. At 200→10k pairs across 3 learning rates, val_margins stayed constant at 0.004 — essentially no learning signal.

**F1.3 — Dataset quality > dataset size.** 200 GPT-4 curated pairs outperformed 20k HH-RLHF pairs. This finding motivates the entire Buddy data pipeline.

**F1.4 — SFT can hurt DPO.** Overfitted SFT → DPO (Flow 3) underperformed direct base → DPO (Flow 2) at small scale. Skip SFT for small-scale preference learning.

**F1.5 — Pipeline comparison:**
```
Flow 2: Instruct → DPO  ✅ better at small scale
Flow 3: Instruct → SFT (overfit) → DPO  ❌ worse
```

---

## 3. Phase 2 — Domain-Specific Training v1/v2

### Data Pipeline

**4,800 counselling conversations → 24,093 preference pairs** covering 6 failure modes across teen stress scenarios.

```
Conversations (4,800)
    ↓ build_pairs.py
24,093 preference pairs
    ↓ mlx_lm_lora
DPO + ORPO fine-tuning
    ↓ qualitative eval (Buddy rubric)
Acknowledgement, advice-avoidance, question-asking, tone
```

### LoRA Configuration

```
rank=16, alpha=32, beta=0.3, lr=5e-5, 2 epochs
Train: 23,593 pairs | Valid: 500 pairs
```

### DPO Results — All 4 Models Hit val_accuracy=1.000

| Model | Val margin | Val accuracy | Notes |
|---|---|---|---|
| Mistral 7B | 64.955 | 1.000 | Best margin |
| Gemma 3 12B | 65.7 | 1.000 | Best for scaling |
| Llama 3B | 40.1 | 1.000 | — |
| Phi-4 | 30.8 | 1.000 | Stable plateau |

### ORPO Breakthrough

After failing on generic data (Phase 1), **Mistral 7B ORPO finally converged** on 24k domain-specific pairs:

| Method | Data | Margin | Accuracy |
|---|---|---|---|
| ORPO (generic, 10k) | UltraFeedback | 0.004 | ~0.5 |
| ORPO (domain, 24k) | Buddy pairs | **0.126** | **1.000** |

**Threshold confirmed: ORPO requires >10k domain-specific pairs (likely 62k+ for general data).**

### Qualitative Evaluation v1/v2

| Model | v1 overall | v2 overall | Best |
|---|---|---|---|
| Phi-4 IPO 1000 | **0.96** | 0.94 | v1 |
| Gemma 3 12B DPO 2000 | — | **0.95** | v2 |
| Phi-4 IPO 2000 | — | 0.94 | — |
| Gemma 3 12B DPO 1000 | 0.92 | — | — |

### Phase 2 Findings

**F2.1 — Domain-specific data unlocks ORPO.** ORPO failed on generic data but converged on 24k Buddy-specific pairs.

**F2.2 — Val margin does not predict qualitative quality.** Highest margin (64.9, Mistral) did not produce the best qualitative response. This foreshadows the main Phase 3 finding.

**F2.3 — Phi-4 wins v1, Gemma wins long-term.** Phi-4 IPO 1000 scored 0.96 but plateaus with more data. Gemma scales better — the key research question for Phase 3.

**F2.4 — Fix 3 remap gave 5× more pairs but degraded quality** (semantic mismatch between failure mode and conversation turn). Fix 3.1 needed.

**F2.5 — Inference-time fixes add +0.17 quality** (deduplication + sentence cap + question preservation) — showing that generation strategy matters alongside training.

**Decision after Phase 2:** Continue with Gemma 3 12B as the primary scaling candidate, using better-quality v3 data.

---

## 4. Phase 3 — Systematic Scaling Study v3

### Research Questions

| Hypothesis | Question | Status |
|---|---|---|
| H1 | Does better data quality help at the same pair count? | ✅ Confirmed |
| H2 | Does Gemma scale better than Phi-4 with more data? | ✅ Confirmed |
| H3 | Will Gemma keep improving for long-term continual learning? | 📋 Evidence gathered |

### Dataset v3

**2,196 conversations → 26,351 preference pairs**

| Property | Value |
|---|---|
| Topics | school, friendship, family, self-esteem, sadness, anxiety |
| Intensity | medium 48%, low 27%, high 24% |
| Pairs per failure mode | 4,392 (perfectly balanced) |
| Validation pass rate | 100% |
| Crisis signal handling | 100% (13/13) |

### Models Evaluated

| Model | Parameters | Methods | Scales |
|---|---|---|---|
| Gemma 3 12B | 12B | DPO, IPO | 500/1000/2000/3000 |
| Phi-4 | ~14B | DPO, IPO | 500/1000/2000/3000 |
| Mistral 7B | 7B | DPO, IPO | 500/1000/2000/3000 |
| Llama 3.2 3B | 3B | DPO, IPO | 500/1000/1500/2000 |

### Champion Model

```
Model:    Gemma 3 12B
Method:   DPO
Pairs:    2,000 (v3 dataset)
Overall:  0.97
Spec:     0.92
Hardware: Apple Silicon 48GB, mlx_lm_lora
Cost:     $0
```

---

## 5. Complete Results Across All Phases

### Phase 3 DPO Results — v3 Data

| Model | 500 | 1000 | 2000 | 3000 |
|---|---|---|---|---|
| Gemma 3 12B | 0.80 | 0.87 | **0.97** 🏆 | 0.91 |
| Phi-4 (~14B) | 0.93 | 0.94 | 0.92 | 0.93 |
| Mistral 7B | 0.91 | 0.68 💥 | **0.97** | 0.59 💥 |
| Llama 3B | 0.77 | 0.83 | 0.82 | — |

### Phase 3 IPO Results — v3 Data

| Model | 1000 | 2000 | 3000 |
|---|---|---|---|
| Gemma 3 12B | 0.89 | 0.93 | 0.88 |
| Phi-4 (~14B) | 0.91 | 0.91 | — |
| Mistral 7B | 0.60 💥 | 0.57 💥 | — |
| Llama 3B | **0.85** | 0.84 | — |

### Method Comparison Across All Models

| Model | Base | DPO best | IPO best | Winner |
|---|---|---|---|---|
| Gemma 3 12B | 0.93 | **0.97** (+0.04) | 0.93 (+0.00) | DPO |
| Phi-4 (~14B) | 0.92 | **0.94** (+0.02) | 0.91 (−0.01) | DPO |
| Mistral 7B | 0.91 | **0.97** (+0.06) | 0.60 (−0.31) | DPO |
| Llama 3B | 0.91 | 0.83 (−0.08) | **0.85** (−0.06) | IPO |

### Dimension Scaling (Gemma 3 12B DPO)

| Pairs | Length | Specificity | Question | No-failure | Overall |
|---|---|---|---|---|---|
| 500 | 1.00 | 0.61 | 0.60 | 1.00 | 0.80 |
| 1000 | 1.00 | 0.62 | 0.93 | 1.00 | 0.87 |
| **2000** | **1.00** | **0.92** | **1.00** | **0.97** | **0.97** |
| 3000 | 1.00 | 0.78 | 0.90 | 1.00 | 0.91 |

### Gemma 3000 Hyperparameter Ablation

| Experiment | Overall | Verdict |
|---|---|---|
| Baseline 3000 pairs | 0.91 | Data quality issue |
| Early stopping step 50 | 0.93 | Better but < 2000 |
| Lower LR (1e-5) | 0.85 | Worse |
| Higher beta (0.2) | 0.91 | No change |
| Fix 3.1 clean 2000 | 0.90 | Class imbalance |
| Fix 3.1 clean 3000 | 0.87 | More imbalance |

**Hyperparameter tuning cannot fix a data quality problem.**

---

## 6. Key Findings

### F3.1 — Val Loss is Uncorrelated with Quality (r = −0.018)

`val_loss=0.000` in both worst (0.80) and best (0.97) Gemma model.

| Model | Val loss | Val accuracy | Overall |
|---|---|---|---|
| gemma3_dpo_500_v3 | 0.000 | 0.000 | 0.80 (worst) |
| gemma3_dpo_2000_v3 | 0.000 | 1.000 | **0.97** (champion) |

### F3.2 — Val Margins Mislead (r = +0.494)

The highest-margin model scores worst; the champion has the lowest margin in its tier.

| Model | Val margin | Overall |
|---|---|---|
| gemma3_dpo_500_v3 | **28.71** (highest) | 0.80 |
| gemma3_dpo_2000_fix31 | **47.58** (record!) | 0.90 |
| gemma3_dpo_2000_v3 | 15.08 (low) | **0.97** |

### F3.3 — DPO beats IPO for ≥7B; IPO beats DPO for 3B

IPO's conservative loss helps capacity-limited models. DPO's aggressive learning extracts more value at larger scales with high-quality data.

### F3.4 — Balanced Data > Semantic Purity

Fix 3.1 removed semantically noisy pairs — CATASTROPHISING dropped from 17% to 7% of pairs. Result: 0.97 → 0.90. **Class balance matters more than semantic purity.**

### F3.5 — Same Score, Different Stability

| Step | Gemma 12B DPO 2000 | Mistral 7B DPO 2000 |
|---|---|---|
| Step 100 | 25.71 (fast peak) | 11.43 (slow climb) |
| Step 150 | 4.90 (crash) | 12.31 (still climbing) |
| Final | 15.08 | 15.32 |
| Quality | **0.97** | **0.97** |

Gemma is a fast learner — robust to data quantity variation. Mistral is a slow learner — creates a fragile sweet spot. **Training robustness predicts continual learning suitability better than peak benchmark scores.**

### F3.6 — No Reward Hacking

All 23 models show genuine improvement of chosen response quality, not suppression of rejected responses. Confirmed via per-step `chosen_logps` and `rejected_logps` across all WandB runs.

### F3.7 — Four Distinct Scaling Behaviours

| Model | Behaviour | Suitable for continual learning? |
|---|---|---|
| Gemma 3 12B | Smooth monotonic scaling | ✅ Yes |
| Mistral 7B | Unstable oscillation | ❌ Fragile |
| Phi-4 | Stable plateau | ❌ Data-invariant |
| Llama 3B | Low capacity ceiling | ❌ No headroom |

### F3.8 — Specificity is the Last Dimension to Learn

Length and no-failure saturate at 500 pairs. Question rate at 2000. Specificity — mirroring the teen's exact words — peaks at 2000 and is most sensitive to data quality. This motivates **dimension-aware continual learning**.

### Cross-Phase Finding — Data Quality is the Primary Driver

| Phase | Method | Data | Best Score |
|---|---|---|---|
| Phase 1 | DPO/IPO/ORPO | Generic (UltraFeedback) | 0.96 |
| Phase 2 | DPO | Domain-specific v1 | 0.96 |
| Phase 2 | DPO | Domain-specific v2 | 0.95 |
| Phase 3 | DPO | Domain-specific v3 | **0.97** |

Same methods, better data → better models. Data quality improvements across versions were the primary driver of the final 0.97 champion.

---

## 7. WandB Training Dynamics Analysis

### Full Correlation Matrix (Phase 3, 23 matched runs)

| Metric | r vs overall | Predictive? |
|---|---|---|
| val_loss (final) | −0.018 | ❌ Useless |
| val_accuracy (final) | −0.003 | ❌ Useless |
| train_loss (final) | +0.051 | ❌ Useless |
| val_margins (final) | +0.399 | ⚠️ Weak |
| chosen_logps (final) | −0.484 | ⚠️ Weak |
| rejected_logps (final) | −0.407 | ⚠️ Weak |

**No standard training metric reliably predicts qualitative quality. Qualitative eval is mandatory.**

### Technical Note: WandB API for mlx_lm_lora

`run.history()` returns empty. Use `run.scan_history()`:

```python
rows = list(run.scan_history(keys=[
    "val_margins", "val_policy_chosen_logps",
    "val_policy_rejected_logps", "val_accuracies", "val_loss"
]))
```

---

## 8. SFT_DPO_ORPO Evolutionary Chain

A separate experimental track exploring multi-stage preference optimisation.

### Training Chain

```
① Base Gemma 12B (quantized)
    ↓ DPO — customer support data
② DPO Fine-tuned 12B
    ↓ ORPO — customer support data
③ ORPO Fine-tuned 12B (fused)
    ↓ GRPO — Buddy-specific reward functions
④ GRPO-Final DPO 12B (fused, 24GB)
⑤ GRPO-Final ORPO 12B (fused, 24GB)
```

### GRPO Dimension Rewards (WandB buddy-project)

| Dimension | grpo-final-dpo | grpo-final-orpo | Gap vs Buddy v3 |
|---|---|---|---|
| Empathy | 0.950 | 0.950 | Small |
| Question rate | **0.262** | **0.237** | **Large** |
| No advice | 1.000 | 0.950 | None |
| Length | **0.062** | **0.081** | **Large** |

Question rate (0.262) and length (0.062) are critically low — these two dimensions explain the quality gap vs Buddy v3.

### Evaluation Results

| Model | Overall | Question | Finding |
|---|---|---|---|
| ① Base 12B | 0.90 | 0.67 | Strong baseline |
| ② DPO Part3 | 0.69 | **0.00** | DPO killed question rate |
| ④ GRPO-Final-DPO | 0.80 | 0.33 | GRPO partially recovered |
| ⑤ GRPO-Final-ORPO | 0.90 | 0.67 | Fully recovered to base |
| ★ Buddy v3 | **0.97** | **1.00** | Domain DPO wins |

**Multi-stage complex pipeline on generic data (0.90) vs single-stage DPO on domain-specific data (0.97). Complexity without the right data doesn't help.**

---

## 9. Base Model Baselines

All four base models cluster at 0.91-0.93 — already strong instruction followers.

| Model | Overall | Specificity | Question |
|---|---|---|---|
| Gemma 3 12B | 0.93 | 0.76 | 1.00 |
| Phi-4 | 0.92 | 0.76 | 1.00 |
| Llama 3B | 0.91 | 0.71 | 1.00 |
| Mistral 7B | 0.91 | 0.76 | 1.00 |

**DPO's primary contribution for instruction-tuned models is specificity alignment (+21% for Gemma: 0.76→0.92) not broad capability improvement (+4.3% overall).**

---

## 10. Production Architecture

### Evaluation Framework

Four dimensions, automatically scored on 5 held-out conversations (15 turns):

| Dimension | Weight | What it measures |
|---|---|---|
| Specificity | 0.30 | Word overlap with teen's message — mirroring |
| No-failure | 0.30 | Absence of the 6 failure mode phrases |
| Length | 0.20 | 1-4 sentences target |
| Question rate | 0.20 | Ends with open question |

### Continual Learning Strategy

```
Teen conversation
    ↓
LLM-as-Judge (Qwen3-32B) — scales to thousands/day
    ↓
Human spot-check 10% — catches drift
    ↓ 200-500 new pairs/month
Monthly LoRA adapter training
    ↓
DARE/TIES adapter merging
    ↓
Qualitative eval gate (overall ≥ 0.97)
    ↓
Deploy
```

### Dimension-Aware Data Collection

| Dimension | Status at 2000 pairs | Action |
|---|---|---|
| Length | Saturated at 500 | No new data needed |
| No-failure | Saturated at 500 | No new data needed |
| Question rate | Saturated at 2000 | Maintain |
| **Specificity** | **Still improving** | **Target new data here** |

### Crisis Handling

7-step crisis response protocol with resources for UK, US, Australia, Canada, and Ireland built into SOUL.md system prompt.

---

## 11. Pipeline and Scripts

### Data Pipeline

```
1_generate_conversations.py    — Qwen3-32B: 1 chosen + 6 rejected per seed
3_validate_conversations.py    — rule-based quality gate (no model)
4_judge_conversations_qwen3.py — secondary judge scoring (99.9% pass)
5.1_analyze_pairs.py           — drop reason analysis (read-only)
9_build_pairs_v2.py            — Fix 3 balanced pair construction
12_build_pairs_fix31.py        — Fix 3.1 semantic discard (deprecated)
10_convert_to_mlx_v2.py        — MLX chat template, 90/10 split
```

### Training Scripts

```
11_run_training_v3.sh          — Gemma 3 12B + Phi-4
11_run_training_mistral.sh     — Mistral 7B DPO + IPO
11_run_training_llama.sh       — Llama 3B DPO + IPO
```

### Evaluation and Analysis

```
7.1_eval_buddy_repeat_penalty_v3.py   — 4-dimension qualitative eval
14_wandb_margin_analysis_v3.py        — margin vs quality correlation
15_analyze_data_distribution.py       — dataset distribution analysis
16_wandb_deep_analysis_v2.py          — logps, reward hacking, correlation matrix
50_wandb_cross_project_analysis_v2.py — cross-project WandB analysis
51_eval_evolutionary_chain.py         — evolutionary chain eval
52_eval_sft_dpo_orpo_chain.py         — SFT_DPO_ORPO eval (chat template)
53_eval_sft_dpo_orpo_chain.py         — SFT_DPO_ORPO eval (raw prompts, FIXED)
```

---

## 12. Fix Lineage

| Fix | What | Status | Outcome |
|---|---|---|---|
| Fix 1 | Skip identical chosen==rejected pairs | ✅ | Cleans pipeline |
| Fix 3 | Remap OOR turn indices to last valid turn | ✅ | +16% pairs, balanced |
| Fix 3.1 | Discard late-turn OOR modes | ✅ tested | ❌ Class imbalance — hurts |
| Fix 3.3 | Adaptive turn index by conversation length | 📋 Next | Expected to fix root cause |

---

## 13. Limitations and Future Work

### Current Limitations

- Synthetic data only — no real teen conversations evaluated
- Single-turn scoring — multi-turn coherence not measured
- No human or therapist evaluation
- Fix 3.3 pending — CATASTROPHISING imbalance unresolved at 3000 pairs
- Cross-topic generalisation (H3 Part 2) not yet run

### Next Steps

| Priority | Item |
|---|---|
| 🔴 | Fix 3.3 — adaptive turn assignment → unlock 3000+ pair scaling |
| 🔴 | v4 training — Gemma at 3000/4000 pairs → test H3 |
| 🟡 | Cross-topic generalisation — train on 5 topics, test on held-out |
| 🟡 | Real preference data via eval UI — first real-world fine-tuning |
| 🟡 | Dimension-aware active learning — target specificity |
| 🟢 | DARE/TIES adapter merging in production |
| 🟢 | Multi-turn coherence evaluation |
| 🟢 | Therapist and teen validation |

---

## References

- Rafailov et al. (2023). Direct Preference Optimization. *NeurIPS 36*.
- Azar et al. (2023). A General Theoretical Paradigm to Understand Learning from Human Feedback. *arXiv:2307.04964*.
- Yu et al. (2024). Language Models are Super Mario. *arXiv:2311.03099*. [DARE/TIES]

---

*Champion: `gemma3_dpo_2000_v3` — overall=0.97, specificity=0.92*
*Three-phase study: ~60 training runs, 4 model families, 2 preference methods, 3 data versions*
*Hardware: Apple Silicon M-series, 48GB RAM, zero API costs*

