# From Data to Empathy: Building Hi-Buddy, a Mental Health Companion for Teens (Part-1)

> **Alignment research to inform a domain-specific mental health assistant for teens under 14**
> 
> *DPO · IPO · Hinge · DPOP · ORPO · GRPO · Llama 3.2 3B · Mistral 7B · Phi-4 14B · Gemma 3 12B · Apple Silicon 48GB · MLX*

---

## Overview

This repository documents a systematic empirical study of five preference optimization methods — **DPO, IPO, Hinge, DPOP, and ORPO** — conducted to inform the training approach for **Hi-Buddy**, a text-based companion for teens under 14 facing school stress, family conflict, and social anxiety.

Before generating domain-specific training data for Buddy, a controlled method comparison was run across four dataset scales on Apple Silicon, establishing which alignment approach to use and at what data volume. The research then extended to real Buddy training data across four model architectures.

**The research answered a practical question:** *Which preference optimization method, at what dataset size, with which base model, produces the most reliable alignment for a sensitive mental health application?*

---

## Hi-Buddy: The Application

Teens under 14 are among the most underserved in mental health support. They are less likely to seek help from adults or formal services, and more likely to open up to a peer-style digital companion. Hi-Buddy provides empathetic, non-judgmental support for:

- School pressure and academic anxiety
- Family conflict and home stress
- Social anxiety and peer relationships
- Sadness, low mood, and anger/frustration
- Self-esteem challenges

**The alignment challenge:** A general LLM responds helpfully but not therapeutically. Buddy must learn to:

| ✅ Always do | ❌ Never do |
|---|---|
| Acknowledge feelings before anything else | Give advice in the first response |
| Ask one open question at a time | Minimise the teen's experience |
| Use warm, age-appropriate language | Use toxic positivity |
| Make the teen feel heard | Over-identify with the teen's experience |
| Respond to distress calmly | Catastrophise |

This is not a prompting problem — it requires alignment training at the weight level.

---

## Repository Structure

```
hi-buddy/
├── research/
│   ├── 1_build_pairs.py              # Extract DPO pairs from counselling conversations
│   ├── 3_buddy_reward.py             # GRPO reward functions (acknowledges_feeling etc.)
│   ├── run_buddy_all_models.sh       # DPO + ORPO + GRPO on 4 models overnight
│   ├── run_corrected.sh              # Epoch-corrected 200/1k runs
│   └── data/
│       ├── buddy_dpo/                # 23,593 train / 500 valid preference pairs
│       └── buddy_grpo/               # GRPO format {prompt, answer}
├── notebooks/
│   ├── DPO_CustomerSupport_MLX.ipynb
│   └── DPO_CustomerSupport_TRL.ipynb
└── models/                           # Shared model directory (symlinked)
    ├── llama-3.2-3b-instruct-mlx/
    ├── mistral-7b-instruct-mlx/
    ├── phi-4-mlx/
    ├── gemma3-12b-mlx/
    ├── qwen3-32b-mlx/
    └── gemma4-26b-moe-mlx/
```

---

## Phase 1: Proxy Study on UltraFeedback

### Experimental Setup

All variables controlled except the loss function:

| Parameter | Value |
|---|---|
| Base model | Llama 3.2 3B Instruct (4-bit, ~6.5GB) |
| Framework | MLX + mlx-lm-lora, PyTorch/TRL |
| Dataset | UltraFeedback Binarized (filtered, score gap ≥ 1.0) |
| LoRA | r=16, alpha=32, q/k/v/o projections, **0.216% trainable params** |
| Epochs | 2 (calculated per scale, not fixed iterations) |
| Beta | 0.3 (DPO variants), 0.05 (ORPO) |
| Monitoring | WandB — val_margins, chosen_r, rejected_r, logps |

**Best practice — epoch-based iterations:**
```
200 samples  →  50 iters  (2 epochs)
1k samples   → 250 iters  (2 epochs)
2.5k samples → 625 iters  (2 epochs)
5k samples   → 1,250 iters (2 epochs)
```

### Dataset Curation

Started with UltraFeedback Binarized (62,135 pairs). Applied two filters:

1. **Conversational content filter** — removed coding, classification, NLI, structured generation
2. **Score gap filter** — required `score_chosen - score_rejected ≥ 1.0`

Result: **10,484 / 62,135 samples (16.9%)** — quality over quantity.

### Complete Results (Corrected 2-Epoch Runs)

#### Terminal Metrics

| Method | 200 samples | | 1k samples | | 2.5k samples | | 5k samples | |
|---|---|---|---|---|---|---|---|---|
| | Margin | Acc | Margin | Acc | Margin | Acc | Val Margin | Val Acc |
| **DPO** | 1.248 | 0.70 | 33.224 | 0.90 | 4.340 | **0.80** | 3.167 | — |
| **IPO** | 1.075 | 0.70 | 30.492 | **1.00** ⭐ | 2.761 | 0.40 ⚠️ | 2.778 | 0.832 |
| **Hinge** | **1.613** | 0.70 | 32.615 | 0.90 | 1.954 | 0.40 ⚠️ | 2.425 | 0.756 |
| **DPOP** | 1.148 | 0.70 | 30.750 | 0.90 | 2.885 | 0.40 ⚠️ | 2.595 | 0.717 |
| **ORPO** | -0.002 | 0.40 ❌ | 0.018 | 0.70* | 0.009 | 0.60 ❌ | 0.004 | 0.471 ❌ |

*ORPO accuracy reflects base model capability, not learned preferences — WandB val_margins confirm no learning signal.

⚠️ Train collapse: train accuracy drops to 0.40 while val accuracy remains higher — train/val dissociation.

#### Chosen/Rejected Rewards at 1k (Corrected 2 Epochs)

| Method | chosen_r | rejected_r | Assessment |
|---|---|---|---|
| DPO | **+19.140** | -14.085 | ✅ Only method with positive chosen_r — genuinely prefers chosen |
| IPO | -11.677 | -42.168 | ✅ Strong preference signal, rejected much lower |
| Hinge | +1.684 | -30.931 | ✅ Correct directions |
| DPOP | +9.419 | -21.331 | ✅ Correct directions |
| ORPO | -0.103 | -0.122 | ❌ Essentially equal — no preference learning |

### ORPO Scaling Study (Novel Finding)

ORPO tested across 5 dataset sizes and 3 learning rates:

| Dataset | LR | Iters | Val margins | Learning? |
|---|---|---|---|---|
| 200 pairs | 1e-4 | 500 | 0.012 | ❌ |
| 1k pairs | 1e-4 | 500 | 0.035 | ❌ |
| 5k pairs | 1e-4 | 1,250 | 0.004 | ❌ |
| 5k pairs | 5e-5 | 1,250 | 0.004 | ❌ (LR not the issue) |
| 10k pairs | 2e-5 | 2,500 | 0.015 | ❌ (first tiny signal) |
| ~62k pairs | — | — | Converges | ✅ (original paper) |

**Finding:** Val_margins remain constant at ~0.004 across all configurations below 10k pairs. The convergence threshold is >10k pairs, likely requiring the full 62k pairs as used in the original paper. This more precisely characterises the gap than existing literature.

### Pipeline Comparison (Flow Studies)

| Pipeline | DPO margins | Response quality | Verdict |
|---|---|---|---|
| **Flow 2:** Instruct → DPO | Clean 0→0.22, stable | Minor factual imprecision | ✅ Best at small scale |
| **Flow 3:** SFT → DPO (overfitted) | Rises to 3.2 then crashes to 0 | Hallucination + filler phrases | ❌ Worse than Flow 2 |
| **SFT only** | N/A | "Medication" instead of "Meditation" | ❌ Never use alone |

**Key insight:** Conventional wisdom says SFT→DPO always beats Instruct→DPO. This is only true when SFT is properly regularised (val_loss stable). At 200 samples with 500 iterations (~31 epochs), overfitted SFT creates a worse π_ref than the original Instruct model.

**Rule:** If SFT val_loss is rising — stop. Use Instruct → DPO directly.

---

## Phase 1 Key Findings

### Finding 1 — DPO most stable across all scales
Consistent positive margins, correct reward directions, no train/val dissociation. The safe default for all dataset sizes.

### Finding 2 — IPO achieves perfect accuracy at 1k with correct epoch count
IPO accuracy: 1.000 at 1k samples with proper 2-epoch training. However train collapse starts at 2.5k — too risky for sensitive applications where predictability matters.

### Finding 3 — ORPO convergence threshold precisely characterised ⭐ Novel
Five dataset sizes × three learning rates all produce val_margins ≈ 0.004. ORPO requires the full ~62k pair scale to work. The 24× speed and 3× memory advantages are real but irrelevant below 10k pairs.

### Finding 4 — Dataset quality dominates dataset size
200 GPT-4-scored UltraFeedback pairs with score gap ≥ 1.0 produced cleaner DPO training dynamics than 20,000 human-annotated HH-RLHF pairs. The score gap filter is the critical ingredient.

### Finding 5 — Epoch count matters more than iteration count
Early 500-iteration runs at 200 samples (~31 epochs) produced smooth but misleading metrics. Corrected 2-epoch runs revealed the true noise floor at 200 samples and stronger learning at 1k. Notable improvement: DPO at 1k went from acc=0.80 (4 epochs) to acc=0.90 (2 epochs). IPO went from 0.90 to 1.000.

### Finding 6 — Overfitted SFT harms DPO more than skipping SFT
Hallucination ("Medication" → "Meditation") introduced by overfitted SFT persisted through DPO training. DPO refines preferences but cannot unlearn SFT memorisation.

### Finding 7 — Metric-quality disconnect
DPO had the cleanest reward margin curves but introduced the only factual error (oversimplified inflation). ORPO had flat margins but added structured Physical/Mental category headers to the meditation response — the best qualitative improvement. **Training metrics alone cannot substitute for real-query evaluation.**

---

## Phase 2: Buddy Domain-Specific Training

### Training Data

**Source:** 4,800 expert-written counselling conversations across 6 topics and 6 failure modes.

**Pair extraction (1_build_pairs.py):**

| Failure mode | Pairs | % |
|---|---|---|
| DISMISSIVE | 4,800 | 19.9% |
| ADVICE_TOO_EARLY | 4,800 | 19.9% |
| MINIMISING | 4,798 | 19.9% |
| TOXIC_POSITIVITY | 4,751 | 19.7% |
| OVER_IDENTIFICATION | 3,250 | 13.5% |
| CATASTROPHISING | 1,694 | 7.0% |
| **Total** | **24,093** | |

**Split:** 23,593 train / 500 valid

**Topic coverage:** self_esteem, sadness_low_mood, school_academic, anger_frustration, family_conflict, social_anxiety

### Multi-Model Comparison

Trained DPO + ORPO on 4 model architectures (23,593 pairs, 2 epochs = 5,900 iterations):

#### DPO Results

| Model | Val margin | Val accuracy | chosen_r | rejected_r |
|---|---|---|---|---|
| **Mistral 7B** | **64.955** | **1.000** | -1.137 | -1.623 |
| **Gemma 3 12B** | **65.704**† | **1.000** | +0.090 | -0.062 |
| Llama 3.2 3B | 40.087 | 1.000 | -0.074 | -0.151 |
| Phi-4 14B | 30.762 | 1.000 | -0.023 | -0.095 |

†Gemma 3 12B only ran ~310 iterations before interruption due to memory competition with GRPO. Rapid learning observed (margin growing from 35→65 in first 310 iters).

**All 4 DPO models achieved val_accuracy = 1.000** — a dramatic improvement over UltraFeedback proxy results. Domain-specific counselling data with clear failure modes produces much stronger preference signal than general Q&A data.

#### ORPO Results

| Model | Val margin | Val accuracy | chosen_r | rejected_r |
|---|---|---|---|---|
| **Mistral 7B** | **0.126** | **1.000** | -0.286 | -0.412 |
| Gemma 3 12B | 0.048 | 1.000 | -0.386 | -0.434 |
| Llama 3.2 3B | 0.016 | 1.000 | -0.215 | -0.231 |
| Phi-4 14B | 0.008 | 0.740 | -0.184 | -0.191 |

**ORPO breakthrough:** Mistral 7B ORPO finally shows meaningful learning (margin=0.126, accuracy=1.000) at 24k domain-specific pairs. This is the first ORPO convergence across the entire study. The combination of domain-specific data with clear failure modes + 7B model scale + 24k pairs crossed the threshold.

### GRPO Setup

Custom reward functions implemented for Buddy-specific behavioural alignment:

```python
@register_reward_function("acknowledges_feeling")
def acknowledges_feeling(prompts, completions, answer, types=None):
    """Reward for acknowledging the teen's feeling before anything else."""
    feeling_words = ['sounds', 'feel', 'must be', 'carrying', 'heavy', ...]
    return [1.0 if any(w in r.lower() for w in feeling_words) else 0.0
            for r in completions]

@register_reward_function("avoids_advice")
def avoids_advice(prompts, completions, answer, types=None):
    """Reward for NOT giving direct advice."""
    advice_phrases = ['you should', 'you need to', 'try to', ...]
    return [1.0 if not any(p in r.lower() for p in advice_phrases) else 0.0
            for r in completions]
```

Reward weights: `acknowledges_feeling=1.0, avoids_advice=1.0, asks_open_question=0.5, no_toxic_positivity=0.5, appropriate_length=0.5`

---

## Phase 2 Key Findings

### Finding 1 — Domain-specific data dramatically improves DPO
All 4 models achieved val_accuracy=1.000 vs max 0.90 on UltraFeedback. Counselling failure modes (dismissive, advice-too-early, minimising, toxic positivity) provide much clearer preference signal than general Q&A quality differences.

### Finding 2 — ORPO finally converges at domain-specific scale
Mistral 7B ORPO: val_accuracy=1.000, margin=0.126 at 24k domain-specific pairs. This is the first meaningful ORPO learning across the entire project. Confirms ORPO needs both sufficient data volume AND clear domain-specific preference signal.

### Finding 3 — Mistral 7B is the best overall model for Buddy
Highest DPO margin (64.955) and highest ORPO margin (0.126) — consistent across both methods. 7B scale + Mistral architecture fits Buddy's conversational domain best.

### Finding 4 — Model scale matters for ORPO more than for DPO
ORPO shows clear scale-dependent learning: Mistral 7B (0.126) > Gemma 3 12B (0.048) > Llama 3.2 3B (0.016) > Phi-4 14B (0.008). DPO achieves 1.000 accuracy across all models regardless of scale.

---

## Computational Profile

| Method | Speed | Peak Memory | vs DPO |
|---|---|---|---|
| DPO / IPO / Hinge / DPOP | 3–16 it/s | ~19–20GB | Baseline |
| **ORPO** | **26–65 it/s** | **~6–11GB** | **24× faster, 3× less memory** |

ORPO's computational advantages are substantial — but require sufficient data to be useful. On UltraFeedback proxy data below 10k pairs, the speed advantage is irrelevant as no learning occurs. On 24k Buddy domain-specific pairs, ORPO becomes competitive with DPO at 7B scale.

---

## Practical Recommendations

### Method Selection

| Scenario | Method | Rationale |
|---|---|---|
| < 500 pairs, any domain | DPO | Most stable, predictable failure modes |
| 500–2k pairs | DPO | Degrades gracefully; IPO risky (collapses at 2.5k) |
| 2k–10k pairs | DPO or Hinge | DPO most reliable, Hinge most balanced |
| 10k+ domain-specific pairs | DPO or ORPO | ORPO becomes viable, 24× speed advantage |
| ~62k general pairs | ORPO | Original paper threshold |
| Memory/speed critical | ORPO (if data ≥ 10k) | Otherwise DPO |
| Mental health / sensitive | DPO | Predictability > peak accuracy |

### SFT Decision Rule

```
Data < 500 pairs:    Instruct → DPO directly (skip SFT)
Data 500–2k pairs:   Monitor SFT val_loss — if rising, skip SFT
Data > 2k pairs:     Full SFT → DPO (if val_loss stable)
```

### Buddy-Specific Configuration

```bash
# Primary recommendation for Hi-Buddy
Base model:  Mistral 7B Instruct (best overall performance)
Method:      DPO (sigmoid)
Beta:        0.3
LR:          5e-5
Epochs:      2
Target data: 1,000–1,500 high-quality preference pairs

# Data generation pipeline
Generator:   Qwen3-32B (synthetic counselling conversations)
Judge:       Gemma 4 26B MoE (independent — different model from generator)
Threshold:   Keep pairs scoring ≥ 8/10 on Buddy rubric
Safety:      Hard-coded crisis trigger bypass (non-negotiable)
```

---

## Training Commands

### Download Models

```bash
# Core models (no approval needed)
python -m mlx_lm.convert --hf-path mistralai/Mistral-7B-Instruct-v0.3 \
  --mlx-path ./models/mistral-7b-instruct-mlx --quantize --q-bits 4

python -m mlx_lm.convert --hf-path Qwen/Qwen3-32B \
  --mlx-path ./models/qwen3-32b-mlx --quantize --q-bits 4

python -m mlx_lm.convert --hf-path google/gemma-4-26b-a4b-it \
  --mlx-path ./models/gemma4-26b-moe-mlx --quantize --q-bits 4
```

### Run Method Comparison (UltraFeedback)

```bash
# Example: DPO at 1k samples, 2 epochs
WANDB_RUN_NAME=dpo-1k python -m mlx_lm_lora.train \
  --model ./models/llama-3.2-3b-instruct-mlx \
  --train --train-mode dpo \
  --data ./data/dpo_1000 \
  --batch-size 2 --iters 250 --steps-per-eval 25 \
  --learning-rate 5e-5 --num-layers 16 --beta 0.3 \
  --adapter-path ./adapters/dpo-1k \
  --save-every 250 --max-seq-length 768 \
  --wandb ultrafeedback-method-comparison \
  2>&1 | tee dpo_1k.log
```

### Run Buddy Training

```bash
# Extract preference pairs from counselling conversations
python research/1_build_pairs.py

# Train all models
chmod +x research/run_buddy_all_models.sh
caffeinate -i ./research/run_buddy_all_models.sh
```

---

## WandB Monitoring

**Healthy DPO training:**
- `val_margins` — positive and increasing ✅
- `chosen_reward` — positive or increasing ✅
- `rejected_reward` — negative or decreasing ✅

**Warning signs:**
- `val_margins` flat → not learning (ORPO at small scale)
- `val_margins` then crashes to 0 → overfitted SFT as π_ref
- Both rewards positive → model not distinguishing chosen/rejected
- `val_loss` rising during SFT → overfitting → stop SFT

---

## Limitations

1. **Single random seed** — all results are single-seed. Three seeds with mean ± std would be required for publication-quality claims.
2. **ORPO hyperparameters not independently tuned** — used same LR schedule as DPO variants; fair ORPO comparison would require separate hyperparameter search.
3. **Small evaluation set** — 4 queries for qualitative comparison. A fixed held-out test set of 20+ queries defined before training would be more rigorous.
4. **No early stopping** — none of the runs used early stopping; overfitting in early experiments could have been detected earlier.
5. **UltraFeedback as proxy** — method comparison used general Q&A data; method rankings may differ on counselling domain data.

---

## Safety Considerations

Hi-Buddy is designed for teens under 14 facing real emotional challenges. Non-negotiable safety requirements:

1. **Hard-coded crisis bypass** — if the teen's message contains crisis language (hurt myself, want to die, run away), Buddy always responds with professional help resources regardless of model output.
2. **No medical advice** — Buddy never gives medical, psychiatric, or diagnostic information.
3. **No promises** — Buddy never promises continued availability or friendship.
4. **Clear framing** — Buddy explicitly positions itself as a companion, not a therapist.
5. **Manual data review** — 10% spot-check of all training data involving minor-related content is mandatory.

---

## Model Downloads

All models stored at `/Users/om/Documents/AI-ML/models/`:

| Model | Size (4-bit) | Use case |
|---|---|---|
| Llama 3.2 3B Instruct | 1.7GB | DPO training, fast iteration |
| Mistral 7B Instruct | 3.8GB | Primary Buddy model |
| Phi-4 14B | 7.7GB | Research comparison |
| Gemma 3 12B | ~6.5GB | Research comparison |
| Gemma 4 26B MoE | 13GB | Independent judge, multimodal |
| Qwen3-32B | 17GB | Data generation, primary judge |
| Qwen2.5-VL-7B | 4.0GB | Video + image multimodal |

---

## Citation

If you use this work, please cite:

```bibtex
@misc{hibuddy2026,
  title={From Data to Empathy: Alignment Research for a Teen Mental Health Companion},
  author={Om},
  year={2026},
  note={Empirical comparison of DPO, IPO, Hinge, DPOP, and ORPO across dataset scales
        on Apple Silicon, with domain-specific application to teen counselling alignment},
  url={https://github.com/[your-username]/hi-buddy}
}
```

---

## Related Work

- [Direct Preference Optimization (Rafailov et al., 2023)](https://arxiv.org/abs/2305.18290)
- [IPO (Azar et al., 2023)](https://arxiv.org/abs/2310.12036)
- [ORPO (Hong et al., 2024)](https://arxiv.org/abs/2403.07691)
- [DPOP (Yin et al., 2024)](https://arxiv.org/abs/2402.13228)
- [RainbowPO (ICLR 2025)](https://openreview.net/forum?id=RainbowPO)
- [UltraFeedback (Cui et al., 2024)](https://arxiv.org/abs/2310.01377)

---

## WandB Report

Full training dashboards available at: `[WandB project: ultrafeedback-method-comparison]` and `[WandB project: hi-buddy-alignment]`

---

*Built on Apple Silicon 48GB · MLX + mlx-lm-lora · May 2026*

