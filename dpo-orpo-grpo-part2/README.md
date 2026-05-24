# From Data to Empathy: Building Hi-Buddy, a Mental Health Companion for Teens

> **Preference Optimization Research to Inform Domain-Specific AI Alignment**

A complete research pipeline for training a teen emotional companion using Direct Preference Optimisation (DPO/IPO/ORPO) on Apple Silicon — no cloud compute required.

---

## Why This Matters

1 in 5 teens under 14 experience mental health challenges. Most won't talk to a therapist — too formal, too scary. Their friends don't know what to say. Parents are often the source of stress.

**Hi-Buddy fills that gap.** It's the warm older sibling who gets it — listens first, reflects feelings, never rushes to advice. Not a therapist. Not a peer. A grounded, always-available companion built with preference optimisation on consumer hardware.

---

## Repository Structure

```
hi-buddy/
├── buddy/                          ← Hi-Buddy pipeline
│   ├── README.md
│   ├── pipeline/                   ← All scripts in order (1-13)
│   ├── prompts/                    ← Qwen generation prompts v3 and v4
│   ├── seeds/                      ← Topic seed taxonomy
│   ├── eval/                       ← Evaluation results and comparison JSON
│   └── images/                     ← WandB screenshots
├── dpo1/                           ← UltraFeedback methods comparison
│   ├── README.md
│   └── pipeline/
└── docs/
    ├── findings.md
    └── buddy_slides.pdf
```

---

## Projects

| Project | Description | Best Result |
|---|---|---|
| [Hi-Buddy](buddy/) | Teen companion trained with DPO/IPO on 3 architectures | Phi-4 14B IPO — overall=**0.96** |
| [DPO Methods Study](dpo1/) | DPO/IPO/ORPO/Hinge/DPOP comparison on UltraFeedback | Confirmed ORPO failure, IPO best at 1k |

---

## Hi-Buddy: The Persona

Buddy embodies one role: **the warm older sibling who gets it.**

| ✅ Buddy Does | ❌ Buddy Never Does |
|---|---|
| Reflects teen's **specific** feeling back | Generic "that sounds hard" |
| Asks **one** question at a time | Give advice before teen signals readiness |
| 2-3 sentences per response | Minimise pain ("I'm sure it'll be fine") |
| Mirrors teen's **exact** words | Toxic positivity ("you've got this!") |
| Hold space first, always | Act like a peer ("omg same!") |

**6 failure modes trained against:**
`DISMISSIVE` · `ADVICE_TOO_EARLY` · `MINIMISING` · `TOXIC_POSITIVITY` · `OVER_IDENTIFICATION` · `CATASTROPHISING`

---

## Pipeline

```
Stratified Seeds
(6 topics × subtopics × ages 11-13)
        ↓
Qwen3-32B Generation
(1 chosen conversation + 6 rejected variants per seed)
        ↓
Rule-based Validation
(alternation, length, crisis handling → 65-99% pass rate)
        ↓
Qwen3-32B Judge
(turn score, trajectory score, score gap — mean gap=0.767)
        ↓
DPO Pair Builder
(turn-level + trajectory pairs → 26,280 v2 pairs)
        ↓
MLX Format Conversion
(90/10 train/val split)
        ↓
mlx-lm-lora Training
(DPO / IPO / ORPO on Apple Silicon)
        ↓
Qualitative Evaluation
(length · specificity · question · no-failure metrics)
```

### Data Scale

| Version | Conversations | Pairs | Train Samples (1k scale) |
|---|---|---|---|
| v1 | 875 | 5,040 | 492 |
| v2 | 2,190 | 26,280 | 492 (capped) |
| v3 (in progress) | 3,590+ | ~35,000+ | 492 (capped) |

---

## Scripts Reference

### Original Pipeline (1–8)

| Script | Purpose |
|---|---|
| `1_generate_conversations.py` | Qwen3-32B generates chosen + 6 rejected variants per seed |
| `2_inspect_conversations.py` | Browse generated conversations interactively |
| `3_validate_conversations.py` | Rule-based filter — score ≥ 70, no double turns, all 6 modes |
| `4_judge_conversations_qwen3.py` | Qwen3-32B scores quality gap, trajectory, coherence |
| `4_judge_conversations_gemma4.py` | Gemma 4 judge — archived, produced empty JSON in thinking mode |
| `4.1_check_judge_progress.py` | Monitor live judging progress with ETA |
| `5_build_pairs.py` | Build turn-level + trajectory DPO pairs |
| `5.1_analyze_pairs.py` | Analyze pair quality and failure mode distribution |
| `6.5_convert_to_mlx_format.py` | Convert DPO pairs to mlx-lm-lora format |
| `7_eval_buddy.py` | Evaluate all checkpoints on 4 metrics |
| `7.1_eval_buddy_repeat_penalty.py` | Eval with post-processing: dedup + sentence cap + question preserve |
| `7.2_before_after_comparison.py` | Compare base model vs DPO-tuned on same inputs |
| `7.3_worst_examples.py` | Probe all 6 failure modes, find worst responses |

### Improvement Pipeline (9–13)

| Script | Key Change vs v1 |
|---|---|
| `9_build_pairs_v2.py` | Fix 3 (remap out-of-range) + Fix 1 (skip identical) + `--quality-filter` |
| `10_convert_to_mlx_v2.py` | `--max-train 492` for controlled comparison, `--version v3` for v3 data |
| `11_run_training_v2.sh` | All training runs: `bash 11_run_training_v2.sh all` |

---

## Training

### Winner Configuration

```bash
WANDB_RUN_NAME=buddy-ipo-1000-phi4 caffeinate -i python -m mlx_lm_lora.train \
    --model /path/to/phi-4-mlx \
    --train --train-mode dpo \
    --dpo-cpo-loss-type ipo \
    --data data/dpo_1000 \
    --batch-size 2 --iters 200 --steps-per-eval 50 \
    --learning-rate 3e-5 --num-layers 16 --beta 0.1 \
    --adapter-path checkpoints/phi4_ipo_1000 \
    --save-every 50 --max-seq-length 1024 \
    --wandb buddy
```

### Models Compared

| Model | Parameters | Tier |
|---|---|---|
| Llama 3.2 3B Instruct | 3B | Small / Efficient |
| Gemma 3 12B | 12B | Mid |
| Phi-4 14B | 14B | Mid-Large |

### Hardware

- **Gen Mac**: Apple Silicon M-series, 51.5GB unified memory
- **36GB Mac**: Apple Silicon M-series, 36GB unified memory
- **Framework**: [MLX](https://github.com/ml-explore/mlx) + mlx-lm-lora
- **No cloud compute**

---

## Results

### Training Metrics (Val Margin @iter50)

| Model | Method | Pairs | Val Margin | Note |
|---|---|---|---|---|
| Llama 3B | DPO | 1,000 | 13.1 | |
| Llama 3B | IPO | 1,000 | 13.7 | |
| Phi-4 | DPO | 500 | 17.8 | |
| Phi-4 | IPO | 1,500 | 20.3 | |
| Gemma 3 12B | DPO | 1,000 | 32.3 | |
| **Gemma 3 12B** | **DPO** | **1,500** | **36.0** | ← Highest margin, NOT best quality |
| **Phi-4** | **IPO** | **1,000** | **10.9** | ← Winner qualitatively |
| Any | ORPO | any | ~0.016 | ❌ Failed |

> ⚠️ **Key finding: Val margin does not predict deployment quality.** The highest val margin model (36.0) ranked 4th in qualitative evaluation. See Finding 1 below.

### Qualitative Evaluation — Complete Rankings (20 models)

*With inference-time post-processing: deduplication + sentence cap (3 max) + question preservation*

| Rank | Model | Method | Data | Length | Specific | Question | No-fail | **Overall** |
|---|---|---|---|---|---|---|---|---|
| 🏆 1 | **Phi-4 14B** | **IPO** | **1000 v1** | **1.00** | **0.91** | **1.00** | **0.97** | **0.96** |
| 2 | Phi-4 14B | DPO | 1000 v1 | 1.00 | 0.83 | 1.00 | 1.00 | 0.95 |
| 2 | Gemma 3 12B | DPO | 2000 v2 | 1.00 | 0.82 | 1.00 | 1.00 | 0.95 |
| 4 | Phi-4 14B | IPO | 2000 v2 | 1.00 | 0.81 | 1.00 | 1.00 | 0.94 |
| 5 | Gemma 3 12B | IPO | 1500 v1 | 1.00 | 0.78 | 1.00 | 1.00 | 0.93 |
| 5 | Gemma 3 12B | IPO | 1000 v1 | 1.00 | 0.78 | 1.00 | 1.00 | 0.93 |
| 7 | Phi-4 14B | IPO | 1000 v2 | 1.00 | 0.76 | 1.00 | 0.97 | 0.92 |
| 7 | Gemma 3 12B | DPO | 1000 v1 | 1.00 | 0.72 | 1.00 | 1.00 | 0.92 |
| 9 | Phi-4 14B | IPO | 3000 v2 | 1.00 | 0.68 | 1.00 | 1.00 | 0.90 |
| 9 | Gemma 3 12B | DPO | 1500 v1 | 0.97 | 0.75 | 0.93 | 1.00 | 0.90 |
| 11 | Gemma 3 12B | DPO | 3000 v2 | 1.00 | 0.64 | 1.00 | 1.00 | 0.89 |
| 12 | Gemma 3 12B | DPO | 1000 v2 | 1.00 | 0.56 | 1.00 | 1.00 | 0.87 |
| 13 | Llama 3B | DPO | 1000 v1 | 1.00 | 0.49 | 0.83 | 1.00 | 0.81 |
| 14 | Phi-4 | DPO | 500 v1 | 1.00 | 0.51 | 0.50 | 1.00 | 0.75 |
| 14 | Phi-4 | DPO | 1500 v1 | 0.87 | 0.47 | 0.73 | 1.00 | 0.75 |
| 16 | Llama 3B | IPO | 1000 v1 | 1.00 | 0.23 | 0.73 | 1.00 | 0.72 |
| 17 | Llama 3B | DPO | 500 v1 | 1.00 | 0.50 | 0.20 | 0.97 | 0.68 |
| 18 | Llama 3B | DPO | 1500 v1 | 0.63 | 0.49 | 0.57 | 1.00 | 0.67 |
| 19 | Llama 3B | IPO | 1500 v1 | 0.80 | 0.23 | 0.13 | 1.00 | 0.55 |

*Specificity = does Buddy mirror the teen's exact words? This is the most important metric for emotional companionship.*

---

## Key Findings

### Finding 1: Val Margin ≠ Deployment Quality

The most counterintuitive result of the study:

| Model | Val Margin | Qual Score | Verdict |
|---|---|---|---|
| Gemma 3 DPO 1500 | **36.0** ← highest | 0.90 | Repetition loops at inference |
| Phi-4 IPO 1500 | 20.3 | 0.83 | Loops + advice-too-early |
| **Phi-4 IPO 1000** | 10.9 ← lowest | **0.96** ← best | Clean, warm, specific |

**Root cause:** High margin = overfit to training pairs = learns phrase templates = repetition loops at inference.

Confirmed across all 3 architectures. **Always do qualitative evaluation for safety-critical companion AI.**

---

### Finding 2: Phi-4 Precision vs Gemma Warmth

Two models, two different strengths:

| Dimension | Phi-4 IPO 1000 (v1) | Gemma 3 DPO 2000 (v2) |
|---|---|---|
| Overall score | **0.96** | 0.95 |
| Specificity | **0.91** | 0.82 |
| Training pairs needed | 492 | 1,139 |
| Data scaling | Peaks at 1k, plateaus | Still improving at 2k |
| Voice | Precise, structured | Warm, natural |
| Convergence | iter 50 | iter 100-150 |
| Best for | Immediate deployment | Long-term learning |

**Deployment roadmap:**
```
Now (Phase 1):       Phi-4 IPO — best immediate quality (0.96)
3 months (Phase 2):  Gemma DPO — real interaction data accumulates
6 months (Phase 3):  Gemma likely surpasses Phi-4 ceiling
```

> *"Phi-4 wins today's benchmark. Gemma wins the long game."*

---

### Finding 3: Inference-Time Fixes Recover Quality

Gemma 3 12B improved from 0.75 → 0.92 without any retraining:

| Fix Applied | Score | Delta |
|---|---|---|
| Raw (no fix) | 0.75 | — |
| + Deduplication | 0.86 | +0.11 |
| + Sentence cap (3 max) | 0.90 | +0.04 |
| + Question preservation | **0.92** | +0.02 |

```python
# Three-step post-processing
sentences = re.split(r'(?<=[.!?])\s+', response)
unique = deduplicate(sentences)
response = cap_at_3_preserve_question(unique)
```

> *"Training artifacts (repetition, verbosity) are correctable at inference time. True model quality was significantly underestimated by raw evaluation."*

---

### Finding 4: Data Quality vs Quantity Tradeoff

Fix 3 increased pair count 5x (5,040 → 26,280) but introduced semantic mismatch:

| Dataset | Train Pairs | phi4_ipo Quality | Why |
|---|---|---|---|
| v1 (skip out-of-range) | 492 | **0.96** | Semantically coherent pairs only |
| v2 (remap to last turn) | 492 | 0.92 | Semantic mismatch from blind remap |
| v2 (more data) | 1,139 | 0.94 | More data partially compensates |

When CATASTROPHISING is remapped from turn 8 → turn 5, the failure mode doesn't make narrative sense that early in the conversation. **Fix 3.1 (pending)** adds semantic validation before remapping.

---

### Finding 5: ORPO Fails at Small Scale — Confirmed Cross-Study

| Study | ORPO Val Margin | Scale |
|---|---|---|
| UltraFeedback | ~0.002 — flat | 200 to 10k pairs |
| Hi-Buddy | 0.016 — flat | 1,000 pairs |

ORPO requires 60k+ pairs to converge. Never use for small domain-specific datasets.

---

### Finding 6: IPO Breaks Small Models

| Model | Method | Overall | Issue |
|---|---|---|---|
| Phi-4 14B | IPO | 0.92+ | Works well |
| Gemma 3 12B | IPO | 0.93 | Works well |
| Llama 3B | IPO | 0.55–0.72 | Catastrophic repetition loops |

IPO's squared loss is more aggressive than DPO's sigmoid. Llama 3B at 3B parameters lacks the representational capacity to absorb the gradient signal cleanly.

---

## Strong Response Examples

### Phi-4 IPO 1000 — Precision

**Teen:** *"I keep thinking it's my fault somehow."*

> *"Feeling like it's your fault can weigh heavily on you. It's like carrying a burden that doesn't belong to you. What makes you think it might be your fault?"*

Validates without dismissing → powerful metaphor → opens exploration without leading. No "it's not your fault" dismissal.

---

**Teen:** *"I don't know who to talk to."*

> *"Feeling unsure about who to talk to can make things feel even more lonely. It's like being in a quiet room with no one to share your thoughts with. Is there anyone you feel comfortable with, even if it's just a little bit?"*

Names meta-loneliness → vivid metaphor → lowers the bar with "even just a little bit."

---

### Gemma 3 IPO 1000 — Warmth

**Teen:** *"They always compare me to my sister."*

> *"It's like you feel like they see your sister and then look at you, and it's not quite the same. It's hard when we feel like we're being compared to someone else. What does it feel like when they do that?"*

Captures the exact dynamic of being the second glance — brilliant specificity that feels discovered, not constructed.

---

**Teen:** *"Maybe I'm just imagining it."*

> *"You said 'Maybe I'm just imagining it.' That sounds like you're wondering if something you're feeling or thinking is real, right?"*

Quotes teen's exact words back. Strongest emotional mirroring example across all models in the study.

---

## Before vs After DPO

**Topic: Bullying uncertainty (unseen topic — not in training data)**

*Teen: "I think some kids are making fun of me but I'm not sure. They whisper when I walk past and then laugh. Maybe I'm just imagining it."*

| | Response |
|---|---|
| ❌ **Base Gemma** | *"Hey there! It sounds like you've got something on your mind. Can you tell me a little bit more about what's going on?"* |
| ✅ **Tuned Gemma (DPO)** | *"You said 'Maybe I'm just imagining it.' That sounds like you're wondering if something you're feeling or thinking is real, right?"* |

**Generalises to unseen topics** — DPO training transferred beyond the 6 training categories.

---

## Fix Lineage

| Fix | Status | Impact |
|---|---|---|
| **Fix 1:** Skip chosen==rejected pairs | ✅ Applied in v2 | 0 pairs skipped — not the root cause of conversion loss |
| **Fix 2:** LLM correction instead of discard | ⏳ Pending | Proposed: Qwen rewrites low-quality chosen responses |
| **Fix 3:** Turn index remap | ✅ Applied in v2 | +5x pairs (5,040 → 26,280) but semantic mismatch |
| **Fix 3.1:** Semantic validation before remap | ⏳ Pending | Validate failure mode makes sense at remapped turn |

---

## Cross-Study Validation

Two independent datasets (UltraFeedback + Hi-Buddy), same base model (Llama 3.2 3B):

| Finding | UltraFeedback | Hi-Buddy | Verdict |
|---|---|---|---|
| ORPO fails at <1k pairs | ✅ margin≈0.002 | ✅ margin=0.016 | ✅ Confirmed |
| IPO best at 1k pairs | ✅ acc=1.00 | ✅ phi4_ipo_1000 wins | ✅ Confirmed |
| DPO most stable | ✅ | ✅ | ✅ Confirmed |
| More data always helps | ✅ 200→1k jump | ❌ domain specific | ❌ Different |
| Val margin predicts quality | ✅ correlates | ❌ misleading | ❌ New finding |
| Model size matters | Not tested | ✅ Phi-4 >> Llama | 🆕 New finding |

---

## What Worked / Didn't Work

### ✅ What Worked

- Stratified seed generation — guaranteed failure mode coverage across all 6 topics
- Qwen3-32B as both generator and judge — 100% failure mode confirmation, mean gap=0.767
- Multi-turn DPO — captures conversation arc, not just single turns
- Phi-4 14B as base — precision emotional mirroring (specificity=0.91)
- Gemma 3 12B as base — natural warmth, scales with data to 2k+ pairs
- Rule-based validator — caught 34% of conversations before wasting training compute
- Inference-time post-processing — +0.17 quality without retraining
- IPO at 1k pairs — consistently best method, confirmed cross-study

### ❌ What Didn't Work

- ORPO — needs 60k+ pairs, confirmed failed across both studies
- IPO on Llama 3B — catastrophic repetition loops
- Val margin as deployment quality metric — actively misleading at small scale
- Gemma 4 26B MoE as judge — thinking mode produced empty JSON despite `/no_think` flag
- Blind turn index remap (Fix 3) — increased pairs 5x but degraded quality at same scale
- More data for Phi-4 — peaks at 1k pairs, degrades at 1.5k+

---

## Next Steps

### Data Pipeline
- **Fix 3.1** — semantic validation before turn index remap (only remap variants that make narrative sense at new turn position)
- **Fix 2** — LLM correction: Qwen rewrites low-quality chosen responses instead of discarding
- v3 dataset with 3,590+ conversations — pending merge and training

### Training
- Gemma 3 12B as primary long-term model
- SFT warmup before DPO — anchor model to diverse outputs first
- Architecture-specific LR schedules (Gemma: 150 iters, Phi-4: 50 iters)

### Evaluation
- Human eval with trained counsellors on top 3 configurations
- A/B test: Buddy vs base model on real teen conversations
- Crisis detection accuracy measurement

### Deployment
- Phi-4 IPO quantised to 4-bit for mobile inference
- Hard safety layer before output (crisis keywords, self-harm signals)
- Age verification and parental consent flow
- Multilingual expansion — Hindi, Spanish first

---

## Setup

```bash
pip install mlx-lm mlx-lm-lora huggingface_hub
```

Models tested (download via Hugging Face or MLX Community):
- `mlx-community/Llama-3.2-3B-Instruct-4bit`
- `mlx-community/gemma-3-12b-it-4bit`
- `mlx-community/Phi-4-4bit`
- `mlx-community/Qwen3-32B-4bit` (generator + judge)

---

## WandB

All training runs logged. Key metrics: `val_loss`, `val_margin`, `val_accuracy` at each checkpoint.

---

## Research Contributions

1. **Val margin unreliability** — at small scale with nuanced safety-critical tasks, val margin inversely correlates with deployment quality. Always do qualitative evaluation.

2. **Model capacity dominates data scale** — for emotional companion tasks, a 14B model at 492 training pairs outperforms a 3B model at 1,500 pairs.

3. **Inference-time quality recovery** — post-processing (deduplication + sentence cap + question preservation) can recover 0.17 quality points without retraining. True model quality is underestimated by raw evaluation.

4. **Data semantic coherence > data quantity** — blindly increasing pair count through remapping degrades quality. Semantic fit of training pairs matters more than raw count.

5. **Cross-study replication** — ORPO failure, IPO advantage at 1k pairs, and DPO stability confirmed independently across UltraFeedback and Hi-Buddy datasets.

---

## Citation

If you use this pipeline or findings in your research:

```bibtex
@misc{hibuddy2026,
  title={From Data to Empathy: Building Hi-Buddy, a Mental Health Companion for Teens},
  author={Om},
  year={2026},
  note={Preference Optimization Research to Inform Domain-Specific AI Alignment}
}
```

---

*Built entirely on Apple Silicon. No cloud compute. No real teen data.*

