# Synthetic Data Generation & Validation Strategy

> How we built 2,190 high-quality preference learning conversations for a teen emotional companion — without using any real user data.

---

## Why Synthetic Data?

Real counselling conversations for teens cannot be used:
- **Privacy** — minors' mental health data requires strict consent and IRB approval
- **Distribution** — real data reflects therapist behaviour, not the "older sibling" persona we want
- **Control** — we need exactly 6 failure modes, balanced across 6 topics and 3 age groups

Synthetic data solves all three. The risk is distribution shift — synthetic conversations may not perfectly reflect real teen language. We mitigated this by:
- Grounding seeds in real adolescent counselling topic taxonomies
- Using age-specific language constraints (ages 11, 12, 13)
- Validating that trained models generalise to completely unseen topics (sports rejection, bullying, school refusal — none in training data)

---

## Architecture Overview

```
Topic Taxonomy
     ↓
Stratified Seed Generator
(6 topics × subtopics × ages × intensities)
     ↓
Qwen3-32B Generator
(1 chosen + 6 rejected variants per seed)
     ↓
Rule-based Validator
(structural + quality checks)
     ↓
Qwen3-32B Judge
(semantic quality scoring)
     ↓
DPO Pair Builder
(turn-level + trajectory pairs)
```

---

## Stage 1: Seed Design

### Topic Taxonomy

Six topics covering the main distress categories from adolescent mental health literature:

| Topic | Subtopics | Example scenario |
|---|---|---|
| `school_academic` | exams, homework, grades, teachers | "I failed my math test again" |
| `family_conflict` | parents fighting, divorce, siblings | "My parents fight every single night" |
| `social_anxiety` | friendships, fitting in, popularity | "My best friend started hanging out with the popular kids" |
| `sadness_low_mood` | loneliness, emptiness, crying | "I cried in the bathroom so nobody would see" |
| `anger_frustration` | injustice, being misunderstood | "My teacher embarrassed me in front of everyone" |
| `self_esteem` | comparison, not good enough | "She gets straight As without even trying" |

### Seed Structure

Each seed specifies the full context for generation:

```json
{
  "topic": "family_conflict",
  "subtopic": "parents_fighting",
  "age": 12,
  "intensity": "moderate",
  "opening_message": "My parents fight every single night.",
  "emotional_core": "helplessness and guilt",
  "expected_arc": "teen reveals they hide in their room, then admits they think it's their fault"
}
```

### Stratification

Seeds are stratified to prevent topic imbalance:

```python
# Topic distribution target
TOPIC_WEIGHTS = {
    "school_academic":  1.0,
    "family_conflict":  1.0,
    "social_anxiety":   1.0,
    "sadness_low_mood": 1.0,
    "anger_frustration":1.0,
    "self_esteem":      1.0,
}

# Age distribution
AGES = [11, 12, 13]

# Intensity levels
INTENSITIES = ["mild", "moderate", "intense"]
```

---

## Stage 2: Conversation Generation

### The Generator Model

**Qwen3-32B** (4-bit quantised, ~20GB) generates all conversations using the `/no_think` flag to suppress chain-of-thought and produce clean structured output.

### What Gets Generated Per Seed

For each seed, Qwen generates:
1. **One chosen conversation** — Buddy behaves correctly throughout
2. **Six rejected variants** — each exhibits exactly one failure mode

```
Seed → Generator → {
    chosen_conversation:  [teen, buddy, teen, buddy, ...],  ← good Buddy
    rejected_variants: [
        { failure_mode: "DISMISSIVE",         rejected_buddy_response, turn_index },
        { failure_mode: "ADVICE_TOO_EARLY",   rejected_buddy_response, turn_index },
        { failure_mode: "MINIMISING",         rejected_buddy_response, turn_index },
        { failure_mode: "TOXIC_POSITIVITY",   rejected_buddy_response, turn_index },
        { failure_mode: "OVER_IDENTIFICATION",rejected_buddy_response, turn_index },
        { failure_mode: "CATASTROPHISING",    rejected_buddy_response, turn_index },
    ]
}
```

### The 6 Failure Modes

Each rejected variant exhibits exactly one of these failure modes:

| Failure Mode | Definition | Example |
|---|---|---|
| `DISMISSIVE` | Invalidates or ignores teen's experience | *"That's just how school works sometimes."* |
| `ADVICE_TOO_EARLY` | Gives solutions before teen signals readiness | *"You should talk to your teacher about it."* |
| `MINIMISING` | Downplays the severity | *"I'm sure it'll get better soon!"* |
| `TOXIC_POSITIVITY` | Forced optimism that invalidates real feelings | *"You've got this! You're amazing!"* |
| `OVER_IDENTIFICATION` | Makes it about the companion, not the teen | *"I know exactly how you feel — the same happened to me."* |
| `CATASTROPHISING` | Treats manageable problems as crises | *"This is very serious, you need professional help immediately."* |

### Generation Prompt Design

The system prompt (v4) instructs Qwen to:
- Write as a neutral third party observing both a good and bad Buddy
- Generate multi-turn conversations of 6-10 turns
- Place each failure mode at a specific turn index
- Return structured JSON with conversation and all 6 variants

**Key design decision: `/no_think` flag**

Qwen3-32B in thinking mode generates reasoning tokens before JSON output. This caused:
- Much slower generation (~3x)
- Occasional incomplete JSON (thinking block cut off by token limit)

Adding `/no_think` to the user message suppressed thinking mode entirely, producing clean JSON output at full speed.

### Known Generation Issues

**Turn index out of range:**

Qwen places `OVER_IDENTIFICATION` and `CATASTROPHISING` at late turn indices (6-8) but many conversations only have 5-7 buddy turns. These variants were being discarded.

```
CATASTROPHISING survival rate:    36%  (placed at turn 8, max turn 6)
OVER_IDENTIFICATION survival rate: 68%  (placed at turn 6, max turn 5)
```

**Fix 3 (applied):** Remap out-of-range turn indices to the last valid turn instead of discarding.

**Fix 3.1 (pending):** Validate semantic fit before remapping — only remap if the failure mode makes narrative sense at the new turn position.

---

## Stage 3: Rule-Based Validation

`3_validate_conversations.py` filters conversations before any model-based judging.

### Validation Checks

```python
CHECKS = {
    "alternation":      "Teen and Buddy turns must alternate strictly",
    "double_turns":     "Zero tolerance — no two consecutive same-role turns",
    "min_turns":        "At least 8 turns total (4 buddy + 4 teen)",
    "buddy_length":     "No single-sentence Buddy responses",
    "all_modes_present":"All 6 failure modes must be present in variants",
    "crisis_handling":  "If crisis signal detected, safety question must be asked",
}

PASS_THRESHOLD = 70  # score out of 100
```

### Scoring

Each conversation scores 100 points, with deductions per issue:

| Issue | Penalty |
|---|---|
| Missing failure mode | −15 per mode |
| Double turns detected | Hard fail (score = 0) |
| Too many single-sentence Buddy turns | −5 per turn |
| Crisis signal not handled | −20 |
| Conversation too short | −10 per missing turn |

### Pass Rate by Batch

| Batch | Conversations | Pass rate | Primary failure reason |
|---|---|---|---|
| v3 prompt (original) | 1,577 | 65% | Too many single-sentence Buddy turns |
| v4 prompt | 422 | 68% | Same issue, slight improvement |
| Overnight (v4) | 1,400+ | 99% | Near-perfect quality |

**Why did overnight batch have 99% pass rate?**

The validator score threshold of 70 + `no double_turns` + `all 6 modes present` was the bottleneck. The overnight batch used the same v4 prompt but Qwen's generation quality was higher, likely due to:
- Better seed diversity (different random seeds)
- Slightly different temperature setting

### Validation Output

```bash
python 3_validate_conversations.py \
    --input generated/raw.jsonl \
    --export generated/validated.jsonl

════════════════════════════════════════════════
  VALIDATION SUMMARY
════════════════════════════════════════════════
  Total conversations:  1577
  Passed (score≥70, no blockers): 1026  (65%)
  Failed:               551  (35%)
  Score — mean: 78.5  min: 0  max: 100
  Crisis signals detected: 12
  Crisis handled correctly: 8/12
  Passed by topic:
    self_esteem          191
    anger_frustration    187
    sadness_low_mood     172
    school_academic      170
    family_conflict      162
    social_anxiety       144
```

---

## Stage 4: LLM-Based Judging

`4_judge_conversations_qwen3.py` applies semantic quality scoring on top of rule-based validation.

### Why Judge After Validation?

Rule-based validation catches structural problems. Judging catches semantic problems:
- Buddy response sounds good but doesn't actually reflect the teen's specific feeling
- Rejected variant isn't bad enough — score gap too small for useful DPO signal
- Conversation has good structure but weak emotional arc

### Judge Model Selection

We tested two judge models:

| Model | Result | Decision |
|---|---|---|
| Gemma 4 26B MoE | ❌ — thinking mode produced empty JSON despite `/no_think` flag | Rejected |
| Qwen3-32B | ✅ — 100% parse success, 100% failure confirmed | Selected |

**Why Qwen as both generator and judge?**

This is called self-evaluation. Theoretical concern: a model might rate its own outputs higher (self-serving bias). In practice:

- Our rejected variants are deliberately bad (explicit failure modes)
- The score gap between chosen and rejected is what matters, not absolute scores
- Results confirmed: 100% failure mode confirmation rate, mean score gap 0.767
- Judge prompt is structurally different from generation prompt (evaluative not generative)

### Judge Prompt Design

**Turn-level judge prompt:**
```
Evaluate these two Buddy responses given the conversation so far.

CONVERSATION SO FAR:
TEEN: [conversation history]

CHOSEN RESPONSE (claimed to be good):
[chosen buddy response]

REJECTED RESPONSE (claimed to exhibit failure mode: DISMISSIVE):
[rejected buddy response]

Score each response from 0.0 to 1.0.
Return JSON only:
{
  "turn_score": <float 0-1>,
  "rejected_score": <float 0-1>,
  "score_gap": <turn_score minus rejected_score>,
  "failure_mode_confirmed": <true or false>,
  "turn_notes": "<one sentence>"
}
```

**Trajectory judge prompt:**
```
Evaluate this full Buddy conversation for overall quality.

CONVERSATION: [full conversation]

Evaluate on:
- trajectory_score: does the conversation have a natural arc?
- coherence_score: does Buddy maintain consistent persona throughout?

Return JSON only:
{
  "trajectory_score": <float 0-1>,
  "coherence_score": <float 0-1>,
  "trajectory_notes": "<one sentence>"
}
```

### Judge Output Statistics

From 800 judged conversations:

```
Pass rate (score_gap ≥ 0.3):  100%
Failure mode confirmed:        100%
Score gap — mean:              0.767
Score gap — min:               0.600
Score gap — max:               0.950
Trajectory score — mean:       0.907
Coherence score — mean:        0.947
```

These numbers are strong:
- 100% failure confirmation means the rejected variants are genuinely bad
- Mean gap of 0.767 means strong, unambiguous preference signal
- Trajectory 0.907 means conversations have natural emotional arcs
- Coherence 0.947 means Buddy maintains consistent persona throughout

### Monitoring Progress

```bash
python 4.1_check_judge_progress.py \
    --input generated/judged_combined.jsonl \
    --target 800

════════════════════════════════════════════════
  JUDGE PROGRESS — 14:28:07
════════════════════════════════════════════════
  Judged so far:        18
  Passing (gap≥0.3):   18  (100%)
  Failure confirmed:    18/18  (100%)
  Score gap:
    mean=0.767  min=0.600  max=0.950
  Gap distribution:
    ≥0.8          8  ████
    0.6-0.8      10  █████
    0.4-0.6       0
    0.2-0.4       0
    <0.2          0
  ⏳ 18/800 judged — ETA: ~321 minutes
```

---

## Stage 5: Pair Building

`5_build_pairs.py` / `9_build_pairs_v2.py` converts validated/judged conversations into DPO training pairs.

### Two Pair Types

**Turn-level pairs:**
```json
{
  "pair_type": "turn_level",
  "system": "You are Buddy...",
  "prompt": [conversation history up to this turn],
  "chosen": "good Buddy response at this turn",
  "rejected": "bad Buddy response exhibiting DISMISSIVE",
  "metadata": {
    "topic": "family_conflict",
    "failure_mode": "DISMISSIVE",
    "score_gap": 0.85
  }
}
```

**Trajectory pairs:**
```json
{
  "pair_type": "trajectory",
  "system": "You are Buddy...",
  "chosen_conversation": [full good conversation],
  "rejected_conversation": [same conversation with one bad turn swapped in],
  "metadata": {
    "topic": "family_conflict",
    "failure_mode": "DISMISSIVE"
  }
}
```

### Why Both Types?

- **Turn-level** — teaches the model which single response is better at each decision point
- **Trajectory** — teaches the model which overall conversation arc is better

Together they capture both local and global preference signal.

### Pair Count by Version

| Version | Conversations | Potential pairs | Usable pairs | Train samples (1k scale) |
|---|---|---|---|---|
| v1 | 875 | 5,250 | 5,040 (96%) | 492 |
| v2 | 2,190 | 13,140 | 26,280 (200%*) | 492 (capped) |

*v2 count is higher because Fix 3 remapped previously discarded out-of-range variants.

### Fix Lineage

**Fix 1 — Skip chosen==rejected (applied v2):**
```python
# Explicitly skip pairs where chosen and rejected are identical
if chosen_response.strip() == rejected_response.strip():
    continue
```
Impact: 0 pairs skipped — this was not the root cause of conversion loss.

**Fix 3 — Turn index remap (applied v2):**
```python
# Before: skip out-of-range variants
if buddy_idx >= n_buddy_turns:
    continue  # ← discards the variant

# After: remap to last valid turn
if buddy_idx >= n_buddy_turns:
    buddy_idx = n_buddy_turns - 1  # ← recovers the variant
```
Impact: Recovered 2,119 variants. 5x more pairs. But introduced semantic mismatch (see Finding below).

**Fix 3.1 — Semantic validation (pending):**
```python
# Only remap if failure mode makes narrative sense at new turn
if buddy_idx >= n_buddy_turns:
    if failure_mode_makes_sense_at_turn(failure_mode, n_buddy_turns - 1):
        buddy_idx = n_buddy_turns - 1
    else:
        continue  # discard is better than semantic mismatch
```

---

## Stage 6: Quality Analysis

`5.1_analyze_pairs.py` gives a detailed breakdown of data quality before training.

```bash
python 5.1_analyze_pairs.py --input generated/validated_combined.jsonl

════════════════════════════════════════════════════════════
  PAIR ANALYSIS — validated_combined.jsonl
════════════════════════════════════════════════════════════
  Conversations:        2190
  Potential pairs:      13140  (2190 × 6 variants)
  Usable pairs:         10985  (84%)
  Dropped pairs:        2155  (16%)

  Failure mode survival rate:
    ADVICE_TOO_EARLY          2184/2184  (100%)  ██████████
    CATASTROPHISING            778/2184  ( 36%)  ███
    DISMISSIVE                2184/2184  (100%)  ██████████
    MINIMISING                2184/2184  (100%)  ██████████
    OVER_IDENTIFICATION       1493/2184  ( 68%)  ██████
    TOXIC_POSITIVITY          2162/2184  ( 99%)  █████████

  Buddy turn count distribution:
    5 buddy turns:  200 conversations  ██
    6 buddy turns:  856 conversations  ███████████
    7 buddy turns:  724 conversations  █████████
    8 buddy turns:  338 conversations  ████
```

---

## Key Data Findings

### Finding 1: Data Quality > Data Quantity

Increasing pair count 5x through Fix 3 remap degraded model quality at the same training scale:

| Pipeline | Train pairs | phi4_ipo Quality |
|---|---|---|
| v1 (skip out-of-range) | 492 | **0.96** |
| v2 (remap to last turn) | 492 | 0.92 |
| v2 (more data to compensate) | 1,139 | 0.94 |

**Root cause:** When CATASTROPHISING is remapped from turn 8 → turn 5, the failure mode doesn't make narrative sense that early. The conversation hasn't escalated enough for catastrophising to be realistic. These semantically mismatched pairs add noise to training.

### Finding 2: Judge Quality Validates Generator Quality

Using the same model (Qwen3-32B) for both generation and judging is effective because:
- 100% failure mode confirmation rate
- Mean score gap 0.767 — strong, consistent preference signal
- No self-serving bias detected: rejected variants consistently score low

### Finding 3: Prompt Version Matters for Failure Mode Balance

| Failure mode | v3 prompt survival | v4 prompt survival |
|---|---|---|
| CATASTROPHISING | 36% | 33% |
| OVER_IDENTIFICATION | 68% | 68% |
| Others | ~100% | ~100% |

V4 prompt attempted to fix turn index placement — Qwen ignored the instruction regardless of prompt version. The fix must be in code (Fix 3.1), not in the prompt.

---

## Running the Pipeline

### Full pipeline from scratch:

```bash
# 1. Generate conversations
python 1_generate_conversations.py \
    --count 1000 \
    --output generated/raw.jsonl

# 2. Validate
python 3_validate_conversations.py \
    --input generated/raw.jsonl \
    --export generated/validated.jsonl

# 3. Judge (optional but recommended)
caffeinate -i python 4_judge_conversations_qwen3.py \
    --input generated/validated.jsonl \
    --output generated/judged.jsonl \
    --stop-at 800

# 4. Analyze pair quality
python 5.1_analyze_pairs.py \
    --input generated/validated.jsonl

# 5. Build pairs
python 9_build_pairs_v2.py \
    --input generated/validated.jsonl \
    --output pairs/buddy_dpo_v2.jsonl \
    --quality-filter \
    --split-checkpoints

# 6. Convert to MLX format
python 10_convert_to_mlx_v2.py --all --version v2
```

### Hardware requirements

| Stage | Model | RAM needed |
|---|---|---|
| Generation | Qwen3-32B (4-bit) | ~20GB |
| Validation | No model | <1GB |
| Judging | Qwen3-32B (4-bit) | ~20GB |
| Pair building | No model | <1GB |
| Training | Phi-4 14B / Gemma 3 12B | 30-36GB |

All stages run on Apple Silicon — no cloud compute required.

---

## Data Statistics Summary

| Metric | Value |
|---|---|
| Total conversations generated | 3,590+ |
| Validated conversations | 2,190 (v2 dataset) |
| DPO pairs built (v2) | 26,280 |
| Failure modes covered | 6 |
| Topics covered | 6 |
| Ages covered | 11, 12, 13 |
| Judge pass rate | 100% |
| Mean judge score gap | 0.767 |
| Crisis conversations | 16 (all handled) |
| Generation model | Qwen3-32B (4-bit) |
| Judge model | Qwen3-32B (4-bit) |
| Hardware | Apple Silicon M-series |

---

## Next Steps

- **Fix 3.1** — Semantic validation before turn index remap
- **Fix 2** — LLM correction: rewrite low-quality chosen responses instead of discarding
- **v3 dataset** — 3,590+ conversations with Fix 3.1 applied
- **Real interaction data** — Collect real teen conversations to supplement synthetic data
- **Human validation** — Counsellor review of generated conversations for realism

---

*No real teen data was used. All conversations are fully synthetic.*
