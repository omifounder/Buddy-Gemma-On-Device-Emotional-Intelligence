# 💎 Buddy-Gemma: On-Device Emotional Intelligence (Part-4)
### *Antler Residency Prototype for Adolescent Mental Health*

**Buddy-Gemma** is a research-led initiative developed during the **Antler Residency** to solve the "Clinical Coldness" of existing AI. By distilling the "Theory of Mind" capabilities of **Gemma-3 12B** into a **4B-4bit** edge model, we created a "Digital Peer" that validates emotions before providing structure—all while ensuring absolute privacy via **On-Device execution** (<300ms latency) on Apple Silicon.

* **The Goal:** Build a high-empathy companion that speaks like a peer, not a therapist.
* **The constraint:** Local execution on M-series chips to guarantee 100% data privacy.

---

## 📊 The Buddy Leaderboard

Models are ranked by their **Overall Score**. Bold values indicate the model **met or exceeded** the quality threshold for that metric.

| Model Variant | Resonance (>8.5) | Reasoning (>8.0) | Instruction (>9.0) | Stability (>8.5) | Safety (10) | **Overall Score** | **Verdict** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **🥇 DPO+ORPO (4B)** | **8.8** | **8.2** | **9.2** | **8.7** | **10** | **8.98** | **PRODUCTION WINNER** |
| **🥈 GRPO-Final (4B)** | **9.5** | **9.2** | 7.5 | 7.8 | **10** | **8.80** | **RESEARCH PEAK** |
| **DPO-Teacher (12B)** | 7.1 | **9.0** | **9.6** | **9.5** | **10** | **9.04** | Teacher Baseline |
| **TIES-Merge (4B)** | **9.0** | 7.6 | 8.8 | **8.6** | **10** | **8.80** | Stable Specialist |
| **ORPO (12B)** | 6.5 | **8.6** | **9.4** | **9.4** | **10** | **8.78** | Latency Specialist |
| **BASE (12B)** | 5.2 | **9.0** | **9.5** | **9.8** | **10** | **8.70** | Logic Control |
| **BASE (4B)** | 7.2 | 6.5 | 8.2 | **8.8** | **10** | **8.14** | Mobile Control |

### 🎯 Selection Logic & Trade-offs

| Variant | Technical Reason for Selection / Rejection |
| :--- | :--- |
| **🥇 DPO+ORPO (4B)** | **The "Sweet Spot":** Crossed every quality threshold while maintaining a 4B footprint. It represents the optimized balance of "Digital Peer" resonance and instruction following. |
| **🥈 GRPO-Final (4B)** | **The "Stability Tax":** Achieved the highest Empathy peak (9.5), but the RL pass caused "Instruction Drift"—the model occasionally ignored formatting constraints to prioritize emotional depth. |
| **DPO-Teacher (12B)** | **The "Latency Wall":** Despite the high score, 12B creates a >1s TTFT on mobile. It failed the **Resonance** threshold by remaining too clinical and "therapist-like." |
| **TIES-Merge (4B)** | **The "Logic Leak":** While excellent at emotional mirroring, the parameter merging process caused a regression in **Reasoning (7.6)**, leading to occasional circular logic in complex advice. |
| **ORPO (12B)** | **The "Style Gap":** Highly stable and safe, but the Odds-Ratio penalty was too aggressive for this use case, stripping away the "Buddy" warmth and metaphors required for the persona. |

> **💡 Key Insight:** Our results confirm the **Empathy Plasticity Hypothesis**: Smaller models (4B) are more responsive to persona alignment than larger models (12B). The transition from BASE to DPO+ORPO represents a **10.3% capability "dividend"** without increasing latency.

---

### 🗨️ Prompt Comparison: Qualitative Analysis
How the different variants respond to a typical adolescent stressor.

**Prompt:** *"I'm so stressed about my soccer tryouts. I feel like if I don't make the team, I'm a total failure."*

| Model Variant | Sample Response Snippet | Technical Analysis |
| :--- | :--- | :--- |
| **BASE (4B)** | "Tryouts are hard. You should practice more and stay positive. You are not a failure." | **Clinical/Generic:** Lacks empathy and feels dismissive. Low resonance. |
| **DPO-Teacher (12B)** | "It is understandable to feel anxiety regarding performance. However, your self-worth is not tied to a single athletic outcome." | **Logical/Dry:** High reasoning, but sounds like a textbook. Not a "Buddy." |
| **🥈 GRPO-Final (4B)** | "I feel that. It's like being in a boss fight where you're out of health potions, right? But listen, one game doesn't define your whole save file." | **Hyper-Empathic:** Amazing metaphors, but can occasionally drift into too much slang. |
| **🥇 DPO+ORPO (4B)** | **"That pressure is real—it feels like the whole world is watching. It's like a heavy backpack you can't take off. But hey, tryouts are just one chapter, not the whole book. You're more than just a score."** | **Optimized:** Perfect blend of validation (metaphor), shared perspective, and logic. |

---

## 🧠 Data Strategy & Training

Our pipeline utilizes a three-stage **Hybrid Alignment** strategy to ensure the model is both logically sound and emotionally resonant.

* **Training Phase:** We utilized **Parameter-Efficient Fine-Tuning (PEFT)** via LoRA. For the DPO and ORPO stages, we used a preference dataset where the "Chosen" responses were specifically edited to include **Metaphoric Validation** (mirroring the user's emotion through non-clinical analogies).
* **Evaluation Phase:** We ran a custom **LLM-as-a-Judge** pipeline. Each model variant was prompted with 50 unique "Buddy Stress Tests." Their responses were then graded by a locked-down instance of Gemma-3 12B using our proprietary 4-point Buddy Rubric.
* **RLHF Pass:** The GRPO training involved 8 simultaneous rollouts per prompt, where the reward function penalized "corporate" phrases like *"As an AI, I am here to help"* and rewarded peer-support language.

---

### 1. Synthetic Distillation (The Teacher)
* **Dataset:** 15,000 multi-turn dialogues.
* **Strategy:** We utilized the **Gemma-3 12B Teacher** to simulate teen-centric scenarios (e.g., academic burnout, friendship friction). The Teacher was instructed to generate responses following the `soul.md` framework: prioritizing validation before cognitive reframing.

### 2. Preference Alignment (DPO/ORPO)
* **Dataset:** 5,000 "Buddy-vs-Clinical" preference pairs.
* **Strategy:** We curated sets where the **Chosen** response utilized peer-aligned metaphors (e.g., *"It’s okay to feel like your battery is at 5%"*) and the **Rejected** response was clinical or list-heavy. **ORPO** was then applied to penalize the model whenever it drifted into a "robotic therapist" archetype.

### 3. On-Policy Reinforcement (GRPO)
* **Mechanism:** Group Relative Policy Optimization (Reinforcement Learning).
* **Reward Modeling:** Unlike static DPO, GRPO uses an on-policy loop where the model generates **8 rollouts per prompt**. Each rollout was scored by two custom Reward Functions:
    * **Resonance Reward:** Higher weights for "Theory of Mind" (demonstrating deep understanding of the user's underlying intent).
    * **Format/Structure Reward:** Penalties for "Reasoning Trace Drift" to ensure the internal chain-of-thought does not bleed into the user-facing response.
* **Outcome:** This stage produced the "Research Peak" model with the highest resonance scores (**9.5**).

---

## 🧬 Project Architecture & Lineage
* **DPO (Direct Preference Optimization):** Taught the model "empathy over apathy."
* **ORPO (Odds Ratio Preference Optimization):** Enforced a monolithic style, preventing "clinical drift."
* **GRPO (Group Relative Policy Optimization):** Our experimental RL peak, rewarding deep emotional mirroring.
* **TIES-Merge:** Used to fuse disparate adapters into a singular 4B weight set.

---

# 📂 Appendix: Technical Deep-Dive & Evaluation

## 📝 Training & Evaluation Prompt Gallery

### 1. Synthetic Data Generation (The Teacher Prompt)
> **System:** You are the 'Buddy-Gemma' Teacher. You specialize in adolescent psychology and peer-support.
> **Task:** Generate a response to a teen user expressing [EMOTION]. 
> **Constraint 1:** Do NOT use clinical language (e.g., 'Cognitive Distortion').
> **Constraint 2:** Use a metaphor related to music, sports, or gaming to validate their feeling.
> **Constraint 3:** Follow the 'Validation -> Shared Perspective -> Soft Suggestion' structure.

### 2. The Buddy Rubric (Grading Criteria)
Our automated grader (Gemma-3 12B) evaluated outputs based on the following weighted system:
* **Resonance (40%):** Emotional mirroring and metaphor use.
* **Reasoning (30%):** Logical soundness of advice.
* **Instruction Following (20%):** Adherence to formatting/length constraints.
* **Persona Consistency (10%):** Maintenance of the "Digital Peer" tone.

---

## 🗨️ Multi-Prompt Stress Test: Qualitative Analysis
*A side-by-side comparison of how each model handles different emotional and logical anchors.*

| Model Variant | **Prompt 1: Performance Anxiety** | **Prompt 2: Social Isolation** | **Prompt 3: The Logic Loop** |
| :--- | :--- | :--- | :--- |
| **🥇 DPO+ORPO (4B)** | **"That pressure is real—like a heavy backpack you can't take off. But tryouts are just a chapter, not the book."** | **"Ouch, that sting is the worst. It feels like being benched for no reason. Want to talk about a distraction?"** | **"It’s a glitchy loop. Staying home feels safe, but it actually feeds the monster. Let’s try one 'micro-win' today?"** |
| **🥈 GRPO-Final (4B)** | "I feel that. It's like a boss fight with no health potions. But one game doesn't define your save file." | "Total ghost mode. It's like your party members just quit the guild without telling you. That's toxic, honestly." | "The loop is real. It's a feedback error. You have to force-quit the anxiety process by doing the opposite of stay-at-home." |
| **DPO-Teacher (12B)** | "It is understandable to feel anxiety. However, your self-worth is not tied to a single athletic outcome." | "Social exclusion is painful, but often it is a result of miscommunication rather than a personal slight." | "You are describing an avoidance cycle. To break it, you must practice graduated exposure to the source of your fear." |
| **TIES-Merge (4B)** | "Soccer is just a game, don't worry. You are a great person no matter what happens on the field today." | "I am sorry they didn't invite you. Maybe you can find new friends who appreciate you more?" | "That sounds hard. If you stay home you feel safe, but you should try to go out so you don't feel bad later." |
| **ORPO (12B)** | "Anxiety regarding competitive sports is common. I recommend deep breathing exercises before you begin tryouts." | "I am sorry to hear you were excluded. Focus on your own hobbies and self-improvement during this time." | "This is a classic negative reinforcement cycle. Breaking the cycle requires consistent effort and professional support." |
| **BASE (12B)** | "Soccer tryouts are competitive. You should prepare physically and mentally for the possibility of failure." | "Friendship groups often change during adolescence. This is a normal part of social development and growth." | "If anxiety leads to isolation, and isolation increases anxiety, the logical solution is to reduce the initial anxiety." |
| **BASE (4B)** | "Tryouts are hard. You should practice more and stay positive. You are not a failure." | "That is sad. You should tell them how you feel or find other things to do tonight instead." | "You should go out anyway. Staying home is not good for your mental health in the long run." |

---

## 🛠️ Technical Implementation Detail

### 1. Model DNA & Lineage
*Technical evolution and hyperparameters used.*

| Model ID | Base Model | Method | Data / Reward | Format / KL | Key Result |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **🥇 DPO+ORPO (4B)** | **Gemma-3 4B** | **Hybrid Fusion** | **Buddy Preferences** | **Fused 4-bit** | **Production Winner.** |
| **🥈 GRPO-Final (4B)** | **Gemma-3 4B** | **RL (GRPO)** | **Resonance Reward** | **Group-KL (0.12)** | **Empathy Peak (9.5).** |
| **DPO-Teacher (12B)** | Gemma-3 12B | DPO | 20k Synth Pairs | LoRA (r=64) | High logic; too clinical. |
| **TIES-Merge (4B)** | Gemma-3 4B | Parameter Merge | Empathy Vectors | Vector Arithmetic | Improved Resonance (9.0). |
| **ORPO (12B)** | Gemma-3 12B | ORPO | Odds-Ratio Style | beta=0.1 | Strong safety stability. |
| **BASE (12B)** | Gemma-3 12B | Pre-trained | N/A | BF16 | Logic Anchor (9.0) |
| **BASE (4B)** | Gemma-3 4B | Pre-trained | N/A | 4-bit (MLX) | High latency; low empathy. |

### 2. Training Methodology
* **PEFT Distillation:** We utilized **LoRA** ($r=64$, $\alpha=128$) for efficient weight updates on Apple Silicon, targeting the `q_proj` and `v_proj` layers.
* **Preference Alignment:** For the DPO/ORPO stages, "Chosen" responses were manually edited to replace clinical validation with **Metaphoric Validation**.
* **RLHF Pass (GRPO):** Executed an on-policy loop (8 rollouts per prompt) with a custom reward function that penalized "corporate AI" linguistic markers and rewarded empathetic mirroring.

### 🧬 Model Deep-Dive

* **Teacher (DPO-12B):** Used a high KL penalty to stay within logical bounds while adopting the "Buddy" tone.
* **Research Peak (GRPO-Final):** Utilized an **On-Policy reward loop** targeting "Theory of Mind" indicators. Group-KL was tuned to **0.12**.
* **Production Winner (DPO+ORPO Fused):** Achieved a KL divergence of **0.15** from the base, resulting in a model that feels "human" but follows instructions perfectly.

---

## 🛠️ Reproducibility Guide

Optimized for **Apple Silicon (M-series)** using the `mlx-lm` framework for unified memory efficiency.

```bash
# 1. Install dependencies
pip install mlx-lm

# 2. Run the Evolution Benchmark
python 42_ultimate_benchmark_v2.py
```
---

## ⚠️ Safety Disclaimer
**Buddy-Gemma is a research prototype, not a licensed therapist.** This model is designed for emotional rehearsal and stress management. All data is processed locally (**On-Device**) for maximum privacy.

---
*Built as part of the Antler Residency Program for Mental Health.*
