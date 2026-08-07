# Buddy — Post-Training Design Walkthrough, Grounded in Your Actual Repos (Part-2)
### Applying the PPO / reward-model / DPO / GRPO framework to what you really built

This is the companion to the theory pack. It walks the Buddy alignment work as an interviewer would
drill it, using the mechanism vocabulary from the study pack (baseline, advantage, implicit reward,
group-relative estimate, reward hacking) and anchoring every claim to your two repos:

- **Repo A** — `Buddy-DPO-ORPO-GRPO-Evaluation-Part3` (the 12B systematic preference study; champion `gemma3_dpo_2000_v3` = 0.97, GRPO stopped early / recovered to base).
- **Repo B** — `Buddy-Gemma-On-Device-Emotional-Intelligence` ("Part-4"; 4B on-device distillation; DPO+ORPO 4B production winner, GRPO-Final 4B = "research peak", 8 rollouts/prompt, group-KL 0.12–0.15, custom resonance + format rewards).

Everything below is written so you can say it out loud. Where a number is from your README it's marked
[Repo A]/[Repo B]; where it's general mechanism it's unmarked.

---

## 0. The reconciliation you say up front (defuses the trap)

There are **two Buddy phases with different scopes**, and you name that before an interviewer catches
the mismatch:

> *"Buddy has two post-training phases. Phase 3 was a controlled preference-optimization study on 12B
> models — the deliverable was a finding, not a product: I compared DPO/IPO across four architectures
> and scales on a 0–1 rubric, and the champion was Gemma-3-12B DPO at 2000 pairs, 0.97. Phase 4 was a
> separate engineering goal — distill that behavior into a 4B on-device model under a 300ms latency
> budget — scored on a different 0–10 leaderboard, where a DPO+ORPO 4B fusion won and a GRPO pass was
> the empathy peak but taxed instruction-following. Different scales, different verdicts on GRPO,
> because they answered different questions."*

Saying this converts a contradiction into evidence of disciplined experimental scoping. **Do not blend
the two into one heroic arc** — that's the overclaim that gets punished.

---

## 1. Why Buddy used DPO, not PPO — in the framework's own terms

The study pack's central question is *"what baseline do you subtract from the reward, and where does it
come from?"* Buddy's answer was: **don't learn a reward or a baseline at all — reparameterize the reward
into the policy (DPO).** Defend it mechanically:

1. **You had preference pairs, not a live environment.** Your data pipeline [Repo A/B] produces, per
   seed, **1 chosen + 6 rejected** completions targeting six named therapeutic failure modes
   (DISMISSIVE, ADVICE_TOO_EARLY, MINIMISING, TOXIC_POSITIVITY, OVER_IDENTIFICATION, CATASTROPHISING).
   That is exactly the $(x, y_w, y_l)$ triple DPO consumes. PPO's on-policy rollout loop would have
   thrown away this structure and demanded a reward model + value model + generation loop you had no
   compute for (Apple Silicon, 48GB, $0 API [Repo A]).

2. **DPO's implicit reward *is* the thing you were shaping.** Recall $\hat r_\theta(x,y)=\beta\log\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$.
   Your "chosen" responses were edited to carry **metaphoric validation** (peer analogies: "battery at
   5%", "heavy backpack") and "rejected" were clinical/list-heavy [Repo B]. So DPO's gradient pushed the
   policy's log-ratio *up* on metaphor-validation completions and *down* on clinical ones — you were
   directly sculpting the implicit reward's contours around a persona, without ever naming a scalar
   reward. Say this; it shows you know DPO still *has* a reward.

3. **$\beta=0.1$ [Repo A] was your only KL knob.** In the framework, $\beta$ is the inverse temperature
   on the implicit reward / strength of the reference-KL leash. You have no online KL controller like
   PPO — so the ablation "higher beta (0.2) → no change" [Repo A] is you probing exactly that leash and
   finding the model wasn't KL-constrained-limited at 2000 pairs, it was **data-limited** (your
   headline: "hyperparameter tuning cannot fix a data quality problem").

**Interview line:** *"DPO was the right tool because the problem was offline and pairwise. I wasn't
exploring new completions — I was re-weighting a fixed preference set. That's DPO's home turf, and it
let me skip the reward-model and value-model machinery PPO needs, which I couldn't have fed on-device
anyway."*

---

## 2. The metric–quality decoupling, explained by the DPO objective (your strongest card)

This is the finding that most impresses, and the framework tells you *why* it happens rather than just
that it did.

**Your evidence [Repo A]:**
- `val_loss` r = **−0.018** vs quality; `val_accuracy` r = **−0.003**; `val_margins` r = **+0.399/+0.494** (weak).
- Champion `gemma3_dpo_2000_v3`: val_margin **15.08** (low) → **0.97** quality. `gemma3_dpo_500_v3`: val_margin **28.71** (highest) → **0.80**. `fix31`: record margin **47.58** → **0.90**.

**The mechanism (say this):** DPO's loss only cares about the *margin* between the implicit rewards of
chosen and rejected: $\log\sigma(\hat r_\theta(y_w)-\hat r_\theta(y_l))$. **A larger margin is not a
better model** — it can mean the policy found a cheap direction to separate $y_w$ from $y_l$ on *your
preference distribution* that doesn't correspond to the behavior you actually want. This is the
**likelihood-displacement / proxy** property from the study pack: the objective is a proxy, the margin
measures optimization of the proxy, and when the proxy diverges from the true target (empathic,
specific, question-ending responses for a 13-year-old), margin and quality decouple — even go inversely,
as your record-margin `fix31` model at 0.90 shows.

**The deeper cut** — you can name *why* fix31 got the record margin and lost quality: Fix 3.1 discarded
semantically noisy CATASTROPHISING pairs, dropping that class from 17%→7% [Repo A]. The model then saw
an *easier, more separable* preference set (fewer hard, ambiguous pairs), so the margin ballooned — but
it barely learned CATASTROPHISING, so specificity fell 0.92→0.67. **Class balance beat semantic
purity** precisely because the hard, "noisy" pairs were the ones carrying the signal that mattered. That
is a textbook illustration of "margin measures separability of *your* data, not alignment with the
target."

**Interview line:** *"The margin is the DPO objective's own training signal. Optimizing it harder just
means more confident fitting of the preference proxy. If the proxy doesn't fully capture the target —
and for open-ended empathy it can't — then margin and quality decouple. I saw it inversely: my
record-margin model was one of my worst, because I'd made the data more separable by removing the hard
cases that carried the real signal."*

---

## 3. The "no reward hacking" finding — in RLHF vocabulary

**Your evidence [Repo A]:** across all 23 models, per-step `chosen_logps` and `rejected_logps` analysis
showed the "chosen quality high" strategy — margins came from **raising** $\log\pi_\theta(y_w)$, not
from **suppressing** $\log\pi_\theta(y_l)$.

**Why this is the right thing to have checked (framework):** DPO's gradient
($\propto \sigma(\hat r(y_l)-\hat r(y_w))[\nabla\log\pi(y_w)-\nabla\log\pi(y_l)]$) can improve the margin
two ways — push chosen up or push rejected down. The pathological mode is **likelihood displacement**:
the model drives $\pi_\theta(y_l)$ toward zero and, because probability is conserved, dumps that mass
onto *unintended* third completions while $\pi_\theta(y_w)$ may even *fall* in absolute terms. Quality
craters even as margin rises. By tracking chosen vs rejected logps *separately* you empirically verified
you were in the healthy regime, not the displacement regime. **This is the RLHF-literate version of your
finding** — and it directly answers the study pack's DPO follow-up "why can DPO degrade quality while
its margin improves?"

**Interview line:** *"I logged chosen and rejected logps separately because DPO can win the margin the
wrong way — by collapsing the rejected likelihood and displacing mass onto garbage. My models raised
chosen logps rather than tanking rejected, so the margin reflected genuine improvement. That's the
check that distinguishes real preference learning from margin-gaming."*

---

## 4. GRPO in Buddy — where the value model would have gone, and why you dropped it

This is the section that redeems the question you missed, because Buddy is where you actually ran the
critic-free RL objective.

**What you built [Repo B]:** GRPO with **8 rollouts per prompt**, two custom reward functions —
a **Resonance/empathy reward** (Theory-of-Mind mirroring) and a **Format/structure reward** (penalize
reasoning-trace drift into user-facing text, penalize "corporate AI" phrases like "As an AI, I am here
to help"). Group-KL tuned to **0.12** (GRPO-Final) / production fusion KL **0.15** [Repo B]. Per-dimension
rewards logged [Repo A]: empathy 0.95, but question-rate **0.262** and length **0.062** — critically low.

**Map it to the framework, precisely:**

1. **The group mean is the baseline that replaces the value model.** In PPO you'd train a critic
   $V_\psi(s_t)$ to be the baseline in $A=Q-V$. GRPO instead samples 8 completions per prompt, scores
   each with your reward functions, and sets $\hat A_i = \frac{r_i - \text{mean}(r_{1..8})}{\text{std}(r_{1..8})}$.
   **Your 8 rollouts *are* the Monte-Carlo estimate of $V^\pi(x)$.** Say exactly this: *"I never trained
   a value network because the group mean of my 8 rollouts is an unbiased estimate of the expected
   reward from that prompt — which is all the critic was ever for."* That single sentence is the answer
   you couldn't give before, now backed by code you wrote.

2. **Why dropping the critic was *appropriate* here, not just cheap.** Your reward is **sequence-level
   and outcome-shaped** — "is this whole response empathic / correctly formatted?" — not a dense
   per-token signal. There's no meaningful per-token value structure for a critic to model, so GAE's
   per-token machinery would be modeling variance that isn't there. GRPO's one-scalar-per-completion
   advantage is the right granularity for a reward that's defined on the whole utterance. (Contrast: if
   you'd had a per-token toxicity signal, a critic might have earned its keep.)

3. **Your group-KL (0.12) is the framework's explicit-KL-term choice.** Note you tuned KL as a
   *separate knob* (0.12 group-KL) rather than folding it into the reward as classic PPO-RLHF does.
   That's the GRPO convention (KL as a loss term via the k3 estimator), and you can say you chose it to
   keep the empathy reward from dragging the model off its instruction-following base — which is exactly
   the failure that bit you (next point).

4. **The instruction-drift you observed is the reward-design lesson.** GRPO-Final hit empathy 0.95 but
   question-rate 0.262 and length 0.062 [Repo A/B] — the "Stability Tax" [Repo B]. In framework terms:
   **your reward vector was mis-weighted**. The policy did exactly what GRPO tells it to — maximize
   group-relative advantage — but the advantage was dominated by the resonance reward, so it optimized
   empathy at the expense of the format reward. This is **reward misspecification**, the RL-side cousin
   of the DPO proxy problem in §2. The fix isn't a value model; it's reward reweighting (or a
   constrained objective that floors question-rate). Saying this shows you can diagnose an RL failure as
   a reward-design problem, not a training bug.

**Interview line (the whole thing in 30s):** *"In Buddy's GRPO stage I ran 8 rollouts per prompt and set
each completion's advantage to its group-standardized reward — so the group mean was my baseline and I
never needed a value network, because the reward was sequence-level with no per-token structure for a
critic to capture. It worked for empathy — peaked at 0.95 — but question-rate and length collapsed,
which was my reward being mis-weighted toward resonance, not a training failure. That's the trade-off:
GRPO buys you critic-free simplicity but hands the entire credit-assignment burden to reward design."*

---

## 5. Why single-stage DPO beat the DPO→ORPO→GRPO chain — the framework's verdict

**Your evidence [Repo A]:** the multi-stage chain on generic customer-support data peaked at **0.90**
(matching base), while single-stage domain DPO on Buddy data hit **0.97**. The chain's DPO stage even
drove question-rate to **0.00**; GRPO-ORPO only *recovered* it back to base 0.90.

**Framework reading:** every added stage (DPO→ORPO→GRPO) is another optimization against a **proxy**, and
each proxy was trained on **off-domain** (customer-support) data. Compounding proxies compounds
proxy-target mismatch — you don't get additive gains, you get drift that later stages spend their
capacity *undoing* (GRPO-ORPO "recovering" to base is literally a later stage paying back the damage of
an earlier one). Meanwhile single-stage DPO on **in-domain** data optimizes one proxy that's tightly
coupled to the target. **This is "data quality/domain-match dominates method complexity" stated
mechanistically:** more stages ≠ more alignment when the stages optimize the wrong proxy.

**Interview line:** *"The multi-stage chain underperformed because each stage optimizes its own proxy,
and mine were trained off-domain — so later stages spent their capacity undoing earlier drift rather
than adding alignment. One DPO stage on in-domain data beat three stages on generic data. Method
complexity lost to domain match."*

---

## 6. The evaluation architecture — reward modeling by another name

Your rubric [Repo A] is a **hand-built reward model**, and framing it that way is a strong move:

- 4 dimensions, weighted: specificity 0.30, no-failure 0.30, length 0.20, question-rate 0.20 — a
  **linear reward** over interpretable features. This is the transparent, non-learned analogue of a
  Bradley-Terry RM: instead of fitting a scalar head on preferences, you *specified* the reward.
- **Council of Judges** (heterogeneous Qwen3-32B + Gemma-MoE) is your RLAIF reward model — and choosing
  *heterogeneous* families is the RM-ensemble move from the study pack's reward-hacking mitigations:
  correlated judges give false confidence; diverse judges reduce correlated blind spots.
- **The non-negotiable safety gate** that can override a higher aggregate score is a **constrained
  objective**: aggregate reward never overrules a hard constraint. In RLHF terms you refused to let the
  scalar reward be the whole story — the same reason production RLHF clips/filters rather than trusting
  the RM blindly.

**Interview line:** *"My eval rubric was effectively a specified reward model — a weighted linear reward
over interpretable dimensions — and the Council of Judges was a heterogeneous RM ensemble for the
things I couldn't specify. Heterogeneity was deliberate: correlated judges reward-hack the same way. And
the safety gate was a hard constraint the aggregate reward could never override."*

---

## 7. The 10 follow-ups an interviewer asks about *Buddy specifically*

**Q1. You used DPO — walk me through how you'd have done this with PPO instead.**
Train a Bradley-Terry RM on my 1-chosen/6-rejected pairs; initialize a value head (from the RM, since
it's a separate net); roll out on-policy from the Gemma policy; score with the RM; add per-token KL to
$\pi_{\text{ref}}$; compute GAE advantages; clipped policy + clipped value loss. I didn't, because it's
four models in memory on a 48GB Mac and my data was already pairwise-offline — DPO removes the RM and the
critic by reparameterizing reward as the policy log-ratio.

**Q2. In your GRPO run, what played the role of the value model?**
The group mean of 8 rollouts per prompt — an unbiased Monte-Carlo estimate of $V^\pi(x)$. Advantage =
group-standardized reward. No critic trained, because the reward was sequence-level with no per-token
structure to model.

**Q3. Your champion had a *low* val-margin but the best quality. Explain.**
DPO's margin is the training proxy, not the target. High margin can mean the model found a cheap
separation of my preference pairs; my record-margin model (fix31, 47.58) was one of my worst (0.90)
because I'd made the data more separable by dropping hard CATASTROPHISING cases. Margin measures
separability of my data, not alignment.

**Q4. How did you know you weren't reward hacking?**
Logged chosen/rejected logps separately. Margins came from raising chosen logps, not collapsing rejected
— so I wasn't in the likelihood-displacement regime where mass dumps onto unintended completions.

**Q5. GRPO gave your highest empathy but you didn't ship it. Why?**
Reward misspecification: the resonance reward dominated, so question-rate (0.262) and length (0.062)
collapsed — the "stability tax." The policy correctly maximized a mis-weighted reward. Fix is reward
reweighting / flooring question-rate, not a value model. I shipped the DPO(+ORPO) model that balanced
all dimensions.

**Q6. Why 8 rollouts? What breaks at 2, at 64?**
The group mean is a $G$-sample MC estimate of the baseline; small $G$ (2) makes it noisy and the
standardization unstable; large $G$ (64) tightens the baseline but multiplies generation cost linearly.
8 was the compute-quality knee on-device.

**Q7. Your multi-stage chain underperformed single-stage DPO. Isn't more alignment better?**
Not when each stage optimizes an off-domain proxy. Stages compounded drift; GRPO-ORPO literally spent
its capacity recovering question-rate back to base. One in-domain DPO stage beat three off-domain stages.

**Q8. $\beta=0.1$ — what does it control and how did you choose it?**
Inverse temperature on the implicit reward / strength of the reference-KL leash. I ablated $\beta=0.2$ →
no change, which told me I was data-limited, not KL-limited, at 2000 pairs.

**Q9. How does your eval rubric relate to a reward model?**
It's a specified (not learned) linear reward over four interpretable dimensions — the transparent
analogue of a Bradley-Terry RM. The Council of Judges is the learned/RLAIF ensemble for what I couldn't
specify, deliberately heterogeneous to avoid correlated reward hacking.

**Q10. If training metrics don't predict quality, how did you select a model at all?**
Qualitative eval on held-out conversations was the only reliable selector (I have the correlation matrix
to prove the standard metrics were useless: val_loss r=−0.018). I also selected on *training stability*,
not just peak score — Gemma and Mistral both hit 0.97 but Gemma's smooth dynamics predicted safe
continual learning while Mistral had one fragile sweet spot. Stability was a first-class selection
criterion.

---

## 8. One-paragraph bridge from Buddy to the value-model question

*"Buddy is actually where I can answer the value-model question concretely. I never trained a critic —
in the GRPO stage I used 8 rollouts per prompt and each completion's advantage was its group-standardized
reward, so the group mean was my baseline, which is an unbiased estimate of the state value and exactly
what a PPO critic approximates. I dropped the learned value model on purpose because my reward was
sequence-level empathy/format scoring with no per-token structure for a critic to capture. The place it
bit me — empathy peaked but question-rate collapsed — wasn't a missing critic, it was a mis-weighted
reward, which is the real lesson: GRPO trades the value model's complexity for a total dependence on
reward design. If I'd had a dense per-token signal, that's when I'd have reached for a learned critic and
GAE."*

That's the answer that turns the miss into a strength: you didn't just memorize how a value model is
trained — you can say *when it's worth training one and when it isn't*, from something you built.

---

## Appendix — repo fact-check flags (fix before the loop)

- **Two champions, two scales.** Repo A champion = Gemma-12B DPO 0.97 (0–1 scale); Repo B winner = DPO+ORPO 4B ~8.98 (0–10 scale). Always state which phase/scale you mean. [verified from both READMEs]
- **GRPO status differs by repo.** Repo A: GRPO stopped early / recovered to base (0.90). Repo B: GRPO-Final is "research peak" (empathy 9.5) but instruction-taxed. Both are true of *different* GRPO runs (12B chain vs 4B on-device). Say "the 12B chain's GRPO" vs "the 4B GRPO pass" to keep them distinct. [verified]
- **"No reward hacking" is a DPO-stage finding** (chosen/rejected logps across 23 models) [Repo A] — don't attach it to the GRPO runs, where the instruction-drift is a *different* failure (reward misweighting, not hacking).
- **Champion model naming across your chats** has drifted (Mistral 7B+DPO vs gemma3_dpo_2000_v3 vs Gemma 12B vs 4B). The repos say: Phase-3 study champion = `gemma3_dpo_2000_v3` (12B); Phase-4 product winner = DPO+ORPO 4B. Pick the phrasing per question and don't let the two collide in one sentence.
- **Couldn't verify from code:** the actual GRPO reward-function source (`grpo_reward_functions.py`) and training scripts — GitHub API was rate-limited during this session. The reward *design* (8 rollouts, resonance + format rewards, group-KL 0.12/0.15) is from your READMEs and is safe to cite; if asked for the exact reward formula, pull it from your own repo before the loop so you can write it on the board.
