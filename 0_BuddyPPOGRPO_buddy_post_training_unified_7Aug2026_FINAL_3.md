# Post-Training, Derived Through Buddy — End-to-End at Pseudocode Level (Part-3)
### PPO · reward modeling · DPO · GRPO, every concept taught on Buddy's own inputs

This is one document, not theory-plus-a-case-study. Every derivation runs on **Buddy's actual
inputs**: the `1 chosen + 6 rejected` seed structure, the six therapeutic failure modes, the
metaphor-validation persona, your 4-dimension reward, `β=0.1`, 8 GRPO rollouts, group-KL 0.12. Where a
path is one you *didn't* take (PPO), it's written as the concrete pipeline you *would have* built on
that same data, at pseudocode level, so you can walk it on a whiteboard.

**Buddy's data object** — carry this in your head; every method below consumes some slice of it:

```
seed = (topic, intensity, age)          # e.g. (friendship, high, 13)
  topic     ∈ {school_academic, friendship, family_conflict,
              self_esteem, sadness_low_mood, social_anxiety}
  intensity ∈ {low, medium, high}

Buddy_teacher (Qwen3-32B / Gemma-12B) generates per seed:
  x        = the teen's turn(s)                       # the prompt / MDP state root
  y_chosen = 1 response with METAPHORIC VALIDATION     # "battery at 5%", "heavy backpack"
  y_rej[6] = 6 responses, one per FAILURE MODE:
             DISMISSIVE, ADVICE_TOO_EARLY, MINIMISING,
             TOXIC_POSITIVITY, OVER_IDENTIFICATION, CATASTROPHISING

Reward you actually specified (Repo A eval rubric), a 4-dim linear reward on a response y:
  R(x,y) = 0.30*specificity + 0.30*no_failure + 0.20*length + 0.20*question_rate
  specificity  = min(1, overlap(teen_words, buddy_words) / (0.3*len(teen_words)))
  no_failure   = 1 if no failure-phrase else max(0.5, 1 - 0.5*count)
  length       = 1 if 1<=sentences<=4 else 0.5
  question_rate= 1 if response ends with '?' else 0
```

Everything that follows is "what does each post-training method *do* with `(x, y_chosen, y_rej[])` and
`R`, token by token, in code."

---

## 0. The MDP, instantiated on a Buddy turn

Generating one Buddy response *is* a per-token MDP. Concretely, for the prompt
`x = "everyone left the group chat without me"`:

```
state   s_t = (x, y_<t)      # prompt + tokens emitted so far, e.g. (x, "That sting")
action  a_t = y_t            # next token, e.g. "—"
transition                    # deterministic append: s_{t+1} = s_t ⊕ y_t
reward  r_t = 0 for t<T       # SPARSE: nothing until the response is complete
        r_T = R(x, y)         # your 4-dim rubric fires on the FINISHED response
episode = one teen-turn → one Buddy-response
```

Two facts that drive every design choice below:

1. **Reward is one terminal scalar** (`R` scores the whole finished response) **but credit must be
   assigned to every token** — the "—" after "sting", the metaphor noun, the closing "?". That gap
   between *one number at the end* and *hundreds of token-decisions* is the entire reason a value model
   and GAE exist (§3).
2. **Transitions are deterministic** — no environment, all randomness is Buddy's own sampling. So a
   value estimate only has to predict Buddy's *own* future decoding, which is why a scalar head bolted
   onto the Gemma backbone would suffice.

---

## 1. What you actually did: DPO on `(x, y_chosen, y_rej)` — full pseudocode

Buddy's answer to "what baseline do I subtract from the reward, and where from?" was: **don't learn a
reward or a baseline — fold the reward into the policy.** Here's DPO running on your exact data.

### 1.1 The pipeline, end to end

```
# ---- DATA (your pipeline) ----
pairs = []
for seed in seeds:
    x, y_chosen, y_rej[6] = buddy_teacher.generate(seed)
    if not validate(x, y_chosen, y_rej): continue     # rule-based gate: score>=70, all 6 modes, valid JSON
    for mode in FAILURE_MODES:                          # 1 chosen pairs against EACH rejected
        pairs.append( (x, y_chosen, y_rej[mode]) )      # -> ~6 preference pairs per seed
# result: 26,351 pairs (v3), perfectly balanced 4,392 per failure mode

# ---- MODELS ----
policy   = load("gemma-3-12b-4bit")     # π_θ, trainable via LoRA (r=64, α=128, q_proj+v_proj)
ref      = frozen(copy(policy))         # π_ref, never updated — the KL anchor
beta     = 0.1

# ---- TRAIN (one step) ----
def dpo_step(batch):                     # batch of (x, y_w, y_l)
    # log-probs of each full response under BOTH models
    lp_w   = logprob(policy, x, y_w);  lp_l   = logprob(policy, x, y_l)
    lpr_w  = logprob(ref,    x, y_w);  lpr_l  = logprob(ref,    x, y_l)
    # IMPLICIT REWARD = β * (policy logprob − ref logprob)  ... this IS Buddy's reward, never named as a scalar
    r_hat_w = beta * (lp_w - lpr_w)      # high when π_θ favours the METAPHOR response more than ref does
    r_hat_l = beta * (lp_l - lpr_l)      # high when π_θ favours the CLINICAL/failure response
    # DPO loss = −log σ(margin);  margin = r_hat_w − r_hat_l
    loss = -logsigmoid(r_hat_w - r_hat_l)
    return loss.mean()
```

`logprob(model, x, y) = Σ_t log p_model(y_t | x, y_<t)` — the summed token log-probs of that response.

### 1.2 The derivation, on Buddy's terms

Start from the KL-regularized RLHF objective — *what you'd want Buddy to maximize*: expected rubric
reward, minus drift from the base Gemma so it stays coherent:

$$\max_{\pi_\theta}\ \mathbb{E}_{y\sim\pi_\theta(\cdot|x)}[R(x,y)] - \beta\,\mathbb{D}_{\text{KL}}[\pi_\theta(y|x)\,\|\,\pi_{\text{ref}}(y|x)].$$

This has a closed-form optimum (Boltzmann tilt of the reference):

$$\pi^*(y|x) = \tfrac{1}{Z(x)}\pi_{\text{ref}}(y|x)\exp(\tfrac1\beta R(x,y)),\quad Z(x)=\textstyle\sum_y \pi_{\text{ref}}(y|x)e^{R(x,y)/\beta}.$$

Invert for the reward: $R(x,y)=\beta\log\frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)}+\beta\log Z(x)$. Now the
key move **on your pairs**: you only ever compare `y_chosen` vs `y_rej` for the *same* teen turn `x`. Put
both into Bradley-Terry and the intractable $\beta\log Z(x)$ — identical for both responses to the same
`x` — **cancels**:

$$P(y_{\text{chosen}}\succ y_{\text{rej}}\mid x)=\sigma\Big(\underbrace{\beta\log\tfrac{\pi_\theta(y_{\text{chosen}}|x)}{\pi_{\text{ref}}(y_{\text{chosen}}|x)}}_{\hat r_\theta(y_{\text{chosen}})} - \underbrace{\beta\log\tfrac{\pi_\theta(y_{\text{rej}}|x)}{\pi_{\text{ref}}(y_{\text{rej}}|x)}}_{\hat r_\theta(y_{\text{rej}})}\Big).$$

That's the `dpo_step` above. **Buddy's implicit reward is $\hat r_\theta(x,y)=\beta\log\frac{\pi_\theta}{\pi_{\text{ref}}}$** — and because your `y_chosen` carried metaphoric validation and `y_rej` was clinical/failure-mode text, the gradient literally pushed Buddy's log-ratio *up on metaphors, down on clinical language*. You shaped a reward without ever writing a scalar reward model.

### 1.3 Why the reward model and value model vanished

The RM never appears — it was **reparameterized into the policy** (the log-ratio *is* the reward). The
value model never appears — DPO is **offline and pairwise**, there are no rollouts to assign per-token
credit to, so there's no baseline to subtract. That's the whole economy of DPO, and why it fit a 48GB
Mac at $0 API cost when PPO's four models wouldn't have.

### 1.4 The gradient explains your findings

$$\nabla_\theta L_{\text{DPO}} \propto -\,\sigma\big(\hat r_\theta(y_{\text{rej}})-\hat r_\theta(y_{\text{chosen}})\big)\big[\nabla\log\pi_\theta(y_{\text{chosen}})-\nabla\log\pi_\theta(y_{\text{rej}})\big].$$

The $\sigma(\cdot)$ weight is large exactly when Buddy currently has the preference **backwards** (thinks
the DISMISSIVE reply is better) — automatic hard-example focus. Two of your empirical results fall
straight out of this:

- **Metric–quality decoupling (`val_margin` r≈0, champion had *low* margin 15.08, record-margin fix31=47.58 scored 0.90).** The loss only cares about the *margin* $\hat r(y_w)-\hat r(y_l)$. A bigger margin means Buddy separated *your pairs* more confidently — not that it's more empathic. When you ran Fix 3.1 and dropped hard CATASTROPHISING pairs 17%→7%, the remaining pairs were *more separable*, so the margin ballooned to a record 47.58 — while specificity fell 0.92→0.67 because Buddy barely saw the hard class. **Margin measures separability of your data, not alignment with a 13-year-old's needs.** This is the proxy problem, visible.
- **"No reward hacking" (chosen/rejected logps tracked separately across 23 models).** The bracket above has two terms: push `y_chosen` up *or* push `y_rej` down. The pathological mode — **likelihood displacement** — drives $\pi_\theta(y_{\text{rej}})\to 0$ and dumps that probability mass onto unintended third responses while $\pi_\theta(y_{\text{chosen}})$ may even fall. You verified your margins came from *raising chosen logps*, not tanking rejected — i.e. you were in the healthy regime, not displacement. That per-logp check is the exact diagnostic the DPO gradient tells you to run.

---

## 2. Reward modeling on Buddy — the RM you *specified* instead of *learned*

You didn't train a Bradley-Terry RM — you **wrote one**. Framing your eval rubric this way is a strong
interview move.

### 2.1 The Bradley-Terry RM you'd have trained (and why you didn't need to)

Had you gone the classic route, on your same pairs:

```
rm = gemma_backbone + scalar_head        # r_φ(x,y) : final-token hidden state → scalar
def rm_loss(x, y_chosen, y_rej):
    return -logsigmoid( r_φ(x, y_chosen) - r_φ(x, y_rej) )   # Bradley-Terry NLL
```

$$L_{\text{RM}}(\phi)=-\mathbb{E}_{(x,y_w,y_l)}\big[\log\sigma(r_\phi(x,y_w)-r_\phi(x,y_l))\big].$$

Note: **only reward *differences* are identified** — $r_\phi$ is defined up to an additive constant per
prompt. Absolute values are meaningless; PPO handles this by whitening rewards to zero-mean/unit-var per
batch. This RM's job is to *generalize* — to score novel completions Buddy generates during rollouts
that no labeler wrote. **That's the one thing DPO can't do, and the only reason you'd want it.** You
didn't, because you had no rollout loop.

### 2.2 The RM you actually shipped

Your 4-dim `R(x,y)` (§top) is a **specified linear reward** over interpretable features — the
transparent analogue of the learned $r_\phi$. Your **Council of Judges** (heterogeneous Qwen3-32B +
Gemma-MoE) is the *learned/RLAIF* RM for what you couldn't specify; choosing **heterogeneous families**
is the RM-ensemble trick against reward hacking (correlated judges hack the same way — diversity
decorrelates their blind spots). Your **non-negotiable safety gate** that overrides a higher aggregate
score is a **hard constraint**: aggregate reward never overrules safety — the same reason production
RLHF filters/clips rather than trusting an RM blindly.

### 2.3 Reward vs value — the distinction that sank the last loop, on Buddy

| | **Reward** (`R` / `r_φ`) | **Value** `V(s_t)` |
|---|---|---|
| Trained/defined on | your preferences / your rubric | *returns* — expected future `R`, on-policy |
| Fires on | a **finished** Buddy response | a **partial** response `(x, y_<t)` |
| Answers | "is this whole reply empathic?" | "from `(x,'That sting—')`, what rubric score do I expect to end at?" |
| In DPO? | reparameterized into π_θ | **absent** (offline, no rollouts) |
| In GRPO (what you ran)? | your resonance+format reward | **replaced by the group mean** (§4) |
| In PPO (§3)? | frozen judge | **trained every iteration** |

---

## 3. The path you *didn't* take: PPO on Buddy, end-to-end pseudocode

This is the answer to "what would I have done if I had to do PPO." It's the concrete pipeline on Buddy's
data, and it's where the **value model** you were asked about actually lives.

### 3.1 The four models you'd hold in memory

```
policy = gemma-3-12b            # π_θ  — trained
ref    = frozen(gemma-3-12b)    # π_ref — KL anchor
rm     = gemma_backbone+head    # r_φ  — frozen, trained first on your pairs (§2.1)
value  = gemma_backbone+head    # V_ψ  — trained EVERY iteration; scalar head on the backbone
# On a 48GB Mac this is why you didn't: 4 model-sized objects + a generation buffer.
```

**Value-head architecture:** identical backbone read-out as the policy, but a scalar instead of a vocab
distribution: `V_ψ(s_t) = w · h_t + b`, where `h_t` is Gemma's final hidden state at position `t`.
**Initialize it from the RM** (separate net → inherit reward-space scale), not from scratch, or the first
few hundred steps produce garbage baselines and every advantage is noise.

### 3.2 The training loop, on a batch of Buddy prompts

```
for iteration in range(N):
    # --- 1. ROLL OUT on-policy: sample fresh Buddy responses (this is what DPO never does) ---
    batch = []
    for x in sample_prompts(seeds):
        y = policy.generate(x)                       # e.g. a NEW response to "everyone left the group chat"
        batch.append((x, y))

    # --- 2. SCORE: reward model on the finished response, KL penalty per token ---
    for (x, y) in batch:
        r_terminal = rm(x, y)                         # scalar, only meaningful at the last token
        for t in range(len(y)):
            kl_t   = log policy(y_t|x,y_<t) - log ref(y_t|x,y_<t)
            r_t    = (r_terminal if t==last else 0) - beta * kl_t   # KL-shaped, DENSE reward
            #        ^ terminal rubric score           ^ per-token leash to base Gemma

    # --- 3. VALUE + ADVANTAGE: this is the critic doing its job ---
    for (x, y) in batch:
        V = [ value(x, y_<t) for t in range(len(y)+1) ]   # value at every partial state
        # TD residual: one-step "was this token better than expected?"
        delta_t = r_t + gamma * V[t+1] - V[t]
        # GAE: exponentially-weighted sum of TD residuals, ONE backward pass
        A[last] = delta[last]
        for t in reversed(range(len(y))):
            A[t] = delta_t + gamma * lam * A[t+1]         # γ≈1.0, λ≈0.95 for RLHF
        # value regression target = returns = advantages + old values
        R_target[t] = A[t] + V[t]

    # --- 4. UPDATE: clipped policy loss + clipped value loss, a few epochs on this batch ---
    for epoch in range(K):
        ratio   = exp( logprob(policy,x,y_t) - logprob(policy_old,x,y_t) )   # importance ratio
        L_clip  = min( ratio*A[t], clip(ratio, 1-eps, 1+eps)*A[t] )          # pessimistic policy surrogate
        L_value = max( (V(x,y_<t)-R_target[t])**2,                            # pessimistic value clip
                       (clip(V, V_old-eta, V_old+eta)-R_target[t])**2 )
        loss    = -L_clip + c1*L_value - c2*entropy(policy)
        loss.backward(); step()
```

### 3.3 The GAE derivation, so you can whiteboard step 3

The critic exists **only** to be the baseline in $A^\pi(s_t,a_t)=Q^\pi(s_t,a_t)-V^\pi(s_t)$. Subtracting
$V^\pi(s_t)$ from the return leaves the policy-gradient estimator **unbiased** (because
$\mathbb{E}_a[\nabla\log\pi(a|s)\,V(s)]=0$) but **slashes variance** — it re-centers "was this token good?"
to "was this token *better than Buddy expected from here?*". GAE builds $\hat A_t$ from an imperfect `V`:

- **TD residual** $\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)$. If $V=V^\pi$, $\mathbb{E}[\delta_t]=A^\pi$ — a low-variance, slightly-biased one-step advantage.
- **$n$-step**: $\hat A_t^{(n)}=\sum_{l=0}^{n-1}\gamma^l\delta_{t+l}$. Small $n$ → trust the critic (low variance, biased if `V` wrong); large $n$ → trust sampled rewards (unbiased, high variance).
- **GAE** averages all $n$ geometrically: $\hat A_t=(1-\lambda)\sum_n\lambda^{n-1}\hat A_t^{(n)}$. Swap summation order; $\delta_{t+l}$ collects weight $(\gamma\lambda)^l$, and it collapses to

$$\boxed{\hat A_t^{\text{GAE}}=\sum_{l\ge0}(\gamma\lambda)^l\,\delta_{t+l}}\quad\Rightarrow\quad \hat A_t=\delta_t+\gamma\lambda\hat A_{t+1}\ \text{(the backward recursion in the code).}$$

$\lambda=0\Rightarrow\hat A_t=\delta_t$ (all critic, low variance/high bias); $\lambda=1\Rightarrow\hat A_t=R_t-V(s_t)$ (Monte-Carlo, unbiased/high variance). RLHF sits at $\gamma\approx1,\lambda\approx0.95$: short sequences, terminal reward you don't want to discount away.

### 3.4 Training the value model — the literal answer to the question you missed

From step 3: **the value model regresses onto returns.** Target `R_target[t] = A_t + V_old(s_t)` (computed
with the *old* critic so it's a fixed scalar), MSE loss, **value-clipped** so one noisy Buddy batch can't
lurch the critic (the `max` of clipped/unclipped squared error is pessimistic, mirroring the policy
clip). It's trained **on-policy, jointly with the policy, every iteration**, because as Buddy improves the
returns move — the critic chases a moving target. Spoken form:

> *"The value model is a scalar head on the Gemma backbone, initialized from the reward model. I roll out
> Buddy responses, score them with the frozen RM, add per-token KL to the base to get dense rewards, then
> compute GAE advantages backward through each response as a γλ-discounted sum of TD residuals. The value
> target is returns = advantages + old values, and I regress the head onto that with a clipped MSE. It's
> retrained every iteration because the return target moves as the policy improves. It exists purely to
> be the variance-reduction baseline in A = Q − V — it never ships; I deploy only the policy."*

### 3.5 Why you'd *still* prefer DPO for Buddy

On-policy exploration and RM generalization are PPO's only real wins — and Buddy needed neither. Your
data was already pairwise and offline; you weren't trying to discover responses no teacher wrote; and you
had one 48GB machine. PPO would have added a reward model, a value model, a generation loop, and
reward-hacking-via-optimizer risk, to buy exploration you didn't want. **DPO was correct, and being able
to say *why PPO was available and you declined it* is stronger than not knowing PPO.**

---

## 4. What you actually ran instead of a critic: GRPO on Buddy

Buddy's GRPO stage is where you *did* run the critic-free RL objective — and it's the cleanest possible
answer to the value-model question, because you can say "I replaced it with a sampled baseline, here's
the code."

### 4.1 The mechanism, on your 8 rollouts

```
# Your GRPO (Repo B): 8 rollouts per prompt, resonance + format rewards, group-KL 0.12
for x in prompts:
    Y = [ policy.generate(x) for _ in range(8) ]          # G=8 rollouts — the group
    r = [ resonance_reward(y) + format_reward(y) for y in Y ]  # your two custom rewards
    #     resonance: Theory-of-Mind mirroring   format: penalize reasoning-trace drift, "As an AI..." phrases
    mean_r, std_r = mean(r), std(r)
    for i, y in enumerate(Y):
        A_i = (r[i] - mean_r) / std_r                      # group-standardized advantage
        # SAME scalar A_i on EVERY token of y_i — no GAE, no per-token critic
    loss = -Σ_i Σ_t min(ratio_{i,t}*A_i, clip(ratio_{i,t},1-eps,1+eps)*A_i)  \
           + 0.12 * KL(policy || ref)                       # KL as explicit term (k3), your group-KL
```

### 4.2 Where the value model went — say exactly this

**The group mean of your 8 rollouts *is* the baseline.** $\text{mean}(r_1..r_8)$ is an unbiased
Monte-Carlo estimate of $V^\pi(x)$ — the expected reward from prompt `x` under Buddy's own sampling —
which is *precisely* what the PPO critic in §3 was trained to approximate. So:

> *"I never trained a value network in GRPO because the group mean of my 8 rollouts already estimates the
> state value — and that's the only thing the critic was for. I traded a learned $O(\text{model})$-sized
> approximator for a sampled statistic: 8× the generation compute, one fewer model, no critic
> instability."*

### 4.3 Why dropping it was *appropriate*, not just cheap

Your reward is **sequence-level and outcome-shaped** — "is this whole response empathic / correctly
formatted?" There is **no per-token value structure** for a critic to model; the response succeeds or
fails as a unit. GAE's per-token machinery (§3.3) would be modeling variance that isn't there. A single
scalar advantage per completion is the *right* granularity for a whole-utterance reward. (Contrast: a
dense per-token toxicity signal *would* have justified a learned critic — that's the boundary.)

### 4.4 Your GRPO failure, diagnosed as reward design (not a missing critic)

GRPO-Final hit empathy **0.95** but question-rate **0.262** and length **0.062** — the "stability tax."
In the framework: the policy did *exactly* what GRPO told it — maximize group-relative advantage — but
your **reward vector was mis-weighted**, resonance dominating format. That's **reward misspecification**,
the RL cousin of the DPO proxy problem in §1.4. The fix is reward reweighting / flooring question-rate,
**not** adding a value model. Naming this shows you diagnose RL failures at the reward, where they live.

### 4.5 Why single-stage DPO beat your DPO→ORPO→GRPO chain

The chain on generic customer-support data peaked at **0.90** (= base); its DPO stage drove question-rate
to **0.00**; GRPO-ORPO only *recovered* to base. Framework reading: **each stage optimizes its own
proxy, and yours were off-domain** — so later stages spent capacity *undoing* earlier drift (GRPO-ORPO
literally paying back the DPO stage's damage) instead of adding alignment. One in-domain DPO stage
(0.97) beat three off-domain stages. **Method complexity lost to domain match** — stated as a mechanism,
not a slogan.

---

## 5. The unifying frame — all four, on Buddy, in one breath

Every method answers **"what baseline do I subtract from Buddy's reward, and where does it come from?"**

| Method | Baseline for Buddy | Cost | Buddy verdict |
|---|---|---|---|
| **REINFORCE** | none / constant | unbiased, huge variance | never used |
| **PPO** (§3) | *learned per-token* critic $V_\psi$ via GAE | 4 models, critic instability | correct tool, wrong budget — declined |
| **GRPO** (§4) | *sampled per-prompt* group mean of 8 rollouts | 8× rollouts, sequence-level only | ran it; empathy peak, format tax |
| **DPO** (§1) | *none* — reward folded into π_θ, $\log Z$ cancels pairwise | offline, no exploration | **shipped** — pairwise, offline, on-device |

Draw $A=Q-V$ in the center; hang all four off "where does the baseline come from." That is the whole
document compressed, and it turns the value-model question into your organizing principle.

---

## 6. End-to-end spoken walkthrough (2 minutes, on Buddy)

> *"Buddy is a preference-optimization problem: I have a teen turn `x`, one chosen response carrying
> metaphoric validation, and six rejected ones each embodying a therapeutic failure mode. Generating a
> response is a per-token MDP with a sparse terminal reward — my 4-dimension rubric only fires on the
> finished reply — so the core challenge is assigning that one score back across every token.*
>
> *I shipped DPO. It reparameterizes the reward as the policy's β-scaled log-ratio to the frozen base;
> because I only compare chosen vs rejected for the same `x`, the intractable partition function cancels
> pairwise, and I'm left with a simple `−log σ(margin)` on my pairs. No reward model, no value model, no
> rollouts — which is why it ran on one 48GB Mac at zero API cost. The implicit reward is
> β·log(π_θ/π_ref), and since chosen carried metaphors and rejected was clinical, the gradient sculpted
> exactly the persona I wanted.*
>
> *That framing also explains my headline finding: training margin didn't predict quality — my
> record-margin model was one of my worst — because the margin only measures how separable my pairs are,
> not how empathic the model is. When I stripped the hard CATASTROPHISING cases the margin spiked and
> quality fell. I confirmed I wasn't reward-hacking by tracking chosen and rejected logps separately:
> margins came from raising chosen, not tanking rejected, so I avoided likelihood displacement.*
>
> *Where I did run RL — GRPO — I used 8 rollouts per prompt and set each completion's advantage to its
> group-standardized reward. So the group mean was my baseline, which is an unbiased estimate of the
> state value — exactly what a PPO critic approximates — and that's why I never trained a value network:
> my reward was whole-utterance, with no per-token structure for a critic to capture. It peaked on
> empathy but tanked question-rate, which was my reward being mis-weighted, not a missing critic.*
>
> *If I'd been forced to run PPO, I'd have trained a Bradley-Terry reward model on my pairs, initialized
> a scalar value head from it, rolled out on-policy, scored with the RM plus a per-token KL leash,
> computed GAE advantages as a γλ-discounted sum of TD residuals, and regressed the critic onto
> returns = advantages + values with a clipped MSE — four models in memory, the critic there purely to
> cut variance. I didn't, because my problem was offline and pairwise and I didn't need exploration.
> Knowing when the value model earns its cost — and when a group mean or a log-ratio replaces it — is the
> actual lesson."*

---

## Appendix — repo fact-check flags (reconcile before the loop)

- **Two phases, two scales.** Phase-3 study (Repo A): champion `gemma3_dpo_2000_v3` = **0.97** on a 0–1 rubric, 12B. Phase-4 on-device (Repo B): **DPO+ORPO 4B** winner ≈ **8.98** on a 0–10 leaderboard. State which phase/scale you mean; never let them collide in one sentence.
- **GRPO status differs by repo** because they're different runs: Repo A's 12B chain GRPO was stopped early / recovered to base (0.90); Repo B's 4B GRPO pass was the empathy "research peak" (9.5) but instruction-taxed. Say "the 12B chain's GRPO" vs "the 4B GRPO pass."
- **"No reward hacking" is a DPO-stage finding** (chosen/rejected logps across 23 models) — don't attach it to GRPO, where the instruction-drift is a *different* failure (reward misweighting).
- **β=0.1, LR 3e-5, batch 2, 150–200 iters, LoRA r=64/α=128 on q_proj+v_proj** — verified from your READMEs; safe to cite.
- **Couldn't verify from source this session:** the exact `grpo_reward_functions.py` formulas (GitHub API rate-limited). Reward *design* is from your READMEs and safe; pull the exact reward code from your own repo before the loop so you can write the formula on the board if pushed.
