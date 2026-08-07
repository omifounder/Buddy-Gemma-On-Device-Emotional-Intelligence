# Post-Training Study Pack — Value Models, PPO, Reward Modeling, DPO, GRPO (Part-1)
### Whiteboard-ready, first-principles, for hands-on Applied Science / ML loops

The question that sank the last loop — *"how do you train the value model for PPO"* — is a
proxy. The interviewer is checking whether you understand that RLHF-PPO has **two** learned
networks with **different targets** (a reward model trained on preferences, a value model trained
on returns), why the second one exists at all, and whether you can defend dropping it (GRPO) or
never learning it (DPO). This pack builds all four methods so the value-model answer falls out as
one node in a connected graph.

Notation used throughout: policy $\pi_\theta$, reference/frozen policy $\pi_{\text{ref}}$, prompt
$x$, response $y = (y_1,\dots,y_T)$ generated token-by-token. A "state" $s_t = (x, y_{<t})$ is the
prompt plus tokens so far; an "action" $a_t = y_t$ is the next token. Reward model $r_\phi$, value
model $V_\psi$.

---

## 0. The MDP framing (say this first — it earns you the rest)

Token generation is a **per-token MDP**:

- **State** $s_t = (x, y_{<t})$ — prompt plus partial completion.
- **Action** $a_t = y_t$ — the next token, drawn from vocabulary $\mathcal{V}$.
- **Transition** is deterministic: $s_{t+1} = s_t \oplus y_t$ (append). No environment stochasticity — all randomness is in the policy's sampling.
- **Reward** is **terminal and sparse**: the reward model scores the *whole* completion, so $r_t = 0$ for $t < T$ and $r_T = r_\phi(x, y)$. (Plus a per-token KL penalty — see §1.4.)
- **Episode** = one prompt→completion. Horizon = sequence length $T$.

Two consequences an interviewer wants you to state unprompted:

1. **Reward is a single terminal scalar, but credit must be assigned per token.** That gap — one number at the end, thousands of token-decisions to grade — is *exactly* what the value model and GAE exist to bridge.
2. **Deterministic transitions** mean the value function only has to model the policy's *own* future sampling, not world dynamics. This is why a value head bolted onto the policy backbone is enough.

---

## 1. PPO for RLHF — the value model (critic) end to end

### 1.1 What the value model is

The value model estimates the **expected return from a state under the current policy**:

$$V^\pi(s_t) = \mathbb{E}_{y_{\ge t}\sim\pi}\!\left[\sum_{k=t}^{T}\gamma^{k-t} r_k \,\middle|\, s_t\right].$$

In words: *"standing at token position $t$ with this partial completion, if I keep sampling from
the current policy to the end, what terminal reward (minus KL) do I expect?"* It is **not** the
reward model. The reward model scores a *finished* completion; the value model predicts the
*eventual* score from a *half-finished* one. Conflating them is the single most common failure in
this question (see §2.3).

### 1.2 Architecture — scalar value head on the policy backbone

Concretely: take the transformer backbone (the same stack that produces the policy logits), and add
a **second head** — a linear map from the final hidden state $h_t \in \mathbb{R}^d$ to a scalar:

$$V_\psi(s_t) = w^\top h_t + b, \qquad w \in \mathbb{R}^d.$$

The policy head is $\text{softmax}(W_{\text{LM}} h_t)$ over the vocabulary; the value head is a
single scalar regressor reading the **same** $h_t$. Design choices:

- **Shared vs separate backbone.** Sharing (one backbone, two heads) saves memory — critical when you're already holding policy + reference + reward model in memory. Cost: the value regression gradient perturbs representations the policy relies on, which can destabilize. Many production setups (and the InstructGPT lineage) use a **separate** value network initialized from the reward model to avoid this coupling. State the trade-off: *shared = cheaper, coupled; separate = stable, 4th model in memory.*
- **Value head init.** A randomly-initialized head produces garbage values for the first few hundred steps, and PPO advantages are only as good as the value baseline. Warm-starting matters (next point).

### 1.3 Initialization — from RM or from SFT?

This is a favorite follow-up. Both are defensible; know why.

- **Init from the reward model.** The RM already maps $(x,y)\to$ scalar in "reward space," so its backbone + scalar head are a natural warm start for a value head. InstructGPT does this. Advantage: values start in roughly the right scale/space. Risk: the RM was trained on *full* sequences, so its scores on *partial* sequences are off-distribution — the value head must still adapt.
- **Init from the SFT/policy model.** Keeps the value backbone in the same representation space as the policy it's evaluating, which helps if you share the backbone. You then train the scalar head (and possibly the backbone) from scratch on returns.

Rule of thumb to say: **RM-init when the value net is separate** (inherit reward-space calibration);
**SFT/policy-init when the backbone is shared** (stay in policy representation space). Either way the
scalar head sees on-policy returns during PPO and adapts.

### 1.4 The reward signal PPO actually optimizes (with KL)

RLHF-PPO does *not* optimize the bare reward — it optimizes reward minus a **per-token KL penalty**
to the reference policy, which keeps the policy from drifting into reward-hacking gibberish the RM
scores highly but humans hate:

$$r_t^{\text{total}} = \underbrace{r_\phi(x,y)\cdot\mathbb{1}[t=T]}_{\text{terminal RM score}} \;-\; \beta\,\underbrace{\log\frac{\pi_\theta(y_t\mid s_t)}{\pi_{\text{ref}}(y_t\mid s_t)}}_{\text{per-token KL}}.$$

The KL term is **dense** (every token) even though the RM term is **sparse** (last token only). This
matters: the value model has *some* dense signal to fit even before the terminal reward lands. $\beta$
is the KL coefficient (often adaptively controlled to hit a target KL). Say explicitly: *the thing
the critic predicts the return of is this KL-shaped reward, not the raw RM output.*

### 1.5 Why the critic exists — variance reduction (the load-bearing "why")

You *could* run policy gradient with the raw return $R_t = \sum_{k\ge t}\gamma^{k-t}r_k$ as the
signal (REINFORCE). The gradient estimator is:

$$\nabla_\theta J = \mathbb{E}\!\left[\sum_t \nabla_\theta\log\pi_\theta(a_t\mid s_t)\,\Psi_t\right].$$

The choice of $\Psi_t$ is what everything hinges on. With $\Psi_t = R_t$ (raw return), the estimator
is **unbiased but catastrophically high-variance** — a single terminal reward gets credited to every
token equally, including tokens that had nothing to do with the outcome. Subtracting a **baseline**
$b(s_t)$ that doesn't depend on the action leaves the estimator **unbiased** (because
$\mathbb{E}_{a}[\nabla\log\pi(a|s)] = 0$, so $\mathbb{E}[\nabla\log\pi(a|s)\,b(s)]=0$) but can slash
variance. The **optimal-ish, tractable** baseline is $V^\pi(s_t)$ — the expected return itself.
Replacing $R_t$ with the **advantage**

$$A^\pi(s_t,a_t) = Q^\pi(s_t,a_t) - V^\pi(s_t)$$

asks the *right* question: *"was this token better or worse than what the policy expected on average
from here?"* Positive advantage → push the token's probability up; negative → down. **The value model
exists to be this baseline.** That's the one-sentence answer to "why train a value model." Everything
else is how.

### 1.6 GAE — full derivation (do this on the whiteboard)

We need $A^\pi(s_t,a_t)$ but we only have samples and an imperfect $V_\psi$. GAE (Generalized
Advantage Estimation, Schulman et al. 2016) is the bias–variance knob that produces it.

**Step 1 — the TD residual.** Define the one-step temporal-difference error:

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t).$$

If $V = V^\pi$ exactly, then $\mathbb{E}[\delta_t] = \mathbb{E}[r_t + \gamma V^\pi(s_{t+1})] - V^\pi(s_t) = Q^\pi(s_t,a_t) - V^\pi(s_t) = A^\pi(s_t,a_t)$.
So **$\delta_t$ is itself a one-step, low-variance-but-biased estimate of the advantage** (biased whenever $V\neq V^\pi$).

**Step 2 — the $n$-step advantage.** We can also estimate the advantage by rolling out $n$ steps
and bootstrapping:

$$
\hat A_t^{(1)} = \delta_t = -V(s_t) + r_t + \gamma V(s_{t+1})
$$
$$
\hat A_t^{(2)} = \delta_t + \gamma\delta_{t+1} = -V(s_t) + r_t + \gamma r_{t+1} + \gamma^2 V(s_{t+2})
$$
$$
\hat A_t^{(n)} = \sum_{l=0}^{n-1}\gamma^l \delta_{t+l} = -V(s_t) + \sum_{l=0}^{n-1}\gamma^l r_{t+l} + \gamma^n V(s_{t+n}).
$$

Read off the trade-off from the telescoped form: small $n$ leans on $V$ (**low variance, high bias**
if $V$ is wrong); large $n$ leans on actual sampled rewards (**low bias, high variance**). $n=\infty$
is the full Monte-Carlo return minus $V(s_t)$ — unbiased, maximal variance. This is *the same
bias–variance axis* from §1.5, now made continuous.

**Step 3 — exponentially-weighted average.** Instead of picking one $n$, GAE takes a
$\lambda$-weighted geometric average over **all** $n$-step estimators:

$$
\hat A_t^{\text{GAE}(\gamma,\lambda)} = (1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}\hat A_t^{(n)}.
$$

The $(1-\lambda)$ normalizes the weights $\sum_n (1-\lambda)\lambda^{n-1}=1$. Substitute
$\hat A_t^{(n)}=\sum_{l=0}^{n-1}\gamma^l\delta_{t+l}$ and swap the order of summation. Each $\delta_{t+l}$
appears in every $\hat A^{(n)}$ with $n>l$, so its total weight is
$(1-\lambda)\gamma^l\sum_{n>l}\lambda^{n-1}=(1-\lambda)\gamma^l\lambda^{l}\frac{1}{1-\lambda}=(\gamma\lambda)^l$.
The $(1-\lambda)$ cancels cleanly and you get the punchline:

$$\boxed{\;\hat A_t^{\text{GAE}} = \sum_{l=0}^{T-t}(\gamma\lambda)^l\,\delta_{t+l}\;}$$

**a discounted sum of TD residuals with discount $\gamma\lambda$.** Interpretation of the two knobs:

- $\gamma$ (discount): how far into the future rewards matter — controls bias from horizon truncation.
- $\lambda$ (GAE): interpolates estimators. $\lambda=0 \Rightarrow \hat A_t=\delta_t$ (pure one-step TD, low variance / high bias, maximal reliance on the critic). $\lambda=1 \Rightarrow \hat A_t=\sum_l\gamma^l\delta_{t+l}=R_t - V(s_t)$ (Monte-Carlo, unbiased / high variance, critic only used as the baseline at $s_t$). Typical RLHF: $\gamma\approx 1.0$ (short sequences, reward at the end — you don't want to discount away the terminal signal), $\lambda\approx 0.95$.

**Step 4 — compute it in one backward pass.** The recursion falls straight out of the boxed form:

$$\hat A_t = \delta_t + \gamma\lambda\,\hat A_{t+1},\qquad \hat A_{T}=\delta_{T}.$$

Iterate backward over the sequence — $O(T)$, one loop. This is worth writing on the board; it shows
you've actually implemented it.

### 1.7 Training the value model — the target

The value model is trained by **regression onto returns**. The target is the GAE-implied return:

$$\hat R_t = \hat A_t^{\text{GAE}} + V_\psi(s_t) \quad(\text{"returns} = \text{advantages} + \text{values"}),$$

computed with the **old** value net (the one used to generate advantages), so the target is a fixed
scalar during the update. The value loss is squared error:

$$L^{V}(\psi) = \mathbb{E}_t\!\left[\big(V_\psi(s_t) - \hat R_t\big)^2\right].$$

Say the loop out loud, because *this is the literal answer to the question you missed*:

1. Sample completions from the current policy on a batch of prompts (on-policy rollouts).
2. Score each completion with the reward model; add per-token KL penalty → per-token rewards $r_t$.
3. Run the current value net forward over every state to get $V_\psi(s_t)$.
4. Compute TD residuals $\delta_t$ and GAE advantages $\hat A_t$ backward through each sequence.
5. Form value targets $\hat R_t = \hat A_t + V_\psi(s_t)$.
6. Regress $V_\psi$ onto $\hat R_t$ with MSE (clipped — §1.8), **jointly** with the policy update, over a few epochs on the same batch.

So: **on-policy, bootstrapped, regressed onto GAE returns, trained simultaneously with the policy.**
The value model chases a moving target (returns shift as the policy improves), which is why it's
retrained every PPO iteration rather than fit once.

### 1.8 The PPO loss — policy clipping *and* value clipping

**Policy objective** (clipped surrogate). Let $\rho_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$
be the importance ratio (rollouts are from $\pi_{\theta_{\text{old}}}$; we do several gradient epochs
on them, so we must importance-correct):

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t\!\left[\min\!\big(\rho_t\hat A_t,\ \text{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t\big)\right].$$

The clip removes the incentive to move $\rho_t$ far outside $[1-\epsilon,1+\epsilon]$ — a trust region
by brute force. The $\min$ makes it a **pessimistic** (lower) bound: when $\hat A_t>0$ the objective is
capped at $(1+\epsilon)\hat A_t$; when $\hat A_t<0$ it's capped at $(1-\epsilon)\hat A_t$. Either way, once
the ratio leaves the band in the "tempting" direction, the gradient vanishes.

**Value objective** (also clipped — this is the part people forget, and a likely follow-up given your
question). To stop the value net from lurching on a single noisy batch, clip its *change* around the
old prediction:

$$L^{V}(\psi)=\mathbb{E}_t\!\Big[\max\!\big((V_\psi(s_t)-\hat R_t)^2,\ (\text{clip}(V_\psi(s_t),\,V_{\text{old}}(s_t)-\eta,\,V_{\text{old}}(s_t)+\eta)-\hat R_t)^2\big)\Big].$$

The $\max$ makes it **pessimistic in the same spirit as the policy clip** — it takes the *larger* of the
unclipped and clipped squared errors, so the value net can't reduce its loss by making a huge jump; a
large move is only rewarded if it genuinely reduces error beyond the clip band. $\eta$ is the value clip
range.

**Full PPO loss:**

$$L^{\text{PPO}} = -L^{\text{CLIP}}(\theta) + c_1 L^{V}(\psi) - c_2\,\mathbb{E}_t[\mathcal{H}(\pi_\theta(\cdot\mid s_t))],$$

with $c_1$ the value-loss coefficient, $c_2$ the entropy-bonus coefficient (encourages exploration /
prevents premature collapse). Minus signs orient everything to minimization. If backbone is shared,
$\theta$ and $\psi$ overlap and this is one optimizer step; if separate, two.

### 1.9 The four-model memory picture (say this to show systems awareness)

At peak, RLHF-PPO holds in memory: **(1)** the policy $\pi_\theta$ (trained), **(2)** the reference
$\pi_{\text{ref}}$ (frozen, for KL), **(3)** the reward model $r_\phi$ (frozen, scores rollouts),
**(4)** the value model $V_\psi$ (trained). That's ~4× the model footprint plus a generation buffer.
This memory/compute burden is the setup for **why GRPO and DPO exist** (§3, §4) — both are, in part,
attacks on this four-model cost. Naming this unprompted signals you see the whole board.

---

## 2. Reward modeling

### 2.1 Bradley–Terry preference loss — derivation

You collect comparisons: for prompt $x$, a labeler prefers completion $y_w$ ("win") over $y_l$
("lose"). The **Bradley–Terry** model posits that the probability one item beats another is a logistic
function of a latent score difference. Let the reward model output a latent scalar "quality"
$r_\phi(x,y)$. Then:

$$P(y_w \succ y_l \mid x) = \frac{\exp r_\phi(x,y_w)}{\exp r_\phi(x,y_w)+\exp r_\phi(x,y_l)} = \sigma\big(r_\phi(x,y_w)-r_\phi(x,y_l)\big),$$

where $\sigma$ is the logistic sigmoid. Fit by maximum likelihood → minimize negative log-likelihood
over the preference dataset $\mathcal{D}$:

$$\boxed{\;L_{\text{RM}}(\phi) = -\,\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\Big[\log\sigma\big(r_\phi(x,y_w)-r_\phi(x,y_l)\big)\Big].\;}$$

Architecture: SFT model backbone + scalar head (final-token hidden state → scalar). Key points to
raise:

- **Only differences are identified.** The loss depends on $r_\phi(x,y_w)-r_\phi(x,y_l)$, so the reward is defined **only up to an additive constant** (per prompt, in fact). Absolute reward values are meaningless — a source of instability that PPO handles via advantage-centering and reward normalization (whitening rewards to zero mean / unit variance per batch). Mention this; it's a favorite gotcha.
- **Overfitting / reward hacking.** RMs are trained on a finite preference set and are then queried massively off-distribution by an adversarial optimizer (the policy). The policy finds inputs where $r_\phi$ is wrong-but-high → reward hacking. Mitigations: KL penalty (§1.4), RM ensembles, early stopping on KL, reward clipping, iterated RM retraining on fresh on-policy data.
- **Ranking-of-$k$ extension.** With $k$ ranked completions you can use a Plackett–Luce likelihood (product of softmaxes over the ranking) rather than pairwise — more data-efficient per prompt.

### 2.2 Reward vs value — the distinction that got you

| | **Reward model $r_\phi(x,y)$** | **Value model $V_\psi(s_t)$** |
|---|---|---|
| Trained on | human **preferences** (pairwise) | **returns** (regression, on-policy) |
| Loss | Bradley–Terry / logistic ranking | MSE onto GAE returns |
| Input | a **complete** completion | a **partial** state $s_t=(x,y_{<t})$ |
| Output | quality of *this finished answer* | expected *future* return from here |
| Frozen during PPO? | **yes** (fixed judge) | **no** (retrained every iteration) |
| Analogy | the exam grader | your predicted score before you finish |

The value model *bootstraps off* the reward: the RM produces the terminal reward, and the value model
learns to predict the (KL-shaped) return that terminal reward implies. They are different objects
trained with different losses on different targets. **Saying this crisply is the fix for the miss.**

### 2.3 Common conflations to avoid (name them, then refute)

1. *"The value model is trained on preferences"* — **no**, that's the reward model. Value is MSE-on-returns.
2. *"The value model is the reward model with a different name"* — no; RM scores finished text, V predicts future return from partial text and is retrained on-policy.
3. *"$Q$ and $V$ and reward are interchangeable"* — reward is the immediate/terminal signal; $V(s)$ is expected return over actions; $Q(s,a)$ conditions on the action; $A=Q-V$.
4. *"Advantage is the reward"* — advantage is reward-relative-to-baseline; it can be negative for a high-reward token if the policy expected even better.
5. *"You need the reward model at inference"* — no, both RM and V are training-time scaffolding; you ship only $\pi_\theta$.

---

## 3. DPO — Direct Preference Optimization

### 3.1 Derivation from the RLHF objective

Start from the **KL-regularized RLHF objective** that PPO approximately solves:

$$\max_{\pi_\theta}\ \mathbb{E}_{x,\,y\sim\pi_\theta}\big[r(x,y)\big] \;-\; \beta\,\mathbb{D}_{\text{KL}}\!\big[\pi_\theta(y\mid x)\,\|\,\pi_{\text{ref}}(y\mid x)\big].$$

This has a **known closed-form optimum** (a standard result — the reverse-KL-regularized reward-maximizing
distribution is a Boltzmann tilt of the reference):

$$\pi^*(y\mid x) = \frac{1}{Z(x)}\,\pi_{\text{ref}}(y\mid x)\,\exp\!\Big(\tfrac{1}{\beta}r(x,y)\Big),\qquad Z(x)=\sum_y \pi_{\text{ref}}(y\mid x)\exp\!\big(\tfrac1\beta r(x,y)\big).$$

*(Derive it in one line if pushed: form the Lagrangian, set the functional derivative w.r.t. $\pi(y|x)$
to zero, $r - \beta(\log\frac{\pi}{\pi_{\text{ref}}}+1)-\text{const}=0$, solve for $\pi$, normalize.)*

Now **invert** for the reward. Take logs and rearrange:

$$r(x,y) = \beta\log\frac{\pi^*(y\mid x)}{\pi_{\text{ref}}(y\mid x)} + \beta\log Z(x).$$

Here's the trick. Substitute this expression for $r$ into the **Bradley–Terry** preference model. In
the difference $r(x,y_w)-r(x,y_l)$, the intractable $\beta\log Z(x)$ term is **the same for both
completions of the same prompt and cancels**:

$$r(x,y_w)-r(x,y_l) = \beta\log\frac{\pi^*(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \beta\log\frac{\pi^*(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}.$$

Replace $\pi^*$ with our trainable $\pi_\theta$ and plug into the BT negative log-likelihood. The
**DPO loss**:

$$\boxed{\;L_{\text{DPO}}(\theta)=-\,\mathbb{E}_{(x,y_w,y_l)}\Big[\log\sigma\Big(\beta\log\tfrac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \beta\log\tfrac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\Big)\Big].\;}$$

### 3.2 Why it removes the explicit reward model

The reward never appears as a separate network — it was **reparameterized as a function of the policy
itself**. The partition function $Z(x)$, the only intractable piece, cancels in the pairwise
difference. So the same preference data that would have trained an RM now trains the policy
**directly**, in one supervised-style stage, with no sampling, no RM, no value model, no PPO loop.

### 3.3 The implicit reward

DPO still has a reward — it's just implicit in the policy:

$$\hat r_\theta(x,y) = \beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)}.$$

The **log-ratio to the reference, scaled by $\beta$, is the reward**. This is worth stating because it
unifies the whole pack: the DPO gradient pushes up the implicit reward of $y_w$ and down that of
$y_l$, weighted by how *wrong* the implicit reward currently is:

$$\nabla_\theta L_{\text{DPO}} \propto -\,\sigma\big(\hat r_\theta(x,y_l)-\hat r_\theta(x,y_w)\big)\big[\nabla\log\pi_\theta(y_w\mid x)-\nabla\log\pi_\theta(y_l\mid x)\big].$$

The $\sigma(\cdot)$ weight is large exactly when the model currently has the preference **backwards** —
an automatic hard-example focus.

### 3.4 Trade-offs vs PPO

**DPO wins on:** simplicity (no RM, no value model, no rollouts, no reward-hacking-via-optimizer),
stability, compute (one model + frozen reference; two forward passes), and reproducibility. This maps
to your Buddy finding — single-stage domain DPO beat the longer DPO→ORPO→GRPO chain qualitatively,
and offline preference methods are far easier to run than a PPO loop.

**PPO wins on / DPO's costs:**

- **Off-policy / fixed dataset.** DPO trains only on the preference pairs you already have; it can't explore *new* completions. PPO generates fresh on-policy data and can discover responses no labeler wrote. This is the deepest difference — DPO is offline, PPO is online.
- **No reward generalization.** A learned RM can *score* novel completions; DPO's implicit reward is only shaped where preference data exists. On distribution shift, PPO's RM can generalize where DPO can't.
- **Overoptimization / likelihood displacement.** DPO can *decrease* $\pi_\theta(y_w|x)$ in absolute terms as long as it decreases $\pi_\theta(y_l|x)$ faster — the loss only cares about the *margin*. This can push probability mass onto unintended third completions and degrade quality even as the training margin improves. (This is precisely your Buddy "training metrics don't predict response quality" result — the val-margin/quality decoupling is a *known structural* property of the DPO objective, not noise. Cite it as design insight.)
- **KL control is implicit.** $\beta$ sets regularization strength but there's no online KL measurement/targeting like PPO's adaptive controller, so drift is harder to monitor.

Variants worth a sentence each (you've studied these): **IPO** replaces the logistic with a squared
loss on the margin to fight DPO's overfitting-to-deterministic-preferences failure; **ORPO** folds a
preference odds-ratio penalty into the SFT loss so no reference model is needed at all (but needs lots
of pairs to show signal — your ~62k observation); **KTO** drops pairwise data entirely and uses
unpaired desirable/undesirable labels via a prospect-theory value function.

---

## 4. GRPO — Group Relative Policy Optimization

### 4.1 The core move — group-relative baseline replaces the critic

GRPO (from the DeepSeekMath / DeepSeek-R1 line) keeps PPO's clipped-ratio *policy* update but
**deletes the value model**. Recall the value model's *only* job was to be the baseline in
$A=Q-V$ (§1.5). GRPO asks: what if we estimate that baseline **empirically from a group of samples**
instead of learning it?

For each prompt $x$, sample a **group** of $G$ completions $\{y_1,\dots,y_G\}$ from the old policy.
Score each with the reward model → $\{r_1,\dots,r_G\}$. The baseline for the group is just the **group
mean**, and the advantage is the **within-group standardized reward**:

$$\hat A_i = \frac{r_i - \text{mean}(\{r_1,\dots,r_G\})}{\text{std}(\{r_1,\dots,r_G\})}.$$

Every token in completion $y_i$ receives this **same** sequence-level advantage $\hat A_i$ (no GAE, no
per-token bootstrapping — the group mean *is* the value baseline). Plug $\hat A_i$ into the identical
PPO clipped surrogate:

$$L_{\text{GRPO}}(\theta)=\mathbb{E}\!\left[\frac1G\sum_{i=1}^{G}\frac{1}{|y_i|}\sum_{t}\min\big(\rho_{i,t}\hat A_i,\ \text{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)\hat A_i\big)\right] - \beta\,\mathbb{D}_{\text{KL}}[\pi_\theta\|\pi_{\text{ref}}].$$

(KL is often applied directly as a term here, via the unbiased $k3$ estimator, rather than folded into
the reward.)

### 4.2 Why the value model can be dropped

Two reasons, both worth stating:

1. **The baseline was the value model's only purpose, and a Monte-Carlo group mean is an unbiased estimate of $V^\pi(x)$** (the expected reward over the policy's own samples from that prompt). With enough samples per group, $\text{mean}(\{r_i\})\approx V^\pi(x)$ — you've replaced a *learned* $O(\text{model})$-sized approximator with a *sampled* statistic. You trade extra rollouts for one whole fewer network.
2. **In many RLHF-for-reasoning settings the reward is terminal and outcome-based** (e.g., verifier: is the final answer correct?). There's no meaningful *per-token* value signal to model — the entire episode succeeds or fails as a unit. A sequence-level advantage is then not just cheaper but *appropriate*; GAE's per-token machinery is modeling variance that isn't really there.

Cost of the trade: you need $G$ completions per prompt (typically 8–64), so **more generation
compute**, and the baseline is noisier for small $G$. But you drop the 4th model, its optimizer state,
its training loop, and the risk that a *bad* critic silently poisons every advantage.

### 4.3 When GRPO wins

- **Verifiable / outcome-reward domains** (math, code, tool-use with a checker) — this is its home turf. Sparse terminal reward + no per-token value structure = the critic was low-value anyway.
- **Memory-constrained training** — dropping the value net's parameters + optimizer state is a large fraction of PPO's footprint (§1.9).
- **When RM/verifier scores are cheap** relative to the value net's instability — you'd rather sample more than fit a critic.

**When PPO's learned critic still wins:** dense or shaped per-token rewards, long horizons where
per-token credit assignment genuinely matters, or when generating a large group per prompt is too
expensive (then one critic amortizes across prompts more cheaply than $G$-sampling every prompt).

### 4.4 The unifying frame (say this to close)

All four methods are answering **"what baseline do I subtract from the reward signal, and where does
it come from?"**

- **REINFORCE:** no baseline (or a constant) → unbiased, high variance.
- **PPO:** a *learned per-state* baseline $V_\psi(s_t)$ via GAE → lowest variance, costs a 4th model + instability risk.
- **GRPO:** a *sampled per-prompt* baseline (group mean) → no extra model, costs $G×$ rollouts, sequence-level granularity.
- **DPO:** *no reward or baseline at all* — reparameterize the reward as the policy's log-ratio and cancel the partition function pairwise → offline, simplest, but can't explore and decouples margin from quality.

That sentence is the whole pack compressed. If you can draw the $A=Q-V$ identity in the center and hang
all four methods off "where does the baseline come from," you've turned the question you missed into
the organizing principle of the answer.

---

## 5. Interviewer follow-up drills (with strong answers)

### 5.A After "how do you train the value model" — the 10 that come next

**Q1. Why not just use the reward model as the value function?**
The RM scores *finished* completions; the value function must score *partial* states $s_t=(x,y_{<t})$,
which are off-distribution for the RM. And the value target is the *return* (GAE-discounted, KL-shaped),
not the RM's raw preference score. You *can* initialize the value head from the RM (shared reward-space
scale), but it must then be trained on-policy against returns to be correct on partial sequences.

**Q2. What exactly is the value model's regression target?**
$\hat R_t = \hat A_t^{\text{GAE}} + V_{\text{old}}(s_t)$ — the GAE advantage plus the old value estimate,
i.e., the empirical return implied by the current rollout and bootstrap. Computed with the frozen old
value net so the target is a constant during the update. MSE loss, value-clipped.

**Q3. Why is the value loss clipped, and how?**
To stop a single noisy batch from lurching the critic and destabilizing every downstream advantage.
Clip $V_\psi(s_t)$ to $V_{\text{old}}(s_t)\pm\eta$ and take the *max* of clipped and unclipped squared
error — a pessimistic bound mirroring the policy clip. A large value move only "counts" if it truly
reduces error past the band.

**Q4. Walk me through GAE. Why the $\lambda$?**
[Do the §1.6 derivation.] $\lambda$ interpolates between one-step TD ($\lambda=0$: low variance, high
bias, max reliance on the critic) and Monte-Carlo ($\lambda=1$: unbiased, high variance, critic only
as the $s_t$ baseline). It's the continuous bias–variance knob over $n$-step returns; the closed form is
a $(\gamma\lambda)$-discounted sum of TD residuals, computable in one backward pass.

**Q5. What $\gamma,\lambda$ would you use for RLHF and why?**
$\gamma\approx1.0$: sequences are short and the meaningful reward is terminal — discounting would
throw away the signal. $\lambda\approx0.95$: mostly trust sampled returns but let the critic damp
variance a little. Contrast with control tasks (long horizons) where $\gamma<1$ matters for stability.

**Q6. Shared vs separate value backbone — pick one and defend it.**
Shared: one backbone, policy + value heads — cheaper memory, but value-regression gradients perturb
policy representations (coupling/instability). Separate: init from RM, stable, but a 4th full model in
memory. For LLM RLHF at scale I'd often go separate/RM-init for stability and reserve sharing for
tighter memory budgets — and I'd state I'd measure it, not assume.

**Q7. What happens if the value model is bad / lags the policy?**
Advantages become biased → the policy chases a wrong baseline → training destabilizes or plateaus.
Symptom: explained-variance of the value predictions drops, KL spikes, reward stalls. Mitigation: more
value epochs per batch, value warm-start, reward/advantage whitening, lower LR on the value head.

**Q8. Why do we normalize/whiten advantages and rewards?**
BT rewards are identified only up to an additive constant (§2.1), so absolute scale is arbitrary;
whitening advantages to zero-mean/unit-variance per batch stabilizes the gradient scale and makes the
clip range $\epsilon$ behave consistently across batches.

**Q9. Where does the KL penalty enter — the reward or the loss?**
Classic PPO-RLHF folds it into the **per-token reward** ($r_t^{\text{total}}$, §1.4), so the value
model learns the return of the *KL-shaped* reward. Some implementations (and GRPO) instead add KL as an
explicit **loss term** via the $k3$ estimator. Know both; the choice changes what the critic is
predicting.

**Q10. At inference, which of these models do you ship?**
Only $\pi_\theta$. The reward model, reference, and value model are all training-time scaffolding. This
is why the value model's job is purely variance reduction *during* training — it never touches
production.

### 5.B Reward-modeling follow-ups

**Q11. Derive the Bradley–Terry loss and state what's identifiable.** [§2.1] Only reward *differences*
are identified; absolute values are arbitrary up to a per-prompt constant.

**Q12. How does reward hacking arise and how do you fight it?** The policy is an adversarial optimizer
querying the RM off-distribution; it finds high-RM/low-quality regions. Fight with KL penalty, RM
ensembles/uncertainty, reward clipping, early stop on KL, and iterated RM retraining on fresh on-policy
completions.

**Q13. RM trained from SFT or from scratch?** From the SFT model backbone + fresh scalar head — you
want the language competence, and the scalar head learns preference scoring. From-scratch wastes the
pretraining.

### 5.C DPO follow-ups

**Q14. Derive DPO and show where the reward model went.** [§3.1] Closed-form RLHF optimum → invert for
$r$ → substitute into BT → $\log Z(x)$ cancels pairwise. Reward is reparameterized as the policy's
$\beta$-scaled log-ratio to the reference.

**Q15. What is DPO's implicit reward?** $\hat r_\theta(x,y)=\beta\log\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$.

**Q16. Why can DPO degrade quality while its training margin improves?** The loss optimizes only the
*margin* $\hat r(y_w)-\hat r(y_l)$; it can lower $\pi_\theta(y_w|x)$ absolutely as long as it lowers
$\pi_\theta(y_l|x)$ faster (likelihood displacement), shifting mass to unintended completions. This is
the structural reason behind the metric–quality decoupling I saw empirically in the Buddy study.

**Q17. When would you still prefer PPO over DPO?** When you need on-policy exploration beyond the fixed
preference set, RM generalization to novel completions, or explicit online KL targeting. DPO is offline
and can't discover responses no labeler produced.

**Q18. $\beta$'s role in DPO?** Regularization strength toward the reference / inverse temperature of the
implicit reward. Small $\beta$ → aggressive, more drift from ref; large $\beta$ → conservative. Unlike
PPO there's no online KL measurement, so $\beta$ is your only (blunt) drift control.

### 5.D GRPO follow-ups

**Q19. How does GRPO estimate advantage without a critic?** Sample $G$ completions per prompt, score
each with the RM/verifier, advantage = within-group standardized reward $(r_i-\text{mean})/\text{std}$;
same scalar to every token of that completion. The group mean *is* the (Monte-Carlo) value baseline.

**Q20. Why is dropping the critic justified — isn't it biased?** The group mean is an unbiased MC
estimate of $V^\pi(x)$; you've swapped a learned approximator for a sampled statistic. In terminal,
outcome-reward settings there's no per-token value structure for a critic to capture anyway, so GAE's
machinery is modeling variance that isn't there.

**Q21. GRPO's costs vs PPO?** Needs $G$ rollouts per prompt (more generation compute), noisier baseline
for small $G$, and sequence-level (not per-token) credit assignment — worse when rewards are dense/shaped
or horizons long. Wins on memory (no 4th model), simplicity, and verifiable-reward domains (math/code/tool
use). Cite DeepSeek-R1-style reasoning as the canonical fit.

**Q22. Unify PPO/GRPO/DPO in one sentence.** They differ only in the baseline subtracted from the reward:
PPO learns a per-state critic (GAE), GRPO samples a per-prompt group mean, DPO removes reward and baseline
entirely by reparameterizing reward as the policy log-ratio and cancelling the partition function pairwise.

---

## 6. 60-second spoken answer to the exact question you missed

*"The value model is the critic — a scalar value head on a transformer backbone, often initialized from
the reward model when it's a separate network or from the policy when the backbone is shared. Its job is
pure variance reduction: it's the baseline in the advantage $A=Q-V$. I train it on-policy — I roll out
completions from the current policy, score them with the frozen reward model, add the per-token KL
penalty to get per-token rewards, then compute GAE advantages backward through each sequence as a
$\gamma\lambda$-discounted sum of TD residuals $\delta_t = r_t+\gamma V(s_{t+1})-V(s_t)$. The value
target is returns = advantages + old values, and I regress the value net onto that with a clipped MSE —
the value clip is a pessimistic max of clipped and unclipped squared error, mirroring the policy clip so
one noisy batch can't lurch the critic. It's trained jointly with the policy every PPO iteration because
the return target moves as the policy improves. Crucially it's not the reward model — the reward model is
trained on preferences with a Bradley–Terry loss and scores finished completions; the value model
regresses onto returns and predicts expected future reward from a partial completion. And I'd note it's
training-time only — we ship just the policy — which is exactly why GRPO can replace it with a
group-mean baseline and DPO can remove it altogether."*

That single paragraph, plus the GAE derivation on the board, closes the gap.

---

## 7. Further reading (canonical anchors)

- Schulman et al. 2016 — *High-Dimensional Continuous Control Using GAE* (the GAE derivation).
- Schulman et al. 2017 — *Proximal Policy Optimization Algorithms* (clipped surrogate).
- Ouyang et al. 2022 — *InstructGPT* (RM-init value net; full RLHF-PPO recipe for LLMs).
- Christiano et al. 2017 — *Deep RL from Human Preferences* (Bradley–Terry RM).
- Rafailov et al. 2023 — *DPO: Your Language Model is Secretly a Reward Model* (§3 derivation).
- Shao et al. 2024 — *DeepSeekMath* / GRPO; DeepSeek-R1 2025 (group-relative advantage).
- Gao et al. 2023 — *Scaling Laws for Reward Model Overoptimization* (reward hacking).
- Azar et al. 2023 (IPO), Hong et al. 2024 (ORPO), Ethayarajh et al. 2024 (KTO) — DPO variants.
