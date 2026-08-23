# Three ways to remember: PPO+GRU, DreamerV3, and STORM

*A self-contained, textbook-style guide to the three agents in this folder.*

All three solve the same `fork_wall` POMDP (see `ENVIRONMENT.md`). They differ in
**how they build and carry the belief** — the internal summary of "which category
is this map?" that must survive the memory corridor. This document explains each
architecture from first principles, with the mathematics, and then compares them.

> **Reading guide.** §0 sets up the common problem. §1 is the model-free baseline
> (PPO+GRU). §2 and §3 are the two model-based agents (DreamerV3, STORM). §4
> compares them and states the empirical lesson we learned on this task. Each
> section is written to be readable on its own.

---

## 0. The common problem: acting under partial observation

At each step the agent sees an observation $o_t \in \mathbb{R}^{3974}$ (the
egocentric crop + scalars) and picks an action $a_t \in \{0,\dots,5\}$. The
environment is a **Partially Observed Markov Decision Process (POMDP)**: the true
state (in particular the hidden map **category** $c \in \{\text{balanced, lakes,
rocky}\}$) is *not* a function of the current observation once the agent has left
the terrain. The optimal action depends on the entire history
$h_t = (o_1,a_1,\dots,o_t)$.

No feed-forward network of $o_t$ alone can solve this — it would have to guess the
door. Every architecture here therefore maintains a **recurrent internal state**
$s_t$ that summarizes the history:

$$
s_t = f_\theta(s_{t-1}, o_t, a_{t-1}), \qquad a_t \sim \pi_\theta(\cdot \mid s_t).
$$

The category $c$, encoded early from the water/rock terrain, must be kept alive
inside $s_t$ across the 16-column grass corridor. We call the part of $s_t$ that
holds $c$ the **belief**. The three agents build $s_t$ in fundamentally different
ways:

| Agent | What $s_t$ is | How memory is trained |
|---|---|---|
| **PPO+GRU** | a GRU hidden state (model-free) | backprop-through-time on the RL loss |
| **DreamerV3** | an RSSM state = GRU deterministic part **+** discrete stochastic latent | trained to *predict the future* (a world model), policy learned in imagination |
| **STORM** | a Transformer's running context over latent tokens | same idea as Dreamer, but attention instead of a GRU |

The rest of the document unpacks each row.

---

## 1. Recurrent PPO + GRU (model-free)

**File:** `ppo/`. **Idea:** learn the policy *directly* from reward, with a GRU to
carry memory. No model of the world is ever built.

### 1.1 Actor–critic

PPO ([Schulman et al., 2017](https://arxiv.org/abs/1707.06347)) is a policy-gradient
method. Two heads sit on top of the recurrent state $s_t$:

* the **actor** $\pi_\theta(a_t\mid s_t)$ — a categorical distribution over the 6
  actions;
* the **critic** $V_\theta(s_t)$ — an estimate of the expected discounted return
  $\mathbb{E}\big[\sum_{k\ge0}\gamma^k r_{t+k}\big]$.

The critic is trained by regression to a return target; the actor is nudged to make
high-advantage actions more likely.

### 1.2 The GRU: where memory lives

The recurrent state is a **Gated Recurrent Unit** ([Cho et al.,
2014](https://arxiv.org/abs/1406.1078)) with hidden size 128. Writing $x_t$ for the
encoded observation (a 256-d MLP embedding of $o_t$) and $s_{t-1}$ for the previous
hidden state:

$$
\begin{aligned}
z_t &= \sigma(W_z x_t + U_z s_{t-1}) &&\text{(update gate)}\\
r_t &= \sigma(W_r x_t + U_r s_{t-1}) &&\text{(reset gate)}\\
\tilde s_t &= \tanh(W_h x_t + U_h (r_t \odot s_{t-1})) &&\text{(candidate)}\\
s_t &= (1-z_t)\odot s_{t-1} + z_t \odot \tilde s_t. &&\text{(new state)}
\end{aligned}
$$

The **update gate** $z_t$ is the memory mechanism: when $z_t\approx 0$ the unit
*copies the old state forward unchanged*, which is exactly what is needed to carry
the category across the information-free corridor. Training pushes the gates to
learn "latch the category when I see terrain, then hold it."

### 1.3 The PPO objective

Let $\hat A_t$ be the **advantage** (how much better $a_t$ was than average),
estimated with Generalized Advantage Estimation (GAE, $\lambda=0.95$, $\gamma=0.99$):

$$
\hat A_t = \sum_{k\ge 0}(\gamma\lambda)^k\,\delta_{t+k},
\qquad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t).
$$

Define the probability ratio between the new and old policy
$\rho_t(\theta) = \dfrac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)}$.
PPO maximizes the **clipped surrogate**, which prevents destructively large updates:

$$
\mathcal L^{\text{clip}}(\theta) = \mathbb{E}_t\Big[\min\big(\rho_t \hat A_t,\;
\mathrm{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\,\hat A_t\big)\Big],\quad \epsilon=0.2.
$$

The full loss adds a value regression term and an entropy bonus (encouraging
exploration):

$$
\mathcal L = -\mathcal L^{\text{clip}} + c_v\,\underbrace{\mathbb{E}_t\big[(V_\theta(s_t)-\hat R_t)^2\big]}_{\text{critic}} - c_e\,\underbrace{\mathbb{E}_t\big[\mathcal H[\pi_\theta(\cdot\mid s_t)]\big]}_{\text{entropy}}.
$$

Here $c_v = 0.5$ and the entropy coefficient $c_e$ is deliberately large and
**annealed** ($c_e = 0.15 \to 0$ over training): this task is only solved when the
policy is **robust under its own stochasticity**, so we train (and evaluate) with a
genuinely stochastic actor. The entropy schedule is not cosmetic — it is what lets
PPO solve the *plain* reward at all (see §1.5).

### 1.4 Auxiliary belief head

Because we care about *interpretability*, the PPO agent carries one extra output: a
small **belief head** that reads $s_t$ and is trained to predict the map category
$c$ with a cross-entropy loss, weighted by `belief_coef = 0.3`:

$$
\mathcal L_{\text{belief}} = 0.3\cdot\mathbb{E}_t\big[\mathrm{CE}(g_\phi(s_t),\,c)\big].
$$

This does **not** change the environment or reward (it is a property of the agent,
not the task); it gently shapes $s_t$ so the belief is linearly decodable, and it
gives a training-time read-out of belief accuracy (`belief_acc ≈ 0.90` at
convergence). DreamerV3 and STORM carry no such head — their belief is read out
*post hoc* with a probe.

### 1.5 Why PPO+GRU works here — and the exploration trap

Backpropagation-through-time flows the door-reward gradient directly back through
the GRU to the moment the terrain was seen, so the gates learn to latch and hold the
category. At convergence PPO reaches **98% held-out** (97.7% decisive-door).

But on the **plain** reward this only works with enough exploration, and the failure
mode is instructive. With default entropy the policy collapses to a **constant
door** (decisive ≈ 50%). Crucially, the GRU *still encodes the category*
(`belief_acc ≈ 0.88`) — the actor simply never learns to *use* it. The reason is
that "always go top" already earns the bonus on 2/3 of maps (rocky + balanced),
and PPO is **on-policy**: once it settles there it stops sampling "bottom on a lakes
map," so the advantage signal that would connect belief→door never appears. This is
an **exploration / credit-assignment** trap, not a representation failure — the
belief is present but *causally unused*.

The fix is a large, **annealed** entropy ($0.15 \to 0$): explore both doors hard
early — before the shortcut basin closes — then commit. A 4-config × 3-seed sweep
confirmed it: `ent 0.15 + anneal` escaped on 2/3 seeds, constant $0.12$ on 1/3, and
the baseline $0.045$ (even with annealing) on 0/3. Two contrasts sharpen the lesson:
model-based DreamerV3/STORM do **not** hit this trap (their imagined rollouts supply
the counterfactual "what if I went bottom?" that on-policy PPO must sample for real),
and over-weighting the auxiliary belief head (`belief_coef 1.0`) actually *hurt*
(0/3) by drowning the RL objective. So the same plain reward is solved by all three,
but the model-free agent alone needs the exploration schedule.

---

## 2. DreamerV3 (model-based, RSSM world model)

**File:** `dreamer/` — the 25M preset. **Idea:** first learn a **world model** that
predicts the future, then learn the policy entirely *inside that model's
imagination*, never touching real data during policy optimization.
([Hafner et al., 2023](https://arxiv.org/abs/2301.04104).)

DreamerV3 has two nested learning problems: (A) fit a generative model of the
environment; (B) do actor–critic on trajectories *dreamed* by that model.

### 2.1 The Recurrent State-Space Model (RSSM)

The heart of Dreamer is the **RSSM**, which factorizes the latent state into two
parts:

* a **deterministic** recurrent state $g_t \in \mathbb{R}^{3072}$ (a GRU carry), and
* a **stochastic** latent $z_t$ — a set of **32 categorical variables**, each with
  **24 classes** (a $32\times 24$ one-hot block; a discrete code, not a Gaussian).

At each step the RSSM computes two distributions over $z_t$:

$$
\begin{aligned}
&\textbf{Prior (imagination):} && \hat z_t \sim p_\theta(\hat z_t \mid g_t)
   &&\text{— what the model expects, before seeing }o_t\\
&\textbf{Posterior (from data):} && z_t \sim q_\theta(z_t \mid g_t, e_t)
   &&\text{— corrected by the encoded observation } e_t=\text{enc}(o_t)\\
&\textbf{Recurrence:} && g_t = \mathrm{GRU}_\theta\big(g_{t-1},\,[z_{t-1}, a_{t-1}]\big).
\end{aligned}
$$

The GRU is a **block GRU** (8 parallel blocks over the 3072-d carry), and the
category distributions get a 1% **unimix** (mixed with a uniform) so no code
collapses to certainty. The full model state is $s_t = [g_t, z_t]$; everything the
agent predicts or acts on is a function of $s_t$.

Four small heads decode $s_t$:

$$
\hat o_t = \text{dec}_\theta(s_t),\quad
\hat r_t \sim p_\theta(r_t\mid s_t),\quad
\hat\gamma_t \sim p_\theta(\text{cont}_t\mid s_t),\quad
a_t \sim \pi_\theta(a_t\mid s_t).
$$

The encoder/decoder are 4-block **MLPs with RMSNorm + SiLU** on the flat $3974$-d
vector (no CNN — the world is symbolic).

### 2.2 World-model loss

The model is trained on real replayed sequences to (a) reconstruct the observation,
reward, and episode-continue flag from the **posterior**, and (b) make the
**prior** match the posterior so it can predict without observations. With
$s_t=[g_t,z_t]$:

$$
\mathcal L_{\text{wm}} = \mathbb{E}\Big[\;
\underbrace{-\ln p_\theta(o_t\mid s_t)}_{\text{recon}}
\underbrace{-\ln p_\theta(r_t\mid s_t)}_{\text{reward}}
\underbrace{-\ln p_\theta(\text{cont}_t\mid s_t)}_{\text{continue}}
+\;\beta\,\mathcal L_{\text{KL}}\;\Big].
$$

The KL term ties prior and posterior together and is **balanced** with **free nats**
to stop it from collapsing:

$$
\mathcal L_{\text{KL}} = 0.5\,\max\!\big(1,\,\mathrm{KL}[\,\mathrm{sg}(q)\,\Vert\,p\,]\big)
\;+\; 0.1\,\max\!\big(1,\,\mathrm{KL}[\,q\,\Vert\,\mathrm{sg}(p)\,]\big),
$$

where $\mathrm{sg}$ is stop-gradient and the two coefficients are the **dynamics**
(train the prior toward the posterior) and **representation** (train the posterior
toward the prior) weights; the $\max(1,\cdot)$ is the free-nats floor. Rewards are
modeled with a **twohot symlog** distribution: the scalar target is `symlog`-
compressed and represented as a soft two-hot over 255 exponentially-spaced bins,
which handles rewards of very different magnitudes without tuning.

### 2.3 Learning the policy in imagination

This is what makes Dreamer *model-based*. Starting from every state in a replay
batch, the frozen world model **rolls itself forward** for a short horizon
$H = 15$ using only the prior (no observations), producing an imagined trajectory
$s_t, s_{t+1}, \dots, s_{t+H}$ with predicted rewards $\hat r$ and continues
$\hat\gamma$. On these dreamed trajectories:

* the **critic** $v_\psi(s)$ regresses to **$\lambda$-returns** ($\lambda=0.95$),
  a bootstrapped mixture of $n$-step targets:

$$
V^\lambda_t = \hat r_t + \hat\gamma_t\big[(1-\lambda)\,v_\psi(s_{t+1}) + \lambda V^\lambda_{t+1}\big];
$$

* the **actor** maximizes those returns. Advantages are scaled by **return
  normalization (RetNorm)** — divide by an EMA of the 5th–95th percentile spread
  $S=\max(1,\,\mathrm{Per}(R,95)-\mathrm{Per}(R,5))$ — so a single entropy setting
  works across reward scales:

$$
\mathcal L_{\text{actor}} = -\,\mathbb{E}\Big[\tfrac{1}{S}\big(V^\lambda_t - v_\psi(s_t)\big)\,\ln\pi_\theta(a_t\mid s_t)\Big]
\; - \; \eta\,\mathcal H[\pi_\theta],\quad \eta = 0.01.
$$

A **slow critic** (an EMA copy, decay 0.98, with a cross-entropy slow-regularizer)
stabilizes the bootstrap. Optimization uses **LaProp** ($\epsilon=10^{-20}$) with
adaptive gradient clipping (**AGC 0.3**), learning rate $4\times10^{-5}$.

### 2.4 The memory subtlety that mattered here

Dreamer's memory quality depends on a training detail specific to this
implementation. The replay buffer samples fixed-length chunks of
`batch_length = 64` steps and does **not** carry the RSSM state across consecutive
chunks (stock DreamerV3's `replay_context` does; this port does not). So the
**effective training context equals `batch_length`**. On fork_wall the
evidence→door dependency spans ~75 steps — *longer than 64* — so at
`batch_length = 64` the world model with a **small** RSSM never sees cause and
consequence in the same window and collapses to a constant door. The fix is to raise
the memory budget, by **either** a longer context (`batch_length = 128`) **or** a
larger RSSM (the 25M preset). The model shipped here (25M) solves it at
`batch_length = 64`; the small 12M model needs `batch_length = 128`. This is a clean
*capacity × context* trade-off — see §4.

---

## 3. STORM (model-based, Transformer world model)

**File:** `storm/`. **Idea:** the same imagination-based recipe as Dreamer, but the
world model's dynamics are a **Transformer** over latent tokens instead of a GRU.
(STORM — Stochastic **T**ransformer-based w**OR**ld **M**odels,
[Zhang et al., 2023](https://arxiv.org/abs/2310.09615).)

### 3.1 Categorical latents + a Transformer dynamics core

As in Dreamer, each observation is encoded and mapped to a **stochastic categorical
latent** $z_t$ — here **32 variables × 32 classes**, with 1% unimix. The difference
is how the next latent is predicted. Instead of a recurrent carry $g_t$, STORM keeps
a **sequence of past latent–action tokens** and predicts the future with
**self-attention**:

$$
h_t = \mathrm{Transformer}_\theta\big([\,(z_{t-L+1},a_{t-L+1}),\dots,(z_t,a_t)\,]\big),
\qquad \hat z_{t+1} \sim p_\theta(\hat z_{t+1}\mid h_t).
$$

The Transformer is **2 layers, 512-dim, 8 heads**, with a maximum context length of
**64** tokens. Because attention connects any two positions in the window directly
(a path length of 1, versus a GRU's $O(\Delta t)$ recurrent path), a Transformer can
in principle bind "terrain at step 12" to "door at step 85" more easily — which is
why STORM solves the task at context 64 where the *small* recurrent model does not
(§4).

### 3.2 World-model loss

Structurally identical to Dreamer: reconstruct observation, reward, and continue
from $z_t$, and match the prior (Transformer prediction) to the posterior
(encoder-corrected) latent via a balanced KL with free nats:

$$
\mathcal L_{\text{wm}} = \mathbb{E}\Big[
-\ln p(o_t\mid z_t) - \ln p(r_t\mid z_t) - \ln p(\text{cont}_t\mid z_t)
+ 0.5\,\mathrm{KL}[\mathrm{sg}(q)\Vert p] + 0.1\,\mathrm{KL}[q\Vert\mathrm{sg}(p)]\Big],
$$

with `free_nats = 1`. Reward and value heads again use **symexp two-hot** over 255
bins; norms are RMSNorm, activations SiLU. A single optimizer (lr $10^{-4}$, AGC 0.3)
trains the whole model.

### 3.3 Imagination

STORM learns its actor–critic in imagination just like Dreamer, but with STORM's
short windows: an imagination **context of 8** steps priming the Transformer, then a
**rollout of 16** imagined steps. The critic uses $\lambda$-returns ($\lambda=0.95$)
with a slow-value EMA (rate 0.02) and percentile **return normalization** (5th–95th).
The actor entropy is `0.03` — like PPO's high entropy, chosen so the greedy policy
does not deadlock and the policy is robust when sampled. Discounting is set through a
horizon: $\gamma = 1 - 1/100 = 0.99$, matching the task discount.

### 3.4 Update-to-data ratio

STORM is trained with a high **train_ratio = 256** (replayed 16×64 batches per env
step, in this framework's units) — far above the paper-default 8 that earlier failed
runs used. Together with the Transformer's easy long-range binding, this is what
lets STORM reach ≈98% success.

---

## 4. Side-by-side and the empirical lesson

### 4.1 Architecture comparison

| | **PPO+GRU** | **DreamerV3** | **STORM** |
|---|---|---|---|
| Family | model-free | model-based (world model) | model-based (world model) |
| Recurrent state | GRU hidden (128) | RSSM: 3072 deterministic **+** 32×24 discrete latent | Transformer context over 32×32 discrete latents |
| Memory mechanism | GRU gates (BPTT) | GRU carry + latent, trained to predict | self-attention over latent tokens |
| Policy trained on | real trajectories | **imagined** rollouts ($H=15$) | **imagined** rollouts (context 8, roll 16) |
| Reward head | scalar critic | two-hot symlog (255 bins) | two-hot symexp (255 bins) |
| Approx. params | ~0.5 M | ~25 M | ~15 M |
| Discount $\gamma$ | 0.99 | 0.997 (agent) / task 0.99 | 0.99 |
| Belief read-out | auxiliary head (`belief_coef 0.3`) | post-hoc probe on $s_t$ | post-hoc probe on $z_t$ |
| Held-out decisive success | 97.7% | 97.0% (25M) | ≈98% |
| Extra ingredient needed | annealed entropy $0.15\to0$ (else trapped) | memory budget (capacity ∨ context) | long-range attention + high train_ratio |

### 4.2 The lesson: memory needs a sufficient "budget"

Sweeping the Dreamer over `{12M, 25M} × {batch_length 64, 128}` on the identical env
gave a clean result:

| model | batch_length | decisive-door success (test) |
|---|---|---|
| 12M | 64 | **49.7 %** (fails — constant door) |
| 12M | 128 | 94.3 % |
| 25M | 64 | **97.0 %** ✅ (shipped here) |
| 25M | 128 | 95.0 % |

Only the **small-model + short-context** corner fails. The memory dependency needs a
threshold "budget" that can be met by **either** more recurrent capacity **or** a
longer training context. This also explains why STORM manages at context 64: its
Transformer binds long-range dependencies directly, so it does not pay the recurrent
model's context tax. Same task, same reward — three different ways to have *enough
memory to remember*.

---

## References

* Schulman, Wolski, Dhariwal, Radford, Klimov (2017). *Proximal Policy Optimization
  Algorithms.* arXiv:1707.06347.
* Cho et al. (2014). *Learning Phrase Representations using RNN Encoder–Decoder.*
  arXiv:1406.1078.
* Schulman, Moritz, Levine, Jordan, Abbeel (2016). *High-Dimensional Continuous
  Control Using Generalized Advantage Estimation.* arXiv:1506.02438.
* Hafner, Pasukonis, Ba, Lillicrap (2023). *Mastering Diverse Domains through World
  Models (DreamerV3).* arXiv:2301.04104.
* Zhang, Wang, Wu et al. (2023). *STORM: Efficient Stochastic Transformer based World
  Models for Reinforcement Learning.* arXiv:2310.09615.
* Ng, Harada, Russell (1999). *Policy Invariance under Reward Transformations
  (potential-based shaping).* ICML.
