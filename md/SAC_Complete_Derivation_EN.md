---
title: "From Q-learning to Soft Actor-Critic: A Complete Derivation"
subtitle: "Bellman optimality, temporal differences, DQN, maximum-entropy policy iteration, and continuous robot control"
date: "2026-10-18"
lang: en-US
toc-title: "Contents"
---

# From Q-learning to Soft Actor-Critic: A Complete Derivation

<!-- course-meta:start -->
[Course home](https://github.com/zengxiong111/Foundation_of_Embodied_AI) · [简体中文](https://github.com/zengxiong111/Foundation_of_Embodied_AI/blob/main/md/SAC_Complete_Derivation_CN.md) · [PDF](https://zengxiong111.github.io/zengxiong.github.io/files/SAC_Complete_Derivation_EN.pdf)

| Item | Details |
|---|---|
| Applicability | Q-learning, TD/SARSA, DQN, soft Bellman equations, modern SAC, twin critics, reparameterization and automatic entropy tuning; instructional pseudocode |
| Documentation review date | 2026-10-18 |
| License | CC BY 4.0 · Xiong Zeng and contributors |

<!-- course-meta:end -->

<!-- course-toc:start -->
## Contents

- [Reading guide: six questions](#course-section-01)
- [1. From MDPs to V and Q: why value functions matter](#course-section-02)
  - [1.1 The reinforcement-learning objective](#course-section-03)
  - [1.2 State values versus action values](#course-section-04)
  - [1.3 Bellman expectation: turning a long horizon into one step](#course-section-05)
  - [1.4 Policy evaluation and improvement](#course-section-06)
- [2. From Bellman expectation to Bellman optimality](#course-section-07)
  - [2.1 Optimal values and the optimality equations](#course-section-08)
  - [2.2 What is the Bellman operator buying us?](#course-section-09)
  - [2.3 A two-action example](#course-section-10)
- [3. Temporal-difference learning, SARSA, and tabular Q-learning](#course-section-11)
  - [3.1 Monte Carlo versus TD: learning before an episode ends](#course-section-12)
  - [3.2 SARSA evaluates the policy being executed](#course-section-13)
  - [3.3 Q-learning can explore while its target is greedy](#course-section-14)
  - [3.4 Comparing the three TD targets](#course-section-15)
  - [3.5 When does tabular Q-learning converge?](#course-section-16)
- [4. DQN: scaling tabular Q-learning to large state spaces](#course-section-17)
  - [4.1 From a Q-table to neural TD regression](#course-section-18)
  - [4.2 Why experience replay helps](#course-section-19)
  - [4.3 Why target networks help](#course-section-20)
  - [4.4 How maximization amplifies value overestimation](#course-section-21)
  - [4.5 Double Q-learning, Double DQN, and SAC's twin critics differ](#course-section-22)
  - [4.6 The deadly triad: why deep value learning can diverge](#course-section-23)
- [5. Why continuous actions push us toward actor-critic methods](#course-section-24)
  - [5.1 The action maximum becomes an expensive inner optimization](#course-section-25)
  - [5.2 An actor is an amortized optimizer of actions](#course-section-26)
  - [5.3 Why does SAC adopt a stochastic actor?](#course-section-27)
- [6. Maximum-entropy reinforcement learning](#course-section-28)
  - [6.1 Policy entropy and temperature](#course-section-29)
  - [6.2 A consistent convention for soft Q and soft V](#course-section-30)
- [7. Hard versus soft Bellman backups: from max to LogSumExp](#course-section-31)
  - [7.1 Optimal soft value for a finite discrete action set](#course-section-32)
  - [7.2 The zero-temperature limit recovers the maximum](#course-section-33)
  - [7.3 Continuous-action integrals and their assumptions](#course-section-34)
  - [7.4 Do not confuse soft optimality with soft policy evaluation](#course-section-35)
- [8. Soft policy improvement: why Boltzmann policies and KL divergence appear](#course-section-36)
  - [8.1 From the ideal distribution to a trainable policy](#course-section-37)
  - [8.2 Reverse KL removes the intractable normalizer](#course-section-38)
  - [8.3 Improvement under exact soft policy iteration](#course-section-39)
- [9. From soft policy iteration to modern SAC](#course-section-40)
  - [9.1 Two conceptual operations, represented by neural networks](#course-section-41)
  - [9.2 Why SAC can reuse data collected by older policies](#course-section-42)
  - [9.3 How this differs from on-policy PPO](#course-section-43)
- [10. The complete SAC critic: target networks, twin Q, and TD regression](#course-section-44)
  - [10.1 From the soft Bellman equation to a sampled target](#course-section-45)
  - [10.2 Why use the minimum of two critics?](#course-section-46)
  - [10.3 True termination versus time-limit truncation](#course-section-47)
  - [10.4 Slowly updating the target critics](#course-section-48)
- [11. The complete SAC actor: reparameterization, bounds, and log probabilities](#course-section-49)
  - [11.1 From KL projection to the actor objective](#course-section-50)
  - [11.2 The reparameterization trick](#course-section-51)
  - [11.3 Correct log probability for a tanh-squashed Gaussian](#course-section-52)
  - [11.4 Normalized versus physical robot actions](#course-section-53)
- [12. Automatic entropy tuning: why learn the temperature alpha?](#course-section-54)
  - [12.1 From fixed temperature to a target-entropy constraint](#course-section-55)
  - [12.2 The temperature objective and a sign check](#course-section-56)
  - [12.3 Why parameterize alpha as exp(eta)?](#course-section-57)
- [13. Putting together the complete modern SAC algorithm](#course-section-58)
  - [13.1 Components and three losses](#course-section-59)
  - [13.2 The full sequence of one SAC update](#course-section-60)
  - [13.3 Original SAC v1 versus modern SAC: is V explicitly learned?](#course-section-61)
- [14. How the main algorithms relate](#course-section-62)
  - [14.1 The most important Q-learning versus SAC distinction](#course-section-63)
  - [14.2 PPO versus SAC: score-function and pathwise gradients](#course-section-64)
- [15. Robot-control limits and common misconceptions](#course-section-65)
  - [15.1 Why critic extrapolation is especially important in manipulation](#course-section-66)
  - [15.2 Joint positions, velocities, torques, and action normalization](#course-section-67)
  - [15.3 When standard SAC should not be copied verbatim](#course-section-68)
  - [15.4 Five misconceptions to avoid](#course-section-69)
- [16. The complete derivation in six equations](#course-section-70)
- [References and further reading](#course-section-71)
<!-- course-toc:end -->

> **Scope.** This self-contained tutorial reorganizes and extends the existing *Soft Actor-Critic (SAC): Complete Derivation and Design Motivations*. Starting with Q-learning, it derives modern continuous-action SAC by following the sequence **problem $\rightarrow$ mathematical principle $\rightarrow$ algorithmic design**. It contains no experimental section. A consistent convention for state/action values, entropy, and termination prevents accidental mixing of ordinary Bellman equations, soft Bellman equations, and their optimality counterparts.
>
> **Original course:** [Course home](https://github.com/zengxiong111/Foundation_of_Embodied_AI) · [Original Chinese notes](https://github.com/zengxiong111/Foundation_of_Embodied_AI/blob/2c62eb6ac720373112d8852e1e3031206c3bde44/md/SAC_Complete_Derivation_CN.md) · [Original English notes](https://github.com/zengxiong111/Foundation_of_Embodied_AI/blob/2c62eb6ac720373112d8852e1e3031206c3bde44/md/SAC_Complete_Derivation_EN.md). This revision is included in the course repository; references to the original notes are pinned to the preceding commit for provenance.

<a id="course-section-01"></a>

## Reading guide: six questions

- What does an action-value $Q(s,a)$ measure, and how does it differ from a state-value $V(s)$?
- Why can Q-learning learn without a dynamics model and update from behavior generated by a different policy?
- What problems do experience replay, target networks, and Double Q-learning address?
- Why is $\max_a Q(s,a)$ difficult for a high-dimensional robot action space?
- How does hard Bellman optimality connect to soft optimality, soft policy evaluation, and finally the SAC actor and critic?
- What specific problems are addressed by twin critics, reparameterization, and automatic entropy tuning?

**Learning path:** Bellman expectation $\to$ Bellman optimality $\to$ TD / Q-learning $\to$ DQN $\to$ continuous-action actor-critic $\to$ maximum-entropy RL $\to$ soft policy iteration $\to$ SAC.

**Notation.** $s$ denotes the state; $a$, the action; $r(s,a)$, the expected one-step reward; $P(s'\mid s,a)$, the transition kernel; $\pi(a\mid s)$, a policy; and $\gamma\in[0,1)$, the discount factor. The termination indicator $d=1$ represents a true MDP termination, not merely an episode time-limit truncation. We write $Q^\pi,V^\pi$ for the values of a fixed policy and $Q^*,V^*$ for optimal values. In maximum-entropy settings we explicitly distinguish *soft* values where needed.

---

<a id="course-section-02"></a>

## 1. From MDPs to V and Q: why value functions matter

<a id="course-section-03"></a>

### 1.1 The reinforcement-learning objective

A discounted Markov decision process is

$$
\mathcal M=(\mathcal S,\mathcal A,P,r,\gamma),\qquad 0\le\gamma<1.
$$

A trajectory $\tau=(s_0,a_0,r_0,s_1,\ldots)$ has discounted return

$$
G_t=\sum_{k=0}^{\infty}\gamma^k r_{t+k},\qquad
J(\pi)=\mathbb E_{\tau\sim\pi}[G_0].
$$

The initial-state distribution is implicit in $J(\pi)$; in manipulation, it may vary with the initial object pose, joint configuration, or contact state. **An immediate reward is not an action value**: the former scores the present step; the latter also accounts for consequences far into the future.

<a id="course-section-04"></a>

### 1.2 State values versus action values

For a fixed policy $\pi$, define

$$
\boxed{V^\pi(s)=\mathbb E_\pi[G_t\mid s_t=s],\qquad
Q^\pi(s,a)=\mathbb E_\pi[G_t\mid s_t=s,a_t=a].}
$$

$V^\pi(s)$ asks, "What is the expected future return from this state if we follow $\pi$?" $Q^\pi(s,a)$ asks, "What if we first take the specified action $a$, then follow $\pi$?" The Q-function directly supports comparisons between actions.

By the law of total expectation, for discrete actions,

$$
\boxed{V^\pi(s)=\sum_a\pi(a\mid s)Q^\pi(s,a).}
$$

For continuous actions replace the sum by $\int_{\mathcal A}\pi(a\mid s)Q^\pi(s,a)\,da$. A grasp can yield zero immediate reward but still have high Q-value if it puts the object in a state where subsequent lifting succeeds.

<a id="course-section-05"></a>

### 1.3 Bellman expectation: turning a long horizon into one step

Using $G_t=r_t+\gamma G_{t+1}$ and the Markov property,

$$
\boxed{Q^\pi(s,a)=r(s,a)+\gamma\mathbb E_{s'\sim P(\cdot\mid s,a)}[V^\pi(s')].}
$$

Substituting $V^\pi(s')=\mathbb E_{a'\sim\pi}[Q^\pi(s',a')]$ gives

$$
\boxed{Q^\pi(s,a)=r(s,a)+\gamma\mathbb E_{s',a'\sim P,\pi}[Q^\pi(s',a')].}
$$

This is the **Bellman expectation equation**: it evaluates a *given* policy, without maximizing over actions. The state-value form is $V^\pi(s)=\mathbb E_{a\sim\pi,s'\sim P}[r(s,a)+\gamma V^\pi(s')]$.

<a id="course-section-06"></a>

### 1.4 Policy evaluation and improvement

Classical policy iteration alternates between two operations. **Evaluation:** $\pi_k\mapsto Q^{\pi_k}$ estimates how good the current policy is. **Improvement:** $\pi_{k+1}(s)\in\arg\max_a Q^{\pi_k}(s,a)$ chooses a greedier policy. SAC preserves this two-step structure but replaces greedy improvement by entropy-regularized improvement of a stochastic policy.

<a id="course-section-07"></a>

## 2. From Bellman expectation to Bellman optimality

<a id="course-section-08"></a>

### 2.1 Optimal values and the optimality equations

Define $V^*(s)=\sup_\pi V^\pi(s)$ and $Q^*(s,a)=\sup_\pi Q^\pi(s,a)$. For a finite discrete action set, when the maximum exists,

$$
\boxed{V^*(s)=\max_a Q^*(s,a),}
$$

$$
\boxed{Q^*(s,a)=r(s,a)+\gamma\mathbb E_{s'\sim P}[\max_{a'}Q^*(s',a')].}
$$

The second equation is **Bellman optimality**. In contrast to policy evaluation, it inserts a maximization over the next action, rather than averaging according to a prescribed policy. It characterizes the optimal action-value function as a fixed point.

<a id="course-section-09"></a>

### 2.2 What is the Bellman operator buying us?

Define

$$
(\mathcal T^*Q)(s,a)
=r(s,a)+\gamma\mathbb E_{s'}\left[\max_{a'}Q(s',a')\right].
$$

Under familiar bounded-reward discounted-MDP assumptions, the fact that $|\max_i x_i-\max_i y_i|\le\max_i|x_i-y_i|$ implies

$$
\boxed{\|\mathcal T^*Q_1-\mathcal T^*Q_2\|_\infty
\le\gamma\|Q_1-Q_2\|_\infty.}
$$

Because $\gamma<1$, the operator is a contraction on the relevant bounded-function space, with a unique fixed point $Q^*$. Exact value iteration $Q_{k+1}=\mathcal T^*Q_k$ converges. **This is not a convergence theorem for general neural-network Q-learning**, whose stochastic approximation and changing function class introduce further difficulties.

<a id="course-section-10"></a>

### 2.3 A two-action example

Consider two grasp choices with zero immediate reward. The optimal value of the state reached by the left grasp is 8, versus 10 for the right grasp. With $\gamma=0.9$,

$$
Q^*(s,a_L)=0+0.9\times8=7.2,\qquad
Q^*(s,a_R)=0+0.9\times10=9.
$$

Q prefers the right grasp even though neither action yields an immediate reward. This is why one-step reward maximization is insufficient for long-horizon manipulation.

<a id="course-section-11"></a>

## 3. Temporal-difference learning, SARSA, and tabular Q-learning

<a id="course-section-12"></a>

### 3.1 Monte Carlo versus TD: learning before an episode ends

Monte Carlo (MC) methods use a completed return $G_t$ as a training target. Temporal-difference (TD) methods use a one-step reward plus an estimate of the next state's value. For policy evaluation, TD(0) uses

$$
\delta_t^V=r_t+\gamma(1-d_t)V(s_{t+1})-V(s_t),
$$

$$
V(s_t)\leftarrow V(s_t)+\eta_t\delta_t^V.
$$

TD can update before an episode ends, but its target **bootstraps** from another imperfect estimate. This introduces approximation-dependent bias and error propagation. It is an oversimplification to claim that the complete MC–TD distinction is always just "unbiased versus biased"; the estimand, sampling process, and approximation assumptions matter.

<a id="course-section-13"></a>

### 3.2 SARSA evaluates the policy being executed

SARSA uses the actual next action $a_{t+1}$ sampled from the behavior policy:

$$
\boxed{y_t^{\mathrm{SARSA}}=r_t+\gamma(1-d_t)Q(s_{t+1},a_{t+1}).}
$$

$$
Q(s_t,a_t)\leftarrow Q(s_t,a_t)
+\eta_t\big(y_t^{\mathrm{SARSA}}-Q(s_t,a_t)\big).
$$

When $a_{t+1}$ follows the same $\varepsilon$-greedy policy being improved, standard SARSA is **on-policy TD control**: its target reflects the exploratory policy's next action.

<a id="course-section-14"></a>

### 3.3 Q-learning can explore while its target is greedy

Tabular Watkins Q-learning uses

$$
\boxed{y_t^Q=r_t+\gamma(1-d_t)\max_{a'}Q_t(s_{t+1},a'),}
$$

$$
\boxed{Q_{t+1}(s_t,a_t)=Q_t(s_t,a_t)
+\eta_t\left[y_t^Q-Q_t(s_t,a_t)\right].}
$$

The bracketed quantity is the TD error $\delta_t$: **new estimate = old estimate + step size × prediction error**. An $\varepsilon$-greedy behavior policy can take a greedy action with probability $1-\varepsilon$ and select a uniform exploratory action with probability $\varepsilon$ (conventions regarding overlap can vary). The **behavior policy** $\beta$ generates transitions. The Q-learning target uses greedy maximization, corresponding to a different **target policy**. Thus tabular Q-learning is **off-policy**.

For a numerical illustration, suppose $Q(s,a)=4$, $r=1$, the highest next-state Q equals 6, $\gamma=0.9$, and $\eta=0.2$. Then

$$
y=1+0.9\times6=6.4,\qquad
Q_{\mathrm{new}}=4+0.2(6.4-4)=4.48.
$$

<a id="course-section-15"></a>

### 3.4 Comparing the three TD targets

| Algorithm | One-step target (nonterminal case) | Next-action rule | Standard data regime |
|:--|:--|:--|:--|
| SARSA | $r+\gamma Q(s',a')$ | Sample $a'$ from behavior policy | On-policy |
| Expected SARSA | $r+\gamma\mathbb E_{a'\sim\pi}Q(s',a')$ | Expectation under chosen target policy | May be on/off-policy |
| Q-learning | $r+\gamma\max_{a'}Q(s',a')$ | Greedy optimality backup | Off-policy |

**Important:** Replay buffers are *not what makes Q-learning off-policy*. Tabular Q-learning can be off-policy even when each sample is processed immediately. A replay buffer is a separate mechanism for reusing transitions and altering their sampling correlations.

<a id="course-section-16"></a>

### 3.5 When does tabular Q-learning converge?

The classical result covers finite tabular state/action settings with a stationary MDP, bounded rewards, sufficiently repeated visits, and appropriate step sizes (e.g., Robbins–Monro-type conditions), among other assumptions. It establishes convergence to $Q^*$ in the specified setting. **It does not automatically cover nonlinear function approximation, inadequate coverage, changing environments, or arbitrary combinations of off-policy training and bootstrapping.** This distinction matters when we move to DQN and SAC. See [Watkins and Dayan (1992)](https://doi.org/10.1007/BF00992698).

<a id="course-section-17"></a>

## 4. DQN: scaling tabular Q-learning to large state spaces

<a id="course-section-18"></a>

### 4.1 From a Q-table to neural TD regression

A high-dimensional observation space makes a Q-table impractical. Introduce a neural approximator $Q_\phi(s,a)$ and the sample target

$$
y=r+\gamma(1-d)\max_{a'}Q_{\bar\phi}(s',a').
$$

Train by minimizing

$$
\boxed{J_Q(\phi)=\frac12\mathbb E_{(s,a,r,s',d)\sim\mathcal D}
\left[\big(Q_\phi(s,a)-\operatorname{stopgrad}(y)\big)^2\right].}
$$

This has the same *regression structure* as SAC's critic loss: both fit $Q$ to a bootstrap-based TD target. The crucial differences concern which action is evaluated in the next state, whether entropy is included, and whether a separately parameterized actor is learned.

<a id="course-section-19"></a>

### 4.2 Why experience replay helps

Consecutive observations are strongly correlated. Updating a neural Q-function only on the most recent transition can make gradients reflect a narrow and changing slice of experience. A replay buffer,

$$
\mathcal D=\{(s_t,a_t,r_t,s_{t+1},d_t)\}_{t=1}^N,
$$

supports minibatch sampling from earlier interactions, weakens short-range temporal dependence, and permits multiple updates from a transition. **Replay does not manufacture coverage:** a dataset consisting entirely of near-identical grasps does not tell us what a never-tried grasp would accomplish.

<a id="course-section-20"></a>

### 4.3 Why target networks help

If the prediction $Q_\phi(s,a)$ and target $r+\gamma\max_{a'}Q_\phi(s',a')$ change aggressively together, the supervised target is continuously moving. DQN introduces a delayed target Q-network $Q_{\bar\phi}$. It may be updated periodically with $\bar\phi\leftarrow\phi$, or, in other actor-critic algorithms, by Polyak averaging:

$$
\boxed{\bar\phi\leftarrow(1-\tau)\bar\phi+\tau\phi,
\qquad 0<\tau\ll1.}
$$

Periodic hard copying is characteristic of classical DQN; small-step Polyak updates are common in modern continuous-control methods such as SAC and TD3. These stabilizers limit target drift, but do not remove the errors inherent in approximation.

<a id="course-section-21"></a>

### 4.4 How maximization amplifies value overestimation

Suppose $\widehat Q(s,a)=Q(s,a)+\epsilon_a$, with unbiased estimation errors for individual actions. Maximizing noisy estimates can still create positive bias:

$$
\mathbb E\left[\max_a(Q_a+\epsilon_a)\right]
\ge\max_a\mathbb E[Q_a+\epsilon_a]=\max_aQ_a.
$$

The inequality follows from the convexity of the maximum (Jensen's inequality). Larger action sets and noisier estimates can make this selection effect more consequential.

<a id="course-section-22"></a>

### 4.5 Double Q-learning, Double DQN, and SAC's twin critics differ

The motivating principle behind Double Q-learning is to **separate action selection from action evaluation**, thereby reducing the bias that arises when both are performed using the same noisy estimator:

- **Tabular Double Q-learning:** two Q-tables alternately choose and evaluate greedy actions, with their roles switched during different updates.
- **Double DQN:** the online network selects the maximizing action, while the target network evaluates it:

$$
\boxed{a^*=\arg\max_{a'}Q_\phi(s',a'),\qquad
 y_{\mathrm{DDQN}}=r+\gamma(1-d)Q_{\bar\phi}(s',a^*).}
$$

- **SAC's clipped double Q:** for an action $a'$ sampled from the actor, the target uses $\min(Q_{\bar\phi_1}(s',a'),Q_{\bar\phi_2}(s',a'))$. This conservative estimator reduces some overoptimism but can underestimate value.

The three mechanisms are related **but their formulas and statistical biases are not identical**. See [van Hasselt (2010)](https://papers.neurips.cc/paper_files/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html) and [Double DQN](https://arxiv.org/abs/1509.06461).

<a id="course-section-23"></a>

### 4.6 The deadly triad: why deep value learning can diverge

The combination of **function approximation, bootstrapping, and off-policy training** is often called the *deadly triad*. A Q-function with limited state-action support may propagate its own prediction errors into future targets. The actor or greedy selector may then exploit the inaccurate values, further concentrating the data or targets in unreliable regions. Experience replay, target networks, and twin critics alleviate some problems; they are not blanket convergence guarantees.

<a id="course-section-24"></a>

## 5. Why continuous actions push us toward actor-critic methods

<a id="course-section-25"></a>

### 5.1 The action maximum becomes an expensive inner optimization

Q-learning uses

$$
\max_{a'\in\mathcal A}Q_\phi(s',a').
$$

For a small discrete action set, we can enumerate all actions. Robot control actions are often continuous, $a\in\mathbb R^d$, e.g., many joint targets or torque commands. A learned $Q_\phi(s,a)$ is typically nonconvex in $a$. Solving a new high-dimensional optimization problem for every Bellman target is expensive and can be trapped at suboptimal local maxima.

<a id="course-section-26"></a>

### 5.2 An actor is an amortized optimizer of actions

Introduce a parameterized deterministic actor $a=\mu_\theta(s)$ and retain a critic $Q_\phi(s,a)$. A standard deterministic actor objective is

$$
J_\mu(\theta)=-\mathbb E_{s\sim\mathcal D}[Q_\phi(s,\mu_\theta(s))],
$$

$$
\nabla_\theta J_\mu
=-\mathbb E_s\left[\nabla_aQ_\phi(s,a)|_{a=\mu_\theta(s)}
\nabla_\theta\mu_\theta(s)\right].
$$

The actor **amortizes action optimization** into a forward pass, instead of repeatedly solving $\arg\max_aQ$ exactly. DDPG exemplifies off-policy deterministic actor-critic control. TD3 augments that family with twin critics, delayed actor updates, and target policy smoothing, among other stabilization choices.

<a id="course-section-27"></a>

### 5.3 Why does SAC adopt a stochastic actor?

Deterministic policies have no intrinsic conditional action entropy and often add separate exploratory noise. SAC instead parameterizes a stochastic conditional policy $\pi_\theta(a\mid s)$ and jointly rewards high expected Q and policy entropy:

$$
\max_\pi\;\mathbb E_{a\sim\pi}[Q(s,a)]
+\alpha\mathcal H(\pi(\cdot\mid s)).
$$

This is a design choice, not a theorem that continuous control *must* have an explicit actor: discretization, sampling-based optimization, and implicit policy methods also exist. SAC uses a tractable, differentiable actor to make continuous soft policy improvement efficient.

<a id="course-section-28"></a>

## 6. Maximum-entropy reinforcement learning

<a id="course-section-29"></a>

### 6.1 Policy entropy and temperature

For discrete actions, $\mathcal H(\pi)=-\sum_a\pi(a\mid s)\log\pi(a\mid s)$. For continuous actions, differential entropy is

$$
\boxed{\mathcal H(\pi(\cdot\mid s))
=-\int_{\mathcal A}\pi(a\mid s)\log\pi(a\mid s)\,da
=\mathbb E_{a\sim\pi}[-\log\pi(a\mid s)].}
$$

Unlike discrete Shannon entropy, differential entropy may be negative and depends on the units of the action coordinates. For a one-dimensional Gaussian, $\mathcal N(\mu,\sigma^2)$, $\mathcal H=\tfrac12\log(2\pi e\sigma^2)$.

With temperature $\alpha\ge0$, maximum-entropy RL optimizes

$$
\boxed{J_{\mathrm{ME}}(\pi)
=\mathbb E_{\tau\sim\pi}\left[
\sum_{t=0}^{\infty}\gamma^t
\big(r_t-\alpha\log\pi(a_t\mid s_t)\big)
\right].}
$$

A larger $\alpha$ puts more weight on entropy; the $\alpha\to0^+$ limit approaches ordinary return maximization. The entropy contribution depends on the **current policy**; it is not a policy-independent environmental reward.

<a id="course-section-30"></a>

### 6.2 A consistent convention for soft Q and soft V

To align the derivation with modern SAC targets, define

$$
\begin{aligned}
V^\pi_{\mathrm{soft}}(s)
&=\mathbb E_\pi\left[
\sum_{t=0}^{\infty}\gamma^t
(r_t-\alpha\log\pi(a_t\mid s_t))\mid s_0=s\right],\\
Q^\pi_{\mathrm{soft}}(s,a)
&=\mathbb E_\pi\left[r_0+\sum_{t=1}^{\infty}\gamma^t
(r_t-\alpha\log\pi(a_t\mid s_t))\mid s_0=s,a_0=a\right].
\end{aligned}
$$

The **current action's entropy is included in soft V**; because the current action is fixed, it is **excluded from soft Q** and begins at the next time step. Consequently,

$$
\boxed{V^\pi_{\mathrm{soft}}(s)
=\mathbb E_{a\sim\pi}\left[Q^\pi_{\mathrm{soft}}(s,a)-\alpha\log\pi(a\mid s)\right],}
$$

$$
\boxed{Q^\pi_{\mathrm{soft}}(s,a)
=r(s,a)+\gamma\mathbb E_{s'}[V^\pi_{\mathrm{soft}}(s')].}
$$

Combining them gives the **soft Bellman expectation equation**:

$$
\boxed{Q^\pi_{\mathrm{soft}}(s,a)
=r(s,a)+\gamma\mathbb E_{s',a'\sim P,\pi}
\left[Q^\pi_{\mathrm{soft}}(s',a')-\alpha\log\pi(a'\mid s')\right].}
$$

Compared with the ordinary expectation equation from Section 1, the structural change is the additional $-\alpha\log\pi$ term inside the next-state soft value.

<a id="course-section-31"></a>

## 7. Hard versus soft Bellman backups: from max to LogSumExp

<a id="course-section-32"></a>

### 7.1 Optimal soft value for a finite discrete action set

Fix state $s$, assume finite actions and $\alpha>0$, and suppose a candidate Q-function is given. Maximize over $p_a\ge0$, $\sum_a p_a=1$,

$$
F(p)=\sum_a p_a Q(s,a)-\alpha\sum_a p_a\log p_a.
$$

With multiplier $\lambda$ for the normalization constraint, stationarity requires

$$
\frac{\partial}{\partial p_a}\left(F(p)+\lambda(\sum_a p_a-1)\right)
=Q(s,a)-\alpha(\log p_a+1)+\lambda=0.
$$

Thus the optimal policy is a **Boltzmann distribution**:

$$
\boxed{p_a^*=
\frac{\exp(Q(s,a)/\alpha)}{\sum_b\exp(Q(s,b)/\alpha)}.}
$$

Substitution produces **LogSumExp**:

$$
\boxed{\max_p F(p)
=\alpha\log\sum_a\exp\big(Q(s,a)/\alpha\big).}
$$

Therefore, the optimal soft Bellman operator for finite discrete actions is

$$
\boxed{(\mathcal T^*_{\mathrm{soft}}Q)(s,a)
=r(s,a)+\gamma\mathbb E_{s'}\left[
\alpha\log\sum_{a'}e^{Q(s',a')/\alpha}\right].}
$$

The $Q$ here follows the soft-Q convention in Section 6, which excludes the current action's entropy. This equation is not obtained by indiscriminately adding LogSumExp to an unrelated definition of conventional Q-value.

<a id="course-section-33"></a>

### 7.2 The zero-temperature limit recovers the maximum

Let $M=\max_i q_i$ and let $n$ be the number of discrete actions. Then

$$
\alpha\log\sum_i e^{q_i/\alpha}
=M+\alpha\log\sum_i e^{(q_i-M)/\alpha}.
$$

Because $1\le\sum_i e^{(q_i-M)/\alpha}\le n$,

$$
\boxed{M\le\alpha\log\sum_i e^{q_i/\alpha}
\le M+\alpha\log n,}
$$

which proves

$$
\boxed{\lim_{\alpha\to0^+}
\alpha\log\sum_i e^{q_i/\alpha}=\max_i q_i.}
$$

This is the central conceptual bridge: **the hard maximum is the zero-temperature limit of an entropy-regularized soft maximum**. With nonzero temperature, multiple high-value actions can retain meaningful probability.

<a id="course-section-34"></a>

### 7.3 Continuous-action integrals and their assumptions

For continuous actions the ideal soft-optimal value takes the form

$$
V^*_{\mathrm{soft}}(s)
=\alpha\log\int_{\mathcal A}
\exp(Q^*_{\mathrm{soft}}(s,a)/\alpha)\,da.
$$

This expression requires a finite normalization integral and depends on the base measure for action densities. Rescaling coordinates changes differential entropy and normalization. For high-dimensional nonconvex Q-functions, the integral is generally intractable. **SAC does not directly evaluate this integral as part of its standard learning loop:** it trains an actor to approximate the corresponding high-value stochastic policy.

<a id="course-section-35"></a>

### 7.4 Do not confuse soft optimality with soft policy evaluation

| Equation / operator | What is being solved? | Treatment of the next action |
|:--|:--|:--|
| $\mathcal T^\pi Q$ | Evaluate a fixed policy $\pi$ | $\mathbb E_{a'\sim\pi}$ |
| $\mathcal T^*Q$ | Unregularized optimal control | $\max_{a'}$ |
| $\mathcal T^*_{\mathrm{soft}}Q$ | Entropy-regularized optimal control | $\alpha\log\sum e^{Q/\alpha}$ (discrete) |
| SAC critic target | Approximate soft evaluation of the **current actor** | Sample $a'\sim\pi_\theta$ and subtract $\alpha\log\pi_\theta$ |

This table is critical. SAC does **not** merely substitute a tractable explicit LogSumExp for Q-learning's max. It performs approximate *soft policy iteration*, with an actor representing the policy whose value is estimated by its critic.

<a id="course-section-36"></a>

## 8. Soft policy improvement: why Boltzmann policies and KL divergence appear

<a id="course-section-37"></a>

### 8.1 From the ideal distribution to a trainable policy

Fix a state $s$ and the soft Q of an old policy: $Q_{\mathrm{old}}(s,a)=Q^{\pi_{\mathrm{old}}}_{\mathrm{soft}}(s,a)$. The ideal improved policy solves

$$
\pi_{\mathrm{new}}
\in\arg\max_\pi\left\{
\mathbb E_{a\sim\pi}[Q_{\mathrm{old}}(s,a)]
+\alpha\mathcal H(\pi(\cdot\mid s))\right\}.
$$

With suitable integrability and density assumptions, the continuous-action solution remains the Boltzmann policy

$$
\boxed{\pi_B(a\mid s)=
\frac{\exp(Q_{\mathrm{old}}(s,a)/\alpha)}{Z_{\mathrm{old}}(s)},
\quad Z_{\mathrm{old}}(s)=\int_{\mathcal A}e^{Q_{\mathrm{old}}(s,a')/\alpha}\,da'.}
$$

However, evaluating $Z_{\mathrm{old}}(s)$ and sampling exactly from $\pi_B$ are difficult in high-dimensional action spaces.

<a id="course-section-38"></a>

### 8.2 Reverse KL removes the intractable normalizer

Introduce a parameterized actor $\pi_\theta$ and fit it to the Boltzmann policy by expanding $D_{\mathrm{KL}}(\pi_\theta\|\pi_B)$:

$$
\begin{aligned}
D_{\mathrm{KL}}(\pi_\theta\|\pi_B)
&=\mathbb E_{a\sim\pi_\theta}\left[
\log\pi_\theta(a\mid s)-\log\pi_B(a\mid s)\right]\\
&=\mathbb E_{a\sim\pi_\theta}\left[
\log\pi_\theta(a\mid s)-Q_{\mathrm{old}}(s,a)/\alpha\right]
+\log Z_{\mathrm{old}}(s).
\end{aligned}
$$

Since $\log Z_{\mathrm{old}}$ does not depend on $\theta$, minimizing this KL divergence is equivalent to

$$
\boxed{\min_\theta\mathbb E_{a\sim\pi_\theta}
\left[\alpha\log\pi_\theta(a\mid s)
-Q_{\mathrm{old}}(s,a)\right].}
$$

This is the theoretical prototype of SAC's actor loss. The direction matters: we are using $D_{\mathrm{KL}}(\pi_\theta\|\pi_B)$, not $D_{\mathrm{KL}}(\pi_B\|\pi_\theta)$. These two projections can have different approximation behavior.

<a id="course-section-39"></a>

### 8.3 Improvement under exact soft policy iteration

Assume the old policy belongs to the admissible policy class and the new policy exactly solves the statewise KL projection. Optimality then implies

$$
\mathbb E_{a\sim\pi_{\mathrm{new}}}
[Q^{\pi_{\mathrm{old}}}_{\mathrm{soft}}(s,a)-\alpha\log\pi_{\mathrm{new}}(a\mid s)]
\ge V^{\pi_{\mathrm{old}}}_{\mathrm{soft}}(s).
$$

Define the soft policy-evaluation operator

$$
(\mathcal T^\pi_{\mathrm{soft}}Q)(s,a)
=r(s,a)+\gamma\mathbb E_{s',a'\sim P,\pi}
[Q(s',a')-\alpha\log\pi(a'\mid s')].
$$

Under suitable boundedness assumptions, this operator is monotone and a $\gamma$-contraction. Repeated application under the improved policy yields

$$
\boxed{Q^{\pi_{\mathrm{new}}}_{\mathrm{soft}}(s,a)
\ge Q^{\pi_{\mathrm{old}}}_{\mathrm{soft}}(s,a).}
$$

This statement concerns **exact policy evaluation and exact improvement**. It does not establish global monotonic improvement or convergence for practical SAC with finite replay data, neural approximators, and stochastic optimization.

<a id="course-section-40"></a>

## 9. From soft policy iteration to modern SAC

<a id="course-section-41"></a>

### 9.1 Two conceptual operations, represented by neural networks

We now have a complete policy-iteration cycle:

1. **Soft policy evaluation:** learn the current policy's soft Q-function through the soft Bellman equation.
2. **Soft policy improvement:** update the actor toward higher-Q, entropy-regularized actions.

With neural function approximation,

$$
\boxed{\text{Actor: }\pi_\theta(a\mid s),\qquad
\text{Critics: }Q_{\phi_1}(s,a),\;Q_{\phi_2}(s,a).}
$$

The critic maps state-action pairs to scalar value estimates. The actor maps states to conditional action distributions. Rewards train the critics, and Q-value gradients then inform actor improvement. In model-free SAC, one does not normally differentiate the environmental reward through the physical simulator directly into the actor.

<a id="course-section-42"></a>

### 9.2 Why SAC can reuse data collected by older policies

For a stationary MDP, the conditional transition law $P(s'\mid s,a)$ does not depend on which behavior policy $\beta$ happened to generate the pair $(s,a)$. A stored transition $(s,a,r,s',d)$ therefore remains a valid sample of environmental dynamics even when the actor has changed. However, the **next action in the target** is sampled from the **current actor**:

$$
y=r+\gamma(1-d)
\left[Q(s',a')-\alpha\log\pi_\theta(a'\mid s')\right],
\qquad a'\sim\pi_\theta(\cdot\mid s').
$$

| Quantity | Source | Recomputed under the current actor? |
|:--|:--|:--|
| $(s,a,r,s',d)$ | Historical replay transition | No |
| $a'$ | Sample $\pi_\theta(\cdot\mid s')$ | Yes |
| $\log\pi_\theta(a'\mid s')$ | Current actor density | Yes |
| $Q(s',a')$ | Target critic | Yes |

Replay improves sample reuse, but not every historical distribution yields a reliable Bellman update. **Insufficient state-action coverage, critic extrapolation, function approximation, and bootstrap error** remain serious issues. Actor updates often use states from the replay distribution rather than the current policy's exact discounted occupancy measure, which is a further approximation.

<a id="course-section-43"></a>

### 9.3 How this differs from on-policy PPO

PPO uses actions sampled from an old policy and the likelihood ratio $\rho_t(\theta)=\pi_\theta(a_t\mid s_t)/\pi_{\mathrm{old}}(a_t\mid s_t)$. Standard PPO typically performs a limited number of optimization epochs on recent rollouts; it does not treat arbitrary ancient rollout actions as if they were sampled from the current actor. SAC instead uses off-policy Bellman evaluation and actor improvement and does not require PPO's probability-ratio clipping. This is a distinction between standard algorithmic settings, not a claim that no PPO-related method could ever use older experience.

<a id="course-section-44"></a>

## 10. The complete SAC critic: target networks, twin Q, and TD regression

<a id="course-section-45"></a>

### 10.1 From the soft Bellman equation to a sampled target

Maintain two critics $Q_{\phi_i}$ and two slowly moving target critics $Q_{\bar\phi_i}$. For $(s,a,r,s',d)$ from replay, sample $a'\sim\pi_\theta(\cdot\mid s')$ and define

$$
\boxed{y=r+\gamma(1-d)\left[
\min_{j=1,2}Q_{\bar\phi_j}(s',a')
-\alpha\log\pi_\theta(a'\mid s')
\right].}
$$

Each critic minimizes a regression loss:

$$
\boxed{J_{Q_i}(\phi_i)=\frac12\mathbb E_{\mathcal D}
\left[(Q_{\phi_i}(s,a)-\operatorname{stopgrad}(y))^2\right],\quad i=1,2.}
$$

The target is a stochastic one-sample approximation of the Bellman expectation. Gradients must not flow through the target branch; otherwise the operation ceases to match this standard semi-gradient TD regression construction.

<a id="course-section-46"></a>

### 10.2 Why use the minimum of two critics?

An actor tends to seek actions with high estimated Q-values. If high Q-values arise from approximation errors, these errors can be exploited by the actor and propagated by bootstrapping. SAC uses $\min_jQ_{\bar\phi_j}(s',a')$ to introduce a conservative estimate for a *given* next action. This reduces certain kinds of positive bias, but **does not make estimates unbiased**. Correlated errors can persist, and taking the minimum may introduce negative bias.

Crucially, classical Q-learning's $\max_{a'}$ is an **action-space optimality backup**. SAC's $\min_j$ is a **reduction over critic estimators**. They operate on different indices and solve different problems.

<a id="course-section-47"></a>

### 10.3 True termination versus time-limit truncation

If $d=1$ represents genuine termination, the target reduces to $y=r$. If an environment truncates a continuing episode only because a step limit was reached (`truncated=True`), bootstrapping should usually be retained. Conflating `terminated` and `truncated` can systematically undervalue states near artificial episode boundaries. For truly finite-horizon problems, one should define the terminal condition and include remaining time in the Markov state when necessary.

<a id="course-section-48"></a>

### 10.4 Slowly updating the target critics

A typical Polyak update is

$$
\boxed{\bar\phi_j\leftarrow(1-\tau)\bar\phi_j+\tau\phi_j,\qquad 0<\tau\ll1.}
$$

A common illustrative setting is $\tau=0.005$, but no single value is universally optimal. A slow-moving target limits short-term target drift; it cannot compensate for flawed reward scaling, inadequate coverage, or unstable action representations.

<a id="course-section-49"></a>

## 11. The complete SAC actor: reparameterization, bounds, and log probabilities

<a id="course-section-50"></a>

### 11.1 From KL projection to the actor objective

Using replay states for approximate policy improvement, a common actor objective is

$$
\boxed{J_\pi(\theta)=
\mathbb E_{\substack{s\sim\mathcal D\\a\sim\pi_\theta(\cdot\mid s)}}
\left[\alpha\log\pi_\theta(a\mid s)
-\min_{j=1,2}Q_{\phi_j}(s,a)\right].}
$$

The first term favors appropriate entropy, and the second favors high-value actions. During actor optimization one usually freezes the critic **parameters**, but preserves $\nabla_aQ_\phi(s,a)$ to transmit an improvement direction to the actor. Some implementations update the actor against just $Q_{\phi_1}$ rather than $\min_jQ_{\phi_j}$; this is an implementation convention to verify, not a mathematical identity.

<a id="course-section-51"></a>

### 11.2 The reparameterization trick

Writing only $a\sim\pi_\theta(a\mid s)$ obscures how derivatives flow through random sampling. Let $\epsilon\sim\mathcal N(0,I)$ be independent of the actor parameters, and let the actor output $\mu_\theta(s)$ and $\sigma_\theta(s)$:

$$
\boxed{u_\theta(s,\epsilon)=\mu_\theta(s)+\sigma_\theta(s)\odot\epsilon,}
$$

$$
\boxed{a_\theta(s,\epsilon)=\tanh u_\theta(s,\epsilon).}
$$

Then $a_\theta=f_\theta(s,\epsilon)$ is differentiable with respect to $\theta$, while the base noise distribution is fixed. The objective becomes

$$
J_\pi(\theta)=\mathbb E_{s\sim\mathcal D,\epsilon\sim\mathcal N(0,I)}
\left[\alpha\log\pi_\theta(f_\theta(s,\epsilon)\mid s)
-Q_{\phi}(s,f_\theta(s,\epsilon))\right].
$$

Here $Q_\phi$ stands for whichever critic reduction is used in the actor update (one critic or the minimum of two). The derivative includes

$$
-\underbrace{\nabla_aQ_\phi(s,a_\theta)}_{\text{action-value gradient}}
\underbrace{\frac{\partial a_\theta}{\partial\theta}}_{\text{actor action Jacobian}},
$$

plus the **total derivative** of $\alpha\log\pi_\theta(a_\theta\mid s)$: both explicit parameter dependence and implicit dependence through the sampled action matter. This pathwise gradient differs from the score-function gradient commonly used in PPO. It is not simply a distinction between deterministic and stochastic policies.

<a id="course-section-52"></a>

### 11.3 Correct log probability for a tanh-squashed Gaussian

For $u\sim\mathcal N(\mu_\theta(s),\operatorname{diag}\sigma_\theta^2(s))$ and $a=\tanh u$, the change-of-variables formula gives

$$
\pi_\theta(a\mid s)
=p_U(u\mid s)\left|\det\frac{\partial a}{\partial u}\right|^{-1}.
$$

Because $\partial a_i/\partial u_i=1-\tanh^2u_i$,

$$
\boxed{\log\pi_\theta(a\mid s)=
\log\mathcal N(u;\mu_\theta(s),\Sigma_\theta(s))
-\sum_{i=1}^{d}\log(1-\tanh^2u_i).}
$$

Substituting $\log\mathcal N(a;\mu,\Sigma)$ is incorrect. For numerical stability near $\tanh u=\pm1$, use

$$
\log(1-\tanh^2u)=2\big(\log 2-u-\operatorname{softplus}(-2u)\big).
$$

<a id="course-section-53"></a>

### 11.4 Normalized versus physical robot actions

Suppose the actual executed action dimension is $a_i^{\mathrm{env}}=c_i+b_i\tanh u_i$, where

$$
c_i=\frac{a_{\max,i}+a_{\min,i}}{2},\qquad
b_i=\frac{a_{\max,i}-a_{\min,i}}{2}>0.
$$

The log density **with respect to physical action coordinates** includes the additional $-\sum_i\log b_i$ term:

$$
\log\pi_\theta(a^{\mathrm{env}}\mid s)
=\log\mathcal N(u;\mu,\Sigma)
-\sum_i\log\big[b_i(1-\tanh^2u_i)\big].
$$

An alternative, common convention defines all probabilities, entropy targets, and critic action inputs in normalized $[-1,1]^d$ coordinates and then applies a fixed environment-level action scaling. Both conventions are valid if applied consistently. The constant log-Jacobian shift affects numerical differential entropy and the interpretation of automatic temperature tuning.

<a id="course-section-54"></a>

## 12. Automatic entropy tuning: why learn the temperature alpha?

<a id="course-section-55"></a>

### 12.1 From fixed temperature to a target-entropy constraint

The effect of a fixed $\alpha$ varies with reward scaling and action dimension. Modern SAC can instead impose an average entropy requirement:

$$
\max_\pi J_R(\pi)
\quad\text{s.t.}\quad
\bar{\mathcal H}(\pi)\ge\mathcal H_{\mathrm{target}}.
$$

The corresponding Lagrangian, in this simplified average-constraint representation, is

$$
\mathcal L(\pi,\alpha)
=J_R(\pi)+\alpha(\bar{\mathcal H}(\pi)-\mathcal H_{\mathrm{target}}),
\qquad\alpha\ge0.
$$

The primal-dual structure is $\max_\pi\min_{\alpha\ge0}\mathcal L(\pi,\alpha)$. Original SAC derivations can use per-time-step entropy constraints; the average-entropy form here conveys the design motivation, but should not be assumed fully equivalent to all other constraint formulations.

<a id="course-section-56"></a>

### 12.2 The temperature objective and a sign check

With the actor held fixed, minimize

$$
\boxed{J_\alpha(\alpha)
=\mathbb E_{s\sim\mathcal D,a\sim\pi_\theta}
\left[\alpha\big(-\log\pi_\theta(a\mid s)-\mathcal H_{\mathrm{target}}\big)\right].}
$$

The derivative is $\partial J_\alpha/\partial\alpha=\bar{\mathcal H}(\pi)-\mathcal H_{\mathrm{target}}$. If measured entropy is below target, this derivative is negative and gradient descent increases $\alpha$; if entropy exceeds target, descent reduces $\alpha$. **Negative numerical targets are perfectly possible for continuous differential entropy**; the intuition from nonnegative discrete entropy does not carry over unchanged.

<a id="course-section-57"></a>

### 12.3 Why parameterize alpha as exp(eta)?

Use $\eta=\log\alpha$ and $\alpha=e^\eta>0$ to preserve positivity. Strict substitution yields

$$
J_\eta(\eta)
=-\mathbb E\left[e^\eta
(\log\pi_\theta(a\mid s)+\mathcal H_{\mathrm{target}})\right].
$$

Many implementations instead use a surrogate with a different gradient scale but the same interior stationary condition:

$$
\boxed{J_\eta^{\mathrm{sur}}(\eta)
=-\mathbb E\left[\eta
(\log\pi_\theta(a\mid s)+\mathcal H_{\mathrm{target}})\right].}
$$

The log probability must be detached during the temperature update to avoid accidentally optimizing the actor through the temperature objective. The common heuristic $\mathcal H_{\mathrm{target}}=-d$ for $d$ normalized action dimensions is **not a theorem**. Correlated joints, high-dimensional hands, and constrained actions require care when interpreting it.

<a id="course-section-58"></a>

## 13. Putting together the complete modern SAC algorithm

<a id="course-section-59"></a>

### 13.1 Components and three losses

| Component | Symbol | Role |
|:--|:--|:--|
| Stochastic actor | $\pi_\theta(a\mid s)$ | Samples valuable continuous actions with nonzero entropy |
| Twin critics | $Q_{\phi_1},Q_{\phi_2}$ | Estimate soft Q and reduce overoptimism |
| Target critics | $Q_{\bar\phi_1},Q_{\bar\phi_2}$ | Provide slowly changing TD targets |
| Replay buffer | $\mathcal D$ | Store and reuse old transitions off-policy |
| Temperature | $\alpha=e^\eta$ | Adjust entropy weight to a target |

The optimization problems are the two critic regressions $J_{Q_1}+J_{Q_2}$, the actor loss $J_\pi$, and the temperature loss $J_\eta$. Their interaction is best understood as **evaluate the actor $\rightarrow$ improve the actor $\rightarrow$ tune entropy $\rightarrow$ slowly track target values**.

<a id="course-section-60"></a>

### 13.2 The full sequence of one SAC update

1. Interact with the environment using the actor (possibly with random warm-up exploration) and save $(s,a,r,s',\mathrm{terminated},\mathrm{truncated})$.
2. Sample a replay minibatch. At each $s'$, resample $a'$ from the current actor; use target critics and $\alpha$ to construct the soft Bellman target $y$ with target-branch gradients disabled.
3. Regress each $Q_{\phi_i}(s,a)$ onto the same target $y$.
4. At replay states $s$, resample $\tilde a$ from the current actor. Update it using $\alpha\log\pi_\theta(\tilde a\mid s)-\min_iQ_{\phi_i}(s,\tilde a)$, keeping critic parameters fixed but the gradient with respect to the input action enabled.
5. Update $\log\alpha$ using detached policy log probabilities to adjust the entropy pressure.
6. Update target critics by Polyak averaging. Continue data collection and optimization, potentially using more than one gradient step per environment step.

The following is **mathematical pseudocode**, not executable Python and not an experimental protocol:

```text
Initialize actor pi_theta; critics Q_phi1,Q_phi2
Initialize target critics Q_barphi1,Q_barphi2; replay D; eta=log(alpha)
Repeat:
    Collect environment transition and append to D
    For each gradient update:
        Sample (s,a,r,s_next,terminated,truncated) from D
        a_next ~ pi_theta(. | s_next)
        y = r + gamma*(1-terminated)*(
                min_i Q_barphi_i(s_next,a_next) - alpha*log pi_theta(a_next|s_next))
        Update each Q_phi_i by gradient of (Q_phi_i(s,a)-stopgrad(y))^2
        a_pi ~ pi_theta(. | s)  # reparameterized sample
        Update actor by gradient of alpha*log pi_theta(a_pi|s)-min_i Q_phi_i(s,a_pi)
        Update eta by entropy dual loss (detach log probability)
        barphi_i <- (1-tau)*barphi_i + tau*phi_i
```

Respecting `terminated` versus `truncated`, normalized action conventions, the tanh density Jacobian, frozen critic weights during actor updates, and replay coverage are essential implementation details rather than optional embellishments.

<a id="course-section-61"></a>

### 13.3 Original SAC v1 versus modern SAC: is V explicitly learned?

The original SAC version additionally fit a state-value network $V_\psi(s)$:

$$
J_V(\psi)=\frac12\mathbb E_s\left[
\left(V_\psi(s)-\operatorname{stopgrad}
(\min_iQ_{\phi_i}(s,\tilde a)-\alpha\log\pi_\theta(\tilde a\mid s))\right)^2\right],
$$

with Q-target $y_Q=r+\gamma(1-d)V_{\bar\psi}(s')$. The modern widely used variant removes the explicit V-network and constructs Q-targets directly from the target Q-networks and the current actor's next-action log probability. Thus implementations often contain an actor, two critics, two target critics, and a learned temperature. **Removing the V-network does not remove the soft V concept from the mathematics.**

<a id="course-section-62"></a>

## 14. How the main algorithms relate

| Method | Typical value backup | Actor / action selection | Data regime | Main design principle |
|:--|:--|:--|:--|:--|
| SARSA | $Q(s',a')$ for next behavior action | Behavior policy | On-policy (standard) | Evaluate the policy being executed |
| Q-learning | $\max_{a'}Q(s',a')$ | Greedy + exploration | Off-policy | Bellman optimality |
| DQN | Neural $\max_{a'}Q_{\bar\phi}$ | Discrete argmax | Off-policy | Replay and target networks |
| Double DQN | Online selection, target evaluation | Discrete argmax | Off-policy | Reduce max-induced overestimation |
| DDPG | Target Q at deterministic actor action | Deterministic actor + exploration noise | Off-policy | Continuous-action policy gradient |
| TD3 | Twin Q with conservative target | Delayed deterministic actor | Off-policy | Reduce approximation error |
| SAC | Soft Bellman with twin critics | Stochastic actor + entropy | Off-policy | Soft policy iteration |
| PPO | Typically V and advantage estimates | Stochastic policy likelihood-ratio updates | On-policy (standard) | Clipped policy objective |

<a id="course-section-63"></a>

### 14.1 The most important Q-learning versus SAC distinction

Classical Q-learning attempts to directly learn optimal Q and uses $\max_{a'}Q(s',a')$ in its target. SAC's critic estimates the **current stochastic actor's soft Q**, while the actor performs entropy-regularized improvement:

$$
\boxed{\begin{aligned}
\text{Q-learning: }&y=r+\gamma\max_{a'}Q(s',a'),\\
\text{SAC: }&y=r+\gamma
\left[\min_jQ_{\bar\phi_j}(s',a')-\alpha\log\pi(a'\mid s')\right],\quad a'\sim\pi.
\end{aligned}}
$$

We omit the true-termination factor for readability; practical implementations multiply the bootstrap term by $1-d$. In SAC, $\min_j$ does not minimize over actions. It chooses the lower of several estimates for a given action.

<a id="course-section-64"></a>

### 14.2 PPO versus SAC: score-function and pathwise gradients

PPO typically uses $\nabla_\theta\log\pi_\theta(a\mid s)\,\hat A(s,a)$ along with clipped likelihood ratios. SAC reparameterizes continuous actions so that $\nabla_aQ(s,a)\,\partial a/\partial\theta$ reaches the actor, together with an entropy gradient. Standard PPO does not require its critic to be differentiable with respect to actions; standard continuous SAC generally does. PPO can be fast in wall-clock time when simulators produce massively parallel fresh rollouts. SAC can be more sample-efficient when physical interactions are expensive. **Sample efficiency is not the same as wall-clock efficiency.**

<a id="course-section-65"></a>

## 15. Robot-control limits and common misconceptions

<a id="course-section-66"></a>

### 15.1 Why critic extrapolation is especially important in manipulation

Robot arms involve mode-switching contacts, nonlinear dynamics, sparse rewards, estimation errors, and action saturation. If replay lacks some contact configurations or recovery motions, the critic may assign unrealistically high values to out-of-coverage actions. An actor that follows Q-gradients may then move toward precisely those spurious values, creating an **actor-critic feedback loop**. Twin critics mitigate part of this risk; they do not replace coverage analysis, safety constraints, reward checks, or monitoring of distribution shift.

<a id="course-section-67"></a>

### 15.2 Joint positions, velocities, torques, and action normalization

A policy action in $[-1,1]^d$ is often not itself the joint position, velocity, or torque sent to the robot. The control interface must apply a clear scale, offset, and physical limits. Saturation can reduce the derivative of tanh; scaling affects density and the entropy target. A correct MDP description should also account for control frequency, sensing latency, actuator bounds, and the chosen action representation. These details often matter more for trustworthy robot behavior than the elegance of the actor loss alone.

<a id="course-section-68"></a>

### 15.3 When standard SAC should not be copied verbatim

Standard SAC assumes actions can be sampled from a policy with an evaluable log density and usually exploits a differentiable path from actor parameters to actions. Discrete actions, implicit generative policies, flow-matching action heads, partial observability with long history, and strict safety constraints can require different objectives or parameterizations. It is invalid to attach $\alpha\log\pi$ to a policy whose density is unavailable and automatically call the result standard SAC. Likewise, replayed data do not by themselves resolve out-of-distribution action values in offline RL; **offline RL and online off-policy RL are different settings**.

<a id="course-section-69"></a>

### 15.4 Five misconceptions to avoid

1. **"Q-learning is off-policy only when a replay buffer is used."** No: the distinction comes from behavior versus target policies.
2. **"Soft Q-learning is just Q-learning's max replaced by an actor sample."** Incomplete: distinguish soft optimality from soft evaluation of a fixed policy.
3. **"Double Q-learning, Double DQN, and SAC min-Q are the same estimator."** No: their selection/evaluation and estimator-combination rules differ.
4. **"Larger temperature always yields higher task return."** No: entropy trades off with task reward and is sensitive to action and reward scales.
5. **"Off-policy reuse guarantees stable, safe learning."** No: support mismatch, bootstrapping error, nonstationarity, and execution constraints still matter.

<a id="course-section-70"></a>

## 16. The complete derivation in six equations

The following six formulas summarize the bridge from classical Q-learning to modern SAC.

**(1) Ordinary Bellman optimality:**

$$
\boxed{Q^*(s,a)=r(s,a)+\gamma\mathbb E_{s'}[\max_{a'}Q^*(s',a')].}
$$

**(2) Q-learning's sampled TD update:**

$$
\boxed{Q(s,a)\leftarrow Q(s,a)+\eta\left[r+\gamma(1-d)\max_{a'}Q(s',a')-Q(s,a)\right].}
$$

**(3) Optimal soft value for finite discrete actions:**

$$
\boxed{V^*_{\mathrm{soft}}(s)=\alpha\log\sum_a\exp(Q^*_{\mathrm{soft}}(s,a)/\alpha),}
$$

whose zero-temperature limit recovers $\max_aQ(s,a)$.

**(4) Soft Bellman evaluation of a given policy:**

$$
\boxed{Q^\pi_{\mathrm{soft}}(s,a)=r+\gamma\mathbb E_{s',a'\sim P,\pi}
[Q^\pi_{\mathrm{soft}}(s',a')-\alpha\log\pi(a'\mid s')].}
$$

**(5) SAC's twin critics and reparameterized actor:**

$$
\boxed{\begin{aligned}
y&=r+\gamma(1-d)\left[\min_jQ_{\bar\phi_j}(s',a')
-\alpha\log\pi_\theta(a'\mid s')\right],\\
J_\pi&=\mathbb E_{s,\epsilon}\left[\alpha\log\pi_\theta(f_\theta(s,\epsilon)\mid s)
-\min_jQ_{\phi_j}(s,f_\theta(s,\epsilon))\right].
\end{aligned}}
$$

**(6) Entropy target and temperature adaptation:**

$$
\boxed{J_\eta^{\mathrm{sur}}=-\mathbb E\left[\eta
(\log\pi_\theta(a\mid s)+\mathcal H_{\mathrm{target}})\right],\quad\eta=\log\alpha.}
$$

**In one sentence:** Q-learning learns optimal action values using **hard Bellman optimality and TD bootstrapping**, whereas SAC combines **soft Bellman policy evaluation, actor policy improvement, and entropy adaptation** to learn a stochastic continuous-control policy. They share Bellman reasoning and TD learning; they differ in how future actions are optimized and how policy entropy is treated.

<a id="course-section-71"></a>

## References and further reading

1. Watkins, C. J. C. H., & Dayan, P. (1992). **Q-learning.** *Machine Learning*, 8, 279–292. [DOI](https://doi.org/10.1007/BF00992698). Classical tabular Q-learning and convergence analysis.
2. Sutton, R. S., & Barto, A. G. (2018). **Reinforcement Learning: An Introduction**, 2nd ed. [Open textbook](http://incompleteideas.net/book/the-book-2nd.html). Bellman equations, TD, SARSA, Q-learning, and policy iteration.
3. Mnih, V., et al. (2015). **Human-level control through deep reinforcement learning.** *Nature*. [DOI](https://doi.org/10.1038/nature14236). DQN, replay, and target networks.
4. van Hasselt, H. (2010). **Double Q-learning.** *NeurIPS*. [Paper](https://papers.neurips.cc/paper_files/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html). Estimation bias and two-estimator learning.
5. van Hasselt, H., Guez, A., & Silver, D. (2016). **Deep Reinforcement Learning with Double Q-learning.** [arXiv:1509.06461](https://arxiv.org/abs/1509.06461). Double DQN.
6. Lillicrap, T. P., et al. (2015). **Continuous control with deep reinforcement learning.** [arXiv:1509.02971](https://arxiv.org/abs/1509.02971). Deterministic continuous actor-critic.
7. Fujimoto, S., van Hoof, H., & Meger, D. (2018). **Addressing Function Approximation Error in Actor-Critic Methods.** [arXiv:1802.09477](https://arxiv.org/abs/1802.09477). TD3 and twin critics.
8. Haarnoja, T., Tang, H., Abbeel, P., & Levine, S. (2017). **Reinforcement Learning with Deep Energy-Based Policies.** *ICML*. [Paper](https://proceedings.mlr.press/v70/haarnoja17a.html). Soft Q-learning and energy-based policies.
9. Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). **Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor.** [arXiv:1801.01290](https://arxiv.org/abs/1801.01290). Soft policy iteration and KL improvement.
10. Haarnoja, T., et al. (2018). **Soft Actor-Critic Algorithms and Applications.** [arXiv:1812.05905](https://arxiv.org/abs/1812.05905). Learned temperature and modern SAC.
11. OpenAI Spinning Up. **Soft Actor-Critic.** [Algorithm guide](https://spinningup.openai.com/en/latest/algorithms/sac.html). Practical loss definitions and implementation conventions.

---

*License note: original course content is credited under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); third-party cited works retain their own licenses. This teaching document includes no experiments, benchmark claims, or guarantees about training outcomes. Equations use standard LaTeX math compatible with GitHub Markdown and Pandoc. Its accompanying PDF is generated from this exact Markdown revision.*
