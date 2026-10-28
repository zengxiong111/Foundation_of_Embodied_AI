---
title: "Dexterous Manipulation with Learned World Models: Wuji Hand2 In-Hand Reorientation"
subtitle: "From a PPO baseline to SuperMimic-style BPTT and Real2Sim2Real"
date: "2026-10-28"
lang: en-US
---

# Dexterous Manipulation with Learned World Models: Wuji Hand2 In-Hand Reorientation

[Course home](https://github.com/zengxiong111/Foundation_of_Embodied_AI) · [简体中文](Dexterous_Manipulation_Learned_World_Models_ZH.md) · [SuperMimic](https://arxiv.org/abs/2610.28480) · [Community Dex-Supermimic](https://github.com/CumulusAlpha/Dex-Supermimic)

| Item | Details |
|---|---|
| Topic | Dexterous World Models |
| Revision | 2026-10-28 |
| License | CC BY-NC-SA 4.0 · Xiong Zeng and contributors |


> **Scope:** An extension of the new *Model-Based RL* tutorial to contact-rich dexterous control. This is a **research design and teaching case**, not an experimentally validated physical-hand deployment.

**Target task:** A Wuji Hand2 reorients a grasped cube toward an arbitrary goal orientation. Establish PPO and learned-world-model+BPTT baselines in mjlab/MuJoCo Warp before discussing transfer to real hardware with known object geometry, RGB-D object pose estimates and measured actuator dynamics.

---

## Contents

- [1. Transferring the idea, not the original results](#supplement-course-section-01)
- [2. Task, state, observation and success](#supplement-course-section-02)
- [3. Simulation and actuator contract](#supplement-course-section-03)
- [4. Building a contact-aware world model](#supplement-course-section-04)
- [5. Comparing three control families](#supplement-course-section-05)
- [6. Training loop, curriculum and recovery](#supplement-course-section-06)
- [7. Controlled experiment matrix](#supplement-course-section-07)
- [8. Real2Sim2Real deployment boundaries](#supplement-course-section-08)
- [9. Self-check and references](#supplement-course-section-09)

<a id="supplement-course-section-01"></a>

## 1. Transferring the idea, not the original results

Original SuperMimic learns a shared humanoid dynamics surrogate from rigid-body simulator rollouts and optimizes motion tracking policies using short-horizon BPTT. **Its reusable contribution here is the training paradigm.** In-hand manipulation involves a moving rigid object, multiple finger contact points, frictional slip and object drops, all of which create more challenging prediction and gradient errors.

- **Transferable ideas:** Dyna-like alternating simulator sampling, replay buffers, autoregressive model fitting, truncated pathwise policy gradients and terminal value bootstrapping.
- **Must be redefined:** object pose representation, contact events, drop/termination criteria, tracking rewards, actuator response, randomized physics and domain context.
- **Must be newly validated:** simulator success, generalization to unseen goals and objects, sensor feedback, deployment safety and physical-sample efficiency.

The independent [Dex-Supermimic project](https://github.com/CumulusAlpha/Dex-Supermimic) includes Wuji Hand 1/2 cube reorientation, Wuji Hand2 pen spinning and a PPO baseline for Hand2 cube reorientation. Its README describes **mjlab/MuJoCo Warp simulated tasks**. These artifacts must not be presented as evidence of physical hand deployment or as the original paper's official code.

<a id="supplement-course-section-02"></a>

## 2. Task, state, observation and success

### 2.1 MDP contract

Let $n$ be the number of controlled hand joints. An object pose $(p_t^o,R_t^o)$ lies in $\mathbb R^3\times SO(3)$ with linear/angular velocity $(v_t^o,\omega_t^o)$:

$$
 s_t=\left(q_t,\dot q_t,p_t^o,R_t^o,v_t^o,\omega_t^o,c_t\right),
 \qquad g_t=R_t^*.
$$

The optional $c_t$ describes contact modes/features that may be inaccessible on real hardware. A deployable policy could observe

$$
 o_t=\left(q_t,\dot q_t,\hat p_t^o,\hat R_t^o,g_t,\{a_{t-j}\}_{j=1}^m\right).
$$

A privileged critic may access a richer simulator state. The actor must use only quantities that actual hardware can sense or reliably estimate. Sending perfect simulator contact forces to the training actor without matching hardware sensors creates a training/deployment mismatch.

### 2.2 Rotation error on SO(3)

If the task does not require special handling for cube symmetries, one may define

$$
 e_R(t)=\left\|\log\left((R_t^*)^\top R_t^o\right)^\vee\right\|_2\in[0,\pi].
$$

Here $\log:SO(3)\to\mathfrak{so}(3)$ and $\vee$ maps a skew matrix into a rotation vector. For cubes with visually or physically equivalent target orientations, specify the relevant symmetry group and minimize only over equivalences allowed by the task definition.

### 2.3 A reproducible success definition

For example, declare success when orientation error remains below $\epsilon_R$ for $M$ consecutive control steps, the object stays grasped and all velocity/torque limits are respected. Set $\epsilon_R$, $M$, timeout and goal distribution **before** inspecting the test results.

<a id="supplement-course-section-03"></a>

## 3. Simulation and actuator contract

### 3.1 Joint commands, PD and control rate

Real hands frequently accept position setpoints or internal motor commands, rather than ideal unlimited torque. A simple position-servo model is

$$
 \tau_t=\operatorname{sat}_{\tau_{\max}}\!\left[K_p(a_t-q_t)-K_d\dot q_t\right].
$$

Check whether $K_p,K_d$ vary per joint. Also verify reflected rotor inertia (armature), torque/velocity limits, transmission ratio, friction, end stops, command filters, controller delay and update frequency. **Do not infer joint-level armature from only the URDF link inertia or transfer numerical values between different hands.**

### 3.2 Minimal asset validation

1. Isolated joint step response, stability and torque saturation without a manipulated object;
2. Fingerpad/palm collision geometry and object mass, inertia, friction and contact solver settings;
3. Small-action tests for unrealistic penetration, bouncing or spontaneously moving objects;
4. Physics step, control interval and learned-model prediction interval;
5. Consistent quaternion ordering, axis conventions, frames and target rotation definitions.

<a id="supplement-course-section-04"></a>

## 4. Building a contact-aware world model

### 4.1 Dynamics, reward and termination

Learn

$$
 \hat s_{t+1}=f_\omega(\hat s_t,a_t,\xi),
 \quad \hat d_{t+1}=d_\omega(\hat s_t,a_t,g_t),
 \quad \hat r_{t+1}=r(\hat s_t,a_t,g_t,\hat s_{t+1}),
$$

where $\xi$ captures object inertia, mass, friction and finger actuator parameters. Rewards may be analytic when the state provides their arguments; an auxiliary reward head is another option. Prefer stable rotation increments or exponential maps for orientation state updates, with validity checks.

### 4.2 Autoregressive training and contact stratification

A structured dynamics loss can be

$$
 \mathcal L_{dyn}
 =\sum_{j=1}^{K}\left[
 w_q\|\hat q_{t+j}-q_{t+j}\|^2+
 w_v\|\hat{\dot q}_{t+j}-\dot q_{t+j}\|^2+
 w_p\|\hat p^o_{t+j}-p^o_{t+j}\|^2+
 w_R d_{SO(3)}(\hat R^o_{t+j},R^o_{t+j})^2
 \right].
$$

Contact mode, slip and drop predictions can be added when labels are available. Stratify multi-step errors by **free motion, stable grip, onset of slip, renewed contact and loss of grasp**. Otherwise the many easy no-contact transitions may dominate the loss and mask critical errors.

### 4.3 Model exploitation

The actor could find actions the inaccurate model incorrectly permits: apparent frictionless turns, suspended objects or missing collision consequences. Decreasing world-model prediction MSE is not equivalent to improving true simulator return. **Every actor improvement cycle must be evaluated in the original MuJoCo Warp environment.**

<a id="supplement-course-section-05"></a>

## 5. Comparing three control families

| Method | What is optimized? | Strength | Limitation |
|---|---|---|---|
| PPO | Policy parameters using simulator feedback | Mature, robust to nonsmooth contact | High simulator interaction cost |
| CEM / MPPI | A planned action sequence per decision step | No model Jacobian required, can search multimodal plans | Many model calls each step; latency must be measured |
| SuperMimic-style BPTT | Policy parameters through learned dynamics | Fast model imagination, lightweight policy execution | Biased surrogate gradients and model exploitation |

**Fair comparison:** use the same hand, cube, goal distribution, actuator interface, reward and termination. Report online planning costs for CEM/MPPI and training GPU-hours/simulator interactions for PPO/BPTT. Never compare MPC test-time compute directly with BPTT training-time compute without a total-cost accounting.

### 5.1 Illustrative reward, not published hyperparameters

$$
 r_t=w_R\exp(-k_Re_R(t))
 +w_h\mathbf1_{\text{object held}}
 -w_a\|a_t-a_{t-1}\|^2
 -w_\tau\|\tau_t\|^2
 -w_d\mathbf1_{\text{drop}}.
$$

Reward checks should verify that orientation progress occurs **while grasp stability is maintained**. All coefficients are symbolic placeholders, **not parameters attributed to SuperMimic or the community implementation**.

### 5.2 Short-horizon BPTT objective

For imagined horizon $H$, optimize

$$
 \hat J_H(\theta)=\mathbb E\left[
 \sum_{t=0}^{H-1}\gamma^t\hat c_t\hat r_{t+1}
 +\gamma^H\hat c_H V_\psi(\hat s_H,g_H,\xi)
 \right].
$$

World-model parameters $\omega$ are frozen during the actor phase, **but gradients must pass through model inputs/outputs**. The best $H$ depends on contact complexity and approximation quality; it should be selected through ablations, not automatically fixed at the paper's 32 steps.

<a id="supplement-course-section-06"></a>

## 6. Training loop, curriculum and recovery

### 6.1 Suggested loop

```text
Validate robot + object + actuator in MuJoCo Warp
                |
                v
Train matched PPO baseline and fix evaluation seeds
                |
                v
Gather simulator transitions (including failures)
                |
                v
Fit autoregressive world model; evaluate contact errors
                |
                v
Imagine short trajectories, optimize actor + critic via BPTT
                |
                v
Return to physical simulator; measure realized policy gain
                |
                +--------------> refresh replay and repeat
```

### 6.2 A staged curriculum (hypothesis, not an observed result)

Increase difficulty from **stable grasp → small-angle turns → large-angle turns → randomized goal orientations → randomized friction/payload → recovery from intermittent contact loss**. Stratify starting states around stable contact, contact boundaries and failures. Reference-state initialization must enforce physically feasible contact and object states; a smoothly interpolated joint trace does not guarantee a feasible in-hand grasp.

### 6.3 Recovery requires more than an extra reward term

Test weakening grasps, sudden slip, recontact, pose-estimator jumps and actuator disturbances explicitly in simulation. Terminal value learning should preserve delayed incentives for recovery. Deployed hardware still needs action clamps, emergency stop, drop detection and a safe fallback posture.

<a id="supplement-course-section-07"></a>

## 7. Controlled experiment matrix

### 7.1 Main groups

| Group | Learned world model | Actor optimization | Critic | Question |
|---|---|---|---|---|
| A: PPO | No | PPO | PPO value | Matched baseline |
| B: BPTT-no-V | Yes | BPTT | No | Truncation-induced myopia? |
| C: BPTT+V | Yes | BPTT | Yes | Proposed principal route |
| D: BPTT+V+contact sampling | Yes | BPTT | Yes | Benefit of contact-aware replay |
| E: CEM/MPPI (optional) | Yes | Per-step trajectory planning | Optional | Test-time compute trade-off |

### 7.2 Key ablations and data efficiency

A reasonable sensitivity grid is $H\in\{8,16,32,64\}$ and dynamics fitting length $K\in\{1,4,8,16\}$; down-select based on compute budget. Hold total simulator interaction budgets, train GPU-hours and evaluation protocols as comparably as possible. Plot success against **real simulator transitions** and against **GPU-hours** separately. Imagined steps are not physical-simulator interactions.

### 7.3 Metrics

- **Task:** success rate, time to goal, SO(3) orientation error, drop rate, recovery after disturbance;
- **Physical:** slip count, contact-mode errors, torque/velocity violation, action acceleration/jitter;
- **Model:** 1/8/32-step prediction accuracy, contact-stratified error, agreement between predicted and realized improvement;
- **Systems:** GPU-hours, simulator FPS, imagination FPS, per-policy inference latency, simulator transition count.

<a id="supplement-course-section-08"></a>

## 8. Real2Sim2Real deployment boundaries

### 8.1 Identify actuator and object dynamics first

Measure each joint's response delay, effective PD gains, friction, armature, backdrivability, velocity limits and motor current/torque caps. On the object side identify mass, CoM, inertia, friction and collision geometry. Begin with safe, instrumented step-response and slip measurements, then choose justified domain randomization ranges rather than arbitrary values.

### 8.2 Real observations require closed-loop state estimation

With known object geometry and RGB-D images, a 6D pose tracker such as FoundationPose can estimate $\hat p_t^o,\hat R_t^o$; add temporal filtering, occlusion confidence, failure detection and reinitialization. During training, inject measured pose noise, latency and frame drops into the actor observation. **Simulator ground-truth object pose is not an available real-time camera measurement.**

### 8.3 Hardware safeguards are separate from learning

Clamp joint position, velocity, torque and command rate. Add an emergency-stop mechanism and a protected zone for dropped objects. A tracking failure, anomalous contact or finger overload should trigger a hold or safe fallback; the policy must not continue blindly. Report real-world success separately from any simulated demo.

<a id="supplement-course-section-09"></a>

## 9. Self-check and references

**Questions:**

1. Why is a geodesic orientation error on $SO(3)$ preferable to naive Euclidean quaternion error?
2. Does freezing world-model parameters force $\partial\hat s_{t+1}/\partial a_t=0$? Why not?
3. Which contact quantities are simulator privileges rather than deployable observations?
4. How could you detect exploitation of incorrect friction predictions?
5. Why must PPO and BPTT be compared in both interaction count and GPU-hours?

**Reading path:** original #6 Dynamics → #7 Isaac/MuJoCo → #8 Motion Control RL → #9 PPO → new *Model-Based RL* → this case study.

Primary references:

- [SuperMimic (original humanoid tracking method)](https://arxiv.org/abs/2610.28480)
- [SHAC (short-horizon actor–critic with differentiable simulation)](https://arxiv.org/abs/2204.07137)
- [Dex-Supermimic (independent community hand adaptation)](https://github.com/CumulusAlpha/Dex-Supermimic)
- [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp)
- [mjlab](https://github.com/mujocolab/mjlab)
- [FoundationPose](https://github.com/NVlabs/FoundationPose)

**Disclaimer:** This is a course/research proposal, not a measurement of a particular dexterous hand's hardware parameters, trained-policy performance, or physical deployment success. New original text follows [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Third-party papers, code, assets and pre-existing course chapters retain their licenses.
