# Advanced Robot Agents: Planning, Skills, Verification and Recovery

<!-- course-meta:start -->
[Course home](https://github.com/zengxiong111/Foundation_of_Embodied_AI) · [简体中文](https://github.com/zengxiong111/Foundation_of_Embodied_AI/blob/main/md/Robot_Agent_Workflow_ZH.md) · [PDF](https://zengxiong111.github.io/zengxiong.github.io/files/Robot_Agent_Workflow_EN.pdf)

| Item | Details |
|---|---|
| Applicability | Agentic planning, verification, recovery and public approaches; no robot validation |
| Documentation review date | 2026-10-18 |
| License | CC BY 4.0 · Xiong Zeng and contributors |

**Verification scope:** teaching and literature review, not experimental reproduction.
<!-- course-meta:end -->

<!-- course-toc:start -->
## Contents

- [1. Responsibilities of Agents and Policies](#course-section-01)
<!-- course-toc:end -->

## 1. Responsibilities of Agents and Policies

A Robot Agent handles goal decomposition, skill selection, state updates, verification and replanning. VLA/imitative policies generate conditional robot actions; a motion controller handles timely execution. Write the hierarchy as $g\to z_k\to\pi_{\rm skill}(a_t\mid o_t,z_k)$, where $g$ is a task and $z_k$ a parameterized skill. An agent must **not replace a high-rate servo loop**.

## 2. Closed-Loop Workflow

Starting from a language goal, an agent reads visual/proprioceptive state, proposes a reachable subgoal, calls a skill, then observes again and verifies success, recovery or stop conditions. Finishing a skill is not the same as satisfying the task. An occluded or unknown outcome cannot be marked successful.

### 2.1 Executable Skill API

Represent a skill as (name, preconditions, arguments, controller, timeout, success_predicate, fallback). A pick(object_id, grasp_pose) skill checks object identity, pose confidence and collision feasibility; on timeout it relocalizes or exits safely instead of retrying forever.

## 3. Representative Robot Agent Approaches

Gemini Robotics ER illustrates vision-language and spatial reasoning for task orchestration; RAX (Robot Agent Exploration) explores feedback-driven experience and skill reuse. Related ideas include SayCan for language-grounded skill affordances and Code as Policies for programmatic compositions. Capabilities depend on model version and robot integration: a paper demonstration does not imply calibration-free autonomous operation on arbitrary robots.

## 4. Planning, Verification and Belief State

For partial observability maintain a belief $b_t(s)=p(s_t\mid o_{\le t},a_{<t})$. A grasp verifier such as $V_{\rm grasp}=\mathbf1[\text{object lifted and retained}]$ should use visual and proprioceptive evidence over time. Low-confidence outcomes enter UNKNOWN and trigger more observation, rather than being labeled SUCCESS.

## 5. Architectures and Safety

Compare LLM/VLM plus fixed skills, VLM agents plus VLA, world-model candidate planning plus skills, and exploration agents plus experience memory. Predicted world-model imagery is not a ground-truth state measurement. Gate all high-level commands through workspace, collision, force/velocity, latency, confidence and emergency-stop constraints. Low-level safety cannot depend on remote LLM response.

## 6. Shared Panda Experiment and Ablation

Shared Panda cube-on-table task: pick from randomized locations, place at goal, then recover from an externally displaced object. Compare A fixed script, B agent plus skills, C visual verification, D replanning, and E reusable memory while fixing the controller and evaluation distribution. Report success $\hat p=\frac1N\sum_i\mathbf1[\mathrm{goal\ achieved}]$, latency, collisions, interventions, retries, recovery and tool/model costs.

## 7. Knowledge Boundaries

**This chapter owns task-level orchestration, verification and recovery only.** Consult [Visual State Estimation](Visual_State_Estimation_EN.md) for masks/depth/pose, [Imitation Learning](Data_Imitation_Learning_EN.md) for collecting demonstrations and fitting BC/ACT/Flow policies, [Flow Matching](Flow_Matching_Step_by_Step_Derivation_EN.md) for full derivations, and [GR00T/SONIC](System1_System0_GR00T_SONIC_EN.md) for system interfaces.

References: [RAX](https://assets.robot-agent-exploration.com/paper/rax.pdf) · [Gemini Robotics](https://deepmind.google/models/gemini-robotics/) · [SayCan](https://say-can.github.io/) · [Code as Policies](https://code-as-policies.github.io/). No hardware benchmark or reproduction is claimed.
