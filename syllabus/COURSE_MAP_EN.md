# Foundation of Embodied AI: Course Map

[简体中文](COURSE_MAP_ZH.md) | **English** | [PDF](COURSE_MAP_EN.pdf)

Fourteen topics are organized as tools → physics and perception → robot RL → demonstrations and generative policies → systems integration. Long-form derivations remain optional deep dives rather than strict prerequisites.

## Contents

- [Six learning modules](#map-section-01)
- [Embodied AI system map](#map-section-02)
- [Reading paths](#map-section-03)
- [Tutorial overview](#map-section-04)

<a id="map-section-01"></a>

## Six learning modules

| Module | Topics | Outcome |
|---|---|---|
| 0. Orientation | Systems map and guide | Explain perception–policy–control loops |
| 1. Tools | Git, ROS 2 | Manage code and understand ROS communication and TF |
| 2. Physics and perception | Dynamics, calibration, vision foundation models, Isaac / MuJoCo | Relate physical models, sensors and simulation |
| 3. Reinforcement learning | Motion RL, PPO, SAC | Design observations, actions, rewards and data collection |
| 4. Demonstrations and models | Imitation Learning, Flow Matching, RL Post-training, GR00T / SONIC | Learn chunk policies from demonstrations and connect hierarchical controllers |
| 5. Robot Agent integration | Robot Agent planning, tool use, verification and recovery | Orchestrate perception, policy and control for long-horizon tasks |

<a id="map-section-02"></a>

## Embodied AI system map

```mermaid
flowchart LR
 A["Task command"] --> B["Visual / state estimation"]
 B --> C["High-level VLA / action policy"]
 C --> D["Motion policy / servo control"]
 D --> E["Robot and environment"]
 E --> B
 F["Demonstrations / imitation"] --> C
 G["Isaac / MuJoCo simulator"] --> H["PPO / SAC optimization"]
 H --> C
 H --> D
 X["Robot Agent / orchestration"] --> B
 X --> C
 E --> X
```

<a id="map-section-03"></a>

## Reading paths

1. Beginners: orientation → essential Git and ROS 2 → dynamics → calibration → vision foundation models → simulation.
2. Motion control: dynamics → simulation → PPO / SAC → motion control RL.
3. VLA: calibration → vision foundation models → Imitation Learning → Flow Matching → RL Post-training → GR00T / SONIC → Robot Agent.
4. Mathematical deep dives: full PPO, SAC, and Flow Matching derivations are reference material; hands-on learning does not require deriving everything first.

<a id="map-section-04"></a>

## Tutorial overview

### 1. Git Basics

[Git Basics](../md/Git_Tutorial_English.md) — Learn Git’s working tree, staging area and commit history, then follow a GitHub workflow covering branches, pull requests, conflict resolution and common recovery commands.

### 2. ROS 2 Basics

[ROS 2 Basics](../md/ROS2_Robot_Learning_Tutorial_EN.md) — Understand nodes, topics, services, actions, QoS and TF. Connect sensors, state estimation, policy inference and controllers through ROS 2 interfaces.

### 3. Camera Calibration

[Camera Calibration](../md/Robot_Camera_Calibration_Tutorial_EN.md) — Study camera intrinsics, distortion, extrinsics and hand–eye calibration. Relate RGB-D observations and coordinate transforms to robot-frame object poses.

### 4. Vision Foundation Models

[Vision Foundation Models](../md/Visual_State_Estimation_EN.md) — Compare segmentation, pose estimation, stereo depth and 3D foundation models. Follow four robotics workflows spanning known objects, stereo geometry, unknown shapes and human demonstrations.

### 5. Robot Dynamics

[Robot Dynamics](../md/Robot_Dynamics_Embodied_EN.md) — Develop rigid-body motion and dynamics through coordinate frames, Jacobians, inertia and equations of motion. Connect contact, friction and control models to robot simulation.

### 6. Isaac / MuJoCo Simulation

[Isaac / MuJoCo Simulation](../md/Isaac_MuJoCo_RL_Complete_Guide_EN.md) — Understand the roles of Isaac Sim, Isaac Lab and MuJoCo. Learn how robot assets, actuators, observations, rewards and environment interfaces form a simulation-based RL pipeline.

### 7. Motion Control RL

[Motion Control RL](../md/Embodied_Motion_Control_RL_Tutorial_EN.md) — Connect MDP design and PPO/SAC to robot motion control. Study observations, action interfaces, rewards, resets, parallel simulation, domain randomization and Sim-to-Real considerations.

### 8. PPO

[PPO](../md/PPO_Complete_Derivation_and_Design_Motivations_English.md) — Derive PPO-Clip and Actor–Critic learning from policy gradients, Bellman targets and GAE. Explain Gaussian policy updates, gradient isolation and asymmetric robot observations.

### 9. SAC

[SAC](../md/SAC_Complete_Derivation_EN.md) — Trace Q-learning, TD/SARSA and DQN toward entropy-regularized RL. Derive soft Bellman equations and modern SAC with twin critics, reparameterization and automatic entropy tuning.

### 10. Imitation Learning

[Imitation Learning](../md/Data_Imitation_Learning_EN.md) — Introduce BC, DAgger and ACT, then connect Diffusion Policy and Flow Matching to action-chunk generation. Explain temporal ensembling and inference-time versus training-time RTC.

### 11. Flow Matching

[Flow Matching](../md/Flow_Matching_Step_by_Step_Derivation_EN.md) — Derive conditional Flow Matching, velocity fields and the training objective. Explain how ODE integration transforms noise into conditional robot action sequences.

### 12. RL Post-training

[RL Post-training](../md/RL_Post_Training_EN.md) — Study RL refinement of pretrained Diffusion, Flow Matching and VLA policies. Compare generator updates with frozen-generator interfaces, Actor–Critic loops and real-robot feedback.

### 13. GR00T & SONIC

[GR00T & SONIC](../md/System1_System0_GR00T_SONIC_EN.md) — Use GR00T and SONIC to explain System 1/System 0 humanoid control. Connect high-level action chunks, motion representations, whole-body policies and low-level execution.

### 14. Robot Agents

[Robot Agents](../md/Robot_Agent_Workflow_EN.md) — Distinguish foundation models, coding agents and agent harnesses. Study model-driven workflows, RAX exploration and compilation, ASPIRE trace repair, skill memory and task-level verification.
