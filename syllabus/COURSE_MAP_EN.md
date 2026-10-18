# Foundation of Embodied AI: Course Map

Fourteen topics are organized as tools → physics and perception → robot RL → demonstrations and generative policies → systems integration. Long-form derivations remain optional deep dives rather than strict prerequisites.

## Six learning modules

| Module | Topics | Outcome |
|---|---|---|
| 0. Orientation | Systems map and guide | Explain perception–policy–control loops |
| 1. Tools | Git, ROS 2 | Manage code and understand ROS communication and TF |
| 2. Physics and perception | Dynamics, calibration, vision foundation models, Isaac / MuJoCo | Relate physical models, sensors and simulation |
| 3. Reinforcement learning | Embodied RL, motion RL, PPO, SAC | Design observations, actions, rewards and data collection |
| 4. Demonstrations and models | Imitation Learning, Flow Matching, GR00T / SONIC | Learn chunk policies from demonstrations and connect hierarchical controllers |
| 5. Robot Agent integration | Robot Agent planning, tool use, verification and recovery | Orchestrate perception, policy and control for long-horizon tasks |

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

## Reading paths

1. Beginners: orientation → essential Git and ROS 2 → dynamics → calibration → visual estimates → simulation.
2. Motion control: dynamics → simulation → embodied RL → PPO / SAC → motion control RL.
3. VLA: calibration → vision foundation models → Imitation Learning → Flow Matching → GR00T / SONIC → Robot Agent.
4. Mathematical deep dives: full PPO, SAC, and Flow Matching derivations are reference material; hands-on learning does not require deriving everything first.

## Shared Panda + cube + table exercise

- **Dynamics:** joint input, contact, end-effector motion.
- **Calibration and vision:** convert camera measurements into a robot-base target; reacquire after lost tracking.
- **Simulation:** use conceptually matched tasks in Isaac Lab / MuJoCo, acknowledging engine/contact differences.
- **RL:** define observation, action, reward, reset; compare data consumption in PPO and SAC.
- **Imitation:** collect demonstrations, then BC / ACT / Flow Matching chunk policies.
- **Integration:** measure safety limits, controller latency, task success and recovery.

## Validation boundary

Documentation, formulae, references and proposed experiments **do not establish that every experiment was run and verified in a common environment**. Record versions, camera calibration error, servo rates, safety logic and test outcomes when deploying hardware.


## Advanced Robot Agents for System Integration

Final topic: [Robot Agents: planning, execution, verification and recovery](../md/Robot_Agent_Workflow_EN.md). It covers RAX exploration and compilation, ASPIRE trace-guided repair, skill reuse and a dexterous-hand screwdriver example. The combined workflow is a teaching example; no code or experiments are included.

## Cross-Chapter Ownership to Avoid Duplication

| Canonical chapter | What it teaches | What related chapters reuse |
|---|---|---|
| Camera Calibration | Intrinsics, extrinsics, hand-eye, TF and frame geometry | Vision uses calibrated transforms without duplicating hand-eye derivations |
| Vision Foundation Models | Model outputs, SAM 2/3/3D, FoundationPose/Stereo and model selection/composition | Robot Agent combines perception outputs for task decisions |
| Imitation Learning | BC/DAgger/ACT, generative action chunks and real-time RTC | GR00T/SONIC focuses on its specific training and deployment interfaces |
| Flow Matching | Full CFM mathematics, velocity fields and ODE derivations | Imitation learning uses only the conditional action-generation interface |
| PPO / SAC and Motion RL | RL objectives and motion-control engineering | Imitation learning does not duplicate reward-optimization derivations |
| Robot Agent | Task planning, skill tools, independent verification, recovery and memory | GR00T/SONIC remains an embodiment-specific System 1/0 case study |
