# Versions, applicability and verification scope

Record the current materials’ applicability and upstream source snapshots. Review dates describe documentation and reference checks, not successful tutorial, training or hardware runs.

[简体中文](VERSIONS.zh-CN.md)

## Contents

- [Applicability](#applicability)
- [Pinned source snapshots](#snapshots)
- [Usage and update rules](#updates)

<a id="applicability"></a>

## Applicability

| Topic | Applicable versions or scope | Documentation review date |
|---|---|---|
| Git Basics | Git 2.x; Ubuntu terminal; GitHub workflow examples | 2026-10-18 |
| ROS 2 Basics | Ubuntu 24.04; ROS 2 Jazzy; Python examples | 2026-10-18 |
| Camera Calibration | OpenCV 4.x concepts; ROS 2 Jazzy; device SDK version must be selected separately | 2026-10-18 |
| Visual Perception and State Estimation | Camera geometry, RGB-D/stereo, segmentation, FoundationPose, SE(3) filtering, uncertainty, asynchronous execution and recovery concepts | 2026-10-18 |
| Robot Dynamics | Rigid-body theory; MuJoCo and Isaac examples are conceptual, not a tested cross-engine pair | 2026-10-18 |
| Isaac / MuJoCo Simulation | Documented tracks: Isaac Lab 2.3.2 / Isaac Sim 5.1; Lab 3.0 Early Access / Sim 6.1. MuJoCo, MJX and MJWarp releases must be selected per task | 2026-10-18 |
| Embodied RL | Algorithm and task taxonomy; no single framework environment is prescribed | 2026-10-18 |
| Motion Control RL | Isaac Lab / RSL-RL workflows are illustrative; task, trainer, asset and actuator versions must be bound together | 2026-10-18 |
| PPO | PPO-Clip / GAE theory; PyTorch-style examples are instructional, not a tested training package | 2026-10-18 |
| SAC | Q-learning, TD/SARSA, DQN, soft Bellman equations, modern SAC, twin critics, reparameterization and automatic entropy tuning; instructional pseudocode | 2026-10-18 |
| Data Collection and Imitation Learning | Demonstrations, datasets, BC, DAgger, ACT, generative action chunks and RTC concepts; no single tested software environment is prescribed | 2026-10-18 |
| Flow Matching | CFM theory; OpenPI pi0.py code reference pinned below; examples are not a tested fine-tuning run | 2026-10-18 |
| GR00T & SONIC | SONIC paper integration: GR00T N1.5; documented G1 workflow: N1.7, 78-dim actions, 2.5 Hz inference / 50 Hz stream. Checkpoint and observation config must match | 2026-10-18 |
| Advanced Robot Agents | ASPIRE and RAX: online exploration, trace-guided repair, program compilation, skill memory and verification concepts | 2026-10-18 |

<a id="snapshots"></a>

## Pinned source snapshots

The following Git refs were resolved to fixed commits through the GitHub API on 2026-10-18. These are locatable reference snapshots, not a guarantee of compatibility between projects. Branch status and documentation pages may subsequently change.

| Project | Original ref | Pinned commit | Upstream commit time (UTC) |
|---|---|---|---|
| Physical-Intelligence/openpi | `main` | [`215abfb217dbac7d5f1273282331b9b1866c0479`](https://github.com/Physical-Intelligence/openpi/commit/215abfb217dbac7d5f1273282331b9b1866c0479) | 2026-08-24T16:27:12Z |
| isaac-sim/IsaacLab | `v2.3.2` | [`37ddf626871758333d6ed89cf64ad702aef127d0`](https://github.com/isaac-sim/IsaacLab/commit/37ddf626871758333d6ed89cf64ad702aef127d0) | 2026-01-29T18:16:56Z |
| isaac-sim/IsaacLab | `release/3.0.0` | [`1693d8539b2404892ad2e1128a5f6a74b56a7dbd`](https://github.com/isaac-sim/IsaacLab/commit/1693d8539b2404892ad2e1128a5f6a74b56a7dbd) | 2026-10-18T01:57:22Z |
| NVIDIA/Isaac-GR00T | `main` | [`dac54b313160c201c450f97df0fd74ebd379c892`](https://github.com/NVIDIA/Isaac-GR00T/commit/dac54b313160c201c450f97df0fd74ebd379c892) | 2026-10-15T21:46:24Z |
| NVlabs/GR00T-WholeBodyControl | `main` | [`45c03facbf848e0c743091b7d9c17952096ff908`](https://github.com/NVlabs/GR00T-WholeBodyControl/commit/45c03facbf848e0c743091b7d9c17952096ff908) | 2026-10-13T20:34:46Z |
| ros2/ros2_documentation | `jazzy` | [`03f26be23292f1b8523b3e4f807f3f123ed5b12a`](https://github.com/ros2/ros2_documentation/commit/03f26be23292f1b8523b3e4f807f3f123ed5b12a) | 2026-10-15T20:23:15Z |

<a id="updates"></a>

## Usage and update rules

1. For experiment reproduction, separately record OS, Python, PyTorch, CUDA/driver, simulator, trainer, assets and checkpoint. This table does not replace a locked configuration of an actually executed environment.
2. Isaac Lab 2.x and 3.0 EA are not interchangeable environments. Check physics backend, quaternion order, asset APIs, actuators and reset/step semantics. EA status is a documentation snapshot; subsequently check fixed source and official releases.
3. Explain the GR00T N1.5 paper integration separately from the N1.7 workflow. SONIC checkpoint, observation configuration, initial token and VLA training data must match.
4. No tutorial has completed training or real-robot validation during this documentation maintenance. Reference availability, a successful PDF build or a cross-engine example is not evidence of performance or hardware admission.
5. When updating versions, edit `materials.json`, record the review date and commit, then run `npm run sync`, `npm run build` and `npm run check`. Synchronize both languages and website copies. Do not add historical material or log archives.

Official resources: [Isaac Lab releases](https://github.com/isaac-sim/IsaacLab/releases), [ROS 2 Jazzy](https://docs.ros.org/en/jazzy/), [GR00T / SONIC workflow](https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/vla_workflow.html).
