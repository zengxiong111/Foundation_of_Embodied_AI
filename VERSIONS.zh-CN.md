# 版本、适用范围与核验边界

记录当前教材的适用范围和上游源码快照。核验日期表示文档整理与引用核对，不代表教程、训练或硬件运行通过。

[English](VERSIONS.md)

## 目录

- [适用范围](#applicability)
- [固定源码快照](#snapshots)
- [使用与更新规则](#updates)

<a id="applicability"></a>

## 适用范围

| 主题 | 适用版本或范围 | 文档核验日期 |
|---|---|---|
| Git 基础 | Git 2.x；Ubuntu 终端；GitHub 协作示例 | 2026-10-18 |
| ROS 2 基础 | Ubuntu 24.04；ROS 2 Jazzy；Python 示例 | 2026-10-18 |
| 相机标定 | OpenCV 4.x 概念；ROS 2 Jazzy；设备 SDK 版本需另行匹配 | 2026-10-18 |
| 视觉感知与状态估计 | 相机几何、RGB-D/双目、分割、FoundationPose、SE(3) 滤波、不确定性、异步执行与恢复概念 | 2026-10-18 |
| 机器人动力学 | 刚体动力学理论；MuJoCo 与 Isaac 示例为概念说明，非已测试的跨引擎组合 | 2026-10-18 |
| Isaac / MuJoCo 仿真 | 文档路线：Isaac Lab 2.3.2 / Isaac Sim 5.1；Lab 3.0 Early Access / Sim 6.1。MuJoCo、MJX、MJWarp 需按任务选择版本 | 2026-10-18 |
| 具身强化学习 | 算法与任务分类；不规定统一框架运行环境 | 2026-10-18 |
| 运动控制 RL | Isaac Lab / RSL-RL 工作流为说明性示例；任务、训练器、资产和执行器版本需一起绑定 | 2026-10-18 |
| PPO | PPO-Clip / GAE 理论；PyTorch 风格示例为教学内容，非已测试训练包 | 2026-10-18 |
| SAC | Q-learning、TD/SARSA、DQN、Soft Bellman 方程、现代 SAC、双 Critic、重参数化与自动熵调节；教学伪代码 | 2026-10-18 |
| 数据采集与模仿学习 | 示教、数据集、BC、DAgger、ACT、生成式动作块与 RTC 概念；不规定统一且经过运行测试的软件环境 | 2026-10-18 |
| Flow Matching | CFM 理论；OpenPI pi0.py 代码引用固定如下；示例不代表已完成微调实验 | 2026-10-18 |
| GR00T 与 SONIC | SONIC 论文集成：GR00T N1.5；文档 G1 工作流：N1.7、78 维动作、2.5 Hz 推理 / 50 Hz 动作流。Checkpoint 与观测配置须匹配 | 2026-10-18 |
| 先进 Robot Agent | ASPIRE 与 RAX：在线探索、执行轨迹诊断、程序修复与编译、技能记忆及验证概念 | 2026-10-18 |

<a id="snapshots"></a>

## 固定源码快照

以下 Git ref 在 2026-10-18 通过 GitHub API 解析到固定 commit；这是可定位的引用快照，不保证这些项目彼此兼容。分支状态和文档页面可能随后变化。

| 项目 | 原 ref | 固定 commit | 上游 commit 时间（UTC） |
|---|---|---|---|
| Physical-Intelligence/openpi | `main` | [`215abfb217dbac7d5f1273282331b9b1866c0479`](https://github.com/Physical-Intelligence/openpi/commit/215abfb217dbac7d5f1273282331b9b1866c0479) | 2026-08-24T16:27:12Z |
| isaac-sim/IsaacLab | `v2.3.2` | [`37ddf626871758333d6ed89cf64ad702aef127d0`](https://github.com/isaac-sim/IsaacLab/commit/37ddf626871758333d6ed89cf64ad702aef127d0) | 2026-01-29T18:16:56Z |
| isaac-sim/IsaacLab | `release/3.0.0` | [`1693d8539b2404892ad2e1128a5f6a74b56a7dbd`](https://github.com/isaac-sim/IsaacLab/commit/1693d8539b2404892ad2e1128a5f6a74b56a7dbd) | 2026-10-18T01:57:22Z |
| NVIDIA/Isaac-GR00T | `main` | [`dac54b313160c201c450f97df0fd74ebd379c892`](https://github.com/NVIDIA/Isaac-GR00T/commit/dac54b313160c201c450f97df0fd74ebd379c892) | 2026-10-15T21:46:24Z |
| NVlabs/GR00T-WholeBodyControl | `main` | [`45c03facbf848e0c743091b7d9c17952096ff908`](https://github.com/NVlabs/GR00T-WholeBodyControl/commit/45c03facbf848e0c743091b7d9c17952096ff908) | 2026-10-13T20:34:46Z |
| ros2/ros2_documentation | `jazzy` | [`03f26be23292f1b8523b3e4f807f3f123ed5b12a`](https://github.com/ros2/ros2_documentation/commit/03f26be23292f1b8523b3e4f807f3f123ed5b12a) | 2026-10-15T20:23:15Z |

<a id="updates"></a>

## 使用与更新规则

1. 复现实验时，另行记录操作系统、Python、PyTorch、CUDA/驱动、仿真器、训练器、资产和 checkpoint；本表不替代已运行环境的锁定配置。
2. Isaac Lab 2.x 与 3.0 EA 不作为可互换环境；核对物理后端、四元数顺序、资产 API、执行器和 reset/step 语义。EA 状态是文档快照，后续以固定源码和官方发布信息核对。
3. GR00T N1.5 论文集成与 N1.7 工作流分开解释；SONIC checkpoint、观测配置、初始 token 和 VLA 训练数据必须匹配。
4. 所有教程均未在本次文档维护中完成训练或真机验证。不要将引用存在、PDF 构建通过或跨引擎示例当作性能或硬件准入证据。
5. 更新版本时编辑 `materials.json`，记录核验日期和 commit，运行 `npm run sync`、`npm run build` 和 `npm run check`，同步中英文及网站副本。不要添加历史素材或日志归档。

官方资料：[Isaac Lab 发布](https://github.com/isaac-sim/IsaacLab/releases)、[ROS 2 Jazzy](https://docs.ros.org/en/jazzy/)、[GR00T / SONIC 工作流](https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/vla_workflow.html)。
