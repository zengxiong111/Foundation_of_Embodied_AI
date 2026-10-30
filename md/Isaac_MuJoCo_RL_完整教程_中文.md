---
title: "具身智能仿真与强化学习：从物理引擎到 Sim-to-Real"
subtitle: "理解 Isaac Sim / Isaac Lab、MuJoCo / mjlab、Genesis World 与 Newton"
date: "2026-10-30"
lang: zh-CN
---

# 具身智能仿真与强化学习：从物理引擎到 Sim-to-Real

[课程首页](https://github.com/zengxiong111/Foundation_of_Embodied_AI) · [English](Isaac_MuJoCo_RL_Complete_Guide_EN.md) · [PDF](../pdf/Isaac_MuJoCo_RL_完整教程_中文.pdf)

| 项目 | 说明 |
|:---|:---|
| 课程定位 | 理解机器人仿真、物理后端、RL 训练框架与真机部署的整体关系。 |
| 贯穿案例 | Franka Panda（基础机械臂）与 BrainCo Revo3（灵巧手）。 |
| 前置知识 | 机器人关节、基础反馈控制及强化学习的入门概念。 |
| 使用边界 | 软件能力随版本和任务变化；本文不对不同求解器的精度或速度作普遍排名。 |
| 许可 | CC BY-NC-SA 4.0 · Xiong Zeng and contributors |

## 目录

- [1. 认识平台：先理解层次，再选择工具](#simulation-section-01)
- [2. 理解机器人仿真模型：为什么它会运动？](#simulation-section-02)
- [3. 理解 RL 环境：策略怎样与仿真交互？](#simulation-section-03)
- [4. 统一 RL 训练流程：不同平台怎样接入？](#simulation-section-04)
- [5. 从仿真到真机：为什么训练成功还不够？](#simulation-section-05)
- [6. 常见问题与最佳实践](#simulation-section-06)
- [参考资料与后续学习](#simulation-section-07)

<a id="simulation-section-01"></a>

## 1. 认识平台：先理解层次，再选择工具

### 1.1 一张图理解完整学习栈

机器人 RL 不由一个工具独立完成。**模型描述**定义机器人和场景，**物理引擎**计算运动和接触，**仿真平台**管理场景与传感器，**RL 框架**组织任务、并行采样及算法接口。一个项目可以跨越多个工具，但每一层必须说明清楚。

| 层级 | 负责什么 | 典型工具或格式 |
|:---|:---|:---|
| **机器人与场景描述** | 连杆、关节、惯量、碰撞体、传感器 | URDF、MJCF、USD |
| **物理求解** | 动力学、约束、接触、积分 | PhysX、MuJoCo、Newton 的求解器、Genesis 物理求解器 |
| **场景与传感器** | 可视化、相机、场景构建与同步 | Isaac Sim、Genesis World、MuJoCo 的仿真接口 |
| **RL 环境框架** | Observation / Action / Reward / Termination / Reset、并行环境 | Isaac Lab、mjlab、MuJoCo Playground、自定义 Genesis 环境 |
| **策略优化与部署** | PPO/SAC 等优化、评估、安全执行 | RSL-RL 等训练器、真机控制接口 |

几种**常见组合**（示意而非唯一配置）：

- **Isaac 路线：** Isaac Sim + PhysX → Isaac Lab → RL 训练器；适合复杂场景、相机和规模化任务组织。
- **Newton 路线：** Newton（例如 MJWarp 求解器）→ Isaac Lab 中受支持的任务 → RL 训练器；Newton 可以在不启动 Isaac Sim 的模式下运行。
- **MuJoCo / mjlab 路线：** MJCF → MuJoCo Warp → mjlab → RL 训练器；突出直接的 MuJoCo 数据访问和模块化任务。
- **Genesis 路线：** URDF/MJCF → Genesis World（物理 + 场景 + 传感器）→ 自定义或兼容的 RL 环境与训练器。

### 1.2 新平台分别是什么？

| 平台或组件 | 主要定位 | 需要特别理解的边界 |
|:---|:---|:---|
| **Isaac Sim** | USD 资产、场景、渲染、传感器及仿真运行时 | 不是 RL 优化算法；通常配合 Isaac Lab 训练。 |
| **Isaac Lab** | 机器人学习的任务与环境框架 | 不等于 PhysX；所能切换的后端与任务有关。 |
| **MuJoCo** | 刚体、关节、执行器和接触的物理仿真 | 本体不等于训练框架；CPU 与批量 GPU 路线须区分。 |
| **MJX / MuJoCo Warp** | MuJoCo 生态的批量仿真实现 | MJX-JAX 与 MJWarp 的计算方式和可微性并不相同。 |
| **mjlab** | 基于 MuJoCo Warp 的轻量级 GPU RL 框架 | 借鉴 Isaac Lab 的 Manager-based 设计；不以高保真 RGB 渲染为核心。 |
| **Genesis World（原 Genesis）** | Python 化、多物理耦合的仿真平台 | 集成刚体、软体、粒子等求解器；功能及可微性仍需逐模块核验。 |
| **Newton** | 基于 NVIDIA Warp 的可扩展 GPU 物理引擎 | 可提供多个求解器；本身不是与 mjlab 同类的 RL 任务框架。 |
| **MuJoCo Playground** | MuJoCo 生态的现成机器人学习任务库 | 适合基准任务和快速研究，不等同于 MuJoCo 引擎本身。 |

**两处关键区分：**（1）Newton 与 MJWarp 不是简单的同义词：Newton 可以通过自己的统一接口使用 MuJoCo Warp 求解器，也支持其他求解器；**某个 Newton 求解器支持梯度**不意味着使用 MJWarp 的所有场景都可微。（2）**mjlab 使用 MuJoCo Warp**，这不代表它自动支持 Newton 的其他求解器。上述接口关系以具体发行版本为准。

### 1.3 如何选择？从任务需求出发

| 研究需求 | 优先了解的路线 | 决定因素 |
|:---|:---|:---|
| 机械臂动力学和控制调试 | MuJoCo | 模型、执行器与接触是否容易检查 |
| 轻量级、结构化的并行 RL | mjlab + MuJoCo Warp | 刚体任务、PyTorch 数据流、任务组合 |
| 多相机、视觉与复杂场景 | Isaac Sim + Isaac Lab | 相机、渲染、传感器与任务兼容性 |
| 软体、颗粒、流体及多物理研究 | Genesis World 或 Newton 的对应求解器 | 材料模型、耦合与数值特性 |
| 物理后端对照和扩展 | Isaac Lab + 受支持的 Newton/PhysX 后端 | 任务支持、接口差异、求解器验证 |

不要因为平台宣称 GPU 并行或可微，就推断其在灵巧手任务上**必然更快、更准或更容易 Sim-to-Real**。本课程以 **Panda 抓取方块**建立基本直觉，再用 **Revo3 抓取物体**说明联动执行器、惯性与指尖接触的挑战。

<a id="simulation-section-02"></a>

## 2. 理解机器人仿真模型：为什么它会运动？

### 2.1 一个机器人，三种常见描述

| 描述 | 首要用途 | 转换时容易丢失或改变的内容 |
|:---|:---|:---|
| **URDF** | 连杆、关节、几何与基本惯性描述 | 执行器细节、接触求解参数、场景配置 |
| **MJCF** | MuJoCo 的模型、执行器、约束与接触配置 | 其他仿真器不一定有同等的求解语义 |
| **USD** | Isaac 的场景层级、资产、物理与渲染属性 | 转换后的驱动、碰撞体与传感器需重新核验 |

不能把“成功导入”视为“物理上等价”。**同一机器人的参数主表**至少应包含：关节名称、轴向、零位、限位、质量、质心、惯量、碰撞几何、执行器类型、末端坐标系与传感器外参。

### 2.2 六层模型：从形状到接触

**（1）运动学。** 关节如何连接？转轴方向、零位和限位是什么？Panda 先检查机械臂各关节与末端工具坐标系（TCP）；Revo3 则还要逐指检查关节顺序、耦合关系及可动范围。

**（2）质量与惯量。** 质量、质心和转动惯量决定惯性与重力响应。只让模型的视觉外观“看起来正确”，并不意味着抓取时的力学行为正确。

**（3）执行器。** 策略给出关节目标后，底层伺服如何产生力矩？需要明确控制模式、刚度 $K_p$、阻尼 $K_d$、速度/力矩限幅和响应延迟。**Armature** 用于表示关节侧的附加等效惯性（常与电机及传动有关）；它不是通用的“越大越好”参数，也不能替代真实惯量识别。灵巧手各关节的值可以不同，但必须有模型或测量依据。

**（4）碰撞与接触。** 可视网格不等于碰撞几何。指尖是否抓得住物体，取决于接触几何、摩擦、数值求解设置和执行器输出；仅增大摩擦系数往往不能解决穿透或抖动。

**（5）传感器。** 关节编码器、相机、力/触觉信号应说明物理含义、坐标系、单位、采样时刻及噪声。仿真可直接读取的物体真值，不一定是真机可用的策略输入。

**（6）约束和耦合。** 若 Revo3 所用模型包含联动关节、传动或肌腱约束，应按实际机械结构检查；不要根据外形假定每个关节都有独立、直接的力矩输入。

### 2.3 “发送位置目标”不等于“瞬移关节”

常见关节位置伺服可概念性写成：

$$
\tau = \operatorname{sat}\!\left(K_p(q_{\mathrm{target}}-q)-K_d\dot q\right),
$$

其中 $\operatorname{sat}$ 表示执行器限幅；真实设备还可能包含重力补偿、摩擦、滤波和内部控制环。**动作是目标位置**时，实际运动仍由伺服和物理引擎决定。若直接把关节状态设置到目标位置，就可能绕开这些约束，导致不可信的 Sim-to-Real 结果。

同时必须区分三种频率：**物理积分频率、低层控制频率、策略/相机采样频率**。例如物理步进为 500 Hz、控制器更新为 250 Hz、策略为 50 Hz，则每次策略决策覆盖 10 个物理步、5 次控制更新；若相机为 30 Hz，图像不能被视为每个策略步都同步更新。上述数值仅为说明时间关系，**不是 Revo3 的设备规格**。

**跨引擎迁移时的检查：** 在 Isaac Sim、MuJoCo、Genesis World 或 Newton 之间导入同一个 URDF/MJCF/USD，并不自动保证碰撞体、关节驱动、约束、摩擦或传感器语义一致；先验证单关节、静态接触，再验证完整抓取。尤其要区分 Genesis 的多物理耦合设置与 Newton 的具体求解器。

**本章要点：** 比较不同仿真器之前，先让机器人结构、执行器语义、动作尺度、时间步及接触配置尽可能一致。

<a id="simulation-section-03"></a>

## 3. 理解 RL 环境：策略怎样与仿真交互？

RL 环境把连续的机器人仿真包装成一个可训练的闭环：**读取观测 → 策略输出动作 → 控制器执行 → 物理前进 → 计算奖励与完成条件 → 必要时重置**。

| 五个接口 | 一句话理解 | Revo3 抓取物体的概念示例 |
|:---|:---|:---|
| **Observation** | 策略能够看到什么 | 手指关节、速度；可部署的物体位姿估计或触觉 |
| **Action** | 策略希望机器人做什么 | 经限幅的关节位置目标或增量 |
| **Reward** | 哪些行为更值得鼓励 | 接近、建立有效接触、稳定抓取；惩罚不安全动作 |
| **Termination** | 回合是否真正结束 | 成功、不可恢复失败；时间上限需单独标记 |
| **Reset** | 下个回合从什么状态开始 | 恢复手指状态并重新放置物体；清空历史缓存 |

**观测不等于仿真全部状态。** 训练时 critic 可以使用物体真值等 privileged information；但若真机 actor 只能看到关节与相机，训练时也不能让 actor 暗中使用物体真值。若策略需要触觉，必须明确真机是否存在对应传感器及其延迟。

**动作不等于电机力矩。** 即使两个环境都输出 20 维动作，也可能一个代表位置增量，另一个代表力矩。比对策略前要统一关节顺序、动作语义、缩放、限幅和执行周期。

**终止不等于超时。** “物体掉落导致任务不可恢复”与“达到最大回合长度”应分开记录；否则可能影响价值学习和成功率统计。重置必须同步清理物理状态、控制器内部状态和历史观测。

**框架只是实现方式。** 无论在 Isaac Lab、mjlab、Playground 还是 Genesis 自定义环境中，这五个接口的数学意义不变；但任务是否使用相同的观察时刻、动作缩放、奖励和 reset 逻辑，需要自己验证。

<a id="simulation-section-04"></a>

## 4. 统一 RL 训练流程：不同平台怎样接入？

### 4.1 所有路线共用八个步骤

**核验机器人模型 → 定义五个 RL 接口 → 检查动作与接触 → 建立简单基线 → 批量训练 → 独立评估 → 加入合理随机化 → 保存可复现配置。** 这是本课程唯一的训练主线；不需要分别记忆 Isaac Lab、mjlab 和 Genesis 的多套操作步骤。

先在固定初始条件下验证行为，再增加多环境采样和困难接触。若训练奖励上涨而抓取成功率没有提高，优先检查成功判据、奖励漏洞及观测信息，而不是直接增大网络。

### 4.2 五条路线的工程差异

| 路线 | 场景与模型 | RL 环境组织 | 优先检查 |
|:---|:---|:---|:---|
| **Isaac Lab（PhysX）** | Isaac Sim、USD、Articulation | Manager-based / Direct；批量任务 | 渲染、传感器、资产和 PhysX 配置 |
| **Isaac Lab（Newton）** | Warp 原生后端；按任务选求解器 | 同一套环境抽象中受支持的任务 | Newton 求解器、功能覆盖和数据/关节顺序 |
| **mjlab / MuJoCo Warp** | MJCF、`MjModel`、`MjData` | Manager-based 风格；GPU 并行 | 执行器、接触、模型数据与 GPU 支持 |
| **MuJoCo Playground** | MuJoCo/MJX 生态资产 | 现成任务、包装器与兼容训练脚本 | 所选 MJX 实现、任务与训练器接口 |
| **Genesis World** | Python 场景与多物理求解器 | 自定义或兼容的 RL 环境接口 | 材料求解器、接触耦合、观测及训练集成 |

**Isaac Lab 的 Manager-based 与 Direct 是任务软件结构，不是两种 RL 算法。** mjlab 采用可组合的 Manager-based 思路，但底层数据模型来自 MuJoCo。Genesis 则将多物理求解与场景接口组织在同一个平台中，RL 训练器仍需按接口接入。Newton 是物理层：经受支持的集成进入训练框架，不应把它误当作另一个 PPO 实现。

### 4.3 PPO、SAC 与可微仿真分别处于哪里？

**PPO** 常用于同步、批量 rollout；**SAC** 使用 replay buffer 做 off-policy 学习。物理引擎本身不保证某个训练器支持 SAC、GPU rollout 或特定的观测结构，应检查实际适配。

**可微仿真**意味着某些求解器/操作允许对模拟结果求导，可能用于参数辨识或基于梯度的优化；它**不是**普通 PPO/SAC 训练的必需条件。Newton 和 Genesis 的可微能力必须逐求解器确认；MuJoCo 的 MJX-JAX 与 MJWarp 也不能混为一谈。

### 4.4 不要只比较峰值 FPS

至少同时记录：**环境步/秒、物理步/秒、策略更新耗时、显存、训练到目标成功率所需墙钟时间**。对于视觉或灵巧手任务，还应统计相机/视觉编码器、接触求解和 CPU-GPU 数据传输耗时。公平比较应固定机器人质量/惯量、动作语义、策略频率、扰动分布、成功判据与训练预算。

**同一策略跨平台回放**是有意义的稳健性检验，但结果差异可能来自建模、驱动、接触和时间步，而不只是“哪个物理引擎更准确”。

<a id="simulation-section-05"></a>

## 5. 从仿真到真机：为什么训练成功还不够？

### 5.1 Sim-to-Real 的五类差异

| 差异来源 | Panda / Revo3 需要关注的例子 |
|:---|:---|
| **几何与惯性** | 关节零位、TCP、连杆惯量、手部负载 |
| **执行器** | 位置/电流/力矩命令语义、$K_p$/$K_d$、饱和与延迟 |
| **接触** | 指尖碰撞几何、摩擦、物体滑移与柔顺性 |
| **观测** | 相机标定、深度误差、遮挡、触觉噪声 |
| **时间与通信** | 控制频率、时间戳、缓存、丢包、动作滤波 |

**系统辨识（System Identification）**先估计实际机器人与物体的关键参数；**域随机化（Domain Randomization）**再让策略在合理范围内适应剩余不确定性。随机化不能代替错误模型的修复，也不是范围越大越好。

推荐的难度顺序：**固定条件下完成任务 → 扩大初始位姿 → 随机化动力学与执行器 → 加入估计噪声/视觉变化 → 扰动与恢复**。Revo3 抓取时，物体质量和摩擦、关节响应、指尖接触以及可能的滑移，往往比花哨的背景纹理更值得先处理。

### 5.2 真机部署必须保持“接口契约”一致

至少检查五件事：**关节顺序和坐标系**一致；**动作含义及限幅**一致；**策略、控制、传感器频率和延迟**合理；**观测只使用真机可得的信息**；**安全保护**（限速、限力、工作空间和急停）独立于 RL 策略工作。

若使用状态 teacher 训练、再蒸馏视觉 student，部署对象应是能从实际传感器得到所需观测的 student；不要误将仅依赖仿真真值的 teacher 当作可部署策略。涉及四元数时，也要明确 `wxyz` 与 `xyzw` 的顺序。

### 5.3 评估看成功和鲁棒性，不只看奖励

建议至少报告：**独立初始条件下的成功率、完成时间、滑移/掉落率、扰动恢复率、不安全事件率**，并说明训练样本量与墙钟时间。多个随机种子和训练外扰动能帮助区分“记住训练分布”与“真正泛化”。跨 PhysX、MuJoCo、Genesis 或 Newton 求解器的验证可暴露模型依赖，但不能保证真机成功。

<a id="simulation-section-06"></a>

## 6. 常见问题与最佳实践

| 现象 | 优先检查 | 不应立刻做什么 |
|:---|:---|:---|
| 机器人抖动或关节发散 | 零位、轴向、$K_p$/$K_d$、步长、力矩限幅 | 盲目提高增益 |
| 指尖穿透、夹持不稳 | 碰撞体、接触参数、摩擦、控制响应 | 仅增大摩擦 |
| RL 奖励上升但抓取失败 | 成功判定、奖励漏洞、终止与 reset | 只延长训练 |
| 并行训练很慢或显存不足 | 环境数、相机、渲染、数据搬运 | 无条件增加环境数 |
| 仿真成功、真机失败 | 关节/坐标、动作语义、延迟、摩擦、标定 | 首先换更大的策略网络 |

**统一排查顺序：** **机器人模型 → 单位和坐标 → 执行器 → 接触与时间步 → 观测/延迟 → RL 环境 → 优化算法**。上游问题没有排除之前，调节 PPO 或 SAC 超参数通常不会解决根本原因。

**课程总结：** MuJoCo 适合分析机器人力学与控制，Isaac Sim 擅长场景和视觉，Isaac Lab 与 mjlab 组织 RL 训练，Genesis World 扩展多物理场问题，Newton 提供可选的 GPU 物理求解能力。**没有脱离任务的“最佳平台”**；正确的模型、清晰的接口和真机验证比平台名称更重要。

<a id="simulation-section-07"></a>

## 参考资料与后续学习

1. [Isaac Sim 官方文档](https://docs.isaacsim.omniverse.nvidia.com/) — 场景、USD、渲染与传感器。
2. [Isaac Lab 官方文档](https://docs.nvidia.com/isaaclab/) 与 [物理后端说明](https://docs.nvidia.com/isaaclab/concepts/physics-backends) — RL 任务、多后端及 Newton 支持范围。
3. [MuJoCo 官方文档](https://mujoco.readthedocs.io/en/stable/) — 动力学、执行器与接触。
4. [MuJoCo MJX 文档](https://mujoco.readthedocs.io/en/latest/mjx.html) — MJX-JAX / MJX-Warp 的区别。
5. [mjlab 官方文档](https://mujocolab.github.io/mjlab/main/) — MuJoCo Warp、Manager-based RL 环境与模型数据。
6. [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) — 机器人学习任务和基准。
7. [Genesis World 官方文档](https://genesis-world.readthedocs.io/en/latest/) — 多物理求解器、场景与传感器。
8. [Newton 官方文档](https://newton-physics.github.io/newton/stable/guide/overview.html) 与 [求解器比较](https://newton-physics.github.io/newton/latest/solvers/index.html) — Warp 引擎及其不同求解器。
9. [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) — 公开机器人模型。
10. [BrainCo 机器人描述仓库](https://github.com/BrainCoTech/brainco-description) — Revo3 的实际模型应按仓库版本验证。
