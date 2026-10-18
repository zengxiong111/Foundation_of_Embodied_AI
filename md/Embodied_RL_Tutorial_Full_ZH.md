# 具身智能中的强化学习教程：算法分类、任务分类与应用场景

<!-- course-meta:start -->
[课程首页](https://github.com/zengxiong111/Foundation_of_Embodied_AI/blob/main/README.zh-CN.md) · [English](https://github.com/zengxiong111/Foundation_of_Embodied_AI/blob/main/md/Embodied_RL_Tutorial_Full_EN.md) · [PDF](https://zengxiong111.github.io/zengxiong.github.io/files/Embodied_RL_Tutorial_Full_ZH.pdf)

| 项目 | 说明 |
|---|---|
| 适用范围 | 算法与任务分类；不规定统一框架运行环境 |
| 文档核验日期 | 2026-10-18 |
| 许可 | CC BY 4.0 · Xiong Zeng and contributors |

**核验边界：** 文档整理与上游引用快照核对；未进行教程环境、训练或硬件验证。 版本与固定源码引用见[版本说明](https://github.com/zengxiong111/Foundation_of_Embodied_AI/blob/main/VERSIONS.zh-CN.md)。原创教材采用 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)，第三方引用、图片及代码遵循其原有许可；注明来源不等于拥有再许可权。
<!-- course-meta:end -->

<!-- course-toc:start -->
## 目录

- [一、具身强化学习的总体分类](#course-section-01)
  - [1.1 分类框架](#course-section-02)
  - [1.2 核心分类表](#course-section-03)
- [二、按算法分类：PPO 与 SAC](#course-section-04)
  - [2.1 On-policy RL：PPO](#course-section-05)
  - [2.2 Off-policy RL：SAC](#course-section-06)
  - [2.3 PPO 与 SAC 的比较](#course-section-07)
- [三、按任务分类：机器人运动控制强化学习](#course-section-08)
  - [3.1 Locomotion RL：机器人运动控制](#course-section-09)
  - [3.2 Manipulation RL：机器人操作强化学习](#course-section-10)
- [四、按任务分类：具身基础模型强化学习后训练](#course-section-11)
  - [4.1 什么是具身基础模型 RL Post-training？](#course-section-12)
  - [4.2 RL 后训练主要解决什么问题？](#course-section-13)
  - [4.3 具身基础模型 RL 后训练的五类方法](#course-section-14)
  - [4.4 五类方法的总结](#course-section-15)
- [五、PPO、SAC 在两类任务中的具体用途](#course-section-16)
  - [5.1 PPO + Locomotion](#course-section-17)
  - [5.2 SAC + Manipulation](#course-section-18)
  - [5.3 PPO 类方法 + VLA](#course-section-19)
  - [5.4 SAC / Off-policy Critic 类方法 + VLA](#course-section-20)
- [六、如何根据任务选择 RL 方法？](#course-section-21)
- [七、两个研究方向的典型实验路线](#course-section-22)
  - [7.1 机器人运控 RL](#course-section-23)
  - [7.2 具身基础模型 RL 后训练](#course-section-24)
- [八、最后总结](#course-section-25)
<!-- course-toc:end -->

**章节知识地图**

```mermaid
flowchart LR
  A["任务与观测"] --> B["PPO / SAC"] --> C["策略表达"] --> D["评估指标"]
```


<a id="course-section-01"></a>

## 一、具身强化学习的总体分类

<a id="course-section-02"></a>

### 1.1 分类框架

```text
                具身智能强化学习
              Embodied Reinforcement Learning
                         │
           ┌─────────────┴─────────────┐
           │                           │
        按算法分类                   按任务分类
           │                           │
     ┌─────┴─────┐           ┌─────────┴─────────┐
     │           │           │                   │
  On-policy   Off-policy   机器人运控          具身基础模型
     │           │           │                 RL 后训练
     │           │           │                   │
    PPO         SAC     ┌─────┴─────┐      ┌──────┴──────┐
                        │           │      │             │
                    Locomotion  Manipulation  Diffusion   Flow/VLA
                        │           │      │             │
                     行走控制     灵巧操作   DPPO/DSRL   OGPO/Q-VGM
```

其中，机器人操作（Manipulation）既可以采用传统的低层控制策略，也可以采用 Diffusion Policy 或 VLA。

因此，**同一个操作任务可以属于传统机器人控制 RL，也可以属于具身基础模型 RL 后训练，关键取决于策略模型和训练目标。**

<a id="course-section-03"></a>

### 1.2 核心分类表

| 分类维度 | 类别 | 代表算法或方法 | 主要用途 |
|---|---|---|---|
| 算法 | On-policy | PPO | 大规模仿真训练、运动控制 |
| 算法 | Off-policy | SAC | 连续控制、数据高效学习、在线改进 |
| 任务 | Locomotion RL | PPO、SAC | 双足、四足、人形机器人运动 |
| 任务 | Manipulation RL | PPO、SAC | 抓取、手内操作、灵巧控制 |
| 任务 | Foundation Model RL | DPPO、DSRL、OGPO 等 | 改进预训练 Diffusion / Flow / VLA 策略 |

需要注意：DPPO、DSRL 和 OGPO 是针对生成式策略设计的方法，不能全部直接归类为标准 PPO 或 SAC。

---

<a id="course-section-04"></a>

## 二、按算法分类：PPO 与 SAC

<a id="course-section-05"></a>

### 2.1 On-policy RL：PPO

**PPO（Proximal Policy Optimization）** 是机器人运控中广泛使用的强化学习算法。

其主要特点是：

- 使用当前策略采集的交互数据进行训练；
- 每轮采样后对策略进行有限次数的更新；
- 训练通常比较稳定；
- 容易结合 GPU 大规模并行仿真；
- 不适合长期重复使用大量旧数据。

#### PPO 主要用于哪些任务？

| 应用 | 具体任务 |
|---|---|
| 四足机器人 | 行走、奔跑、跳跃、复杂地形穿越 |
| 人形机器人 | 双足行走、平衡、抗扰恢复 |
| Whole-Body Control | 全身协调控制、动作跟踪 |
| 灵巧操作 | 灵巧手旋转物体、抓取、接触操作 |
| Sim-to-Real | 仿真训练控制策略并部署到真机 |
| 生成式策略后训练 | DPPO 等针对 Diffusion Policy 的策略梯度方法 |

#### 为什么运控领域喜欢 PPO？

因为运控任务通常可以在 Isaac Lab 等仿真器中同时运行数千个机器人。

例如：

```text
          4096 个并行机器人
                  │
                  ▼
            采集运动数据
                  │
                  ▼
              PPO 训练
                  │
                  ▼
             更新控制策略
                  │
                  ▼
          继续大规模并行采样
```

在这样的设置下，环境交互数据容易大量获得，因此 PPO 的 On-policy 数据使用方式通常是可以接受的。

**典型技术组合：**

`Isaac Lab + RSL-RL + PPO`

Isaac Lab 官方文档也将 RSL-RL 作为新机器人 RL 项目的一种推荐起点。([Isaac Lab 文档](https://isaac-sim.github.io/IsaacLab/develop/source/concepts/reinforcement_learning.html))

<a id="course-section-06"></a>

### 2.2 Off-policy RL：SAC

**SAC（Soft Actor-Critic）** 是机器人连续控制中具有代表性的 Off-policy 算法。

主要特点：

- 可以重复使用历史交互数据；
- 使用 Replay Buffer 存储经验；
- 通常具有较高的环境交互样本效率；
- 适合在线策略改进；
- 可以利用 Critic 指导动作选择；
- 对 Critic 的学习质量比较敏感。

#### SAC 主要用于哪些任务？

| 应用 | 具体任务 |
|---|---|
| 机械臂控制 | 抓取、放置、推物体 |
| 灵巧操作 | 手内旋转、物体重定位、精密操作 |
| 真实机器人 RL | 利用有限真实交互数据改进策略 |
| Residual RL | 在已有控制器上学习动作修正 |
| Latent RL | 学习预训练生成策略的潜变量 |
| 具身模型后训练 | 基于 Critic 的 Off-policy 策略优化 |

#### 为什么 SAC 适合真实机器人学习？

因为真实机器人交互比较昂贵。

例如，一次灵巧手抓取可能涉及：

- 机械臂移动；
- 手指接触；
- 抓取执行；
- 任务成功或失败判断；
- 机器人复位。

如果每次策略更新都需要重新采集大量数据，训练成本会非常高。

SAC 可以通过 Replay Buffer 反复利用已有经验，因此常用于研究数据高效的机器人在线学习。

但需要强调：**SAC 的样本效率优势并不意味着它在所有机器人任务上都比 PPO 更好。** 特别是在拥有大规模并行仿真资源时，PPO 可能获得更高的训练吞吐量。

<a id="course-section-07"></a>

### 2.3 PPO 与 SAC 的比较

| 维度 | PPO | SAC |
|---|---|---|
| 类型 | On-policy | Off-policy |
| 训练数据 | 主要来自当前策略 | 当前和历史策略数据 |
| 历史数据复用 | 有限 | 较充分 |
| Replay Buffer | 通常不需要长期 Replay Buffer | 需要 |
| 样本效率 | 通常较低 | 通常较高 |
| 并行仿真 | 非常成熟 | 同样可以使用 |
| 训练稳定性 | 通常容易控制 | 依赖 Critic 质量 |
| 真实交互成本高的场景 | 通常不占优势 | 往往更有优势 |
| 典型任务 | Locomotion | Manipulation、Online RL |
| 代表框架 | RSL-RL | Stable-Baselines3、其他 SAC 实现 |

#### 如何选择？

**选择 PPO：**

机器人仿真速度快、可以运行大量并行环境、希望从零训练一个稳定的运动控制策略。

**选择 SAC：**

真实机器人数据昂贵、需要重复利用历史经验、希望持续在线提升策略性能。

不过，实际选择还取决于策略结构。如果策略是大型 Diffusion 或 Flow Matching 模型，就不能不加修改地直接使用标准 PPO/SAC。

---

<a id="course-section-08"></a>

## 三、按任务分类：机器人运动控制强化学习

机器人运动控制强化学习主要目标是：

**让机器人通过与环境交互，学会稳定、高效地执行具体的物理动作。**

通常直接训练一个相对轻量的控制策略，并将其部署到机器人控制系统中。

<a id="course-section-09"></a>

### 3.1 Locomotion RL：机器人运动控制

Locomotion RL 主要针对机器人移动与全身运动能力。

#### 典型应用

| 方向 | 任务 | 主要用途 |
|---|---|---|
| Quadruped Locomotion | 四足行走、奔跑、跳跃 | 机器狗运动控制 |
| Humanoid Locomotion | 双足行走、上下楼梯 | 人形机器人运动 |
| Motion Tracking | 跟踪人类或参考运动 | 人形机器人动作模仿 |
| Whole-Body Control | 全身协调运动 | 移动操作、复杂全身动作 |
| Push Recovery | 外部推力扰动恢复 | 提高机器人稳定性 |
| Terrain Adaptation | 斜坡、碎石、楼梯 | 复杂地形适应 |

#### 典型训练方式

```text
机器人模型
    │
    ▼
构建物理仿真环境
    │
    ▼
定义运动任务与奖励
    │
    ▼
使用 PPO 训练
    │
    ▼
Domain Randomization
    │
    ▼
Sim-to-Real
    │
    ▼
真实机器人部署
```

#### 这一方向主要研究什么？

**1. 运动能力**

让机器人完成越来越复杂的运动任务，例如：

- 从平地行走扩展到楼梯；
- 从慢速行走扩展到奔跑；
- 从简单步态扩展到全身动作跟踪。

**2. 鲁棒性**

提高机器人应对不同环境的能力，例如：

- 外部推力；
- 摩擦变化；
- 未知负载；
- 执行器误差。

**3. Sim-to-Real**

减少仿真与真实机器人的差异。

主要技术包括：

- Domain Randomization；
- System Identification；
- Teacher-Student Learning；
- Privileged Learning；
- Online Adaptation。

#### 典型技术栈

| 模块 | 常见选择 |
|---|---|
| 仿真 | Isaac Lab、MuJoCo |
| RL | PPO |
| 实现 | RSL-RL |
| 控制策略 | MLP、RNN |
| 低层控制 | PD / Torque Control |
| 训练方式 | 大规模并行仿真 |
| 部署 | 机器人实时控制系统 |

**这一方向的特点：**

通常可以在仿真中从零训练，主要瓶颈是物理仿真、奖励设计、任务复杂度以及真实部署的鲁棒性。

<a id="course-section-10"></a>

### 3.2 Manipulation RL：机器人操作强化学习

Manipulation RL 主要针对机械臂和灵巧手操作。

虽然它与 Locomotion RL 都属于机器人控制层面的强化学习，但操作任务经常涉及更复杂的接触动力学。

#### 典型应用

| 方向 | 任务 | 主要用途 |
|---|---|---|
| Reaching | 机械臂到达目标位置 | 基础末端控制 |
| Grasping | 抓取不同物体 | 机器人抓取 |
| In-hand Manipulation | 手内旋转与重定位 | 灵巧手精细操作 |
| Object Reorientation | 调整物体姿态 | 灵巧操作 |
| Tool Manipulation | 使用螺丝刀等工具 | 工具操作 |
| Contact-Rich Manipulation | 插孔、旋转、装配 | 精密接触操作 |
| Bimanual Manipulation | 双臂协同 | 双手操作与装配 |

#### 与 Locomotion RL 的区别

Locomotion 更多关注：

- 身体平衡；
- 速度跟踪；
- 地面接触；
- 周期性运动。

Manipulation 更多关注：

- 多点接触；
- 物体位姿；
- 抓取稳定性；
- 接触切换；
- 手指协调；
- 精密操作。

#### 使用 PPO 还是 SAC？

**PPO：**

适合在仿真中并行训练大量灵巧操作任务。

例如，给定不同物体初始姿态，训练灵巧手完成物体旋转。

**SAC：**

适合研究真实机器人在线学习，以及需要较高环境交互样本效率的操作任务。

例如，在已有初始操作策略的基础上，通过少量真实交互逐步提高抓取成功率。

#### 重要区别

这里训练的可能仍然是传统低层控制策略：

```text
Observation
    ↓
MLP Actor
    ↓
Joint Position / Torque
    ↓
Robot
```

它与下一部分的 VLA 强化学习后训练不是同一个问题设定。

---

<a id="course-section-11"></a>

## 四、按任务分类：具身基础模型强化学习后训练

<a id="course-section-12"></a>

### 4.1 什么是具身基础模型 RL Post-training？

具身基础模型通常先利用大量演示数据进行预训练，再利用强化学习提高任务表现。

例如：

- Diffusion Policy；
- Flow Matching Policy；
- π0 / π0.5 等 VLA；
- 其他视觉语言动作生成模型。

训练过程通常是：

```text
             大规模机器人演示数据
                      │
                      ▼
                Imitation Learning
                      │
                      ▼
               Pretrained Policy
                      │
                      ▼
                RL Post-training
                      │
                      ▼
             Improved Robot Policy
```

它与从零训练 Locomotion Policy 的主要区别是：

**后训练阶段已经有一个具备一定任务能力的预训练策略。**

强化学习的目标不再是从零学习基本动作，而是在已有能力基础上进一步提升性能。

<a id="course-section-13"></a>

### 4.2 RL 后训练主要解决什么问题？

| 问题 | RL 后训练的作用 |
|---|---|
| 成功率不足 | 提高抓取、放置、插孔等任务成功率 |
| 动作速度慢 | 缩短任务完成时间 |
| 精度不足 | 提升末端定位和精密操作能力 |
| 遇到新环境失败 | 在线适应环境变化 |
| 模仿学习性能受限 | 利用自主经验超越演示策略 |
| 失败后无法恢复 | 学习错误恢复和重新尝试 |
| 真实任务缺少演示 | 通过自主探索进一步改进策略 |

例如：

一个预训练 VLA 已经能够完成螺丝刀抓取，但成功率只有 70%。

我们希望通过强化学习，提高成功率、减少不必要的动作，并提高对初始姿态变化的鲁棒性。

<a id="course-section-14"></a>

### 4.3 具身基础模型 RL 后训练的五类方法

按照**强化学习究竟更新模型的哪个部分**，可以分为五类。

#### 第一类：Full Policy Fine-tuning

**直接更新预训练生成策略。**

代表方法：

- DPPO；
- OGPO。

其中 DPPO 通过策略梯度微调 Diffusion Policy；OGPO 则利用 Off-policy Critic 和适配生成过程的策略优化目标，实现对生成式控制策略的完整微调。([DPPO](https://arxiv.org/abs/2409.00588))

**主要用途：**

当预训练策略的能力明显不足，希望真正改变其动作生成能力时，采用完整策略微调。

优点是修改范围大，能力提升空间也可能更大。

局限是训练成本较高，并且需要防止破坏原有的预训练能力。

#### 第二类：Latent / Noise-Space RL

**冻结预训练模型，只优化其输入潜变量或噪声分布。**

代表方法：

- DSRL；
- LP-DS。

DSRL 通过学习 Diffusion Policy 的初始噪声分布实现策略改进，而不需要修改基础策略参数。([DSRL](https://arxiv.org/abs/2506.15799))

**主要用途：**

当预训练模型已经具备较好的动作生成能力，只需要选择更优的动作模式时。

优点是：

- 不需要微调整个大模型；
- 在线训练相对轻量；
- 可以复用预训练策略。

局限是：

无法轻易突破冻结生成器本身的表达能力限制。

#### 第三类：Action Residual RL

**在预训练策略输出的动作上增加 RL 修正。**

例如：

```text
Observation
    │
    ▼
Pretrained VLA
    │
    ▼
Base Action ──────────┐
                      │
Observation           │
    │                 │
    ▼                 │
Residual RL Policy    │
    │                 │
    ▼                 ▼
Residual Action ───► Add
                      │
                      ▼
                Final Action
```

**主要用途：**

预训练策略基本正确，但末端误差、姿态误差或接触控制仍然不够精确。

例如：

- 提高插孔精度；
- 修正抓取姿态；
- 对工具操作增加局部控制；
- 改善接触阶段的动作。

这类方法通常不需要重新训练基础 VLA。

#### 第四类：Feature / Token-Level RL

**利用预训练 VLA 的内部特征，训练轻量 RL 模块。**

代表方法：

- RL Token。

RL Token 通过预训练 VLA 提供紧凑的任务相关表示，并训练轻量 Actor-Critic 模块实现在线策略改进。([RL Token](https://arxiv.org/abs/2604.23073))

**主要用途：**

大型 VLA 的在线强化学习。

例如：

- 精密装配；
- 插孔；
- 工具操作；
- 在线提升操作速度。

这类方法适合希望保留基础模型表征、降低在线训练开销的场景。

#### 第五类：Velocity Field Fine-tuning

**直接优化 Flow Matching Action Expert 的 Velocity Field。**

代表研究：

- Q-VGM；
- Reward / Advantage-Guided Velocity Matching 相关方法。

Q-VGM 使用 Critic 提供的价值梯度信号，通过局部 Velocity Matching 改进预训练 Flow Matching Policy。([Q-VGM](https://arxiv.org/abs/2606.08015))

**主要用途：**

希望真正改进 Flow Matching 动作生成模型，但又不希望每次策略更新都对完整生成轨迹进行昂贵反向传播。

这是一条与大型 Flow Matching VLA 后训练高度相关的研究路线。

<a id="course-section-15"></a>

### 4.4 五类方法的总结

| 方法 | 代表 | 更新内容 | 主要用途 |
|---|---|---|---|
| Full Fine-tuning | DPPO、OGPO | 整个生成策略或主要 Action Expert | 提升模型的动作生成能力 |
| Latent RL | DSRL、LP-DS | 输入噪声或潜变量策略 | 轻量级在线改进 |
| Action Residual RL | Residual SAC 等 | 输出动作残差 | 精密控制、局部动作修正 |
| Feature / Token RL | RL Token | 轻量 Actor-Critic 模块 | 大型 VLA 的高效在线学习 |
| Velocity Field RL | Q-VGM 等 | Flow Matching Velocity Field | 直接改进 Flow Matching Action Expert |

其中，前四类的主要区别是优化参数的位置，而第五类进一步利用了 Flow Matching 的生成结构。

这些类别并非完全互斥。例如，可以只训练 Velocity Field 的残差模块，它同时属于 Residual Adaptation 和 Velocity Field Fine-tuning。

---

<a id="course-section-16"></a>

## 五、PPO、SAC 在两类任务中的具体用途

将两个分类维度组合，可以得到下面这张核心表。

| | 机器人运动控制 | 具身基础模型 RL 后训练 |
|---|---|---|
| **On-policy PPO** | 大规模仿真训练运控策略 | DPPO 等生成策略微调 |
| **Off-policy SAC** | 灵巧操作、真实机器人在线学习 | DSRL 等基于 SAC 或 Off-policy Critic 的适配方法 |

<a id="course-section-17"></a>

### 5.1 PPO + Locomotion

**典型任务：**

人形机器人学会上楼梯。

**常用方式：**

Isaac Lab + RSL-RL + PPO。

**主要优势：**

充分利用大规模仿真环境。

<a id="course-section-18"></a>

### 5.2 SAC + Manipulation

**典型任务：**

机器人通过在线交互学习抓取和物体操作。

**主要优势：**

复用 Replay Buffer 中的经验，减少获得新训练数据的需求。

<a id="course-section-19"></a>

### 5.3 PPO 类方法 + VLA

**典型任务：**

使用 DPPO 对 Diffusion Policy 进行强化学习后训练。

**主要优势：**

直接通过策略梯度改进生成策略。

<a id="course-section-20"></a>

### 5.4 SAC / Off-policy Critic 类方法 + VLA

**典型任务：**

使用 DSRL 优化预训练生成策略的潜变量，或者使用 OGPO、Q-VGM 等方法改进生成策略。

**主要优势：**

利用历史交互经验提高数据效率。

这里尤其需要区分：

OGPO 和 Q-VGM 都属于 Off-policy 生成策略优化路线，但并不等价于标准 SAC。

---

<a id="course-section-21"></a>

## 六、如何根据任务选择 RL 方法？

下面给出实际项目中比较有用的选型参考。

| 项目目标 | 优先考虑 | 原因 |
|---|---|---|
| 从零训练四足机器人行走 | PPO | 并行仿真成熟 |
| 训练 G1 等人形机器人运控 | PPO | 成熟的运动控制 RL 工作流 |
| 训练灵巧手在仿真中旋转物体 | PPO | 可利用并行仿真 |
| 真机灵巧手在线改进操作能力 | SAC 或基于演示初始化的 Off-policy RL | 可以复用历史数据 |
| 微调已有 Diffusion Policy | DPPO、OGPO | 直接改进生成策略 |
| 不修改已有 Diffusion/Flow 模型 | DSRL | 冻结生成模型，优化噪声策略 |
| 提升 VLA 的精密操作能力 | Residual RL、RL Token | 训练轻量模块 |
| 直接微调 Flow Matching 的 Velocity Field | Q-VGM 等 | 利用价值信号更新生成模型 |
| 希望避免完整生成链的反向传播 | Latent RL、Token RL、局部 Velocity Matching | 降低在线优化成本 |

实际选型还要考虑模型大小、仿真是否可用、奖励是否稀疏，以及真实交互和计算资源的预算。

---

<a id="course-section-22"></a>

## 七、两个研究方向的典型实验路线

<a id="course-section-23"></a>

### 7.1 机器人运控 RL

推荐路线：

```text
Isaac Lab / MuJoCo
         │
         ▼
      PPO / SAC
         │
         ▼
    训练运控策略
         │
         ▼
  Domain Randomization
         │
         ▼
    Teacher-Student
         │
         ▼
      Sim-to-Real
         │
         ▼
    真实机器人控制
```

这条路线主要关注：

**如何让机器人运动得更稳定、更灵活，并具备更强的物理环境适应能力。**

推荐入门：

[Isaac Lab](https://github.com/isaac-sim/IsaacLab) + [RSL-RL](https://github.com/leggedrobotics/rsl_rl)

<a id="course-section-24"></a>

### 7.2 具身基础模型 RL 后训练

推荐路线：

```text
      Robot Demonstrations
               │
               ▼
         Pretraining
               │
               ▼
     Diffusion / Flow / VLA
               │
               ▼
         RL Post-training
               │
     ┌─────────┼──────────┐
     │         │          │
     ▼         ▼          ▼
 Latent RL  Residual RL  Full Fine-tuning
     │         │          │
    DSRL    RL Token   DPPO / OGPO
                          │
                          ▼
                    Velocity Field
                       Q-VGM
               │
               ▼
        Improved Robot Policy
               │
               ▼
         Real Robot Deployment
```

这条路线主要关注：

**如何利用强化学习，以较少的机器人交互数据提升预训练基础模型的任务能力。**

推荐入门：

[DSRL](https://github.com/ajwagen/dsrl) → [DPPO](https://diffusion-ppo.github.io/) → [OGPO](https://arxiv.org/abs/2605.03065) → [Q-VGM](https://arxiv.org/abs/2606.08015)

---

<a id="course-section-25"></a>

## 八、最后总结

对于具身智能中的强化学习，建议采用以下三层分类来理解整个领域。

| 分类层次 | 主要类别 | 核心问题 |
|---|---|---|
| **第一层：算法** | PPO、SAC | 如何利用环境反馈改进策略？ |
| **第二层：任务** | Locomotion、Manipulation、Foundation Model Post-training | 强化学习要解决什么机器人任务？ |
| **第三层：优化对象** | 低层控制策略、Residual、Latent、Action Expert、Velocity Field | 强化学习究竟更新策略的哪个部分？ |

其中：

**机器人运控强化学习**以 PPO、大规模物理仿真、运动控制和 Sim-to-Real 为主要技术路线。

**具身基础模型强化学习后训练**以 Diffusion/Flow/VLA 为基础，重点研究如何利用 RL 改进预训练模型，包括 Latent RL、Residual RL、Token-Level RL 和 Full Policy Fine-tuning。

二者最核心的差异在于：

- **运控 RL：** 主要解决机器人如何获得稳定的运动与操作控制能力。
- **基础模型 RL 后训练：** 主要解决如何让已经学会操作的模型表现得更好，并进一步提高成功率、速度、精度和鲁棒性。

因此，具身强化学习并不只是 PPO 和 SAC 的选择问题，还包括**任务类型、策略架构、训练数据来源和策略更新范围**的选择。
