---
title: "具身智能中的世界模型与世界动作模型：训练、推理与部署"
subtitle: "训练目标、生成机制、闭环推理与机器人部署"
author: "Foundation of Embodied AI · 教学讲义"
date: "2026-10-19"
lang: zh-CN
toc: true
toc-depth: 2
toc-title: 目录
colorlinks: true
---

# 具身智能中的世界模型与世界动作模型：训练、推理与部署

[课程首页](https://github.com/zengxiong111/Foundation_of_Embodied_AI/blob/main/README.zh-CN.md) · [English](World_Action_Models_EN.md) · [PDF](https://zengxiong111.github.io/zengxiong.github.io/files/World_Action_Models_ZH.pdf)

**定位：** 本讲承接 Flow Matching、机器人模仿学习、GR00T/SONIC，面向需要理解 WAM 的训练接口、推理路径及真机部署约束的读者。**范围：** 不重复连续性方程、PPO 或基础 Transformer 推导；重点解释 world prediction 与 executable action 之间的连接。

**阅读路径：** 概念与四类架构 $\rightarrow$ 数据表示 $\rightarrow$ 训练目标 $\rightarrow$ 推理模式 $\rightarrow$ 实时系统 $\rightarrow$ 接触与失败恢复 $\rightarrow$ 论文地图。

## 目录

- [1. 为什么具身策略需要世界预测？](#wam-section-01)
- [2. WAM 的四种建模范式](#wam-section-02)
- [3. 训练数据究竟长什么样？](#wam-section-03)
- [4. 训练 WAM：从初始化到联合优化](#wam-section-04)
- [5. WAM 的三条推理路径](#wam-section-05)
- [6. 高效推理：延迟预算比模型参数量更重要](#wam-section-06)
- [7. 灵巧操作：真实世界模型为什么更难？](#wam-section-07)
- [8. 如何评价 WAM 的研究贡献？](#wam-section-08)
- [9. 与现有课程的边界和推荐学习路线](#wam-section-09)
- [10. 论文地图与官方资源](#wam-section-10)

<a id="wam-section-01"></a>

## 1. 为什么具身策略需要世界预测？

机器人动作不是与环境无关的文本 token。抓取前物体的姿态、接触后的滑移、手指遮挡和摩擦变化都会改变执行结果。传统视觉语言动作策略（VLA）可以直接拟合条件动作分布

$$\pi_\theta(A_t\mid h_t,\ell),\quad h_t=(o_{t-k:t},q_{t-k:t}),\quad A_t=(a_t,\ldots,a_{t+H-1}).$$

这里 $o$ 是视觉或其他感知量，$q$ 是本体状态，$\ell$ 是语言目标。**这种条件策略并非必然没有世界知识**：训练数据与表征可能已经隐式编码了动力学。引入世界预测，是为了提供更明确的时序监督、反事实推演或规划接口，而不是断言 VLA 完全不能推理物理。

World Model（WM）通常是行动条件的未来预测器。它可以工作在观测空间或潜在状态空间：

$$p_\phi(o_{t+1:t+H}\mid h_t,A_t),\qquad p_\phi(z_{t+1}\mid z_t,a_t).$$

前一个模型可能合成像素视频，后一个模型预测更适合控制的抽象状态。世界动作模型（WAM）强调将未来预测与机器人**可执行动作**相耦合：可以先想象再行动、借助预测特征生成动作、联合生成视频与动作，或仅在训练时用世界预测作为辅助监督。**WAM 不等同于“推理时必须生成视频”**。

### 1.1 三种容易混淆的模型

| 模型 | 典型输出 | 核心监督 | 部署角色 |
|:--|:--|:--|:--|
| 视觉/语言动作策略（VLA） | 动作块 | 动作标签、任务信号 | 直接动作决策 |
| 世界模型（WM） | 未来状态/观测 | 时序变化、动作条件 | 预测、评估、规划 |
| 世界动作模型（WAM） | 关联未来的动作或联合输出 | 预测与动作的耦合信号 | 可执行策略或其训练途径 |

**边界：** 视频生成基础模型如果没有可靠的动作条件和物理一致性，不能自动称为可用于控制的世界模型；同样，能生成视觉上合理的视频也不保证预测的接触力或运动满足动力学。

<a id="wam-section-02"></a>

## 2. WAM 的四种建模范式

### 2.1 Imagine-then-execute：先预测，再生成动作

将希望发生的未来场景 $\hat O^+$ 作为动作生成的中间条件：

$$\hat O^+\sim p_\phi(O^+\mid h_t,\ell),\qquad A_t\sim\pi_\theta(A\mid h_t,\ell,\hat O^+).$$

在理想情况下，预测的视频编码了物体运动路径与目标状态；动作模型再将这类视觉意图映射成具体机器人控制量。缺点是视频采样成本高，生成的未来也可能不可达或与机器人实际动力学不一致。注意：有些实现还会条件化候选动作，不能把上述无动作条件表达视作所有系统的统一定义。

### 2.2 Video-feature-conditioned：利用视频模型的内部表征

先计算世界模型的潜变量或注意力特征 $f_\phi(h_t,\ell)$，再用动作头输出动作：

$$A_t\sim\pi_\theta(A\mid f_\phi(h_t,\ell),q_t).$$

这种方式可以跳过显式解码整段未来视频。但表征是否真正含有可控制、可接触的信息，需要用交互成功率和分布外恢复测试，而不能仅凭视频生成质量推断。

### 2.3 Joint video-action：联合建模视频与动作

联合模型可写成

$$p_{\theta,\phi}(O^+,A\mid h_t,\ell),$$

或者按链式法则分解为 $p(O^+\mid h_t,\ell)p(A\mid O^+,h_t,\ell)$ 等形式。联合建模可共享表征或去噪骨干；**它不意味着视频 token 和动作 token 必须按同一解码顺序生成**。模型设计应区分共享骨干、双分支、交替预测与不同生成时间尺度。

### 2.4 Auxiliary video prediction：只在训练阶段预测视频

训练时通过未来预测促进更具动力学意义的状态表征；执行时可能只保留动作头：

$$\min_{\theta,\phi}\ \mathcal L_{\rm action}+\lambda\mathcal L_{\rm future},\qquad A_t\leftarrow\pi_\theta(h_t,\ell)\ \text{at test time}.$$

Fast-WAM 提供了重要证据：世界预测的收益可以体现在训练表征上，并不一定要求部署时完成显式视频生成。但**这不是所有任务或模型都适用的定理**。[Fast-WAM](https://arxiv.org/abs/2603.16666)

<a id="wam-section-03"></a>

## 3. 训练数据究竟长什么样？

一个最小的机器人交互序列可以写成

$$\mathcal D_R=\{(o_t,q_t,a_t,o_{t+1},r_t,d_t,\ell,\Delta t)\}_{t=0}^{T-1}.$$

$\Delta t$ 是真实时间间隔，$d_t$ 是 episode 终止标志；动作 $a_t$ 必须标注单位、参考坐标系、控制模式与实际执行时刻。这里 $r_t$ 非监督预训练的必需项；它主要服务于规划、价值评估或 RL。还应区分**策略期望命令**与**机器人实际执行命令**，尤其当存在控制饱和、延迟与低层滤波时。

### 3.1 数据来源与可监督目标

| 数据来源 | 可监督内容 | 典型缺陷 |
|:--|:--|:--|
| 互联网/人类视频 | 视觉演化、物体交互先验 | 缺机器人动作标签、尺度和相机标定 |
| 机器人示教 | 同步的观测—动作序列 | 任务与本体覆盖有限 |
| 仿真 rollout | 完整状态、接触、动作、奖励 | 视觉与接触 sim-to-real gap |
| 在线交互数据 | 失败、恢复、状态分布修正 | 成本高、需安全守护与数据治理 |

混用视频与动作数据时，可针对缺失标签做损失掩码；**不能凭空给无动作视频补上“真实机器人动作”**。若利用逆动力学或轨迹重定向得到伪动作，应记录置信度与对应机器人本体。训练前必须同步视觉时间戳、机械臂状态、执行动作；像素预测与动作标签若存在系统性延迟，模型可能学到错误因果关系。

### 3.2 不可忽略的表征设计

视觉：RGB、深度、视频压缩潜变量或 3D token。机器人状态：关节位置/速度、末端位姿、手部编码、控制模式。动作：$\Delta q$、末端 SE(3) 增量、目标姿态、低层参考或 latent action。物理环境：接触事件、法向力/切向力、对象位姿与不确定性。训练/部署必须保持同一动作语义；高层预测可以与低层安全控制采用不同频率。

<a id="wam-section-04"></a>

## 4. 训练 WAM：从初始化到联合优化

### 4.1 阶段 A：视频/状态预测预训练

如果采用 latent world model，将观测编码为 $z_t=E_\eta(o_t)$，预测未来的时空表征：

$$\hat z_{t+1}=F_\phi(z_{\le t},a_{\le t}),\qquad\mathcal L_z=\sum_t\|\hat z_{t+1}-\operatorname{sg}(E_\eta(o_{t+1}))\|_2^2.$$

这是示意性的 latent regression，实际还可能使用 token cross-entropy、时序对比学习、JEPA 式预测、扩散/流匹配目标；是否 stop-gradient 取决于具体训练设计。若只有视频没有动作，可做非动作条件预测，但不能从中直接识别对任意机器人命令的干预因果效应。

### 4.2 阶段 B：动作专家或条件动作生成器

设专家动作块 $A^*$，从高斯噪声 $\varepsilon$ 出发构造插值 $A_\tau=(1-\tau)\varepsilon+\tau A^*$；一个简单的条件 Flow Matching 目标是

$$\mathcal L_{\rm act}=\mathbb E_{A^*,\varepsilon,\tau}\left[\|v_\theta(A_\tau,\tau,c_t)-(A^*-\varepsilon)\|_2^2\right],$$

其中 $c_t$ 包括视觉/世界特征、机器人状态和指令。这里**生成时间 $\tau$ 与机器人物理时间 $t$ 完全不同**，也不同于关节速度。有关一般条件流匹配的推导请见本课程 Flow Matching 专章，本讲只讨论它和世界预测表征的接口。

### 4.3 阶段 C：耦合世界和动作损失

统一而非唯一的教学写法：

$$\mathcal L=\lambda_a\mathcal L_{\rm act}+\lambda_v\mathcal L_{\rm video}+\lambda_z\mathcal L_{\rm latent}+\lambda_c\mathcal L_{\rm consistency}.$$

$\mathcal L_{\rm video}$ 可是视频扩散噪声/速度回归，$\mathcal L_{\rm consistency}$ 可以约束生成动作与预测的未来状态一致，**但一致性损失需要明确采用已知动力学、可微预测器还是对比目标，并不是不加定义就能直接计算**。若视频与机器人动作数据覆盖不同，训练批次应显式掩码不可观测监督。容易出错的地方包括：仅靠视频损失压制动作学习、world backbone 的灾难性遗忘、跨机器人动作尺度不一致、长时预测误差累积。

### 4.4 阶段 D：任务适配与 RL（可选）

监督微调旨在拟合目标本体与目标任务中的有效动作。RL 后训练则以真实或仿真交互回报更新可训练模块，例如动作专家、残差或独立的高层选择器。世界模型还可提供预测奖励或候选动作评分，但若预测器在分布外状态误差增大，策略会利用模型漏洞（model exploitation）。应分开验证 model-based planning 的有效性和 policy improvement 的实际收益。

### 4.5 训练前需要锁定的实验契约

定义明确的 $o_t$ / $a_t$ / $\ell$、训练 horizon $H$、物理时间戳、摄像机外参、动作归一化统计量、episode 边界、世界预测尺度、训练/验证环境划分，以及 unseen-object、unseen-pose 与 recovery 数据切分。对接触丰富的灵巧操作，仅看视频 PSNR 或 FVD 不足以评价控制能力；必须报告任务成功率、重试次数、接触安全和闭环计算时延。

<a id="wam-section-05"></a>

## 5. WAM 的三条推理路径

### 5.1 显式未来想象

每次获取观测 $h_t$ 后，生成未来视频或 latent rollout $\hat O^+$，再解码 $A_t$；只执行动作块前 $K\le H$ 步并重新观测。优点是未来可解释，可检验不同候选轨迹；代价是视频扩散推理和动作生成可能串行叠加，且“看上去合理”的未来可能不可执行。

### 5.2 联合视频—动作生成

单个条件生成过程输出 $(\hat O^+,\hat A)$。应分别明确采样过程的 token 组织、因果掩码、视觉与动作解码顺序，以及是否能够仅从联合模型截取动作。它不自动等价于显式模型预测控制（MPC）：只有实际进行候选评估、优化/筛选，并执行滚动反馈，才能称为相应的规划机制。

### 5.3 推理时仅保留动作分支

共享世界表征在训练期间受未来预测辅助；部署时计算

$$\hat A_t=G_\theta(h_t,q_t,\ell),$$

可省去昂贵的视频解码。需要报告动作分支仍需多少次 Transformer 前向、多少次去噪/ODE 函数评估，以及是否需要访问未来预测中间状态。Fast-WAM 在其实验设置中展示了 190 ms 级延迟；这**不能推广为任意 WAM 的性能承诺**。[Fast-WAM](https://arxiv.org/abs/2603.16666)

<a id="wam-section-06"></a>

## 6. 高效推理：延迟预算比模型参数量更重要

端到端延迟必须分解为

$$T_{\rm e2e}=T_{\rm sense}+T_{\rm encode}+T_{\rm world}+T_{\rm action}+T_{\rm network}+T_{\rm dispatch}.$$

从机器人控制的角度，关键不只是 GPU token/s，而是感知采集到**动作真正生效**的时间。特别是异步推理中，即使高层动作块生成只需数百毫秒，也必须明确旧动作块如何继续执行、新动作块从哪个物理时间戳开始接管。

### 6.1 可以优化哪里？

| 优化手段 | 降低什么成本 | 主要风险 |
|:--|:--|:--|
| 缩短视频预测或跳过视频解码 | 视频采样延迟 | 失去明确未来检查 |
| 减少 flow/diffusion 步数 | 动作生成函数评估 | 模式覆盖或精度下降 |
| KV Cache（因果模型） | 重复前缀注意力计算 | 图像/状态更新与缓存失效条件 |
| FlashAttention | 注意力内存 IO 与临时张量 | 不必然等于整体推理同比例加速 |
| 跨 denoising/replanning cache | 重复骨干计算 | 时间步变化、输入变化导致陈旧特征 |
| 量化/蒸馏/轻量动作头 | 模型内存与计算 | 接触精度、校准及尾部时延变化 |

**技术边界：** KV Cache 适用于满足特定因果结构的自回归推理，不能未经论证直接搬到双向视频 DiT；跨扩散步缓存也不是普通 KV Cache。FlashAttention 可以提高注意力核效率，但高层分辨率视频生成仍可能主要受其他算子与去噪次数限制。这部分的底层算子推导放在“具身基础模型的高效训练与实时推理”专讲。

### 6.2 物理时钟与动作块

令低层周期 $\Delta t_c$、高层输出长度 $H$、计划执行长度 $K$、生成延迟 $d$ 个控制周期。通常必须满足连续命令覆盖或提供稳定的后备控制：高层在计算新动作块时，底层不能无约束暂停。RTC 的已承诺前缀、动作块 inpainting 和训练时条件化已在模仿学习章节讲过；这里强调 **WAM 视频计算额外引入的延迟和可能的相机—动作时间错位**。

<a id="wam-section-07"></a>

## 7. 灵巧操作：真实世界模型为什么更难？

在抓取、插接、旋转等任务中，视觉结果可能无法唯一识别接触状态：同样的图像可以对应不同的法向力、静摩擦裕度或即将发生的滑移。纯 RGB 视频预测需要面对部分可观测性。因此，世界状态最好纳入关节/电机电流、力/触觉、可见对象位姿及置信度；没有触觉时，应诚实地表示不可观测量的不确定性。

一种任务级状态抽象是

$$z_t=(T_{\rm obj},T_{\rm hand},q_t,\hat c_t,\Sigma_t),$$

其中 $\hat c_t$ 是估计接触模式，$\Sigma_t$ 是状态不确定度。训练可用接触事件辅助损失，但不能假定每条视频天然有可靠接触标签。用于控制的测试应包含自由空间轨迹、初次接触、稳定抓持、滑移、掉落与恢复，而非只测试理想起始位置。

### 7.1 一个贯穿案例：任意位置拿起螺丝刀并旋转

**高层目标**是找到工具并建立合适抓握，再移动到固定作业位姿。世界分支预测“若以某种接近角度抓持，工具是否移动/掉落/碰撞”；动作分支输出末端与手指动作；本体感知闭环低层控制器负责频繁纠偏。若视觉预测给出“已抓住”而电流/位置残差显示打滑，不能继续盲目执行动作块，应触发重观测、保持或恢复策略。

### 7.2 真机部署最低检查清单

1. 明确动作类型、坐标系、频率与安全限幅；在仿真和真机一致。
2. 测量相机曝光、编码、网络、GPU 推理、控制下发及实际执行的时间戳。
3. 统计 $p50/p95/p99$ 延迟而非仅单次最短时间；准备推理超时后备策略。
4. 区分预测成功、动作执行成功和真实任务成功；收集失败恢复轨迹。
5. 将世界模型的不确定度当作风险线索，而不是可靠接触力传感器的替代品。

<a id="wam-section-08"></a>

## 8. 如何评价 WAM 的研究贡献？

| 评价维度 | 建议指标 | 容易误判的地方 |
|:--|:--|:--|
| 世界预测 | 长短时预测、运动一致性、校准 | 视频画质并非物理正确性 |
| 策略能力 | 成功率、跨物体/姿态泛化 | 仅训练分布成功率 |
| 闭环鲁棒性 | 外部扰动、遮挡、失败恢复 | 只报告开环重放 |
| 系统实时性 | 端到端 p95 延迟、控制连续性 | 只测纯 GPU forward |
| 数据效率 | 新本体示教需求、样本成本 | 混淆预训练总数据与适配数据 |
| 安全性 | 碰撞、力矩、接触损伤、超时回退 | 靠预测视频保证安全 |

对于一项新 WAM，至少做四组消融：去掉视频预训练、去掉视频辅助损失、去掉显式测试时未来预测、去掉真实闭环重规划。**这些组件影响不同的能力来源，不能仅用“WAM vs VLA”单项比较来判断因果。**

<a id="wam-section-09"></a>

## 9. 与现有课程的边界和推荐学习路线

本讲不重新推导 Flow Matching 的概率路径与连续性方程；这些见 Flow Matching。BC/DAgger/ACT/RTC 的数据协议见 Imitation Learning。冻结/更新基础生成器的 RL 算法见 RL Post-training。GR00T/SONIC 则重点介绍 System 1–System 0 的动作接口。WAM 本讲专注**世界预测如何帮助动作学习与在线决策**，Robot Agent 再讨论外层任务编排、调用工具、成功验证和重规划。

建议学习顺序：世界预测定义 $\rightarrow$ 四范式 $\rightarrow$ 训练数据契约 $\rightarrow$ 视频/动作损失 $\rightarrow$ 三种推理路径 $\rightarrow$ 延迟与安全预算 $\rightarrow$ 接触/扰动测试。实现不同 WAM 之前，先决定究竟需要可视化的 future imagination，还是只想利用未来监督提升表征。

<a id="wam-section-10"></a>

## 10. 论文地图与官方资源

1. Zhang, Zeng, Zhang. [From World Models to World Action Models: A Concise Tutorial for Robotics](https://arxiv.org/abs/2607.00836), 2026. 概念、表示与四类 WAM 范式。
2. Ye et al. [World Action Models are Zero-shot Policies (DreamZero)](https://arxiv.org/abs/2602.15922), 2026. 视频扩散骨干和联合世界—动作建模。
3. Yuan et al. [Fast-WAM: Do World Action Models Need Test-time Future Imagination?](https://arxiv.org/abs/2603.16666), 2026. 训练时视频监督与测试时视频生成的区分。
4. NVIDIA. [Cosmos](https://github.com/nvidia-cosmos), 视频/世界生成的基础设施入口；它并不是单个可直接部署的机器人动作策略。
5. Physical Intelligence. [openpi](https://github.com/Physical-Intelligence/openpi), 机器人动作专家及 Flow Matching 实现参考。
6. Chi et al. [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137), 2023. 动作块扩散基线。
7. Lipman et al. [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747), 2023. 生成式速度场的数学基础。

**文档说明：** 教学表达式未必是各篇论文的原始损失。具体实验数字和计算系统会随论文版本、硬件与复现条件变化。此教程是课程讲义而非复现报告；未执行模型训练和真实机器人实验。
