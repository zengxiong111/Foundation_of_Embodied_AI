---
title: "基于世界模型的强化学习：从模型学习到可微策略优化"
subtitle: "Dyna、PETS、Dreamer、SHAC 与 SuperMimic | Foundation of Embodied AI"
date: "2026-10-28"
lang: zh-CN
---

# 基于世界模型的强化学习：从模型学习到可微策略优化

[课程首页](https://github.com/zengxiong111/Foundation_of_Embodied_AI) · [English](Model_Based_RL_Differentiable_World_Models_EN.md) · [SuperMimic](https://arxiv.org/abs/2610.28480) · [SHAC](https://arxiv.org/abs/2204.07137)

| 项目 | 说明 |
|---|---|
| 主题 | 基于模型的强化学习 |
| 修订日期 | 2026-10-28 |
| 许可 | CC BY-NC-SA 4.0 · Xiong Zeng and contributors |


> **定位：**本课程原有 PPO、SAC、机器人动力学、MuJoCo 与 Motion Tracking 之间的新增桥梁。以公式和系统接口为中心，不包含未经验证的代码或假想实验结果。

---

## 目录

- [1. 为什么需要 Model-Based RL？](#supplement-course-section-01)
- [2. 世界模型：状态、数据和训练目标](#supplement-course-section-02)
- [3. 世界模型的三种主要用途](#supplement-course-section-03)
- [4. 从 Pathwise Gradient 到 BPTT：数学推导](#supplement-course-section-04)
- [5. Short-Horizon Actor–Critic：为什么不仅用 BPTT？](#supplement-course-section-05)
- [6. SuperMimic：共享学习动力学与运动跟踪](#supplement-course-section-06)
- [7. 面向灵巧操作：接触梯度为什么是研究问题？](#supplement-course-section-07)
- [8. 可复现训练与评估计划](#supplement-course-section-08)
- [9. 学习路径、自测题与参考资料](#supplement-course-section-09)

<a id="supplement-course-section-01"></a>

## 1. 为什么需要 Model-Based RL？

在马尔可夫决策过程中，真实或仿真系统按未知转移核 $P(s_{t+1}\mid s_t,a_t)$ 演化，目标是最大化

$$
J(\pi)=\mathbb E_{P,\pi}\left[\sum_{t=0}^{T-1}\gamma^t r(s_t,a_t)\right].
$$

**Model-Free RL**（PPO、SAC）通常只需观测到的奖励和状态转移来更新策略或价值函数，不要求显式预测下一状态。**Model-Based RL（MBRL）**则学习或使用模型 $\hat P$（或确定性 $f_\omega$），使未来变化可以被想象、评估或优化。

三件事必须区分：**环境模型是否存在、模型是否可微、是否对模型进行策略反向传播**。MPC 可以使用不可微模型通过采样优化；Dyna 可以使用模型数据训练 SAC；SuperMimic 需要可微的学习代理模型来计算 Pathwise Gradient。高保真模拟器本身不一定可微。

### 1.1 机器人问题里的数据瓶颈

真实机器人每个动作都需物理时间，接触冲击和失败重置可能很昂贵。并行仿真可以提高样本吞吐，但高度多样化的全身动作跟踪或灵巧接触仍需大量模拟。一个共享的学习动力学模型可以将多种任务中共用的**机械与接触规律**压缩进参数 $\omega$，让策略更新不必每次都重新向物理仿真器询问完整未来。

这并不意味着学习模型自动比模拟器准确。模型偏差与优化利用（Model Exploitation）必须作为系统设计中的一等风险。

<a id="supplement-course-section-02"></a>

## 2. 世界模型：状态、数据和训练目标

### 2.1 状态、动作、观测和环境上下文

以机器人控制为例，先定义

$$
 s_t=(q_t,\dot q_t,x_t^{obj},v_t^{obj}),\quad
 a_t=\text{applied joint command},\quad
 o_t=h(s_t)+\eta_t.
$$

$x_t^{obj}$ 代表可能存在的物体位姿，$\xi$ 是摩擦系数、质量、执行器增益等每个环境的参数。观测 $o_t$ 与仿真特权状态 $s_t$ **不是同一个变量**。世界模型可以用特权仿真状态拟合，部署后的 Actor 仍必须遵守真实可观测信息约束。

数据集

$$
\mathcal B=\{(s_t,a_t,r_{t+1},s_{t+1},d_{t+1},g_t,\xi)\}
$$

应记录**实际执行的动作**（添加探索噪声、位置限幅之后），并明确 $d$ 是失败终止、成功终止还是时间截断。跨 Episode Reset 拼接的序列不应作为连续动力学窗口。

### 2.2 从单步 MSE 到自回归多步损失

学习动力学使用

$$
\hat s_{t+1}=f_\omega(\hat s_t,a_t,\xi),\qquad \hat s_t=s_t
$$

并最小化单步预测误差。训练多步时，依次使用上一时刻**预测状态**作为下一步模型输入：

$$
\hat s_{t+j+1}=f_\omega(\hat s_{t+j},a_{t+j},\xi),\qquad
\mathcal L_{dyn}=\mathbb E_{\mathcal B}\sum_{j=1}^{K}\|\hat s_{t+j}-s_{t+j}\|_{W_s}^2.
$$

多步目标比单步教师强制输入更能暴露模型的误差累积。对于姿态，欧氏四元数距离会受到 $q$ 与 $-q$ 等价性的影响，常须使用测地距离或一致的旋转参数化。

若额外学习奖励、无噪观测及终止概率，可以构造

$$
\mathcal L_{WM}=\mathcal L_{dyn}
+\lambda_r\mathcal L_{reward}
+\lambda_o\mathcal L_{observation}
+\lambda_d\mathcal L_{termination},
$$

其中终止的预测适合二元交叉熵；某些物理上明确可算的目标可用解析函数，而不一定学习对应网络。

### 2.3 不确定性与模型分布外使用

世界模型在 Replay Buffer 中出现频繁的状态上可很准确，在策略优化推动到的未见状态上却可能错误。因此应同时关注独立测试集多步预测、数据覆盖、Ensemble 分歧及真实仿真验证。PETS 倾向于用概率集成和轨迹采样表达预测不确定性；单个确定性模型更轻量，但应通过模型刷新和保守更新控制漂移。

<a id="supplement-course-section-03"></a>

## 3. 世界模型的三种主要用途

| 路线 | 代表 | 模型提供什么？ | 改进动作的机制 |
|---|---|---|---|
| Dyna / 经验增强 | Dyna、MBPO 风格 | 合成状态转移用于更新 Q/Actor | SAC / TD 等模型无关更新 |
| Online Planning | PETS、MPC、CEM/MPPI | 对候选动作序列预测未来 | 每个控制时刻搜索或采样动作 |
| Differentiable Policy Learning | SHAC、SuperMimic | 对动作及状态可微的预测图 | BPTT 优化策略参数 |
| Latent Imagination | Dreamer、部分 TD-MPC 方案 | 紧凑潜状态和未来预测 | 潜空间价值学习、策略优化或规划 |

这些分类会交叉。**是否使用 CEM**与**是否学习 World Model**是独立选择。CEM 和 MPPI 主要用于搜索动作序列；通过 BPTT 训练一个可执行的前馈策略是另一类计算安排。

### 3.1 CEM 与 MPPI 的典型优化变量

对于 $H$ 步动作块 $A=(a_0,\ldots,a_{H-1})$，规划目标为

$$
A^*=\arg\max_A\mathbb E_{\hat P}\left[\sum_{t=0}^{H-1}\gamma^t r(\hat s_t,a_t)\right].
$$

CEM 更新候选动作分布的均值与方差；MPPI 按轨迹回报对采样动作扰动加权。二者都可以把模型当成黑盒，不必计算 $\partial\hat s_{t+1}/\partial a_t$。在线规划持续调用模型；已训练的策略一般部署更快，但需要通过训练将控制模式“编译”进参数 $\theta$。

<a id="supplement-course-section-04"></a>

## 4. 从 Pathwise Gradient 到 BPTT：数学推导

### 4.1 单步求导

设动作由可微确定性策略给出，且模型参数暂时固定：

$$
a_t=\pi_\theta(h(\hat s_t),g_t),\qquad
\hat s_{t+1}=f_\omega(\hat s_t,a_t,\xi).
$$

令 $D_t=\mathrm d\hat s_t/\mathrm d\theta$，则

$$
\begin{aligned}
\frac{\mathrm da_t}{\mathrm d\theta}
&=\frac{\partial\pi_\theta}{\partial\theta}
 +\frac{\partial\pi_\theta}{\partial o_t}
   \frac{\partial h}{\partial s_t}D_t,\\
D_{t+1}
&=\frac{\partial f_\omega}{\partial s_t}D_t
 +\frac{\partial f_\omega}{\partial a_t}
  \frac{\mathrm da_t}{\mathrm d\theta}.
\end{aligned}
$$

这给出动力学模型对策略参数的**全导数递推**。若起始状态来自 Replay Buffer 且与 $\theta$ 无关，则 $D_0=0$，但从 $t=1$ 开始，后续状态仍受到先前动作的影响。

### 4.2 多步回报与梯度

对于光滑奖励 $\hat r(\hat s_t,a_t)$，忽略终止的简化目标为

$$
\hat J_H(\theta)=\sum_{t=0}^{H-1}\gamma^t\hat r(\hat s_t,a_t)
+\gamma^H V_\psi(\hat s_H).
$$

其梯度为

$$
\frac{\mathrm d\hat J_H}{\mathrm d\theta}=
\sum_{t=0}^{H-1}\gamma^t\left(
\frac{\partial\hat r_t}{\partial s_t}D_t+
\frac{\partial\hat r_t}{\partial a_t}\frac{\mathrm da_t}{\mathrm d\theta}
\right)+\gamma^H\frac{\partial V_\psi}{\partial s_H}D_H.
$$

计算图可直接用自动微分实现这组链式法则。与 PPO 的似然比估计器不同，这个梯度依赖可微模型 Jacobian：**它是近似目标 $\hat J_H$ 的梯度，并不等于真实环境 $J$ 的无偏梯度。**

### 4.3 冻结世界模型但保留梯度通路

算法交替优化两套参数：世界模型 $\omega$ 和策略 $\theta$。优化 $\theta$ 时不更新 $\omega$，但不可阻断 $\partial f_\omega/\partial a$。把整段模型想象放在 `no_grad` 中会切断 BPTT。

### 4.4 终止概率和软存活权重

令 $\hat d_{t+1}\in[0,1]$ 为模型预测的 Episode 终止概率，$\hat c_0=1$ 且

$$
\hat c_{t+1}=\hat c_t(1-\hat d_{t+1}).
$$

通过可微存活权重，可定义（与 SuperMimic 形式一致的）想象目标

$$
\boxed{
\hat J_H(\theta)=\mathbb E\left[
\sum_{t=0}^{H-1}\gamma^t\hat c_t\hat r_{t+1}
+\gamma^H\hat c_H V_\psi(\hat s_H,g_H,\xi)
\right].}
$$

该表达式对终止建模方式、Reward 定义和 Episode 语义敏感，不应不加修改地搬到任意任务。

<a id="supplement-course-section-05"></a>

## 5. Short-Horizon Actor–Critic：为什么不仅用 BPTT？

长时域计算图的 Jacobian 连乘可能导致梯度爆炸或消失；模型状态误差随着步数累积，接触切换使局部梯度不稳定。短视野用 $H$ 限定路径长度，而价值函数估计余下回报：

$$
G_t^{(H)}=\sum_{j=0}^{H-1}\gamma^j r_{t+j+1}+\gamma^H V_\psi(s_{t+H}).
$$

对真实终止后的状态应屏蔽尾项；对时间截断则可能需要保留 bootstrap，这取决于训练定义。Critic 可利用 TD($\lambda$)、GAE 风格的目标学习，但不要将**模型里的回报估计**与**真实仿真评估回报**混淆。

SHAC（Short-Horizon Actor-Critic）通过**可微仿真器**对短轨迹求梯度；SuperMimic 通过**从普通仿真数据训练的可微代理模型**求梯度。两者共享短视野+Critic 原理，却对物理求导提出不同要求。

### 5.1 三种常见失效模式

- **Model Bias**：模型低估滑移或碰撞代价，Actor 学到“钻空子”的动作；
- **Gradient Bias**：状态 MSE 低但 $\partial f_\omega/\partial a$ 的方向不可靠；
- **Distribution Shift**：Actor 更新后访问的状态超出 Buffer 支持，产生虚假高回报。

实用缓解包括周期性重采样、动作正则化、限制想象时域、模型集成不确定性阈值、早停和接触分层验证。

<a id="supplement-course-section-06"></a>

## 6. SuperMimic：共享学习动力学与运动跟踪

原论文 [SuperMimic (2026)](https://arxiv.org/abs/2610.28480) 针对人形机器人全身动作跟踪。与在大量 Motion Clips 上重复 Model-Free PPO 的做法不同，它用**一个共享动力学世界模型**学习通用的机器人响应，在该代理模型上高效训练跟踪策略。

### 6.1 三阶段循环

```text
(1) Real simulator rollouts   : policy + MuJoCo Warp -> transitions -> replay
(2) World-model optimization  : replay windows -> dynamics / reward / observation / done
(3) Actor–critic optimization : sampled real states -> imagined rollouts -> BPTT
    repeat from (1) with the improved policy
```

论文中的常用配置包括 **8 步自回归世界模型训练窗口**、**32 步想象轨迹**和利用 Critic 对想象轨迹尾部的回报进行 bootstrapping。可在一个训练循环中多次交替更新，但世界模型参数应在 Actor-BPTT 阶段保持冻结。

### 6.2 数据合同与模型输出

- 输入：真实仿真状态 $s_t$、实际动作 $a_t$、域随机化上下文 $\xi$；
- 输出：预测下一状态 $\hat s_{t+1}$，以及任务相关的奖励、干净观测和终止预测（部分头可使用解析实现）；
- Goal：参考动作、相位或未来指令 $g_t$ 在 Actor/Critic 中保持时间对齐；
- 训练/部署区分：训练中使用真实仿真状态作为世界模型监督，部署 Actor 不应依赖只有模拟器才可获得的特权变量。

### 6.3 与 SONIC 的比较边界

论文报告：单张 H100 训练约 3 小时后，其回报超过使用 128 张 L40、训练约 7 天的 SONIC 已发布 checkpoint；更长训练在论文的运动集合上进一步降低失败数。**这不是相同硬件、相同训练数据和相同优化配置下的因果消融。** 应分开报告 GPU-hours、墙钟时间、训练交互数与完整评估协议，不能把全部硬件效率差异归因于一种梯度估计器。

### 6.4 原论文的适用边界

原始 SuperMimic 论文展示**人形全身动作跟踪**及真机 Unitree G1 运动，**并未直接验证灵巧手掌内旋转的真机部署**。将该思路用于灵巧手，需单独研究物体动力学、接触事件与视觉/触觉状态估计。

<a id="supplement-course-section-07"></a>

## 7. 面向灵巧操作：接触梯度为什么是研究问题？

标准刚体接触包含非穿透与摩擦约束，局部模式可能在 separation、impact、stick、slip 之间切换。若物理映射在切换处不光滑，使用可微代理模型训练策略是一种有效的工程折中，但**代理梯度近似的是训练数据中的连续响应，并不自动匹配真实模式切换**。

在导数存在的局部，除了验证 $\|f_\omega-F_{sim}\|$，还要评估动作方向导数是否能够预测真实仿真器回报变化；在碰撞附近可使用有限扰动方向性测试，而非强行要求不存在的经典梯度。

**灵巧手世界模型推荐输入**：关节位置/速度、物体位姿/速度、目标方向、动作历史，以及可用的摩擦/执行器上下文；视觉部署时需要区分训练特权状态与估计可用状态。

<a id="supplement-course-section-08"></a>

## 8. 可复现训练与评估计划

| 阶段 | 应测量 | 不通过时的处理 |
|---|---|---|
| A. 仿真器正确性 | 质量惯量、关节 PD、控制频率、接触行为 | 修正 URDF/MJCF/USD 资产与执行器参数 |
| B. Model-Free 基线 | PPO 成功率、仿真交互量、GPU-hours | 保持基线可复现，固定种子 |
| C. 世界模型 | 1/8/32 步误差、接触事件误差 | 采用多步训练、重新平衡接触数据 |
| D. BPTT+Critic | 实际仿真回报增量、梯度范数、H 敏感度 | 降低想象时域或模型优化步数 |
| E. 泛化 | 未见命令、环境参数与初始状态 | 加强数据覆盖与域随机化 |
| F. Sim-to-Real | 物体跟踪延迟、掉落、硬件保护 | 回到辨识/标定，不把成功率当作无条件保证 |

### 8.1 必要消融

1. PPO（不学世界模型） vs 使用同一物理仿真样本预算的 Model-Based RL；
2. `BPTT + Critic` vs `BPTT without Critic`；
3. 想象时域 $H\in\{8,16,32,64\}$，长视野是否真的更好由实验判断；
4. 单步训练 vs 8 步自回归训练；
5. 仅更新 Actor/模型 vs 循环收集新真实仿真数据；
6. 终止建模、Contact-Biased Buffer Sampling、Model Ensemble 是否提升稳健性。

### 8.2 报告规范

请分别报告实际仿真 transitions 与 imagined transitions；模型训练、策略优化与仿真采样的各自用时；GPU 种类和数量；独立随机种子及置信区间。对于接触密集任务，单一平均 Reward 不足以替代掉落率、目标姿态误差和任务成功率。

<a id="supplement-course-section-09"></a>

## 9. 学习路径、自测题与参考资料

**建议顺序：**原课程 #6 Dynamics → #7 Simulation → #9 PPO / #10 SAC → 本教程第 1–5 章 → SuperMimic 第 6 章 → 新增灵巧手专题。想进一步研究在线规划，可转向 PETS/CEM/MPPI；研究视觉模型则结合原课程 WAM。

**自测：**

1. 写出 2 步世界模型的 $\mathrm d\hat s_2/\mathrm d\theta$，说明为什么不能只用当前动作的局部导数。
2. PPO、SHAC 和 SuperMimic 分别依赖哪一个梯度计算图？
3. 为什么一个模型能够 1-step 预测准确，却使策略优化方向变差？
4. 为什么 $H=32$ 并不意味着模型每个样本必须训练连续 32 步？
5. 对于掌内方块旋转，设计三个可直接检验的接触失败指标。

**主要参考**：

- [SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model (2026)](https://arxiv.org/abs/2610.28480)
- [Accelerated Policy Learning with Parallel Differentiable Simulation / SHAC (2022)](https://arxiv.org/abs/2204.07137)
- [PETS: Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models (2018)](https://arxiv.org/abs/1805.12114)
- [Dreamer: Dream to Control (2019)](https://arxiv.org/abs/1912.01603)
- [TD-MPC: Temporal Difference Learning for Model Predictive Control (2022)](https://arxiv.org/abs/2203.04955)
- [Dex-Supermimic: independent dexterous-hand adaptation](https://github.com/CumulusAlpha/Dex-Supermimic)（仅作为社区实现）

**版权与边界：**本课程是对既有文献的独立教学整理，不是 SuperMimic 官方教程；比较和推荐属于教学分析，未暗示执行了新的基准实验。本文原创文字采用 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)；论文、代码及既有课程内容分别遵守各自许可。
