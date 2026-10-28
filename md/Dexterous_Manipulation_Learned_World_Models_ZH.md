---
title: "基于学习世界模型的灵巧操作：以 Wuji Hand2 掌内旋转为例"
subtitle: "从 PPO 基线到 SuperMimic 风格的可微策略学习与 Real2Sim2Real"
date: "2026-10-28"
lang: zh-CN
---

# 基于学习世界模型的灵巧操作：以 Wuji Hand2 掌内旋转为例

[课程首页](https://github.com/zengxiong111/Foundation_of_Embodied_AI) · [English](Dexterous_Manipulation_Learned_World_Models_EN.md) · [SuperMimic](https://arxiv.org/abs/2610.28480) · [社区项目 Dex-Supermimic](https://github.com/CumulusAlpha/Dex-Supermimic)

| 项目 | 说明 |
|---|---|
| 主题 | 灵巧操作世界模型 |
| 修订日期 | 2026-10-28 |
| 许可 | CC BY-NC-SA 4.0 · Xiong Zeng and contributors |


> **定位：**配合新增《基于世界模型的强化学习》，将 SuperMimic 的理念迁移到接触丰富的灵巧手任务。这是一份**研究设计与教学案例**，不是已验证的真机系统，也不把独立社区复现视作原论文官方灵巧手结果。

**目标任务：**Wuji Hand2 将已持有方块旋转到目标朝向；先在 MuJoCo Warp/mjlab 比较 PPO 与学习世界模型+BPTT，再讨论已知物体几何、RGB-D 姿态估计及真实执行器建模下的迁移。

---

## 目录

- [1. 方法迁移：哪些概念能直接保留？](#supplement-course-section-01)
- [2. 任务、状态与观测](#supplement-course-section-02)
- [3. 仿真与执行器合同](#supplement-course-section-03)
- [4. 面向掌内接触的世界模型](#supplement-course-section-04)
- [5. 三条有用的控制基线](#supplement-course-section-05)
- [6. 训练闭环、课程学习与失败恢复](#supplement-course-section-06)
- [7. 可发表的实验矩阵](#supplement-course-section-07)
- [8. Real2Sim2Real：从仿真结果到真实手](#supplement-course-section-08)
- [9. 自测题与参考](#supplement-course-section-09)

<a id="supplement-course-section-01"></a>

## 1. 方法迁移：哪些概念能直接保留？

原始 SuperMimic 研究人形机器人全身动作跟踪，使用真实刚体仿真数据学习共享动力学模型，再在可微代理模型上通过短视野 BPTT 优化策略。**可迁移的是训练范式，不是原始结果。** 灵巧手掌内旋转增加可动刚体、手指接触切换和细粒度姿态目标，因此模型误差更可能集中在抓持、滑动和物体脱落时刻。

- **可直接借鉴**：Dyna 式采样-建模-优化闭环、Replay Buffer、多步模型训练、短视野 BPTT、Critic bootstrapping；
- **必须重新定义**：状态与姿态表示、接触/掉落终止、物体奖励、执行器动力学和随机化；
- **必须独立验证**：仿真成功率、对新物体/目标的泛化、视觉反馈、部署安全和真实样本效率。

已有[社区实现 Dex-Supermimic](https://github.com/CumulusAlpha/Dex-Supermimic) 提供 Wuji Hand 1/2 的方块旋转，以及 Wuji Hand2 转笔任务和方块旋转的 PPO 基线。其 README 描述的是 **mjlab/MuJoCo Warp 仿真**；不能据此宣称完成物理手上的 Sim2Real。

<a id="supplement-course-section-02"></a>

## 2. 任务、状态与观测

### 2.1 推荐的 MDP 合同

令 $n$ 为手的可控关节数，物体位姿 $(p_t^o,R_t^o)$ 属于 $\mathbb R^3\times SO(3)$，速度为 $(v_t^o,\omega_t^o)$：

$$
 s_t=\left(q_t,\dot q_t,p_t^o,R_t^o,v_t^o,\omega_t^o,c_t\right),
 \qquad g_t=R_t^*.
$$

$c_t$ 表示可选接触特征/接触模式；它不一定能被真机直接观测。策略观测可用

$$
 o_t=\left(q_t,\dot q_t,\hat p_t^o,\hat R_t^o,g_t,\{a_{t-j}\}_{j=1}^m\right).
$$

训练 Critic 可使用更完整的仿真特权状态，但部署 Actor 应只使用实机可测量/可估计的量。将特权接触力直接送入 Actor 而部署时没有对应传感器，将造成严重的训练-部署信息不一致。

### 2.2 姿态误差

目标是物体旋转而非掌心位移。若不存在方块对称性的特殊处理，可用

$$
 e_R(t)=\left\|\log\left((R_t^*)^\top R_t^o\right)^\vee\right\|_2\in[0,\pi].
$$

这里 $\log:SO(3)\to\mathfrak{so}(3)$，$\vee$ 取旋转向量。对于几何/视觉纹理对称的方块，应在评分中明确是否认为若干等价朝向为同一目标；只有当目标定义允许时，才可最小化其对称群上的旋转误差。

### 2.3 评价终点

一种可复现成功定义是连续 $M$ 个控制周期同时满足 $e_R<\epsilon_R$、物体未掉落，且不超过速度/力矩上限。$\epsilon_R$、$M$、时限及目标分布必须在对比实验前固定；不能在观察结果后调整成功门槛。

<a id="supplement-course-section-03"></a>

## 3. 仿真与执行器合同

### 3.1 动作、PD 与控制速率

实际硬件通常执行关节目标或电机接口，而不是直接实现理想化无限力矩。对于位置目标，仿真中的执行器示意为

$$
 \tau_t=\operatorname{sat}_{\tau_{\max}}\!\left[K_p(a_t-q_t)-K_d\dot q_t\right].
$$

需要明确 $K_p,K_d$ 是否逐关节不同；此外，应核查每个关节的转子反射惯量（armature）、最大力矩/速度、摩擦、传动比、机械限位、动作滤波、通信延迟及控制频率。**不能从 URDF 名义惯量直接推断每个关节的 armature，亦不能将某一设备的数值复用于另一只手。**

### 3.2 资产的最小验收

1. 无物体时每个关节的阶跃响应、稳定性及扭矩限幅；
2. 指尖/掌面碰撞几何与物体质量、转动惯量、材质和摩擦；
3. 小幅动作下物体不应异常弹跳或穿透；
4. 记录物理步长、控制周期和 world-model 预测时间步；
5. 对齐四元数顺序、轴约定、坐标系和目标角度定义。

<a id="supplement-course-section-04"></a>

## 4. 面向掌内接触的世界模型

### 4.1 学习的动力学映射

构建

$$
 \hat s_{t+1}=f_\omega(\hat s_t,a_t,\xi),
 \quad \hat d_{t+1}=d_\omega(\hat s_t,a_t,g_t),
 \quad \hat r_{t+1}=r(\hat s_t,a_t,g_t,\hat s_{t+1}),
$$

其中 $\xi$ 包含物体质量、摩擦、惯量、手指阻尼和执行器参数；可选择将 Reward 按物理含义解析计算，或者学习对应预测头。对 $SO(3)$ 的模型输出应采用稳定的旋转增量/指数映射，并进行数值检查。

### 4.2 多步损失与接触事件

基础损失可写成

$$
 \mathcal L_{dyn}
 =\sum_{j=1}^{K}\left[
 w_q\|\hat q_{t+j}-q_{t+j}\|^2+
 w_v\|\hat{\dot q}_{t+j}-\dot q_{t+j}\|^2+
 w_p\|\hat p^o_{t+j}-p^o_{t+j}\|^2+
 w_R d_{SO(3)}(\hat R^o_{t+j},R^o_{t+j})^2
 \right].
$$

若模型需要识别接触，可另预测接触模式、掉落概率和滑移事件，但这些标签是否可用取决于仿真/传感。请在 **自由运动、抓持稳定、开始滑移、重新接触、失去抓持** 五类窗口上分别报告多步误差，避免常见的无接触样本淹没关键训练事件。

### 4.3 一个重要的风险：世界模型利用

Actor 通过 BPTT 可能偏好世界模型错误允许的“物体悬浮”“无摩擦瞬转”等行为。世界模型的 MSE 下降不一定意味着最终真实仿真成功率上升。安全的教学实验应该把**每次策略更新之后返回 MuJoCo Warp 评估**当作不可省略的验收环节。

<a id="supplement-course-section-05"></a>

## 5. 三条有用的控制基线

| 方法 | 优化变量 | 优势 | 局限 |
|---|---|---|---|
| PPO | 策略参数，使用仿真回报 | 成熟、对不可微接触友好 | 需大量仿真采样 |
| CEM / MPPI | 当前规划窗口中的动作序列 | 不需要模型输入导数，可利用采样探索多峰动作 | 每个控制步都要多次预测，算力与延迟需测量 |
| SuperMimic-style BPTT | 策略参数，通过学习模型求梯度 | 模型想象快，部署时可只执行策略 | 需要正确的代理梯度和防模型利用机制 |

**公平比较原则：**用同一只手、同一方块、同一目标分布、同一执行器接口、同一奖励和终止条件。对 CEM/MPPI 报告测试时规划计算量；对 PPO/BPTT 报告训练期 GPU-hours 和物理仿真交互量。不要用 CEM 的测试时成本与 BPTT 的训练成本直接比较。

### 5.1 训练奖励构成（示意，非论文复现参数）

$$
 r_t=w_R\exp(-k_Re_R(t))
 +w_h\mathbf1_{\text{object held}}
 -w_a\|a_t-a_{t-1}\|^2
 -w_\tau\|\tau_t\|^2
 -w_d\mathbf1_{\text{drop}}.
$$

为防止奖励黑客，需要确认目标姿态误差下降确实发生在物体稳定握持的条件下。**以上权重只是符号占位，不表示已有项目采用该组参数。**

### 5.2 BPTT 目标

在长度 $H$ 的想象轨迹上优化

$$
 \hat J_H(\theta)=\mathbb E\left[
 \sum_{t=0}^{H-1}\gamma^t\hat c_t\hat r_{t+1}
 +\gamma^H\hat c_H V_\psi(\hat s_H,g_H,\xi)
 \right].
$$

模型参数 $\omega$ 在策略阶段冻结、梯度流经 $f_\omega$ 的输入和输出。$H$ 应按**接触复杂性与模型精度**通过消融确定，不应默认为 32 永远最佳。

<a id="supplement-course-section-06"></a>

## 6. 训练闭环、课程学习与失败恢复

### 6.1 建议训练流程

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

### 6.2 Curriculum（建议，不是现成结果）

按难度逐步安排：**稳定握持 → 小角度旋转 → 大角度旋转 → 随机目标方向 → 摩擦/负载扰动 → 短时丢失接触后的恢复**。初始状态可从稳定抓持、多指接触边界和失败附近分层采样。若要采用 RSI（Reference State Initialization），应只在物理可实现的状态上重置，避免把插值关节轨迹误当作有效物体接触状态。

### 6.3 失败恢复不是奖励设计一项能解决的

推荐在真实仿真中显式测试抓持松动、瞬间滑移、接触重建、姿态估计跳变与执行器扰动。策略训练中的短视野值函数应为恢复操作保留长期价值信号。部署系统还必须有动作限幅、硬件急停、掉落检测和安全姿态。

<a id="supplement-course-section-07"></a>

## 7. 可发表的实验矩阵

### 7.1 核心对比

| 实验组 | 世界模型 | Actor 更新 | Critic | 说明 |
|---|---|---|---|---|
| A: PPO | 无 | PPO | PPO Value | 共同任务基线 |
| B: BPTT-no-V | 有 | BPTT | 无 | 检查短视野导致的短视 |
| C: BPTT+V | 有 | BPTT | 有 | 主要方案 |
| D: BPTT+V+contact sampling | 有 | BPTT | 有 | 测试关键接触数据采样 |
| E: CEM/MPPI (可选) | 有 | 动作序列规划 | 可选 | 独立比较测试时算力 |

### 7.2 关键消融与数据效率

建议 $H\in\{8,16,32,64\}$、$K\in\{1,4,8,16\}$；但矩阵应按硬件预算缩小，并至少固定相同模拟器总交互数、训练 GPU-hours 和评估策略。不同方法分别绘制“成功率 vs 真实仿真交互量”和“成功率 vs GPU-hours”；不要把模型想象产生的 transitions 计入真实仿真交互总数。

### 7.3 评价指标

- **任务指标**：成功率、达到目标的时间、SO(3) 朝向误差、掉落/失稳率、失败后恢复率；
- **物理指标**：滑移次数、接触模式误判、力矩/速度越界、动作加速度/抖动；
- **模型指标**：1/8/32 步预测误差、接触条件下的误差、模型预测的改进与真实仿真改进是否一致；
- **系统指标**：GPU-hours、物理仿真 FPS、想象 FPS、策略推理延迟、总真实模拟交互数。

<a id="supplement-course-section-08"></a>

## 8. Real2Sim2Real：从仿真结果到真实手

### 8.1 首先解决动力学辨识

需要估计/检查每个关节的响应时延、PD 增益、摩擦、armature、背驱性、限速、力矩/电流上限；物体端则需质量、质心、惯量、摩擦与碰撞几何。先在有控制安全边界的前提下测量阶跃响应与接触滑移，再决定域随机化范围，而不是随机选“经验范围”。

### 8.2 部署观测需要闭环估计

在有已知物体几何和 RGB-D 摄像头时，可用 FoundationPose 等 6D 位姿估计方法获取 $\hat p_t^o,\hat R_t^o$，并进行短期滤波、遮挡置信度控制和失败重初始化。训练中应向 Actor 注入符合实测分布的估计噪声、延迟和丢帧，**不可把模拟器真值姿态当成真实相机实时观测**。

### 8.3 真实物理试验安全边界

模型和训练策略的输出必须经过关节位置/速度/力矩限制，动作变化率限制以及急停逻辑。对将被丢落的物体设置缓冲区域；遇到无可靠状态估计、接触异常或手指过载应停止或回退，而非让学习策略持续试错。真实成功率必须单独报告，不能由仿真演示推断。

<a id="supplement-course-section-09"></a>

## 9. 自测题与参考

**五个自测题**：

1. 为什么方块朝向误差最好在 $SO(3)$ 上定义，而不是简单使用四元数欧氏误差？
2. BPTT 训练时世界模型权重冻结，是否意味着 $\partial\hat s_{t+1}/\partial a_t=0$？
3. 掌内操作中哪些状态属于真实不可直接观测的特权信息？
4. 如何检验策略是不是利用了错误的摩擦/接触预测？
5. 为什么对比 PPO 和 BPTT 需要同时报告样本数与 GPU-hours？

**阅读建议**：原课程 #6 Dynamics → #7 Isaac/MuJoCo → #8 Motion Control RL → #9 PPO → 新《Model-Based RL》→ 本案例。

参考链接：

- [SuperMimic（原始人形动作跟踪方法）](https://arxiv.org/abs/2610.28480)
- [SHAC（可微仿真中的短视野 Actor–Critic）](https://arxiv.org/abs/2204.07137)
- [Dex-Supermimic（独立社区灵巧手复现）](https://github.com/CumulusAlpha/Dex-Supermimic)
- [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp)
- [mjlab](https://github.com/mujocolab/mjlab)
- [FoundationPose](https://github.com/NVlabs/FoundationPose)

**声明：**这是教学与研究方案，不是对具体灵巧手硬件参数、算法效果或真实部署成功率的实测结论。新增原创文字使用 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)；第三方论文、资产、代码和已有课程内容保留原许可。
