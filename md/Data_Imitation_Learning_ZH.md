---
title: 数据采集与模仿学习
author: Foundation of Embodied AI · 教学讲义
subtitle: 从示教数据到实时闭环动作策略
lang: zh-CN
date: 2026-10-18
---

# 数据采集与模仿学习

<!-- course-meta:start -->
[课程首页](https://github.com/zengxiong111/Foundation_of_Embodied_AI/blob/main/README.zh-CN.md) · [English](https://github.com/zengxiong111/Foundation_of_Embodied_AI/blob/main/md/Data_Imitation_Learning_EN.md) · [PDF](https://zengxiong111.github.io/zengxiong.github.io/files/Data_Imitation_Learning_ZH.pdf)

| 项目 | 说明 |
|---|---|
| 适用范围 | 示教、数据集、BC、DAgger、ACT、生成式动作块与 RTC 概念；不规定统一且经过运行测试的软件环境 |
| 文档核验日期 | 2026-10-18 |
| 许可 | CC BY 4.0 · Xiong Zeng and contributors |

**核验边界：** 仅进行文档整合与排版检查；未进行教程环境、训练或硬件验证。 版本与固定源码引用见[版本说明](https://github.com/zengxiong111/Foundation_of_Embodied_AI/blob/main/VERSIONS.zh-CN.md)。原创教材采用 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)，第三方引用、图片及代码遵循其原有许可；注明来源不等于拥有再许可权。
<!-- course-meta:end -->

<!-- course-toc:start -->
## 目录

- [1. 机器人示教与数据采集](#course-section-01)
  - [1.1 从轨迹到学习问题](#course-section-02)
  - [1.2 主要采集路径与适用范围](#course-section-03)
  - [1.3 一条数据至少应记录什么](#course-section-04)
- [2. 数据表示、同步与质量控制](#course-section-05)
  - [2.1 时钟与因果性](#course-section-06)
  - [2.2 坐标系与控制模式](#course-section-07)
  - [2.3 数据集组织与质量门控](#course-section-08)
- [3. 模仿学习：从行为克隆到 DAgger](#course-section-09)
  - [3.1 模仿学习究竟在学习什么](#course-section-10)
  - [3.2 行为克隆 BC：监督学习目标如何选](#course-section-11)
  - [3.3 为什么均方误差可能学到错误的动作](#course-section-12)
  - [3.4 分布偏移与误差累积：为什么离线准确率不等于成功率](#course-section-13)
  - [3.5 DAgger：逐轮数据聚合的完整逻辑](#course-section-14)
  - [3.6 DAgger 的 no-regret 视角与保证边界](#course-section-15)
  - [3.7 DAgger 在真实机器人上的三种专家接口](#course-section-16)
  - [3.8 BC、DAgger 与生成式模仿学习的关系](#course-section-17)
- [4. 动作块、ACT、Diffusion 与 Flow Matching](#course-section-18)
  - [4.1 为什么输出一个动作序列](#course-section-19)
  - [4.2 ACT：从时序动作块到条件 VAE 与 Transformer](#course-section-20)
  - [4.3 Diffusion Policy：迭代去噪](#course-section-21)
  - [4.4 Flow Matching：学习生成路径上的速度场](#course-section-22)
  - [4.5 方法选择应围绕接口与计算约束](#course-section-23)
- [5. 实时动作块执行与 RTC](#course-section-24)
  - [5.1 两条时间轴：机器人物理时间与模型推理时间](#course-section-25)
  - [5.2 同步执行、朴素异步、Temporal Ensembling 与 RTC 的区别](#course-section-26)
  - [5.3 RTC 的几何关系：承诺前缀与自由后缀](#course-section-27)
  - [5.4 推理时 RTC：将新动作块生成看成 Inpainting](#course-section-28)
  - [5.5 训练时 RTC：把动作前缀直接变成模型条件](#course-section-29)
  - [5.6 RTC 系统实际需要哪些运行时组件](#course-section-30)
  - [5.7 RTC 不解决什么：接触、状态估计与分布偏移](#course-section-31)
  - [5.8 本章核心辨析](#course-section-32)
- [6. 数据增强、重定向与失败纠错](#course-section-33)
  - [6.1 如何构造物理上合法的增强](#course-section-34)
  - [6.2 人类视频重定向的关键缺口](#course-section-35)
  - [6.3 失败样本如何进入数据闭环](#course-section-36)
- [7. 真实部署接口与课程衔接](#course-section-37)
  - [7.1 模块边界](#course-section-38)
  - [7.2 常见失效与定位方向](#course-section-39)
  - [7.3 与课程其他部分的分工](#course-section-40)
- [参考文献与延伸阅读](#course-section-41)
<!-- course-toc:end -->

> **范围与定位**　本文聚焦机器人示教、数据表示、模仿学习与实时执行的技术原理，不设置实验章节或实验操作步骤。机器人强化学习、相机标定及 Flow Matching 的深入理论分别交由对应专题课程展开。
>
> **阅读路线**　采集 → 对齐与质量控制 → BC 的目标与分布偏移 → DAgger 的数据聚合 → ACT 与生成式动作块 → RTC 的时间对齐和前缀条件化 → 失败纠错 → 部署接口。

<a id="course-section-01"></a>

## 1. 机器人示教与数据采集

<a id="course-section-02"></a>

### 1.1 从轨迹到学习问题

机器人的模仿学习不是逐帧复制视频，而是利用专家访问过的状态与选择的动作来学习条件策略：

$$
\pi_\theta(a_t\mid o_{\le t},q_{\le t},g),
$$

其中 $o_t$ 为相机等外感受观测，$q_t$ 为本体感知，$g$ 为任务条件，$a_t$ 为在指定控制接口下执行的动作。真实系统通常满足 $s_{t+1}\sim P(\cdot\mid s_t,u_t)$、$o_t\sim O(\cdot\mid s_t)$。**状态 $s_t$、观测 $o_t$、模型输出 $a_t$、实际下发命令 $u_t$ 与执行反馈不能混用**。安全滤波、限幅和网络延迟可能使 $u_t\ne a_t$。

一个 episode 可表述为

$$
\tau_i=\{(t_k,o_k,q_k,a_k,u_k,\hat q_{k+1},g,e_k)\}_{k=0}^{T_i-1},
$$

其中 $e_k$ 是接触、滑移、接管、失败或恢复等事件标签。任务成功标签属于 episode 层，而不应替代每步动作标签。

<a id="course-section-03"></a>

### 1.2 主要采集路径与适用范围

| 来源 | 动作标签 | 优势 | 局限 |
|:--|:--|:--|:--|
| 关节级拖动示教 | 关节位置或速度目标 | 接近目标机器人的运动学 | 接触力难以一致，示教者负担较大 |
| 主从遥操作（如 GELLO、ALOHA） | 从机控制命令 | 可采真实接触和复杂序列 | 主从映射、延迟及工作空间约束 |
| 手持式通用示教（如 UMI） | 工具/末端运动与视觉 | 可在不同场景采集 | 跨平台运动学和夹爪映射需要处理 |
| 仿真专家与脚本 | 仿真控制动作 | 易规模化且可自动标注 | 仿真到真实的视觉、接触差异 |
| 人类视频重建与重定向 | 推断的人体或物体轨迹 | 无需直接控制目标机器人 | 动作不完全可观测，接触推断不确定 |

**代表性资料**：[GELLO](https://arxiv.org/abs/2309.13037)、[ALOHA / ACT](https://arxiv.org/abs/2304.13705)、[UMI](https://arxiv.org/abs/2402.10329)。这些系统的采集硬件与动作语义不同；不能因为最后都存成数组，就认为训练标签等价。

<a id="course-section-04"></a>

### 1.3 一条数据至少应记录什么

推荐以 episode 为基本单位保存：任务描述、相机序列和各自时间戳、关节角与关节速度、末端位姿、控制模式、模型预测动作、下发命令、实测反馈、夹爪/灵巧手状态、标定版本、设备配置、成功/失败与事件标记。涉及力控时还应记录力矩、力/触觉的量程及采样频率。

动作的语义必须显式标注，例如 $a_t=q^{\rm target}$、$\Delta x^{\rm EE}$、$\dot q^{\rm cmd}$ 或 $\tau^{\rm cmd}$。同一个数值数组若在训练端被理解为“增量位置”、部署端被解释为“绝对位置”，会造成系统性失败。

<a id="course-section-05"></a>

## 2. 数据表示、同步与质量控制

<a id="course-section-06"></a>

### 2.1 时钟与因果性

多模态频率往往不同，例如图像、机器人反馈和触觉存在采样差异。使用统一时间基准或估计时钟偏移，明确曝光时间、接收时间、决策时间和实际执行时间。为时刻 $t$ 的动作构造观测时，原则上只能使用当时**已经可用**的传感器信息：

$$
\tilde o_t=o_{k^*(t)},\qquad k^*(t)=\max\{k:t_k^{\rm available}\le t\}.
$$

用未来帧插值过去时刻的输入会造成训练中的信息泄漏。延迟应进入数据定义而非被不透明的重采样抹去。

<a id="course-section-07"></a>

### 2.2 坐标系与控制模式

统一描述 $T^A_B\in SE(3)$：将 B 系表示的点变换到 A 系。为每个姿态或速度附带参考坐标系与单位。视觉学习尤其需要区分相机系、机器人基座系、末端系、任务系。对角速度、姿态差分及 SE(3) 增量，还需约定左乘还是右乘。**相机外参是常量还是随关节变化**也应写入元数据。

连续动作常用训练集统计量归一化：

$$
\bar a_j=\frac{a_j-\mu_j}{\sigma_j+\epsilon},\qquad
 a_j=\bar a_j(\sigma_j+\epsilon)+\mu_j.
$$

只在训练集统计 $\mu_j,\sigma_j$，保存版本，并对具有严格物理上下界的变量单独处理。绝对位置、角度、旋转表示和二值夹爪命令不应不加区分地使用同一种归一化方案。

<a id="course-section-08"></a>

### 2.3 数据集组织与质量门控

数据结构至少包括 episode 元数据、视频索引、标量/向量时序、动作字段和版本信息。可参考 [LeRobot Dataset](https://huggingface.co/docs/lerobot/lerobot-dataset-v3) 的 episode 与视频组织方式；但具体格式版本可能演进，长期保存时仍须冻结 schema 与依赖版本。

筛选重点包括：时间单调且缺帧可辨识；动作与执行反馈可区分；关节限制与异常速度可检测；遮挡与估计失效有标记；任务标签和事件边界有定义；重复片段与空操作不过度占比。保留具有纠错价值的失败样本，但剔除通信异常造成的错误标签。

数据划分应按**场景、物体实例、示教者、初始姿态或采集日期**隔离，避免相邻视频帧跨集合带来的虚高离线指标。随机按帧拆分不能验证真正的泛化。

<a id="course-section-09"></a>

## 3. 模仿学习：从行为克隆到 DAgger

<a id="course-section-10"></a>

### 3.1 模仿学习究竟在学习什么

模仿学习（Imitation Learning, IL）希望直接从专家行为中得到可闭环执行的策略，而不要求先设计任务奖励或求解完整的环境动力学模型。用隐状态 $s_t$、可观测历史 $h_t=(o_{\le t},q_{\le t},g)$ 和执行动作 $u_t$ 描述系统：

$$
s_{t+1}\sim P(\cdot\mid s_t,u_t),\qquad
h_t\sim \mathcal O(\cdot\mid s_{0:t},u_{0:t-1}),\qquad
a_t\sim\pi_\theta(\cdot\mid h_t).
$$

这里的 $\pi_\theta$ 可以是单步策略，也可以一次输出多个动作。由于相机遮挡、接触隐状态和观测延迟，仅用当前一帧 $o_t$ 未必满足马尔可夫性；历史窗口、机器人本体反馈或状态估计可以降低部分可观测性，但不能保证恢复真实隐状态。

定义长度为 $T$ 的有限时域状态占据分布

$$
d_t^\pi(s)=\Pr(s_t=s\mid\pi),\qquad
d^\pi(s)=\frac1T\sum_{t=0}^{T-1}d_t^\pi(s).
$$

**核心挑战**是：示教数据覆盖专家策略的分布 $d^{\pi_E}$，而部署时策略实际遇到的是 $d^{\pi_\theta}$。BC 主要解决“如何拟合专家动作”，DAgger 进一步解决“遇到自己造成的新状态时该如何行动”。

模仿学习还可区分为：**离线模仿**（固定数据集，典型为 BC）、**交互式模仿**（持续访问专家标注，典型为 DAgger）和**从偏好/轨迹反推奖励**（逆强化学习，属于另一条路线）。本讲义着重前两者；生成式策略的 ACT、Diffusion Policy 和 Flow Matching 本质上也可以使用 BC 监督来训练。

<a id="course-section-11"></a>

### 3.2 行为克隆 BC：监督学习目标如何选

设 $\mathcal D_E=\{(h_i,a_i^E)\}_{i=1}^N$。最简单的确定性行为克隆为

$$
\pi_\theta^*=\arg\min_\theta\frac1N\sum_{i=1}^N
\|\pi_\theta(h_i)-a_i^E\|_W^2,
\qquad \|x\|_W^2=x^\top Wx,\ W\succeq0.
$$

它适合动作在条件 $h$ 下接近单峰且噪声较小的情形；$W$ 可以平衡位置、角度与末端控制维度，但**不能代替正确的物理单位和动作归一化**。平方误差所对应的是一种固定协方差的高斯条件似然。若选择

$$
p_\theta(a\mid h)=\mathcal N\big(a;\mu_\theta(h),\Sigma\big),
$$

则忽略与参数无关的常数，负对数似然为

$$
\mathcal L_{\rm NLL}=
\mathbb E_{\mathcal D_E}\!\left[
\tfrac12(a^E-\mu_\theta(h))^\top\Sigma^{-1}(a^E-\mu_\theta(h))
\right].
$$

若让网络同时预测协方差，还须保留 $\log\det\Sigma_\theta(h)$ 项，以免只靠增大预测方差降低损失。$L_1$ 损失可对应拉普拉斯噪声假设；离散夹爪模式可用交叉熵，但夹爪的开合位置连续控制时不应被错误地二值化。

概率 BC 也可以写成

$$
\min_\theta\ \mathbb E_{h\sim d^{\pi_E}}
D_{\rm KL}\!\left(\pi_E(\cdot\mid h)\|\pi_\theta(\cdot\mid h)\right).
$$

这是专家动作条件分布可被采样且模型表达充分时的总体目标；有限数据实现通常采用 $-\log p_\theta(a^E\mid h)$。它**不等于**在学习策略自身状态分布上的误差最小化。

实际训练还需注意三点：第一，归一化统计量只能由训练集计算；第二，动作、图像和机器人状态应当按因果时间戳配对；第三，若轨迹末尾不足固定窗口长度，应该设置有效时间步掩码 $m_{i,k}\in\{0,1\}$：

$$
\mathcal L_{\rm masked}=
\frac{\sum_{i,k}m_{i,k}\,\ell(\hat a_{i,k},a^E_{i,k})}
{\sum_{i,k}m_{i,k}+\varepsilon}.
$$

否则用零填充却不屏蔽末尾，策略会把“停止输出零动作”当成真实示教。

<a id="course-section-12"></a>

### 3.3 为什么均方误差可能学到错误的动作

当同一观测对应多种合理的专家选择（例如向左或向右绕过障碍物），平方误差的最优预测是条件均值：

$$
f^*(h)=\mathbb E[a^E\mid h].
$$

条件均值却可能恰好落在两条可行轨迹之间的障碍物上。这是**多模态动作分布与单峰回归的结构性不匹配**，不是简单增大模型参数量就能解决的问题。

改善方法可以是：补充任务指令或历史，让不同意图可辨；使用混合密度模型或潜变量动作块；采用 Diffusion Policy、Flow Matching 等生成式行为克隆。另一方面，生成式模型也不会自动修复示教中错误的动作语义、跨模态时间错位或欠覆盖的接触状态。

<a id="course-section-13"></a>

### 3.4 分布偏移与误差累积：为什么离线准确率不等于成功率

假设每一步在专家访问的状态上，学习策略做出一次“关键错误”的概率至多为 $\epsilon$。如果第一次错误可能把系统带离专家轨迹，那么到第 $t$ 步之前出现过错误的概率可以用并集界粗略控制为 $O(t\epsilon)$；将各步风险累加，得到经典的最坏情形标度

$$
\sum_{t=1}^{T}O(t\epsilon)=O(T^2\epsilon).
$$

对比之下，如果在学习策略**自己访问的状态**上仍能保证每步误差为 $\epsilon$，典型的累计误差标度可改善为 $O(T\epsilon)$。这只是有限时域、损失有界、专家比较和状态分布等前提下的解释性结果，**不能**把连续机器人系统的实际成功率严格写成 $1-T^2\epsilon$，也不能把连续动作 MSE 直接当成离散错误概率。

一个直观链条是：抓取点略微偏移 $\rightarrow$ 指尖接触改变 $\rightarrow$ 物体发生滑移 $\rightarrow$ 图像和关节状态离开示教数据分布 $\rightarrow$ 后续动作进一步偏离。更多专家演示、扰动后的恢复示教以及 DAgger 都试图增加这些非典型状态附近的正确监督。

<a id="course-section-14"></a>

### 3.5 DAgger：逐轮数据聚合的完整逻辑

DAgger（Dataset Aggregation）并非一种新的神经网络，而是将**策略引发的状态分布**引入专家监督的训练协议。第 $k$ 轮有当前策略 $\pi_k$ 和累积数据 $\mathcal D_k$，其基本过程为：

1. 用当前策略 $\pi_k$（必要时与专家策略混合）在环境中访问状态或历史 $h_t$。
2. 对访问到的每个被选中样本，查询专家希望采取的动作 $a_t^E=\pi_E(h_t)$；**专家标签不等于机器人当时实际执行的动作**。
3. 将 $(h_t,a_t^E)$ 加入聚合数据集，而不是只保存成功轨迹或只保存失败截图。
4. 在 $\mathcal D_{k+1}$ 上训练得到新策略 $\pi_{k+1}$，继续进行下一轮。

形式上，

$$
\begin{aligned}
\pi_k^{\rm roll}&=\beta_k\pi_E+(1-\beta_k)\pi_k,\\
\mathcal D_{k+1}&=\mathcal D_k\cup
\big\{(h_t,\pi_E(h_t)):h_t\sim d^{\pi_k^{\rm roll}}\big\},\\
\pi_{k+1}&\approx\arg\min_{\pi\in\Pi}
\sum_{(h,a^E)\in\mathcal D_{k+1}}\ell(\pi(h),a^E).
\end{aligned}
$$

混合式中的 $\beta_k$ 表示**按时间步或回合选择专家行为的概率**，不是动作向量数值上做线性插值；后者在接触、抓取等离散模式切换时可能产生危险命令。经典分析要求随着轮数增加，让数据分布趋向学习策略自己诱导的分布；混合日程和专家查询机制应明确记录。

DAgger 与普通“收集更多示教”的根本差别是：**选择哪些状态由当前学习策略决定，而希望采取的动作仍由专家提供**。单纯把失败动作也加入 BC 数据集，并不能构成 DAgger。

<a id="course-section-15"></a>

### 3.6 DAgger 的 no-regret 视角与保证边界

定义第 $k$ 轮在访问分布上的监督风险

$$
L_k(\pi)=\mathbb E_{h\sim d^{\pi_k}}[\ell(\pi(h),\pi_E(h))].
$$

如果监督学习器在连续变化的数据分布上具有较小在线遗憾值

$$
R_N=\sum_{k=1}^{N}L_k(\pi_k)
-\min_{\pi\in\Pi}\sum_{k=1}^{N}L_k(\pi),
$$

则有平均监督风险关系

$$
\frac1N\sum_{k=1}^{N}L_k(\pi_k)
\leq \epsilon_N+\frac{R_N}{N},\quad
\epsilon_N=\min_{\pi\in\Pi}\frac1N\sum_{k=1}^{N}L_k(\pi).
$$

在适当的有界损失、专家代价到达关系、混合收敛和策略选择条件下，可把这种**自身分布上的小监督误差**转化成较好的顺序决策性能，避免朴素 BC 的最坏情形二次误差放大。这里的 $\epsilon_N$ 是策略类别对历轮状态分布的最佳拟合误差；如果专家动作不可观测、标注不一致、模型容量不足或真实接触代价不受控，理论前提就可能失效。不要把 no-regret 结果理解成“DAgger 一定比 BC 成功率高”或“任意机械臂任务都达到统一 $O(T\epsilon)$ 的保证”。

<a id="course-section-16"></a>

### 3.7 DAgger 在真实机器人上的三种专家接口

| 专家接口 | 可获得的监督 | 需要格外注意 |
|:--|:--|:--|
| 离线重标注 | 学习策略访问状态的专家动作 | 录像未必包含足够信息恢复接触隐状态 |
| 实时专家接管 | 接管后的安全纠错序列 | 接管前的错误动作不能自动当成专家标签 |
| 策略执行 + 专家动作查询 | 同一访问状态下的专家建议动作 | 标注时效、控制延迟和时钟对齐 |

建议为数据保留 $\texttt{executed\_action}$、$\texttt{expert\_label}$、$\texttt{policy\_version}$、$\texttt{intervention\_flag}$ 和标签置信度。并不是每个非典型状态都能可靠标注：遮挡、不可逆接触或不安全状态可能需要跳过监督、触发重置，或只记录为危险状态供安全模块处理。

**DAgger 不是强化学习**：它不需要通过奖励估计价值函数，但要求某种形式的专家动作监督。DAgger 也不等同于所有 Human-in-the-Loop 数据收集：只在失败后给出奖励、偏好或二元成功标签，而不提供访问状态的专家动作，一般不构成标准 DAgger。

<a id="course-section-17"></a>

### 3.8 BC、DAgger 与生成式模仿学习的关系

| 方法 | 数据分布来源 | 训练监督 | 主要解决的问题 |
|:--|:--|:--|:--|
| 单步 BC | 固定专家数据 | 当前动作回归/似然 | 低成本模仿专家 |
| Chunked BC / ACT | 固定专家数据 | 未来动作块 | 时序一致性、动作块生成 |
| Diffusion / Flow BC | 固定专家数据 | 生成式动作分布目标 | 条件动作多模态 |
| DAgger + 任一可监督策略 | 由当前策略持续访问的状态 | 访问状态的专家动作/动作块 | 分布偏移与恢复监督 |

因此，**ACT 与 DAgger 不是相互排斥的替代品**：前者指定“如何表示和预测动作序列”，后者指定“如何选择状态和获取专家标签”。将 DAgger 用于动作块模型时，还需明确专家能否提供与当前历史对齐的完整动作块；只补一个单步专家动作并不足以监督整个块。

参考：[Ross et al., DAgger (AISTATS 2011)](https://proceedings.mlr.press/v15/ross11a.html)。

<a id="course-section-18"></a>

## 4. 动作块、ACT、Diffusion 与 Flow Matching

<a id="course-section-19"></a>

### 4.1 为什么输出一个动作序列

给定当前历史 $h_t$，动作块策略预测未来 $H$ 步：

$$
A_t=(a_{t|t},a_{t+1|t},\ldots,a_{t+H-1|t}),\qquad A_t\sim p_\theta(\cdot\mid h_t).
$$

**预测长度 $H$ 不等于开环执行长度 $K$**。每次可仅执行前 $K\le H$ 步后重新观测；$K$ 越大通常节省推理成本，却延长对新反馈的响应周期。动作块统一了 ACT、Diffusion Policy 和 Flow Matching 的输出接口，但三者的概率建模机制不同。

<a id="course-section-20"></a>

### 4.2 ACT：从时序动作块到条件 VAE 与 Transformer

#### 4.2.1 ACT 要解决的两个具体问题

ACT（Action Chunking with Transformers）由 ALOHA 工作提出，目标并不是让 Transformer“记住更长的动作序列”这么简单，而是同时处理：一是单步预测的长时域误差累积，二是人类示教中存在停顿、不同动作速度及多种合理轨迹造成的条件多样性。

在时刻 $t$，令视觉与本体观测为 $h_t$，未来 $H$ 步示教动作块为

$$
A_t^E=[a_t^E,a_{t+1}^E,\ldots,a_{t+H-1}^E]\in\mathbb R^{H\times D_a}.
$$

ACT 的核心预测器输出 $\hat A_t=f_\theta(h_t,z)$。其中 $z$ 是示教轨迹的潜在“风格”变量；**块长 $H$ 是输出预测范围，执行时多久重新观测则由执行器确定**。典型 ACT 产生的是关节**目标位置**序列，由底层位置控制器跟踪；不能在不修改动作接口的前提下把同一张量解释为力矩、速度或关节增量。

#### 4.2.2 条件 VAE 的编码器与解码器分别做什么

ACT 使用条件变分自编码器（CVAE）。训练时的编码器读取当前本体状态 $q_t$ 和真实未来动作块 $A_t^E$，估计

$$
q_\phi(z\mid q_t,A_t^E)
=\mathcal N\!\left(z;\mu_\phi(q_t,A_t^E),
\operatorname{diag}\sigma_\phi^2(q_t,A_t^E)\right).
$$

通过重参数化 $z=\mu_\phi+\sigma_\phi\odot\epsilon$、$\epsilon\sim\mathcal N(0,I)$ 传递梯度。解码器或策略在完整观测（相机、本体状态）和 $z$ 条件下预测动作块：

$$
\hat A_t=f_\theta(h_t,z),\qquad p(z)=\mathcal N(0,I).
$$

原始 ACT 为加速训练，**VAE 编码器不读视觉图像**，主要利用动作序列与关节状态推断 $z$；视觉信息在解码策略侧融合。此处的 $z$ 不是可直接解释的抓取力、操作阶段或任务 ID，也不是每个动作时刻独立采样一次的噪声。

训练目标为重建项加 KL 正则：

$$
\mathcal L_{\rm ACT}=\frac{1}{|\mathcal B|}
\sum_{(h,A^E)\in\mathcal B}
\left[
\frac{\sum_{k=0}^{H-1}m_k\,\|\hat a_{t+k\mid t}-a^E_{t+k}\|_1}
{\sum_k m_k+\varepsilon}
+\beta\,D_{\rm KL}\!\left(q_\phi(z\mid q_t,A_t^E)\|\mathcal N(0,I)\right)
\right].
$$

掩码 $m_k$ 用于区分真实未来动作和 episode 末尾的填充。对于对角高斯后验，KL 项具有闭式表达：

$$
D_{\rm KL}=\frac12\sum_{j=1}^{D_z}
\big(\mu_j^2+\sigma_j^2-\log\sigma_j^2-1\big).
$$

重建项鼓励精确模仿，$\beta$ 调节模型是否依赖潜变量。$\beta$ 太低可能过度依赖示教编码器提供的信息、推理时先验适配不足；过高则可能使 $z$ 失去信息。不同实现可选择 $L_1$、$L_2$ 或其他动作损失；**原始 ACT 论文正文报告采用 $L_1$ 重建损失**，不能把所有 CVAE 的平方误差写成 ACT 唯一的定义。

#### 4.2.3 Transformer 信息流：为什么能同时看多路相机并输出整块动作

原始 ACT 的结构可理解为两条路径：

```text
Training only:
  [joint state, expert future action chunk]
       -> CVAE encoder -> (mu, sigma) -> z

Policy / inference:
  [multi-view images -> visual features]
  [joint state] + [z] -> Transformer encoder (fusion)
                   -> Transformer decoder (H action queries)
                   -> H x D_a future joint targets
```

视觉主干先将各相机图像编码为带二维位置编码的特征 token；关节状态和潜变量也映射为 token，Transformer encoder 完成融合。Decoder 使用对应 $H$ 个未来时刻的查询 token，通过交叉注意力读取融合特征，并**一次性**输出 $H$ 个动作向量；它不是必须逐动作自回归采样的模型。

示例上，若机器人有 $D_a$ 维动作，输出形状是 $H\times D_a$。输出长度增大提高时间覆盖范围，但也增加需要拟合的未来不确定性；隐状态突然变化时，较远时刻的预测更可能失效。对不同机型应更换对应的输入、输出维度和动作语义，而非照搬 ALOHA 双臂的具体数值。

#### 4.2.4 训练与推理为什么不完全对称

**训练阶段**可看到专家未来动作 $A_t^E$，所以能够通过编码器得到 $q_\phi(z\mid q_t,A_t^E)$ 并最小化重建与 KL。**部署阶段**并不知道未来专家动作，训练编码器会被丢弃；原始 ACT 约定使用先验均值 $z=0$，得到确定性块 $f_\theta(h_t,0)$。从先验随机采样 $z\sim\mathcal N(0,I)$ 是一种可以考虑的生成选择，**但不是原始 ACT 默认的部署方式**。

这也解释了两个常见误解：CVAE 训练具有潜变量并不意味着部署动作必然随机；ACT 训练中看到了真实未来动作，不意味着推理时可以把真实未来动作当输入。

#### 4.2.5 Temporal Ensembling：平均“同一执行时刻”的预测

假设策略在多个观测时刻 $r\in\mathcal R(t)$ 各自产生了包含物理时刻 $t$ 的动作块，其针对 $t$ 的预测记为 $\hat a_{t\mid r}$。Temporal Ensembling（时间集成）将这些**指向同一个物理时间点**的预测融合：

$$
\bar a_t=
\frac{\sum_{r\in\mathcal R(t)}w_{r,t}\,\hat a_{t\mid r}}
{\sum_{r\in\mathcal R(t)}w_{r,t}},
\qquad w_{r,t}=\exp[-m\,j(r,t)].
$$

这里 $j(r,t)=0,1,\ldots$ 按“预测创建的先后、从最旧到最新”编号，是一种与原始 ACT 一致的权重约定；在这种编号下，$m$ 越大越强调较旧的计划、变化越保守，$m$ 越小则越快纳入新观测。**切勿把这个加权误写成相邻执行时间点 $a_{t-1},a_t,a_{t+1}$ 的普通滤波**；它融合的是多个预测者对同一个时间点的估计。

这类集成通常要求足够快地重复查询模型。如果推理本身比控制周期慢，单纯进行 Temporal Ensembling 并不能保证动作无缝衔接；这正是第 5 章 RTC 要处理的问题。对接触建立、释放、碰撞后回弹等突然切换，过大的平滑权重也可能把关键瞬态动作抵消。

#### 4.2.6 ACT 的适用边界与常见失效

ACT 适合高质量、动作语义一致的示教，尤其是双臂和具有阶段性连续动作的操作任务；其主要优势是**单次预测一段轨迹、采用监督训练、无需迭代式去噪采样**。但它并不自动具备接触力建模、状态估计、跨机器人重定向或恢复能力。

常见问题包括：较长 $H$ 带来远期预测不确定性；在分布外状态上仍存在 BC 的累积误差；CVAE 可能出现潜变量塌缩；Temporal Ensembling 在时序事件上可能造成过度平滑；训练时的动作标签（如主机位置）与部署时的控制命令语义不一致会直接破坏策略。**ACT 改善动作表示，并未从理论上消除 BC 的分布偏移问题。**

参考：[Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware* (ACT / ALOHA, 2023)](https://arxiv.org/abs/2304.13705)。

<a id="course-section-21"></a>

### 4.3 Diffusion Policy：迭代去噪

令 $A_0$ 为真实动作块，将噪声注入其不同扩散时间 $n$：

$$
A_n=\sqrt{\bar\alpha_n}A_0+\sqrt{1-\bar\alpha_n}\,\epsilon,
\quad \epsilon\sim\mathcal N(0,I).
$$

噪声预测的基本目标为

$$
\mathcal L_{\rm diff}=
\mathbb E_{A_0,h,n,\epsilon}
\left[\|\epsilon-\epsilon_\theta(A_n,n,h)\|^2\right].
$$

生成时从噪声开始进行多步去噪。它可以表达多峰动作分布，但迭代采样计算量、动作一致性和实际推理延迟必须一起考虑。

<a id="course-section-22"></a>

### 4.4 Flow Matching：学习生成路径上的速度场

在动作空间连接噪声 $A_0\sim p_0$ 与数据 $A_1\sim p_{\rm data}(\cdot\mid h)$。使用直线路径

$$
A_s=(1-s)A_0+sA_1,\quad v^*(A_s,s,h)=A_1-A_0,
$$

对应条件流匹配损失

$$
\mathcal L_{\rm FM}=\mathbb E_{s,A_0,A_1,h}
\|v_\theta(A_s,s,h)-(A_1-A_0)\|^2.
$$

生成时积分 $dA_s/ds=v_\theta(A_s,s,h)$。这里的 $s\in[0,1]$ 是**生成过程的虚拟时间**，$v_\theta$ 是**动作空间中的生成速度场**，不是机械臂关节速度或末端线速度。该简化损失需要与实际采样耦合、路径选取和模型实现保持一致。

<a id="course-section-23"></a>

### 4.5 方法选择应围绕接口与计算约束

| 方法 | 动作表示 | 优点 | 主要代价 |
|:--|:--|:--|:--|
| 单步 BC | 单个动作 | 简单、容易高频调用 | 难处理长期依赖与多模态 |
| Chunked BC / ACT | 固定长度动作块 | 时序一致、单次输出多步 | 块间衔接与开环长度 |
| Diffusion Policy | 生成式动作块 | 分布表达力强 | 迭代采样延迟 |
| Flow Matching | ODE 生成式动作块 | 便于连续生成路径建模 | 数值求解与延迟权衡 |

参考：[ACT / ALOHA](https://arxiv.org/abs/2304.13705)、[Diffusion Policy](https://arxiv.org/abs/2303.04137)、[Flow Matching](https://arxiv.org/abs/2210.02747)。

<a id="course-section-24"></a>

## 5. 实时动作块执行与 RTC

<a id="course-section-25"></a>

### 5.1 两条时间轴：机器人物理时间与模型推理时间

动作块策略输出的是**带计划执行时间的动作序列**，但模型计算也需要时间。设机器人控制周期为 $\Delta t_c$，模型从获取可用观测到返回可执行结果的端到端延迟为 $T_{\rm lat}$，则推理延迟大约占据

$$
d=\left\lceil\frac{T_{\rm lat}}{\Delta t_c}\right\rceil
$$

个控制周期。这里 $T_{\rm lat}$ 必须包含所讨论系统中的观测等待、预处理、网络传输、模型推理、结果回传及调度等必要部分，而不能用 GPU 前向传播时间冒充完整的机器人反应时间。对于抖动显著的系统，$d$ 是随请求变化的随机量，工程上需要考虑尾部延迟，而不能永远假设平均值。

设动作块的**预测长度**为 $H$，一块计划连续使用的**执行长度**为 $s\le H$。即使 $H$ 大于 $s$，若新一块推理尚未返回，真实执行器也不能“跳过等待”来取得未来动作。

<a id="course-section-26"></a>

### 5.2 同步执行、朴素异步、Temporal Ensembling 与 RTC 的区别

| 执行机制 | 新块生成时机器人做什么 | 块间一致性怎样处理 |
|:--|:--|:--|
| 同步执行 | 等待推理或保持旧目标 | 可停顿，未解决非静止系统的运动连续性 |
| 朴素异步 | 继续按旧块发送动作 | 新旧块可能在切换瞬间产生突变 |
| Temporal Ensembling | 混合多个块对同一时刻的预测 | 通过加权平均平滑，不要求生成时满足既定前缀 |
| RTC | 执行旧块，同时生成与已承诺动作一致的新块 | 将重叠动作作为新块生成的条件或引导 |

Temporal Ensembling 是 **ACT 中常用的输出融合规则**，RTC 则是**考虑非零推理延迟的生成与执行协议**；二者不是同义词。无论使用何种方法，底层执行器都应严格按绝对时间戳消费动作，并对过期命令进行拒绝或安全处理。

<a id="course-section-27"></a>

### 5.3 RTC 的几何关系：承诺前缀与自由后缀

假设旧块在时刻 $t-s$ 开始，覆盖区间 $[t-s,t-s+H)$；新模型在时刻 $t$ 开始推理，预期时刻 $t+d$ 返回。在等待的 $d$ 个控制周期里，机器人必须继续执行旧块。新块对应的时间区间为 $[t,t+H)$，因此其前 $d$ 个时刻实际上已经被旧块的执行计划占用。

若旧块以局部索引 $0,\ldots,H-1$ 存储，而新块从时刻 $t$ 计索引，则已承诺前缀为

$$
b_i=A^{\rm old}_{s+i},\quad i=0,\ldots,d-1.
$$

最基本的可用条件为

$$
0\le d\le H-s.
$$

**关键点**：这里要求预测地平线有足够的重叠空间。如果 $d>H-s$，旧块已经不能完整覆盖模型下一次推理的等待时间，必须降低执行更新频率、增大预测长度、降低延迟，或启用独立且安全的低层备用行为；不能假设 RTC 凭空补出未知动作。

一个仅用于解释索引的时间示意（非实验设置）：若 $H=16$、$s=6$、$d=3$，则旧块的第 $6,7,8$ 项作为新块第 $0,1,2$ 项的承诺前缀；新块在 $t+3$ 可用后，才执行它的第 $3$ 项及后续动作。

```text
physical step     t       t+1     t+2     t+3      t+4  --->
robot command     old[6]  old[7]  old[8]  new[3]   new[4]
new chunk index     0       1       2       3        4
status           committed prefix  |    free suffix
model              <--- inference --->
```

前缀约束的目标可以概括为

$$
A^{\rm new}_{0:d-1}\approx b_{0:d-1},\qquad
A^{\rm new}_{d:H-1}\ \text{与观测和前缀一致}.
$$

其中“$\approx$”是**生成结果的近似一致性要求**；真正已经执行的旧动作无法被重新生成或覆盖。原始推理时 RTC 用引导式 inpainting 逼近此要求，不能被描述为数学上精确求得任意模型的条件分布。

<a id="course-section-28"></a>

### 5.4 推理时 RTC：将新动作块生成看成 Inpainting

[Black, Galliker & Levine (2025)](https://arxiv.org/abs/2506.07339) 提出：在不重新训练现有 diffusion/flow 动作策略的前提下，利用先前计划的重叠动作来**引导新块生成**。最简单的硬掩码只强调已经承诺的前 $d$ 个动作；但只约束这几个点，可能允许后续轨迹突然改变方向。因此原论文还引入**软掩码（soft masking）**，对尚未承诺但两块仍然重叠的动作施加逐渐衰减的偏好。

若 $Y_i$ 表示旧块投影到新块时间轴上的参考动作，$M_i\in[0,1]$ 表示参考强度，则可以用以下**教学化的目标**理解该方法：

$$
\mathcal L_{\rm match}(A)=\frac12\sum_{i=0}^{H-1}
M_i\,\|A_i-Y_i\|^2,
\qquad
M_i=\begin{cases}
1,&0\le i<d,\\
\rho_i,&d\le i<H-s,\\
0,&H-s\le i<H,
\end{cases}
$$

其中 $\rho_i$ 从接近 1 逐步降至 0。前缀更强、较远的重叠动作更弱，末尾全新的时段可以自由响应最新观测。**这不是对两个已经算出的动作块简单线性插值，而是在生成新块的过程中施加一致性引导。**

对于流策略，设 $x_\tau$ 是生成时间 $\tau\in[0,1]$ 的中间动作块，基础生成速度场为 $v_\theta(x_\tau,\tau,h_t)$，且直线流的末端估计可写成

$$
\hat A_1(x_\tau)=x_\tau+(1-\tau)v_\theta(x_\tau,\tau,h_t).
$$

一个有助于理解的引导形式为

$$
\tilde v(x_\tau)=v_\theta(x_\tau,\tau,h_t)
-\lambda_\tau\nabla_{x_\tau}
\mathcal L_{\rm match}\big(\hat A_1(x_\tau)\big).
$$

真正的 RTC 论文采用基于伪逆引导（pseudoinverse guidance, $\Pi$GDM）的校正与强度裁剪；上式仅展示**“流速度场 + 对末端动作预测的一致性梯度”**这一机制，而非其全部系数的逐式复刻。梯度通过 $\hat A_1$ 对 $x_\tau$ 的向量-雅可比积（VJP）计算，因此**每个去噪/积分步骤可能引入额外反向计算开销**；尤其在采样步数少时，过大引导强度会导致不稳定，需要裁剪。

推理时 RTC 的优势是能够直接用于兼容的已训练 diffusion/flow 策略；局限是附加的引导计算、对推理延迟估计的依赖，以及近似 inpainting 本身不能完全保证所有连接点的一致性。对于普通确定性 ACT 解码器，不能未经改造就套用这一流模型的引导公式。

<a id="course-section-29"></a>

### 5.5 训练时 RTC：把动作前缀直接变成模型条件

[Black et al. (2025)](https://arxiv.org/abs/2512.05964) 的训练时动作条件化方法用**训练阶段模拟推理延迟**替代推理时附加的 inpainting 梯度。对一个真实动作块 $A^E=[a_0^E,\ldots,a_{H-1}^E]$，随机采样前缀长度 $d$。前 $d$ 个动作视为已知、保持无噪声；只对后 $H-d$ 个动作进行 Flow Matching 的加噪与监督。

令 $\tau\in[0,1]$ 为流生成时间、$\epsilon_i\sim\mathcal N(0,I)$，则模型输入可统一写成

$$
x_i^{\tau}=\begin{cases}
a_i^E,&i<d,\\
(1-\tau)\epsilon_i+\tau a_i^E,&i\ge d,
\end{cases}
\qquad
r_i=\begin{cases}1,&i<d,\\ \tau,&i\ge d.\end{cases}
$$

这里 $r_i$ 是**每个动作 token 的生成时间条件**，用于告诉网络哪些 token 已经完成生成。通过掩码仅训练后缀速度场：

$$
\mathcal L_{\rm TT\text{-}RTC}=
\mathbb E\!\left[
\frac{1}{H-d}\sum_{i=d}^{H-1}
\big\|v_\theta(x^\tau,r,h)_i-(a_i^E-\epsilon_i)\big\|^2
\right],\quad 0\le d<H.
$$

部署时，将已经承诺的前缀传入模型，令其 token 保持 $r_i=1$，只从噪声积分生成未知后缀；无需在每步积分中额外计算 inpainting 的 VJP。对于使用共享生成时间嵌入的架构，必须**支持逐 token 的时间条件与后缀掩码**；论文指出这通常可以在不增加模型可学习参数的情况下完成，但并不意味着所有现成模型都无需接口调整。

训练时 RTC 与推理时 RTC 的差别如下：

| 对比 | 推理时 RTC | 训练时 RTC |
|:--|:--|:--|
| 对原有策略重新训练 | 通常不需要 | 必须提供相应条件训练或微调 |
| 主要约束 | 硬前缀 + 软重叠引导 | 显式已知前缀 + 后缀监督 |
| 附加推理开销 | 有 VJP / guidance | 不需要额外 inpainting 反向计算 |
| 对延迟的适应 | 推理时调整掩码和延迟估计 | 依赖训练覆盖的延迟分布 |
| 基础模型要求 | 可在生成过程中施加引导的 diffusion/flow | 能处理逐 token 时间条件和前缀掩码 |

训练采用真实专家前缀，而部署时前缀往往来自前一块模型预测动作，因此仍可能存在**前缀分布偏移**；应谨慎区分论文证明和工程启发。训练时 RTC 也不能自动获得推理时软掩码对更远重叠未来动作的灵活引导。

<a id="course-section-30"></a>

### 5.6 RTC 系统实际需要哪些运行时组件

一个可靠的执行协议需要以下逻辑边界（不是实验流程）：

1. **控制时钟与执行队列：** 低层控制器按固定周期读取已确认的动作，始终能独立于高层推理运行。
2. **带时戳的观测和计划：** 每个请求携带观测可用时刻、计划起始时刻、预期返回截止时刻；新块到达时先检查是否过期。
3. **延迟估计与承诺前缀：** 根据近期端到端延迟估计 $d$，对当前正在执行的旧计划生成新块的对齐前缀；保留必要的时钟和调度余量。
4. **原子切换：** 在明确的物理时刻切换动作来源，不能让多个推理线程无序覆盖执行队列。
5. **安全后备层：** 推理超时、队列耗尽、观测异常或接触危险时，使用受约束的保持、减速、停止或专门恢复控制器；具体策略依赖动力学和动作接口。

有效动作的时间索引比“它来自哪次 API 调用”更重要：如果新块在 $t+d+2$ 才到达，就不能继续把该块第 $d$ 项当作当前动作执行，必须正确丢弃过期前缀、重对齐或触发后备机制。

<a id="course-section-31"></a>

### 5.7 RTC 不解决什么：接触、状态估计与分布偏移

RTC 处理的是**计算延迟期间持续执行与跨块一致性**，并不保证生成动作在物理上安全或任务上正确。即使满足命令平滑约束

$$
\|u_{t+1}-u_t\|\le\delta_u,
$$

仍可能在高刚度接触时产生过大的交互力；而某些必要的接触切换需要真实的不连续模式变化，不应一味做低通平滑。对位置、速度、力矩和触觉反馈，其安全边界和动作变换完全不同。

RTC 也不能消除 BC 的分布偏移、修复摄像头遮挡、推断未观测摩擦系数，或凭空生成掉落后可靠的重拾能力。这些问题分别要求良好的观测与状态估计、数据覆盖、交互纠错、反馈控制或更高级的任务重规划。

<a id="course-section-32"></a>

### 5.8 本章核心辨析

**Action Chunking** 定义“一次预测未来多少个动作”；**ACT** 是用 Transformer/CVAE 学习动作块的一种策略；**Temporal Ensembling** 是融合不同预测块对同一时刻输出的方法；**Asynchronous Execution** 是在等待推理时继续运行控制器；**RTC** 在此基础上主动约束新块与已承诺的执行前缀一致；**Training-Time RTC** 把这种前缀条件纳入训练。它们分别处在模型表示、输出融合、实时系统和训练协议的不同层次，应该组合讨论而不是混为一个算法。

参考：[Real-Time Execution of Action Chunking Flow Policies (NeurIPS 2025)](https://arxiv.org/abs/2506.07339)；[Training-Time Action Conditioning for Efficient Real-Time Chunking (2025)](https://arxiv.org/abs/2512.05964)。

<a id="course-section-33"></a>

## 6. 数据增强、重定向与失败纠错

<a id="course-section-34"></a>

### 6.1 如何构造物理上合法的增强

轨迹增强不是把单帧图像任意变形，而是对初始状态、目标物体、轨迹与后续标签施加相容变换。若使用场景刚体变换 $G\in SE(3)$，末端参考轨迹和物体位姿应采用一致的坐标规则；但机器人基座、关节限制和重力方向未必允许相同变换。

可将增强视为约束映射

$$
\tau'=\mathcal T(\tau,\xi),\quad
\text{s.t.}\; c_{\rm kin}(\tau')\le0,
\;c_{\rm col}(\tau')\le0,
\;c_{\rm contact}(\tau')\le0.
$$

其中 $c_{\rm kin},c_{\rm col},c_{\rm contact}$ 分别代表运动学、碰撞和接触一致性。对动作轨迹进行坐标变换不保证动态可执行；接触切换附近的误差尤其需要专门处理。可参考 [MimicGen](https://proceedings.mlr.press/v229/mandlekar23a.html) 与 [DexMimicGen](https://dexmimicgen.github.io/)。

<a id="course-section-35"></a>

### 6.2 人类视频重定向的关键缺口

从人类视频推断机器人动作通常涉及：人体/手部与物体的 3D 重建；轨迹平滑、尺度与坐标对齐；接触候选识别；机器人关节/末端目标求解；物理可行性检查与必要的策略修复。视觉轨迹只能提供不完全观测，**遮挡下的接触力、摩擦及真实意图不能简单从像素唯一恢复**。

对于灵巧手，重要事件包括初次接触、稳定抓取、滑移、再抓取、释放、掉落与恢复。可使用多源信号（视觉、关节、电流/力矩、触觉）进行弱标注，并保留事件标签的置信度。

<a id="course-section-36"></a>

### 6.3 失败样本如何进入数据闭环

推荐区分三类数据：**失败前有效动作**、**失败但不应模仿的错误动作**、**专家纠错/恢复动作**。只把所有失败轨迹合并为 BC 正例容易教会策略重复失败。人类接管示教可以监督恢复行为；专家对策略访问状态的动作重标注适用于 DAgger；奖励和在线策略优化则属于强化学习后训练。

数据更新链为

$$
\mathcal D_0\xrightarrow{\text{train}}\pi_0
\xrightarrow{\text{deploy / observe}}\mathcal F_0
\xrightarrow{\text{correct / relabel}}\Delta\mathcal D_0
\xrightarrow{\text{curate}}\mathcal D_1.
$$

每轮更新应保留数据来源、示教者、模型版本与是否人工接管，防止将模型生成但未经验证的动作再次当作真实专家标签。

<a id="course-section-37"></a>

## 7. 真实部署接口与课程衔接

<a id="course-section-38"></a>

### 7.1 模块边界

一个可维护的系统通常区分：**感知与状态估计**（将传感器流转为带时戳的观测）、**策略**（由历史和任务生成动作块）、**实时执行器**（缓存、时间对齐、RTC 或其他衔接）、**安全控制层**（限幅、碰撞与紧停）以及**数据记录层**（同步记录决策、下发与反馈）。这些模块的契约比模型名称更重要。

<a id="course-section-39"></a>

### 7.2 常见失效与定位方向

| 症状 | 优先排查 |
|:--|:--|
| 离线预测准确，部署频繁偏离 | 数据分布偏移、观测延迟、坐标系不一致 |
| 动作块边界出现顿挫 | 同步等待、旧/新块时间轴未对齐、RTC 前缀不匹配 |
| 接触动作被平均掉 | Temporal Ensembling 过度平滑、多模态标签混合 |
| 跨物体泛化差 | 按帧拆分导致泄漏、物体/初始状态覆盖不足 |
| 恢复动作无效 | 错误失败标签、缺少真实纠错示教、状态估计失效 |
| 真机动作方向或尺度错误 | 单位、坐标系、绝对/相对控制、归一化版本 |

<a id="course-section-40"></a>

### 7.3 与课程其他部分的分工

本讲义负责**数据如何成为可训练的动作监督，以及动作块如何在真实时间线上运行**。相机标定与视觉状态估计课程负责相机/世界/机器人变换和遮挡恢复；Flow Matching 专题负责生成路径、求解器和概率流理论；强化学习课程负责奖励驱动的策略优化；Robot Agent 专题负责长程任务规划、失败判断与子任务重规划。模块之间采用明确接口，避免在每门课程中重复整套算法。

<a id="course-section-41"></a>

## 参考文献与延伸阅读

1. Ross et al., *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning* (DAgger, 2011). https://proceedings.mlr.press/v15/ross11a.html
2. Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware* (ACT / ALOHA, 2023). https://arxiv.org/abs/2304.13705
3. Chi et al., *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion* (2023). https://arxiv.org/abs/2303.04137
4. Lipman et al., *Flow Matching for Generative Modeling* (2022/2023). https://arxiv.org/abs/2210.02747
5. Wu et al., *GELLO: A General, Low-Cost, and Intuitive Teleoperation Framework for Robot Manipulators* (2023). https://arxiv.org/abs/2309.13037
6. Chi et al., *Universal Manipulation Interface* (2024). https://arxiv.org/abs/2402.10329
7. Mandlekar et al., *MimicGen: A Data Generation System for Scalable Robot Learning using Human Demonstrations* (2023). https://proceedings.mlr.press/v229/mandlekar23a.html
8. *DexMimicGen* project. https://dexmimicgen.github.io/
9. Black, Galliker & Levine, *Real-Time Execution of Action Chunking Flow Policies* (2025). https://arxiv.org/abs/2506.07339
10. Black et al., *Training-Time Action Conditioning for Efficient Real-Time Chunking* (2025). https://arxiv.org/abs/2512.05964
11. LeRobot Dataset v3 documentation. https://huggingface.co/docs/lerobot/lerobot-dataset-v3

---

*资料说明：本文为教学性重编写，遵循原讲义的课程范围；算法公式为便于理解的标准化或示意性表述，并非对所有具体实现的逐行复刻。本文没有实验部分，也不宣称实际完成模型训练或真机测试。*
