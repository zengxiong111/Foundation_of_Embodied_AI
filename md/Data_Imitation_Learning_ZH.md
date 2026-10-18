# 数据采集与模仿学习

<!-- course-meta:start -->
[课程首页](https://github.com/zengxiong111/Foundation_of_Embodied_AI/blob/main/README.zh-CN.md) · [English](https://github.com/zengxiong111/Foundation_of_Embodied_AI/blob/main/md/Data_Imitation_Learning_EN.md) · [PDF](https://zengxiong111.github.io/zengxiong.github.io/files/Data_Imitation_Learning_ZH.pdf)

| 项目 | 说明 |
|---|---|
| 适用范围 | 示教数据、BC、DAgger、ACT、生成式动作块策略 |
| 文档核验日期 | 2026-10-18 |
| 许可 | CC BY 4.0 · Xiong Zeng and contributors |

**核验边界：** 方法与实验设计，未实际复现训练。 版本与固定源码引用见[版本说明](https://github.com/zengxiong111/Foundation_of_Embodied_AI/blob/main/VERSIONS.zh-CN.md)。原创教材采用 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)，第三方引用、图片及代码遵循其原有许可；注明来源不等于拥有再许可权。
<!-- course-meta:end -->

<!-- course-toc:start -->
## 目录

- [1. 从演示到可训练的数据](#course-section-01)
  - [1.1 一条轨迹应包含什么](#course-section-02)
  - [1.2 时间和坐标系先于网络结构](#course-section-03)
- [2. 数据采集与质量控制](#course-section-04)
  - [2.1 成功演示之外，还需要什么](#course-section-05)
  - [2.2 数据划分比指标数字更重要](#course-section-06)
- [3. 模仿学习：从 BC 到交互式纠错](#course-section-07)
  - [3.1 为什么训练误差低也会失败](#course-section-08)
  - [3.2 DAgger 是什么](#course-section-09)
- [4. 从单步动作到生成式动作块](#course-section-10)
  - [4.1 ACT 的核心思路](#course-section-11)
  - [4.2 Diffusion Policy 与 Flow Matching](#course-section-12)
- [5. 增强、再标定与仿真数据](#course-section-13)
  - [5.1 接触事件和恢复样本](#course-section-14)
  - [5.2 数据增强的安全边界](#course-section-15)
- [6. 统一 Panda 桌面抓取实验](#course-section-16)
  - [6.1 最小实验表](#course-section-17)
  - [6.2 常见失败排查](#course-section-18)
- [7. 统一数据模型：从轨迹到动作块](#course-section-19)
  - [7.1 为什么要区分真实状态、观测与命令](#course-section-20)
  - [7.2 Action Chunk 的学习目标](#course-section-21)
  - [7.3 时间戳、坐标系与归一化](#course-section-22)
- [8. BC 的数学形式、设计动机与局限](#course-section-23)
  - [8.1 确定性 BC 与概率 BC](#course-section-24)
  - [8.2 分布偏移：为什么测试时误差会累积](#course-section-25)
  - [8.3 动作块监督与平滑的取舍](#course-section-26)
- [9. 从离线 BC 到交互式模仿学习](#course-section-27)
  - [9.1 DAgger 的数据聚合与专家查询](#course-section-28)
  - [9.2 失败轨迹、恢复示教与标签质量](#course-section-29)
- [10. 从 ACT 到扩散与 Flow Matching 动作策略](#course-section-30)
  - [10.1 ACT：CVAE 与 Transformer 动作块](#course-section-31)
  - [10.2 Diffusion Policy：以噪声预测学习动作块分布](#course-section-32)
  - [10.3 Flow Matching：动作速度场与物理速度不是一回事](#course-section-33)
  - [10.4 生成式策略和 BC 的公平对照](#course-section-34)
- [11. 人类视频重定向、接触与数据增强](#course-section-35)
  - [11.1 从人类视频到机器人动作不是直接复制像素](#course-section-36)
  - [11.2 轨迹增强必须保持物理和标签一致](#course-section-37)
- [12. 建议的训练与评估规范](#course-section-38)
  - [12.1 区分离线预测、闭环成功率和扰动恢复](#course-section-39)
  - [12.2 一个贯穿课程的最小实验](#course-section-40)
- [13. 课程衔接与参考资料](#course-section-41)
- [8. 深入：从示教轨迹到监督学习目标](#course-section-42)
  - [8.1 数据集、闭环系统与行为策略](#course-section-43)
  - [8.2 确定性与概率式 BC](#course-section-44)
  - [8.3 分布偏移与 DAgger](#course-section-45)
- [9. 深入：动作块、ACT 与生成式模仿学习](#course-section-46)
  - [9.1 预测长度和执行长度](#course-section-47)
  - [9.2 ACT 的 CVAE 目标](#course-section-48)
  - [9.3 Diffusion Policy 的去噪训练](#course-section-49)
  - [9.4 Flow Matching 形式的动作块学习](#course-section-50)
- [10. 数据工程与评估的完整清单](#course-section-51)
  - [10.1 轨迹质量、标签可靠性与事件分层](#course-section-52)
  - [10.2 Panda 统一实验与最小消融](#course-section-53)
  - [10.3 从模仿学习到 Robot Agent 的接口](#course-section-54)
<!-- course-toc:end -->

**学习目标：** 识别数据质量问题；比较 BC、ACT 与生成式动作策略；设计可复现的抓取实验。

<a id="course-section-01"></a>

## 1. 从演示到可训练的数据

模仿学习的目标不是保存一段看起来正确的视频，而是把观察、机器人状态、动作和任务条件组织成可对齐、可复现的监督信号。对于闭环机器人，最终学习的是条件策略 $\pi_\theta(a_t\mid o_{\le t},g)$，其中 $g$ 是目标。

<a id="course-section-02"></a>

### 1.1 一条轨迹应包含什么

建议每条 episode 至少保存：任务与场景 ID、相机时间戳、RGB 或 RGB-D、关节状态、执行动作、动作坐标系、控制模式、成功与否、异常事件。**动作标签必须表示实际发给机器人的命令**，不能把目标关节角、实际反馈位置和关节速度混为一谈。对视觉模型，还需注明相机外参是静态还是时变。

<a id="course-section-03"></a>

### 1.2 时间和坐标系先于网络结构

使用统一的时钟基准或经验证的时间对齐。标明 $T_{B}^{C}$、$T_{B}^{E}$、$T_{B}^{O}$ 的定义，所有姿态矩阵使用相同的乘法约定。相机通常低频，机器人本体状态可能更高频；可采用与动作时刻对齐的最近帧或带明确延迟的插值，避免未来信息泄漏。

<a id="course-section-04"></a>

## 2. 数据采集与质量控制

典型数据源包括遥操作、拖动示教、脚本专家、仿真 RL 专家和人类视频重定向。它们在采样效率、动作精度、接触真实性和设备成本上各有差异。

```mermaid
flowchart LR
  N0["任务定义"]
  N1["演示采集"]
  N2["时间与坐标对齐"]
  N3["质量筛选"]
  N4["训练数据集"]
  N5["评估与失败回收"]
  N0 --> N1
  N1 --> N2
  N2 --> N3
  N3 --> N4
  N4 --> N5
  N5 --> N1
```

**实用原则：** 一开始优先获得质量稳定且可复现的小数据集，而不是尽快积累大量未标注或不同步的视频。

<a id="course-section-05"></a>

### 2.1 成功演示之外，还需要什么

保留有效的失败轨迹、纠错动作与恢复片段，并标注事件边界，例如接触开始、滑落、重新抓取。不要把所有失败都作为“照做”的标签；区分应模仿的恢复策略与应避免的危险命令。明确数据使用权限、隐私、许可证及真实机器人安全审核。

<a id="course-section-06"></a>

### 2.2 数据划分比指标数字更重要

按对象实例、场景布局、操作者、初始姿态和采集日期划分训练与测试；**禁止同一连续轨迹切块后同时出现在训练集和测试集**。记录真实 episode 级成功率，而不仅是逐动作 MSE。多相机或多帧观测应整体划分，防止相邻帧泄漏。

<a id="course-section-07"></a>

## 3. 模仿学习：从 BC 到交互式纠错

Behavior Cloning（BC）直接优化数据上的动作预测。对于确定性回归可采用 $\mathcal{L}_{BC}=\mathbb{E}_{(o,a)\sim D}\|\pi_\theta(o)-a\|_2^2$；对于概率策略则最大化动作条件对数似然。这两种目标并不自动解决测试时的分布偏移。

<a id="course-section-08"></a>

### 3.1 为什么训练误差低也会失败

闭环执行中，小的动作误差会改变下一时刻的观测，机器人可能进入训练集从未覆盖的状态。应在测试时记录状态偏移、触碰失败、纠错次数，以及是否能从偏离轨迹的状态恢复。可加入安全受控的小扰动或恢复示教，扩大有效状态覆盖。

<a id="course-section-09"></a>

### 3.2 DAgger 是什么

DAgger 的思想是让当前策略收集会访问的状态，并由专家为这些状态提供适当动作，再聚合数据更新策略。对真实机器人要有明确的安全干预机制；如果无法在线查询专家，就不能简单地把普通离线训练称为 DAgger。

<a id="course-section-10"></a>

## 4. 从单步动作到生成式动作块

生成动作块可减少频繁重规划的开销，但会引入执行中观测过时的风险。区分 chunk horizon（预测长度）、execution horizon（执行长度）与 policy query rate（模型调用频率）。动作块的控制参考系必须和机器人的实际执行器接口一致。

<a id="course-section-11"></a>

### 4.1 ACT 的核心思路

ACT 使用 Transformer 预测一段连续的动作，并在论文实现中借助潜变量建模演示差异。与逐步 BC 相比，动作块适合示教数据中连续且结构化的操作，但需要正确处理 chunk 的对齐与组合。

<a id="course-section-12"></a>

### 4.2 Diffusion Policy 与 Flow Matching

Diffusion Policy 与 Flow Matching 都可用生成式机制表示多模态动作分布，但**扩散去噪过程、概率流或速度场求解与机器人关节速度是不同概念**。

```mermaid
flowchart LR
  N0["历史观测与目标"]
  N1["动作块先验噪声"]
  N2["条件生成网络"]
  N3["去噪或 ODE 求解"]
  N4["候选动作块"]
  N5["控制接口"]
  N0 --> N2
  N1 --> N2
  N2 --> N3
  N3 --> N4
  N4 --> N5
```

在比较方法时，固定观察模态、示教数据量、推理预算、控制频率与评估环境；不能只比较离线训练损失。

<a id="course-section-13"></a>

## 5. 增强、再标定与仿真数据

在机器人操作中，几何变换必须一致地作用于图像、物体目标、轨迹参考和动作标签。仅把图像旋转而不调整动作的增强通常会产生错误监督。仿真专家轨迹可以扩充姿态覆盖，但接触参数、视觉渲染和执行器时延的 Sim-to-Real 差异需要单独验证。

<a id="course-section-14"></a>

### 5.1 接触事件和恢复样本

报告均匀采样基线，再评估接触切换、夹持失败和物体掉落附近的增采样。事件采样改变训练数据分布，需要明确其权重以及是否扭曲自然状态分布；也要评估无事件偏置的测试集。

<a id="course-section-15"></a>

### 5.2 数据增强的安全边界

不应把任意初始条件直接传播到不满足物理约束的接触轨迹。对旋转、平移、尺度、时序扰动都检查可达性、碰撞、接触一致性和控制限幅；可先做短时物理回放再收录。

<a id="course-section-16"></a>

## 6. 统一 Panda 桌面抓取实验

**实验任务：** 在 Isaac Lab 或 MuJoCo 中让 Panda 抓取随机放置的方块，先生成脚本专家或遥操作轨迹，再训练 BC / ACT / 生成式策略。

**控制变量：** 固定相机视角、训练任务数、机器人动作接口、控制频率、训练/测试对象划分、评估回合数。以简单 BC 作为所有生成式方法的必要基线。

**指标：** 成功率、接触失败率、轨迹偏差、恢复率、平均策略推理延迟、平均控制延迟与数据量。每种方法报告多个随机种子和失败模式，而非宣称某方法天然最优。

<a id="course-section-17"></a>

### 6.1 最小实验表

| 对照 | 目的 |
|---|---|
| BC 单步 | 数据与控制接口是否正确 |
| BC 动作块 | 排除 chunking 本身的贡献 |
| ACT | 检验 Transformer 行为生成 |
| Flow/Diffusion | 检验多模态动作建模 |
| 增加恢复示教 | 检验闭环扰动恢复 |

<a id="course-section-18"></a>

### 6.2 常见失败排查

优先检查时间对齐、动作/反馈混淆、相机外参、动作缩放、延迟、策略执行长度、数据划分与初始化分布，最后再考虑扩大模型。不能用只在训练集上计算的 action MSE 代替真机任务成功率。

<a id="course-section-19"></a>

## 7. 统一数据模型：从轨迹到动作块

<a id="course-section-20"></a>

### 7.1 为什么要区分真实状态、观测与命令

在机器人控制中，真实世界状态 $s_t$（物体速度、摩擦系数、接触模式等）一般无法被完整观测。机器人可以记录图像 $o_t^{\mathrm{cam}}$、关节反馈 $q_t$、实际发送的控制命令 $a_t$ 和任务目标 $g$。将一条示教定义为

$$
\tau_i=\{g_i,(o_t^{\mathrm{cam}},q_t,a_t,r_t^{\mathrm{event}})_{t=0}^{T_i-1}\}.
$$

事件标注 $r_t^{\mathrm{event}}$ 可以表示接触、滑落、恢复与任务完成，**不是必须使用的强化学习奖励**。不要把仿真器的真实物体位姿、示教者未来的动作或后处理的成功标签偷放到部署策略输入。若监督信号来自动作命令而观测来自延迟的反馈，应明确这一时序关系。

<a id="course-section-21"></a>

### 7.2 Action Chunk 的学习目标

策略从最近 $L$ 帧观测、当前本体状态和目标构造条件 $c_t=(o_{t-L+1:t},q_t,g)$，预测长度为 $H$ 的动作块

$$
A_t=(a_t,a_{t+1},\ldots,a_{t+H-1}),\qquad
p_\theta(A_t\mid c_t).
$$

$H$ 是**预测长度**，不是必须开环执行的步数。实际执行 $h\le H$ 步后重新观测，构成滚动时域控制。策略查询周期、机器人低层控制周期、动作块执行长度必须分别记录。例如视觉为 30 Hz、低层伺服为 100 Hz 时，策略可以低于伺服频率运行，但中间的插值和超时机制不可省略。

<a id="course-section-22"></a>

### 7.3 时间戳、坐标系与归一化

动作时间为 $t_k^a$ 时，输入图像必须满足其**可用时间**不晚于 $t_k^a$。图像采集、传输与推理的延迟记为 $\delta$，采用 $t_j^{\rm image}\le t_k^a-\delta$ 的历史图像；不要使用未来图像对齐训练样本。对末端位姿动作，若增量定义在基座系可写作 $T_{BE,t+1}=\Delta T_{B,t}T_{BE,t}$；若定义在末端系则为 $T_{BE,t+1}=T_{BE,t}\Delta T_{E,t}$。**两种约定不可混用。**

对训练集中的动作分量计算 $\mu_D,\sigma_D$ 后归一化：

$$
\tilde a=(a-\mu_D)/(\sigma_D+\varepsilon),\qquad
a_{\rm deployed}=\tilde a(\sigma_D+\varepsilon)+\mu_D .
$$

不同关节角、夹爪开合或末端旋转需要正确的单位与限幅。检查归一化统计是否意外使用测试集数据。

<a id="course-section-23"></a>

## 8. BC 的数学形式、设计动机与局限

<a id="course-section-24"></a>

### 8.1 确定性 BC 与概率 BC

给定示教数据分布 $d_E(c,a)$，最简单的确定性 BC 优化

$$
\mathcal L_{\rm BC}(\theta)=\mathbb E_{(c,a)\sim d_E}\|\pi_\theta(c)-a\|_2^2.
$$

它在均方损失下趋向条件均值 $\mathbb E[a\mid c]$。当专家行为近似单峰时，它是低成本、值得保留的强基线；当同一条件下“从左绕开”和“从右绕开”都是合理动作时，平均行为可能撞上障碍物。

用概率模型更一般地写为最大似然：

$$
\mathcal L_{\rm NLL}(\theta)=-\mathbb E_{(c,a)\sim d_E}\log p_\theta(a\mid c).
$$

若 $p_\theta$ 为方差固定的各向同性高斯分布，NLL 与带常数系数的 MSE 等价。多峰分布需要混合密度模型或生成式动作策略，**不能只因为动作维度高就认定 BC 必然失败**。

<a id="course-section-25"></a>

### 8.2 分布偏移：为什么测试时误差会累积

离线 BC 在专家访问的状态分布 $d_{\pi_E}$ 上训练，但部署后的数据来自 $d_{\pi_\theta}$，两者可能不同：

$$
\min_\theta\,\mathbb E_{s\sim d_{\pi_E}}\ell(\pi_\theta(s),\pi_E(s))
\quad\not\Rightarrow\quad
\min_\theta\,\mathbb E_{s\sim d_{\pi_\theta}}\ell(\pi_\theta(s),\pi_E(s)).
$$

一个小的接触误差可能触发物体滑动，使后续观察脱离训练分布。专家示教集中在成功轨迹时，模型可能根本没有见过“物体掉落后重新抓取”的状态。这解释了为什么测试需要扰动恢复回合，而不仅是随机抽样的动作回归误差。

<a id="course-section-26"></a>

### 8.3 动作块监督与平滑的取舍

在带掩码的变长轨迹中，动作块回归可以写为

$$
\mathcal L_{\rm chunk}=
\frac{\sum_{t,k}m_{t,k}w_k\|\hat a_{t+k\mid t}-a_{t+k}\|_1}
{\sum_{t,k}m_{t,k}w_k},
$$

其中 $m_{t,k}\in\{0,1\}$ 排除越界步，$w_k$ 可以减小远期预测权重。可选的动作平滑正则为 $\lambda_s\sum_k\|\hat a_{k+1}-\hat a_k\|^2$，但精密插装、接触切换时动作变化可能是必要的，**过度平滑会抹掉真实接触事件**。比较正则前后成功率和力/速度峰值，而不是只看曲线是否光滑。

<a id="course-section-27"></a>

## 9. 从离线 BC 到交互式模仿学习

<a id="course-section-28"></a>

### 9.1 DAgger 的数据聚合与专家查询

DAgger 的要点不是简单再训练，而是让当前策略暴露它自己会遇到的状态，专家再标注这些状态。第 $i$ 轮执行混合策略 $\pi_i=\beta_i\pi_E+(1-\beta_i)\pi_{\theta_i}$，采集 $s\sim d_{\pi_i}$，专家提供动作 $\pi_E(s)$，然后更新

$$
D_{i+1}=D_i\cup\{(s,\pi_E(s)):s\sim d_{\pi_i}\},\qquad
\theta_{i+1}=\arg\min_\theta\sum_{(s,a)\in D_{i+1}}\ell(\pi_\theta(s),a).
$$

$\beta_i$ 通常逐步降低。真实机器人上必须给专家接管、安全停止和强制限幅预留控制权；不能直接让未经审查的探索命令在硬件上自由执行。

<a id="course-section-29"></a>

### 9.2 失败轨迹、恢复示教与标签质量

示教可以分为成功完成、可恢复失败、不可恢复失败、人工中断四类。只对**可安全模仿的动作**建立监督，记录接触开始、滑移、掉落、重新定位、重新抓取等事件。对于不可安全复现的行为，应保留结果标签与日志用于分析，而不是直接做正例 BC。

建议在均匀时间采样的 baseline 之后再加入 event-biased sampling。若事件权重为 $w_t$，可以采用 $\mathcal L=\sum_tw_t\ell_t/\sum_tw_t$；必须披露其改变的训练分布，并在无偏的测试集上评估。

<a id="course-section-30"></a>

## 10. 从 ACT 到扩散与 Flow Matching 动作策略

<a id="course-section-31"></a>

### 10.1 ACT：CVAE 与 Transformer 动作块

Action Chunking with Transformers（ACT）通过 Transformer 预测整段关节命令，原始方案还使用条件变分自编码器处理示教的风格差异。训练时编码器近似 $q_\phi(z\mid A,c)$，解码器给出 $p_\theta(A\mid z,c)$，常见的损失为

$$
\mathcal L_{\rm ACT}=
\mathbb E_{z\sim q_\phi}\big[-\log p_\theta(A\mid z,c)\big]
+\beta\,D_{\rm KL}(q_\phi(z\mid A,c)\|p(z)).
$$

第二项把示教编码分布约束到先验 $p(z)$；推理阶段没有真实未来动作块 $A$ 可用，应从约定先验采样或使用其确定性代表。$\beta$ 过大可能造成后验塌缩，过小则易使训练与推理潜变量分布不一致。

<a id="course-section-32"></a>

### 10.2 Diffusion Policy：以噪声预测学习动作块分布

令 $A_0$ 为真实动作块、$\epsilon\sim\mathcal N(0,I)$、扩散时间 $k$，构造

$$
A_k=\sqrt{\bar\alpha_k}A_0+\sqrt{1-\bar\alpha_k}\epsilon.
$$

一个常见的条件噪声预测目标为

$$
\mathcal L_{\rm diff}=
\mathbb E_{A_0,c,k,\epsilon}
\|\epsilon-\epsilon_\theta(A_k,k,c)\|_2^2.
$$

推理时从高斯噪声迭代去噪得到动作块，并只执行其中一部分，再基于新观测重规划。计算量由去噪步数、动作维度和视觉编码器共同决定；“生成式”本身不保证实时。

<a id="course-section-33"></a>

### 10.3 Flow Matching：动作速度场与物理速度不是一回事

为便于解释，设数据端点 $A_1\sim p_{\rm data}(\cdot\mid c)$、噪声 $A_0\sim\mathcal N(0,I)$，选择直线路径 $A_\tau=(1-\tau)A_0+\tau A_1$。一个条件速度回归目标为

$$
\mathcal L_{\rm FM}=
\mathbb E_{c,A_0,A_1,\tau}
\|v_\theta(A_\tau,\tau,c)-(A_1-A_0)\|_2^2.
$$

采样时求解生成空间 ODE：$dA_\tau/d\tau=v_\theta(A_\tau,\tau,c)$。其中 $v_\theta$ 是**动作样本在生成时间 $\tau$ 上的变化方向**，不是机械臂关节速度 $\dot q$。不同论文的时间方向和路径不同；严格的连续性方程、条件期望推导及 OpenPI 的时间约定统一放在 [Flow Matching 专题](Flow_Matching_详细推导_精简版.md)，本章只说明与模仿学习的接口。

<a id="course-section-34"></a>

### 10.4 生成式策略和 BC 的公平对照

先用相同数据、同一观测编码、同一动作表示和相同执行长度比较：单步 BC、chunked BC、ACT、Diffusion Policy、Flow Matching。报告训练数据量、推理频率、生成步数、动作延迟和实际任务成功率。不能把“ACT 的动作块收益”误记成“Transformer 架构收益”，也不能把更多推理计算导致的增益归因于算法本身。

<a id="course-section-35"></a>

## 11. 人类视频重定向、接触与数据增强

<a id="course-section-36"></a>

### 11.1 从人类视频到机器人动作不是直接复制像素

可采用“视频中的手/工具位姿估计 → 相机与尺度对齐 → 目标机器人运动学重定向 → 接触时间段识别 → 物理回放与不可达轨迹修复 → 学习”的链条。人类手与灵巧手的自由度、连杆长度及摩擦条件不同，简单的关节角映射通常不足以复现接触。重定向的目标可包含手部追踪、接触位置和约束软惩罚：

$$
\mathcal L_{\rm retarget}
=w_p\|p_{\rm robot}-p_{\rm target}\|^2
+w_R\,d_{SO(3)}(R_{\rm robot},R_{\rm target})^2
+w_c\mathcal L_{\rm contact}+w_j\mathcal L_{\rm limits}.
$$

这里 $\mathcal L_{\rm contact}$ 必须针对具体任务定义，例如接触点距离、接触法线、抓握保持或物体相对滑移；不等于“两个物体距离近”就保证存在力学接触。

<a id="course-section-37"></a>

### 11.2 轨迹增强必须保持物理和标签一致

对物体姿态的随机旋转与平移，需同步变换目标、观测几何和所有依赖参考系的动作标签。图像旋转不伴随动作坐标变换通常会导致错误监督。接触阶段还应在短时物理回放中检查指尖穿透、物体滑落、关节限位、控制频率和接触力，删去不可信增强片段。

<a id="course-section-38"></a>

## 12. 建议的训练与评估规范

<a id="course-section-39"></a>

### 12.1 区分离线预测、闭环成功率和扰动恢复

| 测试 | 应测指标 | 不能替代什么 |
|---|---|---|
| 留出示教集 | NLL / 动作误差、chunk 接口一致性 | 不能代替闭环成功率 |
| 常规随机初始位 | 成功率、完成时间、碰撞与接触失败率 | 不能代替扰动测试 |
| 扰动 / 掉落后恢复 | Recovery success、恢复时长、重新抓取次数 | 不能代替正常性能 |
| 跨场景和跨对象 | 分组划分的泛化成功率 | 不能使用相邻轨迹帧泄漏 |
| 系统实时性 | 策略推理 p50/p95 延迟、执行器延迟 | 不能只报告 GPU 帧率 |

成功率估计 $\hat p=k/N$ 应同时报告独立试验数 $N$、训练随机种子及置信区间；粗略标准误差为 $\sqrt{\hat p(1-\hat p)/N}$，当 $N$ 小或成功率接近 0/1 时采用 Wilson 区间。不能将相互相关的同一条轨迹的动作帧当作独立 episode。

<a id="course-section-40"></a>

### 12.2 一个贯穿课程的最小实验

在同一 Panda + 方块 + 桌面场景训练五个基线，固定训练 episode 数、相机、控制模式与任务划分。按阶段报告：首先验证动作标签，随后比较 BC 与 chunked BC，再比较 ACT、Diffusion/Flow，最后专门加入恢复示教并做随机外力或物体位姿扰动。在 Isaac / MuJoCo 交叉测试时明确控制器和接触模型可能不等价。

**章节边界：** 本章深入讲采集、示教监督、分布偏移、生成动作策略和数据设计；PPO/SAC 的理论推导仍在各自专题，Flow Matching 的严格数学推导见相应章节，视觉 6D 位姿与 SAM/Foundation 系列见 [视觉感知与状态估计](Visual_State_Estimation_ZH.md)。


<a id="course-section-41"></a>

## 13. 课程衔接与参考资料

本章连接[相机标定](Robot_Camera_Calibration_Tutorial_ZH.md)、[视觉状态估计](Visual_State_Estimation_ZH.md)、[Flow Matching](Flow_Matching_详细推导_精简版.md)与[GR00T / SONIC](System1_System0_GR00T_SONIC_CN.md)。

- [ACT / ALOHA](https://arxiv.org/abs/2304.13705)
- [Diffusion Policy](https://arxiv.org/abs/2303.04137)
- [DAgger](https://arxiv.org/abs/1011.0686)
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- [LeRobot](https://github.com/huggingface/lerobot)

**核验边界：** 本章是方法与实验设计导论；没有声称这些算法已在本课程统一 Panda 环境复现。


<a id="course-section-42"></a>

## 8. 深入：从示教轨迹到监督学习目标

<a id="course-section-43"></a>

### 8.1 数据集、闭环系统与行为策略

将一个 episode 表示为 $\tau_i=(g_i,\{o_t,q_t,a_t,r_t,\delta_t\}_{t=0}^{T_i-1})$，其中 $o_t$ 可以包含头部和腕部图像，$q_t$ 是机器人反馈，$a_t$ 是**实际发出的控制命令**，$\delta_t$ 标记终止或异常事件。目标数据集 $D=\{\tau_i\}_{i=1}^N$ 应保留每个观测与控制信号的独立时间戳，而不是仅保存重采样后的视频帧。

区分三种策略：专家 $\pi_E$、离线数据收集行为策略 $\mu$ 和待学习策略 $\pi_\theta$。如果混入随机探索、失败恢复和多个操作者的数据，通常 $\mu\ne\pi_E$；不能默认全部动作都是最优监督标签。对训练样本可设置质量权重 $w_i\ge 0$，但需要在无权重的数据分布上进行独立评估。

时间对齐的因果约束是 $t_{\mathrm{obs}}\le t_{\mathrm{act}}$，考虑感知和通信延迟 $d$ 时，策略真实使用的是 $\pi_\theta(a_t\mid o_{t-d},q_{t-d},g)$。训练时使用未来帧会导致离线指标偏高但部署失败。

<a id="course-section-44"></a>

### 8.2 确定性与概率式 BC

对于确定性动作回归，经典损失为

$$
\mathcal L_{\mathrm{BC}}(\theta)=\frac1{|D|}\sum_{(o,a)\in D}\|\pi_\theta(o)-a\|_2^2.
$$

当示教是多模态分布，例如抓取物体可以从左右两侧接近，单一均值动作可能落在两种模式之间。概率式 BC 优化

$$
\mathcal L_{\mathrm{NLL}}(\theta)=-\mathbb E_{(o,a)\sim D}\log\pi_\theta(a\mid o,g).
$$

对高斯策略 $\pi_\theta=\mathcal N(\mu_\theta,\Sigma_\theta)$，负对数似然包含 $\frac12(a-\mu_\theta)^\top\Sigma_\theta^{-1}(a-\mu_\theta)$ 与 $\frac12\log|\Sigma_\theta|$，因此固定协方差时才近似为加权 MSE。关节角、末端位姿和手指控制量的量纲不同，应先定义动作尺度或明确损失权重。

<a id="course-section-45"></a>

### 8.3 分布偏移与 DAgger

离线训练最小化专家访问分布 $d^{\pi_E}$ 下的单步误差，闭环部署却访问 $d^{\pi_\theta}$。简单情况下误差会随时域 $T$ 复合累积，不能把单步高准确率直接等同于任务成功。

DAgger 通过迭代方式缓解偏移：

$$
D_{k+1}=D_k\cup\{(o,\pi_E(o)):o\sim d^{\pi_{\theta_k}}\},\qquad
\theta_{k+1}=\arg\min_\theta\mathbb E_{(o,a)\sim D_{k+1}}\ell(\pi_\theta(o),a).
$$

关键是为学习器真正访问的状态提供专家标签；对机器人应设置接管、紧急停止、速度和力矩上限。仅仅对旧数据重复训练**不是** DAgger。

<a id="course-section-46"></a>

## 9. 深入：动作块、ACT 与生成式模仿学习

<a id="course-section-47"></a>

### 9.1 预测长度和执行长度

令 $A_t=(a_t,\ldots,a_{t+H-1})$ 为长度为 $H$ 的动作块。策略输入 $c_t=(o_{t-k:t},q_t,g)$，输出 $\hat A_t=f_\theta(c_t)$。实际只执行前 $h\le H$ 步再重新观测，从而平衡计算开销与闭环纠错能力。**区分**策略查询频率 $f_\pi$、低层控制频率 $f_c$ 和执行时长 $h/f_c$。观测发生遮挡、物体掉落或接触变化时，应优先允许提前中止当前动作块。

<a id="course-section-48"></a>

### 9.2 ACT 的 CVAE 目标

ACT 以 Transformer/CVAE 预测动作块。其典型训练目标为

$$
\mathcal L_{\mathrm{ACT}}=
\mathbb E_{q_\phi(z\mid A,c)}
\|A-f_\theta(c,z)\|_1
+\beta D_{\mathrm{KL}}\!\left(q_\phi(z\mid A,c)\|p(z)\right).
$$

第一项拟合示教动作块，第二项让训练时使用的潜变量分布靠近推理先验。实践中还需明确多时刻预测重叠时如何做 temporal ensembling，以及不同接触阶段是否适合平均动作。

<a id="course-section-49"></a>

### 9.3 Diffusion Policy 的去噪训练

定义干净动作块 $A^0$ 和加噪变量 $A^k=\sqrt{\bar\alpha_k}A^0+\sqrt{1-\bar\alpha_k}\epsilon$，$\epsilon\sim\mathcal N(0,I)$。常见噪声预测目标是

$$
\mathcal L_{\mathrm{diff}}=
\mathbb E_{A^0,c,k,\epsilon}
\left[\|\epsilon-\epsilon_\theta(A^k,k,c)\|_2^2\right].
$$

生成时从噪声逐步迭代得到动作块；这种去噪步数属于**生成过程**，不等于机器人的控制步数。对不同推理步数，应报告真实硬件延迟和任务成功率。

<a id="course-section-50"></a>

### 9.4 Flow Matching 形式的动作块学习

以 $x_0\sim\mathcal N(0,I)$、$x_1=A$，定义 $x_s=(1-s)x_0+s x_1$ 和监督速度 $u=x_1-x_0$。一个常用的条件 Flow Matching 目标是

$$
\mathcal L_{\mathrm{FM}}=
\mathbb E_{s,x_0,x_1,c}\|v_\theta(x_s,s,c)-(x_1-x_0)\|_2^2.
$$

推理可求解 $\mathrm dx_s/\mathrm ds=v_\theta(x_s,s,c)$，从初始噪声生成动作块。该式采用噪声到数据的时间方向；具体模型若使用相反参数化，应相应调整速度符号。**完整推导、边缘速度场与条件期望证明统一放在 [Flow Matching 专章](Flow_Matching_详细推导_精简版.md)，这里聚焦数据接口和训练实施。**

<a id="course-section-51"></a>

## 10. 数据工程与评估的完整清单

<a id="course-section-52"></a>

### 10.1 轨迹质量、标签可靠性与事件分层

建议将数据集划分为成功示教、专家纠错、失败恢复、不可模仿的失败四类。可记录二值接触标签 $c_t\in\{0,1\}$，优先使用接触传感器或经过验证的物理事件作为标签；仅从图像推断的接触必须标明为估计。

对于事件偏置采样，在接触切换时刻 $t_j$ 周围提高采样权重，例如 $w_t=1+\lambda\sum_j\exp[-(t-t_j)^2/(2\sigma^2)]$。这是一种**人为训练分布选择**，并不保证比均匀采样更优；需要做事件采样与均匀采样消融。

<a id="course-section-53"></a>

### 10.2 Panda 统一实验与最小消融

统一平台使用 Panda + 桌面 + 方块，保持相机标定、动作接口和控制频率固定。比较单步 BC、chunk BC、ACT、Diffusion Policy、Flow Matching。消融实验逐次只改变数据量、执行长度 $h$、示教失败恢复数据比例、视觉噪声或推理预算。

在 $N$ 次独立测试中，成功率估计 $\hat p=\frac1N\sum_i\mathbf1[\mathrm{success}_i]$。报告置信区间、训练任务与测试任务的对象/场景拆分、接触失败率、干扰恢复率、每次推理延迟以及完整 episode 时长。训练 MSE 只作诊断，不能替代闭环成功率。

<a id="course-section-54"></a>

### 10.3 从模仿学习到 Robot Agent 的接口

本章输出的是可执行的**策略或技能接口**，例如“拿起物体”“调整工具姿态”“重新抓取”；任务级规划、成功判定与失败后的重规划统一在 [Robot Agent 专章](Robot_Agent_Workflow_ZH.md) 讨论。视觉分割与状态估计的具体模型选型统一在 [视觉感知专章](Visual_State_Estimation_ZH.md) 讨论。
