# 先进 Robot Agent：任务规划、技能执行与闭环恢复

<!-- course-meta:start -->
[课程首页](https://github.com/zengxiong111/Foundation_of_Embodied_AI/blob/main/README.zh-CN.md) · [English](https://github.com/zengxiong111/Foundation_of_Embodied_AI/blob/main/md/Robot_Agent_Workflow_EN.md) · [PDF](https://zengxiong111.github.io/zengxiong.github.io/files/Robot_Agent_Workflow_ZH.pdf)

| 项目 | 说明 |
|---|---|
| 适用范围 | RAX、Gemini Robotics ER、SayCan、Code as Policies：概念比较 |
| 文档核验日期 | 2026-10-18 |
| 许可 | CC BY 4.0 · Xiong Zeng and contributors |

**核验边界：** 公开研究工作流与方法综述；未进行真机部署验证。 版本与固定源码引用见[版本说明](https://github.com/zengxiong111/Foundation_of_Embodied_AI/blob/main/VERSIONS.zh-CN.md)。原创教材采用 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)，第三方引用、图片及代码遵循其原有许可；注明来源不等于拥有再许可权。
<!-- course-meta:end -->

<!-- course-toc:start -->
## 目录

- [1. Agent 与策略的分工](#course-section-01)
- [2. 闭环工作流](#course-section-02)
  - [2.1 可执行技能接口](#course-section-03)
- [3. 代表性 Robot Agent 路线](#course-section-04)
- [4. 规划、验证与信念状态](#course-section-05)
- [5. 部署架构与安全](#course-section-06)
- [6. 统一 Panda 实验与消融](#course-section-07)
- [7. 系统化设计：从 Agent 的输出到机器人的约束动作](#course-section-08)
  - [7.1 层级工具接口与任务状态](#course-section-09)
  - [7.2 真实闭环包含“验收”，而不止“反馈”](#course-section-10)
  - [7.3 多个频率不是同一个环](#course-section-11)
- [8. 前沿路线的具体工作流与关键差异](#course-section-12)
  - [8.1 RAX：在线试错 → 编译程序 → 留出场景验证](#course-section-13)
  - [8.2 Gemini Robotics ER 2：持续视频 + 具身推理 + 工具编排](#course-section-14)
  - [8.3 RAX 与 ER 2 并非必须二选一](#course-section-15)
- [9. 成功验证、技能记忆与风险控制](#course-section-16)
  - [9.1 状态估计之外的任务验证](#course-section-17)
  - [9.2 候选技能的收益与风险](#course-section-18)
  - [9.3 技能库只能保存“带证据的成功”](#course-section-19)
- [10. 错误恢复与真实机器人接口](#course-section-20)
  - [10.1 失败诊断需要分层](#course-section-21)
  - [10.2 可复用的 Panda / 工具案例](#course-section-22)
- [11. 训练前后应该怎样评价 Robot Agent](#course-section-23)
  - [11.1 最小消融设计](#course-section-24)
  - [11.2 评价指标与边界](#course-section-25)
- [12. 与其他章节的知识边界](#course-section-26)
<!-- course-toc:end -->

<a id="course-section-01"></a>

## 1. Agent 与策略的分工

Robot Agent 负责高级任务解析、技能选择、世界状态更新、结果验证和失败重规划。VLA/模仿学习策略负责条件动作生成；System 0/运动控制器负责实时跟踪和物理约束。分层关系可写成 $g\to z_k\to\pi_{\rm skill}(a_t\mid o_t,z_k)$，其中 $g$ 是语言目标、$z_k$ 是参数化技能或子目标。Agent **不应直接代替高频伺服环**。

<a id="course-section-02"></a>

## 2. 闭环工作流

从语言目标开始，Agent 读取视觉状态与机器人状态，提出可达的子目标，调用技能，等待动作结束，再重新观察、判断是否成功并选择继续/恢复/终止。必须区分“技能执行结束”与“目标已完成”；遮挡或不确定观测不能被视为成功。

<a id="course-section-03"></a>

### 2.1 可执行技能接口

建议把技能声明为 (name, preconditions, arguments, controller, timeout, success_predicate, fallback)。如 pick(object_id, grasp_pose) 必须检查目标 ID、姿态置信度和碰撞风险；超时后进入重定位或安全退出，而不是无限尝试。

<a id="course-section-04"></a>

## 3. 代表性 Robot Agent 路线

Gemini Robotics ER 系列体现视觉语言空间推理与机器人任务编排，RAX (Robot Agent Exploration) 强调环境探索、反馈、经验/技能复用。其他重要路线包括 SayCan 的语言—技能可行性选择、Code as Policies 的程序化技能组合。应按版本和部署形态区分其能力，不能将论文演示推广为“任何机器人零标定通用自治”。

<a id="course-section-05"></a>

## 4. 规划、验证与信念状态

在部分可观测场景，维护近似信念 $b_t(s)=p(s_t\mid o_{\le t},a_{<t})$。任务谓词如 $V_{\rm grasp}=\mathbf1[\text{物体离开桌面且抓持稳定}]$，应同时依赖视觉、本体反馈及时间窗口。若验证置信度不足，则进入 UNKNOWN，先移动相机或重新感知，而非将 UNKNOWN 判为 SUCCESS。

<a id="course-section-06"></a>

## 5. 部署架构与安全

对比四类架构：LLM/VLM + 固定技能库、VLM Agent + VLA、世界模型候选计划 + 技能、探索 Agent + 经验记忆。世界模型的预测并非真实状态，必须用新观测验证。所有高层指令均需通过工作空间、碰撞、力矩/速度、延迟、置信度和急停检查，低层安全控制不能依赖云端大模型响应。

<a id="course-section-07"></a>

## 6. 统一 Panda 实验与消融

同一 Panda + 方块 + 桌面任务：随机位置抓取、移动到目标、物体被打落后的重定位与重抓。对照 A 固定脚本、B Agent+技能、C 增加视觉验证、D 增加失败重规划、E 增加经验记忆。统一底层策略和场景划分。成功率 $\hat p=\frac1N\sum_i\mathbf1[\mathrm{goal\ achieved}]$，同时记录推理/执行延迟、碰撞、人工干预、重复尝试、恢复成功率和 token/工具成本。


<a id="course-section-08"></a>

## 7. 系统化设计：从 Agent 的输出到机器人的约束动作

<a id="course-section-09"></a>

### 7.1 层级工具接口与任务状态

建议以结构化状态表而非长篇自由文本作为任务记忆：物体 ID、相机和物体姿态时间戳、置信度、已完成步骤、夹爪状态、工具返回值、重试次数和安全事件。每个技能 $\omega=(I_\omega,\pi_\omega,\beta_\omega)$ 包含可启动集合 $I_\omega$、技能内部动作策略 $\pi_\omega$ 和终止条件 $\beta_\omega$，并额外定义可独立计算的成功验证器与恢复策略。

Agent 调用的工具可包括目标搜索、六维姿态查询、基座/末端导航、抓手操作、运行 VLA 技能、进度检验及安全停止。每个 API 都应明确：坐标系、单位、参数范围、执行超时、动作权限、可观测的完成条件、错误码和审计日志。**接到“命令已提交”的 API 回复，不能把它当成“抓取成功”的证明。**

<a id="course-section-10"></a>

### 7.2 真实闭环包含“验收”，而不止“反馈”

~~~mermaid
flowchart TD
  G["自然语言任务"] --> P["任务分解与技能选择"]
  P --> O["视觉 / 本体观测"]
  O --> C{"前置条件与安全检查"}
  C -- "失败" --> R["补充观察 / 人工介入"]
  C -- "通过" --> A["调用技能 / VLA / 规划器"]
  A --> S["受限的底层执行器"]
  S --> V["独立进度与结果验证"]
  V -- "完成" --> M["提交技能证据与下一子任务"]
  V -- "未知 / 失败" --> F["局部诊断与恢复"]
  F --> O
  M --> P
~~~

校验应该覆盖工具**动作已结束**、目标**物理上已达成**、当前状态**可安全进行下一步**三个不同判断。模型即使输出“任务完成”，也不能绕过物理后置条件。视觉估计的数学原理见[视觉课程](Visual_State_Estimation_ZH.md)。

<a id="course-section-11"></a>

### 7.3 多个频率不是同一个环

高层 Agent 根据任务事件调用工具；视觉估计器按相机帧更新；动作生成器可能输出较长 Action Chunk；关节伺服在更高的独立频率运行。不得把 Agent 的 token 流当作执行器直接命令，也不能把离线脚本的成功误认为闭环自主执行成功。各层均需 timeout、stale-state 检查和安全降级。频率必须以具体硬件的实测延迟为依据。

<a id="course-section-12"></a>

## 8. 前沿路线的具体工作流与关键差异

<a id="course-section-13"></a>

### 8.1 RAX：在线试错 → 编译程序 → 留出场景验证

[RAX](https://robot-agent-exploration.com/) 的公开视频和仓库展示了一条不同于大规模模仿学习的路线：通用 Coding Agent（例如 Codex CLI、Claude Code）通过机器人工具接口观察并试探环境，记录动作前置条件和结果，再编译为可独立运行的控制程序。三阶段可归纳为 **Explore → Compile → Evaluate**。系统采用隔离的工具服务和机器人 backend，编译后的程序可以脱离 Coding Agent 独立执行，减少部署推理成本。

RAX 发布的示例涵盖不同机械形态及仿真/真机，研究报告探索回合增加带来未见布局上的成功率提升；具体实验数字属于其数据集和任务，不能推广为任意设备上的保证。重要技术贡献是跨 episode 技能记忆、前置条件、检查代码与无人干预的验证证据，**不应描述成“RAX 训练了另一个流模型”**。

<a id="course-section-14"></a>

### 8.2 Gemini Robotics ER 2：持续视频 + 具身推理 + 工具编排

[Gemini Robotics ER 2 官方介绍](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/gemini-robotics-er-2/)将其定位为机器人的高层 Embodied Reasoner：用视频与语言理解当前任务进度，选择下一步工具或低层 VLA，支持复杂步骤、任务中断后的修正以及多机器人协作。它的输出主要用于任务层决策和工具参数，**不是直接每个伺服周期输出电机扭矩**。

与一次性静态规划不同，持续视频输入使 Agent 可以在执行过程中评估“是否已经完成当前步骤”。进度分类、关键事件定位和失败识别是验证机制的组成部分，但需再结合机器人状态与具体任务后置条件。根据官方 Model Card，其能力、API 和安全限制需要随具体版本核对。

<a id="course-section-15"></a>

### 8.3 RAX 与 ER 2 并非必须二选一

| 维度 | RAX | Gemini Robotics ER 2 |
|---|---|---|
| 核心机制 | 探索机器人 API 并把经验编译为技能程序 | 多模态 VLM 在任务期间推理和调用工具 |
| 如何适应新场景 | 交互探索、经验记忆、验证技能 | 视觉/语言泛化、实时任务进度与重新规划 |
| 部署主要开销 | 编译技能可以少调用 Agent | 持续的高层推理与流式观测 |
| 与低层策略关系 | 技能可调用确定性规划或 learned policy | 对接 VLA、导航和其他技能 API |
| 主要挑战 | 安全探索、技能泛化、编译的错误分支 | 视频误判、延迟、工具权限与安全护栏 |

潜在的组合方式是 RAX 式探索生成可审计技能，ER 式高层 Reasoner 调度技能；这是教学上的系统构想，不是任何原论文已经验证的组合结果。

<a id="course-section-16"></a>

## 9. 成功验证、技能记忆与风险控制

<a id="course-section-17"></a>

### 9.1 状态估计之外的任务验证

对于每项技能，定义独立后置条件 $V_\omega(o_{0:t},q_{0:t},g)$。抓取应检查物体是否随末端运动且保持稳定，而不只是“闭合过夹爪”；插装应检测物体相对位姿、接触与最终固定状态，而不只是末端到达目标位置。验证可能产生 SUCCESS、FAILURE 或 UNKNOWN：**UNKNOWN 需要追加感知，不能直接当成功。**

<a id="course-section-18"></a>

### 9.2 候选技能的收益与风险

用以下概念性目标表达 Agent 的选择（非 RAX 或 ER 2 内部优化公式）：

$
\omega_t^\star=\arg\max_{\omega\in\Omega(b_t)}
\left[
\mathbb E[R_{\rm goal}\mid b_t,\omega]
-\lambda_c C(\omega)-\lambda_r {\rm Risk}(b_t,\omega)
\right].
$

$\Omega(b_t)$ 必须先满足技能前置条件和机器人安全许可。执行层可进行动作限制、禁区判定及安全停止；语言模型建议不构成安全认证。

<a id="course-section-19"></a>

### 9.3 技能库只能保存“带证据的成功”

每个 skill 应包含适用机器人、相机/URDF/控制器版本、前置条件、运行日志、失败案例、成功验证器及独立测试次数。RAX Skills Library 展示了将编译程序与其无人干预的运行证据绑定的设计。更换灵巧手或外参后应重新验证，而不是只根据自然语言技能名称直接复用。

<a id="course-section-20"></a>

## 10. 错误恢复与真实机器人接口

<a id="course-section-21"></a>

### 10.1 失败诊断需要分层

| 故障 | 可观测证据 | 局部恢复 |
|---|---|---|
| 目标丢失 | LOST、Mask 异常、图像过期 | 暂停危险动作并重新定位 |
| 抓取滑落 | 相对位姿突变、夹爪异常 | 安全退让、重新抓取 |
| 不可达 | IK 失败、关节/工作区限制 | 更换接近位姿或技能 |
| 工具调用超时 | 无完成事件、执行延迟过长 | 停止、检查状态而不盲重试 |
| 任务进度未知 | 多源验证器不一致 | 增加观测或请求人工确认 |
| 人工急停 | 物理停止信号 | 立即停机，禁止自主恢复 |

真实设备接入时建议 Agent 仅能通过审计的机器人服务访问硬件；供应商低层安全栈、急停和工作空间限制保留独立权威。未知状态下不能让 Agent 任意“试探”高接触力或高速度命令。

<a id="course-section-22"></a>

### 10.2 可复用的 Panda / 工具案例

在 Panda + 方块 + 桌面任务中，控制变量保持不变，把随机初始位、遮挡、物体被推走、抓取失败作为不同测试切片。对应流程是定位 → 6D/深度观测 → 选择 grasp skill → 执行并验证 → 放置 → 最终验证；失败只重做受影响的子任务。扩展到灵巧手螺丝刀时，要增加接触稳定性、工具姿态与重新抓取判据。前两章解释感知与策略训练，本章只负责跨技能的状态管理与恢复。

<a id="course-section-23"></a>

## 11. 训练前后应该怎样评价 Robot Agent

<a id="course-section-24"></a>

### 11.1 最小消融设计

在相同相机、控制器和技能 API 下比较：A）固定 FSM / 行为树，B）一次性 VLM 计划，C）Agent + 执行反馈，D）C + 独立成功验证，E）D + 失败重规划，F）E + 持久技能记忆；另外比较探索式 Agent 与编译后技能的部署成本。**所有组保持同一低层策略和随机化测试分布**，否则无法归因。

<a id="course-section-25"></a>

### 11.2 评价指标与边界

报告未见场景成功率、步骤完成率、失败检测召回率、恢复成功率、工具调用次数、人工介入数、Agent 模型调用与 token 成本、实际墙钟时间、控制延迟和危险动作阻断次数。把探索预算和部署预算分开。Agent 工作流在高风险装配或协作任务中必须有明确操作者接管和最大重试次数。


<a id="course-section-26"></a>

## 12. 与其他章节的知识边界

**本章只讲任务级 Agent 循环和验证/恢复。** 视觉 mask、深度和 6D 位姿请查阅[视觉状态估计](Visual_State_Estimation_ZH.md)；数据和 BC/ACT/Flow Policy 训练请查阅[数据采集与模仿学习](Data_Imitation_Learning_ZH.md)；Flow Matching 的完整数学推导见[专章](Flow_Matching_详细推导_精简版.md)；System 1/System 0 接口见[GR00T 与 SONIC](System1_System0_GR00T_SONIC_CN.md)。

资料：[RAX](https://robot-agent-exploration.com/) · [RAX 代码](https://github.com/robot-agent-exploration/rax) · [技能库](https://github.com/robot-agent-exploration/skills-lib) · [Gemini Robotics ER 2](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/gemini-robotics-er-2/) · [SayCan](https://say-can.github.io/) · [Code as Policies](https://code-as-policies.github.io/)。以上为架构与实验设计，不构成硬件复现声明。
