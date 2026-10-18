# 先进 Robot Agent：任务规划、技能执行与闭环恢复

<!-- course-meta:start -->
[课程首页](https://github.com/zengxiong111/Foundation_of_Embodied_AI/blob/main/README.zh-CN.md) · [English](https://github.com/zengxiong111/Foundation_of_Embodied_AI/blob/main/md/Robot_Agent_Workflow_EN.md) · [PDF](https://zengxiong111.github.io/zengxiong.github.io/files/Robot_Agent_Workflow_ZH.pdf)

| 项目 | 说明 |
|---|---|
| 适用范围 | 任务规划、验证、失败恢复与公开 Agent 路线；无真机验证 |
| 文档核验日期 | 2026-10-18 |
| 许可 | CC BY 4.0 · Xiong Zeng and contributors |

**核验边界：** 教学方法与文献综述，不代表已复现实验。
<!-- course-meta:end -->

<!-- course-toc:start -->
## 目录

- [1. Agent 与策略的分工](#course-section-01)
<!-- course-toc:end -->

## 1. Agent 与策略的分工

Robot Agent 负责高级任务解析、技能选择、世界状态更新、结果验证和失败重规划。VLA/模仿学习策略负责条件动作生成；System 0/运动控制器负责实时跟踪和物理约束。分层关系可写成 $g\to z_k\to\pi_{\rm skill}(a_t\mid o_t,z_k)$，其中 $g$ 是语言目标、$z_k$ 是参数化技能或子目标。Agent **不应直接代替高频伺服环**。

## 2. 闭环工作流

从语言目标开始，Agent 读取视觉状态与机器人状态，提出可达的子目标，调用技能，等待动作结束，再重新观察、判断是否成功并选择继续/恢复/终止。必须区分“技能执行结束”与“目标已完成”；遮挡或不确定观测不能被视为成功。

### 2.1 可执行技能接口

建议把技能声明为 (name, preconditions, arguments, controller, timeout, success_predicate, fallback)。如 pick(object_id, grasp_pose) 必须检查目标 ID、姿态置信度和碰撞风险；超时后进入重定位或安全退出，而不是无限尝试。

## 3. 代表性 Robot Agent 路线

Gemini Robotics ER 系列体现视觉语言空间推理与机器人任务编排，RAX (Robot Agent Exploration) 强调环境探索、反馈、经验/技能复用。其他重要路线包括 SayCan 的语言—技能可行性选择、Code as Policies 的程序化技能组合。应按版本和部署形态区分其能力，不能将论文演示推广为“任何机器人零标定通用自治”。

## 4. 规划、验证与信念状态

在部分可观测场景，维护近似信念 $b_t(s)=p(s_t\mid o_{\le t},a_{<t})$。任务谓词如 $V_{\rm grasp}=\mathbf1[\text{物体离开桌面且抓持稳定}]$，应同时依赖视觉、本体反馈及时间窗口。若验证置信度不足，则进入 UNKNOWN，先移动相机或重新感知，而非将 UNKNOWN 判为 SUCCESS。

## 5. 部署架构与安全

对比四类架构：LLM/VLM + 固定技能库、VLM Agent + VLA、世界模型候选计划 + 技能、探索 Agent + 经验记忆。世界模型的预测并非真实状态，必须用新观测验证。所有高层指令均需通过工作空间、碰撞、力矩/速度、延迟、置信度和急停检查，低层安全控制不能依赖云端大模型响应。

## 6. 统一 Panda 实验与消融

同一 Panda + 方块 + 桌面任务：随机位置抓取、移动到目标、物体被打落后的重定位与重抓。对照 A 固定脚本、B Agent+技能、C 增加视觉验证、D 增加失败重规划、E 增加经验记忆。统一底层策略和场景划分。成功率 $\hat p=\frac1N\sum_i\mathbf1[\mathrm{goal\ achieved}]$，同时记录推理/执行延迟、碰撞、人工干预、重复尝试、恢复成功率和 token/工具成本。

## 7. 与其他章节的知识边界

**本章只讲任务级 Agent 循环和验证/恢复。** 视觉 mask、深度和 6D 位姿请查阅[视觉状态估计](Visual_State_Estimation_ZH.md)；数据和 BC/ACT/Flow Policy 训练请查阅[数据采集与模仿学习](Data_Imitation_Learning_ZH.md)；Flow Matching 的完整数学推导见[专章](Flow_Matching_详细推导_精简版.md)；System 1/System 0 接口见[GR00T 与 SONIC](System1_System0_GR00T_SONIC_CN.md)。

资料：[RAX](https://assets.robot-agent-exploration.com/paper/rax.pdf) · [Gemini Robotics](https://deepmind.google/models/gemini-robotics/) · [SayCan](https://say-can.github.io/) · [Code as Policies](https://code-as-policies.github.io/)。以上为架构与实验设计，不构成硬件复现声明。
