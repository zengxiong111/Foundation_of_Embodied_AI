# 具身智能基础

**简体中文** | [English](README.md)

为未来「具身智能基础」课程准备的中英文学习资料，基于我们的项目经验，在 ChatGPT 的协助下生成。

本课程分为 **Part I：理论与算法** 和 **Part II：工程与系统**，共 **6 个模块、20 个主题**。理论部分介绍基础原理、机器人学习算法和具身基础模型；工程部分介绍开发工具、仿真训练、实时推理与机器人系统集成。两部分可并行学习，不要求严格顺序。

**推荐路径：** 理论研究 / 机器人工程 / 具身智能全栈。详见 [课程大纲](syllabus/COURSE_MAP_ZH.md) · <a href="https://zengxiong111.github.io/zengxiong.github.io/files/COURSE_MAP_ZH.pdf" target="_blank">PDF</a>。

**目录：** [课程结构](#课程结构) · [课程资料](#课程资料) · [如何学习](#如何学习) · [使用、许可与反馈](#使用许可与反馈)

## 课程结构

| 部分 | 模块 | 主题 | 学习重点 |
|---|---|---|---|
| Part I：理论与算法 | A. 基础理论 | 深度学习、视觉、动力学、控制 | 理解机器人学习、感知、物理与控制的基本原理 |
| Part I：理论与算法 | B. 机器人学习算法 | PPO、SAC、基于模型的强化学习、模仿学习、流匹配 | 理解策略如何通过强化学习、模仿与生成建模得到 |
| Part I：理论与算法 | C. 具身基础模型 | GR00T/SONIC、世界动作模型、后训练 | 理解分层架构、VLA/WAM 与监督及强化学习后训练 |
| Part II：工程与系统 | D. 机器人开发基础 | Git、ROS 2、相机标定 | 管理代码、组织 ROS 2 系统并完成传感器标定 |
| Part II：工程与系统 | E. 仿真与策略训练 | 仿真与强化学习、运动控制强化学习 | 搭建仿真、定义任务并训练和评估运动策略 |
| Part II：工程与系统 | F. 训练、部署与系统集成 | 高效训练与实时推理、通信、机器人智能体 | 满足计算与实时约束，运行机器人反馈闭环 |

## 课程资料

20 个主题，40 份 Markdown 和 40 份 PDF。点击表格中的链接阅读或下载。原编号是稳定主题 ID，不代表阅读顺序。

### Part I：理论与算法（12 个主题）

| 模块 | 原编号 | 主题 | 中文 Markdown | 英文 Markdown | 中文 PDF | 英文 PDF |
|---|---|---|---|---|---|---|
| A. 基础理论 | 4 | 深度学习基础 | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Deep_Learning_Foundations_ZH.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Deep_Learning_Foundations_EN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Deep_Learning_Foundations_ZH.pdf" target="_blank">PDF</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Deep_Learning_Foundations_EN.pdf" target="_blank">PDF</a> |
| A. 基础理论 | 5 | 视觉基础模型 | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Visual_State_Estimation_ZH.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Visual_State_Estimation_EN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Visual_State_Estimation_ZH.pdf" target="_blank">PDF</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Visual_State_Estimation_EN.pdf" target="_blank">PDF</a> |
| A. 基础理论 | 6 | 机器人动力学 | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Robot_Dynamics_Embodied_CN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Robot_Dynamics_Embodied_EN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Robot_Dynamics_Embodied_CN.pdf" target="_blank">PDF</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Robot_Dynamics_Embodied_EN.pdf" target="_blank">PDF</a> |
| A. 基础理论 | 7 | 底层控制 | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Robot_Low_Level_Control_ZH.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Robot_Low_Level_Control_EN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Robot_Low_Level_Control_ZH.pdf" target="_blank">PDF</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Robot_Low_Level_Control_EN.pdf" target="_blank">PDF</a> |
| B. 机器人学习算法 | 10 | PPO / 行动者–评论家 | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/PPO_Complete_Derivation_CN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/PPO_Complete_Derivation_EN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/PPO_Complete_Derivation_CN.pdf" target="_blank">PDF</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/PPO_Complete_Derivation_EN.pdf" target="_blank">PDF</a> |
| B. 机器人学习算法 | 11 | Q 学习 / SAC | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/SAC_Complete_Derivation_CN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/SAC_Complete_Derivation_EN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/SAC_Complete_Derivation_CN.pdf" target="_blank">PDF</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/SAC_Complete_Derivation_EN.pdf" target="_blank">PDF</a> |
| B. 机器人学习算法 | 12 | 基于模型的强化学习 | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Model_Based_RL_Differentiable_World_Models_ZH.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Model_Based_RL_Differentiable_World_Models_EN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Model_Based_RL_Differentiable_World_Models_ZH.pdf" target="_blank">PDF</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Model_Based_RL_Differentiable_World_Models_EN.pdf" target="_blank">PDF</a> |
| B. 机器人学习算法 | 13 | 模仿学习 | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Data_Imitation_Learning_ZH.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Data_Imitation_Learning_EN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Data_Imitation_Learning_ZH.pdf" target="_blank">PDF</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Data_Imitation_Learning_EN.pdf" target="_blank">PDF</a> |
| B. 机器人学习算法 | 14 | 流匹配 | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Flow_Matching_CN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Flow_Matching_Step_by_Step_Derivation_EN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Flow_Matching_CN.pdf" target="_blank">PDF</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Flow_Matching_Step_by_Step_Derivation_EN.pdf" target="_blank">PDF</a> |
| C. 具身基础模型 | 15 | GR00T 与 SONIC | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/System1_System0_GR00T_SONIC_CN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/System1_System0_GR00T_SONIC_EN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/System1_System0_GR00T_SONIC_CN.pdf" target="_blank">PDF</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/System1_System0_GR00T_SONIC_EN.pdf" target="_blank">PDF</a> |
| C. 具身基础模型 | 16 | 世界动作模型 | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/World_Action_Models_ZH.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/World_Action_Models_EN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/World_Action_Models_ZH.pdf" target="_blank">PDF</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/World_Action_Models_EN.pdf" target="_blank">PDF</a> |
| C. 具身基础模型 | 18 | 后训练 | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Embodied_Post_Training_ZH.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Embodied_Post_Training_EN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Embodied_Post_Training_ZH.pdf" target="_blank">PDF</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Embodied_Post_Training_EN.pdf" target="_blank">PDF</a> |

### Part II：工程与系统（8 个主题）

| 模块 | 原编号 | 主题 | 中文 Markdown | 英文 Markdown | 中文 PDF | 英文 PDF |
|---|---|---|---|---|---|---|
| D. 机器人开发基础 | 1 | Git 基础 | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Git_Tutorial_CN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Git_Tutorial_EN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Git_Tutorial_CN.pdf" target="_blank">PDF</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Git_Tutorial_EN.pdf" target="_blank">PDF</a> |
| D. 机器人开发基础 | 2 | ROS 2 基础 | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/ROS2_Robot_Learning_Tutorial_ZH.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/ROS2_Robot_Learning_Tutorial_EN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/ROS2_Robot_Learning_Tutorial_ZH.pdf" target="_blank">PDF</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/ROS2_Robot_Learning_Tutorial_EN.pdf" target="_blank">PDF</a> |
| D. 机器人开发基础 | 3 | 相机标定 | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Robot_Camera_Calibration_Tutorial_ZH.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Robot_Camera_Calibration_Tutorial_EN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Robot_Camera_Calibration_Tutorial_ZH.pdf" target="_blank">PDF</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Robot_Camera_Calibration_Tutorial_EN.pdf" target="_blank">PDF</a> |
| E. 仿真与策略训练 | 8 | 仿真与强化学习 | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Isaac_MuJoCo_RL_%E5%AE%8C%E6%95%B4%E6%95%99%E7%A8%8B_%E4%B8%AD%E6%96%87.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Isaac_MuJoCo_RL_Complete_Guide_EN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Isaac_MuJoCo_RL_%E5%AE%8C%E6%95%B4%E6%95%99%E7%A8%8B_%E4%B8%AD%E6%96%87.pdf" target="_blank">PDF</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Isaac_MuJoCo_RL_Complete_Guide_EN.pdf" target="_blank">PDF</a> |
| E. 仿真与策略训练 | 9 | 运动控制强化学习 | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Embodied_Motion_Control_RL_Tutorial_ZH.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Embodied_Motion_Control_RL_Tutorial_EN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Embodied_Motion_Control_RL_Tutorial_ZH.pdf" target="_blank">PDF</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Embodied_Motion_Control_RL_Tutorial_EN.pdf" target="_blank">PDF</a> |
| F. 训练、部署与系统集成 | 17 | 高效训练与实时推理 | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Efficient_Training_Real_Time_Inference_ZH.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Efficient_Training_Real_Time_Inference_EN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Efficient_Training_Real_Time_Inference_ZH.pdf" target="_blank">PDF</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Efficient_Training_Real_Time_Inference_EN.pdf" target="_blank">PDF</a> |
| F. 训练、部署与系统集成 | 19 | 通信与实时系统 | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Robot_Communication_Real_Time_Systems_ZH.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Robot_Communication_Real_Time_Systems_EN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Robot_Communication_Real_Time_Systems_ZH.pdf" target="_blank">PDF</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Robot_Communication_Real_Time_Systems_EN.pdf" target="_blank">PDF</a> |
| F. 训练、部署与系统集成 | 20 | 机器人智能体 | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Robot_Agent_Workflow_RPent_ZH.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Robot_Agent_Workflow_RPent_EN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Robot_Agent_Workflow_RPent_ZH.pdf" target="_blank">PDF</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Robot_Agent_Workflow_RPent_EN.pdf" target="_blank">PDF</a> |

## 如何学习

- **理论研究：** 深度学习基础 → 模仿学习与生成式策略 → 强化学习与基于模型的优化 → VLA/WAM 与后训练；按需补足视觉、动力学与控制。
- **机器人工程：** Git/ROS 2（按需）→ 相机标定 → 动力学/底层控制 → 仿真与强化学习 → 实时推理/通信 → 机器人智能体。
- **具身智能全栈：** 交叉学习两部分，以“感知 → 策略 → 控制 → 机器人 → 反馈”连接训练与部署。

详细路线与交叉关系见 [课程大纲](syllabus/COURSE_MAP_ZH.md)。

## 使用、许可与反馈

欢迎大家在适用许可范围内下载、修改和提交 PR！反馈或另行商业授权申请请发送至 [zengx3@sustech.edu.cn](mailto:zengx3@sustech.edu.cn)。

**新增原创内容：** 自 2026-10-22 起，在[许可分界提交](https://github.com/zengxiong111/Foundation_of_Embodied_AI/tree/1456e30d30c54b3cb999e9468b677a2640a360cf)之后首次发布、且许可人有权授权的新增原创内容，采用 [CC BY-NC-SA 4.0](LICENSE)：

- **署名：** 注明创作者、提供来源和许可链接，并说明修改；不得暗示获得作者背书。
- **非商业性使用：** 本许可不授权商业使用；商业使用须向相关权利人另行申请授权。
- **相同方式共享：** 对这些新增内容进行改编后再分享，须采用 CC BY-NC-SA 4.0 或官方认可的兼容许可。
