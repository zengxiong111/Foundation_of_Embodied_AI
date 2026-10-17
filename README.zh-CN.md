# 具身智能基础  

**简体中文** | [English](README.md)

为未来开设「具身智能基础」课程准备的公开教学资料库，汇集机器人学习、工程工具与核心算法的中英文教程。当前处于课程筹备阶段，尚未形成正式授课安排。


## 目录

- [课程定位](#课程定位)
- [资料目录](#资料目录)
- [建议学习路线](#建议学习路线)
- [快速使用](#快速使用)
- [仓库结构](#仓库结构)
- [课程建设计划](#课程建设计划)
- [使用说明与限制](#使用说明与限制)

## 课程定位

面向希望学习具身智能、机器人控制与机器人学习的学生和研究者。建议具备 Python 编程、线性代数、概率论和基础微积分知识；控制与机器学习背景有助于理解进阶内容。

学习目标：

- 使用 Git、ROS 2 和相机标定方法组织机器人学习工程。
- 理解机器人动力学、执行器与接触如何连接策略动作和物理状态变化。
- 理解 Isaac Sim、Isaac Lab 与 MuJoCo 的分工，以及仿真、强化学习训练和跨引擎验证流程。
- 理解具身强化学习的任务分类、运动控制与 Sim-to-Real 工作流程。
- 理解 PPO、SAC 的核心推导、设计动机及适用场景。
- 理解 Flow Matching 的基本构造及推导，为后续学习生成式策略做准备。
- 理解人形机器人中 VLA 与全身运动控制的分工，以及 GR00T 与 SONIC 的系统衔接。

## 资料目录

当前收录 **11 个主题、22 个 Markdown 文件和 22 个 PDF 文件**，每个主题均提供中文和英文资料。Markdown 适合在线阅读和编辑，PDF 适合下载、打印与课堂阅读。

| 主题 | 中文 Markdown | English Markdown | 中文 PDF | English PDF |
|---|---|---|---|---|
| Git 与协作基础 | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Git_Tutorial_CN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Git_Tutorial_EN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Git_Tutorial_CN.pdf" target="_blank">PDF</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Git_Tutorial_EN.pdf" target="_blank">PDF</a> |
| ROS 2 与机器人学习 | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/ROS2_Robot_Learning_Tutorial_ZH.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/ROS2_Robot_Learning_Tutorial_EN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/ROS2_Robot_Learning_Tutorial_ZH.pdf" target="_blank">PDF</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/ROS2_Robot_Learning_Tutorial_EN.pdf" target="_blank">PDF</a> |
| 机器人相机标定 | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Robot_Camera_Calibration_Tutorial_ZH.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Robot_Camera_Calibration_Tutorial_EN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Robot_Camera_Calibration_Tutorial_ZH.pdf" target="_blank">PDF</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Robot_Camera_Calibration_Tutorial_EN.pdf" target="_blank">PDF</a> |
| 具身强化学习概览 | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Embodied_RL_Tutorial_Full_ZH.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Embodied_RL_Tutorial_Full_EN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Embodied_RL_Tutorial_Full_ZH.pdf" target="_blank">PDF</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Embodied_RL_Tutorial_Full_EN.pdf" target="_blank">PDF</a> |
| 运动控制强化学习 | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Embodied_Motion_Control_RL_Tutorial_ZH.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Embodied_Motion_Control_RL_Tutorial_EN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Embodied_Motion_Control_RL_Tutorial_ZH.pdf" target="_blank">PDF</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Embodied_Motion_Control_RL_Tutorial_EN.pdf" target="_blank">PDF</a> |
| PPO 推导与设计动机 | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/PPO_Complete_Derivation_CN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/PPO_Complete_Derivation_EN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/PPO_Complete_Derivation_CN.pdf" target="_blank">PDF</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/PPO_Complete_Derivation_EN.pdf" target="_blank">PDF</a> |
| SAC 推导与设计动机 | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/SAC_Complete_Derivation_CN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/SAC_Complete_Derivation_EN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/SAC_Complete_Derivation_CN.pdf" target="_blank">PDF</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/SAC_Complete_Derivation_EN.pdf" target="_blank">PDF</a> |
| Flow Matching 逐步推导 | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Flow_Matching_CN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Flow_Matching_Step_by_Step_Derivation_EN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Flow_Matching_CN.pdf" target="_blank">PDF</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Flow_Matching_Step_by_Step_Derivation_EN.pdf" target="_blank">PDF</a> |
| 人形机器人 System 1 / System 0：GR00T 与 SONIC | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/System1_System0_GR00T_SONIC_CN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/System1_System0_GR00T_SONIC_EN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/System1_System0_GR00T_SONIC_CN.pdf" target="_blank">PDF</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/System1_System0_GR00T_SONIC_EN.pdf" target="_blank">PDF</a> |
| Isaac Sim、Isaac Lab 与 MuJoCo 仿真及强化学习 | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Isaac_MuJoCo_RL_%E5%AE%8C%E6%95%B4%E6%95%99%E7%A8%8B_%E4%B8%AD%E6%96%87.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Isaac_MuJoCo_RL_Complete_Guide_EN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Isaac_MuJoCo_RL_%E5%AE%8C%E6%95%B4%E6%95%99%E7%A8%8B_%E4%B8%AD%E6%96%87.pdf" target="_blank">PDF</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Isaac_MuJoCo_RL_Complete_Guide_EN.pdf" target="_blank">PDF</a> |
| 具身智能中的机器人动力学 | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Robot_Dynamics_Embodied_CN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Robot_Dynamics_Embodied_EN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Robot_Dynamics_Embodied_CN.pdf" target="_blank">PDF</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Robot_Dynamics_Embodied_EN.pdf" target="_blank">PDF</a> |

## 建议学习路线

以下为课程筹备建议，可按学生基础调整，并非正式课表。

1. **工程准备**：Git 与协作基础 → ROS 2 与机器人学习。
2. **感知基础**：相机内参、外参、手眼标定与坐标系。
3. **动力学与仿真**：机器人动力学 → Isaac Sim、Isaac Lab 与 MuJoCo，建立动作、执行器、接触与仿真训练之间的联系。
4. **任务全景**：具身强化学习概览 → 运动控制强化学习。
5. **算法推导**：PPO → SAC，比较 on-policy 与 off-policy 学习。
6. **生成式方法**：Flow Matching 逐步推导，衔接后续策略学习专题。
7. **系统架构**：System 1 / System 0，结合 GR00T 与 SONIC 理解高层动作生成和低层全身控制。
8. **综合实践（待建设）**：结合感知、通信与控制设计课程项目。

## 快速使用

1. 从上方资料目录选择主题、语言和格式，链接直接指向个人网站 `files/` 目录中的对应文件。PDF 可在浏览器中阅读；Markdown 文件可能显示为文本或下载，具体取决于浏览器。
2. 如需本地阅读或参与资料整理，在终端执行：

```bash
git clone https://github.com/zengxiong111/Foundation_of_Embodied_AI.git
cd Foundation_of_Embodied_AI
```

3. 在 `md/` 中编辑教程，在 `pdf/` 中查看对应阅读版本。运行教程示例前，请按各篇文档准备依赖环境；本仓库没有统一安装或训练入口。

## 仓库结构

```text
Foundation_of_Embodied_AI/
├── README.md       # English course overview (default)
├── README.zh-CN.md # 中文课程入口
├── md/             # 22 份中英文 Markdown 教程
└── pdf/            # 22 份中英文 PDF 资料
```

仓库标识为 `Foundation_of_Embodied_AI`，课程中文名为「具身智能基础」，英文展示名为 “Foundation of Embodied AI”。现有资料保留原文件名（包括部分 PDF 的 `-1` 后缀），所有原有文件均完整收录。

## 课程建设计划

以下内容计划逐步补充，当前尚未交付：

- 正式教学大纲、课时分配与每讲学习目标。
- 配套课件、阅读清单与课堂演示。
- 分层作业、实验环境与可复现实验代码。
- 综合课程项目、评价标准与学生贡献指南。

## 使用说明与限制

- 当前主要交付阅读资料；教程中的示例代码不代表已在统一环境中完成验证。
- 现有中英文资料按主题配对，PDF 与 Markdown 的逐段一致性尚未审校。
- 软件版本、设备接口及硬件参数以各教程说明和实际环境为准；硬件实验需独立完成设备核对与安全验证。
- 后续修改教程时同步维护中英文内容，并更新对应 PDF 和本索引。
- 本仓库当前未声明统一开源许可证；公开可访问不等于授予任意再分发或商业使用许可。引用资料请注明来源，并遵守其中第三方材料的许可要求。
