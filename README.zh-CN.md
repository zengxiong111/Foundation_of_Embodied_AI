# 具身智能基础  

**简体中文** | [English](README.md)

为未来开设「具身智能基础」课程准备的公开教学资料库，汇集机器人学习、工程工具与核心算法的中英文教程。当前处于课程筹备阶段，尚未形成正式授课安排。

**内容来源说明：** 本仓库资料基于我们的项目经验，在 ChatGPT 的协助下生成。

## 目录

- [课程定位](#课程定位)
- [资料目录](#资料目录)
- [建议学习路线](#建议学习路线)
- [快速使用](#快速使用)
- [仓库结构](#仓库结构)
- [使用许可](#使用许可)
- [版本与核验范围](#版本与核验范围)
- [构建校验与发布](#构建校验与发布)
- [课程建设计划](#课程建设计划)
- [参与贡献与反馈](#参与贡献与反馈)
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
| Git 基础 | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Git_Tutorial_CN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Git_Tutorial_EN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Git_Tutorial_CN.pdf" target="_blank">PDF</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Git_Tutorial_EN.pdf" target="_blank">PDF</a> |
| ROS 2 基础 | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/ROS2_Robot_Learning_Tutorial_ZH.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/ROS2_Robot_Learning_Tutorial_EN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/ROS2_Robot_Learning_Tutorial_ZH.pdf" target="_blank">PDF</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/ROS2_Robot_Learning_Tutorial_EN.pdf" target="_blank">PDF</a> |
| 相机标定 | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Robot_Camera_Calibration_Tutorial_ZH.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Robot_Camera_Calibration_Tutorial_EN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Robot_Camera_Calibration_Tutorial_ZH.pdf" target="_blank">PDF</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Robot_Camera_Calibration_Tutorial_EN.pdf" target="_blank">PDF</a> |
| 机器人动力学 | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Robot_Dynamics_Embodied_CN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Robot_Dynamics_Embodied_EN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Robot_Dynamics_Embodied_CN.pdf" target="_blank">PDF</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Robot_Dynamics_Embodied_EN.pdf" target="_blank">PDF</a> |
| Isaac / MuJoCo 仿真 | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Isaac_MuJoCo_RL_%E5%AE%8C%E6%95%B4%E6%95%99%E7%A8%8B_%E4%B8%AD%E6%96%87.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Isaac_MuJoCo_RL_Complete_Guide_EN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Isaac_MuJoCo_RL_%E5%AE%8C%E6%95%B4%E6%95%99%E7%A8%8B_%E4%B8%AD%E6%96%87.pdf" target="_blank">PDF</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Isaac_MuJoCo_RL_Complete_Guide_EN.pdf" target="_blank">PDF</a> |
| 具身强化学习 | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Embodied_RL_Tutorial_Full_ZH.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Embodied_RL_Tutorial_Full_EN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Embodied_RL_Tutorial_Full_ZH.pdf" target="_blank">PDF</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Embodied_RL_Tutorial_Full_EN.pdf" target="_blank">PDF</a> |
| 运动控制 RL | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Embodied_Motion_Control_RL_Tutorial_ZH.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Embodied_Motion_Control_RL_Tutorial_EN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Embodied_Motion_Control_RL_Tutorial_ZH.pdf" target="_blank">PDF</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Embodied_Motion_Control_RL_Tutorial_EN.pdf" target="_blank">PDF</a> |
| PPO | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/PPO_Complete_Derivation_CN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/PPO_Complete_Derivation_EN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/PPO_Complete_Derivation_CN.pdf" target="_blank">PDF</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/PPO_Complete_Derivation_EN.pdf" target="_blank">PDF</a> |
| SAC | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/SAC_Complete_Derivation_CN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/SAC_Complete_Derivation_EN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/SAC_Complete_Derivation_CN.pdf" target="_blank">PDF</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/SAC_Complete_Derivation_EN.pdf" target="_blank">PDF</a> |
| Flow Matching | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Flow_Matching_CN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Flow_Matching_Step_by_Step_Derivation_EN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Flow_Matching_CN.pdf" target="_blank">PDF</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Flow_Matching_Step_by_Step_Derivation_EN.pdf" target="_blank">PDF</a> |
| GR00T 与 SONIC | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/System1_System0_GR00T_SONIC_CN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/System1_System0_GR00T_SONIC_EN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/System1_System0_GR00T_SONIC_CN.pdf" target="_blank">PDF</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/System1_System0_GR00T_SONIC_EN.pdf" target="_blank">PDF</a> |

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
├── pdf/            # 22 份 PDF 与当前构建清单
├── materials.json  # 资料、版本与网站映射
├── VERSIONS.md     # 英文版本说明（配套中文版）
├── scripts/        # 同步、构建、校验与发布
├── tests/          # 校验器测试
├── LICENSE         # 原创教材：CC BY 4.0
└── LICENSE-CODE    # 构建工具：MIT
```

仓库标识为 `Foundation_of_Embodied_AI`，课程中文名为「具身智能基础」，英文展示名为 “Foundation of Embodied AI”。现有资料保留原文件名（包括部分 PDF 的 `-1` 后缀），所有原有文件均完整收录。

## 使用许可

| 内容 | 许可与适用范围 |
|---|---|
| 本仓库有权许可的原创教材正文、公式说明和自绘图示；对应 Markdown 与 PDF | [CC BY 4.0](LICENSE)；允许复制、修改、再分发和商业使用，需署名、链接许可并注明修改 |
| 新增构建、校验、同步和发布工具（`scripts/`、`tests/` 与构建配置） | [MIT](LICENSE-CODE)；保留版权和许可声明 |
| 第三方论文、引用段落、图片、代码、模型、数据和商标 | 仍遵循原有许可；不因本仓库引用而自动纳入 CC BY 或 MIT |

建议署名：**Xiong Zeng and contributors, Foundation of Embodied AI**，附[仓库地址](https://github.com/zengxiong111/Foundation_of_Embodied_AI)、[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 和修改说明。引用原论文或官方代码时同时保留其来源。法律正文保留标准英文原文，本节提供中文范围说明。欢迎提交你有权按相应许可贡献的内容；提交 PR 时说明来源和第三方限制。

## 版本与核验范围

[中英文版本矩阵](VERSIONS.zh-CN.md)记录每个主题的适用范围、2026-10-18 文档核验日期和固定上游 commit。核验内容是文档结构与引用快照，**不代表安装、训练或真机运行通过**。复现实验仍需锁定完整环境、资产、配置与 checkpoint。

`materials.json` 是主题、语言配对、版本和网站文件映射的统一清单；`npm run sync` 据此更新教程导航、可点击目录和版本说明。22 份教程均提供课程首页、另一语言和 PDF 入口。资料表中的个人网站直达地址保持不变。

## 构建校验与发布

### 环境与首次构建

以下命令在仓库根目录执行。需要 Node.js 24、npm、Python 3、Chromium、Poppler、Fontconfig 和 Noto CJK 字体；构建脚本不运行教程示例或训练任务。

```bash
sudo apt-get update
sudo apt-get install -y poppler-utils fontconfig fonts-noto-cjk
npm ci --no-audit --no-fund
npx playwright install --with-deps chromium
npm run sync
npm test
npm run build
npm run check
```

已有 Chromium 时，可通过 `PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH` 指定其绝对路径。`npm run build -- sac` 可只重建 SAC 双语 PDF；主题 ID 见 `materials.json`。如果修改了渲染器、样式、依赖锁文件或清单，需重建全部主题。

### 日常修改与检查

1. 同步修改中英文正文；版本变化在 `materials.json` 中记录范围、核验日期与固定 commit。
2. 运行 `npm run sync`、`npm test`、`npm run build` 和 `npm run check`。检查要求单一标题、正确标题层级、有效目录锚点、语言配对、资料链接和当前 PDF 绑定。
3. 抽看生成 PDF 的目录、长公式、表格、流程图和分页。渲染器自动拒绝公式错误、显示公式溢出和未渲染的 Mermaid 图；自动检查不等于全部页面已人工审校。
4. 提交 Markdown、当前 PDF、`pdf/build-manifest.json` 及必要构建文件。依赖、临时 HTML/图片、备份和日志不入库。

依赖锁定在 `package-lock.json`，打印样式在 `scripts/print.css`。`pdf/build-manifest.json` 记录源文件/PDF 哈希、构建输入指纹、浏览器版本、字体哈希及页数。它用于发现过期 PDF；浏览器、字体和 PDF 内部时间戳可能影响二进制结果，因此不承诺跨环境逐字节重现。

### 网站同步

需要 GitHub CLI 已登录，且对课程仓库与个人网站仓库拥有相应写权限。以下发布命令仅更新清单列出的个人网站资料文件；课程仓库改动仍需正常 commit 和 push。

```bash
python3 scripts/publish.py --publish --wait
python3 scripts/publish.py --verify-only
```

发布使用非强制的原子 commit，保留个人网站其他文件；等待现有 Pages 工作流完成后，下载并校验所有 44 个资料文件。含 YAML 元数据的 Markdown 网站副本会增加开头空行，以便 Jekyll 按静态文件提供下载；正文保持一致。`npm run check` 自动校验已提交 PDF 的绑定；只有显式执行发布命令才会更新网站。

## 课程建设计划

以下内容计划逐步补充，当前尚未交付：

- 正式教学大纲、课时分配与每讲学习目标。
- 配套课件、阅读清单与课堂演示。
- 分层作业、实验环境与可复现实验代码。
- 综合课程项目、评价标准与学生贡献指南。

## 参与贡献与反馈

欢迎大家下载、修改和提交 PR！反馈可以发送到我的邮箱：[zengx3@sustech.edu.cn](mailto:zengx3@sustech.edu.cn)。

## 使用说明与限制

- 当前主要交付阅读资料；教程中的示例代码不代表已在统一环境中完成验证。
- 现有中英文资料按主题配对，PDF 与 Markdown 的逐段一致性尚未审校。
- 软件版本、设备接口及硬件参数以各教程说明和实际环境为准；硬件实验需独立完成设备核对与安全验证。
- 后续修改教程时同步维护中英文内容，并更新对应 PDF 和本索引。
- 许可范围见上方“使用许可”；第三方材料不因本仓库公开或标注来源而自动获得再许可。
