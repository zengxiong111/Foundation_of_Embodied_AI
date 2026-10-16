# Foundation of Embodied AI  

[简体中文](README.zh-CN.md) | **English**

A public teaching-material repository for a future course titled “Foundation of Embodied AI” (具身智能基础), collecting Chinese and English tutorials on robot learning, engineering tools, and core algorithms. The course is currently in preparation; a formal teaching schedule is not yet available.


## Contents

- [Course scope](#course-scope)
- [Material index](#material-index)
- [Suggested learning path](#suggested-learning-path)
- [Quick start](#quick-start)
- [Repository structure](#repository-structure)
- [Course development plan](#course-development-plan)
- [Usage notes and limitations](#usage-notes-and-limitations)

## Course scope

For students and researchers interested in embodied AI, robot control, and robot learning. Familiarity with Python, linear algebra, probability, and basic calculus is recommended; control and machine-learning experience helps with advanced topics.

Learning objectives:

- Organize robot-learning engineering workflows with Git, ROS 2, and camera calibration.
- Understand embodied RL task categories, motion control, and Sim-to-Real workflows.
- Understand the core derivations, design motivations, and use cases of PPO and SAC.
- Understand the basic construction and derivation of Flow Matching as preparation for generative policies.
- Understand the roles of VLA and whole-body control in humanoids, and how GR00T and SONIC connect across those roles.

## Material index

The collection contains **9 topics, 18 Markdown files, and 18 PDF files**, with Chinese and English materials for every topic. Markdown supports online reading and editing; PDF supports downloading, printing, and classroom reading.

| Topic | Chinese Markdown | English Markdown | Chinese PDF | English PDF |
|---|---|---|---|---|
| Git and collaboration | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Git_Tutorial_CN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Git_Tutorial_EN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Git_Tutorial_CN.pdf" target="_blank">PDF</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Git_Tutorial_EN.pdf" target="_blank">PDF</a> |
| ROS 2 for robot learning | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/ROS2_Robot_Learning_Tutorial_ZH.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/ROS2_Robot_Learning_Tutorial_EN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/ROS2_Robot_Learning_Tutorial_ZH.pdf" target="_blank">PDF</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/ROS2_Robot_Learning_Tutorial_EN.pdf" target="_blank">PDF</a> |
| Robot camera calibration | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Robot_Camera_Calibration_Tutorial_ZH.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Robot_Camera_Calibration_Tutorial_EN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Robot_Camera_Calibration_Tutorial_ZH.pdf" target="_blank">PDF</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Robot_Camera_Calibration_Tutorial_EN.pdf" target="_blank">PDF</a> |
| Embodied reinforcement learning overview | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Embodied_RL_Tutorial_Full_ZH.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Embodied_RL_Tutorial_Full_EN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Embodied_RL_Tutorial_Full_ZH.pdf" target="_blank">PDF</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Embodied_RL_Tutorial_Full_EN.pdf" target="_blank">PDF</a> |
| Reinforcement learning for motion control | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Embodied_Motion_Control_RL_Tutorial_ZH.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Embodied_Motion_Control_RL_Tutorial_EN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Embodied_Motion_Control_RL_Tutorial_ZH.pdf" target="_blank">PDF</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Embodied_Motion_Control_RL_Tutorial_EN.pdf" target="_blank">PDF</a> |
| PPO derivation and design motivations | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/PPO_Complete_Derivation_CN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/PPO_Complete_Derivation_EN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/PPO_Complete_Derivation_CN.pdf" target="_blank">PDF</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/PPO_Complete_Derivation_EN.pdf" target="_blank">PDF</a> |
| SAC derivation and design motivations | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/SAC_Complete_Derivation_CN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/SAC_Complete_Derivation_EN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/SAC_Complete_Derivation_CN.pdf" target="_blank">PDF</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/SAC_Complete_Derivation_EN.pdf" target="_blank">PDF</a> |
| Flow Matching step-by-step derivation | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Flow_Matching_CN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Flow_Matching_Step_by_Step_Derivation_EN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Flow_Matching_CN.pdf" target="_blank">PDF</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Flow_Matching_Step_by_Step_Derivation_EN.pdf" target="_blank">PDF</a> |
| Humanoid System 1 / System 0: GR00T and SONIC | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/System1_System0_GR00T_SONIC_CN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/System1_System0_GR00T_SONIC_EN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/System1_System0_GR00T_SONIC_CN.pdf" target="_blank">PDF</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/System1_System0_GR00T_SONIC_EN.pdf" target="_blank">PDF</a> |

## Suggested learning path

This is a preparation proposal that can be adapted to students' backgrounds, rather than a formal timetable.

1. **Engineering preparation**: Git and collaboration → ROS 2 for robot learning.
2. **Perception foundations**: camera intrinsics, extrinsics, hand-eye calibration, and coordinate frames.
3. **Task landscape**: embodied RL overview → motion-control RL.
4. **Algorithm derivations**: PPO → SAC, comparing on-policy and off-policy learning.
5. **Generative methods**: Flow Matching derivation, leading into future policy-learning topics.
6. **System architecture**: System 1 / System 0, using GR00T and SONIC to understand high-level action generation and low-level whole-body control.
7. **Integrated practice (planned)**: course projects combining perception, communication, and control.

## Quick start

1. Choose a topic, language, and format from the index above. Links open the corresponding files under the personal website’s `files/` directory. PDFs can be read in the browser; Markdown files may display as text or download, depending on the browser.
2. To read locally or contribute to the materials, run in a terminal:

```bash
git clone https://github.com/zengxiong111/Foundation_of_Embodied_AI.git
cd Foundation_of_Embodied_AI
```

3. Edit tutorials in `md/` and view their reading versions in `pdf/`. Prepare dependencies according to each tutorial before running examples; this repository has no unified installation or training entry point.

## Repository structure

```text
Foundation_of_Embodied_AI/
├── README.md       # English course overview (default)
├── README.zh-CN.md # Chinese course overview
├── md/             # 18 Chinese and English Markdown tutorials
└── pdf/            # 18 Chinese and English PDF materials
```

The repository identifier is `Foundation_of_Embodied_AI`. The Chinese course title is “具身智能基础”, and the English display title is “Foundation of Embodied AI”. Existing filenames are preserved, including the `-1` suffix on some PDFs. All original files are included in full.

## Course development plan

The following items are planned and are not yet delivered:

- Formal syllabus, teaching hours, and learning objectives for each lecture.
- Slides, reading lists, and classroom demonstrations.
- Graded assignments, lab environments, and reproducible experiment code.
- Integrated course projects, assessment criteria, and student contribution guidelines.

## Usage notes and limitations

- The current delivery primarily consists of reading materials; tutorial examples have not been validated together in a unified environment.
- Chinese and English materials are paired by topic. Paragraph-level consistency between PDF and Markdown has not been reviewed.
- Software versions, device interfaces, and hardware parameters depend on each tutorial and the actual environment. Hardware experiments require separate equipment checks and safety validation.
- Maintain Chinese and English content together when revising tutorials, and update the corresponding PDFs and this index.
- No repository-wide open-source license is currently declared. Public access does not grant unrestricted redistribution or commercial-use rights. Credit sources and comply with licenses for third-party materials.
