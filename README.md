# Foundation of Embodied AI · 具身智能基础

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

## Material index

The collection contains **8 topics, 16 Markdown files, and 16 PDF files**, with Chinese and English materials for every topic. Markdown supports online reading and editing; PDF supports downloading, printing, and classroom reading.

| Topic | Chinese Markdown | English Markdown | Chinese PDF | English PDF |
|---|---|---|---|---|
| Git and collaboration | [MD](md/Git_%E4%BD%BF%E7%94%A8%E6%95%99%E7%A8%8B_%E4%B8%AD%E6%96%87%E7%89%88.md) | [MD](md/Git_Tutorial_English.md) | [PDF](pdf/Git_%E4%BD%BF%E7%94%A8%E6%95%99%E7%A8%8B_%E4%B8%AD%E6%96%87%E7%89%88.pdf) | [PDF](pdf/Git_Tutorial_English-1.pdf) |
| ROS 2 for robot learning | [MD](md/ROS2_Robot_Learning_Tutorial_ZH.md) | [MD](md/ROS2_Robot_Learning_Tutorial_EN.md) | [PDF](pdf/ROS2_Robot_Learning_Tutorial_ZH.pdf) | [PDF](pdf/ROS2_Robot_Learning_Tutorial_EN-1.pdf) |
| Robot camera calibration | [MD](md/Robot_Camera_Calibration_Tutorial_ZH.md) | [MD](md/Robot_Camera_Calibration_Tutorial_EN.md) | [PDF](pdf/Robot_Camera_Calibration_Tutorial_ZH-1.pdf) | [PDF](pdf/Robot_Camera_Calibration_Tutorial_EN-1.pdf) |
| Embodied reinforcement learning overview | [MD](md/Embodied_RL_Tutorial_Full_ZH.md) | [MD](md/Embodied_RL_Tutorial_Full_EN.md) | [PDF](pdf/Embodied_RL_Tutorial_Full_ZH-1.pdf) | [PDF](pdf/Embodied_RL_Tutorial_Full_EN-1.pdf) |
| Reinforcement learning for motion control | [MD](md/Embodied_Motion_Control_RL_Tutorial_ZH.md) | [MD](md/Embodied_Motion_Control_RL_Tutorial_EN.md) | [PDF](pdf/Embodied_Motion_Control_RL_Tutorial_ZH.pdf) | [PDF](pdf/Embodied_Motion_Control_RL_Tutorial_EN.pdf) |
| PPO derivation and design motivations | [MD](md/PPO_%E5%AE%8C%E6%95%B4%E6%8E%A8%E5%AF%BC%E4%B8%8E%E8%AE%BE%E8%AE%A1%E5%8A%A8%E6%9C%BA_%E4%B8%AD%E6%96%87.md) | [MD](md/PPO_Complete_Derivation_and_Design_Motivations_English.md) | [PDF](pdf/PPO_%E5%AE%8C%E6%95%B4%E6%8E%A8%E5%AF%BC%E4%B8%8E%E8%AE%BE%E8%AE%A1%E5%8A%A8%E6%9C%BA_%E4%B8%AD%E6%96%87.pdf) | [PDF](pdf/PPO_Complete_Derivation_and_Design_Motivations_English-1.pdf) |
| SAC derivation and design motivations | [MD](md/SAC_Complete_Derivation_CN.md) | [MD](md/SAC_Complete_Derivation_EN.md) | [PDF](pdf/SAC_Complete_Derivation_CN-1.pdf) | [PDF](pdf/SAC_Complete_Derivation_EN-1.pdf) |
| Flow Matching step-by-step derivation | [MD](md/Flow_Matching_%E8%AF%A6%E7%BB%86%E6%8E%A8%E5%AF%BC_%E7%B2%BE%E7%AE%80%E7%89%88.md) | [MD](md/Flow_Matching_Step_by_Step_Derivation_EN.md) | [PDF](pdf/Flow_Matching_%E8%AF%A6%E7%BB%86%E6%8E%A8%E5%AF%BC_%E7%B2%BE%E7%AE%80%E7%89%88.pdf) | [PDF](pdf/Flow_Matching_Step_by_Step_Derivation_EN.pdf) |

## Suggested learning path

This is a preparation proposal that can be adapted to students' backgrounds, rather than a formal timetable.

1. **Engineering preparation**: Git and collaboration → ROS 2 for robot learning.
2. **Perception foundations**: camera intrinsics, extrinsics, hand-eye calibration, and coordinate frames.
3. **Task landscape**: embodied RL overview → motion-control RL.
4. **Algorithm derivations**: PPO → SAC, comparing on-policy and off-policy learning.
5. **Generative methods**: Flow Matching derivation, leading into future policy-learning topics.
6. **Integrated practice (planned)**: course projects combining perception, communication, and control.

## Quick start

1. Choose a topic, language, and format from the index above. Read Markdown on GitHub or download PDF without installing software.
2. To read locally or contribute to the materials, run in a terminal:

```bash
git clone https://github.com/zengxiong111/Foundation_of_Emboadied_AI.git
cd Foundation_of_Emboadied_AI
```

3. Edit tutorials in `md/` and view their reading versions in `pdf/`. Prepare dependencies according to each tutorial before running examples; this repository has no unified installation or training entry point.

## Repository structure

```text
Foundation_of_Emboadied_AI/
├── README.md       # English course overview (default)
├── README.zh-CN.md # Chinese course overview
├── md/             # 16 Chinese and English Markdown tutorials
└── pdf/            # 16 Chinese and English PDF materials
```

The repository identifier remains `Foundation_of_Emboadied_AI`. The Chinese course title is “具身智能基础”, and the English display title is “Foundation of Embodied AI”. Existing filenames are preserved, including the `-1` suffix on some PDFs. All original files are included in full.

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
