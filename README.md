# Foundation of Embodied AI  

[简体中文](README.zh-CN.md) | **English**

A public teaching-material repository for a future course titled “Foundation of Embodied AI” (具身智能基础), collecting Chinese and English tutorials on robot learning, engineering tools, and core algorithms. The course is currently in preparation; a formal teaching schedule is not yet available.

**Content provenance:** These materials are based on our project experience and were generated with the assistance of ChatGPT.

## Contents

- [Course scope](#course-scope)
- [Material index](#material-index)
- [Suggested learning path](#suggested-learning-path)
- [Quick start](#quick-start)
- [Repository structure](#repository-structure)
- [Licensing](#licensing)
- [Versions and verification scope](#versions-and-verification-scope)
- [Build validation and publication](#build-validation-and-publication)
- [Course development plan](#course-development-plan)
- [Contributions and feedback](#contributions-and-feedback)
- [Usage notes and limitations](#usage-notes-and-limitations)

## Course scope

For students and researchers interested in embodied AI, robot control, and robot learning. Familiarity with Python, linear algebra, probability, and basic calculus is recommended; control and machine-learning experience helps with advanced topics.

Learning objectives:

- Organize robot-learning engineering workflows with Git, ROS 2, and camera calibration.
- Understand how robot dynamics, actuation, and contact connect policy actions to physical state changes.
- Understand the roles of Isaac Sim, Isaac Lab, and MuJoCo, and workflows for simulation, RL training, and cross-engine validation.
- Understand embodied RL task categories, motion control, and Sim-to-Real workflows.
- Understand the core derivations, design motivations, and use cases of PPO and SAC.
- Understand the basic construction and derivation of Flow Matching as preparation for generative policies.
- Understand the roles of VLA and whole-body control in humanoids, and how GR00T and SONIC connect across those roles.

## Material index

The collection contains **11 topics, 22 Markdown files, and 22 PDF files**, with Chinese and English materials for every topic. Markdown supports online reading and editing; PDF supports downloading, printing, and classroom reading.

| Topic | Chinese Markdown | English Markdown | Chinese PDF | English PDF |
|---|---|---|---|---|
| Git Basics | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Git_Tutorial_CN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Git_Tutorial_EN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Git_Tutorial_CN.pdf" target="_blank">PDF</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Git_Tutorial_EN.pdf" target="_blank">PDF</a> |
| ROS 2 Basics | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/ROS2_Robot_Learning_Tutorial_ZH.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/ROS2_Robot_Learning_Tutorial_EN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/ROS2_Robot_Learning_Tutorial_ZH.pdf" target="_blank">PDF</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/ROS2_Robot_Learning_Tutorial_EN.pdf" target="_blank">PDF</a> |
| Camera Calibration | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Robot_Camera_Calibration_Tutorial_ZH.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Robot_Camera_Calibration_Tutorial_EN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Robot_Camera_Calibration_Tutorial_ZH.pdf" target="_blank">PDF</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Robot_Camera_Calibration_Tutorial_EN.pdf" target="_blank">PDF</a> |
| Robot Dynamics | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Robot_Dynamics_Embodied_CN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Robot_Dynamics_Embodied_EN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Robot_Dynamics_Embodied_CN.pdf" target="_blank">PDF</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Robot_Dynamics_Embodied_EN.pdf" target="_blank">PDF</a> |
| Isaac / MuJoCo Simulation | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Isaac_MuJoCo_RL_%E5%AE%8C%E6%95%B4%E6%95%99%E7%A8%8B_%E4%B8%AD%E6%96%87.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Isaac_MuJoCo_RL_Complete_Guide_EN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Isaac_MuJoCo_RL_%E5%AE%8C%E6%95%B4%E6%95%99%E7%A8%8B_%E4%B8%AD%E6%96%87.pdf" target="_blank">PDF</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Isaac_MuJoCo_RL_Complete_Guide_EN.pdf" target="_blank">PDF</a> |
| Embodied RL | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Embodied_RL_Tutorial_Full_ZH.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Embodied_RL_Tutorial_Full_EN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Embodied_RL_Tutorial_Full_ZH.pdf" target="_blank">PDF</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Embodied_RL_Tutorial_Full_EN.pdf" target="_blank">PDF</a> |
| Motion Control RL | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Embodied_Motion_Control_RL_Tutorial_ZH.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Embodied_Motion_Control_RL_Tutorial_EN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Embodied_Motion_Control_RL_Tutorial_ZH.pdf" target="_blank">PDF</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Embodied_Motion_Control_RL_Tutorial_EN.pdf" target="_blank">PDF</a> |
| PPO | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/PPO_Complete_Derivation_CN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/PPO_Complete_Derivation_EN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/PPO_Complete_Derivation_CN.pdf" target="_blank">PDF</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/PPO_Complete_Derivation_EN.pdf" target="_blank">PDF</a> |
| SAC | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/SAC_Complete_Derivation_CN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/SAC_Complete_Derivation_EN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/SAC_Complete_Derivation_CN.pdf" target="_blank">PDF</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/SAC_Complete_Derivation_EN.pdf" target="_blank">PDF</a> |
| Flow Matching | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Flow_Matching_CN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Flow_Matching_Step_by_Step_Derivation_EN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Flow_Matching_CN.pdf" target="_blank">PDF</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/Flow_Matching_Step_by_Step_Derivation_EN.pdf" target="_blank">PDF</a> |
| GR00T & SONIC | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/System1_System0_GR00T_SONIC_CN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/System1_System0_GR00T_SONIC_EN.md" target="_blank">MD</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/System1_System0_GR00T_SONIC_CN.pdf" target="_blank">PDF</a> | <a href="https://zengxiong111.github.io/zengxiong.github.io/files/System1_System0_GR00T_SONIC_EN.pdf" target="_blank">PDF</a> |

## Suggested learning path

This is a preparation proposal that can be adapted to students' backgrounds, rather than a formal timetable.

1. **Engineering preparation**: Git and collaboration → ROS 2 for robot learning.
2. **Perception foundations**: camera intrinsics, extrinsics, hand-eye calibration, and coordinate frames.
3. **Dynamics and simulation**: robot dynamics → Isaac Sim, Isaac Lab, and MuJoCo, connecting actions, actuators, contact, and simulation-based training.
4. **Task landscape**: embodied RL overview → motion-control RL.
5. **Algorithm derivations**: PPO → SAC, comparing on-policy and off-policy learning.
6. **Generative methods**: Flow Matching derivation, leading into future policy-learning topics.
7. **System architecture**: System 1 / System 0, using GR00T and SONIC to understand high-level action generation and low-level whole-body control.
8. **Integrated practice (planned)**: course projects combining perception, communication, and control.

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
├── md/             # 22 Chinese and English Markdown tutorials
├── pdf/            # 22 PDFs and current build manifest
├── materials.json  # Material, version and website mapping
├── VERSIONS.md     # English version notes (Chinese counterpart provided)
├── scripts/        # Synchronization, build, validation and publication
├── tests/          # Validator tests
├── LICENSE         # Original teaching content: CC BY 4.0
└── LICENSE-CODE    # Build tools: MIT
```

The repository identifier is `Foundation_of_Embodied_AI`. The Chinese course title is “具身智能基础”, and the English display title is “Foundation of Embodied AI”. Existing filenames are preserved, including the `-1` suffix on some PDFs. All original files are included in full.

## Licensing

| Content | License and scope |
|---|---|
| Original teaching prose, equation explanations and course-created diagrams that this repository has the right to license; corresponding Markdown and PDFs | [CC BY 4.0](LICENSE); copying, adaptation, redistribution and commercial use are allowed with attribution, a license link and an indication of changes |
| New build, validation, synchronization and publication tools (`scripts/`, `tests/` and build configuration) | [MIT](LICENSE-CODE); retain copyright and license notices |
| Third-party papers, quoted passages, figures, code, models, data and trademarks | Retain their original terms; citation here does not automatically place them under CC BY or MIT |

Suggested attribution: **Xiong Zeng and contributors, Foundation of Embodied AI**, with the [repository URL](https://github.com/zengxiong111/Foundation_of_Embodied_AI), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) and a description of changes. Preserve original paper or official-code attribution where applicable. Legal texts retain their standard English wording; this section explains scope. Contributions are welcome when you have the right to contribute under the applicable license; identify sources and third-party restrictions in your PR.

## Versions and verification scope

The [bilingual version matrix](VERSIONS.md) records applicability, the 2026-10-18 documentation review date and pinned upstream commits for each topic. Review covers documentation structure and reference snapshots; **it does not mean installation, training or real-robot execution passed**. Experiment reproduction still requires a locked environment, assets, configuration and checkpoint.

`materials.json` is the single inventory for topics, language pairs, versions and website filenames. `npm run sync` uses it to update tutorial navigation, clickable contents and version notes. All 22 tutorials provide course-home, counterpart-language and PDF links. Existing direct personal-website URLs in the material table are preserved.

## Build validation and publication

### Environment and first build

Run these commands from the repository root. Requirements are Node.js 24, npm, Python 3, Chromium, Poppler, Fontconfig and Noto CJK fonts. The build does not execute tutorial examples or training tasks.

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

To use an existing Chromium, set `PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH` to its absolute path. `npm run build -- sac` rebuilds only the SAC language pair; topic IDs are listed in `materials.json`. Rebuild every topic after changing the renderer, stylesheet, dependency lockfile or inventory.

### Routine edits and checks

1. Update both language bodies together. Record version scope, review date and pinned commits in `materials.json` when they change.
2. Run `npm run sync`, `npm test`, `npm run build` and `npm run check`. Checks require one title, a valid heading hierarchy, valid contents anchors, language pairing, material URLs and current PDF bindings.
3. Visually inspect PDF contents, long equations, tables, diagrams and pagination. The renderer rejects math errors, display-equation overflows and unrendered Mermaid diagrams; automated checks do not imply manual review of every page.
4. Commit Markdown, current PDFs, `pdf/build-manifest.json` and necessary build files. Keep installed dependencies, temporary HTML/images, backups and logs out of the repository.

Dependencies are locked in `package-lock.json`; print styling lives in `scripts/print.css`. `pdf/build-manifest.json` records source/PDF hashes, the build-input fingerprint, browser version, font hashes and page count. It detects stale PDFs. Browser/font differences and internal PDF timestamps can affect binary output, so byte-for-byte reproduction across environments is not promised.

### Website synchronization

GitHub CLI must be authenticated with appropriate write access to the course and personal-website repositories. The publication command only updates manifest-listed personal-site materials; course repository changes still require the normal commit and push.

```bash
python3 scripts/publish.py --publish --wait
python3 scripts/publish.py --verify-only
```

Publication uses a non-force atomic commit and preserves other website files. After the existing Pages workflow completes, it downloads and checks all 44 materials. A leading blank line is added to website Markdown copies containing YAML metadata so Jekyll serves them statically; body content is preserved. `npm run check` automatically checks committed PDF bindings; website updates require the explicit publication command.

## Course development plan

The following items are planned and are not yet delivered:

- Formal syllabus, teaching hours, and learning objectives for each lecture.
- Slides, reading lists, and classroom demonstrations.
- Graded assignments, lab environments, and reproducible experiment code.
- Integrated course projects, assessment criteria, and student contribution guidelines.

## Contributions and feedback

Everyone is welcome to download, modify, and submit pull requests! You can send feedback to my email: [zengx3@sustech.edu.cn](mailto:zengx3@sustech.edu.cn).

## Usage notes and limitations

- The current delivery primarily consists of reading materials; tutorial examples have not been validated together in a unified environment.
- Chinese and English materials are paired by topic. Paragraph-level consistency between PDF and Markdown has not been reviewed.
- Software versions, device interfaces, and hardware parameters depend on each tutorial and the actual environment. Hardware experiments require separate equipment checks and safety validation.
- Maintain Chinese and English content together when revising tutorials, and update the corresponding PDFs and this index.
- See Licensing above for scope; public access or source attribution does not automatically relicense third-party materials.
