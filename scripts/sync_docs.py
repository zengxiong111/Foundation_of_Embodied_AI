#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Synchronize generated navigation, metadata and version tables; preserve tutorial bodies."""
import argparse
import json
from pathlib import Path
import re
from urllib.parse import quote
from check import ROOT, prose_lines


def metadata(config, topic, lang, doc):
    zh = lang == 'zh'
    repo = 'https://github.com/' + config['repository']
    peer = topic['documents']['en' if zh else 'zh']['md']
    nav = (f'[课程首页]({repo}/blob/main/README.zh-CN.md) · [English]({repo}/blob/main/{quote(peer)}) · [PDF]({config["site_base"] + quote(doc["site_pdf"])})' if zh else
           f'[Course home]({repo}) · [简体中文]({repo}/blob/main/{quote(peer)}) · [PDF]({config["site_base"] + quote(doc["site_pdf"])})')
    table = (f'| {"项目" if zh else "Item"} | {"说明" if zh else "Details"} |\n|---|---|\n'
             f'| {"适用范围" if zh else "Applicability"} | {topic["versions_zh" if zh else "versions_en"]} |\n'
             f'| {"文档核验日期" if zh else "Documentation review date"} | {topic["reviewed_at"]} |\n'
             f'| {"许可" if zh else "License"} | CC BY 4.0 · Xiong Zeng and contributors |')
    notes = topic['review_scope_zh' if zh else 'review_scope_en']
    scope = (f'**核验边界：** {notes} 版本与固定源码引用见[版本说明]({repo}/blob/main/VERSIONS.zh-CN.md)。原创教材采用 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)，第三方引用、图片及代码遵循其原有许可；注明来源不等于拥有再许可权。' if zh else
             f'**Verification scope:** {notes} See [version notes]({repo}/blob/main/VERSIONS.md) for pinned source references. Original teaching content uses [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); third-party quotations, figures and code retain their original licenses. Citation alone does not grant relicensing rights.')
    return '<!-- course-meta:start -->\n' + nav + '\n\n' + table + '\n\n' + scope + '\n<!-- course-meta:end -->'


def synchronize_tutorial(text, meta, lang):
    text, n = re.subn(r'<!-- course-meta:start -->[\s\S]*?<!-- course-meta:end -->', lambda _: meta, text)
    if n != 1:
        raise ValueError('Expected one generated metadata block')
    text = re.sub(r'<!-- course-toc:start -->[\s\S]*?<!-- course-toc:end -->\n*', '', text)
    text = re.sub(r'<a id="course-section-\d+"></a>\n*', '', text)
    # prose_lines filters code blocks; count identical headings in source order.
    headings = [line for line in prose_lines(text) if re.match(r'^#{2,3} ', line)]
    lines = text.splitlines(); toc = []; cursor = 0
    for number, heading in enumerate(headings, 1):
        i = lines.index(heading, cursor)
        anchor = f'course-section-{number:02d}'
        lines.insert(i, f'<a id="{anchor}"></a>\n'); cursor = i + 2
        match = re.match(r'^(#{2,3}) (.+)$', heading)
        label = re.sub(r'\[([^]]+)\]\([^)]*\)', r'\1', match[2])
        toc.append(('  ' if len(match[1]) == 3 else '') + f'- [{label}](#{anchor})')
    text = '\n'.join(lines) + '\n'
    block = '<!-- course-toc:start -->\n## ' + ('目录' if lang == 'zh' else 'Contents') + '\n\n' + '\n'.join(toc) + '\n<!-- course-toc:end -->'
    return text.replace('<!-- course-meta:end -->', '<!-- course-meta:end -->\n\n' + block)


def version_document(config, zh):
    title = '版本、适用范围与核验边界' if zh else 'Versions, applicability and verification scope'
    other = '[English](VERSIONS.md)' if zh else '[简体中文](VERSIONS.zh-CN.md)'
    if zh:
        intro = '记录当前教材的适用范围和上游源码快照。核验日期表示文档整理与引用核对，不代表教程、训练或硬件运行通过。'
        sections = [('适用范围', 'applicability'), ('固定源码快照', 'snapshots'), ('使用与更新规则', 'updates')]
        table = '| 主题 | 适用版本或范围 | 文档核验日期 |\n|---|---|---|\n'
    else:
        intro = 'Record the current materials’ applicability and upstream source snapshots. Review dates describe documentation and reference checks, not successful tutorial, training or hardware runs.'
        sections = [('Applicability', 'applicability'), ('Pinned source snapshots', 'snapshots'), ('Usage and update rules', 'updates')]
        table = '| Topic | Applicable versions or scope | Documentation review date |\n|---|---|---|\n'
    s = '# ' + title + '\n\n' + intro + '\n\n' + other + '\n\n## ' + ('目录' if zh else 'Contents') + '\n\n'
    s += '\n'.join(f'- [{t}](#{a})' for t, a in sections) + '\n\n<a id="applicability"></a>\n\n## ' + sections[0][0] + '\n\n' + table
    for t in config['topics']:
        s += '| ' + t['title_zh' if zh else 'title_en'] + ' | ' + t['versions_zh' if zh else 'versions_en'] + ' | ' + t['reviewed_at'] + ' |\n'
    s += '\n<a id="snapshots"></a>\n\n## ' + sections[1][0] + '\n\n'
    s += ('以下 Git ref 在 2026-10-18 通过 GitHub API 解析到固定 commit；这是可定位的引用快照，不保证这些项目彼此兼容。分支状态和文档页面可能随后变化。\n\n' if zh else
          'The following Git refs were resolved to fixed commits through the GitHub API on 2026-10-18. These are locatable reference snapshots, not a guarantee of compatibility between projects. Branch status and documentation pages may subsequently change.\n\n')
    s += ('| 项目 | 原 ref | 固定 commit | 上游 commit 时间（UTC） |\n|---|---|---|---|\n' if zh else '| Project | Original ref | Pinned commit | Upstream commit time (UTC) |\n|---|---|---|---|\n')
    for r in config['upstream_snapshots']:
        s += f'| {r["repo"]} | `{r["ref"]}` | [`{r["sha"]}`]({r["url"]}) | {r["date"]} |\n'
    s += '\n<a id="updates"></a>\n\n## ' + sections[2][0] + '\n\n'
    s += ('''1. 复现实验时，另行记录操作系统、Python、PyTorch、CUDA/驱动、仿真器、训练器、资产和 checkpoint；本表不替代已运行环境的锁定配置。
2. Isaac Lab 2.x 与 3.0 EA 不作为可互换环境；核对物理后端、四元数顺序、资产 API、执行器和 reset/step 语义。EA 状态是文档快照，后续以固定源码和官方发布信息核对。
3. GR00T N1.5 论文集成与 N1.7 工作流分开解释；SONIC checkpoint、观测配置、初始 token 和 VLA 训练数据必须匹配。
4. 所有教程均未在本次文档维护中完成训练或真机验证。不要将引用存在、PDF 构建通过或跨引擎示例当作性能或硬件准入证据。
5. 更新版本时编辑 `materials.json`，记录核验日期和 commit，运行 `npm run sync`、`npm run build` 和 `npm run check`，同步中英文及网站副本。不要添加历史素材或日志归档。

官方资料：[Isaac Lab 发布](https://github.com/isaac-sim/IsaacLab/releases)、[ROS 2 Jazzy](https://docs.ros.org/en/jazzy/)、[GR00T / SONIC 工作流](https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/vla_workflow.html)。
''' if zh else '''1. For experiment reproduction, separately record OS, Python, PyTorch, CUDA/driver, simulator, trainer, assets and checkpoint. This table does not replace a locked configuration of an actually executed environment.
2. Isaac Lab 2.x and 3.0 EA are not interchangeable environments. Check physics backend, quaternion order, asset APIs, actuators and reset/step semantics. EA status is a documentation snapshot; subsequently check fixed source and official releases.
3. Explain the GR00T N1.5 paper integration separately from the N1.7 workflow. SONIC checkpoint, observation configuration, initial token and VLA training data must match.
4. No tutorial has completed training or real-robot validation during this documentation maintenance. Reference availability, a successful PDF build or a cross-engine example is not evidence of performance or hardware admission.
5. When updating versions, edit `materials.json`, record the review date and commit, then run `npm run sync`, `npm run build` and `npm run check`. Synchronize both languages and website copies. Do not add historical material or log archives.

Official resources: [Isaac Lab releases](https://github.com/isaac-sim/IsaacLab/releases), [ROS 2 Jazzy](https://docs.ros.org/en/jazzy/), [GR00T / SONIC workflow](https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/vla_workflow.html).
''')
    return s


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--check', action='store_true'); args = parser.parse_args()
    config = json.loads((ROOT / 'materials.json').read_text()); changed = []
    outputs = {}
    for t in config['topics']:
        for lang, doc in t['documents'].items():
            p = ROOT / doc['md']; outputs[p] = synchronize_tutorial(p.read_text(), metadata(config, t, lang, doc), lang)
    outputs[ROOT / 'VERSIONS.md'] = version_document(config, False)
    outputs[ROOT / 'VERSIONS.zh-CN.md'] = version_document(config, True)
    for p, content in outputs.items():
        if not p.exists() or p.read_text() != content:
            changed.append(str(p.relative_to(ROOT)))
            if not args.check: p.write_text(content)
    if args.check and changed:
        raise SystemExit('Generated documentation is stale; run npm run sync:\n' + '\n'.join(changed))
    print(('PASS: generated documentation is current.' if args.check else f'Synchronized {len(changed)} documentation files.'))


if __name__ == '__main__':
    main()
