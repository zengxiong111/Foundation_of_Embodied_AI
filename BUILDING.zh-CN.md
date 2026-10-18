# 构建与维护

面向资料维护者；学习者可直接访问[课程首页](README.zh-CN.md)下载教材。命令均在仓库根目录执行。

[English](BUILDING.md)

## 目录

- [环境与首次构建](#环境与首次构建)
- [日常修改与检查](#日常修改与检查)
- [网站同步](#网站同步)
- [许可与贡献范围](#许可与贡献范围)

## 环境与首次构建

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

## 日常修改与检查

1. 同步修改中英文正文；版本变化在 `materials.json` 中记录范围、核验日期与固定 commit。
2. 运行 `npm run sync`、`npm test`、`npm run build` 和 `npm run check`。检查要求单一标题、正确标题层级、有效目录锚点、语言配对、资料链接和当前 PDF 绑定。
3. 抽看生成 PDF 的目录、长公式、表格、流程图和分页。渲染器自动拒绝公式错误、显示公式溢出和未渲染的 Mermaid 图；自动检查不等于全部页面已人工审校。
4. 提交 Markdown、当前 PDF、`pdf/build-manifest.json` 及必要构建文件。依赖、临时 HTML/图片、备份和日志不入库。

PDF 默认采用黑白为主的打印样式：黑色正文、标题、公式和带下划线的链接，浅灰表头，以及灰度流程图。正文保持可搜索，链接仍可点击。

依赖锁定在 `package-lock.json`，打印样式在 `scripts/print.css`。`pdf/build-manifest.json` 记录源文件/PDF 哈希、构建输入指纹、浏览器版本、字体哈希及页数。它用于发现过期 PDF；浏览器、字体和 PDF 内部时间戳可能影响二进制结果，因此不承诺跨环境逐字节重现。

## 课程合订本

执行 `npm run build`、`npm run check` 成功后，安装构建依赖 `python3 -m pip install pymupdf==1.26.7`，再运行 `python3 scripts/merge_collections.py`。它将生成 `books/Foundation_of_Embodied_AI_Complete_Chinese.pdf` 和 `books/Foundation_of_Embodied_AI_Complete_English.pdf`，包含可点击总目录与章节书签。合订本直接使用验证通过的 28 份独立章节 PDF。

## 网站同步

需要 GitHub CLI 已登录，且对课程仓库与个人网站仓库拥有相应写权限。以下发布命令仅更新清单列出的个人网站资料文件；课程仓库改动仍需正常 commit 和 push。

```bash
python3 scripts/publish.py --publish --wait
python3 scripts/publish.py --verify-only
```

发布使用非强制的原子 commit，保留个人网站其他文件；等待现有 Pages 工作流完成后，下载并校验所有 56 个章节资料文件。含 YAML 元数据的 Markdown 网站副本会增加开头空行，以便 Jekyll 按静态文件提供下载；正文保持一致。`npm run check` 自动校验已提交 PDF 的绑定；只有显式执行发布命令才会更新网站。

## 许可与贡献范围

| 内容 | 许可与适用范围 |
|---|---|
| 本仓库有权许可的原创教材正文、公式说明和自绘图示；对应 Markdown 与 PDF | [CC BY 4.0](LICENSE)；允许复制、修改、再分发和商业使用，需署名、链接许可并注明修改 |
| 新增构建、校验、同步和发布工具（`scripts/`、`tests/` 与构建配置） | [MIT](LICENSE-CODE)；保留版权和许可声明 |
| 第三方论文、引用段落、图片、代码、模型、数据和商标 | 仍遵循原有许可；不因本仓库引用而自动纳入 CC BY 或 MIT |

建议署名：**Xiong Zeng and contributors, Foundation of Embodied AI**，附[仓库地址](https://github.com/zengxiong111/Foundation_of_Embodied_AI)、[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 和修改说明。引用原论文或官方代码时同时保留其来源。法律正文保留标准英文原文，本节提供中文范围说明。欢迎提交你有权按相应许可贡献的内容；提交 PR 时说明来源和第三方限制。
