# Build and maintenance

For material maintainers; learners can download tutorials from the [course home](README.md). Run commands from the repository root.

[简体中文](BUILDING.zh-CN.md)

## Contents

- [Environment and first build](#environment-and-first-build)
- [Routine edits and checks](#routine-edits-and-checks)
- [Website synchronization](#website-synchronization)
- [Licensing and contribution scope](#licensing-and-contribution-scope)

## Environment and first build

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

## Routine edits and checks

1. Update both language bodies together. Record version scope, review date and pinned commits in `materials.json` when they change.
2. Run `npm run sync`, `npm test`, `npm run build` and `npm run check`. Checks require one title, a valid heading hierarchy, valid contents anchors, language pairing, material URLs and current PDF bindings.
3. Visually inspect PDF contents, long equations, tables, diagrams and pagination. The renderer rejects math errors, display-equation overflows and unrendered Mermaid diagrams; automated checks do not imply manual review of every page.
4. Commit Markdown, current PDFs, `pdf/build-manifest.json` and necessary build files. Keep installed dependencies, temporary HTML/images, backups and logs out of the repository.

PDFs use a print-friendly monochrome style: black text, headings, equations and underlined links, light-gray table headers, and grayscale diagrams. Text remains searchable and links remain clickable.

Dependencies are locked in `package-lock.json`; print styling lives in `scripts/print.css`. `pdf/build-manifest.json` records source/PDF hashes, the build-input fingerprint, browser version, font hashes and page count. It detects stale PDFs. Browser/font differences and internal PDF timestamps can affect binary output, so byte-for-byte reproduction across environments is not promised.

## Whole-course PDFs

After successful `npm run build` and `npm run check`, install the independent book generator once with `python3 -m pip install pymupdf==1.26.7 fonttools` and run `python3 scripts/merge_collections.py`. It generates `books/Foundation_of_Embodied_AI_Complete_Chinese.pdf` and `books/Foundation_of_Embodied_AI_Complete_English.pdf` with clickable contents and chapter bookmarks. Book PDFs are derived from the 26 validated per-topic chapter PDFs, not independently edited versions.

## Website synchronization

GitHub CLI must be authenticated with appropriate write access to the course and personal-website repositories. The publication command only updates manifest-listed personal-site materials; course repository changes still require the normal commit and push.

```bash
python3 scripts/publish.py --publish --wait
python3 scripts/publish.py --verify-only
```

Publication uses a non-force atomic commit and preserves other website files. After the existing Pages workflow completes, it downloads and checks all 52 chapter materials. A leading blank line is added to website Markdown copies containing YAML metadata so Jekyll serves them statically; body content is preserved. `npm run check` automatically checks committed PDF bindings; website updates require the explicit publication command.

## Licensing and contribution scope

| Content | License and scope |
|---|---|
| Original teaching prose, equation explanations and course-created diagrams that this repository has the right to license; corresponding Markdown and PDFs | [CC BY 4.0](LICENSE); copying, adaptation, redistribution and commercial use are allowed with attribution, a license link and an indication of changes |
| New build, validation, synchronization and publication tools (`scripts/`, `tests/` and build configuration) | [MIT](LICENSE-CODE); retain copyright and license notices |
| Third-party papers, quoted passages, figures, code, models, data and trademarks | Retain their original terms; citation here does not automatically place them under CC BY or MIT |

Suggested attribution: **Xiong Zeng and contributors, Foundation of Embodied AI**, with the [repository URL](https://github.com/zengxiong111/Foundation_of_Embodied_AI), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) and a description of changes. Preserve original paper or official-code attribution where applicable. Legal texts retain their standard English wording; this section explains scope. Contributions are welcome when you have the right to contribute under the applicable license; identify sources and third-party restrictions in your PR.
