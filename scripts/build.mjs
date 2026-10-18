// SPDX-License-Identifier: MIT
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';
import matter from 'gray-matter';
import MarkdownIt from 'markdown-it';
import texmath from 'markdown-it-texmath';
import katex from 'katex';
import { chromium } from 'playwright';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
process.chdir(root);
const sha = data => crypto.createHash('sha256').update(data).digest('hex');
const config = JSON.parse(fs.readFileSync('materials.json', 'utf8'));
const inputs = ['scripts/build.mjs', 'scripts/print.css', 'package-lock.json', 'materials.json'];
const inputHash = sha(Buffer.concat(inputs.map(p => fs.readFileSync(p))));
const reportPath = 'pdf/build-manifest.json';
const old = fs.existsSync(reportPath) ? JSON.parse(fs.readFileSync(reportPath)) : { documents: {} };
const selected = process.argv.slice(2);
if (selected.some(id => !config.topics.some(t => t.id === id))) throw Error('Unknown topic ID');
const topics = config.topics.filter(t => !selected.length || selected.includes(t.id));
const esc = s => s.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');
const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'embodied-course-'));
const katexDir = path.resolve('node_modules/katex/dist');
const mathCSS = fs.readFileSync(path.join(katexDir, 'katex.min.css'), 'utf8')
  .replaceAll('url(fonts/', `url(file://${katexDir}/fonts/`);
const fonts = ['Noto Serif CJK SC', 'Noto Sans CJK SC', 'Noto Sans Mono CJK SC'].map(family => {
  const file = execFileSync('fc-match', ['-f', '%{file}', family], { encoding: 'utf8' });
  const actual = execFileSync('fc-match', ['-f', '%{family}', family], { encoding: 'utf8' });
  if (!actual.includes(family)) throw Error(`Required font missing: ${family}`);
  return { family, sha256: sha(fs.readFileSync(file)) };
});
const browser = await chromium.launch({
  headless: true,
  ...(process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH } : {}),
  args: ['--no-sandbox'],
});
const report = { schema_version: 1, documents: { ...old.documents } };
try {
  for (const topic of topics) for (const [lang, doc] of Object.entries(topic.documents)) {
    const source = fs.readFileSync(doc.md);
    const { content } = matter(source.toString());
    const title = /^# (.+)$/m.exec(content)?.[1];
    if (!title) throw Error(`Missing title: ${doc.md}`);
    // texmath catches KaTeX exceptions and emits plain error text, without a
    // .katex-error node. Track errors at the engine boundary so none reach PDFs.
    const mathErrors = [];
    const engine = { renderToString(tex, options) {
      try { return katex.renderToString(tex, options); }
      catch (error) { mathErrors.push(error.message); throw error; }
    } };
    const md = new MarkdownIt({ html: true }).use(texmath, {
      engine, delimiters: 'dollars', katexOptions: { throwOnError: true, strict: 'ignore' },
    });
    const baseFence = md.renderer.rules.fence.bind(md.renderer.rules);
    let diagrams = 0;
    md.renderer.rules.fence = (tokens, idx, opts, env, self) => {
      if (tokens[idx].info.trim() === 'mermaid') {
        diagrams++;
        return `<pre class="mermaid">${esc(tokens[idx].content)}</pre>`;
      }
      return baseFence(tokens, idx, opts, env, self);
    };
    const html = `<!doctype html><html lang="${lang === 'zh' ? 'zh-CN' : 'en'}"><head><meta charset="utf-8"><title>${esc(title)}</title><style>${mathCSS}\n${fs.readFileSync('scripts/print.css', 'utf8')}</style></head><body>${md.render(content)}</body></html>`;
    if (mathErrors.length) throw Error(`${doc.md}: ${mathErrors.join('\n')}`);
    const htmlPath = path.join(temp, `${topic.id}-${lang}.html`);
    fs.writeFileSync(htmlPath, html);
    const page = await browser.newPage({ viewport: { width: 658, height: 950 } });
    // Building is offline: external links remain links, and no remote asset may silently change a PDF.
    await page.route(/^https?:/, route => route.abort());
    await page.goto(`file://${htmlPath}`);
    if (diagrams) {
      await page.addScriptTag({ path: path.resolve('node_modules/mermaid/dist/mermaid.min.js') });
      await page.evaluate(async () => {
        mermaid.initialize({
          startOnLoad: false, securityLevel: 'strict', theme: 'base', deterministicIds: true,
          themeVariables: {
            primaryColor: '#f2f2f2', primaryTextColor: '#000000', primaryBorderColor: '#555555',
            secondaryColor: '#ffffff', secondaryTextColor: '#000000', secondaryBorderColor: '#555555',
            tertiaryColor: '#e6e6e6', tertiaryTextColor: '#000000', tertiaryBorderColor: '#555555',
            lineColor: '#444444', textColor: '#000000', mainBkg: '#f2f2f2',
            nodeBorder: '#555555', clusterBkg: '#f7f7f7', clusterBorder: '#999999',
            edgeLabelBackground: '#ffffff',
          },
        });
        await mermaid.run({ querySelector: '.mermaid' });
      });
    }
    await page.evaluate(() => document.fonts.ready);
    const quality = await page.evaluate(() => {
      const width = document.body.clientWidth;
      for (const e of document.querySelectorAll('.katex')) {
        const w = e.getBoundingClientRect().width;
        if (w > width - 12) e.style.fontSize = `${100 * (width - 12) / w}%`;
      }
      return {
        math_nodes: document.querySelectorAll('.katex').length,
        math_errors: document.querySelectorAll('.katex-error').length,
        diagrams_rendered: document.querySelectorAll('.mermaid svg').length,
        equation_overflows: [...document.querySelectorAll('eqn')].filter(e => e.scrollWidth > width + 3).length,
      };
    });
    if (quality.math_errors || quality.equation_overflows || quality.diagrams_rendered !== diagrams) {
      throw Error(`${doc.md}: ${JSON.stringify(quality)}`);
    }
    await page.pdf({
      path: doc.pdf, format: 'A4', printBackground: true,
      margin: { top: '18mm', bottom: '18mm', left: '18mm', right: '18mm' },
      displayHeaderFooter: true, headerTemplate: '<span></span>',
      footerTemplate: '<div style="font:8px sans-serif;color:#777;width:100%;text-align:center"><span class="pageNumber"></span> / <span class="totalPages"></span></div>',
      tagged: true, outline: true,
    });
    const info = execFileSync('pdfinfo', [doc.pdf], { encoding: 'utf8' });
    report.documents[doc.md] = {
      pdf: doc.pdf, source_sha256: sha(source), pdf_sha256: sha(fs.readFileSync(doc.pdf)),
      build_inputs_sha256: inputHash, browser: browser.version(), fonts,
      pages: Number(/Pages:\s+(\d+)/.exec(info)[1]), ...quality,
    };
    await page.close();
    console.log(`${topic.id}/${lang}: ${report.documents[doc.md].pages} pages; ${quality.math_nodes} math nodes; ${diagrams} diagrams`);
  }
  fs.writeFileSync(reportPath, `${JSON.stringify(report, null, 2)}\n`);
} finally {
  await browser.close();
  fs.rmSync(temp, { recursive: true, force: true });
}
