#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Validate course structure, bilingual inventory, and current PDF bindings."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
BUILD_INPUTS = ['scripts/build.mjs', 'scripts/print.css', 'package-lock.json', 'materials.json']


def digest(data):
    return hashlib.sha256(data).hexdigest()


def body(text):
    return re.sub(r'^---\n[\s\S]*?\n---\n', '', text, count=1)


def prose_lines(text):
    fence = None
    for line in body(text).splitlines():
        match = re.match(r'^\s*(`{3,}|~{3,})', line)
        if match:
            mark = match[1][0]
            fence = mark if fence is None else (None if mark == fence else fence)
            continue
        if fence is None:
            yield line


def slug(text):
    text = text.lower().replace(' ', '-')
    return ''.join(c for c in text if unicodedata.category(c)[0] in 'LNM' or c in '-_')


def document_errors(text):
    lines = list(prose_lines(text))
    headings = [re.match(r'^(#{1,6}) (.+)$', l) for l in lines]
    headings = [m for m in headings if m]
    errors = []
    if sum(len(m[1]) == 1 for m in headings) != 1:
        errors.append('exactly one H1 is required')
    previous = 0
    for m in headings:
        level = len(m[1])
        if level > previous + 1:
            errors.append(f'heading level skipped: {m[2]}')
        previous = level
    ids = re.findall(r'<a id="([^"]+)"', '\n'.join(lines))
    if len(ids) != len(set(ids)):
        errors.append('duplicate explicit anchor')
    targets = set(ids) | {slug(m[2]) for m in headings}
    for target in re.findall(r'\]\(#([^)]*)\)', '\n'.join(lines)):
        if target not in targets:
            errors.append(f'broken contents anchor: {target}')
    if not re.search(r'\]\(#course-section-\d+\)', text):
        errors.append('generated clickable contents missing')
    return errors


def binding_errors(record, source, pdf, fingerprint):
    errors = []
    for field, expected in [('source_sha256', digest(source)), ('pdf_sha256', digest(pdf)),
                            ('build_inputs_sha256', fingerprint)]:
        if record.get(field) != expected:
            errors.append(f'outdated build binding: {field}')
    if record.get('math_errors') != 0 or record.get('equation_overflows') != 0:
        errors.append('PDF renderer quality check failed')
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--structure-only', action='store_true')
    args = parser.parse_args()
    subprocess.run(['python3', str(ROOT / 'scripts/sync_docs.py'), '--check'], check=True)
    config = json.loads((ROOT / 'materials.json').read_text())
    errors = []
    expected_md, expected_pdf = set(), set()
    expected_urls = []
    build = {} if args.structure_only else json.loads((ROOT / 'pdf/build-manifest.json').read_text())['documents']
    fingerprint = digest(b''.join((ROOT / p).read_bytes() for p in BUILD_INPUTS))
    for topic in config['topics']:
        headings_by_lang = []
        for lang, doc in topic['documents'].items():
            expected_md.add(doc['md'])
            expected_pdf.add(doc['pdf'])
            text = (ROOT / doc['md']).read_text()
            errors += [f'{doc["md"]}: {e}' for e in document_errors(text)]
            headings_by_lang.append([len(m[1]) for line in prose_lines(text)
                                     if (m := re.match(r'^(#{1,6}) (.+)$', line))])
            peer = topic['documents']['en' if lang == 'zh' else 'zh']['md']
            from urllib.parse import quote
            if quote(peer) not in text:
                errors.append(f'counterpart navigation missing: {doc["md"]}')
            if topic['reviewed_at'] not in text or 'CC BY 4.0' not in text:
                errors.append(f'version/license metadata missing: {doc["md"]}')
            if not args.structure_only:
                rec = build.get(doc['md'], {})
                errors += [f'{doc["md"]}: {e}' for e in binding_errors(
                    rec, (ROOT / doc['md']).read_bytes(), (ROOT / doc['pdf']).read_bytes(), fingerprint)]
                info = subprocess.check_output(['pdfinfo', str(ROOT / doc['pdf'])], text=True)
                pages = int(re.search(r'Pages:\s+(\d+)', info)[1])
                if pages != rec.get('pages') or pages < 1:
                    errors.append(f'PDF page metadata differs: {doc["pdf"]}')
                extracted = subprocess.check_output(['pdftotext', str(ROOT / doc['pdf']), '-'], text=True)
                if 'CC BY 4.0' not in extracted or topic['reviewed_at'] not in extracted:
                    errors.append(f'PDF provenance not extractable: {doc["pdf"]}')
        if headings_by_lang[0] != headings_by_lang[1]:
            errors.append(f'bilingual heading structure differs: {topic["id"]}')
        for ext in ['md', 'pdf']:
            for lang in ['zh', 'en']:
                from urllib.parse import quote
                expected_urls.append(config['site_base'] + quote(topic['documents'][lang]['site_' + ext]))
    if expected_md != {str(p.relative_to(ROOT)) for p in (ROOT / 'md').glob('*.md')}:
        errors.append('Markdown inventory differs from materials.json')
    if expected_pdf != {str(p.relative_to(ROOT)) for p in (ROOT / 'pdf').glob('*.pdf')}:
        errors.append('PDF inventory differs from materials.json')
    for readme in ['README.md', 'README.zh-CN.md']:
        text = (ROOT / readme).read_text()
        if re.findall(r'<a href="([^"]+)" target="_blank">(?:MD|PDF)</a>', text) != expected_urls:
            errors.append(f'{readme}: topic order or material URLs differ from manifest')
    for p in ROOT.rglob('*'):
        if p.is_file() and p.parent == ROOT and p.name.lower().startswith('changelog'):
            errors.append(f'forbidden changelog: {p.name}')
    if errors:
        print('\n'.join(errors), file=sys.stderr)
        return 1
    print(f'PASS: {len(config["topics"])} bilingual topics, {len(expected_md)} Markdown files, '
          f'{len(expected_pdf)} PDFs; navigation, inventory' +
          ('' if args.structure_only else ', source/PDF/build hashes, PDF provenance') + ' validated.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
