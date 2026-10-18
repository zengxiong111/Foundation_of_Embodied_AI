#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generate both bilingual course volumes from the 26 manifest-listed PDFs.

The text, math, hyperlinks and illustrations in chapter PDFs are retained.
"""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import pymupdf as fitz
from fontTools import subset
from fontTools.ttLib import TTCollection, TTFont

W, H = 595.28, 841.89
INK, ACCENT, GRAY = (0.11, .17, .29), (.04, .43, .52), (.40, .44, .50)
LIGHT = (.93, .97, .98)


def font_subset(manifest, target):
    path = subprocess.check_output(
        ['fc-match', '-f', '%{file}', 'Noto Sans CJK SC'], text=True).strip()
    if not Path(path).exists():
        raise RuntimeError('Missing Noto Sans CJK SC font')
    if path.lower().endswith('.ttc'):
        fonts = TTCollection(path).fonts
        font = next((f for f in fonts if (f['name'].getDebugName(1) or '').endswith('SC')), fonts[0])
    else:
        font = TTFont(path)
    strings = [
        'FOUNDATION OF EMBODIED AI', '中文课程讲义', 'English Course Reader',
        '具身智能基础 · 全套课程讲义', 'Foundation of Embodied AI',
        '个专题 chapters', '源文件以课程仓库的构建清单为准',
        'Chapters follow the repository manifest', '目录 Contents',
        '点击章节标题可跳转到正文。', 'Select a chapter to jump to the start.',
        '各章保留原始公式、插图与引用；独立 PDF 页码与全书页码可能不同。',
        'Original chapter math and links are preserved; local page numbers may differ.',
        'zengxiong111 / Foundation_of_Embodied_AI', '0123456789',
    ] + [t['title_zh'] for t in manifest['topics']] + [t['title_en'] for t in manifest['topics']]
    opts = subset.Options()
    opts.layout_features = ['*']
    worker = subset.Subsetter(options=opts)
    worker.populate(text=' '.join(strings))
    worker.subset(font)
    font.save(target)
    return str(target)


def write_line(page, x, y, value, size, color=INK):
    page.insert_text((x, y), value, fontname='Course', fontsize=size, color=color)


def make_one(root, config, lang, fontfile):
    zh = lang == 'zh'
    title = '中文课程讲义' if zh else 'English Course Reader'
    entries = []
    for topic in config['topics']:
        file = root / topic['documents'][lang]['pdf']
        if not file.is_file():
            raise FileNotFoundError('Missing chapter PDF: ' + str(file))
        with fitz.open(file) as source:
            if len(source) == 0 or source.is_encrypted:
                raise RuntimeError('Unreadable chapter PDF: ' + str(file))
            entries.append(dict(title=topic['title_zh' if zh else 'title_en'],
                                file=file, count=len(source), outline=source.get_toc()))
    if not 1 <= len(entries) <= 17:
        raise RuntimeError('One-page contents supports 1–17 topics')

    cursor = 2
    for entry in entries:
        entry['start'] = cursor
        cursor += entry['count']

    book = fitz.open()
    cover = book.new_page(width=W, height=H)
    cover.insert_font(fontname='Course', fontfile=fontfile)
    cover.draw_rect(fitz.Rect(0, 0, W, 13), color=ACCENT, fill=ACCENT)
    cover.draw_rect(fitz.Rect(53, 158, 60, 353), color=ACCENT, fill=ACCENT)
    write_line(cover, 79, 183, 'FOUNDATION OF', 27)
    write_line(cover, 79, 222, 'EMBODIED AI', 29)
    write_line(cover, 79, 284, title, 23, ACCENT)
    write_line(cover, 79, 329, '具身智能基础 · 全套课程讲义' if zh else 'Foundation of Embodied AI', 12, GRAY)
    write_line(cover, 79, 405, f'{len(entries)} ' + ('个专题' if zh else 'chapters'), 13)
    write_line(cover, 79, 450, '源文件以课程仓库的构建清单为准' if zh
               else 'Chapters follow the repository manifest', 9, GRAY)
    write_line(cover, 79, 770, 'zengxiong111 / Foundation_of_Embodied_AI', 9, GRAY)

    contents = book.new_page(width=W, height=H)
    contents.insert_font(fontname='Course', fontfile=fontfile)
    contents.draw_rect(fitz.Rect(0, 0, W, 8), color=ACCENT, fill=ACCENT)
    write_line(contents, 52, 64, '目录' if zh else 'Contents', 24)
    write_line(contents, 52, 91, '点击章节标题可跳转到正文。' if zh
               else 'Select a chapter to jump to the start.', 9, GRAY)
    contents.draw_line((52, 110), (543, 110), color=GRAY, width=.7)
    bookmarks = []
    for i, entry in enumerate(entries):
        y = 147 + i * 39
        if i % 2 == 0:
            contents.draw_rect(fitz.Rect(50, y - 21, 545, y + 10),
                               color=LIGHT, fill=LIGHT)
        write_line(contents, 57, y, f'{i + 1:02}', 10, ACCENT)
        write_line(contents, 91, y, entry['title'], 11)
        write_line(contents, 503, y, str(entry['start'] + 1), 10, GRAY)
        bookmarks.append([1, entry['title'], entry['start'] + 1])
    write_line(contents, 52, 759,
               '各章保留原始公式、插图与引用；独立 PDF 页码与全书页码可能不同。' if zh
               else 'Original chapter math and links are preserved; local page numbers may differ.',
               8, GRAY)

    for entry in entries:
        with fitz.open(entry['file']) as source:
            before = len(book)
            book.insert_pdf(source, links=True, annots=True)
            if len(book) != before + entry['count']:
                raise RuntimeError('Page count differs: ' + str(entry['file']))
        inner = 0
        for level, name, page in entry['outline']:
            if level == 2 and 1 <= page <= entry['count'] and inner < 10:
                bookmarks.append([2, name, entry['start'] + page])
                inner += 1

    for i, entry in enumerate(entries):
        y = 147 + i * 39
        book[1].insert_link(dict(kind=fitz.LINK_GOTO,
                                 **{'from': fitz.Rect(54, y - 22, 543, y + 11)},
                                 page=entry['start']))
    bookmarks.sort(key=lambda item: (item[2], item[0]))
    book.set_toc(bookmarks)
    book.set_metadata(dict(title=f'Foundation of Embodied AI - {title}',
                           author='Xiong Zeng and contributors',
                           subject='Bilingual Foundation of Embodied AI course notes'))
    output = root / 'books' / ('Foundation_of_Embodied_AI_Complete_Chinese.pdf' if zh
                               else 'Foundation_of_Embodied_AI_Complete_English.pdf')
    output.parent.mkdir(parents=True, exist_ok=True)
    book.save(output, garbage=4, deflate=True)
    book.close()

    with fitz.open(output) as check:
        assert len(check) == 2 + sum(t['count'] for t in entries)
        assert sum(level == 1 for level, _, _ in check.get_toc()) == len(entries)
        jump_links = [a for a in check[1].get_links() if a['kind'] == fitz.LINK_GOTO]
        assert len(jump_links) == len(entries)
        for link, entry in zip(jump_links, entries):
            assert link['page'] == entry['start'], (link, entry)
    print(f'{output}: {cursor} pages, {len(entries)} topics, {len(jump_links)} TOC links')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    config = json.loads((root / 'materials.json').read_text(encoding='utf-8'))
    with tempfile.TemporaryDirectory() as folder:
        font = font_subset(config, Path(folder) / 'subset.otf')
        for lang in ('zh', 'en'):
            make_one(root, config, lang, font)


if __name__ == '__main__':
    main()
