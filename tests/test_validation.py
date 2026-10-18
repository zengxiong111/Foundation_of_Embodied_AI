# SPDX-License-Identifier: MIT
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('course_check', Path(__file__).resolve().parents[1] / 'scripts/check.py')
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)
GOOD = '# Title\n\n## Contents\n\n- [Section](#course-section-01)\n\n<a id="course-section-01"></a>\n\n## Section\n'


class CourseValidationTests(unittest.TestCase):
    def test_valid_document(self):
        self.assertEqual(check.document_errors(GOOD), [])

    def test_code_comments_do_not_become_headings(self):
        self.assertEqual(check.document_errors(GOOD + '\n```python\n# a comment\n```\n'), [])

    def test_duplicate_title_rejected(self):
        self.assertIn('exactly one H1 is required', check.document_errors(GOOD + '\n# Other title\n'))

    def test_broken_toc_rejected(self):
        self.assertTrue(any('broken contents anchor' in e for e in check.document_errors(GOOD.replace('](#course-section-01)', '](#course-section-99)'))))

    def test_skipped_level_rejected(self):
        self.assertTrue(any('heading level skipped' in e for e in check.document_errors(GOOD.replace('## Section', '#### Section'))))

    def test_duplicate_explicit_anchor_rejected(self):
        self.assertIn('duplicate explicit anchor', check.document_errors(GOOD + '\n<a id="course-section-01"></a>'))

    def test_stale_source_pdf_and_renderer_rejected(self):
        self.assertEqual(len(check.binding_errors({'math_errors': 0, 'equation_overflows': 0}, b'changed source', b'changed PDF', 'new renderer')), 3)


if __name__ == '__main__':
    unittest.main()
