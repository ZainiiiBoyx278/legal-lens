"""Checks for benchmark scoring and corpus identity; no network or model required."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

SPEC = importlib.util.spec_from_file_location('legal_lens_eval', Path(__file__).with_name('run.py'))
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)
REVIEW_SPEC = importlib.util.spec_from_file_location('legal_lens_review', Path(__file__).with_name('review_answers.py'))
review = importlib.util.module_from_spec(REVIEW_SPEC)
REVIEW_SPEC.loader.exec_module(review)


def response(items):
    return {'results': items, 'retrieval': {'method': 'bm25', 'warning': None}, 'elapsed_ms': 2.5}


class EvaluationScoringTests(unittest.TestCase):
    def test_hit_requires_both_labelled_page_and_section_marker(self):
        case = {'section': 302, 'pages': [1077]}
        items = [{'page': 1077, 'excerpt': 'A different section [s 301].'},
                 {'page': 17, 'excerpt': 'Some reference [s 302].'},
                 {'page': 1077, 'excerpt': '[s 302] Punishment for murder.'}]
        scored = evaluation.score(case, response(items))
        self.assertTrue(scored['hit'])
        self.assertEqual(scored['rank'], 3)
        self.assertAlmostEqual(scored['reciprocal_rank'], 1 / 3)

    def test_wrong_page_or_section_is_a_miss(self):
        case = {'section': 302, 'pages': [1077]}
        scored = evaluation.score(case, response([{'page': 17, 'excerpt': '[s 302]'}, {'page': 1077, 'excerpt': '[s 300]'}]))
        self.assertFalse(scored['hit'])
        self.assertIsNone(scored['rank'])
        self.assertEqual(scored['reciprocal_rank'], 0)

    def test_corpus_hash_and_page_labels_must_match(self):
        dataset = {'corpus_sha256': 'abc', 'cases': [{'id': 'h', 'section': 302, 'pages': [1077]}]}
        corpus = SimpleNamespace(pdf_hash='wrong', pages=[1077], texts=['[s 302]'])
        with self.assertRaisesRegex(ValueError, 'PDF hash'):
            evaluation.label_validation(corpus, dataset)
        corpus.pdf_hash = 'abc'
        corpus.texts = ['[s 300]']
        with self.assertRaisesRegex(ValueError, 'section marker'):
            evaluation.label_validation(corpus, dataset)
        corpus.texts = ['[s 302]']
        evaluation.label_validation(corpus, dataset)

    def test_hybrid_fallback_cannot_inflate_hybrid_results(self):
        results = [
            {'hit': True, 'reciprocal_rank': 1.0, 'elapsed_ms': 5, 'actual_method': 'bm25'},
            {'hit': False, 'reciprocal_rank': 0.0, 'elapsed_ms': 20, 'actual_method': 'hybrid'},
            {'unavailable': 'no model'},
        ]
        summary = evaluation.summarize_method(results, 'hybrid')
        self.assertEqual((summary['hits'], summary['total'], summary['fallbacks'], summary['unavailable']), (0, 1, 1, 1))
        self.assertEqual(summary['p50_ms'], 20)

    def test_manual_review_keeps_unknown_and_unreviewed_separate(self):
        rows = [{'id': 'a', 'correctness': 'uncertain', 'reviewer': 'reviewer', 'response_reference': 'private-1'},
                {'id': 'b', 'correctness': ''}]
        summary = review.summarize(rows)
        self.assertEqual((summary['reviewed'], summary['unreviewed']), (1, 1))
        self.assertEqual(summary['ratings']['correctness']['uncertain'], 1)
        self.assertEqual(summary['ratings']['correctness']['yes'], 0)

    def test_manual_review_requires_source_reference_and_reviewer(self):
        with self.assertRaisesRegex(ValueError, 'reviewer and response reference'):
            review.summarize([{'id': 'a', 'correctness': 'yes'}])


if __name__ == '__main__':
    unittest.main()
