import copy
import unittest
from scripts.score_answer_review import score


class GroundingMetricTests(unittest.TestCase):
    def setUp(self):
        self.generation = {"cases": [
            {"id": "a", "answerable": True, "kind": "answerable", "mode": "generated", "sources": [{"id": "s", "excerpt": "The limit is one month."}],
             "claims": [{"text": "The limit is one month.", "citation_ids": ["s"]}]},
            {"id": "b", "answerable": True, "kind": "answerable", "mode": "model_abstention", "sources": [], "claims": []},
            {"id": "c", "answerable": False, "kind": "unrelated", "mode": "validation_failure", "sources": [], "claims": []},
        ]}
        self.review = {"reviewer": "Agent test fixture, no independent review", "cases": [
            {"id": "a", "completeness": "complete", "reason": "Answers the requested limit exactly.",
             "claims": [{"index": 0, "support": "supported", "reason": "The claim restates the cited limit.", "evidence": [{"source_id": "s", "quote": "The limit is one month."}]}]},
            {"id": "b", "completeness": "not_answered", "reason": "The model explicitly declined to answer.", "claims": []},
            {"id": "c", "completeness": "not_answered", "reason": "Malformed generation was not delivered.", "claims": []},
        ]}

    def test_abstentions_cannot_inflate_grounded_answer_coverage(self):
        result = score(self.generation, self.review)
        self.assertEqual(result["claim_support"]["supported_rate"], 1.0)
        self.assertEqual(result["fully_supported_complete_answers"]["rate"], 0.5)
        self.assertEqual(result["negative_outcomes"]["unrelated"]["explicit_abstentions"], 0)
        self.assertEqual(result["negative_outcomes"]["unrelated"]["validation_or_context_failures"], 1)

    def test_unsupported_claim_reduces_support_and_success(self):
        review = copy.deepcopy(self.review)
        review["cases"][0]["claims"][0].update(support="unsupported", evidence=[])
        result = score(self.generation, review)
        self.assertEqual(result["claim_support"]["supported_rate"], 0)
        self.assertEqual(result["fully_supported_complete_answers"]["rate"], 0)

    def test_invented_quote_missing_judgment_and_duplicate_case_rejected(self):
        for change in ("quote", "missing", "duplicate"):
            review = copy.deepcopy(self.review)
            if change == "quote":
                review["cases"][0]["claims"][0]["evidence"][0]["quote"] = "Not present in the source at all."
            elif change == "missing":
                review["cases"][0]["claims"] = []
            else:
                review["cases"].append(copy.deepcopy(review["cases"][0]))
            with self.assertRaises(ValueError):
                score(self.generation, review)


if __name__ == "__main__":
    unittest.main()
