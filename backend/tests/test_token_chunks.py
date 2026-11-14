"""Token boundaries, degraded operation, and source/citation integrity."""
import json
import re
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from app.services import passage_chunking as chunking, retrieval_service as retrieval
from app.services.evidence_contract import parse_answer, build_messages


class OffsetTokenizer:
    model_max_length = 256

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def __call__(self, text, add_special_tokens=True, **kwargs):
        matches = list(re.finditer(r"\w+|[^\w\s]", text))
        return {"input_ids": [1] * (len(matches) + (2 if add_special_tokens else 0)),
                "offset_mapping": [(m.start(), m.end()) for m in matches]}


class ChunkingTests(unittest.TestCase):
    def test_chunks_fit_encoder_and_preserve_source_tokens(self):
        tokenizer = OffsetTokenizer()
        text = "\n\n".join(" ".join(f"word{n}" for n in range(i, i + 80)) + "." for i in range(0, 800, 80))
        chunks = list(chunking.token_chunks(text, tokenizer, budget=96, overlap=12))
        self.assertGreater(len(chunks), 1)
        for part in chunks:
            self.assertIn(part, text)
            self.assertLessEqual(len(tokenizer(part)["input_ids"]), 96)
        for word in re.findall(r"\w+", text):
            self.assertTrue(any(word in part.split() or word + "." in part.split() for part in chunks), word)

    def test_short_empty_and_invalid_budget(self):
        tokenizer = OffsetTokenizer()
        self.assertEqual(list(chunking.token_chunks("", tokenizer)), [])
        self.assertEqual(list(chunking.token_chunks("Short source.", tokenizer)), ["Short source."])
        with self.assertRaises(ValueError):
            list(chunking.token_chunks("source", tokenizer, budget=10, overlap=10))

    def test_missing_tokenizer_uses_lexical_fallback_without_download(self):
        with patch.dict(chunking._profiles, {}, clear=True), patch.dict(chunking._failures, {}, clear=True), patch("transformers.AutoTokenizer.from_pretrained", side_effect=OSError("not cached")) as loader:
            first, second = chunking.profile("uncached"), chunking.profile("uncached")
            self.assertIsNone(first.tokenizer)
            self.assertEqual(first, second)
            self.assertEqual(loader.call_count, 1)
            self.assertTrue(loader.call_args.kwargs["local_files_only"])
            self.assertEqual(list(chunking.chunks("original evidence", first)), ["original evidence"])

    def test_lexical_normalization_handles_ligatures_and_function_words(self):
        self.assertEqual(retrieval.tokenize("Has the historical IPC sufficient e\ufb00ect?"), ["sufficient", "effect"])

    def test_oversized_sources_never_get_silently_embedded(self):
        corpus = SimpleNamespace(vectors=None, texts=["long"], fingerprint="test")
        model = MagicMock(max_seq_length=4)
        model.tokenizer.return_value = {"input_ids": [[1] * 5]}
        with patch.object(retrieval, "_model", return_value=model):
            with self.assertRaises(retrieval.EmbeddingInputTooLongError):
                retrieval._vectors(corpus)
        model.encode.assert_not_called()


class EvidenceContractTests(unittest.TestCase):
    source = {"id": "exact-source-id", "title": "Book", "page": 1, "excerpt": "Evidence.", "corpus_status": "historical"}

    def test_abstention_cannot_smuggle_claims(self):
        payload = {"sufficient_evidence": False, "claims": [{"text": "Unsupported", "citation_ids": [self.source["id"]]}]}
        with self.assertRaisesRegex(ValueError, "must not contain"):
            parse_answer(json.dumps(payload), [self.source])

    def test_valid_membership_is_not_grounding(self):
        # Deliberately false content passes syntax: an entailment review is still needed.
        payload = {"sufficient_evidence": True, "claims": [{"text": "The moon is cheese.", "citation_ids": [self.source["id"]]}]}
        self.assertEqual(len(parse_answer(json.dumps(payload), [self.source]).claims), 1)

    def test_unknown_citation_and_blank_claim_are_rejected(self):
        for text, citation in (("Claim", "unknown"), ("  ", self.source["id"]), ("Claim [S1]", self.source["id"])):
            with self.assertRaises(ValueError):
                parse_answer(json.dumps({"sufficient_evidence": True, "claims": [{"text": text, "citation_ids": [citation]}]}), [self.source])

    def test_prompt_uses_bounded_history_and_unmodified_evidence(self):
        history = [{"role": "user", "content": "x" * 2000} for _ in range(12)]
        messages = build_messages("Question?", history, [self.source])
        payload = json.loads(messages[1]["content"])
        self.assertEqual(len(payload["conversation_context"]), 8)
        self.assertEqual(len(payload["conversation_context"][0]["content"]), 1500)
        self.assertEqual(payload["sources"][0], self.source)


if __name__ == "__main__":
    unittest.main()
