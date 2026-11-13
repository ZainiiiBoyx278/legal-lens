"""Failure-injection tests: no network, paid generation or downloaded weights."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import httpx
import numpy as np
from openai import APITimeoutError, AuthenticationError, RateLimitError
import pymupdf

from app.services import chatbot_service as chat, retrieval_service as retrieval, source_service
from scripts.evaluate_retrieval import normalize, query_metrics, validate_questions

READY = {"status": "ready", "model_version": "test", "scope": "historical", "intent": "explanation", "note": "test"}
UNAVAILABLE = {**READY, "status": "unavailable", "scope": "uncertain", "intent": "uncertain"}
SOURCE = {"id": "ipc-book:test:p1:c1", "document_id": "ipc-book", "title": "Test source", "page": 1,
          "excerpt": "A short historical passage about criminal law.", "url": "/sources/ipc-book/pdf#page=1",
          "corpus_status": "historical", "score": 1.0, "rank": 1}
RESPONSE = {"results": [SOURCE], "retrieval": {"method": "bm25", "elapsed_ms": 1.0, "warning": None}, "corpus_status": "historical"}


class ChatReliabilityTests(unittest.TestCase):
    def setUp(self):
        self.settings = SimpleNamespace(OPENAI_API_KEY="fake-test-key", OPENAI_MODEL="gpt-3.5-turbo")
        for target, value in (("get_settings", self.settings),):
            patcher = patch.object(chat, target, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)
        for target, value in ((chat.classification_service, READY), (chat.retrieval_service, RESPONSE)):
            name = "classify" if target is chat.classification_service else "retrieve"
            patcher = patch.object(target, name, return_value=value)
            mock = patcher.start()
            setattr(self, name, mock)
            self.addCleanup(patcher.stop)
        patcher = patch.dict(chat._provider_circuits, {}, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def ask(self, **kwargs):
        return chat.chat("ipc", "Explain this book's historical theft definition.", **kwargs)

    def test_missing_key_preserves_page_linked_evidence(self):
        self.settings.OPENAI_API_KEY = None
        with patch.object(chat, "_generate") as generate:
            result = self.ask()
        generate.assert_not_called()
        self.assertEqual(result["mode"], "excerpts")
        self.assertEqual(result["sources"], [SOURCE])
        self.assertIn("No AI provider key", result["reason"])
        self.assertIn("not an AI-generated answer", result["response"])

    def test_explicit_excerpt_mode_skips_provider(self):
        with patch.object(chat, "_generate") as generate:
            self.assertEqual(self.ask(answer_mode="excerpts")["mode"], "excerpts")
        generate.assert_not_called()

    def test_missing_or_uncertain_classifier_skips_provider(self):
        for classification in (UNAVAILABLE, {**READY, "scope": "uncertain"}):
            self.classify.return_value = classification
            with patch.object(chat, "_generate") as generate:
                result = self.ask()
            generate.assert_not_called()
            self.assertEqual(result["mode"], "excerpts")

    def test_empty_retrieval_does_not_generate(self):
        self.retrieve.return_value = {**RESPONSE, "results": []}
        with patch.object(chat, "_generate") as generate:
            result = self.ask()
        generate.assert_not_called()
        self.assertEqual(result["mode"], "no_evidence")
        self.assertEqual(result["sources"], [])

    def test_current_law_guard_works_without_classifier(self):
        self.classify.return_value = UNAVAILABLE
        for query in ("Compare IPC with BNS", "What is the current criminal law?"):
            with patch.object(chat, "_generate") as generate:
                result = chat.chat("ipc", query)
            self.retrieve.assert_not_called()
            generate.assert_not_called()
            self.assertEqual(result["mode"], "no_evidence")
            self.assertIn("Retrieval skipped", result["retrieval"]["warning"])

    def test_timeout_falls_back_and_circuit_retries_after_expiry(self):
        clock = [100.0]
        error = APITimeoutError(request=httpx.Request("POST", "https://provider.invalid"))
        with patch.object(chat, "monotonic", side_effect=lambda: clock[0]), patch.object(chat, "_generate", side_effect=error) as generate:
            first, second = self.ask(), self.ask()
            self.assertEqual(generate.call_count, 1)
            self.assertEqual(first["mode"], "excerpts")
            self.assertEqual(second["sources"], [SOURCE])
            clock[0] = 161.0
            self.ask()
            self.assertEqual(generate.call_count, 2)

    def test_auth_and_quota_use_five_minute_cooldown(self):
        response = httpx.Response(401, request=httpx.Request("POST", "https://provider.invalid"))
        errors = [AuthenticationError("fake-test-key must never leak", response=response, body={}),
                  RateLimitError("quota", response=response, body={"code": "insufficient_quota"})]
        for error in errors:
            chat._provider_circuits.clear()
            with patch.object(chat, "monotonic", return_value=100), patch.object(chat, "_generate", side_effect=error) as generate:
                result = self.ask()
                self.ask()
            self.assertEqual(generate.call_count, 1)
            self.assertEqual(next(iter(chat._provider_circuits.values()))[0], 400)
            self.assertEqual(result["mode"], "excerpts")
            self.assertNotIn("fake-test-key", json.dumps(result))

    def test_insufficient_generated_evidence_is_explicit(self):
        with patch.object(chat, "_generate", return_value=chat.EvidenceAnswer(sufficient_evidence=False, claims=[])):
            result = self.ask()
        self.assertEqual(result["mode"], "no_evidence")
        self.assertEqual(result["sources"], [SOURCE])

    def test_validated_generation_and_history_are_request_scoped(self):
        answer = chat.EvidenceAnswer(sufficient_evidence=True, claims=[chat.EvidenceClaim(text="Test claim", citation_ids=[SOURCE["id"]])])
        history = [{"role": "user", "content": "previous request"}]
        with patch.object(chat, "_generate", return_value=answer) as generate:
            first = self.ask(history=history)
            self.assertEqual(generate.call_args.args[2], history)
            self.ask()
            self.assertEqual(generate.call_args.args[2], [])
        self.assertEqual(first["response"], "Test claim [S1]")
        self.assertEqual(first["mode"], "generated")

    def test_real_validation_rejects_unknown_citation_malformed_or_truncated_json(self):
        cases = [('{"sufficient_evidence":true,"claims":[{"text":"Claim","citation_ids":["invented"]}]}', "stop"),
                 ('not json', "stop"),
                 ('{"sufficient_evidence":true,"claims":[]}', "stop"),
                 (json.dumps({"sufficient_evidence": True, "claims": [{"text": "See https://invented.invalid", "citation_ids": [SOURCE["id"]]}]}), "stop"),
                 ('{}', "length")]
        for content, finish in cases:
            client = MagicMock()
            client.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish, message=SimpleNamespace(content=content))])
            with patch("openai.OpenAI") as constructor:
                constructor.return_value.__enter__.return_value = client
                result = self.ask()
            self.assertEqual(result["mode"], "excerpts")
            self.assertIn("validation", result["reason"])
            self.assertEqual(constructor.call_args.kwargs["timeout"], 30.0)
            self.assertEqual(constructor.call_args.kwargs["max_retries"], 0)


class RetrievalReliabilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "source.pdf"
        self.write_pdf("Unique quokka evidence concerning wildlife.")
        self.document = source_service.SourceDocument("test-source", "Test", ("ipc",), self.path, "unverified")
        patcher = patch.object(source_service, "for_domain", return_value=self.document)
        patcher.start()
        self.addCleanup(patcher.stop)
        for cache in (retrieval._corpora, retrieval._dense_failures):
            patcher = patch.dict(cache, {}, clear=True)
            patcher.start()
            self.addCleanup(patcher.stop)

    def write_pdf(self, first_page):
        pdf = pymupdf.open()
        for text in (first_page, "Garden flowers and sunny weather.", "Computers execute arithmetic instructions.", "Cooking recipes describe ingredients."):
            pdf.new_page().insert_text((72, 72), text)
        pdf.save(self.path)
        pdf.close()

    def test_missing_dense_model_falls_back_and_retries_after_cooldown(self):
        clock = [100.0]
        with patch.object(retrieval, "monotonic", side_effect=lambda: clock[0]), patch.object(retrieval, "_vectors", side_effect=OSError("missing local model")) as vectors:
            first = retrieval.retrieve("unique quokka", method="hybrid")
            second = retrieval.retrieve("unique quokka", method="hybrid")
            self.assertEqual(vectors.call_count, 1)
            clock[0] = 161
            third = retrieval.retrieve("unique quokka", method="hybrid")
            self.assertEqual(vectors.call_count, 2)
        for result in (first, second, third):
            self.assertEqual(result["retrieval"]["method"], "bm25")
            self.assertIn("unavailable", result["retrieval"]["warning"])
            self.assertEqual(result["results"][0]["page"], 1)
            self.assertIn("?version=", result["results"][0]["url"])

    def test_absent_matches_and_stop_words_do_not_call_dense_model(self):
        with patch.object(retrieval, "_model") as model:
            for query in ("zzzzqqqq", "the and of"):
                self.assertEqual(retrieval.retrieve(query)["results"], [])
        model.assert_not_called()

    def test_source_change_invalidates_corpus_and_evidence_ids(self):
        original = retrieval.retrieve("unique quokka", method="bm25")["results"][0]
        self.path.unlink()
        self.write_pdf("Unique quokka NEW evidence concerning wildlife.")
        updated = retrieval.retrieve("unique quokka", method="bm25")["results"][0]
        self.assertNotEqual(original["id"], updated["id"])
        self.assertNotEqual(original["url"], updated["url"])
        self.assertIn("NEW", updated["excerpt"])
        self.assertEqual(len(retrieval._corpora), 1)

    def test_missing_pdf_and_scanned_only_pdf_fail_explicitly(self):
        self.path.unlink()
        with self.assertRaisesRegex(ValueError, "No PDF"):
            retrieval.retrieve("quokka")
        pdf = pymupdf.open()
        pdf.new_page()
        pdf.save(self.path)
        pdf.close()
        with self.assertRaisesRegex(ValueError, "OCR"):
            retrieval.retrieve("quokka")

    def test_embedding_cache_is_reused_and_corruption_rebuilt(self):
        corpus = retrieval._get_corpus("ipc")
        model = MagicMock()
        model.encode.return_value = np.ones((len(corpus.ids), 3), dtype=np.float32)
        with patch.object(retrieval, "BASE_DIR", Path(self.temp.name)), patch.object(retrieval, "_model", return_value=model):
            first = retrieval._vectors(corpus)
            corpus.vectors = None
            np.testing.assert_array_equal(first, retrieval._vectors(corpus))
            self.assertEqual(model.encode.call_count, 1)
            corpus.vectors = None
            next((Path(self.temp.name) / "instance/embeddings").glob("*.npz")).write_bytes(b"broken")
            retrieval._vectors(corpus)
            self.assertEqual(model.encode.call_count, 2)


class BenchmarkMetricTests(unittest.TestCase):
    def test_ranks_count_chunks_and_do_not_confuse_page_with_passage(self):
        result = query_metrics(["p1:wrong", "p2:wrong", "p1:right"], {"p1:right"})
        self.assertEqual(result, {"hit_at_1": 0, "hit_at_5": 1, "hit_at_10": 1, "mrr_at_10": 1 / 3})
        self.assertEqual(query_metrics([], {"right"})["mrr_at_10"], 0)

    def test_unicode_ligatures_and_whitespace_normalize(self):
        self.assertEqual(normalize("su\ufb03cient\n cause"), "sufficient cause")

    def test_wrong_page_annotation_is_rejected(self):
        topic = {"id": "x", "page": 1, "quote": "This is a sufficiently long but nonexistent gold passage.", "section_lookup": "a", "paraphrase": "b"}
        with self.assertRaisesRegex(ValueError, "not found"):
            validate_questions({"topics": [topic], "negative_queries": []}, ["different text"])


if __name__ == "__main__":
    unittest.main()
