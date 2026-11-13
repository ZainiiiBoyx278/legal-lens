import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_temp = tempfile.TemporaryDirectory()
os.environ["DATABASE_URL"] = "sqlite:///" + str(Path(_temp.name) / "test.db")

from fastapi.testclient import TestClient
from app.main import app
from app import config
from app.services import bm25_service, chatbot_service, knowledge_graph_service


class RegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)
        from app.database import engine
        engine.dispose()
        _temp.cleanup()

    def setUp(self):
        bm25_service._corpus_df = None
        bm25_service._tokens_list = None
        bm25_service._bm25_index = None

    def test_dotenv_loaded_before_database_and_secret_settings(self):
        def fake_load():
            os.environ["SECRET_KEY"] = "test-secret"
            os.environ["DATABASE_URL"] = "sqlite:///from-dotenv.db"
        with patch.dict(os.environ), patch.object(config, "_load_dotenv", side_effect=fake_load):
            settings = config.get_settings()
            self.assertEqual(settings.SECRET_KEY, "test-secret")
            self.assertEqual(settings.DATABASE_URL, "sqlite:///from-dotenv.db")

    def test_demo_search_and_no_matches(self):
        with patch.object(bm25_service._settings, "DATA_PATH", None):
            response = self.client.post("/search/bm25", json={"query": "maternity benefit"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["results"][0]["name"], "maternity.txt")
            self.assertEqual(self.client.post("/search/bm25", json={"query": "zzzzzzzzz"}).json()["results"], [])

    def test_invalid_search_inputs(self):
        for payload in ({"query": " "}, {"query": "law", "top_k": 0}, {"query": "law", "top_k": -1}, {"query": "law", "top_k": 51}):
            self.assertEqual(self.client.post("/search/bm25", json=payload).status_code, 422)

    def test_invalid_corpus_returns_actionable_error(self):
        cases = ["wrong,name\nvalue,file\n", "text,name\n", "text,name\n!!!,file\n"]
        for content in cases:
            self.setUp()
            path = Path(_temp.name) / "corpus.csv"
            path.write_text(content)
            with patch.object(bm25_service._settings, "DATA_PATH", str(path)):
                response = self.client.post("/search/bm25", json={"query": "law"})
                self.assertEqual(response.status_code, 503)
        self.setUp()
        with patch.object(bm25_service._settings, "DATA_PATH", str(Path(_temp.name)/"missing.csv")):
            self.assertEqual(self.client.post("/search/bm25", json={"query": "law"}).status_code, 503)

    def test_duplicate_document_names_keep_separate_nodes(self):
        docs = [{"name": "same", "text": "maternity benefit law"}, {"name": "same", "text": "child labour law"}]
        response = self.client.post("/knowledge-graph/generate", json={"documents": docs})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len({node["id"] for node in data["nodes"]}), 2)
        self.assertTrue(all(edge["source"] != edge["target"] for edge in data["edges"]))
        self.assertEqual(len(data["top_3"]), 2)
        self.assertEqual(self.client.post("/rerank/cosine", json={"documents": docs}).status_code, 200)

    def test_graph_empty_and_malformed_documents(self):
        self.assertEqual(self.client.post("/knowledge-graph/generate", json={"documents": []}).json()["nodes"], [])
        for doc in ({"text": None, "name": "x"}, {"text": 123, "name": "x"}, {"text": " ", "name": "x"}):
            self.assertEqual(self.client.post("/knowledge-graph/generate", json={"documents": [doc]}).status_code, 422)

    def test_feedback_persistence_and_validation(self):
        response = self.client.post("/feedback/submit", json={"message": "Test feedback"})
        self.assertEqual(response.status_code, 200)
        from app.database import SessionLocal
        from app.models.feedback import Feedback
        with SessionLocal() as db:
            self.assertEqual(db.get(Feedback, response.json()["id"]).message, "Test feedback")
        for body in ({"message": " "}, {"message": "test", "email": "invalid"}):
            self.assertEqual(self.client.post("/feedback/submit", json=body).status_code, 422)

    def test_chat_status_no_key_keeps_excerpt_mode_available(self):
        from app.services import classification_service, retrieval_service
        classification = {"status": "unavailable", "model_version": None, "scope": "uncertain", "intent": "uncertain", "note": "test"}
        retrieval = {"results": [], "retrieval": {"method": "bm25", "elapsed_ms": 0.0, "warning": None}, "corpus_status": "historical"}
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}), \
             patch.object(classification_service, "classify", return_value=classification), \
             patch.object(retrieval_service, "retrieve", return_value=retrieval):
            data = self.client.get("/chatbot/status").json()
            self.assertTrue(data["ipc"]["available"])
            self.assertEqual(data["ipc"]["mode"], "excerpts")
            response = self.client.post("/chatbot/ipc", json={"message": "Historical theft definition"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["mode"], "no_evidence")

    def test_missing_chat_domain_pdf_returns_503(self):
        with patch.dict(os.environ, {"CHATBOT_MURDER_PDF": str(Path(_temp.name) / "missing.pdf")}):
            self.assertEqual(self.client.post("/chatbot/murder", json={"message": "Historical definition"}).status_code, 503)

    def test_stale_source_version_and_unknown_source_are_rejected(self):
        from app.services import source_service
        with patch.object(source_service, "metadata", return_value=(1, "a" * 64)):
            self.assertEqual(self.client.get("/sources/ipc-book/pdf", params={"version": "b" * 64}).status_code, 409)
        self.assertEqual(self.client.get("/sources/not-a-source/pdf").status_code, 404)

    def test_retrieval_schema_and_missing_pdf(self):
        for payload in ({"query": " "}, {"query": "law", "method": "unknown"}, {"query": "law", "top_k": 21}):
            self.assertEqual(self.client.post("/search/retrieve", json=payload).status_code, 422)
        with patch.dict(os.environ, {"CHATBOT_IPC_PDF": str(Path(_temp.name) / "missing.pdf")}):
            self.assertEqual(self.client.post("/search/retrieve", json={"query": "historical theft"}).status_code, 503)

    def test_chat_validation_and_removed_auth_routes(self):
        for path, body in [("/chatbot/unknown", {"message": "Hello"}), ("/chatbot/ipc", {"message": " "}), ("/chatbot/ipc", {"message": "Hi", "history": [{"role": "system", "content": "bad"}]}), ("/chatbot/ipc", {"message": "Hi", "domain": "child"})]:
            self.assertEqual(self.client.post(path, json=body).status_code, 422)
        self.assertEqual(self.client.post("/auth/register", json={}).status_code, 404)

if __name__ == "__main__":
    unittest.main()
