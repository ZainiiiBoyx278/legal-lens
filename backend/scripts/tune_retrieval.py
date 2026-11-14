"""Small development-only sweep; never reads the held-out queries or labels."""
import argparse
import json
import os
from pathlib import Path
import sys

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
from scripts.evaluate_retrieval import normalize, sha256, query_metrics, mean_metrics
from scripts.evaluation_integrity import code_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output path")
    os.environ.update(EMBEDDING_MODEL=str(args.model_path.resolve()), OPENAI_API_KEY="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    from app.services import retrieval_service as service
    path = BACKEND / "evaluation/retrieval/questions.json"
    protocol = json.loads((BACKEND / "evaluation/retrieval_v2/protocol.json").read_text())
    if sha256(path) != protocol["development_sha256"]:
        raise ValueError("Development questions changed")
    data = json.loads(path.read_text(encoding="utf-8"))
    service.BASE_DIR = BACKEND / "instance/evaluation_v2/v2"
    corpus = service._get_corpus("ipc")
    if corpus.pdf_hash != data["source_sha256"]:
        raise ValueError("Source changed")
    service._vectors(corpus)
    gold = {t["id"]: {i for i, text in zip(corpus.ids, corpus.texts) if normalize(t["quote"]) in normalize(text)} for t in data["topics"]}
    trials = []
    # Declared grid, equal to six candidates; no held-out queries consulted.
    for coverage in (0.0, 0.15, 0.35):
        for rank_constant in (20, 60):
            service.MIN_QUERY_COVERAGE, service.RRF_RANK_CONSTANT = coverage, rank_constant
            rows, negative = [], []
            for topic in data["topics"]:
                for variant in ("section_lookup", "paraphrase"):
                    response = service.retrieve(topic[variant], top_k=10)
                    if response["retrieval"]["warning"]:
                        raise ValueError(response["retrieval"]["warning"])
                    ids = [s["id"] for s in response["results"]]
                    rows.append({"id": topic["id"] + "-" + variant, "ids": ids, "metrics": query_metrics(ids, gold[topic["id"]])})
            for q in data["negative_queries"]:
                response = service.retrieve(q["query"], top_k=10)
                if response["retrieval"]["warning"]:
                    raise ValueError(response["retrieval"]["warning"])
                negative.append({"id": q["id"], "ids": [s["id"] for s in response["results"]], "abstained": not response["results"]})
            result = {"coverage": coverage, "rrf_rank_constant": rank_constant, "overall": mean_metrics(rows), "negative_abstentions": sum(r["abstained"] for r in negative), "queries": rows, "negatives": negative}
            trials.append(result)
            print(json.dumps({k: v for k, v in result.items() if k not in {"queries", "negatives"}}), flush=True)
    # First maximize abstention among the eight probes; then Hit@5, then MRR.
    # This is model selection on a tiny development set, never a test-set score.
    chosen = max(trials, key=lambda t: (t["negative_abstentions"], t["overall"]["hit_at_5"], t["overall"]["mrr_at_10"], -t["coverage"]))
    output = {"split": "development", "questions_sha256": sha256(path), "selection_rule": "Maximize unrelated abstentions, then Hit@5, then MRR@10; lower coverage wins exact ties",
              "code_sha256": {p: code_sha256(BACKEND / p) for p in ("scripts/tune_retrieval.py", "app/services/retrieval_service.py", "app/services/passage_chunking.py")},
              "chosen": {k: chosen[k] for k in ("coverage", "rrf_rank_constant", "overall", "negative_abstentions")}, "trials": trials}
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
