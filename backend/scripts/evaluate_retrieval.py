"""Offline passage-grounded BM25/hybrid comparison. Never calls a generation API."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import random
import statistics
import sys
import tempfile
from time import perf_counter
import unicodedata

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def query_metrics(ids, relevant):
    rank = next((rank for rank, item in enumerate(ids, 1) if item in relevant), None)
    return {"hit_at_1": int(rank is not None and rank <= 1),
            "hit_at_5": int(rank is not None and rank <= 5),
            "hit_at_10": int(rank is not None and rank <= 10),
            "mrr_at_10": 1 / rank if rank is not None and rank <= 10 else 0.0}


def validate_questions(benchmark, pages):
    ids = [topic["id"] for topic in benchmark["topics"]] + [q["id"] for q in benchmark["negative_queries"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate benchmark ID")
    for topic in benchmark["topics"]:
        if len(normalize(topic["quote"])) < 40:
            raise ValueError("Gold passage is too short to identify meaningful evidence")
        if not 1 <= topic["page"] <= len(pages) or normalize(topic["quote"]) not in normalize(pages[topic["page"] - 1]):
            raise ValueError(f"Gold passage not found on its annotated page: {topic['id']}")
        for variant in ("section_lookup", "paraphrase"):
            if not topic[variant].strip():
                raise ValueError("Empty positive query")


def mean_metrics(rows):
    return {metric: statistics.mean(row["metrics"][metric] for row in rows)
            for metric in ("hit_at_1", "hit_at_5", "hit_at_10", "mrr_at_10")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=BACKEND / "evaluation/retrieval/questions.json")
    parser.add_argument("--model-path", type=Path, required=True, help="Local immutable MiniLM snapshot directory; no download")
    parser.add_argument("--model-revision", required=True, help="Recorded upstream snapshot SHA (also content-hashed)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.output.exists() or args.repeats < 1 or not args.model_path.is_dir():
        parser.error("Require a new output path, an existing local snapshot, and positive repeats")
    if len(args.model_revision) != 40 or any(c not in "0123456789abcdef" for c in args.model_revision):
        parser.error("Record a full immutable model revision")
    # Backend reads this local model path, and requests cannot download models.
    os.environ["EMBEDDING_MODEL"] = str(args.model_path.resolve())
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["OPENAI_API_KEY"] = ""
    import numpy as np
    import pymupdf
    from app.services import retrieval_service as service, source_service
    benchmark = json.loads(args.questions.read_text(encoding="utf-8"))
    document = source_service.for_domain(benchmark["domain"])
    if document.document_id != benchmark["document_id"] or sha256(document.path) != benchmark["source_sha256"]:
        raise ValueError("Configured source differs from the frozen annotated PDF")
    with pymupdf.open(document.path) as pdf:
        pages = [page.get_text(sort=True) for page in pdf]
    validate_questions(benchmark, pages)
    print(json.dumps({"event": "annotations_validated", "topics": len(benchmark["topics"]), "pages": len(pages)}), flush=True)
    start = perf_counter()
    corpus = service._get_corpus(benchmark["domain"])
    lexical_build_seconds = perf_counter() - start
    normalized_chunks = [normalize(text) for text in corpus.texts]
    gold, questions = {}, []
    for topic in benchmark["topics"]:
        # Identical copied passages elsewhere in the same PDF are valid evidence.
        relevant = {corpus.ids[i] for i, text in enumerate(normalized_chunks) if normalize(topic["quote"]) in text}
        gold[topic["id"]] = sorted(relevant)
        for variant in ("section_lookup", "paraphrase"):
            questions.append({"id": topic["id"] + "-" + variant, "topic": topic["id"],
                              "variant": variant, "query": topic[variant], "answerable": True})
    questions.extend({**q, "variant": "negative", "answerable": False} for q in benchmark["negative_queries"])
    model_files = {str(path.relative_to(args.model_path)).replace("\\", "/"): sha256(path)
                   for path in sorted(args.model_path.rglob("*")) if path.is_file()}
    # Isolate from production caches. Every benchmark starts with a fresh index;
    # cold initialization is reported separately from warm query latency.
    cache_root = BACKEND / "instance"
    cache_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="benchmark-", dir=cache_root) as temporary:
        service.BASE_DIR = Path(temporary)
        start = perf_counter()
        model = service._model()
        model_load_seconds = perf_counter() - start
        start = perf_counter()
        vectors = service._vectors(corpus)
        dense_build_seconds = perf_counter() - start
        if not np.isfinite(vectors).all() or vectors.shape[0] != len(corpus.ids):
            raise RuntimeError("Invalid dense index")
        # Measure truncation directly, without changing the production chunker.
        lengths = model.tokenizer(corpus.texts, truncation=False, add_special_tokens=True, return_length=True)["length"]
        truncation = {"max_seq_length": model.max_seq_length,
                      "chunks_exceeding_encoder_limit": sum(n > model.max_seq_length for n in lengths),
                      "total_chunks": len(lengths)}
        print(json.dumps({"event": "index_ready", "chunks": len(corpus.ids), "dense_build_seconds": dense_build_seconds}), flush=True)
        for method in ("bm25", "hybrid"):
            service.retrieve("historical criminal offences", benchmark["domain"], 10, method)
        rng = random.Random(20261113)
        timings = {method: [] for method in ("bm25", "hybrid")}
        records = {}
        for repeat in range(args.repeats):
            jobs = [(question, method) for question in questions for method in ("bm25", "hybrid")]
            rng.shuffle(jobs)
            for question, method in jobs:
                start = perf_counter()
                response = service.retrieve(question["query"], benchmark["domain"], 10, method)
                elapsed_ms = (perf_counter() - start) * 1000
                timings[method].append(elapsed_ms)
                if response["retrieval"]["warning"]:
                    raise RuntimeError("Hybrid benchmark silently degraded: " + response["retrieval"]["warning"])
                if method == "hybrid" and response["results"] and response["retrieval"]["method"] != "hybrid":
                    raise RuntimeError("Nonempty hybrid response did not use the dense model")
                ids = [result["id"] for result in response["results"]]
                key = (question["id"], method)
                if key in records:
                    if ids != [result["id"] for result in records[key]["results"]]:
                        raise RuntimeError("Repeated ranking changed within one benchmark run")
                    records[key]["latency_ms"].append(elapsed_ms)
                    continue
                record = {**question, "requested_method": method, "actual_method": response["retrieval"]["method"],
                          "latency_ms": [elapsed_ms], "results": response["results"]}
                if question["answerable"]:
                    record["relevant_chunk_ids"] = gold[question["topic"]]
                    record["metrics"] = query_metrics(ids, gold[question["topic"]])
                else:
                    record["abstained"] = not bool(ids)
                records[key] = record
            print(json.dumps({"event": "repeat_complete", "repeat": repeat + 1}), flush=True)
    summaries = {}
    for method in ("bm25", "hybrid"):
        positives = [row for (_, m), row in records.items() if m == method and row["answerable"]]
        negatives = [row for (_, m), row in records.items() if m == method and not row["answerable"]]
        summaries[method] = {"positive_queries": len(positives), "overall": mean_metrics(positives),
                             "by_variant": {variant: mean_metrics([row for row in positives if row["variant"] == variant])
                                            for variant in ("section_lookup", "paraphrase")},
                             "negative_abstention": {"empty_results": sum(row["abstained"] for row in negatives), "queries": len(negatives)},
                             "warm_latency_ms": {"p50": float(np.percentile(timings[method], 50)),
                                                 "p95": float(np.percentile(timings[method], 95)), "samples": len(timings[method])},
                             "actual_methods": dict(Counter(row["actual_method"] for (_, m), row in records.items() if m == method))}
    result = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "questions_sha256": sha256(args.questions), "source_sha256": corpus.pdf_hash,
              "model": {"id": "sentence-transformers/all-MiniLM-L6-v2", "revision": args.model_revision, "files_sha256": model_files},
              "environment": {"python": platform.python_version(), "platform": platform.platform(), "device": "cpu",
                              "versions": {package: version(package) for package in ("torch", "numpy", "sentence-transformers", "transformers", "PyMuPDF", "rank-bm25")}},
              "code_sha256": {name: sha256(BACKEND / name) for name in ("app/services/retrieval_service.py", "app/services/source_service.py", "scripts/evaluate_retrieval.py")},
              "setup_seconds": {"pdf_text_and_annotation_validation_excluded": True, "lexical_index": lexical_build_seconds,
                                "model_load": model_load_seconds, "fresh_dense_index_and_cache_write": dense_build_seconds},
              "encoder_truncation": truncation, "chunk_cache_version": service.CACHE_VERSION,
              "annotation_coverage": {"topics": len(gold), "topics_with_at_least_one_complete_chunk": sum(bool(ids) for ids in gold.values())},
              "protocol": {"top_k": 10, "warm_repeats": args.repeats, "order_seed": 20261113,
                           "timing": "Interleaved randomized serial service calls on CPU; excludes HTTP, generation, source extraction validation and cold initialization. Warm-up uses one fixed nonbenchmark query.",
                           "relevance": "A chunk must contain the complete normalized gold quote. Duplicate copies count. Other valid paraphrases or evidence are unjudged; scores are not exhaustive recall.",
                           "tuning": "No retriever hyperparameters tuned on this set. Agent-authored, source-aware small benchmark; paired variants share 20 topics; no independent legal review."},
              "summary": summaries, "queries": [records[key] for key in sorted(records)]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
