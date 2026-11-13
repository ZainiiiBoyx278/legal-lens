"""Compare the preserved v1 retriever with v2, using a frozen question file.

No generation, network, or provider calls. Indexes live in an ignored local cache.
New output files are mandatory so a failed experiment cannot be overwritten.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.util
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import random
import sys
from time import perf_counter

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
from scripts.evaluate_retrieval import normalize, sha256, query_metrics, validate_questions, mean_metrics
from scripts.evaluation_integrity import code_sha256


def load_baseline():
    path = BACKEND / "evaluation/retrieval_v2/baselines/retrieval_v1.py"
    spec = importlib.util.spec_from_file_location("retrieval_v1", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--coverage", type=float, help="Development-only coverage override")
    args = parser.parse_args()
    if args.output.exists() or args.repeats < 1:
        parser.error("Use a new output path and positive repeat count")
    os.environ.update(EMBEDDING_MODEL=str(args.model_path.resolve()), OPENAI_API_KEY="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    import numpy as np
    import pymupdf
    from app.services import retrieval_service as v2, source_service
    protocol = json.loads((BACKEND / "evaluation/retrieval_v2/protocol.json").read_text())
    expected_hash = protocol["development_sha256" if args.development else "holdout_sha256"]
    if sha256(args.questions) != expected_hash:
        raise ValueError("Question file differs from frozen protocol")
    if args.coverage is not None:
        if not args.development or not 0 <= args.coverage <= 1:
            parser.error("Coverage overrides require development mode and a value in [0, 1]")
        v2.MIN_QUERY_COVERAGE = args.coverage
    code_paths = ["app/services/retrieval_service.py", "app/services/passage_chunking.py", "app/services/source_service.py", "scripts/compare_retrieval.py", "scripts/evaluate_retrieval.py", "scripts/evaluation_integrity.py"]
    code_hashes = {p: code_sha256(BACKEND / p) for p in code_paths}
    model_hashes = {p.relative_to(args.model_path).as_posix(): sha256(p) for p in args.model_path.rglob("*") if p.is_file()}
    if not args.development:
        frozen = json.loads((BACKEND / "evaluation/retrieval_v2/candidate.json").read_text())
        if frozen["retrieval_code_sha256"] != code_hashes:
            raise ValueError("Retrieval code differs from frozen candidate")
        if frozen["embedding_files_sha256"] != model_hashes:
            raise ValueError("Embedding model differs from frozen candidate")
    baseline = load_baseline()
    if sha256(Path(baseline.__file__)) != protocol["baseline_sha256"]:
        raise ValueError("Baseline snapshot changed")
    services = {"v1": baseline, "v2": v2}
    benchmark = json.loads(args.questions.read_text(encoding="utf-8"))
    document = source_service.for_domain(benchmark["domain"])
    if document.document_id != benchmark["document_id"] or sha256(document.path) != benchmark["source_sha256"]:
        raise ValueError("Wrong source PDF")
    with pymupdf.open(document.path) as pdf:
        validate_questions(benchmark, [page.get_text(sort=True) for page in pdf])
    questions = [{"id": t["id"] + "-" + variant, "topic": t["id"], "variant": variant, "kind": "answerable", "answerable": True, "query": t[variant]}
                 for t in benchmark["topics"] for variant in ("section_lookup", "paraphrase")]
    questions += [{**q, "kind": q.get("kind", "unrelated"), "answerable": False} for q in benchmark["negative_queries"]]
    setup, gold = {}, {}
    for name, service in services.items():
        service.BASE_DIR = BACKEND / "instance/evaluation_v2" / name
        start = perf_counter()
        corpus = service._get_corpus(benchmark["domain"])
        index_seconds = perf_counter() - start
        model = service._model()
        start = perf_counter()
        service._vectors(corpus)
        dense_seconds = perf_counter() - start
        lengths = model.tokenizer(corpus.texts, add_special_tokens=True, truncation=False)["input_ids"]
        normalized = [normalize(t) for t in corpus.texts]
        gold[name] = {t["id"]: [chunk_id for chunk_id, text in zip(corpus.ids, normalized) if normalize(t["quote"]) in text] for t in benchmark["topics"]}
        setup[name] = {"chunks": len(corpus.ids), "over_encoder_limit": sum(len(t) > model.max_seq_length for t in lengths), "encoder_limit": model.max_seq_length,
                       "topics_with_complete_gold_chunk": sum(bool(ids) for ids in gold[name].values()), "lexical_build_seconds": index_seconds,
                       "dense_load_or_build_seconds": dense_seconds, "corpus_fingerprint": corpus.fingerprint}
        for method in ("bm25", "hybrid"):
            service.retrieve("historical criminal offences", top_k=10, method=method)
        print(json.dumps({"event": "index_ready", "system": name, **setup[name]}), flush=True)
    records = {}
    rng = random.Random(20261113)
    for repeat in range(args.repeats):
        jobs = [(q, name, method) for q in questions for name in services for method in ("bm25", "hybrid")]
        rng.shuffle(jobs)
        for q, name, method in jobs:
            start = perf_counter()
            response = services[name].retrieve(q["query"], benchmark["domain"], top_k=10, method=method)
            elapsed = (perf_counter() - start) * 1000
            if response["retrieval"]["warning"] or (method == "hybrid" and response["results"] and response["retrieval"]["method"] != "hybrid"):
                raise ValueError("Dense benchmark degraded: " + str(response["retrieval"]))
            system = name + "_" + method
            key = (q["id"], system)
            ids = [s["id"] for s in response["results"]]
            if key in records:
                if ids != [s["id"] for s in records[key]["results"]]:
                    raise ValueError("Nondeterministic repeated ranking")
                records[key]["latency_ms"].append(elapsed)
                continue
            row = {**q, "system": system, "actual_method": response["retrieval"]["method"], "latency_ms": [elapsed], "results": response["results"]}
            if q["answerable"]:
                row.update(relevant_chunk_ids=gold[name][q["topic"]], metrics=query_metrics(ids, gold[name][q["topic"]]))
            else:
                row["abstained"] = not bool(ids)
            records[key] = row
        print(json.dumps({"event": "repeat_complete", "repeat": repeat + 1}), flush=True)
    summaries = {}
    for system in sorted({row["system"] for row in records.values()}):
        rows = [r for r in records.values() if r["system"] == system]
        pos, neg = [r for r in rows if r["answerable"]], [r for r in rows if not r["answerable"]]
        timings = [t for r in rows for t in r["latency_ms"]]
        summaries[system] = {"positive_queries": len(pos), "overall": mean_metrics(pos),
                             "by_variant": {v: mean_metrics([r for r in pos if r["variant"] == v]) for v in ("section_lookup", "paraphrase")},
                             "negative_abstention": {kind: {"empty_results": sum(r["abstained"] for r in neg if r["kind"] == kind), "queries": sum(r["kind"] == kind for r in neg)} for kind in sorted({r["kind"] for r in neg})},
                             "warm_latency_ms": {"p50": float(np.percentile(timings, 50)), "p95": float(np.percentile(timings, 95)), "samples": len(timings)},
                             "actual_methods": dict(Counter(r["actual_method"] for r in rows))}
    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "split": "development" if args.development else "holdout", "questions_sha256": sha256(args.questions),
              "source_sha256": benchmark["source_sha256"], "retrieval_code_sha256": code_hashes, "baseline_sha256": protocol["baseline_sha256"],
              "code_hash_normalization": "UTF-8 with LF newlines; data and model hashes use raw bytes",
              "model_files_sha256": model_hashes,
              "environment": {"python": platform.python_version(), "platform": platform.platform(), "device": "cpu", "versions": {p: version(p) for p in ("torch", "transformers", "sentence-transformers", "numpy", "PyMuPDF", "rank-bm25")}},
              "settings": {"coverage": v2.MIN_QUERY_COVERAGE, "dense_similarity": v2.MIN_DENSE_SIMILARITY, "rrf_rank_constant": v2.RRF_RANK_CONSTANT, "repeats": args.repeats, "seed": 20261113},
              "setup": setup, "summary": summaries, "queries": [records[k] for k in sorted(records)]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
