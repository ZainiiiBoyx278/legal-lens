"""Validate an explicit, output-bound grounding review and compute denominators.

Quoted evidence is checked mechanically; entailment/completeness judgments remain
the reviewer's judgments. This script does not turn citation presence into truth.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
from scripts.evaluate_retrieval import normalize, sha256

SUPPORT = {"supported", "unsupported", "contradicted", "uncertain"}
COMPLETENESS = {"complete", "partial", "incorrect", "not_answered"}


def review_template(generation, benchmark, generation_hash):
    refs = {t["id"] + "-paraphrase": t["reference_answer"] for t in benchmark["topics"]}
    refs.update({q["id"]: q["reference_answer"] for q in benchmark["negative_queries"]})
    return {"schema_version": 1, "generation_sha256": generation_hash,
            "reviewer": "Codex agent; questions and reference answers also agent-authored; no independent human or legal review",
            "rubric": "Supported means every material assertion follows from the cited excerpts, with no missing condition. Unsupported means absent; contradicted means the excerpt conflicts; uncertain means ambiguous. Completeness is separately assessed against the question and source-grounded reference. A supported but irrelevant answer is not complete.",
            "cases": [{"id": case["id"], "reference_answer": refs[case["id"]], "completeness": None, "reason": None,
                       "claims": [{"index": i, "support": None, "evidence": [], "reason": None} for i, _ in enumerate(case["claims"])]}
                      for case in generation["cases"]]}


def score(generation, review):
    cases = {c["id"]: c for c in generation["cases"]}
    reviewed = {c["id"]: c for c in review["cases"]}
    if len(cases) != len(generation["cases"]) or len(reviewed) != len(review["cases"]) or cases.keys() != reviewed.keys():
        raise ValueError("Review must contain every generated case exactly once")
    labels = Counter()
    fully_supported_complete = 0
    per_case = []
    for case_id, case in cases.items():
        rating = reviewed[case_id]
        if rating["completeness"] not in COMPLETENESS or len((rating["reason"] or "").strip()) < 20:
            raise ValueError("Missing answer assessment: " + case_id)
        if (case["mode"] == "generated") == (rating["completeness"] == "not_answered"):
            raise ValueError("Completeness conflicts with generated/abstained mode: " + case_id)
        if not case["answerable"] and rating["completeness"] != ("incorrect" if case["mode"] == "generated" else "not_answered"):
            raise ValueError("Unanswerable case cannot be marked complete")
        if len(rating["claims"]) != len(case["claims"]) or {r["index"] for r in rating["claims"]} != set(range(len(case["claims"]))):
            raise ValueError("Every claim requires exactly one review: " + case_id)
        sources = {s["id"]: s for s in case["sources"]}
        for claim_review in rating["claims"]:
            claim = case["claims"][claim_review["index"]]
            label = claim_review["support"]
            if label not in SUPPORT or len((claim_review["reason"] or "").strip()) < 20:
                raise ValueError("Missing claim judgment: " + case_id)
            if label in {"supported", "contradicted"} and not claim_review["evidence"]:
                raise ValueError("Supported/contradicted judgments require a source quote")
            for evidence in claim_review["evidence"]:
                if evidence["source_id"] not in claim["citation_ids"] or evidence["source_id"] not in sources:
                    raise ValueError("Review quotes a source the claim did not cite")
                quote = normalize(evidence["quote"])
                if len(quote) < 12 or quote not in normalize(sources[evidence["source_id"]]["excerpt"]):
                    raise ValueError("Review quote does not occur in the cited excerpt: " + case_id)
            labels[label] += 1
        all_supported = bool(case["claims"]) and all(r["support"] == "supported" for r in rating["claims"])
        success = case["answerable"] and rating["completeness"] == "complete" and all_supported
        fully_supported_complete += int(success)
        per_case.append({"id": case_id, "mode": case["mode"], "completeness": rating["completeness"], "all_claims_supported": all_supported,
                         "fully_supported_complete_answer": bool(success)})
    positive = [c for c in cases.values() if c["answerable"]]
    negative = [c for c in cases.values() if not c["answerable"]]
    generated_positive = sum(c["mode"] == "generated" for c in positive)
    claims_total = sum(labels.values())
    latency = sorted(c["generation_seconds"] for c in cases.values() if c.get("generation_seconds") is not None)
    # Order-statistic p95; all model attempts, including validation failures.
    p95_index = max(0, __import__("math").ceil(0.95 * len(latency)) - 1)
    result = {"cases": len(cases), "positive_queries": len(positive), "negative_queries": len(negative),
              "mode_counts": dict(Counter(c["mode"] for c in cases.values())),
              "answerable_generation_coverage": {"generated": generated_positive, "total": len(positive), "rate": generated_positive / len(positive) if positive else None},
              "fully_supported_complete_answers": {"count": fully_supported_complete, "total_answerable": len(positive), "rate": fully_supported_complete / len(positive) if positive else None},
              "claim_support": {"labels": {key: labels[key] for key in sorted(SUPPORT)}, "total": claims_total, "supported_rate": labels["supported"] / claims_total if claims_total else None},
              "negative_outcomes": {kind: {"queries": sum(c["kind"] == kind for c in negative),
                                           "explicit_abstentions": sum(c["kind"] == kind and c["mode"] in {"scope_abstention", "retrieval_abstention", "model_abstention"} for c in negative),
                                           "validation_or_context_failures": sum(c["kind"] == kind and c["mode"] in {"validation_failure", "context_failure"} for c in negative),
                                           "generated_answers": sum(c["kind"] == kind and c["mode"] == "generated" for c in negative)} for kind in sorted({c["kind"] for c in negative})},
              "generation_latency_seconds": {"attempts": len(latency), "p50": statistics.median(latency) if latency else None, "p95": latency[p95_index] if latency else None},
              "judgment_limit": review["reviewer"], "per_case": per_case}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation", required=True, type=Path)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--make-template", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    generation = json.loads(args.generation.read_text(encoding="utf-8"))
    if args.make_template:
        if args.review.exists():
            parser.error("Review already exists")
        questions = BACKEND / "evaluation/retrieval_v2/holdout.json"
        if sha256(questions) != generation["questions_sha256"]:
            raise ValueError("Wrong question set")
        payload = review_template(generation, json.loads(questions.read_text(encoding="utf-8")), sha256(args.generation))
        args.review.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
        return
    if not args.output or args.output.exists():
        parser.error("Use a new output path")
    review = json.loads(args.review.read_text(encoding="utf-8"))
    if review["generation_sha256"] != sha256(args.generation):
        raise ValueError("Review is not bound to these exact model outputs")
    result = score(generation, review)
    result.update(generation_sha256=sha256(args.generation), review_sha256=sha256(args.review), scorer_sha256=sha256(__file__))
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in result.items() if k != "per_case"}, indent=2))


if __name__ == "__main__":
    main()
