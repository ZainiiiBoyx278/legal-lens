"""Paired topic-level bootstrap of the frozen retrieval results (not tuning)."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
from scripts.evaluate_retrieval import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output path")
    data = json.loads(args.results.read_text(encoding="utf-8"))
    positives = [q for q in data["queries"] if q["answerable"]]
    topics = sorted({q["topic"] for q in positives})
    rng = np.random.default_rng(20261113)
    samples = rng.integers(0, len(topics), size=(10000, len(topics)))
    comparisons = {}
    for method in ("bm25", "hybrid"):
        indexed = {v: {q["id"]: q for q in positives if q["system"] == v + "_" + method} for v in ("v1", "v2")}
        metrics = {}
        for metric in ("hit_at_1", "hit_at_5", "hit_at_10", "mrr_at_10"):
            differences = np.array([np.mean([indexed["v2"][q]["metrics"][metric] - indexed["v1"][q]["metrics"][metric]
                                            for q in indexed["v1"] if indexed["v1"][q]["topic"] == topic]) for topic in topics])
            means = differences[samples].mean(axis=1)
            metrics[metric] = {"v2_minus_v1": float(differences.mean()), "paired_topic_bootstrap_95_percentile_interval": np.percentile(means, [2.5, 97.5]).tolist()}
        comparisons[method] = {"metrics": metrics,
                               "hit_at_5_gains": [q for q in indexed["v1"] if indexed["v2"][q]["metrics"]["hit_at_5"] > indexed["v1"][q]["metrics"]["hit_at_5"]],
                               "hit_at_5_losses": [q for q in indexed["v1"] if indexed["v2"][q]["metrics"]["hit_at_5"] < indexed["v1"][q]["metrics"]["hit_at_5"]]}
    output = {"results_sha256": sha256(args.results), "analysis_sha256": sha256(__file__), "topics": len(topics), "bootstrap_resamples": 10000, "seed": 20261113,
              "interpretation": "Paired resampling keeps both variants of a section together. Descriptive sampling uncertainty only: not a correction for agent-authored labels, selection bias, or related legal topics.", "comparisons": comparisons}
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
