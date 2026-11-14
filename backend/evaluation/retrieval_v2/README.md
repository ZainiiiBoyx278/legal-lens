# Token-bounded retrieval and local answer-support evaluation

This follow-up separates retrieval quality, generated-claim support, and answer
completeness. Questions, reference answers, and claim reviews are **agent-authored**;
there is **no independent human or legal review**. The source is the bundled
historical IPC book, not a verified current-law collection.

## Held-out retrieval results

The frozen run returned the following results on 80 positive queries. All 40
annotated passages were representable as complete chunks in both indexes.

| Metric | v1 BM25 | v2 BM25 | v1 hybrid | v2 hybrid |
| --- | ---: | ---: | ---: | ---: |
| Passage Hit@1 | 33.75% | 36.25% | 33.75% | 28.75% |
| Passage Hit@5 | 82.50% (66/80) | 86.25% (69/80) | 81.25% (65/80) | 80.00% (64/80) |
| Passage Hit@10 | 90.00% | 91.25% | 86.25% | 87.50% |
| MRR@10 | 0.5369 | 0.5600 | 0.5211 | 0.4773 |
| Section-lookup Hit@5 | 85.0% | 95.0% | 92.5% | 87.5% |
| Paraphrase Hit@5 | 80.0% | 77.5% | 70.0% | 72.5% |
| Unrelated queries with empty results | 7/10 | 7/10 | 7/10 | 10/10 |
| Warm CPU service p50 | 22.3 ms | 20.2 ms | 37.1 ms | 36.0 ms |
| Warm CPU service p95 | 31.9 ms | 31.7 ms | 51.5 ms | 49.7 ms |

Encoder-over-limit chunks fell from **1,205/5,474 (22.0%) to 0/6,959**. The
new chunker increases the chunk count by 27.1%; it does not simply discard the
text beyond the encoder limit. The dense index is therefore larger. Both runs
use the same pinned MiniLM encoder and original PDF.

The result is **not a general hybrid-ranking improvement**. v2 hybrid eliminates
the three unrelated-query false matches, but loses one net Hit@5 query and has
lower Hit@1 and MRR. v2 BM25 outperforms hybrid on positive Hit@5 in this set.
No ranker was changed or selected after these results were seen. The answer
experiment continues to use the hybrid configuration frozen beforehand.

A paired bootstrap resampling the 40 topics (keeping their two variants
together) gives a 95% percentile interval for the Hit@5 change of **-10.0 to
+7.5 percentage points** for hybrid and **-3.75 to +11.25 points** for BM25.
These intervals include zero and exclude neither improvement nor deterioration.
They do not account for annotation/author bias. See
[raw results](holdout-retrieval.json) and [paired comparison](retrieval-comparison.json).

All four systems return related passages for the six insufficient-context and
four current-law queries. This does not establish that those questions are
answerable; source guards and answer generation must handle that separately.

### Development history

[The six-candidate sweep](development-grid.json) used only the original 20 topics
and eight unrelated probes. It selected coverage 0.35 and RRF rank constant 20
by negative abstention, then Hit@5, then MRR. Semantic retrieval was allowed to
recover queries rejected by the lexical fusion filter. Standalone BM25 and
missing-model fallback retain their broader content-term matches.

[The final development check](development-final.json) has hybrid Hit@5 28/40
versus v1's 29/40, Hit@10 35/40 versus 32/40, and empty unrelated results 8/8
versus 7/8. Earlier, overly restrictive diagnostic candidates remain in
[development-01](development-01.json) and [development-02](development-02.json).
They were not used as held-out evidence. Later harness changes recorded
cross-platform code hashes and pooling configuration hashes before freezing.

The unchanged evidence prompt also produced a malformed-JSON failure in a
[development smoke test](development-generation-smoke.json). It was retained.
The output budget was increased to match production's 1,600 tokens before the
[final smoke test](development-generation-final.json); the same failure remained.
There was no prompt repair or output selection to hide it.

[Candidate code/configuration](candidate.json) was committed in `a4baef9` before
any held-out scores or completions were inspected. **46 offline backend tests
passed**, including source/policy cache invalidation, missing-model fallback,
no silent embedding truncation, semantic recovery, provider failures, invalid
citations, and grounding-metric denominator checks.

## Local answer results

The local experiment completed on **14 November 2026 (Asia/Calcutta)**. The
review covers all 60 cases and all **110 claims** in the 32 structurally valid
generated responses. Judgments are agent-authored, including the questions and
references; there is no independent human review.

| Measure | Observed result |
| --- | ---: |
| Answerable questions producing a valid generated answer | 30/40 (75.0%) |
| Answerable questions with complete requested facts, irrespective of added claim support | 15/40 (37.5%) |
| Complete answers with every emitted claim supported by its citations | **5/40 (12.5%)** |
| Supported emitted claims | 51/110 (46.4%) |
| Unsupported / contradicted / uncertain claims | 50 / 6 / 3 |
| Schema, citation or completion failures | 13/60 (10 positive, 3 negative) |
| Local model attempts | 46; other cases stopped before generation |
| Generation p50 / p95 | 17.55 s / 63.15 s |

The complete-and-supported successes were `s201`, `s25`, `s277`, `s351`, and
`s96`. This strict measure counts every answerable query in its denominator:
abstentions and validation failures cannot inflate it. An answer that addresses
the requested fact but adds an unsupported assertion also fails this measure.
Repeated claims count separately in the claim totals and are not independent
facts. Borderline wording and garbled statements may be labeled uncertain;
the per-claim explanations make those judgments reviewable.

| Negative category | Cases | Explicit abstentions | Validation failures | Generated answers |
| --- | ---: | ---: | ---: | ---: |
| Unrelated | 10 | 10, by retrieval | 0 | 0 |
| Current law/applicability | 4 | 4, by source guard | 0 | 0 |
| Information absent from the source | 6 | 1, by model | 3 | 2 |

The two generated insufficient-context responses contain unsupported claims.
The 13 validation failures would trigger the production assistant's excerpt
fallback; they are not counted as successful semantic abstentions. The local
model therefore **does not demonstrate reliable legal answering**. These are
results for Qwen2.5-1.5B-Instruct under this protocol, not measurements of the
production OpenAI model.

The experiment exposes several distinct failure modes:

- `n11`: an unrelated arrest passage is incorrectly attributed to person A in a
  theft illustration. The citation exists, but the claimed identity is absent.
- `s20`: the model drops a negation and says a court ceases to exist when a judge
  leaves office; the citation says that is **not** the consequence.
- `s269`: a penalty is present in a retrieved passage, but the claim cites a
  different passage that does not contain the penalty.
- `s374`: both claims accurately reproduce their sources, but those sources
  concern inflammatory writing rather than the question about compelled labour.
- `s79`: the answer permits mistake of law where the cited provision expressly
  distinguishes and excludes it.
- `s506`: corrupted text and an editorial marker become part of the generated
  legal rule; long repeated output also reduces usefulness.

Inspect [all raw completions and excerpts](holdout-generation.json), the
[full claim review](answer-review.json), and the [computed metrics](grounding-metrics.json).
No invalid JSON was repaired and no repeated outputs were selected for quality.
Before making answer-quality claims, a future iteration should address
structured decoding, citation-to-claim alignment, unnecessary extra claims, and
abstention when related text does not contain the answer. That iteration needs
new evaluation data; these cases are now inspected diagnostic examples.

### Interruption recovery

The workspace moved during execution, interrupting the run after 56 saved cases.
The [original partial output](holdout-generation-interrupted.json) is retained.
[A narrow recovery helper](../../scripts/resume_local_answers.py) loaded the
same frozen evaluator, checked model/configuration/source/environment identities,
and generated only the four unfinished cases. All 56 saved case objects compare
equal before and after recovery. Prompt, decoder, source selection and settings
were unchanged; the final output records both the archive hash and recovery-helper
hash. This was execution recovery, not a rerun or selection of better answers.

## Frozen design

- [Protocol](protocol.json), [questions](holdout.json), and the [unchanged v1
  retriever](baselines/retrieval_v1.py) were committed in `dd1533c` before v2
  implementation and held-out ranking.
- The original 40 questions over 20 sections are now **development data**. The
  new set has **80 positive queries over 40 different sections**, with one
  section-number lookup and one paraphrase per section. These are correlated
  pairs, not 80 independent topics.
- The 20 negative probes contain 10 unrelated questions, six questions requiring
  information absent from the source, and four current-law/applicability queries.
  Retrieval may correctly find related passages for an unanswerable question:
  absence of an answer is a separate test from absence of related text.
- All systems index the same 1,871-page PDF. A relevant chunk must contain the
  complete normalized gold quote, including copied occurrences elsewhere in the
  book. This is **passage Hit@k**, not exhaustive recall: other useful passages
  are unjudged. NFKC, case and whitespace normalization handle PDF extraction.
- Compare the preserved v1 and final v2 with both BM25 and hybrid retrieval.
  Settings and code are frozen before the new test run. No changes are selected
  from held-out retrieval or generated answers.

## Implementation

`passage_chunking.py` uses a cached fast tokenizer's source offsets to make
chunks of at most 224 tokens, including special tokens, with 32-token overlap.
It prefers paragraph/sentence boundaries and checks the token count again after
slicing. Evidence stays an exact substring of the PDF extraction.

If the tokenizer is missing, BM25 still works with character chunks. Dense
indexing checks every chunk against the actual encoder limit and refuses silent
truncation. Oversized queries fall back to BM25 without putting other requests
into the model-failure cooldown. Chunk-policy identities invalidate cached
indexes and citation IDs when boundaries change.

Lexical matching normalizes Unicode ligatures and removes function/query words.
The lexical leg of hybrid fusion filters incidental matches using IDF-weighted
query coverage; standalone BM25 and missing-model fallback keep the broader list
requiring at least two distinct content-term matches (one for a one-term query).
Semantic
matching can recover a paraphrase that fails that lexical filter, provided at
least one query term occurs in the corpus and a chunk passes the dense threshold.
All-OOV and stop-word-only queries skip the encoder. These are retrieval
heuristics, not calibrated probabilities of relevance.

## Generated-answer experiment

Run local **Qwen2.5-1.5B-Instruct** at revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, on one RTX 4050 Laptop GPU, using
BF16, SDPA, greedy decoding, seed 20261113, at most 1,600 new tokens, and at most
6,144 input tokens. [The model manifest](local-model.json) pins the exact files.
No paid inference API is used.

The experiment uses one paraphrase per held-out topic and all 20 negatives:
**60 cases**. Each receives the top six final hybrid excerpts. The existing
source-scope rule can abstain before generation. Empty retrieval also abstains.
The remaining cases use the **same prompt, schema and citation validator as the
production assistant**, extracted into `evidence_contract.py`. No retry, JSON
repair, best-of selection, excerpt replacement, or silent prompt truncation is
allowed. Raw failures remain in the output.

This is a **local generator component evaluation**, not a measurement of the
production OpenAI model, learned classifier, HTTP routing, or a deployed service.
Classification is bypassed deliberately to exercise answer generation. API and
classifier fallback behavior is covered separately by offline regression tests.

### Review rubric and denominators

Every delivered claim is reviewed against **only the passages it cites**:

- **Supported:** all material assertions follow from those passages, with no
  missing qualification. Save an exact supporting quote and a reason.
- **Unsupported:** a material assertion is absent from the cited evidence.
- **Contradicted:** the cited passage conflicts with a material assertion.
- **Uncertain:** support cannot be established confidently from the excerpt.

Answer completeness is separately labeled complete, partial, incorrect, or not
answered against the question and source-grounded reference answer. A true but
irrelevant claim is not a complete answer. Citation-ID validation alone does not
establish support.

The scorer verifies every case and claim was reviewed, checks evidence quotes
occur in sources the claim actually cited, and binds the review to the exact
generation-file hash. It validates the review's structure, **not the reviewer's
semantic judgment**. A supported-claim percentage excludes abstentions; the
fully supported-and-complete answer percentage uses **all 40 answerable cases**.
Negative explicit abstentions, validation failures, and generated answers are
reported separately. A parsing failure is not scored as successful abstention.

## Limits

The set is small, source-aware and agent-authored. The same agent wrote questions
and reviewed outputs, so correlated judgment errors and author bias remain.
This is a diagnostic artifact, not expert certification, legal correctness, a
production SLA, or proof of generalization to user traffic. It excludes current
law, multi-document reasoning, exhaustive relevance labels, multilingual input,
adversarial testing, and independent annotation.

Warm CPU service-call latency excludes HTTP, model/index initialization and
generation. Calls are serial and interleaved on a development laptop, with no
load isolation. Generation timing includes every attempted completion, including
failures, but excludes loading weights. Neither latency measure is a server
throughput benchmark.


## Reproduce without an inference API

Use Python 3.12. The retrieval experiment uses the application's locked CPU
requirements; follow the repository's normal backend setup. Keep the GPU
experiment in a **separate environment** to avoid replacing the application's
PyTorch/Transformers versions.

Download the pinned model files once, with network access. These commands return
snapshot directories; supply those directories in the commands below. Do not
pass a mutable branch name as the revision.

```powershell
# CPU/backend environment; run from backend/
python -c "from huggingface_hub import snapshot_download; print(snapshot_download('sentence-transformers/all-MiniLM-L6-v2', revision='1110a243fdf4706b3f48f1d95db1a4f5529b4d41', allow_patterns=['*.json','*.safetensors','vocab.txt','README.md','1_Pooling/*']))"
python scripts/compare_retrieval.py --questions evaluation/retrieval_v2/holdout.json --model-path <MINILM_SNAPSHOT> --output evaluation/retrieval_v2/holdout-retrieval-rerun.json --repeats 3
```

Use an immutable snapshot directory containing exactly the model files recorded
in `candidate.json`, including the pooling configuration. Code hashes normalize
LF/CRLF newlines; model files, questions, outputs and reviews use exact-byte
SHA-256 hashes. The baseline and frozen JSON artifacts are protected against
Git newline conversion. New output paths are mandatory; repeated experiments
cannot silently overwrite the recorded run.

```powershell
# Separate Python 3.12 GPU environment; run from backend/
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r evaluation/retrieval_v2/generation-environment.txt
python -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', revision='989aa7980e4cf806f80c7fef2b1adb7bc71aa306', allow_patterns=['*.json','*.safetensors','merges.txt','vocab.json','README.md','LICENSE']))"
python scripts/evaluate_local_answers.py --retrieval evaluation/retrieval_v2/holdout-retrieval-rerun.json --model-path <QWEN_SNAPSHOT> --output evaluation/retrieval_v2/holdout-generation-rerun.json
python scripts/score_answer_review.py --generation evaluation/retrieval_v2/holdout-generation-rerun.json --review evaluation/retrieval_v2/review-rerun.json --make-template
# Review every emitted claim against its cited excerpts and every answer against
# the source-grounded reference. Fill the review; do not copy old judgments.
python scripts/score_answer_review.py --generation evaluation/retrieval_v2/holdout-generation-rerun.json --review evaluation/retrieval_v2/review-rerun.json --output evaluation/retrieval_v2/grounding-metrics-rerun.json
```

After downloads, evaluation disables Hub network access and uses local files.
The local generator contains no inference API client. Greedy decoding reduces
sampling variation, but results can still differ across hardware/library
versions. The output records model files, configuration, library versions, and
hardware; a regenerated output requires a fresh review bound to its new hash.

To recompute the published review's aggregates without loading any model:

```powershell
python scripts/score_answer_review.py --generation evaluation/retrieval_v2/holdout-generation.json --review evaluation/retrieval_v2/answer-review.json --output evaluation/retrieval_v2/grounding-metrics-recomputed.json
```
