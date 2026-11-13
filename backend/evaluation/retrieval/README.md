# Passage-grounded retrieval evaluation

This benchmark compares the **existing** BM25 and MiniLM + BM25 reciprocal-rank
fusion implementations. No ranking thresholds, fusion weights, chunking settings,
or model weights were tuned on this set.

## Measured result

On 40 positive queries over the bundled 1,871-page historical IPC PDF, hybrid
retrieval found a chunk containing the annotated supporting passage in its top
five for **29 queries (72.5%)**, compared with **26 (65.0%)** for BM25. This is a
7.5 percentage-point difference on this small benchmark, not a production or
legal-answer accuracy claim.

| Metric | BM25 | Hybrid |
| --- | ---: | ---: |
| Passage Hit@1 | 25.0% (10/40) | 37.5% (15/40) |
| Passage Hit@5 | 65.0% (26/40) | 72.5% (29/40) |
| Passage Hit@10 | 75.0% (30/40) | 80.0% (32/40) |
| MRR@10 | 0.4346 | 0.5159 |
| Section-lookup Hit@5 | 75.0% (15/20) | 80.0% (16/20) |
| Paraphrase Hit@5 | 55.0% (11/20) | 65.0% (13/20) |
| Unrelated queries with no results | 7/8 | 7/8 |
| Warm service-call p50 | 19.3 ms | 31.0 ms |
| Warm service-call p95 | 31.4 ms | 49.8 ms |

All figures come from [results.json](results.json), which retains every query,
ranked excerpt, gold chunk ID, latency sample, model-file hash, source hash,
dependency version and evaluation-code hash.

## What is being measured

- **40 source-grounded questions:** 20 topics, each with a section-number lookup
  and a paraphrase. These paired variants are correlated, not 40 independent
  legal topics. Eight separate unrelated queries probe empty-result behavior.
- **Annotation before scoring:** questions and supporting statutory passages were
  authored after inspecting the source PDF and before viewing retrieval scores.
  They are agent-authored and source-aware, with no independent legal review.
- **Evidence, not just a page number:** a hit requires a returned chunk to contain
  the complete gold passage after Unicode NFKC/case/whitespace normalization.
  Another chunk on the same page does not count. Identical copies elsewhere in
  the PDF do count. All 20 gold passages occur in at least one indexed chunk.
- **Non-exhaustive judgments:** valid alternative passages and paraphrases are
  unjudged. The metrics measure recovery of selected evidence; they are not
  exhaustive relevance recall. Hit@k is binary per query. MRR@10 is the reciprocal
  rank of the first matching chunk, or zero if absent from the first ten.
- **Frozen corpus:** SHA-256
  `5706a1b995df774c4c4ea1868223e18a13ba619977d323d3cab76a1cc095e237`;
  physical PDF page numbers, 5,474 chunks, chunk-cache version
  `page-aware-1100-140-v1`.
- **Pinned encoder:** `sentence-transformers/all-MiniLM-L6-v2`, upstream snapshot
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Every local model file is hashed.
  The benchmark aborts if semantic retrieval degrades to BM25 with a warning.
  Seven empty hybrid requests are labeled BM25 because the service deliberately
  skips dense retrieval when no lexical candidates survive; they are counted,
  not dropped from the negative-query results.

## Latency protocol and limits

Both methods run serially on CPU, in shuffled interleaved order, for three warm
repeats of all 48 queries: 144 timing samples per method. Timings include the
retrieval service call, excluding HTTP, generation and initial index/model setup.
This was a Windows development laptop, not an isolated load-test environment;
other local development work can affect timing. The p95 is descriptive of this
run, not a throughput/SLA guarantee. Requests are not concurrent.

Fresh lexical indexing took 34.7 seconds, model loading 54.0 seconds and fresh
dense indexing plus cache write 237.8 seconds. PDF extraction for annotation
validation occurs before those measurements and is explicitly excluded. These
are setup components, not a complete first-request latency measurement.

## Weaknesses found

1. **Encoder truncation:** 1,205/5,474 chunks (22.0%) exceed MiniLM's 256-token
   input limit. The existing character-based chunker can leave trailing content
   outside the semantic representation. BM25 still sees the whole chunk. The
   benchmark records this; it does not silently change chunking after scoring.
2. **Unrelated-query false positive:** the GPU/tensor-core question (`n04`) returns
   a legal passage in both methods. Keyword overlap and the dense filter do not
   establish relevance or justify an answer. Empty-result behavior is imperfect.
3. **Remaining retrieval misses:** hybrid misses eight selected passages in its
   top ten, including both variants of the abetment topic. Some returned passages
   discuss the correct topic but do not contain the annotated rule. Inspect the
   raw results before interpreting every miss as completely irrelevant retrieval.
4. **Limited coverage:** this set focuses on individual provisions and short
   evidence lookups, excluding multihop reasoning, adversarial users, OCR corpora,
   real legal professionals and production traffic. It cannot establish current
   law, answer correctness, legal applicability or hallucination rates.

Future ranking/chunking changes should use a separate development set, with a new
held-out benchmark for final claims. The published v1 set is now visible and must
not be represented as untouched after tuning against it.

## Reproduce

Install the backend environment described in the root README. Download the
immutable MiniLM snapshot once (no API key required):

```powershell
python -c "from huggingface_hub import snapshot_download; print(snapshot_download('sentence-transformers/all-MiniLM-L6-v2', revision='1110a243fdf4706b3f48f1d95db1a4f5529b4d41', local_dir='models/minilm-eval', allow_patterns=['*.json','*.safetensors','vocab.txt','1_Pooling/*']))"
cd backend
python scripts/evaluate_retrieval.py --model-path ../models/minilm-eval --model-revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 --output evaluation/retrieval/results-rerun.json
```

The evaluator needs an existing local model, disables Hub network access, clears
the provider key for its process and calls no generation API. It checks the source
hash and every annotated passage before ranking. It builds an isolated temporary
embedding cache and refuses to overwrite existing results. For the recorded
environment versions, see `results.json` rather than assuming a newer install
will reproduce timings or floating-point tie ordering exactly.

## Reliability checks

From `backend`:

```powershell
$env:HF_HUB_OFFLINE = '1'
$env:OPENAI_API_KEY = ''
python -m unittest discover -s tests -v
```

**30 tests passed** in the recorded local run. The suite now exercises the current
assistant instead of removed LangChain helper functions. CI runs these tests.
Failure-injection tests use local PDF fixtures and mocked provider responses;
they neither spend API credits nor demonstrate provider uptime.

Covered behaviors include missing keys, explicitly requested excerpts, missing
or uncertain classification, empty retrieval, current-law source guards,
provider timeouts and cooldown expiry, authentication/quota cooldowns, invalid
JSON/citations/incomplete responses, insufficient evidence, request-scoped
history, missing embeddings and BM25 fallback, cache corruption, PDF changes,
missing/scanned PDFs, stale source links and HTTP validation/error responses.
Citation-ID validation establishes membership in the retrieved set, not semantic
entailment of a generated claim.

## Defensible portfolio wording

> Built a passage-grounded retrieval benchmark with 40 questions across 20 IPC
> topics; measured hybrid Hit@5 of 72.5% versus 65.0% for BM25, and added offline
> failure-injection tests for retrieval fallback, provider errors and citations.

Keep the benchmark size and qualification when using these results. Do not turn
72.5% passage retrieval into an AI-answer accuracy or production reliability claim.
