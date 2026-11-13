# Token-bounded retrieval and local answer-support evaluation

This follow-up separates retrieval quality, generated-claim support, and answer
completeness. Questions, reference answers, and claim reviews are **agent-authored**;
there is **no independent human or legal review**. The source is the bundled
historical IPC book, not a verified current-law collection.

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
