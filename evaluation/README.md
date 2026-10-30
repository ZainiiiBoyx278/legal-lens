# Historical IPC evaluation pilot

This folder measures whether the existing Legal Lens search can locate selected
passages in its bundled **historical** Indian Penal Code book. It also tests
whether the real chat service abstains on examples requiring current-law,
cross-era, or dated-applicability evidence. It does not evaluate current law or
replace legal review.

## Read the results

Open [`results/REPORT.md`](results/REPORT.md) for the full comparison and every
question. [`results/results.json`](results/results.json) contains the individual
method outcomes and actual chat response modes. The report shows successes,
misses, unsupported cases, actual methods, and warm response times.

The 35-question set contains 21 explicit section lookups, 6 paraphrases, and 8
unsupported questions. Questions were authored for this pilot and their target
section markers and physical page numbers were checked against the PDF. They are
**not** a representative sample, independently adjudicated, or reviewed by a
legal professional. Each question records only selected relevant pages; other
passages can be relevant, so a miss need not be a wrong answer. Numbers are
development observations, not general quality claims.

## Method and boundaries

- All comparisons use the same page-aware corpus chunks and source hash.
- **BM25** and **hybrid** call the existing application retrieval service.
- **Semantic** is an evaluation-only dense baseline using the same MiniLM
  encoder and similarity cutoff as the hybrid service. It is not a new product
  endpoint. It applies the app's provision-number filter when applicable.
- The benchmark checks for the expected section marker *and* one labelled page
  in the top five results. It reports Hit@5 and reciprocal rank, not legal-answer
  correctness. Hybrid results identify and count BM25 fallbacks and exclude them
  from the hybrid metric denominator.
- Unsupported cases exercise the app's real `chat()` scope path in excerpt
  mode. The report records failures as well as passes.
- The actual chat service is also run in excerpt mode on each supported question.
  Its returned mode, source pages, and evidence marker are shown. The chat
  service retrieves up to six results; those hits are not comparable to Hit@5.
  A marker hit does not establish that the passage answers the question.
- No generative AI provider is called. No generated-answer accuracy, citation
  entailment, or legal correctness score is claimed. Answer-quality review
  requires independently annotated reference answers, unsupported cases and
  claim-level evidence checks, ideally with qualified legal reviewers.
- Timings are warm in-process wall-clock measurements on one machine. They
  exclude PDF indexing and model initialization, can vary substantially, and
  the small-sample p95 is illustrative. No production performance claim is made.

## Run

Install the project's locked backend dependencies, and download/cache the
configured SentenceTransformers model once using the project's setup guide.
Then from the repository root:

```sh
python evaluation/run.py
python -m unittest discover -s evaluation -p 'test_evaluation.py' -v
```

The runner forces `OPENAI_API_KEY` empty and sets Hugging Face offline mode by
default. It does not transmit questions to an AI provider. If the model is not
available, semantic is marked unavailable and any hybrid-to-BM25 fallback is
reported rather than counted as hybrid. It aborts if the PDF hash changes or a
labelled section marker is absent. Results go to `evaluation/results/` by
default; pass `--out-dir` to choose a different local folder.

## Extending the set

Recruit independent reviewers to label relevant page ranges and answer support
without seeing system rankings. Add diverse real-user research questions only
with appropriate consent, including multiple relevant passages, difficult
paraphrases, unrelated questions, and incomplete or conflicting evidence. Keep a
separate held-out set and record reviewer disagreements. Have a legal expert
review any legal-answer labels. Re-run all methods against the same frozen PDF
and encoder, and publish both failure cases and metrics with denominators.

## Manual answer review

Create a blank worksheet **outside the repository** and have an independent
reviewer inspect actual outputs alongside the cited pages. Keep responses and
completed worksheets private if they contain sensitive or licensed material:

```sh
python evaluation/review_answers.py --template /path/to/private-review.csv
python evaluation/review_answers.py --score /path/to/private-review.csv
```

For each actual response, record its mode and a private response reference, the
reviewer's identity, and `yes`, `no`, `uncertain`, or `not_applicable` for
correctness, citation support, and appropriate abstention. Leave unreviewed
fields blank. The scorer reports counts, including uncertainty and unreviewed
rows, and does not manufacture legal correctness or a percentage. Human review
has not yet been performed for the included pilot.
