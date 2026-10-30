# Legal Lens 2.0

**A legal-document research workspace with hybrid search, page-linked evidence, a grounded assistant, and local question classification.**

Legal Lens 2.0 extends the original CSE508 group project into a runnable FastAPI + Next.js application. Users can search the bundled Indian Penal Code book, inspect original PDF passages, ask questions, and export a research brief. Search, document excerpts, and local classification can run without a paid API key. Hybrid retrieval and classification use a locally cached MiniLM model. The dashboard opens directly, without signup.

The bundled book is **historical IPC research material**. Current-law BNS/BNSS/BSA coverage and legal-applicability verification are future work.

**Retrieval evaluation pilot:** [35 labelled historical/scope questions and per-question results](evaluation/results/REPORT.md) compare application BM25 and hybrid retrieval with an evaluation-only semantic baseline. The author-created set is not representative or legally expert-reviewed; evidence-location hits are not answer-accuracy scores. [Method and reproduction](evaluation/README.md).

[Run locally](#run-locally-on-windows) · [What we built](#what-we-built) · [Architecture](ARCHITECTURE.md) · [Demo guide](docs/DEMO.md) · [Classifier model card](backend/app/data/classification/README.md)

## What we built

The 2.0 work focused on connecting retrieval, machine learning, evidence inspection, and failure handling into one usable research workflow.

| Area we worked on | Implemented work | Where to inspect it |
| --- | --- | --- |
| **Information retrieval** | A shared PDF retrieval service for search and chat; BM25 keyword ranking, cached MiniLM semantic embeddings, and reciprocal rank fusion (RRF). Section numbers remain searchable. | [Retrieval service](backend/app/services/retrieval_service.py) · [PR #4](https://github.com/YashwantRana23/CSE508_Winter2024_Project/pull/4) |
| **Document processing and provenance** | Page-aware text chunks, stable source IDs, PDF page links, a registered source catalogue, and SHA-256 version checks that reject stale links. | [Source service](backend/app/services/source_service.py) · [Source API](backend/app/routers/sources.py) |
| **Retrieval-augmented generation (RAG)** | An assistant that retrieves evidence before generation, requests structured claims, checks citation IDs, and clearly distinguishes generated answers, excerpts, and no evidence. | [Assistant service](backend/app/services/chatbot_service.py) · [PR #5](https://github.com/YashwantRana23/CSE508_Winter2024_Project/pull/5) |
| **Local NLP and classification** | Frozen MiniLM embeddings with two logistic-regression heads for question scope and intent. Acceptance thresholds allow an explicit `uncertain` result; uncertain questions use excerpts. | [Classifier service](backend/app/services/classification_service.py) · [PR #9](https://github.com/YashwantRana23/CSE508_Winter2024_Project/pull/9) |
| **ML training and evaluation** | Reviewable train/validation/test splits, an offline training script, JSON model weights, recorded confusion matrices, validation-selected thresholds, and a model card documenting weaknesses. | [Training script](backend/scripts/train_question_classifier.py) · [Dataset and evaluation](backend/app/data/classification/README.md) |
| **Frontend and research experience** | A responsive workspace with shared corpus/search controls, source inspection, visible question-understanding labels, cancellation/retry, and Markdown brief export containing evidence and classification metadata. | [Dashboard](frontend/src/app/dashboard/page.tsx) · [Assistant UI](frontend/src/components/ChatbotUI.tsx) · [PR #6](https://github.com/YashwantRana23/CSE508_Winter2024_Project/pull/6) |
| **Backend reliability and source scope** | Typed request/response schemas, keyless excerpt fallback, provider timeouts and cooldowns, missing-model fallback, historical/current-law scope checks, and SQLite feedback storage. Sources never switch silently. | [API schemas](backend/app/schemas) · [Scope policy](backend/app/services/question_policy.py) |
| **Local delivery and CI** | Direct dashboard access, a Windows startup script, locked Python dependencies, Docker Compose packaging, and GitHub checks for frontend lint/types/build, backend dependencies/imports, and Compose configuration. | [Launcher](start-local.ps1) · [CI workflow](.github/workflows/build.yml) · [PR #3](https://github.com/YashwantRana23/CSE508_Winter2024_Project/pull/3) · [PR #7](https://github.com/YashwantRana23/CSE508_Winter2024_Project/pull/7) |

The initial 2.0 feature stack was integrated through [PR #8](https://github.com/YashwantRana23/CSE508_Winter2024_Project/pull/8), followed by local classification in [PR #9](https://github.com/YashwantRana23/CSE508_Winter2024_Project/pull/9).

## How the workflow fits together

1. **Select a source and search method.** Search and chat use the same registered PDFs and retrieval service.
2. **Understand the question.** In chat, the local classifier estimates scope and intent; independent source checks identify requests requiring evidence beyond the selected corpus.
3. **Retrieve evidence.** BM25 or hybrid retrieval returns ranked excerpts with physical PDF page numbers and version-pinned source links. Requests outside the supported chat scope explicitly report skipped retrieval.
4. **Read or generate.** Excerpt mode returns original passages. Auto mode can generate an answer when a funded provider is configured and classification/source checks permit it; failures or uncertainty use excerpts.
5. **Inspect and export.** Open the source page and download a brief retaining the question, answer mode, classification, excerpts, and citations.

## Technology stack

| Layer | Technologies |
| --- | --- |
| Frontend | Next.js 16, React 19, TypeScript, Tailwind CSS 4 |
| Backend | Python 3.12, FastAPI, Pydantic, Uvicorn |
| Retrieval and PDF processing | PyMuPDF, BM25, SentenceTransformers MiniLM, NumPy, RRF |
| Local classification | Frozen MiniLM encoder + scikit-learn logistic regression; JSON coefficient artifacts |
| Optional generation | OpenAI SDK with JSON responses and local schema/citation validation |
| Persistence and delivery | SQLite/SQLAlchemy, local embedding cache, PowerShell, Docker Compose, GitHub Actions |

## Run locally on Windows

Requirements: **Git, Python 3.12, Node.js 22, and npm**. Run these commands in PowerShell. Installation and the one-time model download require internet access; local search, excerpts, and classification do not require a provider API key.

### First-time setup

Skip the first two commands if you are already inside the cloned repository.

```powershell
git clone https://github.com/YashwantRana23/CSE508_Winter2024_Project.git
cd CSE508_Winter2024_Project

# Create a Python 3.12 environment and install the locked backend dependencies.
py -3.12 -m venv backend/.venv
.\backend\.venv\Scripts\python.exe -m pip install -r backend/requirements-lock.txt

# Keep an existing local configuration if one is already present.
if (!(Test-Path backend/.env)) { Copy-Item backend/.env.example backend/.env }

# Install the frontend dependencies.
Push-Location frontend
npm ci
Pop-Location

# Download MiniLM once to enable both hybrid retrieval and local classification.
.\backend\.venv\Scripts\python.exe -c "from dotenv import load_dotenv; load_dotenv('backend/.env'); from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Start both servers.
.\start-local.ps1
```

If the Windows `py` launcher is unavailable, use `python -m venv backend/.venv` after confirming `python --version` reports Python 3.12. The download command reads `backend/.env` first so any configured `HF_HOME` cache location matches the app. Model training is **not required** to run the app: trained classifier coefficients are included. If you skip the MiniLM download, BM25 and excerpts remain available, while classification reports unavailable and hybrid retrieval falls back to BM25.

### Start again after setup

From the repository root:

```powershell
.\start-local.ps1
```

For the existing checkout used during development:

```powershell
Set-Location 'E:\2026\AllProjects\CSE508_Winter2024_Project'
.\start-local.ps1
```

| Service | Local URL |
| --- | --- |
| Research dashboard | [http://localhost:3000/dashboard](http://localhost:3000/dashboard) |
| Interactive API documentation | [http://localhost:8000/docs](http://localhost:8000/docs) |
| Backend health | [http://localhost:8000/health](http://localhost:8000/health) |

The launcher starts background processes and writes `backend.stdout.log`, `backend.stderr.log`, `frontend.stdout.log`, and `frontend.stderr.log` in the project root. It leaves occupied ports alone; it does not restart an existing backend after code/configuration changes. First-use model loading and PDF extraction can take longer than later requests.

### Run in two terminals instead

Use this option to see server logs directly and stop each server with **Ctrl+C**, or if PowerShell prevents running the startup script. Start each terminal at the repository root, with ports 8000 and 3000 free.

**Terminal 1 — backend:**

```powershell
cd backend
.\.venv\Scripts\python.exe run.py
```

**Terminal 2 — frontend:**

```powershell
cd frontend
npm run dev -- --hostname 127.0.0.1
```

### Try the completed features

Select the **IPC corpus** and **Document excerpts** answer mode for a demo without generation credits.

| Action / question | What to inspect |
| --- | --- |
| Search: “Section 302 punishment for murder” | Ranked passages, actual retrieval method, PDF pages, and original source links. |
| Ask: “Today I am studying the old IPC. Explain Section 302.” | Historical research remains usable despite incidental “today”; uncertain labels still allow excerpts. |
| Ask: “Compare Section 299 and Section 300 of the historical IPC.” | Historical scope can coexist with comparison intent. |
| Ask: “How does IPC Section 302 compare with BNS?” | The source limitation and explicitly skipped retrieval. |
| Download a brief from the assistant | The original question, output mode, question understanding, and returned evidence. |

Check the API directly from PowerShell while the backend is running:

```powershell
Invoke-RestMethod -Uri 'http://localhost:8000/health'

$body = @{ question = 'Compare Section 299 and Section 300 of the historical IPC.' } | ConvertTo-Json
Invoke-RestMethod -Uri 'http://localhost:8000/questions/classify' -Method Post -ContentType 'application/json' -Body $body
```

### Optional AI-generated answers

Set `OPENAI_API_KEY` and a model available to your API project in `OPENAI_MODEL` inside **backend/.env**, restart the backend, then choose **Auto** in the assistant. A funded provider account and model access are required for generation. Keep credentials server-side; do not commit them or place them in frontend variables.

Missing credentials, exhausted credits, provider errors, or uncertain classification leave document retrieval usable through excerpts. Quota/authentication failures pause provider attempts for five minutes; transient failures pause them for one minute. The returned mode always identifies what happened.

## Local classifier: training and measured scope

The classifier reuses `sentence-transformers/all-MiniLM-L6-v2`; only the logistic-regression heads are trained. It estimates:

- **Scope:** historical, current, cross-era comparison, or uncertain.
- **Intent:** section lookup, explanation, comparison, unrelated, or uncertain.

The dataset contains **164 training, 56 validation, and 56 held-out questions**, authored and reviewed by coding agents. It is an illustrative English dataset, without human legal-expert or representative real-user validation. Thresholds were chosen on validation only; the held-out set was not used for tuning.

| Head | Held-out accuracy before abstention | Accuracy on accepted labels | Coverage after abstention |
| --- | --- | --- | --- |
| Scope | 42/48 (87.50%) | 39/42 (92.86%) | 42/48 (87.50%) |
| Intent | 53/56 (94.64%) | 44/46 (95.65%) | 46/56 (82.14%) |

Scope evaluation excludes the eight unrelated questions. These figures measure question labels, **not legal-answer accuracy**. Cross-era scope recall was only **3/8**, so independent source checks remain necessary. See the [model card](backend/app/data/classification/README.md) and [evaluation report](backend/app/data/classification/evaluation.json) for confusion matrices, thresholds, and limitations. This is a local task-specific classifier, not a Jev implementation.

To reproduce training separately from normal app startup, with MiniLM already cached:

```powershell
# From the repository root; writes fresh artifacts to ignored local runtime storage.
.\backend\.venv\Scripts\python.exe backend/scripts/train_question_classifier.py --output-dir backend/instance/classifier-retraining
```

## Repository map

```text
backend/app/                     FastAPI routes, schemas, retrieval, assistant and classifier
backend/app/data/classification/ Authored datasets, trained JSON weights and evaluation report
backend/scripts/                 Offline classifier training
frontend/src/                    Next.js dashboard, source viewer, assistant and brief export
LegalLaw/Chatbot/                 Bundled historical IPC PDF and legacy chatbot material
docs/                            Architecture support, demo, deployment and verification notes
.github/workflows/               Build-only CI
start-local.ps1                  Windows launcher
```

## Configuration and APIs

Process environment variables override `backend/.env`. Frontend configuration belongs in `frontend/.env.local`. Environment files, runtime caches, databases, and logs are Git-ignored.

| Setting | Purpose |
| --- | --- |
| `OPENAI_API_KEY`, `OPENAI_MODEL` | Optional server-side generation configuration; the existing model default is `gpt-3.5-turbo`. |
| `EMBEDDING_MODEL` | Defaults to `sentence-transformers/all-MiniLM-L6-v2`. Changing it requires compatible retrained classifier heads. |
| `CHATBOT_IPC_PDF` | Optional replacement PDF; replacements are marked unverified. |
| `CHATBOT_MURDER_PDF`, `CHATBOT_CHILD_PDF`, `CHATBOT_MATERNITY_PDF` | Optional domain PDFs. Missing sources remain unavailable. |
| `DATABASE_URL` | SQLite feedback database by default, under `backend/instance`. |
| `DATA_PATH` | Optional CSV for legacy `/search/bm25`; separate from the shared PDF registry. |
| `NEXT_PUBLIC_API_URL` | Browser-accessible backend URL, default `http://localhost:8000`; baked into production frontend builds. |
| `HF_HOME` | Optional local Hugging Face model-cache directory. |

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Application health and configuration status. |
| `GET /sources` | Registered sources, availability, and historical/unverified status. |
| `GET /sources/{document_id}/pdf` | Registered PDF with version-pinned page links. |
| `POST /search/retrieve` | Shared BM25/hybrid PDF retrieval. |
| `POST /questions/classify` | Local scope/intent classification. |
| `GET /chatbot/status` | Per-domain availability and provider configuration. |
| `POST /chatbot/{domain}` | Evidence-backed chat, classification metadata, and actual answer mode. |
| `POST /feedback/submit` | Validated feedback storage. |

Full request/response schemas are available at `/docs`. Chat history stays in the browser page; changing the corpus or refreshing clears it. Search-method changes preserve the conversation.

## Verification and packaging

Frontend lint, TypeScript and production build, backend dependency/syntax/import checks, and focused API/browser demos were completed for the implemented release and classifier. GitHub frontend/backend checks passed on [classifier PR #9](https://github.com/YashwantRana23/CSE508_Winter2024_Project/pull/9). The work did not run the historical regression suite.

To run build checks yourself from the repository root:

```powershell
.\backend\.venv\Scripts\python.exe -m pip check
.\backend\.venv\Scripts\python.exe -m compileall -q backend/app backend/scripts backend/run.py backend/prepare_corpus.py
Push-Location frontend
npm run lint
npx tsc --noEmit --incremental false
npm run build
Pop-Location
```

Docker Compose packaging is included. The commands below require a working Docker engine using Linux containers. Stop the local servers first so ports 3000 and 8000 are free:

```powershell
docker compose config --quiet
docker compose up --build
```

Compose configuration was verified; container build/run remains unverified in the development environment. The image does not bundle MiniLM weights. See [deployment notes](docs/DEPLOYMENT.md) for its model cache, optional provider configuration, volumes, and runtime limits.

## Current boundaries and next steps

- Only the historical IPC book is bundled: **1,871 physical PDF pages**. The older preparation produced 1,844 text-bearing page records from this book, rather than 1,844 separate judgments.
- Citation-ID validation checks source membership; it does not prove legal correctness or that every claim is supported. Scope rules and classification can make mistakes.
- The retained graph is TF-IDF passage similarity. Legal entity/relation extraction is future work.
- Verified current-law ingestion, OCR, multilingual evaluation, cross-encoder reranking, private persistent workspaces, and public production deployment remain future work.

Further detail: [architecture](ARCHITECTURE.md), [demo guide](docs/DEMO.md), [classification design](docs/CLASSIFICATION.md), [release scope](docs/RELEASE.md), and [verification record](docs/VERIFICATION.md).

## Contributors and project history

The original project was developed as **CSE508 Winter 2024, Group 44**, with contributions from:

- Harsh Patel — MT23056
- Sahil More — MT23079
- Sarthak Pol — MT23082
- Vinayak Katoch — MT23105
- Yashwant Rana — MT23107

The original notebooks, Flask/Streamlit components, and Llama experiments remain in the repository. The `backend/` and `frontend/` directories contain the later FastAPI/Next.js application; [the refactor commit](https://github.com/YashwantRana23/CSE508_Winter2024_Project/commit/d681f4b01e1ca9872389c9a47ae59097a888d7a0) is attributed to `yashwant938`. The feature PRs above document the 2.0 additions while preserving the original group attribution.

Original project references: [legal-document structuring corpus](https://arxiv.org/abs/2201.13125), [legal judgment prediction](https://arxiv.org/abs/2112.06370), [Indian court judgment NER](https://arxiv.org/abs/2211.03442), and [LegalEval](https://arxiv.org/abs/2304.09548).
