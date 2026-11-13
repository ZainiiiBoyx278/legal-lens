"""Shared page-aware BM25 + optional local dense retrieval for search and chat."""
from dataclasses import dataclass
import hashlib
import logging
import os
from pathlib import Path
import re
import tempfile
import unicodedata
from threading import RLock
from time import monotonic, perf_counter

import numpy as np
from rank_bm25 import BM25Okapi
from app.config import BASE_DIR, get_settings
from app.services import source_service, passage_chunking

logger = logging.getLogger(__name__)
CACHE_VERSION = passage_chunking.POLICY_VERSION + "-nfkc-coverage-v2"
# A retrieval filter, not a calibrated probability of relevance or correctness.
MIN_DENSE_SIMILARITY = 0.35
MIN_QUERY_COVERAGE = 0.35
RRF_RANK_CONSTANT = 20
STOP_WORDS = frozenset("a an the and or is are was were be been being of to in on for with by at from this that these those it its as what which who whom how when where why can could would should may might must please explain tell me about under does do did give has have had i my you your we our they their he his she her them than then also according historical ipc book passage text section sec".split())
_lock = RLock()
_model_lock = RLock()
_corpora = {}
_models = {}
_dense_failures = {}


@dataclass
class Corpus:
    document: source_service.SourceDocument
    fingerprint: str
    pdf_hash: str
    texts: list[str]
    pages: list[int]
    ids: list[str]
    tokens: list[list[str]]
    bm25: BM25Okapi
    vectors: np.ndarray | None = None


def tokenize(text: str) -> list[str]:
    # Keep numbers and suffixes (302, 304B), unlike stemming preprocessors.
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return [token for token in re.findall(r"[^\W_]+", normalized, flags=re.UNICODE) if token not in STOP_WORDS]


def _chunks(text: str, profile=None):
    selected = profile or passage_chunking.profile(get_settings().EMBEDDING_MODEL)
    yield from (chunk for chunk in passage_chunking.chunks(text, selected) if tokenize(chunk))


def _get_corpus(domain: str) -> Corpus:
    document = source_service.for_domain(domain)
    if not document.path.is_file():
        raise ValueError("No PDF is configured for this domain.")
    _, pdf_hash = source_service.metadata(document)
    profile = passage_chunking.profile(get_settings().EMBEDDING_MODEL)
    key = (document.document_id, pdf_hash, CACHE_VERSION, profile.fingerprint)
    with _lock:
        if key in _corpora:
            return _corpora[key]
        import pymupdf
        texts, pages, ids = [], [], []
        with pymupdf.open(document.path) as pdf:
            for page_number, page in enumerate(pdf, start=1):
                for chunk_number, text in enumerate(_chunks(page.get_text(sort=True), profile), start=1):
                    texts.append(text)
                    pages.append(page_number)
                    ids.append(f"{document.document_id}:{pdf_hash[:12]}:{profile.fingerprint[:10]}:p{page_number}:c{chunk_number}")
        if not texts:
            raise ValueError("The configured PDF has no extractable text. OCR is required for scanned documents.")
        tokens = [tokenize(text) for text in texts]
        fingerprint = hashlib.sha256((pdf_hash + CACHE_VERSION + profile.fingerprint).encode()).hexdigest()
        corpus = Corpus(document, fingerprint, pdf_hash, texts, pages, ids, tokens, BM25Okapi(tokens))
        # Discard previous versions of this source from the in-process cache.
        for stale in [item for item in _corpora if item[0] == document.document_id]:
            del _corpora[stale]
        _corpora[key] = corpus
        return corpus


def _model():
    model_name = get_settings().EMBEDDING_MODEL
    with _model_lock:
        if model_name not in _models:
            import torch
            from sentence_transformers import SentenceTransformer
            torch.set_num_threads(min(4, os.cpu_count() or 1))
            # Requests never download models. An uncached model falls back to BM25.
            _models[model_name] = SentenceTransformer(model_name, device="cpu", local_files_only=True)
        return _models[model_name]


class EmbeddingInputTooLongError(ValueError):
    """Raised when a caller requires embeddings without silent truncation."""


def encode_texts(texts: list[str], *, reject_truncation: bool = False) -> np.ndarray:
    """Reuse the cached CPU encoder without downloading or persisting input text."""
    with _model_lock:
        model = _model()
        if reject_truncation:
            tokens = model.tokenizer(texts, truncation=False, add_special_tokens=True)
            if any(len(ids) > model.max_seq_length for ids in tokens["input_ids"]):
                raise EmbeddingInputTooLongError("Text exceeds the local encoder token limit.")
        return np.asarray(model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)



def _vectors(corpus: Corpus) -> np.ndarray:
    if corpus.vectors is not None:
        return corpus.vectors
    model_name = get_settings().EMBEDDING_MODEL
    with _model_lock:
        if corpus.vectors is not None:
            return corpus.vectors
        model = _model()
        # A lexical fallback index must never silently become a truncated dense
        # index. This also protects custom encoders with shorter token limits.
        lengths = model.tokenizer(corpus.texts, truncation=False, add_special_tokens=True)["input_ids"]
        if any(len(ids) > model.max_seq_length for ids in lengths):
            raise EmbeddingInputTooLongError("Source chunks exceed the local encoder limit; serving BM25 only.")
        revision = getattr(model[0].auto_model.config, "_commit_hash", None)
        identity = f"{corpus.fingerprint}:{model_name}:{revision}:{model.max_seq_length}"
        fingerprint = hashlib.sha256(identity.encode()).hexdigest()
        cache_dir = BASE_DIR / "instance" / "embeddings"
        cache_file = cache_dir / f"retrieval-{fingerprint}.npz"
        if cache_file.is_file():
            try:
                with np.load(cache_file, allow_pickle=False) as saved:
                    vectors = saved["vectors"]
                    valid = (
                        vectors.ndim == 2 and vectors.shape[0] == len(corpus.texts)
                        and np.isfinite(vectors).all()
                        and saved["ids"].tolist() == corpus.ids
                        and saved["pages"].tolist() == corpus.pages
                        and saved["texts"].tolist() == corpus.texts
                    )
                    if valid:
                        corpus.vectors = vectors.astype(np.float32, copy=False)
                        return corpus.vectors
            except (OSError, ValueError, KeyError):
                pass
        corpus.vectors = np.asarray(model.encode(corpus.texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)
        temporary_path = None
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=cache_dir, suffix=".npz", delete=False) as temporary:
                temporary_path = Path(temporary.name)
                np.savez_compressed(temporary, vectors=corpus.vectors, texts=np.asarray(corpus.texts), pages=np.asarray(corpus.pages), ids=np.asarray(corpus.ids))
            os.replace(temporary_path, cache_file)
        except OSError:
            logger.warning("Embedding cache could not be saved; retrieval remains available in memory.")
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return corpus.vectors


def retrieve(query: str, domain: str = "ipc", top_k: int = 6, method: str = "hybrid") -> dict:
    started = perf_counter()
    corpus = _get_corpus(domain)
    query_tokens = tokenize(query)
    warning = None
    used_method = "bm25"
    scores = corpus.bm25.get_scores(query_tokens) if query_tokens else np.zeros(len(corpus.texts))
    lexical = [int(i) for i in np.argsort(-scores, kind="stable")[:60] if scores[i] > 0]
    # Explicit provision numbers must survive ranking rather than being stemmed away.
    section_match = re.search(r"\b(?:section|sec|s\.)\s*(\d+[a-z]*)\b", query, flags=re.I)
    if section_match:
        provision = section_match.group(1).casefold()
        lexical = [i for i in lexical if provision in corpus.tokens[i]]
    fusion_lexical = lexical
    if not section_match:
        required_overlap = min(2, len(set(query_tokens)))
        unique = set(query_tokens)
        lexical = [i for i in lexical if len(unique.intersection(corpus.tokens[i])) >= required_overlap]
        # Out-of-vocabulary terms still contribute to the denominator. A pair
        # of incidental common words should not dominate hybrid fusion. Keep
        # the broader lexical list for standalone BM25 and model-failure fallback.
        unseen_weight = float(np.log(1 + len(corpus.texts)))
        weights = {token: max(0.1, corpus.bm25.idf.get(token, unseen_weight)) for token in unique}
        total_weight = sum(weights.values()) or 1.0
        fusion_lexical = [i for i in lexical if sum(weights[token] for token in unique.intersection(corpus.tokens[i])) / total_weight >= MIN_QUERY_COVERAGE]
    ranked = lexical
    combined = {i: float(scores[i]) for i in lexical}
    # A paraphrase can lack the lexical overlap needed for BM25 but still have
    # strong semantic evidence. All-OOV/stop-word queries still skip the encoder.
    if method == "hybrid" and query_tokens and np.any(scores > 0):
        failure_key = (corpus.fingerprint, get_settings().EMBEDDING_MODEL)
        with _model_lock:
            failed_until = _dense_failures.get(failure_key, 0)
        cooling_down = failed_until > monotonic()
        try:
            if cooling_down:
                raise RuntimeError("Local semantic retrieval is cooling down after a failure.")
            vectors = _vectors(corpus)
            with _model_lock:
                query_vector = encode_texts([query], reject_truncation=True)[0]
            dense_scores = vectors @ query_vector
            dense = [int(i) for i in np.argsort(-dense_scores, kind="stable")[:60] if dense_scores[i] >= MIN_DENSE_SIMILARITY]
            if section_match:
                dense = [i for i in dense if provision in corpus.tokens[i]]
            combined = {}
            for ranking in (fusion_lexical, dense):
                for position, index in enumerate(ranking, start=1):
                    combined[index] = combined.get(index, 0.0) + 1.0 / (RRF_RANK_CONSTANT + position)
            ranked = sorted(combined, key=lambda i: (-combined[i], i))
            used_method = "hybrid"
        except EmbeddingInputTooLongError:
            # An oversized query is an input constraint, not an unavailable
            # model: it must not disable semantic retrieval for other requests.
            warning = "Text exceeds the local encoder token limit; showing BM25 keyword matches without truncated embeddings."
        except Exception:
            if not cooling_down:
                with _model_lock:
                    _dense_failures[failure_key] = monotonic() + 60
            warning = "Semantic retrieval is unavailable locally; showing BM25 keyword matches. Check the cached encoder and tokenizer to enable hybrid retrieval."
            logger.warning("Local semantic retrieval unavailable; serving lexical results.")
    results = []
    for rank, index in enumerate(ranked[:max(1, min(top_k, 20))], start=1):
        results.append({
            "id": corpus.ids[index], "document_id": corpus.document.document_id,
            "title": corpus.document.title, "page": corpus.pages[index],
            "excerpt": corpus.texts[index],
            "url": f"/sources/{corpus.document.document_id}/pdf?version={corpus.pdf_hash}#page={corpus.pages[index]}",
            "corpus_status": corpus.document.corpus_status,
            "score": combined[index], "rank": rank,
        })
    return {
        "query": query, "results": results,
        "retrieval": {"method": used_method, "elapsed_ms": round((perf_counter() - started) * 1000, 2), "warning": warning},
        "corpus_status": corpus.document.corpus_status,
    }
