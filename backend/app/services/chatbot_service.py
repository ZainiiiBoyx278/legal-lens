"""Evidence-first chat over the same retrieval service used by search.

Retrieval needs no provider credentials. Generated claims carry validated source
IDs; validation establishes citation membership, not legal truth or entailment.
"""
import hashlib
from threading import RLock
from time import monotonic
from uuid import uuid4

from pydantic import ValidationError
from app.services.evidence_contract import EvidenceClaim, EvidenceAnswer, build_messages, parse_answer
from app.config import get_settings
from app.services import classification_service, retrieval_service, source_service
from app.services.question_policy import explicit_historical_reference, source_scope_reason

PROVIDER_TIMEOUT_SECONDS = 30.0
TRANSIENT_COOLDOWN_SECONDS = 60
CONFIGURATION_COOLDOWN_SECONDS = 300
_provider_lock = RLock()
_provider_circuits: dict[str, tuple[float, str]] = {}


def _provider_id(settings) -> str:
    return hashlib.sha256(((settings.OPENAI_API_KEY or "") + ":" + settings.OPENAI_MODEL).encode()).hexdigest()


def _cooldown_reason(settings) -> str | None:
    key = _provider_id(settings)
    with _provider_lock:
        expires, reason = _provider_circuits.get(key, (0, ""))
        if expires > monotonic():
            return reason
        _provider_circuits.pop(key, None)
    return None


def _pause_provider(settings, reason: str, seconds: int) -> str:
    with _provider_lock:
        _provider_circuits[_provider_id(settings)] = (monotonic() + seconds, reason)
    return reason


def domain_status(domain: str) -> dict:
    settings = get_settings()
    document = source_service.for_domain(domain)
    available = document.path.is_file()
    configured = bool(settings.OPENAI_API_KEY)
    reason = ""
    if not available:
        reason = "No PDF is configured for this domain."
    elif not configured:
        reason = "Document-excerpt mode is ready. No AI provider key is configured."
    else:
        reason = _cooldown_reason(settings) or ""
    return {
        "available": available,
        "reason": reason,
        "mode": "unavailable" if not available else "auto" if configured and not reason else "excerpts",
        "provider_configured": configured,
        "provider_health": "not_checked",
        "source_count": int(available),
        "corpus_status": document.corpus_status,
        "title": document.title,
    }


def _excerpt_response(sources: list[dict], reason: str) -> str:
    passages = [f"[S{source['rank']}] {source['title']} - PDF page {source['page']}\n{source['excerpt']}" for source in sources]
    return "Document-excerpt mode. These are retrieved passages, not an AI-generated answer.\n\n" + "\n\n".join(passages)


def _generate(settings, message: str, history: list[dict], sources: list[dict]) -> EvidenceAnswer:
    from openai import OpenAI
    with OpenAI(api_key=settings.OPENAI_API_KEY, timeout=PROVIDER_TIMEOUT_SECONDS, max_retries=0) as client:
        # Preserve compatibility with an existing legacy-model configuration.
        token_budget = {"max_tokens": 1600} if settings.OPENAI_MODEL.startswith("gpt-3.5") else {"max_completion_tokens": 1600}
        completion = client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=build_messages(message, history, sources),
            response_format={"type": "json_object"},
            **token_budget,
        )
    if not completion.choices or completion.choices[0].finish_reason != "stop":
        raise ValueError("The provider did not return a complete answer.")
    return parse_answer(completion.choices[0].message.content or "", sources)


def chat(domain: str, message: str, history: list[dict] | None = None, *, method: str = "hybrid", answer_mode: str = "auto") -> dict:
    settings = get_settings()
    scope_reason = source_scope_reason(message)
    classification = classification_service.classify(message)
    document = source_service.for_domain(domain)
    # The model describes a question, never the currency of its source. Explicit
    # scope rules continue to work when the local classifier is unavailable.
    labels_accepted = classification["status"] == "ready" and "uncertain" not in (classification["scope"], classification["intent"])
    scope_conflict = (
        labels_accepted and scope_reason is None
        and classification["scope"] in ("current", "comparison")
        and explicit_historical_reference(message)
    )
    if labels_accepted and scope_reason is None and not scope_conflict and classification["scope"] in ("current", "comparison"):
        scope_reason = "The local classifier identified a current-law or cross-era question; this is a routing estimate, not a legal determination."
    if scope_reason and document.corpus_status in ("historical", "unverified"):
        return {
            "response": "The selected source cannot verify current law, cross-era mappings, or legal applicability. For historical research, ask specifically about a passage or provision in the selected book. Source search remains available to inspect the text.",
            "domain": domain, "mode": "no_evidence",
            "reason": f"{scope_reason} Selected source status: {document.corpus_status}.",
            "sources": [],
            "retrieval": {"method": method, "elapsed_ms": 0.0, "warning": "Retrieval skipped: the question needs source verification beyond the selected corpus."},
            "request_id": str(uuid4()), "classification": classification,
        }
    # Search and chat use precisely the same query/index. History affects generation
    # only, preventing stale previous topics from silently changing retrieved evidence.
    retrieval = retrieval_service.retrieve(message, domain, top_k=6, method=method)
    sources = retrieval["results"]
    result = {
        "response": "", "domain": domain, "mode": "excerpts", "reason": None,
        "sources": sources, "retrieval": retrieval["retrieval"], "request_id": str(uuid4()),
        "classification": classification,
    }
    if not sources:
        result.update(mode="no_evidence", reason="No matching evidence was found in the selected PDF.", response="No matching passages were found in the selected source. Try a specific provision number or more precise legal keywords.")
        return result
    reason = None
    if scope_conflict:
        reason = "The classifier estimate conflicts with an explicit historical-source reference. Showing source excerpts without an AI-generated interpretation."
    elif answer_mode == "excerpts":
        reason = "Document excerpts were requested."
    if reason is None and (classification["status"] != "ready" or "uncertain" in (classification["scope"], classification["intent"])):
        reason = "Question classification is uncertain or unavailable. Showing source excerpts without an AI-generated interpretation."
    if reason is None and not settings.OPENAI_API_KEY:
        reason = "No AI provider key is configured. Local document retrieval remains available."
    if reason is None:
        reason = _cooldown_reason(settings)
    if reason is None:
        from openai import APIError, AuthenticationError, BadRequestError, PermissionDeniedError, RateLimitError
        try:
            answer = _generate(settings, message, history or [], sources)
            if not answer.sufficient_evidence:
                result.update(mode="no_evidence", reason="The retrieved passages do not sufficiently support an answer.", response="The retrieved passages are insufficient to answer this question. You can inspect them below or try a narrower question.")
                return result
            labels = {source["id"]: f"S{source['rank']}" for source in sources}
            result.update(
                mode="generated",
                response="\n\n".join(claim.text.strip() + " [" + ", ".join(dict.fromkeys(labels[source_id] for source_id in claim.citation_ids)) + "]" for claim in answer.claims),
            )
            return result
        except (AuthenticationError, PermissionDeniedError):
            reason = _pause_provider(settings, "The AI provider rejected this configuration. Showing document excerpts; generation will retry after five minutes.", CONFIGURATION_COOLDOWN_SECONDS)
        except RateLimitError as error:
            quota = error.code in ("insufficient_quota", "credit_balance_exhausted")
            reason = _pause_provider(settings, "AI API credits are exhausted. Showing document excerpts; generation will retry after five minutes." if quota else "The AI provider is rate-limited. Showing document excerpts; generation will retry after one minute.", CONFIGURATION_COOLDOWN_SECONDS if quota else TRANSIENT_COOLDOWN_SECONDS)
        except BadRequestError:
            reason = _pause_provider(settings, "The configured model could not process this request. Showing document excerpts; generation will retry after five minutes.", CONFIGURATION_COOLDOWN_SECONDS)
        except APIError:
            reason = _pause_provider(settings, "The AI provider is temporarily unavailable. Showing document excerpts; generation will retry after one minute.", TRANSIENT_COOLDOWN_SECONDS)
        except (ValueError, ValidationError):
            reason = "The generated answer failed evidence or citation validation. Showing retrieved excerpts instead."
    result["reason"] = reason
    result["response"] = _excerpt_response(sources, reason or "")
    return result
