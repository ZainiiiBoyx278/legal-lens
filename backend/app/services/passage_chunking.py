"""Source-exact chunks bounded by a locally cached encoder tokenizer.

Loading this module never loads model weights or makes network requests. If the
tokenizer is absent, lexical retrieval retains its character-chunk fallback.
"""
from dataclasses import dataclass
import hashlib
import re
from threading import RLock
from time import monotonic

TOKEN_BUDGET = 224  # Includes special tokens; validated against the actual encoder.
TOKEN_OVERLAP = 32
POLICY_VERSION = "source-offset-token224-overlap32-v2"
_lock = RLock()
_profiles = {}
_failures = {}


@dataclass(frozen=True)
class ChunkProfile:
    tokenizer: object | None
    fingerprint: str


def profile(model_name: str) -> ChunkProfile:
    with _lock:
        if model_name in _profiles:
            return _profiles[model_name]
        if _failures.get(model_name, 0) <= monotonic():
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, local_files_only=True)
                if not tokenizer.is_fast:
                    raise ValueError("A fast tokenizer is required for source offsets.")
                identity = model_name + POLICY_VERSION + tokenizer.backend_tokenizer.to_str()
                result = ChunkProfile(tokenizer, hashlib.sha256(identity.encode()).hexdigest())
                _profiles[model_name] = result
                _failures.pop(model_name, None)
                return result
            except (OSError, ValueError, ImportError):
                _failures[model_name] = monotonic() + 60
        return ChunkProfile(None, hashlib.sha256((model_name + "character1100-overlap140-v2").encode()).hexdigest())


def character_chunks(text: str):
    start = 0
    while start < len(text):
        end = min(start + 1100, len(text))
        if end < len(text):
            boundary = max(text.rfind("\n", start + 650, end), text.rfind(" ", start + 650, end))
            if boundary > start:
                end = boundary
        if text[start:end].strip():
            yield text[start:end].strip()
        if end == len(text):
            break
        start = max(start + 1, end - 140)


def token_chunks(text: str, tokenizer, budget: int = TOKEN_BUDGET, overlap: int = TOKEN_OVERLAP):
    """Slice original text with tokenizer offsets; never decode/rewrite evidence.

    Prefer paragraph/sentence ends in the latter part of a window. Re-tokenize
    each slice because starting midway through a word can change its encoding.
    """
    capacity = min(budget, tokenizer.model_max_length) - tokenizer.num_special_tokens_to_add(pair=False)
    if capacity <= overlap or overlap < 0:
        raise ValueError("The token budget must exceed overlap and special tokens.")
    offsets = tokenizer(text, add_special_tokens=False, truncation=False, return_offsets_mapping=True)["offset_mapping"]
    offsets = [(a, b) for a, b in offsets if b > a]
    start = 0
    while start < len(offsets):
        end = min(start + capacity, len(offsets))
        if end < len(offsets):
            lower = offsets[start + max(1, int(capacity * 0.6))][0]
            upper = offsets[end - 1][1]
            window = text[lower:upper]
            boundaries = list(re.finditer(r"\n\s*\n", window)) or list(re.finditer(r"(?<=[.!?])\s+", window))
            if boundaries:
                boundary = lower + boundaries[-1].end()
                while end > start + 1 and offsets[end - 1][1] > boundary:
                    end -= 1
        while end > start:
            chunk = text[offsets[start][0]:offsets[end - 1][1]].strip()
            length = len(tokenizer(chunk, add_special_tokens=True, truncation=False)["input_ids"])
            if length <= min(budget, tokenizer.model_max_length):
                break
            end -= 1
        if end <= start:
            raise ValueError("A source token cannot fit in the encoder budget.")
        if chunk:
            yield chunk
        if end == len(offsets):
            break
        start = max(start + 1, end - overlap)


def chunks(text: str, selected: ChunkProfile):
    if selected.tokenizer is None:
        yield from character_chunks(text)
    else:
        yield from token_chunks(text, selected.tokenizer)
