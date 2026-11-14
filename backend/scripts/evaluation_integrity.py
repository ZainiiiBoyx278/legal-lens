"""Cross-platform source identities; data/model hashes retain their raw bytes."""
import hashlib
from pathlib import Path


def code_sha256(path):
    # Git may check Python out with CRLF on Windows and LF on Linux. Universal
    # newline decoding makes the source identity independent of that conversion.
    return hashlib.sha256(Path(path).read_text(encoding="utf-8").encode("utf-8")).hexdigest()
