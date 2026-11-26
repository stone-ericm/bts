"""C1 rank 4a: one archived input read once, plain or gzipped (X-E1: "preserve both plain and gzipped static input
support"; review r1 R8).

`read(base)` looks for `<base>.json` and `<base>.json.gz`:
- **neither:** `None` (absent);
- **one:** its stored bytes are read once; a gzip file is decompressed from that same buffer;
- **both (the conflict rule):** both are read and decoded. The same decoded content means the plain file is used and
  both buffers are kept as consumed. Different content, or either file unreadable, is an error.

An unreadable file, a corrupt gzip or a conflict sets `error`; the buffers read so far are still returned, so the
caller can retain exactly what it consumed. The caller decides: an erroneous slate refuses acceptance, and an
erroneous feed makes its game unknown.
"""
from __future__ import annotations

import gzip
import hashlib
import zlib
from dataclasses import dataclass, field
from pathlib import Path

FORMATS = (("json", ".json"), ("json.gz", ".json.gz"))


@dataclass
class Stored:
    files: list = field(default_factory=list)       # [(format, file name, stored bytes)], as consumed
    used: str | None = None                          # the format whose decoded bytes are parsed
    decoded: bytes | None = None
    error: str | None = None
    note: str | None = None

    def manifest(self) -> dict:
        return {"files": [{"format": f, "file": n, "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}
                          for f, n, b in self.files],
                "used": self.used, "error": self.error, "note": self.note,
                "decoded_sha256": None if self.decoded is None else hashlib.sha256(self.decoded).hexdigest()}


def _decode(fmt: str, b: bytes) -> bytes:
    return gzip.decompress(b) if fmt == "json.gz" else b


def read(base: Path) -> Stored | None:
    out = Stored()
    decoded = {}
    for fmt, suffix in FORMATS:
        p = base.with_name(base.name + suffix)
        try:
            if not p.exists():
                continue
            b = p.read_bytes()
        except OSError as exc:
            out.error = f"{p.name}: unreadable: {exc}"[:300]
            return out
        out.files.append((fmt, p.name, b))
        try:
            decoded[fmt] = _decode(fmt, b)
        except (OSError, EOFError, zlib.error) as exc:
            out.error = f"{p.name}: corrupt gzip: {exc}"[:300]
            return out
    if not out.files:
        return None
    if len(decoded) == 2:
        if decoded["json"] != decoded["json.gz"]:
            out.error = "both the plain and the gzipped file exist, with different content"
            return out
        out.note = "both formats present with identical content; the plain file was used"
    out.used = "json" if "json" in decoded else "json.gz"
    out.decoded = decoded[out.used]
    return out
