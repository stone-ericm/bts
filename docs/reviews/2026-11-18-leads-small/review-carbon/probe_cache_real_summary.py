"""Record exact metrics for real-copy cache equivalence (base df454725 vs HEAD b0c11135)."""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from probe_cache import (
    REAL_COPY_CACHE,
    REAL_COPY_RAW,
    REVIEW_DIR,
    _call_build,
    _load_base_compute,
    check_equivalence_real_copies,
    check_equivalence_synthetic,
    check_interruptions,
)
from bts.features import compute as head_compute


def main() -> None:
    base_compute = _load_base_compute()
    real_bytes = REAL_COPY_CACHE.read_bytes()
    real_sha = hashlib.sha256(real_bytes).hexdigest()

    with tempfile.TemporaryDirectory(dir=REVIEW_DIR) as td:
        root = Path(td)
        c_head = root / "head.json"
        shutil.copy2(REAL_COPY_CACHE, c_head)
        lookup, out, err = _call_build(head_compute, c_head, REAL_COPY_RAW)
        assert out == "" and err == ""
        after_sha = hashlib.sha256(c_head.read_bytes()).hexdigest()
        assert after_sha == real_sha

    summary = {
        "real_cache_bytes": len(real_bytes),
        "real_cache_sha256": real_sha,
        "real_cache_entries": len(lookup),
        "synthetic": check_equivalence_synthetic(base_compute),
        "real_copies": check_equivalence_real_copies(base_compute),
        "interruptions": check_interruptions(base_compute),
    }
    (REVIEW_DIR / "probe_cache_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
