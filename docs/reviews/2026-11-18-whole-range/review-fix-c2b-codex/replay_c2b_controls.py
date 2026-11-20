"""Refresh obsolete receipt expectations in separate copies of the older probes."""

from pathlib import Path
import runpy
import sys


REVIEW = Path(__file__).resolve().parent
CARBON = REVIEW / "replay" / "carbon"


def replace_once(source, expected, updated):
    assert source.count(expected) == 1, expected
    return source.replace(expected, updated)


def adapted_probe(kind):
    pattern = {"ack": "probe_3b_*.py", "damage": "probe_4b_*.py"}[kind]
    source, = CARBON.glob(pattern)
    text = source.read_text(encoding="utf-8")
    if kind == "ack":
        text = replace_once(text, '                                "message_id": reply,\n',
                            '                                "message_id": reply,\n'
                            '                                "body_sha256": sha256(u7_copy().encode()).hexdigest(),\n')
    else:
        text = replace_once(text, '                            "sent_sources": ["CRITICAL:late_delivery"], "message_id": "shared-old",\n',
                            '                            "sent_sources": ["CRITICAL:late_delivery"], "message_id": "shared-old",\n'
                            '                            "body_sha256": sha256(u7_copy().encode()).hexdigest(),\n')
    text = replace_once(text, "from pathlib import Path\n", "from pathlib import Path\nfrom hashlib import sha256\n")
    generated = CARBON / f"c2b-{source.name}"
    generated.write_text(text, encoding="utf-8")
    runpy.run_path(str(generated), run_name="__main__")


def diagnostics():
    sys.path.insert(0, str(REVIEW / "replay" / "codex"))
    import probe_failures
    import probe_support as support

    archived = runpy.run_path(str(REVIEW / "round2" / "codex" / "probe_diagnostics.py"))
    support.HERE = Path("/tmp")
    probe_failures.main()
    with support.deadline(15), support.fixtures("c2b-diagnostics") as (root, patch, opening):
        for route in ("direct", "classification"):
            archived["long_path_failure"](root / route, patch, opening, route)


if __name__ == "__main__":
    mode, = sys.argv[1:]
    if mode == "diagnostics":
        diagnostics()
    else:
        adapted_probe(mode)
