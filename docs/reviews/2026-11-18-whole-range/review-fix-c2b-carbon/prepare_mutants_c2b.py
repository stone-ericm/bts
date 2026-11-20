"""Prepare isolated scratch mutants under .codex-review/wr-c2b/mutants/ without touching tracked files."""

from __future__ import annotations

from pathlib import Path
import shutil

REVIEW = Path(__file__).resolve().parent
SOURCE = REVIEW.parents[1] / "src"
MUTANTS_DIR = REVIEW / "mutants"
ALERT_REL = "bts/health/alert.py"

KEYSET_LINE = (
    '            and prior.keys() == {"schema_version", "status", "sent_day", "sent_sources", "message_id", "body_sha256"}\n'
)
PREDICATE_BLOCK = (
    "        if (\n"
    "            prior is not None\n"
    '            and prior.keys() == {"schema_version", "status", "sent_day", "sent_sources", "message_id", "body_sha256"}\n'
    '            and prior.get("schema_version") == HEALTH_DM_STATUS_SCHEMA_VERSION\n'
    '            and prior.get("status") == "sent"\n'
    '            and prior.get("sent_day") == day\n'
    '            and prior.get("sent_sources") == alert_key\n'
    '            and _usable_message_id(prior.get("message_id"))\n'
    '            and prior.get("body_sha256") == body_sha256\n'
    "        ):\n"
)

MUTANTS: dict[str, list[tuple[str, str, str]]] = {
    "m1_drop_keyset_check": [
        (ALERT_REL, KEYSET_LINE, ""),
    ],
    "m2_accept_superset_keys": [
        (
            ALERT_REL,
            'prior.keys() == {"schema_version", "status", "sent_day", "sent_sources", "message_id", "body_sha256"}',
            'prior.keys() >= {"schema_version", "status", "sent_day", "sent_sources", "message_id", "body_sha256"}',
        ),
    ],
    "m3_drop_duplicate_member_rejection": [
        (
            ALERT_REL,
            "data = json.load(handle, object_pairs_hook=_unique_receipt_members)",
            "data = json.load(handle)",
        ),
    ],
    "m4_drop_body_hash_comparison": [
        (ALERT_REL, '            and prior.get("body_sha256") == body_sha256\n', ""),
    ],
    "m5a_hash_constant": [
        (
            ALERT_REL,
            '    body_sha256 = hashlib.sha256(body.encode("utf-8")).hexdigest()',
            '    body_sha256 = hashlib.sha256(b"constant").hexdigest()',
        ),
    ],
    "m5b_hash_alert_message": [
        (
            ALERT_REL,
            "def _dispatch_with_acknowledgement(\n"
            "    body: str,\n"
            "    dm_recipient: str | None,\n"
            "    status_path: Path | str | None,\n"
            "    day: str,\n"
            "    alert_key: list[str],\n"
            ") -> bool:\n"
            '    """Use a dedicated receipt and keep the DM eligible if storage is unavailable."""\n'
            '    body_sha256 = hashlib.sha256(body.encode("utf-8")).hexdigest()',
            "def _dispatch_with_acknowledgement(\n"
            "    body: str,\n"
            "    dm_recipient: str | None,\n"
            "    status_path: Path | str | None,\n"
            "    day: str,\n"
            "    alert_key: list[str],\n"
            "    hash_source: str | None = None,\n"
            ") -> bool:\n"
            '    """Use a dedicated receipt and keep the DM eligible if storage is unavailable."""\n'
            '    body_sha256 = hashlib.sha256((body if hash_source is None else hash_source).encode("utf-8")).hexdigest()',
        ),
        (
            ALERT_REL,
            "    if require_message_id:\n"
            "        return _dispatch_with_acknowledgement(body, dm_recipient, status_path, now_day, alert_key)",
            "    if require_message_id:\n"
            '        msg_only = "\\n".join(a.message for a in critical + warn_attention)\n'
            "        return _dispatch_with_acknowledgement(body, dm_recipient, status_path, now_day, alert_key, msg_only)",
        ),
    ],
    "m6a_compare_sent_day_only": [
        (
            ALERT_REL,
            PREDICATE_BLOCK,
            '        if prior is not None and prior.get("sent_day") == day:\n',
        ),
    ],
    "m6b_keyset_and_sent_day_only": [
        (
            ALERT_REL,
            PREDICATE_BLOCK,
            "        if (\n"
            "            prior is not None\n"
            '            and prior.keys() == {"schema_version", "status", "sent_day", "sent_sources", "message_id", "body_sha256"}\n'
            '            and prior.get("sent_day") == day\n'
            "        ):\n",
        ),
    ],
}


def main() -> None:
    if MUTANTS_DIR.exists():
        shutil.rmtree(MUTANTS_DIR)
    MUTANTS_DIR.mkdir(parents=True)
    for name, edits in MUTANTS.items():
        dst = MUTANTS_DIR / name / "src"
        shutil.copytree(SOURCE, dst, ignore=shutil.ignore_patterns("__pycache__"))
        for rel, old, new in edits:
            target = dst / rel
            text = target.read_text(encoding="utf-8")
            count = text.count(old)
            assert count == 1, f"{name}:{rel} expected 1 occurrence, found {count}"
            patched = text.replace(old, new, 1)
            compile(patched, str(target), "exec")
            target.write_text(patched, encoding="utf-8")
        print(f"prepared {name}")
    print(f"ALL {len(MUTANTS)} SCRATCH MUTANTS PREPARED")


if __name__ == "__main__":
    main()
