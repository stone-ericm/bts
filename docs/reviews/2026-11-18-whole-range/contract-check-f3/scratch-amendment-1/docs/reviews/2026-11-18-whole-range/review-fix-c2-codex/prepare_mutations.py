"""Prepare independent source overlays from an archived, unmodified HEAD."""

from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parent / "mutations"
SCHEDULER = "bts/scheduler.py"
ALERT = "bts/health/alert.py"
CLASSIFICATION_CALL = "            if attempted and not delivered:\n                _alert_delivery_outcome_unknown(config, picks_dir, date)\n"
EARLY_MARKER = "            _write_classification_decision(\n                picks_dir, date,"
DST = 'status_path=picks_dir.parent / "health_state" / "delivery_outcome_unknown" / f"{slate}.json"'


MUTATIONS = {
    "wall-key": [
        (SCHEDULER, "now_et_date=slate,", "now_et_date=_now_et().date(),"),
    ],
    "wall-filename": [
        (SCHEDULER, DST, DST.replace("{slate}", "{_now_et().date()}")),
    ],
    "shared-file": [
        (SCHEDULER, DST, 'status_path=picks_dir.parent / "health_state" / "health_dm_delivery_status.json"'),
    ],
    "ack-none": [
        (ALERT, "return isinstance(value, str) and bool(value.strip())",
         "return value is None or (isinstance(value, str) and bool(value.strip()))"),
    ],
    "drop-classification": [
        (SCHEDULER, CLASSIFICATION_CALL, ""),
    ],
    "early-classification": [
        (SCHEDULER, CLASSIFICATION_CALL, ""),
        (SCHEDULER, EARLY_MARKER, CLASSIFICATION_CALL + EARLY_MARKER),
    ],
    "retry-banner": [
        (ALERT, '        payload = {"schema_version": HEALTH_DM_STATUS_SCHEMA_VERSION}',
         '        if prior and prior.get("status") == "failed":\n'
         '            body = _resurface_note({**prior, "critical_count": 1}) + "\\n\\n" + body\n'
         '        payload = {"schema_version": HEALTH_DM_STATUS_SCHEMA_VERSION}'),
    ],
    "follow-symlinks": [
        (ALERT, "follow_symlinks=False", "follow_symlinks=True"),
        (ALERT, "os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=directory_fd",
         "os.O_RDONLY | os.O_NONBLOCK, dir_fd=directory_fd"),
        (ALERT, "os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)", "os.O_RDONLY | os.O_DIRECTORY)"),
    ],
    "follow-parent": [
        (ALERT, "os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)", "os.O_RDONLY | os.O_DIRECTORY)"),
    ],
    "require-receipt-schema": [
        (ALERT, '        if (prior and prior.get("status") == "sent" and prior.get("sent_day") == day',
         '        if (prior and prior.get("schema_version") == HEALTH_DM_STATUS_SCHEMA_VERSION\n'
         '                and prior.get("status") == "sent" and prior.get("sent_day") == day'),
    ],
}


def main():
    original = ROOT / "reference" / "src"
    for mutation, edits in MUTATIONS.items():
        source = ROOT / mutation / "src"
        shutil.copytree(original, source)
        for name, old, new in edits:
            path = source / name
            text = path.read_text()
            assert text.count(old) == 1, (mutation, name, old, text.count(old))
            path.write_text(text.replace(old, new))
        print(mutation, f"{len(edits)} source edit(s)")


if __name__ == "__main__":
    main()
