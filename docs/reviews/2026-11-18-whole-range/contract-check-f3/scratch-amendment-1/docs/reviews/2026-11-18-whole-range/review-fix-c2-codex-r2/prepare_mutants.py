"""Build six one-change mutants and one candidate repair outside the tracked tree."""

from pathlib import Path
import shutil


REVIEW = Path(__file__).resolve().parent
SOURCE = REVIEW.parents[1] / "src"
SCRATCH = REVIEW / "mutants"
ALERT = "bts/health/alert.py"
SCHEDULER = "bts/scheduler.py"

SCHEMA_CHECK = '        if (prior and prior.get("schema_version") == HEALTH_DM_STATUS_SCHEMA_VERSION\n'
DECISION = '''            _write_classification_decision(
                picks_dir, date,
                action=("double" if ld.double_down else "single"),
                delivered=delivered,
                attempted=attempted,
                primary=_row_from_daily(ld.pick),
                double_down=_row_from_daily(ld.double_down),
                state=state, daily=ld,
            )
'''
PAGE = '''            if attempted and not delivered:
                _alert_delivery_outcome_unknown(config, picks_dir, date)
'''
RESURFACE = '''    prior = None
    if not require_message_id:
        prior = _read_health_dm_delivery_status(status_path)
        resurface = _resurface_note(prior)  # H6: resurface a prior failed DM
        if resurface:
            body = resurface + ("\\n\\n" + body if body else "")
'''
MUTANTS = {
    "drop-schema-check": (ALERT, SCHEMA_CHECK + '                and prior.get("status")',
                          '        if (prior and prior.get("status")'),
    "before-classification-decision": (SCHEDULER, DECISION + PAGE, PAGE + DECISION),
    "follow-receipt-directory": (ALERT,
        "os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)",
        "os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)"),
    "drop-post-send-entry-recheck": (ALERT,
        "        _check_receipt_entry(directory_fd, name)\n        os.replace(temporary, name,",
        "        os.replace(temporary, name,"),
    "drop-resurface-guard": (ALERT, RESURFACE,
        ''.join(line[4:] for line in RESURFACE.splitlines(keepends=True)[2:])),
    "accept-none": (ALERT,
        '    return isinstance(value, str) and bool(value.strip())\n',
        '    return value is None or (isinstance(value, str) and bool(value.strip()))\n'),
}


def scratch_source(label):
    source = SCRATCH / label / "src"
    shutil.copytree(SOURCE, source, ignore=shutil.ignore_patterns("__pycache__"))
    return source


def replace_once(source, relative, old, new):
    path = source / relative
    original = path.read_text()
    assert original.count(old) == 1, (relative, old)
    patched = original.replace(old, new, 1)
    compile(patched, str(path), "exec")
    path.write_text(patched)


def prepare_candidate():
    source = scratch_source("candidate-receipt-check")
    replace_once(source, ALERT, "def _read_receipt(directory_fd", '''def _unique_receipt_members(pairs):
    members = {}
    for name, value in pairs:
        if name in members:
            raise ValueError(f"duplicate receipt key: {name}")
        members[name] = value
    return members


def _read_receipt(directory_fd''')
    replace_once(source, ALERT, "            data = json.load(handle)\n",
                 "            data = json.load(handle, object_pairs_hook=_unique_receipt_members)\n")
    replace_once(source, ALERT, SCHEMA_CHECK, '''        body_sha256 = hashlib.sha256(body.encode("utf-8")).hexdigest()
        sent_members = {"schema_version", "status", "sent_day", "sent_sources", "message_id", "body_sha256"}
        if (prior and prior.keys() == sent_members and prior.get("body_sha256") == body_sha256
                and prior.get("schema_version") == HEALTH_DM_STATUS_SCHEMA_VERSION
''')
    replace_once(source, ALERT,
                 '                payload.update(status="sent", message_id=message_id, sent_day=day, sent_sources=alert_key)',
                 '                payload.update(status="sent", message_id=message_id, sent_day=day,\n'
                 '                               sent_sources=alert_key, body_sha256=body_sha256)')


def main():
    for name, (file, old, new) in MUTANTS.items():
        replace_once(scratch_source(name), file, old, new)
        print(f"{name}: {file}")
    prepare_candidate()
    print("candidate-receipt-check: duplicate-member rejection, exact sent shape and body binding")


if __name__ == "__main__":
    main()
