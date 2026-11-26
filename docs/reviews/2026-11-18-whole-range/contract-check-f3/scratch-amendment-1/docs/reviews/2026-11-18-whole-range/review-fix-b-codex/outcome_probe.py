"""Replay day-outcome display and receipt binding on both revisions."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from probe_support import REVIEW, NOW, dump, invoke, offline
from bts import daily_decision, entry_receipt, receipt_io, scheduler
from tests.test_day_outcome import _receipt

DAY = NOW.date().isoformat()

cases = {
    "single": {"action": "single"},
    "double": {"action": "double"},
    "single-runner-list": {"double_down": ["unused"]},
    "single-primary-list": {"primary": ["invalid"]},
    "single-primary-null": {"primary": None},
    "double-secondary-null": {"action": "double", "double_down": None},
    "double-secondary-int": {"action": "double", "double_down": {"batter_name": 42}},
    "single-primary-space": {"primary": {"batter_name": "  "}},
    "missing-action": {},
    "unknown-action": {"action": "triple"},
    "list-action": {"action": ["single"]},
}
rows = {}
with TemporaryDirectory(dir=REVIEW, prefix="outcome-fixtures-") as directory:
    root = Path(directory)
    config = root / "config.toml"
    config.write_text('[scheduler]\npick_delivery = "dm"\nentry_intent = "enter"\n')
    with offline() as (send, _), patch.object(scheduler, "fetch_schedule", return_value=[{"gamePk": 1}]):
        for name, overrides in cases.items():
            picks = root / name
            record = daily_decision.write_decision(
                DAY, picks, action="single", source="mdp", primary={"batter_name": "A"},
                double_down={"batter_name": "B"}, delivery_status="delivered", scoreable=True,
            ) | overrides
            if name == "missing-action":
                record.pop("action")
            path = picks / DAY / "decision.json"
            path.write_text(json.dumps(record))
            before = path.read_bytes()
            rows[name] = []
            for confirmed in (False, True):
                if confirmed:
                    receipt = _receipt(date=DAY, sha=hashlib.sha256(before).hexdigest())
                    receipt_io.publish(entry_receipt.receipts_root(picks) / DAY / "bound.json",
                                       json.dumps(receipt).encode())
                send.reset_mock()
                result = invoke(picks, "day-outcome", "--date", DAY, "--config", str(config),
                                "--expected-username", "stonehengee", "--dm-recipient", "offline.example")
                assert path.read_bytes() == before
                rows[name].append(dict(receipt=confirmed, exit=result.exit_code,
                                       reported=json.loads(result.stdout) if result.stdout else None,
                                       exception=type(result.exception).__name__ if result.exception else None,
                                       dm=[c.args[1] for c in send.call_args_list]))

dump("outcome", rows)
