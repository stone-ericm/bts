"""Compare watchdog copy and checkpoints with real entry classification."""
import datetime as dt
import hashlib
import itertools
import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from probe_support import REVIEW, NOW, dump, invoke, offline
from bts import contest_fetch, entry_receipt, picks as pick_io
from tests.test_day_outcome_entry_sequence import _setup_committed_day

DAY = NOW.date().isoformat()
rows = {}
with TemporaryDirectory(dir=REVIEW, prefix="watchdog-fixtures-") as directory:
    root = Path(directory)
    for delivery, reason in itertools.product(
        ("locked_unconfirmed", "delivered", "private_locked", "legacy", "disagree", "stale"),
        ("no_pick", "mismatch"),
    ):
        picks = _setup_committed_day(root / f"{delivery}-{reason}", DAY, game_time_utc=f"{DAY}T23:10:00Z")
        path = picks / DAY / "decision.json"
        record = json.loads(path.read_bytes())
        daily = pick_io.load_pick(DAY, picks)
        daily.notification_id = "ack-id" if delivery in {"delivered", "legacy", "disagree", "stale"} else None
        if delivery in {"locked_unconfirmed", "private_locked"}:
            daily.notification_sent = False
        if delivery in {"locked_unconfirmed", "disagree", "stale"}:
            daily.delivery_attempted = True
            daily.double_down = replace(daily.pick, batter_id=123, batter_name="Second batter")
            record["action"] = "double"
            record["double_down"] = {"batter_id": 123, "batter_name": "Second batter"}
        if delivery == "legacy":
            path.unlink()
        else:
            record["delivery_status"] = "locked_unconfirmed" if delivery in {"disagree", "stale"} else delivery
            if delivery == "disagree":
                record["primary"]["batter_name"] = "Different record batter"
            if delivery == "stale":
                record["date"] = (NOW.date()-dt.timedelta(days=1)).isoformat()
            path.write_text(json.dumps(record))
        pick_io.save_pick(daily, picks)
        marker = picks.parent / "health_state/pick_entry_check.json"
        checkpoints = []
        with offline() as (send, _), patch("bts.cli._fetch_bts_to_mlb", return_value={88: 592450, 89: 123, 99: 997, 98: 998}):
            for clock, entry in (
                ("18:20", reason), ("18:22", reason), ("18:40", reason), ("18:55", reason),
                ("18:56", reason), ("18:57", "match"), ("18:58", "match"),
            ):
                players = [] if entry == "no_pick" else [99, 98] if entry == "mismatch" else [88, 89]
                pending = [dict(roundId=2, playerId=p) for p in players]
                send.reset_mock()
                with patch.object(contest_fetch, "fetch_pending_predictions", return_value=pending):
                    result = invoke(picks, "check-pick-entered", "--now-et", f"{DAY}T{clock}:00",
                                    "--dm-recipient", "offline.example", "--expected-username", "offline")
                current = json.loads(marker.read_bytes())
                receipt_id = current.pop("receipt")
                available = entry_receipt.discover(picks, DAY)
                bound = [r for r in available if r["attempt_id"] == receipt_id]
                assert len(bound) == 1 and bound[0]["et_date"] == DAY
                assert bound[0]["verifier"]["reason"] == entry
                commit = bound[0]["selection"]["commit"]
                assert commit["decision_sha256"] == (hashlib.sha256(path.read_bytes()).hexdigest()
                                                      if path.exists() else None)
                checkpoints.append(dict(exit=result.exit_code,
                                        console=result.output.replace(str(picks), "PICKS"),
                                        marker=current, total_receipts=len(available),
                                        dm=[c.args[1].replace(str(picks), "PICKS") for c in send.call_args_list]))
        rows[f"{delivery}-{reason}"] = checkpoints
        assert [x["exit"] for x in checkpoints] == [1, 1, 1, 1, 1, 0, 0]
        assert [len(x["dm"]) for x in checkpoints] == [1, 0, 1, 1, 0, 1, 0]
        assert checkpoints[5]["dm"] == [f"✅ BTS pick entry confirmed for {DAY} (match)."]

dump("watchdog", rows)
