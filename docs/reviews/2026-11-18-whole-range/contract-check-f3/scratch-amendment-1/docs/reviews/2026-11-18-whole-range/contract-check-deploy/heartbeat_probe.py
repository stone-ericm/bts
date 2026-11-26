"""Exercise the base heartbeat checker CLI with a fake network and clock."""

import contextlib
import importlib.util
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PATH = REPO / "scripts" / "check_heartbeat.py"
SPEC = importlib.util.spec_from_file_location("checker", PATH)
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


class Clock(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2027, 1, 1, 0, 5, tzinfo=timezone.utc).astimezone(tz)


def check(heartbeat, replace_contents=None):
    if replace_contents is not None:
        heartbeat.write_text(json.dumps(replace_contents))
    out = io.StringIO()
    url = "https://heartbeat.invalid/id"
    argv = [str(PATH), "--heartbeat-path", str(heartbeat), "--ping-url", url]
    with (patch.object(CHECKER, "datetime", Clock), patch.object(CHECKER, "ping") as ping,
          patch.object(CHECKER, "read_nrestarts", return_value=None) as churn,
          patch.object(sys, "argv", argv), contextlib.redirect_stdout(out)):
        try:
            CHECKER.main()
        except SystemExit as stopped:
            return {"exit": stopped.code, "stdout": out.getvalue().strip(),
                    "pings": ["".join(call.args) for call in ping.call_args_list],
                    "churn_called": churn.called}
    raise AssertionError("checker did not exit")


if __name__ == "__main__":
    with TemporaryDirectory(prefix="heartbeat-probe-", dir=HERE) as temp:
        heartbeat = Path(temp) / ".heartbeat"
        marker = {"state": "deploy_failed", "timestamp": "2027-01-01T00:00:00Z",
                  "step": "sync", "target_sha": "acbd123", "old_sha": "619abda"}
        sleeping = {"state": "sleeping", "timestamp": "2026-12-31T20:00:00Z",
                    "sleeping_until": "2027-01-01T15:00:00Z"}
        results = {"marker_first": check(heartbeat, marker), "marker_next": check(heartbeat),
                   "old_sleeping_heartbeat": check(heartbeat, sleeping)}
        assert results["marker_first"] == results["marker_next"]
        assert results["marker_first"]["exit"] == 1 and not results["marker_first"]["churn_called"]
        assert results["marker_first"]["pings"] == ["https://heartbeat.invalid/id/fail"]
        assert results["old_sleeping_heartbeat"]["pings"] == ["https://heartbeat.invalid/id"]
        print(json.dumps(results, indent=2))
