"""Item 2 supplement: verify historical contest_state health read at end-of-slate microsecond and huge JSON int parity."""
from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
ET = ZoneInfo("America/New_York")


def main() -> int:
    from click.testing import CliRunner
    from bts import cli
    from bts.health import contest_state as health_cs

    problems = []
    day = dt.date(2027, 7, 15)
    end_of_slate = dt.datetime.combine(day, dt.time.max, ET)
    after_midnight = dt.datetime(2027, 7, 16, 0, 5, tzinfo=ET)

    with tempfile.TemporaryDirectory(prefix="h2c-item2b-") as tmp:
        picks_dir = Path(tmp) / "picks"
        picks_dir.mkdir()
        # Write valid dated 400/400 directly with pinned timestamps
        man_path = picks_dir / "account_state" / "contest_streak.manual.json"
        man_path.parent.mkdir(parents=True)
        good_dated = {
            "schema_version": "bts_contest_streak_manual_v3",
            "effective_date": "2027-07-15",
            "season": 2027,
            "expires_at": end_of_slate.isoformat(),
            "active_streak": 400,
            "best_streak": 400,
            "saver_state": "used",
            "recorded_at": "2027-07-15T14:00:00Z",
            "source": "manual_cli",
        }
        man_path.write_text(json.dumps(good_dated))
        auto_path = picks_dir / "account_state" / "contest_streak.json"
        auto_path.write_text(json.dumps({"active_streak": 401, "best_streak": 401}))

        # At slate's last microsecond (how C1 runner evaluates historical_day), dated 400/400 wins and gives []
        alerts_eod = health_cs.check(picks_dir, expected=True, now=end_of_slate, historical_day=day)
        if alerts_eod != []:
            problems.append(f"historical EOD at slate end alerted on valid 400/400: {alerts_eod}")

        # After midnight, dated 2027-07-15 is expired, so auto 401/401 is read and refused!
        alerts_post = health_cs.check(picks_dir, expected=True, now=after_midnight, historical_day=day)
        if len(alerts_post) != 1 or "The automatic contest streak file has an invalid streak or best streak." not in alerts_post[0].message:
            problems.append(f"after-midnight fallback to auto 401/401 unexpected: {alerts_post}")

        # Path-normalized huge 5000-digit integer parity check between base_tree and head
        huge_results = {}
        for side, src_dir in (("base", OUT_DIR / "base_tree" / "src"), ("head", REPO / "src")):
            code = (
                "import json, tempfile; from pathlib import Path; from bts import contest_state as cs\n"
                "out = []\n"
                "with tempfile.TemporaryDirectory() as d:\n"
                "    for fn in ('contest_streak.manual.json', 'contest_streak.json'):\n"
                "        p = Path(d) / fn\n"
                "        p.write_text('{\"active_streak\": ' + ('9'*5000) + '}')\n"
                "        for h in (False, True):\n"
                "            try:\n"
                "                cs._read_state_file(p, require_dated=False, historical=h)\n"
                "            except Exception as e:\n"
                "                out.append([type(e).__name__, str(e).replace(str(p), '<PATH>')[:120]])\n"
                "print(json.dumps(out))\n"
            )
            proc = subprocess.run(
                [str(REPO / ".venv/bin/python"), "-c", code],
                env=dict(os.environ, PYTHONPATH=str(src_dir)),
                capture_output=True, text=True, check=True, timeout=15,
            )
            huge_results[side] = json.loads(proc.stdout)
        if huge_results["base"] != huge_results["head"]:
            problems.append(f"huge int mismatch: {huge_results}")

    msg = f"historical EOD + huge-int parity ok; problems={len(problems)}\n"
    (OUT_DIR / "item2_historical_eod.log").write_text(msg, encoding="utf-8")
    print(msg, end="")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
