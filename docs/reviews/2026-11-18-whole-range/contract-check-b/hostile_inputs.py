"""Bound the FIFO probes and expose malformed day-outcome rendering inputs."""

import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from bts.day_outcome import classify_day


def fifo_blocks(root, target, statement):
    target.parent.mkdir(parents=True, exist_ok=True)
    os.mkfifo(target)
    try:
        subprocess.run([sys.executable, "-c", statement, str(root), str(target)],
                       check=True, timeout=4, capture_output=True, text=True)
    except subprocess.TimeoutExpired as error:
        assert b"before read" in (error.stdout or b""), error
        print(f"FIFO blocks: {target.relative_to(root)}")
    finally:
        target.unlink()


def main():
    try:
        classify_day(date="2027-07-15", games=1, decision_sha256=None, receipts=[],
                     intent="enter", expected_username="offline", decision={
                         "action": "single", "scoreable": True, "primary": {"batter_name": "A"},
                         "double_down": ["unused and malformed"],
                     })
    except AttributeError as error:
        print(f"Malformed unused runner-up crashes day-outcome: {error}")
    else:
        raise AssertionError("Expected the base renderer to read the unused field")

    with TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
        root = Path(folder)
        (root / "account_state").mkdir()
        (root / "account_state/saver_state.json").write_text(json.dumps({"state": "used", "season": 2027}))
        fifo_blocks(root, root / "account_state/contest_ledger.jsonl", """
from datetime import datetime, timezone
from pathlib import Path
import sys
from bts.season_contract import SwitchState, inspection_lines
print('before read', flush=True)
inspection_lines(Path(sys.argv[1]), datetime(2027, 7, 15, tzinfo=timezone.utc), SwitchState('absent'), season=2027)
""")
        fifo_blocks(root, root / "health_state/contest_streak_fetch_status.json", """
from pathlib import Path
import sys
from bts.cli import _contest_fetch_alert
print('before read', flush=True)
_contest_fetch_alert(Path(sys.argv[2]), None, 'synthetic failure')
""")


if __name__ == "__main__":
    main()
