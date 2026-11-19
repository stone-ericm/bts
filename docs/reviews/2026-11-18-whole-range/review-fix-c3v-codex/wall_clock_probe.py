"""Verify the shifted wall clock and run the complete focused suite in one process."""

from datetime import date, datetime
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo


def main():
    import pytest

    year = int(sys.argv[1])
    et = ZoneInfo("America/New_York")
    assert date.today().year == datetime.now(et).year == datetime.fromtimestamp(time.time(), et).year == year
    print(f"Verified wall clock {datetime.now(et).isoformat()}", flush=True)
    return pytest.main(["-p", "no:cacheprovider", "-q", "--tb=short",
                        "tests/test_wr_c3v_corpus.py", "tests/test_wr_c3v_delivery.py",
                        "tests/test_wr_c3v_reconcile.py", "tests/test_wr_c3_r2_entered_pick_blockers.py",
                        "tests/test_streak_replay.py", "tests/test_streak_season_cs2.py",
                        "tests/test_reconcile_receipt.py", "tests/test_lightgbm_optional.py",
                        "tests/test_web_saver.py"])


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    raise SystemExit(main())
