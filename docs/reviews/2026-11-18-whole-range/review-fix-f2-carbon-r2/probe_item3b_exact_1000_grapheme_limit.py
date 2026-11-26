"""Item 3b probe: exact 1,000-grapheme chat-limit boundary (1,957 code points, 1,000 graphemes) preserved byte-for-byte."""

from __future__ import annotations

import datetime as dt
import json
import sys
import unicodedata
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest
from click.testing import CliRunner

import bts
from bts import cli, contest_state
from bts.health import alert, contest_state as health, runner

ET = ZoneInfo("America/New_York")
REAL_DT = dt.datetime
DAY = REAL_DT(2027, 7, 15, 22, 15, tzinfo=ET)
MAIN_STATUS = "health_dm_delivery_status.json"
MISMATCH_STATUS = MAIN_STATUS + ".dated_used_mismatch.json"


def _clock(fixed_now: dt.datetime):
    class Clock(REAL_DT):
        @classmethod
        def now(cls, tz=None):
            return fixed_now.astimezone(tz) if tz else fixed_now.replace(tzinfo=None)
    return Clock


def test_exact_1000_grapheme_digest_preserved_and_mismatch_sent_second(tmp_path):
    picks = tmp_path / "data" / "picks"
    clk = _clock(DAY)
    with patch.object(dt, "datetime", clk), patch.object(cli, "datetime", clk):
        assert CliRunner().invoke(cli.cli, [
            "set-contest-streak", "--date", "2027-07-15", "--streak", "14",
            "--best-streak", "15", "--saver", "used", "--picks-dir", str(picks),
        ]).exit_code == 0

    flag_path = picks / "account_state" / "saver_state.json"
    flag_path.write_text('{"season": 2027, "state": "used"}', encoding="utf-8")

    # Header "BTS health CRITICAL alert(s):\n- [near_cap] " is 43 graphemes; 43 + 957 = 1,000 graphemes (1,957 code points)
    near_crit = alert.Alert("CRITICAL", "near_cap", "e\u0301" * 957)
    grapheme_counts = []

    def run_eod():
        sent = []
        orig = runner._safe_run

        def run_check(name, fn):
            if name == "contest_state":
                return orig(name, fn)
            return [near_crit] if name == "calibration" else []

        def strict_1000_transport(to, body):
            g = sum(not unicodedata.combining(ch) for ch in body)
            grapheme_counts.append(g)
            if g > 1000:
                raise ValueError(f"exceeds 1000 graphemes: {g}")
            sent.append((to, body))
            return f"ack-{len(sent)}"

        with patch.object(dt, "datetime", clk), \
             patch.object(contest_state, "datetime", clk), \
             patch.object(health, "datetime", clk), \
             patch.object(runner, "datetime", clk), \
             patch.object(alert, "datetime", clk), \
             patch.object(runner, "_safe_run", run_check), \
             patch.object(alert, "send_dm", strict_1000_transport):
            raw = runner.run_all_checks(
                picks, picks.parent / "models", "ops.bsky.social",
                today=DAY.date(), contest_state_expected=True,
            )
        return raw, sent

    _, ctrl_sends = run_eod()
    assert len(ctrl_sends) == 1
    assert len(ctrl_sends[0][1]) == 1957
    hs = picks.parent / "health_state"
    ctrl_status_bytes = (hs / MAIN_STATUS).read_bytes()
    assert json.loads(ctrl_status_bytes)["status"] == "sent"

    (hs / MAIN_STATUS).unlink()
    flag_path.write_text('{"season": 2027, "state": "active"}', encoding="utf-8")
    _, cand_sends = run_eod()
    assert len(cand_sends) == 2
    assert cand_sends[0] == ctrl_sends[0]
    assert (hs / MAIN_STATUS).read_bytes() == ctrl_status_bytes
    assert json.loads((hs / MISMATCH_STATUS).read_text())["status"] == "sent"
    assert grapheme_counts == [1000, 1000, 501]


if __name__ == "__main__":
    print(f"bts.__file__ = {bts.__file__}")
    sys.exit(pytest.main([__file__, "-v", "-s", "-rA"]))
