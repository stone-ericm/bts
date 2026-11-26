"""Follow-up for E1 maximum representable ET timestamp and E2 exact documentation check."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from zoneinfo import ZoneInfo

from click.testing import CliRunner
from bts import cli, contest_state, daily_decision, day_outcome, dm, scheduler
from tests.test_day_outcome import DATE, _decision


REPO = Path(__file__).resolve().parents[2]
ET = ZoneInfo("America/New_York")


def main() -> None:
    skip_dec = _decision("skip", scoreable=False) | {"streak": 30, "saver_available": False}
    end_et = dt.datetime.combine(dt.date.fromisoformat(DATE), dt.time.max, ET).isoformat()

    def make_dated(streak: int, recorded_at: str) -> dict:
        return {
            "schema_version": contest_state.DATED_MANUAL_SCHEMA,
            "season": int(DATE[:4]),
            "effective_date": DATE,
            "expires_at": end_et,
            "active_streak": streak,
            "best_streak": max(streak, 30),
            "saver_state": "used",
            "recorded_at": recorded_at,
            "source": "manual_cli",
        }

    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        picks = root / "picks"
        account = picks / "account_state"
        account.mkdir(parents=True)
        dec_path = daily_decision.decision_path(DATE, picks)
        dec_path.parent.mkdir(parents=True)
        dec_path.write_text(json.dumps(skip_dec))
        # Both timestamps convert cleanly to ET (min ET = 0001-01-01T00:00:00 LMT; max UTC = 9999-12-31T23:59:59Z = 18:59:59 EST)
        (account / "contest_streak.manual.revoked-min.json").write_text(
            json.dumps(make_dated(4, "0001-01-01T04:56:02Z"))
        )
        (account / "contest_streak.manual.revoked-max.json").write_text(
            json.dumps(make_dated(5, "9999-12-31T23:59:59Z"))
        )
        cfg = root / "config.toml"
        cfg.write_text('[scheduler]\nentry_intent = "enter"\npick_delivery = "dm"\n')
        sends = []
        with patch.object(scheduler, "fetch_schedule", return_value=[{}]), \
             patch.object(dm, "send_dm", side_effect=lambda r, m: sends.append((r, m)) or "ok"):
            res = CliRunner().invoke(
                cli.cli,
                ["day-outcome", "--date", DATE, "--config", str(cfg),
                 "--picks-dir", str(picks), "--expected-username", "stonehengee",
                 "--dm-recipient", "operator.bsky.social"],
            )
        body = json.loads(res.stdout)
        assert res.exit_code == 1 and body["state"] == "skip_state_review" and "records streak 5" in body["detail"]
        print("E1 representable ET min/max ranking:", json.dumps(body))

    req_skip = (
        "A deliberate MDP skip can prompt `skip_state_review` if the selected dated observation for that day "
        "records a different streak or policy saver availability. The report uses a valid installed observation "
        "when present and otherwise checks retained archives. In enter mode it also prompts review if the switch "
        "remains revoked for that season, the revocation was on the skip date in ET, and no valid retained dated "
        "observation matches both inputs."
    )
    req_missing = (
        "Missing or unusable optional contest evidence can leave a deliberate skip clean. The base outcome checks "
        "report missing schedule, decision or entry evidence where required."
    )
    claude_line_66 = (REPO / "CLAUDE.md").read_text().splitlines()[65]
    doc_norm = " ".join((day_outcome.__doc__ or "").split())

    assert req_skip in claude_line_66, claude_line_66
    assert req_missing in claude_line_66, claude_line_66
    assert "Missing evidence reads as unverifiable, never as a pass." not in claude_line_66
    assert req_skip in doc_norm, doc_norm
    assert req_missing in doc_norm, doc_norm
    assert "Missing or failed evidence is reported as unverifiable" not in doc_norm
    print("E2 exact text check passed on CLAUDE.md:66 and day_outcome.__doc__.")


if __name__ == "__main__":
    main()
