import datetime as dt
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
from zoneinfo import ZoneInfo

from click.testing import CliRunner
from bts import cli, daily_decision, dm, scheduler, util

ET = ZoneInfo("America/New_York")
DAY = "2027-12-31"


def probe(root):
    picks_dir = root / "picks with space"
    account = picks_dir / "account_state"
    account.mkdir(parents=True)
    log = account / "contest_manual_requests.jsonl"
    manual = account / "contest_streak.manual.json"
    switch = account / "season_contract.json"
    config = root / "config.toml"
    config.write_text('[scheduler]\nentry_intent = "enter"\npick_delivery = "dm"\n')
    now = dt.datetime(2027, 12, 31, 10, tzinfo=ET)

    class Clock(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return now.astimezone(tz) if tz else now.replace(tzinfo=None)

    def command(*args, code=0):
        with patch("datetime.datetime", Clock):
            result = CliRunner().invoke(cli.cli, [*args, "--picks-dir", str(picks_dir)])
        assert result.exit_code == code, (args, result.output, result.exception, code)
        return result.output

    def dated(streak, saver, *args, code=0):
        return command("set-contest-streak", "--date", DAY, "--streak", str(streak),
                       "--best-streak", "30", "--saver", saver, *args, code=code)

    def expect_count(count):
        assert len(log.read_text().splitlines()) == count

    command("season-contract", "enable", "--season", "2027")
    selected_switch = json.loads(switch.read_text())
    assert selected_switch["status"] == "automatic" and selected_switch["revoked_at"] is None
    dated(30, "used")
    first = json.loads(log.read_text())
    observed = json.loads(manual.read_text())
    assert first == dict(observed, requested_at=observed["recorded_at"])
    print("Initial request:", json.dumps(first, sort_keys=True))

    now = now.replace(hour=11)
    assert "Observation unchanged" in dated(30, "used")
    expect_count(1)
    assert json.loads(manual.read_text()) == observed
    dated(30, "used", "--reason", "same values, new reason")
    expect_count(2)
    installed = manual.read_bytes()

    now = now.replace(hour=15)
    with patch.object(util, "atomic_write_text", side_effect=PermissionError("installed file denied")):
        failure = dated(1, "not_earned", code=1)
    expect_count(3)
    assert manual.read_bytes() == installed
    print("Failed write:", failure.strip(), "Last request:", log.read_text().splitlines()[-1])
    assert "Observation unchanged" in dated(30, "used", "--reason", "same values, new reason")
    expect_count(3)

    now = now.replace(hour=16, minute=10)
    decision = daily_decision.write_decision(
        DAY, picks_dir, action="skip", source="mdp", streak=30, saver_available=False,
        state_source="contest", state_status="fresh", contest_source_date=DAY, now=now,
        delivery_status="not_applicable", scoreable=False,
    )
    assert decision["streak"] == json.loads(manual.read_text())["active_streak"]

    def outcome(label, expected_exits=(1,), games=None):
        sent = []
        when = now
        before = {p.name: p.read_bytes() for p in account.iterdir() if p.is_file()}
        with patch.object(scheduler, "fetch_schedule", return_value=[{}] if games is None else games), \
             patch.object(dm, "send_dm", side_effect=lambda who, text: sent.append(text) or "id"):
            raw = command("day-outcome", "--date", DAY, "--config", str(config),
                          "--dm-recipient", "offline.invalid", code=expected_exits[0])
        assert before == {p.name: p.read_bytes() for p in account.iterdir() if p.is_file()}
        print(label, str(when), raw.strip(), "DM:", sent)

    outcome("Failed pre-skip request flagged even though final install matches skip")

    now = now.replace(hour=18, minute=0)
    command("season-contract", "revoke", "--season", "2027", "--reason", "restored from the offseason snapshot")
    revoked = json.loads(switch.read_text())
    assert revoked["status"] == "revoked" and revoked["switched_at"] == selected_switch["switched_at"]
    print("Revoke:", json.dumps(revoked), "Remaining files:", sorted(p.name for p in account.iterdir()))

    now = now.replace(hour=18, minute=15)
    command("season-contract", "enable", "--season", "2027")
    enabled = json.loads(switch.read_text())
    assert enabled["status"] == "automatic" and enabled["revoked_at"] == revoked["revoked_at"]
    assert enabled["activation_id"] != revoked["activation_id"]
    print("Re-enable:", json.dumps(enabled))
    preserved = switch.read_bytes()
    now = now.replace(hour=18, minute=30)
    command("season-contract", "enable", "--season", "2027")
    assert switch.read_bytes() == preserved

    now = now.replace(hour=23, minute=55)
    dated(2, "used")
    assert json.loads(log.read_text().splitlines()[-1])["requested_at"] == "2028-01-01T04:55:00Z"

    now = now.replace(year=2028, month=1, day=1, hour=0, minute=5)
    command("season-contract", "revoke", "--season", "2028", "--reason", "restored from the offseason snapshot")
    new_season = json.loads(switch.read_text())
    assert new_season["switched_at"] is None and new_season["season"] == 2028
    print("Next-year revoke:", json.dumps(new_season))
    assert "must be today (2028-01-01)" in dated(1, "used", code=2)
    assert "Observation written" in command(
        "set-contest-streak", "--date", "2028-01-01", "--streak", "0",
        "--best-streak", "0", "--saver", "not_earned"
    )
    expect_count(5)
    now = now.replace(hour=8, minute=30)
    print("Default season status:", command("season-contract", "status"), sep="\n")
    print("Prior-season status:", command("season-contract", "status", "--season", "2027"), sep="\n")
    outcome("Wrong or superseded historical correction, synthetic playable Dec 31")
    outcome("Dec 31 with no regular-season games", expected_exits=(0,), games=[])


with tempfile.TemporaryDirectory() as temp:
    probe(Path(temp))
