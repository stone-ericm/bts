import datetime as dt
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
from zoneinfo import ZoneInfo

from click.testing import CliRunner
from bts import cli, contest_state, daily_decision, day_outcome as trial, dm, entry_receipt, scheduler, util
from tests.test_day_outcome import DATE, SHA, _decision, _receipt

spec = importlib.util.spec_from_file_location("amendment_base_outcome", Path.cwd() / "src/bts/day_outcome.py")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)

ET = ZoneInfo("America/New_York")
SKIP = _decision("skip", scoreable=False) | {
    "streak": 30, "saver_available": False, "state_source": "contest",
}


def dated(day=DATE, *, streak=1, stamp=None):
    effective = dt.date.fromisoformat(day)
    end = dt.datetime.combine(effective, dt.time.max, ET)
    return {
        "schema_version": contest_state.DATED_MANUAL_SCHEMA, "season": effective.year,
        "effective_date": day, "expires_at": end.isoformat(),
        "active_streak": streak, "best_streak": 30, "saver_state": "used",
        "recorded_at": stamp or f"{day}T14:00:00Z", "source": "manual_cli",
    }


def switch(day=DATE, **changes):
    return {
        "schema_version": "bts_season_contract_v1", "season": int(day[:4]), "status": "revoked",
        "activation_id": "a" * 32, "switched_at": None, "revoked_at": f"{day}T20:00:00Z", "note": None,
    } | changes


def snapshot(root):
    return {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def cfg(root, intent="enter", body=None):
    delivery = "dm" if intent == "enter" else "private"
    config = root / "config.toml"
    config.write_text(body or f'[scheduler]\nentry_intent = "{intent}"\npick_delivery = "{delivery}"\n')
    return config


def command(config, picks, schedule, module, *, day=DATE, send_failure=None):
    sends = []

    def send(who, message):
        sends.append((who, message))
        if send_failure:
            raise send_failure
        return "offline"

    with patch.object(trial, "gather", module.gather), patch.object(scheduler, "fetch_schedule", side_effect=schedule), \
         patch.object(dm, "send_dm", side_effect=send):
        result = CliRunner().invoke(cli.cli, ["day-outcome", "--date", day, "--config", str(config),
                                              "--picks-dir", str(picks), "--expected-username", "stonehengee",
                                              "--dm-recipient", "offline.invalid"])
    return (result.exit_code, result.output, sends, type(result.exception).__name__)


def parity(root):
    picks = root / "picks"
    account = picks / "account_state"
    account.mkdir(parents=True)
    (account / "contest_streak.manual.json").write_text(json.dumps(dated()))
    (account / "season_contract.json").write_text(json.dumps(switch()))
    batter = _receipt(confirmed=False)
    batter["qualification"]["slots"] = [{"state": "unit_unverified"}]
    game = [{}]
    committed = _decision()
    absent = [_receipt(confirmed=False, reason="no_pick")]
    confirmed = [_receipt()]
    cases = [
        ("entered", committed, game, "enter", confirmed),
        ("entered_game_unverified", committed, game, "enter", [batter]),
        ("entered_late", committed, game, "enter", [_receipt(before_cutoff=False)]),
        ("not_entered", committed, game, "enter", absent),
        ("entry_unverifiable", committed, game, "enter", []),
        ("no_decision", None, game, "enter", []),
        ("no_games", SKIP, [], "enter", []),
        ("no_games", SKIP, [{"status": {"detailedState": "Postponed"}}], "enter", []),
        ("commit_not_scoreable", _decision(scoreable=False), game, "enter", []),
        ("research_commit", _decision(delivery_status="private_locked"), game, "research", []),
        ("delivered_under_research", committed, game, "research", []),
        ("contradiction", committed, [], "enter", []),
        ("intent_unknown", SKIP, game, "broken", []),
        ("schedule_unverifiable", SKIP, OSError("schedule offline"), "enter", []),
        ("skip_not_deliberate", SKIP | {"source": "other"}, game, "enter", []),
    ]
    for label, decision, games, intent, receipts in cases:
        def schedule(_):
            if isinstance(games, Exception):
                raise games
            return games

        config = cfg(root, intent)
        with patch.object(daily_decision, "load_decision_bytes", return_value=(decision, b"static")), \
             patch.object(entry_receipt, "_sha", return_value=SHA), \
             patch.object(entry_receipt, "discover", return_value=receipts), \
             patch.object(trial, "_read_evidence", side_effect=AssertionError("non-skip attempted evidence")):
            before = command(config, picks, schedule, base)
            after = command(config, picks, schedule, trial)
        assert before == after and json.loads(before[1])["state"] == label, (label, before, after)
        print("K3/K4 parity", label, before[0], len(before[2]))

    path = daily_decision.decision_path(DATE, picks)
    path.parent.mkdir()
    path.write_text(json.dumps(SKIP))
    result = command(cfg(root, "research"), picks, lambda _: game, trial)
    assert result[0] == 1 and "A dated observation" in result[1] and "records a revocation" not in result[1]
    print("Research dated reason remains enabled", result[2])

    path.write_text(json.dumps(SKIP | {"action": {}}))
    before = command(cfg(root), picks, lambda _: game, base)
    after = command(cfg(root), picks, lambda _: game, trial)
    assert before == after == (1, "", [], "TypeError")
    print("K5 pre-existing malformed action exception", before)

    for case in ({"streak": True}, {"streak": -1}, {"streak": None}, {"saver_available": 0},
                 {"saver_available": None}):
        path.write_text(json.dumps(SKIP | case))
        before = command(cfg(root), picks, lambda _: game, base)
        after = command(cfg(root), picks, lambda _: game, trial)
        assert before == after and before[0] == 0
    print("K5 malformed streak or saver fields retain the base result for both reasons")

    for case in ({"date": "2027-04-01"}, {"scoreable": True}):
        path.write_text(json.dumps(SKIP | case))
        before = command(cfg(root), picks, lambda _: game, base)
        after = command(cfg(root), picks, lambda _: game, trial)
        assert before[0] == 0 and after[0] == 1
        print("K5 poorly formed but base-readable skip still upgraded:", case, after[2])


def input_matrix(root):
    picks = root / "picks"
    account = picks / "account_state"
    account.mkdir(parents=True)
    paths = {
        "current": account / "contest_streak.manual.json",
        "archive": account / "contest_streak.manual.revoked-20270402T203000000000Z.json",
        "switch": account / "season_contract.json",
    }
    path = daily_decision.decision_path(DATE, picks)
    path.parent.mkdir()
    path.write_text(json.dumps(SKIP))
    config = cfg(root)

    def check():
        return command(config, picks, lambda _: [{}], trial)

    def valid(name):
        return switch() if name == "switch" else dated()

    def wrong_date(name):
        return (switch("2027-04-01") if name == "switch" else dated("2027-04-01"))

    def wrong_season(name):
        return valid(name) | {"season": 2028}

    writers = {
        "absent": lambda p, _: None,
        "directory": lambda p, _: p.mkdir(),
        "FIFO": lambda p, _: os.mkfifo(p),
        "invalid UTF-8": lambda p, _: p.write_bytes(b"\xff"),
        "malformed JSON": lambda p, _: p.write_text('{"x":'),
        "deep JSON": lambda p, _: p.write_text("[" * 2000 + "0" + "]" * 2000),
        "long number": lambda p, _: p.write_text('{"x":' + "1" * 5000 + "}"),
        "oversize, otherwise valid": lambda p, n: p.write_text(json.dumps(valid(n) | {
            "ignored": "x" * (trial.EVIDENCE_MAX_BYTES + 1),
        })),
        "wrong date": lambda p, n: p.write_text(json.dumps(wrong_date(n))),
        "wrong season": lambda p, n: p.write_text(json.dumps(wrong_season(n))),
        "wrong schema": lambda p, n: p.write_text(json.dumps(valid(n) | {"schema_version": "legacy"})),
    }
    for label, write in writers.items():
        for name, evidence in paths.items():
            write(evidence, name)
            assert check()[0] == 0, (label, name)
            good = "current" if name == "switch" else "switch"
            paths[good].write_text(json.dumps(valid(good)))
            assert check()[0] == 1, (label, name)
            paths[good].unlink()
            if evidence.is_dir():
                evidence.rmdir()
            else:
                evidence.unlink(missing_ok=True)
        print("K5 each of current, archive, switch ignored independently:", label)

    for name, evidence in paths.items():
        evidence.write_text(json.dumps(valid(name)))
        good = "archive" if name == "switch" else "switch"

        def denied(target):
            if target == evidence:
                raise PermissionError("simulated EACCES")
            return util.read_regular_file_text(target)

        with patch.object(trial, "read_regular_file_text", side_effect=denied):
            assert check()[0] == 0
            paths[good].write_text(json.dumps(valid(good)))
            assert check()[0] == 1
        paths[good].unlink()
        evidence.unlink()
    print("K5 simulated EACCES on all three paths")

    paths["switch"].write_text(json.dumps(switch(revoked_at="0001-01-01T00:00:00Z")))
    assert check()[0] == 0
    print("K5 validator-accepted extreme revoke timestamp ignored on ET conversion")

    paths["current"].write_text(json.dumps(dated(streak=1)))
    paths["archive"].write_text(json.dumps(dated(streak=30, stamp=f"{DATE}T10:00:00-04:00")))
    outcome = check()
    assert "gives streak 1" in outcome[1]
    paths["current"].unlink()
    paths["archive"].rename(paths["current"])
    paths["archive"].write_text(json.dumps(dated(streak=1)))
    assert check()[0] == 0
    print("Equal instants with different offsets and values need a tie rule")

    paths["current"].write_text(json.dumps(dated()))
    regular_reads = []

    def note_read(path):
        text = util.read_regular_file_text(path)
        regular_reads.append((path.name, len(text.encode("utf-8"))))
        return text

    paths["archive"].write_text(json.dumps(dated() | {"reason": "x" * 96 * 1024}))
    with patch.object(trial, "read_regular_file_text", side_effect=note_read):
        assert check()[0] == 1
    assert any(bytes_read > trial.EVIDENCE_MAX_BYTES for _, bytes_read in regular_reads)
    print("Reader buffers the whole oversized archive before its rejection", regular_reads)

    paths["archive"].unlink()
    paths["switch"].unlink()
    before = snapshot(root)
    failed = command(config, picks, lambda _: [{}], trial, send_failure=OSError("DM failed"))
    assert failed[0] == 1 and "DM failed: OSError" in failed[1] and len(failed[2]) == 1
    assert snapshot(root) == before
    bad = cfg(root, body="[invalid")
    for module in (base, trial):
        outcome = command(bad, picks, lambda _: [{}], module)
        assert outcome[0] == 2 and "THE CHECK COULD NOT RUN" in outcome[2][0][1]
        print("K6 CLI could not run", outcome)


def year_boundary(root):
    day = "2027-12-31"
    picks = root / "picks"
    picks.mkdir(parents=True)
    config = cfg(root)
    now = dt.datetime(2027, 12, 31, 19, tzinfo=ET)

    class Clock(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return now.astimezone(tz) if tz else now.replace(tzinfo=None)

    def call(*args):
        with patch("datetime.datetime", Clock):
            result = CliRunner().invoke(cli.cli, [*args, "--picks-dir", str(picks)])
        assert result.exit_code == 0, (args, result.output, result.exception)
        return result.output

    record = daily_decision.write_decision(day, picks, action="skip", source="mdp", streak=30,
                                          saver_available=False, delivery_status="not_applicable",
                                          scoreable=False, now=now)
    assert record["date"] == day
    now = now.replace(hour=23, minute=50)
    call("season-contract", "revoke", "--season", "2027", "--reason", "restored from the offseason snapshot")
    assert "2028-01-01T04:50:00Z" in (picks / "account_state/season_contract.json").read_text()
    now = now.replace(year=2028, month=1, day=1, hour=0, minute=30)
    affects_dec31 = command(config, picks, lambda _: [{}], trial, day=day)
    assert affects_dec31[0] == 1 and "23:50 ET" in affects_dec31[2][0][1]
    print("R64 Dec 31 revoke retained into Jan 1, rendered:", affects_dec31[2])
    call("season-contract", "revoke", "--season", "2028", "--reason", "restored from the offseason snapshot")
    now = now.replace(hour=8, minute=30)
    after_boundary = command(config, picks, lambda _: [{}], trial, day=day)
    assert after_boundary[0] == 0 and json.loads(after_boundary[1])["state"] == "deliberate_skip"
    print("K7 Jan 1 revoke is outside D, and also overwrote the earlier D revoke", after_boundary)
    print("Status on Jan 1, R64 scenario:\n" + call("season-contract", "status"))


with tempfile.TemporaryDirectory() as temp:
    root = Path(temp)
    parity(root / "parity")
    input_matrix(root / "evidence")
    year_boundary(root / "year")
