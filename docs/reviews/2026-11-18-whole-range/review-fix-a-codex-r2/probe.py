"""Offline response replay and full-day final fallback confirmation."""

import codecs
from contextlib import redirect_stderr
from importlib.util import module_from_spec, spec_from_file_location
from io import BytesIO, StringIO
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DAY = "2027-03-24"
EMPTY = b'{"dates":[]}'
TOKENS = (b"NaN", b"Infinity", b"-Infinity")
GAME = (
    b'{"gamePk":123456,"gameType":"R","gameDate":"2027-03-24T23:10:00Z",'
    b'"doubleHeader":"N","status":{"abstractGameState":"Preview","statusCode":"S"},'
    b'"teams":{"away":{"team":{"id":147},"leagueRecord":{"pct":0.625}},'
    b'"home":{"team":{"id":137},"probablePitcher":{"id":543037}}},'
    b'"numericProbe":[1,-7,0.375,6.25e-2,-1.5E+2,123456789012345678901234567890]}'
)


def number_and_constant_cases():
    for token in TOKENS:
        error = f"non-JSON constant {token.decode()}"
        yield f"{token.decode()} as the root value", token, "AttributeError", error
        yield f"{token.decode()} as a root field", b'{"dates":[],"totalGames":' + token + b'}', [], error
        yield (
            f"{token.decode()} inside a filtered game",
            b'{"dates":[{"games":[{"gameType":"A","nested":[' + token + b'] }]}]}',
            [], error,
        )
    yield "empty dates", EMPTY, [], []
    yield "empty dates, zero games", b'{"dates":[],"totalGames":0}', [], []
    yield "constant spellings as strings", b'{"dates":[],"notes":["NaN","Infinity","-Infinity"]}', [], []
    game = GAME[:-1] + b',"notes":"NaN, Infinity and -Infinity"}'
    expected_game = {
        "gamePk": 123456, "gameType": "R", "gameDate": "2027-03-24T23:10:00Z", "doubleHeader": "N",
        "status": {"abstractGameState": "Preview", "statusCode": "S"},
        "teams": {"away": {"team": {"id": 147}, "leagueRecord": {"pct": 0.625}},
                  "home": {"team": {"id": 137}, "probablePitcher": {"id": 543037}}},
        "numericProbe": [1, -7, 0.375, 0.0625, -150.0, 123456789012345678901234567890],
        "notes": "NaN, Infinity and -Infinity",
    }
    body = (b'{"totalItems":2,"totalGames":2,"dates":[{"date":"2027-03-24","totalGames":2,'
            b'"games":[{"gameType":"S"},' + game + b']}]}')
    yield "hydrated slate with ordinary numbers and quoted spellings", body, [expected_game], [expected_game]


def edge_cases():
    first_games = b'"dates":[{"games":[{}]}]'
    empty_dates = b'"dates":[]'
    yield "duplicate dates, empty last", b'{' + first_games + b',' + empty_dates + b'}', [], []
    yield "duplicate dates, populated last", b'{' + empty_dates + b',' + first_games + b'}', [{}], [{}]
    yield "duplicate games, including escaped key", b'{"dates":[{"games":[{}],"g\\u0061mes":[]}]}', [], []
    yield "UTF-8 BOM", codecs.BOM_UTF8 + EMPTY, [], []
    for encoding in ("utf-16", "utf-16-be", "utf-32", "utf-32-be"):
        yield encoding, EMPTY.decode().encode(encoding), [], []
    yield "Windows-1252 metadata", b'{"dates":[],"note":"caf\xe9"}', "UnicodeDecodeError", "codec can't decode"
    yield "raw UTF-8 surrogate", b'{"dates":[],"note":"\xed\xa0\x80"}', [], []
    yield "trailing whitespace", EMPTY + b' \t\r\n', [], []
    yield "trailing object", EMPTY + b'{}', "JSONDecodeError", "Extra data"
    yield "trailing non-JSON text", EMPTY + b'Service Unavailable', "JSONDecodeError", "Extra data"
    yield "JSON fragment in HTML error", b'<html><p>503: ' + EMPTY + b'</p></html>', "JSONDecodeError", "Expecting value"
    yield "empty dates with positive count and error text", b'{"dates":[],"totalGames":1,"error":"failure"}', [], []
    yield "valid exponent exceeds float range in ignored metadata", b'{"dates":[],"totalGames":1e9999}', [], []


def replay(commit):
    from bts import scheduler

    print(f"{commit} imported {scheduler.__file__}")
    assert Path(scheduler.__file__).resolve().is_relative_to(Path(os.environ["PYTHONPATH"]))
    for label, body, base, head in (*number_and_constant_cases(), *edge_cases()):
        expected = head if commit == "HEAD" else base
        if commit == "9cb84536" and isinstance(head, str):
            if base == "AttributeError":
                expected = "MLB schedule must be a JSON object"
            elif not head.startswith("non-JSON constant"):
                expected = head
        with patch.object(scheduler, "retry_urlopen", return_value=BytesIO(body)) as request:
            try:
                actual = scheduler.fetch_schedule(DAY)
            except Exception as exc:
                if commit == "9e0ec28a":
                    assert type(exc).__name__ == expected, (label, expected, exc)
                elif expected == "MLB schedule must be a JSON object":
                    assert type(exc) is ValueError and str(exc) == expected, (label, exc)
                else:
                    assert type(exc) is ValueError, (label, exc)
                    assert str(exc).startswith("Invalid MLB schedule JSON: ") and expected in str(exc), (label, exc)
                print(f"{label} -> {type(exc).__name__}: {exc}")
            else:
                assert actual == expected, (label, expected, actual)
                if actual and "numericProbe" in actual[0]:
                    assert [type(value) for value in actual[0]["numericProbe"]] == [int, int, float, float, float, int]
                print(f"{label} -> {actual!r}")
            request.assert_called_once()


def archived_replay(commit):
    with TemporaryDirectory(prefix="source-", dir=HERE) as directory:
        archive = subprocess.check_output(["git", "archive", commit, "src"], cwd=ROOT)
        with tarfile.open(fileobj=BytesIO(archive)) as bundle:
            bundle.extractall(directory, filter="data")
        subprocess.run(
            [sys.executable, __file__, commit], cwd=ROOT, check=True,
            env={**os.environ, "PYTHONPATH": str(Path(directory) / "src")},
        )


def offline_fixture():
    location = ROOT / "docs/reviews/2026-11-18-whole-range/slice-3-codex/offline.py"
    spec = spec_from_file_location("offline_schedule_review", location)
    fixture = module_from_spec(spec)
    spec.loader.exec_module(fixture)
    fixture.ROOT = ROOT
    fixture.HERE = HERE
    return fixture


def callers(fixture):
    from bts import cli, scheduler
    from click.testing import CliRunner

    for token in (*TOKENS, None, b"0"):
        body = EMPTY if token is None else b'{"dates":[],"totalGames":' + token + b'}'
        rejects = token in TOKENS
        with fixture.offline(fixture.instant(DAY, 10)) as (world, overrides):
            days = []
            idle_calls = []

            def response(url, **_):
                path = urlsplit(url)
                assert path.path == "/api/v1/schedule"
                date, = parse_qs(path.query)["date"]
                days.append(date)
                return BytesIO(body if date == DAY else EMPTY)

            overrides.setattr(scheduler, "retry_urlopen", response)
            overrides.setattr(scheduler, "_idle_until_next_wakeup", lambda *args: idle_calls.append(args))
            result = CliRunner().invoke(cli.cli, ["schedule", "--date", DAY, "--config", str(world.config_path)])
            assert result.exit_code == int(rejects), (result.output, result.exception)
            assert ("No games scheduled" in result.stderr) is not rejects
            assert len(idle_calls) == int(not rejects)
            assert days == ([DAY] if rejects else [DAY, "2027-03-25"]), days
            if rejects:
                assert type(result.exception) is ValueError
                assert str(result.exception) == f"Invalid MLB schedule JSON: non-JSON constant {token.decode()}"
            assert scheduler.load_state(DAY, world.picks) is None
            assert not world.sends("dm") and not world.sends("health-dm")

            outcome = CliRunner().invoke(cli.cli, [
                "day-outcome", "--date", DAY, "--config", str(world.config_path),
                "--picks-dir", str(world.picks), "--dm-recipient", "offline.invalid",
            ], catch_exceptions=False)
            state = json.loads(outcome.stdout.splitlines()[0])["state"]
            assert outcome.exit_code == int(rejects), outcome.output
            assert state == ("schedule_unverifiable" if rejects else "no_games"), outcome.output
            assert len(world.sends("dm")) == int(rejects)
            assert days[-1] == DAY
            print(f"{body!r} -> schedule exit={result.exit_code} idle_calls={len(idle_calls)} "
                  f"error={result.exception if rejects else None}; "
                  f"day-outcome exit={outcome.exit_code} state={state} dms={len(world.sends('dm'))}")


def final_fallback(fixture, correct_at_final):
    from bts import picks, scheduler
    from bts.daily_decision import load_decision

    with fixture.offline(fixture.instant(DAY, 18, 25)) as (world, overrides):
        world.dated(0)
        world.config["scheduler"]["cluster_min"] = 0
        world.schedule[DAY][1]["gameDate"] = fixture.instant(DAY, 19, 15).isoformat()
        deliver = scheduler._deliver_and_lock_pick
        attempts = []

        def observe_and_correct(daily, config, directory, state, day, label, *, selection=None):
            if label == "final fallback":
                assert selection is not None and selection.decision is not None
                assert selection.pick_result.daily is daily
                assert world.now < picks.submission_cutoff_et(daily)
            if label != "final fallback" or correct_at_final:
                world.dated(0, best=daily.chosen_on["dated"]["best_streak"] + 1)
            delivered = deliver(daily, config, directory, state, day, label, selection=selection)
            stored = picks.load_pick(DAY, world.picks)
            record = load_decision(DAY, world.picks)
            attempts.append((label, delivered, stored.delivery_attempted, picks.pick_was_delivered(stored),
                             state.pick_locked, record and record["delivery_status"]))
            return delivered

        overrides.setattr(scheduler, "_deliver_and_lock_pick", observe_and_correct)
        journal = StringIO()
        with redirect_stderr(journal):
            scheduler.run_day(DAY, world.config)
        print(journal.getvalue(), end="")
        for record in attempts:
            print("delivery", record)
        assert [attempt[0] for attempt in attempts] == ["lineup", "lineup", "fallback", "final fallback"]
        assert attempts[-1][1:] == ((False, False, False, False, None) if correct_at_final
                                    else (True, True, True, True, "delivered"))
        assert all(row[1:] == (False, False, False, False, None) for row in attempts[:-1])
        assert len(world.sends("dm")) == int(not correct_at_final) and not world.sends("post")
        health_messages = world.sends("health-dm")
        assert sum("contest_state" in message for message in health_messages) == 1
        assert sum("missed_pick" in message for message in health_messages) == int(correct_at_final)
        print(f"final fallback correction={correct_at_final}; pick DMs={len(world.sends('dm'))}; "
              f"health DMs={len(health_messages)}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        replay(sys.argv[1])
    else:
        archived_replay("9e0ec28a")
        archived_replay("9cb84536")
        replay("HEAD")
        fixture = offline_fixture()
        callers(fixture)
        final_fallback(fixture, correct_at_final=True)
        final_fallback(fixture, correct_at_final=False)
