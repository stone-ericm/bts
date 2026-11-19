"""Test the hydrated schedule parser at the scheduler and watchdog entry points."""

from datetime import timedelta
from io import BytesIO
import json
from urllib.error import URLError
from urllib.parse import parse_qs, urlsplit

import pytest
from click.testing import CliRunner

from bts import cli, dm, scheduler
from tests.test_wr_a_pick import DAY, review_world


@pytest.mark.parametrize("case", ["malformed", "failed_alert", "off_day"])
def test_bad_schedule_shape_propagates_and_day_outcome_alerts(review_world, case, capsys):
    with review_world.offline(review_world.instant(DAY, 10)) as (world, patch):
        malformed = case != "off_day"
        payload = b'{"message":"synthetic upstream error"}' if malformed else b'{"dates":[]}'
        patch.setattr(scheduler, "retry_urlopen", lambda *a, **k: BytesIO(payload))
        attempts = []
        if case == "failed_alert":
            def fail(recipient, body):
                attempts.append(body)
                raise RuntimeError("synthetic send failure")

            patch.setattr(dm, "send_dm", fail)
        if malformed:
            with pytest.raises(ValueError, match="dates"):
                scheduler.run_day(DAY, world.config)
            assert "No games scheduled" not in capsys.readouterr().err
        else:
            scheduler.run_day(DAY, world.config)
        assert scheduler.load_state(DAY, world.picks) is None
        result = CliRunner().invoke(cli.cli, [
            "day-outcome", "--date", DAY, "--config", str(world.config_path),
            "--picks-dir", str(world.picks), "--dm-recipient", "offline.invalid",
        ], catch_exceptions=False)
        assert result.exit_code == int(malformed)
        assert json.loads(result.stdout.splitlines()[0])["state"] == (
            "schedule_unverifiable" if malformed else "no_games"
        )
        assert len(attempts or world.sends("dm")) == int(malformed)
        if case == "failed_alert":
            assert "DM failed: RuntimeError" in result.stderr


@pytest.mark.parametrize("payload,location", [
    (b"{", "JSON"), (b"\xff", "JSON"), (b"[]", "object"),
    (b'{"dates":null}', "dates"), (b'{"dates":{}}', "dates"),
    (b'{"dates":[null]}', r"dates\[0\]"),
    (b'{"dates":[{}]}', r"dates\[0\].games"),
    (b'{"dates":[{"games":null}]}', r"dates\[0\].games"),
    (b'{"dates":[{"games":"bad"}]}', r"dates\[0\].games"),
    (b'{"dates":[{"games":[{"gameType":"S"},42]}]}', r"dates\[0\].games\[1\]"),
    (b'{"dates":[{"games":[{"gameType":"R"}]},false]}', r"dates\[1\]"),
])
def test_malformed_schedule_reports_location(monkeypatch, payload, location):
    monkeypatch.setattr(scheduler, "retry_urlopen", lambda *a, **k: BytesIO(payload))
    with pytest.raises(ValueError, match=location):
        scheduler.fetch_schedule(DAY)


@pytest.mark.parametrize("body,expected", [
    ({"dates": []}, []),
    ({"dates": [{"games": []}]}, []),
    ({"dates": [{"games": [{"gameType": "S"}, {"gameType": "R", "gamePk": 101}]}]}, [101]),
])
def test_valid_schedule_lists_and_filtering(monkeypatch, body, expected):
    monkeypatch.setattr(scheduler, "retry_urlopen",
                        lambda *a, **k: BytesIO(json.dumps(body).encode()))
    assert [game["gamePk"] for game in scheduler.fetch_schedule(DAY)] == expected


@pytest.mark.parametrize("failure", [KeyboardInterrupt, SystemExit, URLError])
def test_interrupted_read_and_restart_refetch_original_slate(review_world, failure):
    day = "2027-12-31"
    with review_world.offline(review_world.instant(day, 23, 59)) as (world, patch):
        requests = []

        class InterruptedResponse:
            def read(self):
                world.now += timedelta(minutes=2)
                raise failure("synthetic read failure")

        def request(url, **kwargs):
            requests.append(parse_qs(urlsplit(url).query)["date"])
            return InterruptedResponse() if len(requests) == 1 else BytesIO(b'{"dates":[]}')

        patch.setattr(scheduler, "retry_urlopen", request)
        with pytest.raises(failure, match="synthetic read failure"):
            scheduler.run_day(day, world.config)
        assert scheduler.load_state(day, world.picks) is None
        scheduler.run_day(day, world.config)
        assert requests == [[day], [day], ["2028-01-01"]]
        assert not world.sends("dm")
