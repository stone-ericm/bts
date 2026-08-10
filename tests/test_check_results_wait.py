"""check-results: shadow reconciliation on every exit path, --wait-deadline-et,
and the stale-scoring guard.

Regression suite for the 2026-07-10 stranded shadow: SF@COL wasn't final at the
01:00 ET cron, two exit paths returned before attempting shadow reconciliation,
and no later nightly run revisits an old date. Fix shape (2026-08-09
wait-not-sweep design + Codex r2): always attempt shadow reconciliation before
returning; retry in-process on a capped cadence until the grader itself reports
everything settled or a hard deadline; and refuse streak-bearing scoring for
dates older than STALE_SCORING_MAX_AGE_DAYS without --allow-stale-scoring —
update_streak applies results against CURRENT streak state, so out-of-order
historical grading corrupts the streak.
"""

from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

from click.testing import CliRunner

import bts.cli as cli_mod
from bts.cli import cli
from bts.picks import (
    Pick, DailyPick, save_pick, save_streak,
    load_pick, load_shadow_pick, load_streak, save_shadow_pick,
)
from bts.shadow_eval import stamp_shadow_version

ET = ZoneInfo("America/New_York")
PROD_BATTER = 700363
SHADOW_BATTER = 600123


def _pick(batter_id, game_pk, **overrides):
    defaults = dict(
        batter_name=f"Batter {batter_id}",
        batter_id=batter_id,
        team="ATH",
        lineup_position=1,
        pitcher_name="Jose Suarez",
        pitcher_id=660761,
        p_game_hit=0.83,
        flags=[],
        projected_lineup=False,
        game_pk=game_pk,
        game_time="2026-04-01T23:10:00Z",
    )
    defaults.update(overrides)
    return Pick(**defaults)


def _daily(batter_id=PROD_BATTER, game_pk=778899, **overrides):
    defaults = dict(
        date="2026-04-01",
        run_time="2026-04-01T15:00:00+00:00",
        pick=_pick(batter_id, game_pk),
        double_down=None,
        runner_up=None,
        bluesky_posted=True,
        bluesky_uri=None,
    )
    defaults.update(overrides)
    return DailyPick(**defaults)


def _save_shadow(picks_dir, batter_id=SHADOW_BATTER, game_pk=778901):
    shadow = _daily(batter_id=batter_id, game_pk=game_pk, bluesky_posted=False)
    return save_shadow_pick(stamp_shadow_version(shadow), picks_dir)


def _by_batter(results):
    """check_hit side effect keyed on batter_id (2nd positional arg)."""
    def side_effect(game_pk, batter_id, *args, **kwargs):
        return results[batter_id]
    return side_effect


def _install_clock(monkeypatch, start="01:05", day=2):
    """Fake ET clock (2026-04-<day>) advancing by each _sleep's duration."""
    state = {"now": datetime(2026, 4, day, *map(int, start.split(":")), tzinfo=ET)}
    sleeps = []

    def fake_now():
        return state["now"]

    def fake_sleep(seconds):
        sleeps.append(seconds)
        state["now"] = state["now"] + cli_mod.timedelta(seconds=seconds)

    monkeypatch.setattr(cli_mod, "_now_et", fake_now)
    monkeypatch.setattr(cli_mod, "_sleep", fake_sleep)
    return sleeps


def _no_sleep(monkeypatch, reason="must not sleep"):
    def boom(seconds):
        raise AssertionError(reason)
    monkeypatch.setattr(cli_mod, "_sleep", boom)


def _invoke(picks_dir, *extra):
    return CliRunner().invoke(cli, [
        "check-results", "--date", "2026-04-01",
        "--picks-dir", str(picks_dir), *extra,
    ])


class TestShadowReconcileOnEveryExit:
    @patch("bts.picks.get_game_statuses_detailed", return_value={})
    @patch("bts.picks.check_hit")
    def test_no_production_pick_still_reconciles_shadow(
        self, mock_check, _statuses, tmp_path,
    ):
        picks_dir = tmp_path / "picks"
        picks_dir.mkdir()
        _save_shadow(picks_dir)
        mock_check.side_effect = _by_batter({SHADOW_BATTER: True})

        result = _invoke(picks_dir)

        assert result.exit_code == 0
        assert "No pick found" in result.output
        assert load_shadow_pick("2026-04-01", picks_dir).result == "hit"

    @patch("bts.picks.get_game_statuses_detailed", return_value={})
    @patch("bts.picks.check_hit")
    def test_production_pending_still_reconciles_shadow(
        self, mock_check, _statuses, tmp_path, monkeypatch,
    ):
        picks_dir = tmp_path / "picks"
        picks_dir.mkdir()
        save_pick(_daily(), picks_dir)
        save_streak(3, picks_dir)
        _save_shadow(picks_dir)
        _install_clock(monkeypatch)  # cron-night clock: date is 1 day old
        mock_check.side_effect = _by_batter({PROD_BATTER: None, SHADOW_BATTER: True})

        result = _invoke(picks_dir)

        assert result.exit_code == 0
        assert "WARNING" in result.output
        assert load_streak(picks_dir) == 3
        assert load_pick("2026-04-01", picks_dir).result is None
        assert load_shadow_pick("2026-04-01", picks_dir).result == "hit"

    @patch("bts.picks.get_game_statuses_detailed", return_value={})
    @patch("bts.picks.check_hit")
    def test_production_resolver_error_still_reconciles_shadow(
        self, mock_check, _statuses, tmp_path, monkeypatch,
    ):
        picks_dir = tmp_path / "picks"
        picks_dir.mkdir()
        save_pick(_daily(), picks_dir)
        save_streak(3, picks_dir)
        _save_shadow(picks_dir)
        _install_clock(monkeypatch)

        def side_effect(game_pk, batter_id, *args, **kwargs):
            if batter_id == PROD_BATTER:
                raise RuntimeError("transient API failure")
            return True
        mock_check.side_effect = side_effect

        result = _invoke(picks_dir)

        assert result.exit_code == 0
        assert load_streak(picks_dir) == 3
        assert load_shadow_pick("2026-04-01", picks_dir).result == "hit"


class TestStaleScoringGuard:
    @patch("bts.picks.get_game_statuses_detailed", return_value={})
    @patch("bts.picks.check_hit")
    def test_old_unresolved_scoreable_production_never_scores(
        self, mock_check, _statuses, tmp_path, monkeypatch,
    ):
        """THE Codex-r2 blocker regression: old date + unresolved scoreable
        production + resolvable boxscore -> shadow reconciles, streak and
        production result stay untouched, no waiting."""
        picks_dir = tmp_path / "picks"
        picks_dir.mkdir()
        save_pick(_daily(), picks_dir)
        save_streak(3, picks_dir)
        _save_shadow(picks_dir)
        _install_clock(monkeypatch, day=10)  # pick is 9 days old
        _no_sleep(monkeypatch, "stale-refused date must not wait")
        mock_check.side_effect = _by_batter({PROD_BATTER: True, SHADOW_BATTER: True})

        result = _invoke(picks_dir, "--wait-deadline-et", "06:00")

        assert result.exit_code == 0
        assert "refusing" in result.output.lower()
        assert load_streak(picks_dir) == 3
        assert load_pick("2026-04-01", picks_dir).result is None
        assert load_shadow_pick("2026-04-01", picks_dir).result == "hit"

    @patch("bts.picks.get_game_statuses_detailed", return_value={})
    @patch("bts.picks.check_hit")
    def test_allow_stale_scoring_override_scores(
        self, mock_check, _statuses, tmp_path, monkeypatch,
    ):
        picks_dir = tmp_path / "picks"
        picks_dir.mkdir()
        save_pick(_daily(), picks_dir)
        save_streak(3, picks_dir)
        _install_clock(monkeypatch, day=10)
        mock_check.side_effect = _by_batter({PROD_BATTER: True})

        result = _invoke(picks_dir, "--allow-stale-scoring")

        assert result.exit_code == 0
        assert "HIT!" in result.output
        assert load_streak(picks_dir) == 4

    @patch("bts.picks.get_game_statuses_detailed", return_value={})
    @patch("bts.picks.check_hit")
    def test_yesterday_scores_without_override(
        self, mock_check, _statuses, tmp_path, monkeypatch,
    ):
        picks_dir = tmp_path / "picks"
        picks_dir.mkdir()
        save_pick(_daily(), picks_dir)
        save_streak(3, picks_dir)
        _install_clock(monkeypatch)  # day=2: the normal cron case
        mock_check.side_effect = _by_batter({PROD_BATTER: True})

        result = _invoke(picks_dir)

        assert result.exit_code == 0
        assert "HIT!" in result.output
        assert load_streak(picks_dir) == 4


class TestWaitDeadline:
    @patch("bts.picks.get_game_statuses_detailed", return_value={})
    @patch("bts.picks.check_hit")
    def test_wait_retries_until_production_resolves(
        self, mock_check, _statuses, tmp_path, monkeypatch,
    ):
        picks_dir = tmp_path / "picks"
        picks_dir.mkdir()
        save_pick(_daily(), picks_dir)
        save_streak(3, picks_dir)
        sleeps = _install_clock(monkeypatch)
        mock_check.side_effect = [None, True]  # attempt 1 pending, attempt 2 final

        result = _invoke(picks_dir, "--wait-deadline-et", "06:00")

        assert result.exit_code == 0
        assert "HIT!" in result.output
        assert load_streak(picks_dir) == 4
        assert sleeps == [900]

    @patch("bts.picks.get_game_statuses_detailed", return_value={})
    @patch("bts.picks.check_hit")
    def test_deadline_is_a_hard_bound(
        self, mock_check, _statuses, tmp_path, monkeypatch,
    ):
        """05:50 start: the sleep is capped to the deadline (600s, not 900),
        one final attempt runs AT 06:00, and no attempt starts after it."""
        picks_dir = tmp_path / "picks"
        picks_dir.mkdir()
        save_pick(_daily(), picks_dir)
        save_streak(3, picks_dir)
        sleeps = _install_clock(monkeypatch, start="05:50")
        mock_check.return_value = None  # never resolves

        result = _invoke(picks_dir, "--wait-deadline-et", "06:00")

        assert result.exit_code == 0
        assert load_streak(picks_dir) == 3
        assert sleeps == [600]
        assert "deadline" in result.output.lower()

    @patch("bts.picks.get_game_statuses_detailed", return_value={})
    @patch("bts.picks.check_hit")
    def test_default_is_single_attempt_no_sleep(
        self, mock_check, _statuses, tmp_path, monkeypatch,
    ):
        picks_dir = tmp_path / "picks"
        picks_dir.mkdir()
        save_pick(_daily(), picks_dir)
        save_streak(3, picks_dir)
        _install_clock(monkeypatch)
        _no_sleep(monkeypatch, "must not sleep without --wait-deadline-et")
        mock_check.return_value = None

        result = _invoke(picks_dir)

        assert result.exit_code == 0
        assert "WARNING" in result.output
        assert load_streak(picks_dir) == 3

    @patch("bts.picks.get_game_statuses_detailed", return_value={})
    @patch("bts.picks.check_hit")
    def test_wait_already_past_deadline_single_attempt(
        self, mock_check, _statuses, tmp_path, monkeypatch,
    ):
        picks_dir = tmp_path / "picks"
        picks_dir.mkdir()
        save_pick(_daily(), picks_dir)
        save_streak(3, picks_dir)
        sleeps = _install_clock(monkeypatch, start="14:00")
        mock_check.return_value = None

        result = _invoke(picks_dir, "--wait-deadline-et", "06:00")

        assert result.exit_code == 0
        assert sleeps == []
        assert mock_check.call_count == 1  # exactly one attempt ran
        assert load_streak(picks_dir) == 3

    @patch("bts.picks.get_game_statuses_detailed", return_value={})
    @patch("bts.picks.check_hit")
    def test_wait_driven_by_shadow_alone(
        self, mock_check, _statuses, tmp_path, monkeypatch,
    ):
        picks_dir = tmp_path / "picks"
        picks_dir.mkdir()
        _save_shadow(picks_dir)
        sleeps = _install_clock(monkeypatch)
        mock_check.side_effect = [None, True]

        result = _invoke(picks_dir, "--wait-deadline-et", "06:00")

        assert result.exit_code == 0
        assert sleeps == [900]
        assert load_shadow_pick("2026-04-01", picks_dir).result == "hit"

    @patch("bts.picks.get_game_statuses_detailed", return_value={})
    @patch("bts.picks.check_hit")
    def test_wait_settles_immediately_when_already_resolved(
        self, mock_check, _statuses, tmp_path, monkeypatch,
    ):
        """Old-date manual runs with the flag: resolved production = settled,
        zero sleeps, streak untouched."""
        picks_dir = tmp_path / "picks"
        picks_dir.mkdir()
        daily = _daily()
        daily.result = "miss"
        save_pick(daily, picks_dir)
        save_streak(5, picks_dir)
        _install_clock(monkeypatch, day=10)
        _no_sleep(monkeypatch, "resolved date must not wait")

        result = _invoke(picks_dir, "--wait-deadline-et", "06:00")

        assert result.exit_code == 0
        assert "Already resolved" in result.output
        assert load_streak(picks_dir) == 5

    def test_wait_interval_must_be_positive(self, tmp_path):
        picks_dir = tmp_path / "picks"
        picks_dir.mkdir()

        result = _invoke(
            picks_dir, "--wait-deadline-et", "06:00", "--wait-interval-min", "0",
        )

        assert result.exit_code != 0

    @patch("bts.picks.get_game_statuses_detailed", return_value={})
    @patch("bts.picks.check_hit")
    def test_vanished_pick_file_stays_pending(
        self, mock_check, _statuses, tmp_path, monkeypatch,
    ):
        """Codex r2 #9: the fail-closed vanished-file branch must not read as
        settled — keep waiting (the file may be restored later tonight)."""
        picks_dir = tmp_path / "picks"
        picks_dir.mkdir()
        daily = _daily()
        save_pick(daily, picks_dir)
        save_streak(3, picks_dir)
        sleeps = _install_clock(monkeypatch)
        mock_check.return_value = True
        # First load (attempt head) sees the pick; the locked reload — and
        # every later load — sees nothing, as if the file vanished mid-score.
        monkeypatch.setattr(
            "bts.picks.load_pick",
            _sequential_loader([daily, None]),
        )

        result = _invoke(picks_dir, "--wait-deadline-et", "06:00")

        assert result.exit_code == 0
        assert "failing" in result.output.lower()
        assert sleeps == [900]
        assert "No pick found" in result.output  # attempt 2 sees no file
        assert load_streak(picks_dir) == 3


def _sequential_loader(values):
    """load_pick replacement yielding `values` in order, then repeating the last."""
    state = {"i": 0}

    def loader(date, picks_dir):
        idx = min(state["i"], len(values) - 1)
        state["i"] += 1
        return values[idx]
    return loader
