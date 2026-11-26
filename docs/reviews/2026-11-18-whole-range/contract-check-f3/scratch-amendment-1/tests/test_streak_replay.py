"""Forward streak replay with saver state machine (audit D4, Codex-validated).

reconcile recomputes the current streak from scratch. The old BACKWARD walk
couldn't replay the saver — whether a miss was forgiven depends on the forward
streak AT the miss (must be in [10,15]) and whether the saver was already used.
Forward replay is the sound approach. These encode Codex's overcount traps and
the fail-closed contract on incomplete history.
"""
import json

from bts.picks import _apply_streak_day, _replay_season_streak, Pick, DailyPick, save_pick


def _replay(seq, streak=0, saver=True):
    for is_hit, inc in seq:
        streak, saver = _apply_streak_day(streak, saver, is_hit, inc)
    return streak, saver


# --- state machine: Codex's overcount traps ---

def test_saver_consumed_then_later_miss_resets():
    # 12 hits, miss(saver@12), 3 hits ->15, miss(reset, saver gone), 4 hits -> 4
    seq = [(True, 1)] * 12 + [(False, 0)] + [(True, 1)] * 3 + [(False, 0)] + [(True, 1)] * 4
    assert _replay(seq) == (4, False)


def test_miss_below_window_does_not_consume_saver():
    seq = [(True, 1)] * 9 + [(False, 0)] + [(True, 1)] * 3   # miss@9 -> reset, saver intact
    assert _replay(seq) == (3, True)


def test_miss_at_upper_boundary_is_eligible():
    seq = [(True, 1)] * 15 + [(False, 0)] + [(True, 1)] * 2  # miss@15 eligible -> saver
    assert _replay(seq) == (17, False)


def test_miss_above_window_not_eligible():
    seq = [(True, 1)] * 16 + [(False, 0)] + [(True, 1)] * 2  # miss@16 -> reset, saver intact
    assert _replay(seq) == (2, True)


def test_double_down_increment():
    assert _apply_streak_day(0, True, True, 2) == (2, True)


# --- file replay ---

def _save_hit(picks_dir, d, double_down=False):
    def _pk(gp):
        return Pick(batter_name="B", batter_id=1, team="X", lineup_position=1,
                    pitcher_name="P", pitcher_id=2, p_game_hit=0.7, flags=[],
                    projected_lineup=False, game_pk=gp, game_time=f"{d}T23:00:00Z")
    daily = DailyPick(date=d, run_time=f"{d}T15:00:00+00:00", pick=_pk(100),
                      double_down=_pk(200) if double_down else None, runner_up=None,
                      result="hit")
    save_pick(daily, picks_dir)


def test_replay_simple_hits(tmp_path):
    for d in ("2026-05-01", "2026-05-02", "2026-05-03"):
        _save_hit(tmp_path, d)
    assert _replay_season_streak(tmp_path, 2026, "2026-06-10") == (3, True)


def test_replay_double_down_counts_two(tmp_path):
    _save_hit(tmp_path, "2026-05-01", double_down=True)  # +2
    assert _replay_season_streak(tmp_path, 2026, "2026-06-10") == (2, True)


def test_replay_fail_closed_on_unresolved_past_day(tmp_path):
    _save_hit(tmp_path, "2026-05-01")
    (tmp_path / "2026-05-02.json").write_text(json.dumps({"date": "2026-05-02", "result": None}))
    assert _replay_season_streak(tmp_path, 2026, "2026-06-10") is None


def test_replay_skips_today_and_later(tmp_path):
    _save_hit(tmp_path, "2026-05-01")
    # a future preview pick (result None) dated >= today must NOT fail-close the replay
    (tmp_path / "2026-06-11.json").write_text(json.dumps({"date": "2026-06-11", "result": None}))
    assert _replay_season_streak(tmp_path, 2026, "2026-06-10") == (1, True)
