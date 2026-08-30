"""late_delivery — EOD backstop for the delivery-cutoff guard (2026-08-30)."""
import json
from datetime import date

from bts.picks import DailyPick, Pick, save_pick

D = date(2026, 8, 30)


def _delivered(tmp_path, delivered_at, dd=False):
    pick = Pick(batter_name="Kwan", batter_id=1, team="CLE", lineup_position=1, pitcher_name="L",
                pitcher_id=2, p_game_hit=0.75, flags=[], projected_lineup=False, game_pk=100,
                game_time="2026-08-30T17:40:00Z")
    double = None
    if dd:
        double = Pick(batter_name="McNeil", batter_id=2, team="ATH", lineup_position=2,
                      pitcher_name="B", pitcher_id=3, p_game_hit=0.74, flags=[],
                      projected_lineup=False, game_pk=200, game_time="2026-08-30T20:05:00Z")
    d = DailyPick(date="2026-08-30", run_time="x", pick=pick, double_down=double, runner_up=None,
                  notification_sent=True, notification_id="m", notification_channel="bluesky_dm",
                  delivered_at=delivered_at)
    save_pick(d, tmp_path)


def _state(tmp_path, **extra):
    (tmp_path / "2026-08-30").mkdir(exist_ok=True)
    body = {"date": "2026-08-30", "schedule_fetched_at": "x", "games": [], "confirmed_game_pks": [],
            "runs_completed": [], "pick_locked": False, "pick_locked_at": None,
            "result_status": None, "next_wakeup": None}
    body.update(extra)
    (tmp_path / "2026-08-30" / "scheduler_state.json").write_text(json.dumps(body))


def test_after_cutoff_is_critical(tmp_path):
    from bts.health import late_delivery
    _delivered(tmp_path, "2026-08-30T13:36:14-04:00")
    a = late_delivery.check(tmp_path, today=D)
    assert len(a) == 1 and a[0].level == "CRITICAL" and a[0].source == "late_delivery"
    assert "13:35" in a[0].message and "13:36" in a[0].message


def test_at_cutoff_is_critical(tmp_path):
    from bts.health import late_delivery
    _delivered(tmp_path, "2026-08-30T13:35:00-04:00")
    assert late_delivery.check(tmp_path, today=D)[0].level == "CRITICAL"


def test_inside_reserve_is_warn(tmp_path):
    from bts.health import late_delivery
    _delivered(tmp_path, "2026-08-30T13:30:00-04:00")
    a = late_delivery.check(tmp_path, today=D)
    assert len(a) == 1 and a[0].level == "WARN" and "5 min" in a[0].message


def test_comfortable_delivery_is_silent(tmp_path):
    from bts.health import late_delivery
    _delivered(tmp_path, "2026-08-30T12:50:00-04:00")
    assert late_delivery.check(tmp_path, today=D) == []


def test_utc_timestamp_and_earlier_double_down(tmp_path):
    """delivered_at may be UTC; the cutoff keys on the EARLIEST slot."""
    from bts.health import late_delivery
    _delivered(tmp_path, "2026-08-30T17:36:00Z", dd=True)      # 13:36 ET; primary game 13:40
    assert late_delivery.check(tmp_path, today=D)[0].level == "CRITICAL"


def test_falls_back_to_pick_locked_at(tmp_path):
    from bts.health import late_delivery
    _delivered(tmp_path, None)
    _state(tmp_path, pick_locked=True, pick_locked_at="2026-08-30T13:36:14-04:00")
    assert late_delivery.check(tmp_path, today=D)[0].level == "CRITICAL"


def test_no_timestamp_is_silent(tmp_path):
    from bts.health import late_delivery
    _delivered(tmp_path, None)
    assert late_delivery.check(tmp_path, today=D) == []


def test_refusal_on_state_is_critical(tmp_path):
    from bts.health import late_delivery
    _state(tmp_path, delivery_refusals=[{
        "at": "2026-08-30T13:36:14-04:00", "label": "lineup", "batter": "Kwan",
        "double_down": None, "cutoff_et": "2026-08-30T13:35:00-04:00", "late_min": 1.2,
        "archive": "refused_delivery_x.json"}])
    a = late_delivery.check(tmp_path, today=D)
    assert len(a) == 1 and a[0].level == "CRITICAL" and "refused" in a[0].message.lower()
    assert "Kwan" in a[0].message


def test_no_pick_is_silent(tmp_path):
    from bts.health import late_delivery
    assert late_delivery.check(tmp_path, today=D) == []


def test_undelivered_pick_is_silent(tmp_path):
    from bts.health import late_delivery
    pick = Pick(batter_name="Kwan", batter_id=1, team="CLE", lineup_position=1, pitcher_name="L",
                pitcher_id=2, p_game_hit=0.75, flags=[], projected_lineup=False, game_pk=100,
                game_time="2026-08-30T17:40:00Z")
    save_pick(DailyPick(date="2026-08-30", run_time="x", pick=pick, double_down=None,
                        runner_up=None), tmp_path)
    assert late_delivery.check(tmp_path, today=D) == []


def test_registered_in_runner():
    from bts.health import runner
    import inspect
    assert "late_delivery" in inspect.getsource(runner)
