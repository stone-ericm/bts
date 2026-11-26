"""The 90 pick shapes and seven histories per shape from the round-one C3 review."""

from dataclasses import dataclass
from datetime import date, timedelta
import json

import pytest

from tests.test_wr_c3v_reconcile import RUN, base_modules, compare_reconcile, deadline


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def load(path):
    return json.loads(path.read_text())


def slot(day, game=101):
    return dict(batter_name="Synthetic batter", batter_id=100001, team="PIT", lineup_position=1,
                pitcher_name="Synthetic pitcher", pitcher_id=100002, p_game_hit=0.8, flags=[],
                projected_lineup=False, game_pk=game, game_time=f"{day}T23:00:00Z")


def pick_data(day, result=None, **extra):
    return dict(date=day, run_time=f"{day}T19:00:00Z", pick=slot(day), double_down=None,
                runner_up=None, result=result) | extra


def decision(root, day, *, scoreable, status=None, schema=None):
    from bts.daily_decision import decision_path, write_decision

    status = status or ("delivered" if scoreable else "not_applicable")
    record = write_decision(day, root, action="single" if scoreable else "skip", source="probe",
                            delivery_status=status, scoreable=scoreable, now=RUN)
    assert record is not None
    path = decision_path(day, root)
    if schema is not None:
        dump(path, record | {"schema_version": schema})
    return path


def seed_streak(root, *, streak=21, saver=False, markers=(), season=2027):
    data = dict(streak=streak, saver_available=saver, applied_dates=list(markers),
                updated="2027-04-20T05:00:00+00:00")
    return dump(root / "streak.json", data if season is None else data | {"season": season})


def stagger(day, offset):
    return (date.fromisoformat(day) + timedelta(days=offset)).isoformat()


# The regeneration writer at 11059dd1, retained as a fixture.
def early_pick_file(record) -> dict:
    """Convert a regenerated HistoricalPickRecord to a pick file shape.

    Note: some fields (batter_id, pitcher info, p_game_hit) cannot be
    recovered from Bluesky alone and are left as None. A follow-up
    pass could use the MLB API to backfill batter_id from name+team+date.
    """
    return {
        "date": record.date,
        "run_time": f"{record.date}T12:00:00+00:00",
        "pick": {
            "batter_name": record.batter_name,
            "batter_id": None,
            "team": record.team,
            "pitcher_name": None,
            "pitcher_id": None,
            "game_pk": None,
            "game_time": None,
            "p_game_hit": None,
            "p_hit_pa": None,
            "projected_lineup": False,
        },
        "double_down": {
            "batter_name": record.double_down_batter,
            "batter_id": None,
            "team": record.double_down_team,
            "pitcher_name": None, "pitcher_id": None,
            "game_pk": None, "game_time": None,
            "p_game_hit": None, "p_hit_pa": None,
            "projected_lineup": False,
        } if record.is_double_down else None,
        "runner_up": None,
        "bluesky_posted": True,
        "bluesky_uri": record.bluesky_uri,
        "result": record.result,
    }


TODAY = "2026-08-20"
EDGE = "2026-08-12"
ORDINARY = (
    (1, 1, 1), (2, 1, "V", 2, 1), (2,) * 5 + ("M", 1, 2, "M", 1),
    (2,) * 7 + (1, "M", 1), (2,) * 8 + ("M", 2),
    ("M", 2, "V", 2, "M", 1), (2,) * 7 + ("M", "V", 1, "M"),
)


@dataclass(frozen=True)
class Case:
    name: str
    transform: object


def cases():
    from bts.state.regenerate import HistoricalPickRecord, _hist_to_pick_file, _record_to_pick_file

    def d(fields, result="hit"):
        return lambda day, root: pick_data(day, result) | fields

    def s(fields, result="hit", *, second=False):
        key = "double_down" if second else "pick"
        return lambda day, root: pick_data(day, result) | {key: slot(day, 102 if second else 101) | fields}

    def omit(field, result="hit", *, nested=False):
        def edit(day, root):
            data = pick_data(day, result)
            del (data["pick"] if nested else data)[field]
            return data
        return edit

    def decided(*, scoreable, result=None, status=None, missing=False, schema=None):
        def edit(day, root):
            decision(root, day, scoreable=scoreable, status=status, schema=schema)
            return None if missing else pick_data(day, result)
        return edit

    def bad_decision(edit, result="hit"):
        def mutate(day, root):
            path = decision(root, day, scoreable=True)
            record = json.loads(path.read_text())
            path.write_text(edit(record))
            return pick_data(day, result)
        return mutate

    def marker(tag, marker_day):
        def edit(day, root):
            seed_streak(root, markers=(marker_day,), season=tag)
            return pick_data(day)
        return edit

    def legacy_record(result, writer):
        def edit(day, root):
            record = HistoricalPickRecord(day, "Synthetic batter", "PIT", False, None, None,
                                          "at://synthetic", result, None)
            return writer(record)
        return edit

    def snapshot(fields):
        def edit(day, root):
            # The export does not coerce typed identifiers. Regeneration does not coerce non-null IDs.
            from bts.state.export import _pick_to_historical

            original = pick_data(day, "hit", pick=slot(day) | fields)
            return _hist_to_pick_file(_pick_to_historical(original))
        return edit

    def historical_snapshot(day, root):
        return _hist_to_pick_file({"date": day, "result": "miss", "pick": None})

    def sparse_snapshot(day, root):
        primary = {key: value for key, value in slot(day).items() if key not in
                   {"lineup_position", "flags", "projected_lineup", "pitcher_id", "game_time"}}
        return _hist_to_pick_file({"date": day, "result": "miss", "pick": primary})

    def partial(day, root):
        return pick_data(day, "hit", double_down=slot(day, 102),
                         slot_results={"pick": "void", "double_down": "hit"})

    def snapshot_partial(day, root):
        from bts.state.export import _pick_to_historical

        return _hist_to_pick_file(_pick_to_historical(partial(day, root)))

    def refused(day, root):
        from bts.contest_state import DecisionStreakState
        from bts.picks import chosen_on_record

        state = DecisionStreakState(0, True, True, "model", "fresh", 0, True)
        return pick_data(day, chosen_on=chosen_on_record(state, required=False))

    return [
        Case("normal", d({})),
        Case("miss", d({}, "miss")),
        Case("void", d({}, "void")),
        Case("missing_pitcher_team", d({})),
        Case("nullable_game_pk", s({"game_pk": None})),
        Case("extra_top_key", d({"streak": 21, "other": []})),
        Case("extra_slot_key_miss", s({"unmapped": 1}, "miss")),
        Case("extra_slot_key_hit", s({"unmapped": 1})),
        Case("minimal_miss", lambda day, root: {"result": "miss"}),
        Case("minimal_void", lambda day, root: {"result": "void"}),
        Case("missing_top_date_miss", omit("date", "miss")),
        Case("missing_top_runtime_miss", omit("run_time", "miss")),
        Case("missing_top_runner_up_miss", omit("runner_up", "miss")),
        Case("missing_top_double_down_void", omit("double_down", "void")),
        Case("missing_slot_game_pk_void", omit("game_pk", "void", nested=True)),
        Case("missing_slot_name_miss", omit("batter_name", "miss", nested=True)),
        Case("missing_slot_team_miss", omit("team", "miss", nested=True)),
        Case("missing_slot_id_miss", omit("batter_id", "miss", nested=True)),
        Case("pick_is_null_void", d({"pick": None}, "void")),
        Case("pick_is_array_miss", d({"pick": []}, "miss")),
        Case("empty_array_double_hit", d({"double_down": []})),
        Case("empty_object_double_hit", d({"double_down": {}})),
        Case("empty_string_double_hit", d({"double_down": ""})),
        Case("primary_id_numeric_string", snapshot({"batter_id": "100001"})),
        Case("game_id_numeric_string", snapshot({"game_pk": "101"})),
        Case("double_id_numeric_string", s({"batter_id": "100001"}, second=True)),
        Case("double_game_id_numeric_string", s({"game_pk": "102"}, second=True)),
        Case("primary_batter_null_existing", s({"batter_id": None})),
        Case("primary_batter_float", s({"batter_id": 100001.0})),
        Case("primary_game_float", s({"game_pk": 101.0})),
        Case("primary_batter_bool", s({"batter_id": True})),
        Case("primary_name_null", s({"batter_name": None})),
        Case("primary_team_null", s({"team": None})),
        Case("double_name_null", s({"batter_name": None}, second=True)),
        Case("double_team_numeric", s({"team": 101}, second=True)),
        Case("runtime_null", d({"run_time": None})),
        Case("runtime_invalid_string", d({"run_time": "not-a-time"})),
        Case("bluesky_int", d({"bluesky_posted": 1})),
        Case("notification_sent_string", d({"notification_sent": "true"})),
        Case("notification_attempt_null", d({"delivery_attempted": None})),
        Case("notification_id_int", d({"notification_id": 1234})),
        Case("notification_id_blank", d({"notification_id": ""})),
        Case("slot_results_non_dict_miss", d({"slot_results": []}, "miss")),
        Case("slot_results_non_dict_hit", d({"slot_results": []})),
        Case("slot_results_null_leg", d({"slot_results": {"pick": "hit", "double_down": None}})),
        Case("slot_results_pending_leg", d({"slot_results": {"pick": "hit", "double_down": "suspended"}})),
        Case("slot_results_extra_leg", d({"slot_results": {"pick": "hit", "unknown": "hit"}})),
        Case("slot_results_empty_hit", d({"slot_results": {}})),
        Case("date_filename_mismatch", d({"date": "2026-01-01"})),
        Case("utf8_bom",
             lambda day, root: "\ufeff" + json.dumps(pick_data(day, "hit"))),
        Case("non_utf8", lambda day, root: b"\xff\xfe"),
        Case("json_array", lambda day, root: [pick_data(day, "miss")]),
        Case("json_null", lambda day, root: "null"),
        Case("json_truncated", lambda day, root: "{"),
        Case("unknown_result", d({"result": "mystery"})),
        Case("today_unknown_result", d({"result": "mystery"})),
        Case("today_committed_unresolved", d({"bluesky_posted": True}, None)),
        Case("today_terminal", d({})),
        Case("refused_preview", refused),
        Case("uncommitted_unresolved", d({"result": "unresolved"})),
        Case("uncommitted_suspended", d({"result": "suspended"})),
        Case("delivered_suspended", d({"bluesky_posted": True, "result": "suspended"})),
        Case("delivered_unresolved", d({"notification_sent": True, "notification_id": "receipt"}, "unresolved")),
        Case("bare_attempt", d({"delivery_attempted": True}, None)),
        Case("private_pending", decided(scoreable=True, status="private_locked")),
        Case("locked_unconfirmed", decided(scoreable=True, status="locked_unconfirmed")),
        Case("terminal_private", decided(scoreable=True, result="hit", status="private_locked")),
        Case("skip_leftover", decided(scoreable=False)),
        Case("skip_without_root", decided(scoreable=False, missing=True)),
        Case("committed_without_root", decided(scoreable=True, missing=True)),
        Case("old_decision_v1", decided(scoreable=True, result="hit", schema="bts_daily_decision_v1")),
        Case("old_decision_v2", decided(scoreable=True, result="hit", schema="bts_daily_decision_v2")),
        Case("decision_future_schema", bad_decision(lambda r: json.dumps(r | {"schema_version": "v4"}))),
        Case("decision_date_mismatch", bad_decision(lambda r: json.dumps(r | {"date": "2026-08-11"}))),
        Case("decision_scoreable_string", bad_decision(lambda r: json.dumps(r | {"scoreable": "true"}))),
        Case("decision_truncated", bad_decision(lambda r: "{")),
        Case("past_marker_matched", marker(2026, EDGE)),
        Case("past_marker_tagless", marker(None, EDGE)),
        Case("past_marker_foreign", marker(2025, EDGE)),
        Case("today_marker_matched", marker(2026, TODAY)),
        Case("today_marker_tagless", marker(None, TODAY)),
        Case("today_marker_foreign", marker(2025, TODAY)),
        Case("double_partial_void", partial),
        Case("legacy_double_hit", lambda day, root: pick_data(day, "hit", double_down=slot(day, 102))),
        Case("regenerated_current", legacy_record("hit", _record_to_pick_file)),
        Case("regenerated_partial_void", snapshot_partial),
        Case("regenerated_early_miss", legacy_record("miss", early_pick_file)),
        Case("regenerated_early_hit", legacy_record("hit", early_pick_file)),
        Case("regenerated_historical_null", historical_snapshot),
        Case("regenerated_historical_missing_pitcher_id", sparse_snapshot),
    ]


def ordinary(root, day, event):
    result = {"V": "void", "M": "miss"}.get(event, "hit")
    data = pick_data(day, result)
    if event == 2:
        data |= {"double_down": slot(day, 102), "slot_results": {"pick": "hit", "double_down": "hit"}}
    dump(root / f"{day}.json", data)


def build_history(picks, case, index):
    ordinary_events = ORDINARY[index]
    for offset, event in enumerate(ordinary_events):
        ordinary(picks, stagger("2026-07-01", offset), event)
    day = TODAY if case.name.startswith("today_") else EDGE
    edge = case.transform(day, picks)
    file = picks / f"{day}.json"
    if isinstance(edge, str):
        file.write_text(edge)
    elif isinstance(edge, bytes):
        file.write_bytes(edge)
    elif edge is not None:
        dump(file, edge)
    if day < TODAY and index % 2:
        ordinary(picks, TODAY, 1)
    ordinary(picks, "2025-12-31", 2)
    ordinary(picks, "2026-08-21", 2)
    ordinary(picks, "2027-01-01", 2)
    (picks / "2026-08-11.shadow.json").write_text("{")
    dump(picks / "archive" / "2026-08-10.json", pick_data("2026-08-10", "miss"))



@pytest.mark.parametrize("index", range(len(ORDINARY)))
@pytest.mark.parametrize("case", cases(), ids=lambda case: case.name)
def test_review_corpus(tmp_path, monkeypatch, base_modules, case, index):
    from datetime import datetime
    from tests.test_wr_c3v_reconcile import ET

    root = tmp_path / "current" / "picks"
    root.mkdir(parents=True)
    seed_streak(root, season=2026)
    build_history(root, case, index)
    compare_reconcile(root, monkeypatch, base_modules, clock=lambda: datetime(2026, 8, 20, 2, tzinfo=ET))
