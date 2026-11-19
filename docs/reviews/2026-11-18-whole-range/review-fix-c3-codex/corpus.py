"""Generate mixed whole-season histories, compare revisions, and audit the forward fold."""

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
import sys
from types import ModuleType

from probe_support import BASE, HERE, HEAD, archive, decision, dump, offline, pick_data
from probe_support import python, scratch, seed_streak, slot, stagger


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
    event: int | str | None
    classification: str
    origin: str
    transform: object


def cases(legacy):
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

    synthetic = "synthetic only; no normal writer located"
    legacy_parser = "base load_pick permits it; no normal producer located"
    inherited = "current export/regenerate preserve non-null value; original source unverified"
    return [
        Case("normal", 1, "=", "save_pick", d({})),
        Case("miss", "M", "=", "save_pick", d({}, "miss")),
        Case("void", "V", "=", "save_pick", d({}, "void")),
        Case("missing_pitcher_team", 1, "=", "pre-pitcher_team writer", d({})),
        Case("nullable_game_pk", 1, "=", "I-008 actual 3/29 and 3/30 shape", s({"game_pk": None})),
        Case("extra_top_key", 1, "=", "historical DailyPick.streak plus unknown metadata", d({"streak": 21, "other": []})),
        Case("extra_slot_key_miss", None, "b", synthetic, s({"unmapped": 1}, "miss")),
        Case("extra_slot_key_hit", None, "b", synthetic, s({"unmapped": 1})),
        Case("minimal_miss", None, "b", synthetic, lambda day, root: {"result": "miss"}),
        Case("minimal_void", None, "b", synthetic, lambda day, root: {"result": "void"}),
        Case("missing_top_date_miss", None, "b", synthetic, omit("date", "miss")),
        Case("missing_top_runtime_miss", None, "b", synthetic, omit("run_time", "miss")),
        Case("missing_top_runner_up_miss", None, "b", synthetic, omit("runner_up", "miss")),
        Case("missing_top_double_down_void", None, "b", synthetic, omit("double_down", "void")),
        Case("missing_slot_game_pk_void", None, "b", synthetic, omit("game_pk", "void", nested=True)),
        Case("missing_slot_name_miss", None, "b", synthetic, omit("batter_name", "miss", nested=True)),
        Case("missing_slot_team_miss", None, "b", synthetic, omit("team", "miss", nested=True)),
        Case("missing_slot_id_miss", None, "b", synthetic, omit("batter_id", "miss", nested=True)),
        Case("pick_is_null_void", None, "b", "regenerate passes it through if snapshot has null primary", d({"pick": None}, "void")),
        Case("pick_is_array_miss", None, "b", synthetic, d({"pick": []}, "miss")),
        Case("empty_array_double_hit", None, "b", legacy_parser, d({"double_down": []})),
        Case("empty_object_double_hit", 1, "=", legacy_parser, d({"double_down": {}})),
        Case("empty_string_double_hit", None, "b", legacy_parser, d({"double_down": ""})),
        Case("primary_id_numeric_string", None, "c", inherited, snapshot({"batter_id": "100001"})),
        Case("game_id_numeric_string", None, "c", inherited, snapshot({"game_pk": "101"})),
        Case("double_id_numeric_string", None, "c", legacy_parser, s({"batter_id": "100001"}, second=True)),
        Case("double_game_id_numeric_string", None, "c", legacy_parser, s({"game_pk": "102"}, second=True)),
        Case("primary_batter_null_existing", None, "b", legacy_parser, s({"batter_id": None})),
        Case("primary_batter_float", None, "b", legacy_parser, s({"batter_id": 100001.0})),
        Case("primary_game_float", None, "b", legacy_parser, s({"game_pk": 101.0})),
        Case("primary_batter_bool", None, "b", legacy_parser, s({"batter_id": True})),
        Case("primary_name_null", None, "b", legacy_parser, s({"batter_name": None})),
        Case("primary_team_null", None, "b", legacy_parser, s({"team": None})),
        Case("double_name_null", None, "b", legacy_parser, s({"batter_name": None}, second=True)),
        Case("double_team_numeric", None, "b", legacy_parser, s({"team": 101}, second=True)),
        Case("runtime_null", None, "b", legacy_parser, d({"run_time": None})),
        Case("runtime_invalid_string", 1, "=", legacy_parser, d({"run_time": "not-a-time"})),
        Case("bluesky_int", None, "b", legacy_parser, d({"bluesky_posted": 1})),
        Case("notification_sent_string", None, "b", legacy_parser, d({"notification_sent": "true"})),
        Case("notification_attempt_null", None, "b", legacy_parser, d({"delivery_attempted": None})),
        Case("notification_id_int", None, "b", "transport return written verbatim; provider declares string", d({"notification_id": 1234})),
        Case("notification_id_blank", 1, "=", legacy_parser, d({"notification_id": ""})),
        Case("slot_results_non_dict_miss", None, "b", synthetic, d({"slot_results": []}, "miss")),
        Case("slot_results_non_dict_hit", None, "b", synthetic, d({"slot_results": []})),
        Case("slot_results_null_leg", None, "b", legacy_parser, d({"slot_results": {"pick": "hit", "double_down": None}})),
        Case("slot_results_pending_leg", None, "b", legacy_parser, d({"slot_results": {"pick": "hit", "double_down": "suspended"}})),
        Case("slot_results_extra_leg", 2, "=", legacy_parser, d({"slot_results": {"pick": "hit", "unknown": "hit"}})),
        Case("slot_results_empty_hit", 0, "=", legacy_parser, d({"slot_results": {}})),
        Case("date_filename_mismatch", None, "b", "wrong filename or hand-edit; save_pick derives name", d({"date": "2026-01-01"})),
        Case("utf8_bom", None, "b", "no emitter adds BOM; I-046 concerns non-UTF-8/AppleDouble",
             lambda day, root: "\ufeff" + json.dumps(pick_data(day, "hit"))),
        Case("non_utf8", None, "b", "I-046 reported unreadable paths", lambda day, root: b"\xff\xfe"),
        Case("json_array", None, "b", synthetic, lambda day, root: [pick_data(day, "miss")]),
        Case("json_null", None, "b", synthetic, lambda day, root: "null"),
        Case("json_truncated", None, "b", "possible pre-atomic save crash", lambda day, root: "{"),
        Case("unknown_result", None, "b", synthetic, d({"result": "mystery"})),
        Case("today_unknown_result", None, "b", synthetic, d({"result": "mystery"})),
        Case("today_committed_unresolved", "S", "=", "normal scorer pending before cutoff", d({"bluesky_posted": True}, None)),
        Case("today_terminal", 1, "=", "normal scored today", d({})),
        Case("refused_preview", "S", "a", "W14/W23, real preview metadata", refused),
        Case("uncommitted_unresolved", "S", "a", "scheduler unresolved", d({"result": "unresolved"})),
        Case("uncommitted_suspended", "S", "a", "recognized daily result", d({"result": "suspended"})),
        Case("delivered_suspended", None, "b", "scheduler suspended", d({"bluesky_posted": True, "result": "suspended"})),
        Case("delivered_unresolved", None, "b", "scheduler pending", d({"notification_sent": True, "notification_id": "receipt"}, "unresolved")),
        Case("bare_attempt", None, "b", "scheduler crash before confirmation", d({"delivery_attempted": True}, None)),
        Case("private_pending", None, "b", "scheduler private lock", decided(scoreable=True, status="private_locked")),
        Case("locked_unconfirmed", None, "b", "scheduler restart", decided(scoreable=True, status="locked_unconfirmed")),
        Case("terminal_private", 1, "=", "scheduler private lock", decided(scoreable=True, result="hit", status="private_locked")),
        Case("skip_leftover", "S", "a", "scheduler final skip", decided(scoreable=False)),
        Case("skip_without_root", "S", "=", "scheduler final skip", decided(scoreable=False, missing=True)),
        Case("committed_without_root", None, "b", "decision survives pick loss", decided(scoreable=True, missing=True)),
        Case("old_decision_v1", 1, "=", "v1 decision", decided(scoreable=True, result="hit", schema="bts_daily_decision_v1")),
        Case("old_decision_v2", 1, "=", "v2 decision", decided(scoreable=True, result="hit", schema="bts_daily_decision_v2")),
        Case("decision_future_schema", None, "b", synthetic, bad_decision(lambda r: json.dumps(r | {"schema_version": "v4"}))),
        Case("decision_date_mismatch", None, "b", synthetic, bad_decision(lambda r: json.dumps(r | {"date": "2026-08-11"}))),
        Case("decision_scoreable_string", None, "b", synthetic, bad_decision(lambda r: json.dumps(r | {"scoreable": "true"}))),
        Case("decision_truncated", None, "b", synthetic, bad_decision(lambda r: "{")),
        Case("past_marker_matched", None, "b", "scorer crash", marker(2026, EDGE)),
        Case("past_marker_tagless", None, "b", "legacy scorer crash", marker(None, EDGE)),
        Case("past_marker_foreign", "S", "a", "foreign season tag", marker(2025, EDGE)),
        Case("today_marker_matched", None, "b", "scorer crash", marker(2026, TODAY)),
        Case("today_marker_tagless", None, "b", "legacy scorer crash", marker(None, TODAY)),
        Case("today_marker_foreign", "S", "a", "foreign season tag; C3 discards the marker", marker(2025, TODAY)),
        Case("double_partial_void", 1, "=", "normal grading partial-void file", partial),
        Case("legacy_double_hit", 2, "=", "pre-slot_results", lambda day, root: pick_data(day, "hit", double_down=slot(day, 102))),
        Case("regenerated_current", 1, "=", "current _record_to_pick_file", legacy_record("hit", _record_to_pick_file)),
        Case("regenerated_partial_void", 2, "=", "current export/regenerate drops slot_results", snapshot_partial),
        Case("regenerated_early_miss", None, "c", "11059dd1 _record_to_pick_file", legacy_record("miss", legacy._record_to_pick_file)),
        Case("regenerated_early_hit", None, "b", "11059dd1 _record_to_pick_file; old hit loader already throws", legacy_record("hit", legacy._record_to_pick_file)),
        Case("regenerated_historical_null", None, "b", "current _hist_to_pick_file can pass null pick", historical_snapshot),
        Case("regenerated_historical_missing_pitcher_id", None, "b", "current _hist_to_pick_file does not backfill missing pitcher_id; source unverified", sparse_snapshot),
    ]


def reference_fold(events):
    counter = 0
    has_saver = True
    for event in events:
        if event is None:
            return None
        if event == "M":
            if 10 <= counter <= 15 and has_saver:
                has_saver = False
            else:
                counter = 0
        elif event not in ("V", "S"):
            counter += event
    return [counter, has_saver]


def ordinary(root, day, event):
    result = {"V": "void", "M": "miss"}.get(event, "hit")
    data = pick_data(day, result)
    if event == 2:
        data |= {"double_down": slot(day, 102), "slot_results": {"pick": "hit", "double_down": "hit"}}
    dump(root / f"{day}.json", data)


def build(root, legacy):
    for case in cases(legacy):
        for index, ordinary_events in enumerate(ORDINARY):
            name = f"{case.name}-{index}"
            picks = root / name
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
            events = (*ordinary_events, case.event)
            if day < TODAY and index % 2:
                ordinary(picks, TODAY, 1)
                events += (1,)
            # These must never affect the season fold.
            ordinary(picks, "2025-12-31", 2)
            ordinary(picks, "2026-08-21", 2)
            ordinary(picks, "2027-01-01", 2)
            (picks / "2026-08-11.shadow.json").write_text("{")
            dump(picks / "archive" / "2026-08-10.json", pick_data("2026-08-10", "miss"))
            yield name, case, reference_fold(events)


def replay_all(root):
    from bts import picks

    print(f"SOURCE {picks.__file__}", file=sys.stderr)
    out = {}
    for season in sorted(root.iterdir()):
        try:
            out[season.name] = picks._replay_season_streak(season, 2026, TODAY)
        except Exception as error:
            out[season.name] = {"raised": type(error).__name__}
    return out


def main():
    with scratch("corpus") as work, offline():
        work = Path(work)
        histories = work / "histories"
        path = archive("11059dd1", work / "legacy", "src/bts/state/regenerate.py") / "src/bts/state/regenerate.py"
        legacy = ModuleType("early_regenerate")
        sys.modules[legacy.__name__] = legacy
        exec(compile(path.read_text(), str(path), "exec"), legacy.__dict__)
        manifest = list(build(histories, legacy))
        sources = [archive(ref, work / ref, "src") / "src" for ref in (BASE, HEAD)]
        results = [python(src, __file__, "--worker", histories, timeout=35) for src in sources]
        for result in results:
            print(result.stderr.strip())
        before, after = [json.loads(result.stdout) for result in results]
        summary = Counter()
        groups = defaultdict(Counter)
        diff_path = HERE / "corpus-differences.jsonl"
        with diff_path.open("w") as differences:
            for name, case, expected in manifest:
                base, head = before[name], after[name]
                summary["histories"] += 1
                if head is not None and not isinstance(head, dict):
                    summary["fold_checked"] += 1
                    if head != expected:
                        summary["unexpected_fold"] += 1
                if base != head:
                    summary[case.classification] += 1
                    groups[case.name][json.dumps([base, head])] += 1
                    differences.write(json.dumps(dict(history=name, category=case.classification,
                                                      origin=case.origin, base=base, head=head,
                                                      oracle=expected)) + "\n")
        print(json.dumps(dict(summary), sort_keys=True))
        for case in cases(legacy):
            print(case.name, case.classification, case.origin, dict(groups[case.name]))
        assert not summary["unexpected_fold"]


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        with offline():
            print(json.dumps(replay_all(Path(sys.argv[2]))))
    else:
        main()
