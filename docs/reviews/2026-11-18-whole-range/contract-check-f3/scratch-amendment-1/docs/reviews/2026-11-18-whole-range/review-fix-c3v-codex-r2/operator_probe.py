"""Run checklist B8b with real private locks and synthetic contest observations."""

import builtins
from datetime import date, datetime
import json
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
from unittest.mock import patch

from rendering_probe import deadline
from round1.codex.probe_common import ET, RUN, examine, receipt, refuse_network


REVIEW = Path(__file__).resolve().parent
FIRST = "2027-04-18"
SECOND = "2027-04-19"
LOCK_TIME = datetime(2027, 4, 18, 18, tzinfo=ET)
ARCHIVE = "set_aside_never_entered.json"


def prepare(work):
    from tests.test_wr_c3v_corpus import dump, pick_data, seed_streak, stagger

    root = work / "data" / "picks"
    seed_streak(root, streak=9, saver=True)
    for offset in range(9):
        day = stagger("2027-04-01", offset)
        dump(root / f"{day}.json", pick_data(day, "hit", bluesky_posted=True))
    return root


def assert_blocked(root, day):
    result = examine(root)
    assert result["strict"][0:3] == [["return", None], 0, None]
    assert result["strict"][3] == result["strict"][4]
    assert result["stderr"] == (
        f"Season replay unavailable for {day} (no final result on the pick); "
        "streak retained, not recalculated.\n"
    )
    assert receipt(result)["replay"]["blocked_date"] == day
    return result["stderr"].rstrip()


def assert_rebuilt(root, streak):
    result = examine(root)
    assert result["strict"][0:3] == [["return", (streak, True)], 0, None]
    assert result["stderr"] == ""
    assert receipt(result)["replay"] == {"state": "saved", "streak": streak, "saver_available": True}
    assert json.loads((root / "streak.json").read_bytes())["streak"] == streak
    return streak


def decision_read(path):
    try:
        return json.loads(path.read_bytes())["scoreable"]
    except FileNotFoundError:
        return "missing"
    except PermissionError:
        return "unreadable"


def move(work, day):
    root = work / "data" / "picks"
    source = root / f"{day}.json"
    target = root / day / ARCHIVE
    saved = source.read_bytes()
    target.parent.mkdir(exist_ok=True)
    subprocess.run(
        ["/bin/mv", f"data/picks/{day}.json", f"data/picks/{day}/{ARCHIVE}"],
        cwd=work, check=True, timeout=3,
    )
    assert not source.exists() and target.read_bytes() == saved
    return target


def readers(root, accesses):
    from bts import daily_decision, picks, scheduler, skip_policy_shadow, web
    from bts.data import backup, lineup_analyze
    from bts.health import (
        analytics_artifacts_missing, fallback_defer, late_delivery, result_resolution, scheduler_state_integrity,
    )
    from bts.state import export as state_export, verify as state_verify

    old_path_open, old_open = Path.open, builtins.open

    def record(file):
        if isinstance(file, (str, Path)) and Path(file).name == ARCHIVE:
            accesses.append(str(file))

    def path_open(file, *args, **kwargs):
        record(file)
        return old_path_open(file, *args, **kwargs)

    def builtin_open(file, *args, **kwargs):
        record(file)
        return old_open(file, *args, **kwargs)

    with patch.object(Path, "open", path_open), patch.object(builtins, "open", builtin_open), \
         patch.object(web, "PICKS_DIR", root):
        assert_rebuilt(root, 9)
        assert all(picks.load_pick(day, root) is None for day in (FIRST, SECOND))
        for day in (FIRST, SECOND):
            daily_decision.load_decision_bytes(day, root)
            scheduler.load_state(day, root)
            web.load_scheduler_state(day)
            analytics_artifacts_missing._load_scheduler_state(root, day)
            late_delivery.check(root, today=date.fromisoformat(day), now=RUN)
        skip_policy_shadow.record_pending_skips(root, now=RUN)
        skip_policy_shadow.find_aged_contradictions(root, now=RUN)
        lineup_analyze.backfill_from_scheduler_state(root)
        result_resolution.check(root, today=RUN.date())
        scheduler_state_integrity.check(root, today=RUN.date())
        fallback_defer.check(root, today=RUN.date())
        analytics_artifacts_missing.check(root, today=RUN.date())
        by_date = state_export.export_initial_state(root, root.parents[1] / "export.json")
        assert not {FIRST, SECOND} & {item["date"] for item in by_date["historical_picks"]}
        state_verify.diff_pick_files(root, root)
        backup.verify_restored_ops_tree(root.parents[1])
        assert not {FIRST, SECOND} & {pick["date"] for pick in web.load_all_picks()}


def genuine_previews(work, days):
    from bts import daily_decision
    from tests.test_wr_c3v_corpus import pick_data, dump

    root = prepare(work)
    for day in days:
        dump(root / f"{day}.json", pick_data(day))
        if day == FIRST:
            daily_decision.write_decision(
                day, root, action="skip", source="mdp", delivery_status="not_applicable",
                scoreable=False, now=LOCK_TIME,
            )
    assert decision_read(root / FIRST / "decision.json") is False
    assert decision_read(root / SECOND / "decision.json") == "missing"
    warnings = [assert_blocked(root, FIRST), assert_blocked(root, FIRST)]
    for index, day in enumerate(days):
        # Fixture provenance confirms these files came solely from previews.
        move(work, day)
        if index + 1 < len(days):
            warnings.append(assert_blocked(root, days[index + 1]))
    poison = root / "2027-04-12" / ARCHIVE
    poison.parent.mkdir()
    poison.write_bytes(b"\xff\xfe{")
    accesses = []
    readers(root, accesses)
    assert not accesses
    return {"warnings": warnings, "streak_after": 9, "archive_opens": len(accesses)}


def private_lock(root, shape):
    from bts import daily_decision, entry_intent, picks, scheduler
    from tests.test_wr_c3v_corpus import pick_data

    if shape == "false":
        stale = daily_decision.write_decision(
            FIRST, root, action="skip", source="mdp", delivery_status="not_applicable",
            scoreable=False, now=LOCK_TIME.replace(hour=17),
        )
        assert stale["scoreable"] is False
    config = {"scheduler": {"pick_delivery": "private", "entry_intent": "research"}}
    assert entry_intent.check_entry_intent(config, include_live_gate=True).ok
    daily = picks.parse_daily_pick(json.dumps(pick_data(FIRST)))
    state = scheduler.SchedulerState(FIRST, LOCK_TIME.isoformat(), [], [], [], False, None, None, None)
    interrupted = shape in {"false", "missing"}

    def capture(*_args):
        if interrupted:
            raise KeyboardInterrupt()

    with patch.object(scheduler, "_now_et", return_value=LOCK_TIME), \
         patch.object(scheduler, "_void_committed_games", return_value={}), \
         patch.object(scheduler, "_trigger_live_forward_capture_on_lock", capture):
        try:
            assert scheduler._deliver_and_lock_pick(daily, config, root, state, FIRST, "probe")
        except KeyboardInterrupt:
            assert interrupted

    assert scheduler.load_state(FIRST, root).pick_locked is True
    pending = picks.load_pick(FIRST, root)
    assert pending.result is None and not picks.pick_was_delivered(pending)
    path = daily_decision.decision_path(FIRST, root)
    if shape == "unreadable":
        assert decision_read(path) is True
        path.chmod(0)
    return path


def locked_cases(scratch):
    from tests.test_wr_c3v_corpus import dump, pick_data

    outcomes = {}
    for shape, observed in (("true", True), ("false", False), ("missing", "missing"),
                            ("unreadable", "unreadable")):
        work = scratch / shape
        root = prepare(work)
        decision_path = private_lock(root, shape)
        assert decision_read(decision_path) == observed
        warning = assert_blocked(root, FIRST)
        outcomes[shape] = {
            "mlb_entered": False, "private_lock": True, "observed_scoreable": observed,
            "warning": warning, "safe_action": "leave the pick in place",
        }
        if shape == "true":
            assert (root / f"{FIRST}.json").exists()
            continue
        if shape == "unreadable":
            decision_path.chmod(0o600)
            assert decision_read(decision_path) is True
            continue

        # Demonstrate what the documented move does if the only named checks
        # (empty MLB history and absence of a true decision) are used as proof.
        kept = scratch / f"kept-{shape}" / "picks"
        shutil.copytree(root, kept)
        dump(kept / f"{FIRST}.json", pick_data(FIRST, "hit"))
        outcomes[shape]["streak_after_kept_pick_is_resolved_as_hit"] = assert_rebuilt(kept, 10)
        move(work, FIRST)
        outcomes[shape]["streak_after_moving_committed_pick"] = assert_rebuilt(root, 9)
    return outcomes


def noncanonical(scratch):
    work = scratch / "lf-path"
    root = prepare(work)
    (root / f"{FIRST}\n.json").write_text("{}")
    result = examine(root)
    assert receipt(result)["replay"]["blocked_date"] == FIRST + "\n"
    shown = FIRST + r"\n"
    assert shown in result["stderr"]
    target_dir = root / shown
    target_dir.mkdir()
    move_result = subprocess.run(
        ["/bin/mv", f"data/picks/{shown}.json", f"data/picks/{shown}/{ARCHIVE}"],
        cwd=work, capture_output=True, text=True, timeout=3,
    )
    assert move_result.returncode != 0 and (root / f"{FIRST}\n.json").is_file()
    return {"warnings": result["stderr"], "literal_display_move_exit": move_result.returncode,
            "true_filename_move_exit": 0 if move(work, FIRST + "\n").is_file() else 1}


def main():
    checklist = REVIEW.parents[1] / "docs" / "ops" / "2027-season-start.md"
    [b8b] = [line for line in checklist.read_text().splitlines() if line.startswith("| B8b |")]
    assert "`Season replay unavailable for <date> (no final result on the pick); streak retained, not recalculated.`" in b8b
    assert "`mv data/picks/<date>.json data/picks/<date>/set_aside_never_entered.json`" in b8b
    assert 'If the decision file exists and says `"scoreable": true`' in b8b
    with tempfile.TemporaryDirectory(dir=REVIEW) as folder, deadline(), \
         patch.object(socket.socket, "connect", refuse_network), \
         patch.object(socket.socket, "connect_ex", refuse_network), \
         patch.object(socket, "getaddrinfo", refuse_network):
        scratch = Path(folder)
        report = {"single": genuine_previews(scratch / "single", [FIRST]),
                  "two": genuine_previews(scratch / "two", [FIRST, SECOND]),
                  "locked": locked_cases(scratch), "escaped_filename": noncanonical(scratch)}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
