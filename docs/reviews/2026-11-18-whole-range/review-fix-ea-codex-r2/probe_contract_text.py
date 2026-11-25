"""Read expected wording from the contract, then capture the real envelopes."""
from datetime import datetime, timedelta
from io import StringIO
from contextlib import redirect_stderr
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from unittest.mock import patch

from bts import contest_state, picks, scheduler
from bts.health import alert
from scripts.audit.season_walk import walk
from tests.test_pick_chosen_on_cs5a import daily_pick


work = Path(__file__).resolve().parent
repo = work.parent.parent
docs = repo / "docs/reviews/2026-11-18-whole-range"
original = (docs / "kickoff-fix-e.md").read_text().split("## E2, composed")[1]
ea = (docs / "kickoff-fix-ea.md").read_text()
warning_envelope = "BTS health WARN attention:\n- [contest_state] "
refusal_envelope = "BTS health CRITICAL alert(s):\n- "


def after(doc, marker):
    return doc.split(marker, 1)[1].split("`")[1]


morning = after(ea, "The message becomes exactly:")
delivery = after(original, "**After a delivery:**").replace(
    "For a later date, follow B8b(1).", "For a later slate date, see B8a.")
late_original = after(original, "**Late:**")
late = after(ea, "The late refusal DM's first sentence becomes exactly")
late += " " + late_original.split(". ", 1)[1]
void = after(original, "**Postponed:**")
bullet = re.search(r"<li>After the switch: (.*?)</li>",
                   (repo / "docs/ops/2027-season-start.md").read_text()).group(1)
amendment = ea.split("### Applied by the session to the one B8a bullet")[1].split("### Must still catch")[0]
for expected in re.findall(r"`([^`]+)`", amendment):
    if expected.startswith(("A confirmed delivery using", "When a dated manual", "**For a later slate",
                            "Read the latest fetch status")):
        assert expected in bullet.replace("`", ""), expected
print("B8a contains each round-2 replacement, ignoring Markdown code markup.")

day = "2027-07-15"
now = datetime.fromisoformat(day + "T10:00:00-04:00")
with TemporaryDirectory(dir=work) as tmp, walk.offline(now):
    path = Path(tmp) / "picks"
    cfg = walk._config(path) | {"bluesky": {"dm_recipient": "capture.invalid"}}
    received = []
    def send(_, body):
        received.append(body)
        return "captured"
    stamps = (
        "0001-01-01T00:00:00." + "0" * 102 + "+00:00",
        (now + timedelta(hours=1)).isoformat(),
        "invalid",
    )
    for index, stamp in enumerate(stamps):
        for route in ("morning", "delivery"):
            directory = path / str(index) / route / "picks"
            reading = contest_state.DecisionStreakState(
                12, True, True, "contest", "fresh", None, None,
                observation_kind="automatic", contest_recorded_at=stamp)
            record = contest_state.automatic_reading_record(reading, now)
            if record["reason"] == "old":
                detail = f'last accepted fetch {stamp}, about {record["age_hours"]:.2f} hours ago'
            elif record["reason"] == "future":
                detail = f"its recorded_at {stamp} is in the future"
            else:
                detail = "it has no usable recorded_at"
            with patch.object(alert, "send_dm", side_effect=send), \
                    patch.object(contest_state, "load_decision_streak_state", return_value=reading), \
                    redirect_stderr(StringIO()) as log:
                if route == "morning":
                    scheduler._morning_state_check(cfg, directory, day, now, now + timedelta(hours=8))
                    expected = morning.format(D=day, detail=detail)
                    assert log.getvalue() == f"  MORNING STATE CHECK — stale contest reading: {detail}\n"
                else:
                    scheduler._warn_stale_contest_reading(record, cfg, directory)
                    expected = delivery.format(D=day, detail=detail)
                    assert log.getvalue() == f"  STALE CONTEST READING — {expected}\n"
            assert received[-1] == warning_envelope + expected, (received[-1], expected)
            assert len(received[-1]) < 900
            print(f"{route} {record['reason']}: exact agreement, {len(received[-1])} characters")

    for route in ("late", "void"):
        folder = path / route / "picks"
        daily = daily_pick()
        cutoff = picks.submission_cutoff_et(daily)
        walk._set_clock(cutoff)
        picks.save_pick(daily, folder)
        state = scheduler.SchedulerState(day, "pinned", [], [], [], False, None, None, None)
        with patch.object(alert, "send_dm", side_effect=send), redirect_stderr(StringIO()):
            if route == "late":
                scheduler._refuse_late_delivery(daily, cfg, folder, state, day, route, cutoff, cutoff)
                expected = "[late_delivery] " + late.format(cutoff=cutoff.strftime("%H:%M ET"),
                                                            names=daily.pick.batter_name)
            else:
                scheduler._refuse_void_game_delivery(daily, cfg, folder, state, day, route, {101: "Postponed"})
                expected = "[postponed_pick] " + void.format(date=day, games="game 101 is Postponed")
        assert received[-1] == refusal_envelope + expected, (received[-1], expected)
        print(f"{route}: exact refusal agreement")
