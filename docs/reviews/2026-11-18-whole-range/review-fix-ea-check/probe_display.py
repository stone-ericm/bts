"""Check exact B8a copy and skip text through state, journal, DM and dashboard."""

from contextlib import nullcontext, redirect_stderr
from dataclasses import replace
from datetime import datetime
from html import escape
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import ModuleType
from unittest.mock import patch

from bts import orchestrator, picks, scheduler, web
from scripts.audit.season_walk import walk


WORK = Path(__file__).resolve().parent
ROOT = WORK.parent.parent
DAY = "2027-07-15"
NOW = datetime.fromisoformat(DAY + "T18:20:00-04:00")


def b8a():
    checklist = (ROOT / "docs/ops/2027-season-start.md").read_text()
    contract = (ROOT / "docs/reviews/2026-11-18-whole-range/kickoff-fix-ea.md").read_text()
    followup = contract.split("### Applied by the session to the one B8a bullet\n", 1)[1]
    table = followup.split("### The author's items", 1)[0]
    line = next(line for line in checklist.splitlines() if line.startswith("| B8a |"))
    bullet = line.split("<li>After the switch, ", 1)[1].split("</li>", 1)[0]
    bullet = ("After the switch, " + bullet).replace("**", "").replace("`", "")
    expected = [line.split(" | ")[2].strip("` ") for line in table.splitlines()
                if line.startswith(("| Codex", "| Carbon"))]
    assert len(expected) == 4
    for value in expected:
        assert bullet.count(value) == 1, value
        print("B8a exact visible copy:", value)
    for cmd in ("bts saver-state --picks-dir DIR",
                "bts set-contest-streak --date TODAY --streak N --best-streak B --saver S --picks-dir DIR"):
        assert f"`{cmd}`" in line


def load_legacy():
    location = "63669c42:src/bts/scheduler.py"
    source = subprocess.run(["git", "show", location], cwd=ROOT, timeout=5,
                            capture_output=True, check=True, text=True).stdout
    old = ModuleType("bts._ea3_legacy_scheduler")
    sys.modules[old.__name__] = old
    exec(compile(source, location, "exec"), old.__dict__)
    return old


def check(module, path, statuses, override_reason=None):
    config = walk._config(path)
    frame = walk._predictions(NOW.date(), .75)
    frame.loc[0, "batter_name"] = "Batter below the line"
    frame.loc[0, "team"] = "Team below the line"
    real_choose = orchestrator.run_and_pick

    def choose(*args, **kwargs):
        rows, selection, tier = real_choose(*args, **kwargs)
        if override_reason is not None:
            selection = replace(selection, decision=replace(selection.decision, reason=override_reason))
        return rows, selection, tier

    journal = StringIO()
    with patch.object(orchestrator, "run_cascade", return_value=(frame, "pinned")), \
            patch.object(orchestrator, "run_and_pick", side_effect=choose), \
            patch.object(picks, "get_game_statuses_detailed", return_value=statuses), \
            patch.object(module, "count_new_confirmations", return_value=3), \
            patch.object(module, "heartbeat_watchdog", side_effect=lambda *a, **k: nullcontext()), \
            redirect_stderr(journal):
        result = module.run_single_check(DAY, list(statuses), set(), config, .03)
    return result, journal.getvalue()


def saved(summary, stamp=None):
    return scheduler.SchedulerState(DAY, "pinned", [], [], [], False, None, None, None,
                                    skip_summary=summary, skip_notified_at=stamp)


def render_loaded(path, summary, expected, journal=None):
    encoded = scheduler.save_state(saved(summary), path)
    loaded = scheduler.load_state(DAY, path)
    assert summary == loaded.skip_summary == json.loads(encoded.read_text())["skip_summary"]
    assert scheduler.format_skip_summary(loaded.skip_summary) == expected
    assert scheduler.format_skip_dm(DAY, loaded.skip_summary) == f"BTS {DAY}: No pick at this check. {expected}"
    banner = web.render_skip_banner(loaded.skip_summary)
    assert banner == ('<div class="skip-banner"><strong>LAST RECORDED NO-PICK SUMMARY</strong><span>'
                      + escape(expected) + "</span></div>")
    with patch.object(web, "PICKS_DIR", path), patch.object(web, "fetch_bluesky_posts", return_value=[]):
        assert banner in web.render_page()
    if journal is not None:
        assert f"  SKIP — {expected}\n" in journal
    print("Skip exact copy:", expected)


def skip_checks(root, old):
    path = root / "picks"
    assert not walk.install_state(path, NOW.date(), walk.State("dated", 10, 14, "used", .75))
    walk._set_clock(NOW)
    finals = {pk: {"abstract": "F", "detailed": "Final"} for pk in range(70000, 70003)}
    result, _ = check(old, path, finals)
    legacy = result["skip_summary"]
    assert result["selection"].decision is None and "basis" not in legacy and legacy["streak"] == 10
    old.save_state(saved(legacy, NOW.isoformat()), path)
    restarted = scheduler.carry_forward_skip_state(saved(None), scheduler.load_state(DAY, path))
    assert restarted.skip_summary == legacy and restarted.skip_notified_at == NOW.isoformat()
    config = walk._config(path) | {"bluesky": {"dm_recipient": "capture.invalid"}}
    assert not scheduler.maybe_notify_skip(restarted, legacy, config, now_iso=NOW.isoformat(),
                                           send=lambda *args: raise_unexpected_send())

    label = "Top model prediction (eligibility unverified): Batter under the line (Team under the line) 75.0%"
    body = label + ". No probability comparison is available for this check."
    for value in ({}, {"basis": "legacy"}, {"basis": []}, {"basis": None}):
        render_loaded(path, legacy | value, body + " Streak recorded with this check: 10.")
    render_loaded(path, legacy | {"basis": "decision"}, body + " Streak used at this check: 10.")
    render_loaded(path, legacy | {"basis": "no_decision"}, "No candidate was chosen. " + label + ".")
    for invalid in (True, -1, "10", None):
        render_loaded(path, legacy | {"streak": invalid}, body)

    statuses = {pk: {"abstract": "P", "detailed": "Scheduled"} for pk in range(70000, 70003)}
    result, journal = check(scheduler, path, statuses, "Observations\nbelow the expected count")
    summary = result["skip_summary"]
    assert result["selection"].action == "skip" and summary["basis"] == "decision"
    assert summary["reason"] == "Observations\nbelow the expected count"
    candidate = "Candidate used in the decision: Batter under the line (Team under the line) 75.0%"
    expected = candidate + ". Reason: Observations under the expected count. Streak used at this check: 10."
    render_loaded(path, summary, expected, journal)
    fresh = saved(summary)
    dms = []
    send = lambda recipient, body: dms.append((recipient, body))
    assert scheduler.maybe_notify_skip(fresh, summary, config, now_iso=NOW.isoformat(), send=send)
    assert not scheduler.maybe_notify_skip(fresh, summary, config, now_iso=NOW.isoformat(), send=send)
    assert dms == [("capture.invalid", f"BTS {DAY}: No pick at this check. {expected}")]
    k2 = scheduler.format_skip_summary({key: value for key, value in summary.items() if key != "reason"})
    assert k2.count("below the") == 1 and "below the streak-10 bar" in k2
    print("K2 retains exactly one comparison:", k2)
    prefix = scheduler.format_skip_summary(summary | {"reason": "Counts below theoretical minimum"})
    assert "Reason: Counts under theoretical minimum." in prefix
    print("Substring inside a longer word is also replaced:", prefix)
    false_match = scheduler.format_skip_summary(summary | {"reason": "Model ID below themodel"})
    assert "Reason: Model ID under themodel." in false_match
    print("Stored identifier mutation:", false_match)


def raise_unexpected_send():
    raise AssertionError("A resumed skip notice must not re-send")


if __name__ == "__main__":
    b8a()
    legacy = load_legacy()
    with TemporaryDirectory(dir=WORK) as temp, walk.offline(NOW):
        skip_checks(Path(temp), legacy)
