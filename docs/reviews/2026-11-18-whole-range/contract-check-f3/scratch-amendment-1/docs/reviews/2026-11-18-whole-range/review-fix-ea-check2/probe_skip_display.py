"""Compare real journal, captured skip DM and dashboard rendering on both revisions."""

from contextlib import nullcontext, redirect_stderr
from dataclasses import asdict, replace
from html import escape
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from bts import orchestrator, picks, scheduler, web
from scripts.audit.season_walk import walk

from probe_delivery_warning_r2 import copy_audit, source_audit
from probe_later_slate_r2 import WORK, at, config, scheduler_state


DAY = "2027-07-15"
NOW = at(f"{DAY}T18:20")
CASES = (
    ("Model ID below themodel", "Model ID below themodel", "Model ID under themodel"),
    ("below the line; xbelow the line; below the_model; below the2; (below the)",
     "under the line; xbelow the line; below the_model; below the2; (under the)",
     "under the line; xunder the line; under the_model; under the2; (under the)"),
    ("Observations below\nthe expected count", "Observations under the expected count",
     "Observations under the expected count"),
    ("Below the line, below The line, BELOW THE line", "Below the line, below The line, BELOW THE line",
     "Below the line, below The line, BELOW THE line"),
    ("Counts below theoretical minimum", "Counts below theoretical minimum", "Counts under theoretical minimum"),
    (None, None, None),
)


def run_check(module, path, frame, reason):
    statuses = {pk: {"abstract": "P", "detailed": "Scheduled"} for pk in range(70000, 70003)}
    real_select = orchestrator.run_and_pick

    def select(*args, **kwargs):
        predictions, choice, tier = real_select(*args, **kwargs)
        assert choice.action == "skip" and choice.decision.pick_bar is not None
        return predictions, replace(choice, decision=replace(choice.decision, reason=reason)), tier

    journal = StringIO()
    with (patch.object(orchestrator, "run_cascade", return_value=(frame, "pinned")),
          patch.object(orchestrator, "run_and_pick", side_effect=select),
          patch.object(picks, "get_game_statuses_detailed", return_value=statuses),
          patch.object(module, "count_new_confirmations", return_value=3),
          patch.object(module, "heartbeat_watchdog", side_effect=lambda *a, **kw: nullcontext()),
          redirect_stderr(journal)):
        result = module.run_single_check(DAY, list(statuses), set(), config(path), .03)
    assert result["pick_result"] is None and not result["should_post"]
    return result["skip_summary"], asdict(result["selection"].decision), journal.getvalue()


def render(module, path, frame, reason, expected, expected_batter, expected_team):
    summary, decision, journal = run_check(module, path, frame, reason)
    state = scheduler_state(DAY)
    state.skip_summary = summary
    encoded = module.save_state(state, path)
    loaded = module.load_state(DAY, path)
    assert summary == loaded.skip_summary == json.loads(encoded.read_text())["skip_summary"]
    if reason is not None:
        assert summary["reason"] == reason
    text = module.format_skip_summary(loaded.skip_summary)
    assert f"{expected_batter} ({expected_team}) 75.0%" in text
    if expected is not None:
        assert f"Reason: {expected}." in text
    else:
        assert ", below the streak-10 bar (" in text and "using unrounded values." in text
    assert "\n" not in text and f"  SKIP — {text}\n" in journal
    sent = []
    capture = lambda recipient, body: sent.append((recipient, body))
    stamp = NOW.isoformat()
    assert module.maybe_notify_skip(loaded, loaded.skip_summary, config(path), now_iso=stamp, send=capture)
    assert not module.maybe_notify_skip(loaded, loaded.skip_summary, config(path), now_iso=stamp, send=capture)
    assert sent == [("capture.invalid", f"BTS {DAY}: No pick at this check. {text}")]
    assert loaded.skip_notified_at == stamp
    if module is scheduler:
        banner = web.render_skip_banner(loaded.skip_summary)
        assert banner == ('<div class="skip-banner"><strong>LAST RECORDED NO-PICK SUMMARY</strong><span>'
                          + escape(text) + "</span></div>")
        with patch.object(web, "PICKS_DIR", path), patch.object(web, "fetch_bluesky_posts", return_value=[]):
            assert banner in web.render_page()
    print(module.__name__, repr(reason), text)
    return summary, decision, loaded.skip_notified_at


if __name__ == "__main__":
    old = source_audit(*copy_audit())
    with TemporaryDirectory(dir=WORK) as temp, walk.offline(NOW):
        root = Path(temp)
        path = root / "picks"
        path.mkdir()
        assert walk.install_state(path, NOW.date(), walk.State("dated", 10, 14, "used", .75)) == []
        walk._set_clock(NOW)
        frame = walk._predictions(NOW.date(), .75)
        frame.loc[0, "batter_name"] = "<Batter> below the line"
        frame.loc[0, "team"] = "Team below themodel"
        for original, head_text, base_text in CASES:
            baseline = render(old, path, frame, original, base_text, "<Batter> under the line", "Team under themodel")
            current = render(scheduler, path, frame, original, head_text, "<Batter> under the line", "Team below themodel")
            assert baseline == current
        assert picks.load_pick(DAY, path) is None
        print("PARITY: identical stored summary fields, policy decisions, no-pick outcomes, and once-per-slate notice timing.")
