"""Pass decoded legacy/current summaries through both dashboard and scheduler renderers."""
import contextlib
import html
import importlib.util
import io
import json
import sys
import tempfile
from pathlib import Path

from pytest import MonkeyPatch

HERE = Path(__file__).resolve().parent
BASE = HERE / "base/src/bts/scheduler.py"
DAY = "2027-07-15"


def state(module, summary):
    return module.SchedulerState(
        date=DAY, schedule_fetched_at=None, games=[], confirmed_game_pks=[], runs_completed=[],
        pick_locked=False, pick_locked_at=None, result_status=None, next_wakeup=None, skip_summary=summary,
    )


def run(root, mp):
    from bts import scheduler, web

    spec = importlib.util.spec_from_file_location("wr_e_base_scheduler", BASE)
    base = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = base
    spec.loader.exec_module(base)
    mp.setattr(web, "PICKS_DIR", root)
    original = {"best_batter": "A", "best_team": "B", "best_p": .79, "bar": .79, "streak": 10}
    base.save_state(state(base, original), root)
    assert scheduler.load_state(DAY, root).skip_summary == original
    assert "eligibility unverified" in web.render_skip_banner(web.load_scheduler_state(DAY)["skip_summary"])
    new = original | {"basis": "decision", "candidate_verified": True}
    path = scheduler.save_state(state(scheduler, new), root)
    assert base.load_state(DAY, root).skip_summary == new
    print("state round-trips: base→HEAD and HEAD→base")

    selected = []
    for summary in [
        None, {}, "string", [1], 2,
        *[dict(new, best_p=value, bar=value, streak=value) for value in
          ("0.99", True, None, {}, [-1], float("nan"), float("inf"), 10**400, 1e308)],
        dict(new, best_p=1e308, bar=1e307),
        dict(new, best_batter='<i>A</i>\r\nB\x85C', best_team='"\n&', reason='<img src=x>\u2028Line!'),
        dict(new, best_batter=["A"], best_team={"a": 1}, reason=["note"], basis=["no_decision"]),
        dict(new, reason="Stopped!", basis="no_decision"),
        dict(new, basis="no_decision", best_p=.75, bar=.8),
        dict(new, streak=None, best_p=.75, bar=.8),
    ]:
        scheduler.save_state(state(scheduler, summary), root)
        loaded = scheduler.load_state(DAY, root).skip_summary
        banner = web.render_skip_banner(web.load_scheduler_state(DAY)["skip_summary"])
        dm = scheduler.format_skip_dm(DAY, loaded)
        full = scheduler.format_skip_summary(loaded)
        assert "\n" not in dm and "\r" not in dm
        assert dm.endswith(full)
        if not isinstance(summary, dict) or not summary:
            assert banner == ""
        else:
            assert banner.split("<span>")[1] == html.escape(full) + "</span></div>"
        in_case_k2 = isinstance(summary, dict) and summary.get("bar") == .8 and summary.get("basis") == "decision"
        assert ("below the" in dm) is in_case_k2
        if "inf%" in dm:
            selected.append(dm)
        print(repr(summary), dm, banner)
    assert len(selected) == 2

    path.write_text("{")
    with contextlib.redirect_stderr(io.StringIO()) as errors:
        assert scheduler.load_state(DAY, root) is None
    assert "quarantined" in errors.getvalue()
    path.mkdir()
    assert web.load_scheduler_state(DAY) == {}
    try:
        scheduler.load_state(DAY, root)
    except IsADirectoryError:
        print("scheduler directory load raises before renderer; dashboard maps it to {}")
    else:
        raise AssertionError("expected unchanged scheduler loader error")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(dir=HERE) as directory, MonkeyPatch.context() as monkeypatch:
        run(Path(directory), monkeypatch)
