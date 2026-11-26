"""Carry a real base no-decision summary into a same-day branch restart."""
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

import bts
from bts import orchestrator, picks, scheduler, web
from scripts.audit.season_walk import walk


work = Path(__file__).resolve().parent
mode = sys.argv[1]
src = work / "base/src" if mode == "base" else work.parent.parent / "src"
assert Path(bts.__file__).is_relative_to(src), bts.__file__
now = datetime.fromisoformat("2027-07-15T18:35:00-04:00")
day = now.date().isoformat()
path = work / "legacy-restart" / "picks"
state = scheduler.SchedulerState(day, "pinned", [], [], [], False, None, None, None)
with walk.offline(now):
    if mode == "base":
        path.mkdir(parents=True, exist_ok=True)
        assert not walk.install_state(path, now.date(), walk.State("dated", 10, 14, "used", .99))
        walk._set_clock(now)
        rows = walk._predictions(now.date(), .99)
        statuses = {pk: {"abstract": "F", "detailed": "Final"} for pk in range(70000, 70003)}
        with patch.object(orchestrator, "run_cascade", return_value=(rows, "probe")), \
                patch.object(picks, "get_game_statuses_detailed", return_value=statuses), \
                patch.object(scheduler, "count_new_confirmations", return_value=3), \
                patch.object(scheduler, "heartbeat_watchdog", side_effect=lambda *a, **k: nullcontext()):
            result = scheduler.run_single_check(day, list(statuses), set(), walk._config(path), .03)
        assert result["selection"].decision is None
        state.skip_summary = result["skip_summary"]
        assert state.skip_summary["streak"] == 10 and "basis" not in state.skip_summary
        state.skip_notified_at = now.isoformat()
        scheduler.save_state(state, path)
        print("Base result had no decision:", state.skip_summary)
    else:
        restarted = scheduler.carry_forward_skip_state(state, scheduler.load_state(day, path))
        assert "basis" not in restarted.skip_summary and restarted.skip_notified_at == now.isoformat()
        banner = web.render_skip_banner(restarted.skip_summary)
        assert "Streak used at this check: 10." in banner
        with patch.object(web, "PICKS_DIR", path), patch.object(web, "fetch_bluesky_posts", return_value=[]):
            assert banner in web.render_page()
        print("Branch dashboard after same-day upgrade:", banner)

    with TemporaryDirectory(dir=work) as tmp:
        damaged = Path(tmp) / day / "scheduler_state.json"
        damaged.parent.mkdir()
        damaged.write_text("{")
        with patch.object(web, "PICKS_DIR", Path(tmp)):
            assert web.load_scheduler_state(day) == {}
        assert scheduler.load_state(day, Path(tmp)) is None
        assert list(damaged.parent.glob("scheduler_state.json.corrupt-*"))
        damaged.mkdir()
        with patch.object(web, "PICKS_DIR", Path(tmp)):
            assert web.load_scheduler_state(day) == {}
        try:
            scheduler.load_state(day, Path(tmp))
        except IsADirectoryError:
            print("Directory in place of scheduler_state.json: dashboard ignores, scheduler loader raises.")
        else:
            raise AssertionError("Expected the unchanged directory read error")
