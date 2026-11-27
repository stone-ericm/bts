import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
from unittest.mock import patch

from run_tests import offline

sys.addaudithook(offline)
import bts
from bts import dm, picks, posting
from bts.health import alert
from tests.test_stale_delivery_cs4c import seed_automatic

DOCS = Path.cwd() / "docs/reviews/2026-11-18-whole-range"
sys.path.insert(0, str(DOCS / "review-fix-k2-codex"))
from probe_startup import DAY, Startup, game, scheduler


def start_cases(root):
    for mode in ("private", "dm", "public"):
        for age in (None, 3, 20):
            for at in ("15:54:55", "15:55:01"):
                workers, observations = [], []
                startup = Startup(root / f"{mode}-{age}-{at}", mode=mode, required=True)
                now = datetime.fromisoformat(f"{DAY}T{at}").replace(tzinfo=scheduler.ET)
                if age is not None:
                    recorded = (now - timedelta(hours=age)).isoformat()
                    seed_automatic(startup.root / "picks", recorded, now=now, streak=20)

                class RecordedThread(Thread):
                    def start(self):
                        workers.append(self)
                        super().start()

                extra = []
                if hasattr(scheduler, "_season_boundary_check"):
                    extra.extend([patch.object(scheduler, "Thread", RecordedThread),
                                  patch.object(scheduler, "_write_season_boundary_line", observations.append)])
                real_dispatch = alert.dispatch_dm_for_health_alerts
                with (patch.object(alert, "dispatch_dm_for_health_alerts", wraps=real_dispatch) as dispatch,
                      patch.object(dm, "send_dm") as direct_dm,
                      patch.object(posting, "post_to_bluesky") as public_post):
                    def lag(_, body):
                        startup.now += timedelta(seconds=11)
                        return "captured-id"

                    checks, _ = startup.run([game()], at=at, transport=lag, extra=extra)
                    for worker in workers:
                        worker.join(1)
                        assert not worker.is_alive()
                    assert not direct_dm.called and not public_post.called
                    wanted = int(mode != "private" and at == "15:54:55" and age != 3)
                    assert dispatch.call_count == len(startup.bodies) == wanted, startup.bodies
                    assert not any("season_boundary" in str(p) for p in startup.root.rglob("*"))
                    assert checks == [(f"{DAY}T16:05:00-04:00", [])], checks
                if extra:
                    assert len(workers) == len(observations) == 1
                yield mode, age, at, checks, startup.bodies


def sent_pick(root):
    startup = Startup(root, mode="public", required=True)
    directory = Path(startup.config["orchestrator"]["picks_dir"])
    daily = picks.DailyPick(
        date=DAY, run_time=f"{DAY}T15:00:00+00:00",
        pick=picks.Pick(batter_name="Already sent", batter_id=11, team="AWY",
                        lineup_position=1, pitcher_name="Starter", pitcher_id=22,
                        p_game_hit=.82, flags=[], projected_lineup=False, game_pk=301,
                        game_time=f"{DAY}T23:05:00Z"),
        double_down=None, runner_up=None, bluesky_posted=True, bluesky_uri="at://offline/prior",
    )
    path = picks.save_pick(daily, directory)
    original = path.read_bytes()
    with patch.object(scheduler, "_write_season_boundary_line", lambda _: None, create=True):
        checks, _ = startup.run([game()], at="15:54:55")
    assert path.read_bytes() == original and not startup.bodies
    assert checks == [(f"{DAY}T16:05:00-04:00", [])]


if __name__ == "__main__":
    assert Path(bts.__file__).resolve() == Path(sys.argv[1]).resolve() / "bts" / "__init__.py"
    with TemporaryDirectory() as scratch:
        root = Path(scratch)
        print(json.dumps(list(start_cases(root)), ensure_ascii=False))
        sent_pick(root / "sent-pick")
        print("previously sent pick byte-identical; no additional DM")
