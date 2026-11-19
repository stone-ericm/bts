"""Compare unchanged callers with both revisions under the same clocks and local inputs."""

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
import sys
from unittest.mock import patch

from probe_support import BASE, DAY, ET, HEAD, RUN, STREAK, archive, decision, load, offline
from probe_support import pick_data, plant, python, scratch, seed_streak, slot
from reconcile_probe import menu


class PinnedDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return RUN.replace(tzinfo=None) if tz is None else RUN.astimezone(tz)


def invoke_checks(work):
    from bts import cli, picks, scheduler, web
    from bts.health import result_resolution, streak_validation
    from click.testing import CliRunner

    root = work / "picks"
    root.mkdir(parents=True)
    result = {}

    def current():
        return (root / STREAK).read_text()

    with patch.object(picks, "datetime", PinnedDatetime):
        for name, initial, saver, markers, season, updates in (
            ("dated hit", 7, True, ["2027-04-18"], 2027, [([True], DAY), ([True], DAY)]),
            ("dated saved miss", 10, True, [], 2027, [([False], DAY)]),
            ("dated miss with used saver", 12, False, [], 2027, [([False], DAY)]),
            ("dateless hit keeps tag", 7, True, ["2027-04-18"], 2027, [([True, True], None)]),
            ("cross season", 15, False, ["2026-12-31"], 2026, [([True], DAY)]),
        ):
            seed_streak(root, streak=initial, saver=saver, markers=markers, season=season)
            series = []
            for legs, day in updates:
                count = picks.update_streak(legs, root, date=day)
                series.append({"count": count, "exact_bytes": current()})
            result[name] = series

        cases = [("single hit", DAY, True, None, "hit"),
                 ("private miss", DAY, False, "private_locked", "miss"),
                 ("single void", DAY, True, None, "void"),
                 ("skip leftover", DAY, False, "skipped", None),
                 ("no decision preview", DAY, False, None, None),
                 ("resolved short-circuit", DAY, False, None, None)]
        for index, (name, day, delivered, status, answer) in enumerate(cases):
            target = work / f"check-{index}" / "picks"
            seed_streak(target, streak=10, saver=True)
            data = plant(target, day, "hit" if name == "resolved short-circuit" else None, bluesky_posted=delivered)
            if status:
                decision(target, day, status=status, scoreable=status == "private_locked")
            with patch.object(cli, "_now_et", return_value=RUN), menu(answer):
                invocation = CliRunner().invoke(cli.cli, ["check-results", "--picks-dir", str(target), "--date", day])
            assert invocation.exit_code == 0, invocation.output
            result["check-results " + name] = {"stdout": invocation.stdout, "stderr": invocation.stderr,
                                               "streak_bytes": (target / STREAK).read_text(), "pick": load(data)}

        for index, (name, status, now, answer) in enumerate((
            ("final", "final", RUN, "hit"), ("pending final", "final", RUN, None),
            ("suspended", "suspended", RUN, None), ("cap", "live", RUN.replace(hour=5), None),
        )):
            target = work / f"poll-{index}" / "picks"
            seed_streak(target, streak=4, saver=True)
            data = plant(target, DAY, bluesky_posted=True)
            out, err = StringIO(), StringIO()
            with patch.object(scheduler, "_now_et", return_value=now), \
                 patch.object(scheduler, "poll_game_result", return_value=status), \
                 patch.object(picks, "get_game_statuses_detailed", return_value={}), \
                 patch.object(picks, "resolve_daily_slot_results", return_value={"pick": answer} if answer else None), \
                 redirect_stdout(out), redirect_stderr(err):
                state = scheduler.run_result_polling(101, DAY, target)
            result["polling " + name] = {"result": state, "stdout": out.getvalue(), "stderr": err.getvalue(),
                                          "streak_bytes": (target / STREAK).read_text(), "pick": load(data)}

    target = work / "dashboard" / "picks"
    seed_streak(target, streak=10, saver=True)
    plant(target, DAY, "hit", double_down=slot(DAY, 102), slot_results={"pick": "void", "double_down": "hit"})
    plant(target, "2027-04-17", "unresolved", bluesky_posted=True)
    plant(target, "2027-04-12", "suspended", notification_sent=True, notification_id="synthetic")
    plant(target, "2027-04-16", "unresolved")
    decision(target, "2027-04-16", scoreable=False)
    today = RUN.date().isoformat()
    plant(target, today)
    with patch.object(web, "PICKS_DIR", target), patch.object(web, "datetime", PinnedDatetime), \
         patch.object(web, "fetch_bluesky_posts", return_value=[]), patch.object(web, "fetch_live_scorecard", return_value=None):
        page = web.render_page()
        assert "Synthetic batter" in page and "Current Streak" in page
        result["dashboard"] = {"streak": web.load_streak(), "pick_count": len(web.load_all_picks()),
                               "page_sha256": sha256(page.encode()).hexdigest()}
    result["result health"] = [asdict(alert) for alert in result_resolution.check(target, today=RUN.date())]
    result["streak health"] = [asdict(alert) for alert in streak_validation.check(target)]
    (target / STREAK).write_text('{"streak": true, "saver_available": "yes"}')
    result["streak health invalid"] = [asdict(alert) for alert in streak_validation.check(target)]

    value = pick_data(DAY, "hit", pick=slot(DAY) | {"batter_id": "100001", "game_pk": None},
                      extra_legacy="ignored")
    result["legacy parser"] = asdict(picks.parse_daily_pick(json.dumps(value)))
    return result


if __name__ == "__main__":
    with offline():
        if len(sys.argv) > 1:
            print(json.dumps(invoke_checks(Path(sys.argv[1])), default=str, sort_keys=True))
        else:
            with scratch("callers") as work:
                work = Path(work)
                observations = []
                for revision in (BASE, HEAD):
                    source = archive(revision, work / revision, "src") / "src"
                    try:
                        observations.append(json.loads(python(source, __file__, work / "data", timeout=20).stdout))
                    except Exception as error:
                        if hasattr(error, "stdout"):
                            print(error.stdout, error.stderr, file=sys.stderr)
                        raise
                    # Reset only the generated caller fixtures, with the same path in both observations.
                    (work / "data").rename(work / f"data-{revision}")
                base, head = observations
                for name in head:
                    print(json.dumps({"caller": name, "equal": base[name] == head[name], "base": base[name], "head": head[name]}))
                assert base == head
