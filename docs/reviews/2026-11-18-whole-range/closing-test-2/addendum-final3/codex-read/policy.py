import hashlib
import io
import json
import sys
from contextlib import nullcontext
from dataclasses import asdict, replace
from datetime import date, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import pandas as pd
from pytest import MonkeyPatch

import bts
from bts import cli, orchestrator, picks, posting, scheduler, strategy
from bts.features import park_drag
from bts.model import predict
from scripts.audit.season_walk import walk


DAY = date(2027, 9, 27)
ET = scheduler.ET


def status_for(frame):
    return {pk: {"abstract": "P", "detailed": "Scheduled", "code": "S"} for pk in frame.game_pk}


def select(day, directory, streak, best, saver, p=.80):
    frame = walk._predictions(day, p)
    return strategy.select_pick(frame, day.isoformat(), directory, streak, saver,
                                game_statuses_detailed=status_for(frame),
                                best_streak=best, best_status="trusted")


def digest_add(digest, row):
    digest.update(json.dumps(row, sort_keys=True, default=str).encode() + b"\n")


def grid(directory, mdp):
    first = date(2027, 1, 1)
    days = [first + timedelta(days=i) for i in range((DAY - first).days)]
    resolver = strategy.resolve_policy_decision
    with patch.object(strategy, "resolve_policy_decision", wraps=resolver) as record:
        select(days[0], directory, 20, 21, False)
    template = record.call_args.args[0]
    boundaries = set(mdp["boundaries"]) | set(mdp["tail"].boundaries)
    qualities = sorted({.01, .50, .99} | boundaries | {p - .000001 for p in boundaries})
    digest = hashlib.sha256()
    cases = 0
    for day in days:
        for streak in (0, 9, 20, 53, 54, 55, 56, 57, 120, 400):
            for saver in (False, True):
                for i, p in enumerate(qualities):
                    best, trust = ((streak, "trusted"), (min(400, streak + 4), "trusted"),
                                   (None, "untrusted"))[i % 3]
                    ctx = replace(template, date=day.isoformat(), mdp=mdp, primary_p=p,
                                  second_p=max(0, p - .01) if i % 5 else None,
                                  allow_double=i % 7 != 0, best_streak=best, best_status=trust)
                    decision = resolver(ctx, streak, saver)
                    digest_add(digest, (day, p, saver, asdict(decision)))
                    cases += 1
    print(f"positive-grid dates={len(days)} rows={cases} sha256={digest.hexdigest()}")
    digest = hashlib.sha256()
    sample = days[::13] + days[-5:]
    for day in sample:
        for streak in (20, 55, 56, 57, 120):
            sel = select(day, directory, streak, streak, bool(streak % 2))
            result = sel.pick_result
            bats = None if not result else (result.daily.pick.batter_id, getattr(result.daily.double_down, "batter_id", None))
            digest_add(digest, (day, asdict(sel.decision), bats))
    print(f"positive-selection dates={len(sample)} rows={len(sample)*5} sha256={digest.hexdigest()}")


def makeup(directory, mdp, head):
    rows = ((20, 20, "double"), (20, 25, "skip"), (55, 55, "double"),
            (56, 56, "single"), (57, 57, "double"), (120, 120, "double"))
    for streak, best, want in rows:
        for saver in ("used", "active"):
            direct = strategy.resolve_policy_decision(
                strategy.DecisionContext(.8, .79, True, str(DAY), True, mdp, best, "trusted"), streak, saver == "active")
            assert (direct.action, direct.days_effective) == ("skip", 0)
            assert strategy.effective_pick_bar(streak, str(DAY), saver == "active") is None
            selected = select(DAY, directory, streak, best, saver == "active").decision
            on = datetime(2027, 9, 27, 12, tzinfo=ET)
            state = walk.State("dated", streak, best, saver, .8)
            with walk.offline(on), TemporaryDirectory() as tmp:
                root = Path(tmp) / "picks"
                root.mkdir()
                assert not walk.install_state(root, DAY, state)
                frame = walk._predictions(DAY, .8)
                with patch.object(orchestrator, "run_cascade", return_value=(frame, "fixture")), \
                     patch.object(picks, "get_game_statuses_detailed", return_value=status_for(frame)):
                    _, selection, _ = orchestrator.run_and_pick(walk._config(root), str(DAY))
                assert selection.decision == selected
                if selection.pick_result:
                    assert selection.pick_result.daily.chosen_on["dated"]["effective_date"] == str(DAY)
            outcome = (selected.action, selected.days_effective)
            assert outcome == ((want, 1) if head else ("skip", 0)), (streak, best, saver, outcome)
            print(f"makeup streak={streak} best={best} saver={saver} => {outcome} {selected.objective}")
    if head:
        records = [walk.walk_one(DAY, walk.State("dated", s, b, "used", .8)) for s, b, _ in rows]
        records.append(walk.walk_one(date(2027, 9, 28), walk.State("dated", 57, 57, "used", .8), empty_slate=True))
        assert not walk.check(records)
        print("season-walk makeup and explicit empty records accepted")


def off_days(root):
    day = "2027-10-01"
    now = datetime.fromisoformat(day).replace(tzinfo=ET, hour=12)
    processed = root / "processed"
    processed.mkdir()
    (processed / "pa_synthetic.parquet").touch()
    regular = {"gamePk": 100, "gameType": "R", "officialDate": "2027-09-30",
               "gameDate": f"{day}T19:00:00-04:00", "status": {"abstractGameCode": "P", "detailedState": "Scheduled"},
               "teams": {"away": {"team": {"name": "Away"}}, "home": {"team": {"name": "Home"}}}}
    for label, games in (("empty", []), ("postseason", [regular | {"gameType": "F", "officialDate": day}]),
                         ("resumed only", [regular])):
        payload = json.dumps({"dates": [{"date": day, "games": games}]}).encode()

        def schedule_response(url, **kw):
            assert "/api/v1/schedule?" in url, url
            return io.BytesIO(payload if f"date={day}" in url else b'{"dates": []}')

        directory = root / label
        config = walk._config(directory / "picks")
        config["orchestrator"].update(data_dir=str(processed), heartbeat_path=str(directory / "heartbeat"))
        config["health_checks"]["enabled"] = False
        args = ["--date", day, "--data-dir", str(processed), "--models-dir",
                config["orchestrator"]["models_dir"], "--picks-dir", config["orchestrator"]["picks_dir"]]
        with walk.offline(now), MonkeyPatch.context() as mp:
            mp.setattr(predict, "urlopen", schedule_response)
            mp.setattr(predict, "_refresh_season_data", Mock())
            mp.setattr(pd, "read_parquet", lambda _: pd.DataFrame({"date": [day]}))
            mp.setattr(predict, "compute_all_features", lambda df: df)
            mp.setattr(predict, "train_model", lambda *a, **kw: object())
            mp.setattr(predict, "train_blend", lambda *a, **kw: {})
            mp.setattr(predict, "save_blend", Mock())
            mp.setattr(predict, "_build_feature_lookups", lambda _: {})
            mp.setattr(park_drag, "get_table", lambda: {})
            mp.setattr(park_drag, "get_manifest", lambda: {})
            mp.setattr(scheduler, "retry_urlopen", schedule_response)
            mp.setattr(scheduler, "heartbeat_watchdog", lambda *a, **kw: nullcontext())
            mp.setattr(scheduler, "_idle_until_next_wakeup", Mock())
            mp.setattr(scheduler, "notify_watchdog", Mock())
            mp.setattr(scheduler, "notify_ready", Mock())
            mp.setattr(scheduler, "check_confirmed_lineups", lambda *a: {})
            mp.setattr(scheduler, "_watchdog_ping_sleep", lambda delay: walk._set_clock(scheduler._now_et() + timedelta(seconds=delay)))
            mp.setattr(picks, "get_game_statuses_detailed", Mock(return_value={}))
            mp.setattr(orchestrator, "run_cascade", lambda *a: (predict.run_pipeline(day, str(processed)), "fixture"))
            mp.setattr(posting, "post_to_bluesky", Mock(side_effect=AssertionError("unexpected post")))
            scheduler.run_day(day, config)
            for command in ("run", "preview", "predict-json"):
                opts = args if command != "predict-json" else args[:-2]
                rc, output = walk._invoke(command, *opts)
                assert rc == 0, (command, output)
                assert ("[]" if command == "predict-json" else "No games found") in output, output
            scheduler._run_shadow_prediction(config, day, "fixture")
            assert not posting.post_to_bluesky.called
            assert picks.load_pick(day, Path(config["orchestrator"]["picks_dir"])) is None
            assert picks.load_shadow_pick(day, Path(config["orchestrator"]["picks_dir"])) is None
        print(f"off-day {label}: scheduler, run, preview, remote predict-json and shadow saved no pick")


def main():
    print(f"bts.__file__={bts.__file__}", flush=True)
    mdp = strategy._load_mdp()
    assert mdp["policy_table"] is not None and mdp["tail"] is not None
    print(f"artifacts base={mdp['base_sha256']} tail={mdp['tail'].sha256}", flush=True)
    head = "base/src" not in str(Path(bts.__file__).as_posix())
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        if sys.argv[1] == "grid":
            grid(root / "picks", mdp)
        else:
            makeup(root / "picks", mdp, head)
            off_days(root)


if __name__ == "__main__":
    main()
