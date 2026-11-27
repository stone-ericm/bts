import io
import json
import socket
import tempfile
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from unittest.mock import patch

from bts import orchestrator, picks, scheduler, strategy
from bts.model import predict as model
from scripts.audit.season_walk import walk

from probe_scheduler import game, offline

D = "2027-09-27"
DAY = date.fromisoformat(D)


def schedule(games, listed_day=D):
    return {"dates": [{"date": listed_day, "games": games}]}


def fetch_payload(body):
    return io.BytesIO(json.dumps(body).encode())


def model_slots(games, listed_day=D):
    feed = {"gameData": {"venue": {"id": 1},
                         "teams": {"away": {"id": 1, "abbreviation": "A"},
                                   "home": {"id": 2, "abbreviation": "H"}}},
            "liveData": {"boxscore": {"teams": {"away": {"players": {}},
                                               "home": {"players": {}}}},
                         "plays": {"allPlays": []}}}

    def fetch(url, **kwargs):
        return fetch_payload(schedule(games, listed_day) if "/schedule?" in url else feed)

    with patch.object(model, "urlopen", fetch), patch.object(
        model, "_fetch_prior_lineup", return_value=[{"batter_id": 510, "batter_name": "Projected", "lineup": 1}]
    ):
        return [[s["game_pk"], s["projected"]] for s in model._fetch_game_slots(D)]


def run_cli(command, root, frame, games):
    with walk.offline(datetime(2027, 9, 27, 10, tzinfo=walk.ET)):
        problems = walk.install_state(root, DAY, walk.State("dated", 57, 57, "used", .8))
        resp = schedule(games)
        posts = []
        with patch.object(model, "run_pipeline", return_value=frame), patch.object(
            picks, "_fetch_json", return_value=resp
        ), patch("bts.posting.should_post_now", return_value=True), patch(
            "bts.posting.post_to_bluesky", side_effect=lambda text: posts.append(text) or "at://stub"
        ):
            code, message = walk._invoke(command, "--date", D, "--picks-dir", root,
                                         "--models-dir", root / "models")
        daily = picks.load_pick(D, root)
    return {"setup": problems, "exit": code, "text": message, "posts": posts,
            "pick": asdict(daily) if daily else None}


def run_orchestrator(root, frame, games, *, scheduler_check=False):
    config = walk._config(root)
    with walk.offline(datetime(2027, 9, 27, 10, tzinfo=walk.ET)):
        problems = walk.install_state(root, DAY, walk.State("dated", 57, 57, "used", .8))
        with patch.object(orchestrator, "run_cascade", return_value=(frame, "captured")), +                patch.object(picks, "_fetch_json", return_value=schedule(games)):
            if scheduler_check:
                from contextlib import nullcontext
                with patch.object(scheduler, "heartbeat_watchdog", return_value=nullcontext()), +                        patch.object(scheduler, "count_new_confirmations", return_value=1):
                    result = scheduler.run_single_check(D, [70000], set(), config, .03)
                summary = result["skip_summary"]
                return {"setup": problems, "summary": summary, "rendered": scheduler.format_skip_dm(D, summary)}
            _, sel, _ = orchestrator.run_and_pick(config, D)
    return {"setup": problems, "decision": asdict(sel.decision) if sel and sel.decision else None,
            "no_pick_reason": sel.no_pick_reason if sel else None}


def main(root):
    regular = [game(D, n) for n in range(3)]
    nonreg = [game(D, n, kind="F") for n in range(3)]
    prior = [game("2027-09-26", n) for n in range(3)]
    frame = walk._predictions(DAY, .8)
    stale = walk._predictions(date(2027, 9, 26), .8)
    out = {"real_slots": {"postseason": model_slots(nonreg),
                          "spring": model_slots([game(D, kind="S")]),
                          "projected": model_slots(regular),
                          "resume": model_slots([game(D, official="2027-09-26")]),
                          "wrong_day_response_future": model_slots([game("2027-09-28")], "2027-09-28"),
                          "missing_game_type": model_slots([{k: v for k, v in game(D).items() if k != "gameType"}])}}
    try:
        with patch.object(model, "urlopen", return_value=io.BytesIO(b"{broken")):
            model._fetch_game_slots(D)
    except Exception as error:
        out["prediction_schedule_error"] = type(error).__name__
    for command in ("run", "preview", "orchestrator"):
        for name, rows, games in (("regular", frame, regular), ("empty", frame.iloc[:0], []),
                                  ("stale_offday", stale, []), ("foreign_regular", stale, prior),
                                  ("invalid_nonregular_rows", frame, nonreg)):
            picks_path = root / f"{command}-{name}" / "picks"
            picks_path.mkdir(parents=True)
            call = run_orchestrator if command == "orchestrator" else lambda *a: run_cli(command, *a)
            out[f"{command}_{name}"] = call(picks_path, rows, games)
    picks_path = root / "skip-bar" / "picks"
    picks_path.mkdir(parents=True)
    out["no_verified_candidate_skip_notice"] = run_orchestrator(picks_path, frame, [],
                                                               scheduler_check=True)
    with tempfile.TemporaryDirectory() as tmp:
        mdp = strategy._load_mdp()
        context = strategy.DecisionContext(.8, .79, True, "2027-12-31", True, mdp, 57, "trusted")
        out["decide_action_without_schedule"] = strategy.decide_action(context, 57, False)
        try:
            statuses = {int(pk): {"abstract": "P", "detailed": "Scheduled"} for pk in frame.game_pk}
            strategy.select_pick(frame, "bad-date", Path(tmp), 57, False, game_statuses_detailed=statuses)
        except Exception as error:
            out["invalid_date"] = f"{type(error).__name__}: {error}"
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp, patch.object(socket.socket, "connect", offline), +            patch.object(socket, "getaddrinfo", offline):
        main(Path(tmp))
