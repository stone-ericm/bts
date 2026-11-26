import json
import socket
import tempfile
from datetime import date, datetime
from pathlib import Path
from unittest.mock import patch

from bts import orchestrator, picks
from bts.model import predict as model
from scripts.audit.season_walk import walk

from probe_scheduler import D, game, offline


def attempt(command, kind, root):
    day = date.fromisoformat(D)
    if kind == "prior_date":
        row_day = date(2027, 9, 26)
        now = datetime(2027, 9, 27, 19, 20, tzinfo=walk.ET)
    else:
        row_day = day
        now = datetime(2027, 9, 27, 17, 0, tzinfo=walk.ET)
    frame = walk._predictions(row_day, .8)
    games = [game(row_day.isoformat(), i, kind="F" if kind == "postseason" else "R")
             for i in range(3)]
    schedule = {"dates": [{"date": row_day.isoformat(), "games": games}]}
    with walk.offline(now):
        setup = walk.install_state(root, day, walk.State("dated", 57, 57, "used", .8))
        walk._set_clock(now)
        posts = []
        config = walk._config(root)
        config.setdefault("bluesky", {})["dm_recipient"] = "offline.test"
        config_file = root.parent / "orchestrator.toml"
        config_file.touch()
        with (patch.object(model, "run_pipeline", return_value=frame),
              patch.object(picks, "_fetch_json", return_value=schedule),
              patch.object(orchestrator, "run_cascade", return_value=(frame, "stub")),
              patch.object(orchestrator, "load_config", return_value=config),
              patch("bts.posting._now_et", return_value=now),
              patch("bts.posting.post_to_bluesky", side_effect=lambda text: posts.append(text) or "at://stub")):
            args = ("--config", config_file) if command == "orchestrate" else (
                "--picks-dir", root, "--models-dir", root / "models")
            code, output = walk._invoke(command, "--date", D, *args)
        daily = picks.load_pick(D, root)
    return {"setup": setup, "exit": code, "output": output, "posts": posts,
            "objective": daily.policy_decision["objective"] if daily else None,
            "saved_game_time": daily.pick.game_time if daily else None}


if __name__ == "__main__":
    results = {}
    with (tempfile.TemporaryDirectory() as tmp, patch.object(socket, "getaddrinfo", offline),
          patch.object(socket.socket, "connect", offline)):
        for command in ("run", "orchestrate"):
            for kind in ("prior_date", "postseason"):
                case = f"{command}-{kind}"
                picks_dir = Path(tmp) / case / "picks"
                picks_dir.mkdir(parents=True)
                results[case] = attempt(command, kind, picks_dir)
    print(json.dumps(results, indent=2))
