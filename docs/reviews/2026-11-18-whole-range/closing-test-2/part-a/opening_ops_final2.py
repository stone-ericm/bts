"""Offline A5 cron install, B7 park-drag refresh, and cold-build probes."""

from datetime import timedelta
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from opening_rehearsal import DAYS, HERE, OLD_DAY, REPO, Clock, Season, at, invoke, read_json, sandbox, write_json


def cron_shim():
    if sys.argv[1] == "uv":
        from bts.cli import cli

        assert sys.argv[2:5] == ["run", "bts", "entry-intent"], sys.argv
        cli.main(args=sys.argv[4:], prog_name="bts")
    else:
        table = Path(os.environ["WR_CRON_TABLE"])
        if sys.argv[2:] == ["-l"]:
            if not table.exists():
                print("no crontab for synthetic", file=sys.stderr)
                raise SystemExit(1)
            sys.stdout.write(table.read_text())
        else:
            assert sys.argv[2:] == ["-"], sys.argv
            table.write_text(sys.stdin.read())


def cron_matrix(root):
    """The only shell edit relocates the script's HOME-derived paths."""
    season = Season(root)
    shell_root = root / "cron-sandbox"
    bindir = shell_root / "bin"
    uv = shell_root / ".local/bin/uv"
    project = shell_root / "projects/bts"
    table = shell_root / "crontab"
    for path in (bindir, uv.parent, project):
        path.mkdir(parents=True)
    (project / ".env").touch()
    for path, mode in ((bindir / "crontab", "crontab"), (uv, "uv")):
        path.write_text(f'#!/bin/sh\nexec "$WR_PYTHON" "$WR_CRON_SHIM" {mode} "$@"\n')
        path.chmod(0o700)
    misc_job = "22 4 * * * /bin/true"
    table.write_text(misc_job + "\n")

    def run(version, action, expected=0):
        source = version / "scripts/cron-setup-hetzner.sh"
        isolated_script = shell_root / "isolated-cron.sh"
        isolated_script.write_text(source.read_text().replace("$HOME/", "$WR_CRON_ROOT/"))
        env = dict(os.environ, PYTHONPATH=str(version / "src"), PYTHONDONTWRITEBYTECODE="1",
                   PATH=f"{bindir}:/usr/bin:/bin", WR_CRON_ROOT=str(shell_root),
                   WR_CRON_TABLE=str(table), WR_CRON_SHIM=str(Path(__file__).resolve()),
                   WR_PYTHON=sys.executable, HEALTHCHECKS_PING_URL="https://example.invalid/capture")
        result = subprocess.run(["/bin/bash", str(isolated_script), action], cwd=project,
                                capture_output=True, text=True, check=False, env=env, timeout=20)
        assert result.returncode == expected, (result.returncode, result.stdout, result.stderr)
        return result.stdout + result.stderr

    config = shell_root / ".bts-orchestrator.toml"

    def configure(mode, intent, required):
        season.write_config(mode, intent, required)
        config.write_text(season.config.read_text())

    configure("private", "research", False)
    old_log = run(HERE / "base", "install")
    old_table = table.read_text()
    assert "check-pick-entered" not in old_table and "40 7 * * *" in old_table
    assert sum("fetch-contest-streak" in line for line in old_table.splitlines()) == 4
    configure("dm", "enter", False)
    refused = run(REPO, "install", 1)
    assert table.read_text() == old_table
    configure("dm", "enter", True)
    shown = run(REPO, "show")
    assert "Would install:" in shown and "check-pick-entered" in shown
    assert table.read_text() == old_table
    installed = run(REPO, "install")
    target = table.read_text()
    assert target.count(misc_job) == 1
    # VARIANT (closing test re-run): FINAL2's watchdog line names its lock file .check-pick-entered.lock (F1b),
    # so count the LINES that run the command rather than occurrences of the word.
    assert sum("bts check-pick-entered" in line for line in target.splitlines()) == target.count("day-outcome") == 1
    assert "30 8 * * *" in target and "40 7 * * *" in target and "45 7 * * *" in target
    fetch_lines = [line for line in target.splitlines() if "fetch-contest-streak" in line]
    assert len(fetch_lines) == 4 and all("--dm-recipient" in line for line in fetch_lines)
    configure("private", "research", False)
    research = run(REPO, "show")
    displayed = research.split("Would install:", 1)[1].splitlines()
    research_fetches = [line for line in displayed if "fetch-contest-streak" in line]
    assert len(research_fetches) == 4 and all("--dm-recipient" not in line for line in research_fetches)
    assert not any("check-pick-entered" in line for line in displayed)
    return {"old_install": old_log, "old_crontab": old_table, "refusal": refused,
            "show": shown, "install": installed, "live_crontab": target,
            "research_show": research}


def park_matrix(root):
    import pandas as pd
    from bts.features import park_drag_producer as producer
    from bts.health.park_drag_freshness import check

    clock = Clock(at(DAYS[0], "07:45"))
    park = root / "data/external/park_drag"
    inputs = park / "producer"
    inputs.mkdir(parents=True)

    def pitches(day, game_pk):
        row = [game_pk, day, "NYY", 95.0, 0.0, -135.0, 0.0, 0.0, 29.0, -20.0]
        return pd.DataFrame([row] * 30, columns=producer.REQUIRED_PITCH_COLS)

    old = pitches(OLD_DAY, 260)
    opener = pitches(DAYS[0], 27240)
    next_day = pitches(DAYS[1], 27250)
    old.to_csv(inputs / "pitches_current.csv.gz", index=False)
    (inputs / "games_meta.csv").write_text("game_pk,start_utc,day_night,temp_f,condition,wind,venue_id,venue\n"
                                         "260,2026-09-27T23:05:00Z,night,70,Clear,,3313,Yankee Stadium\n")
    write_json(park / "park_drag_manifest.json", {"max_source_game_date": OLD_DAY})
    (park / "park_drag_export.csv").write_text("game_pk\n260\n")
    pd.DataFrame([[260, 2026, OLD_DAY, 3313, "Yankee Stadium", 30, .35]],
                 columns=["game_pk", "season", "game_date", "venue_id", "venue",
                          "n_pitch", "cd_trim"]).to_csv(inputs / "game_level_static.csv", index=False)
    requests = []

    def captured_http(url, **kwargs):
        requests.append({"at": clock.now.isoformat(), "url": url})
        if "statcast_search" in url:
            frame = opener if clock.now.date().isoformat() == DAYS[1] else pd.concat([opener, next_day])
            return SimpleNamespace(status_code=200, raise_for_status=lambda: None, text=frame.to_csv(index=False))
        assert "/feed/live" in url, url
        response = {"gameData": {"weather": {"temp": 70, "condition": "Clear"},
                                 "datetime": {"dateTime": f"{DAYS[0]}T23:05:00Z", "dayNight": "night"},
                                 "venue": {"id": 3313, "name": "Yankee Stadium"}}}
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: response)

    real_refresh = producer.refresh
    with sandbox(clock, root):
        with patch.object(producer, "refresh", lambda *a, **kw: real_refresh(*a, http_get=captured_http, **kw)), \
                patch.object(producer, "_jitter_sleep", lambda: None):
            old_status = json.loads(invoke("park-drag-refresh", "--root", park, expected=1).output)
            assert "store season 2026 is behind 2027" in old_status["error"] and not requests
            clock.move(at(DAYS[0], "21:15"))
            opening_health = [vars(alert) for alert in check(park, today=clock.now.date())]
            assert any(alert["level"] == "CRITICAL" for alert in opening_health)
            clock.move(at(DAYS[1], "07:45"))
            old.iloc[:0].to_csv(inputs / "pitches_current.csv.gz", index=False)
            empty = json.loads(invoke("park-drag-refresh", "--root", park, expected=1).output)
            assert not empty["ok"] and not requests
            opener.to_csv(inputs / "pitches_current.csv.gz", index=False)
            reseed = json.loads(invoke("park-drag-refresh", "--root", park).output)
            assert reseed["max_source_game_date"] == DAYS[0] and reseed["current_season_games"] == 1
            after_reseed = [vars(alert) for alert in check(park, today=clock.now.date())]
            assert all(alert["level"] != "CRITICAL" for alert in after_reseed)
            clock.move(at(DAYS[2], "07:45"))
            following = json.loads(invoke("park-drag-refresh", "--root", park).output)
            assert following["max_source_game_date"] == DAYS[1] and following["current_season_games"] == 2
    return {"old_store": old_status, "empty_store": empty, "opening_health": opening_health,
            "after_reseed_health": after_reseed, "populated_reseed": reseed, "next_refresh": following,
            "manifest": read_json(park / "park_drag_manifest.json"), "requests": requests}


def cold_build(root):
    raw = root / "raw"
    out = root / "processed"
    clock = Clock(at(DAYS[0], "03:00") - timedelta(days=1))
    with sandbox(clock, root):
        args = ["data", "build", "--seasons", "2027", "--raw-dir", raw, "--out-dir", out]
        absent = invoke(*args).output
        write_json(raw / "2027/10.json", {"gameData": {"game": {"type": "S"}}})
        write_json(raw / "2027/10_weather.json", {"humidity": 20})
        spring_only = invoke(*args).output
        assert "nothing written" in absent and "nothing written" in spring_only
        assert not out.exists()
    return {"absent": absent, "spring_only_with_weather": spring_only}


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cron_shim()
    else:
        with TemporaryDirectory(prefix="wr-opening-ops-") as directory:
            root = Path(directory)
            results = {"cron": cron_matrix(root), "park": park_matrix(root), "cold_build": cold_build(root)}
            write_json(HERE / "opening_ops.json", results)
            print("A5 cron, B7 park-drag and pre-opener data build assertions passed")
