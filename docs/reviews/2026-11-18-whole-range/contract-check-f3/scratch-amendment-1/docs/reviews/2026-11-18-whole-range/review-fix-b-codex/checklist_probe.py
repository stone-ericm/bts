"""Rehearse B8a's real CLI steps on failed, settled and unsettled cutovers."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx
from click.testing import CliRunner
from probe_support import REVIEW, NOW, REV, auto, dump, fetch, invoke, offline, put
from bts import cli, contest_state

YEAR = str(NOW.year)
DAY = NOW.date().isoformat()
rows = {}
with TemporaryDirectory(dir=REVIEW, prefix="checklist-fixtures-") as directory:
    root = Path(directory)
    config = root / "orchestrator.toml"
    config.write_text('[scheduler]\ncontest_state_required=true\npick_delivery="dm"\n'
                      'entry_intent="enter"\n[health_checks]\ncontest_state_expected=true\n')
    intent = CliRunner().invoke(cli.cli, ["entry-intent", "--config", str(config)])
    assert (intent.exit_code, intent.stdout.strip()) == (0, "enter")
    for scenario in ("settled", "unsettled", "incomplete-then-retry"):
        picks = root / scenario / "picks"
        legacy = put(picks, "contest_streak.json",
                     auto(schema_version="bts_contest_streak_auto_v1", season=2026))
        put(picks, "saver_state.json", {"season": 2026, "state": "used"})
        changes = []

        def keep(label, result):
            changes.append({"step": label, "exit": result.exit_code,
                            "output": result.output.replace(str(picks.parent), "FIXTURE")})
            return result

        with offline() as (_, profile):
            assert keep("revoke restored state", invoke(
                picks, "season-contract", "revoke", "--season", YEAR,
                "--reason", "restored from the offseason snapshot",
            )).exit_code == 0
            assert keep("pre-enable status", invoke(picks, "season-contract", "status")).exit_code == 0
            assert keep("morning observation", invoke(
                picks, "set-contest-streak", "--date", DAY, "--streak", "0",
                "--best-streak", "0", "--saver", "not_earned",
            )).exit_code == 0
            before = legacy.read_bytes()
            assert keep("pre-enable fetch", fetch(picks)).exit_code == 0
            assert legacy.read_bytes() == before
            assert keep("compare observations", invoke(picks, "season-contract", "status")).exit_code == 0
            assert keep("enable", invoke(picks, "season-contract", "enable", "--season", YEAR)).exit_code == 0
            good = dict(activeStreak=7, seasonBestStreak=9, predictions=[dict(roundId=1, result="hit")])
            if scenario == "unsettled":
                good["predictions"] = []
            if scenario == "incomplete-then-retry":
                profile.side_effect = httpx.ConnectError("offline")
                assert keep("failed fetch", fetch(picks)).exit_code == 2
                result = keep("dated masks unready automatic", invoke(picks, "season-contract", "status"))
                assert "live decision now: accepted" in result.output
                if REV == "head":
                    assert "automatic path alone: refused" in result.output
                today = contest_state.load_decision_streak_state(picks, now=NOW, require_contest_state=True)
                assert today.observation_kind == "dated"
                profile.side_effect = None
            profile.return_value = good
            assert keep("successful post-enable fetch", fetch(picks)).exit_code == 0
            status = keep("check independent readiness", invoke(picks, "season-contract", "status"))
            if REV == "head":
                assert "automatic path alone: accepted" in status.output
            read = keep("read saver", invoke(picks, "saver-state", "--season", YEAR))
            assert read.exit_code == 0
            if scenario == "unsettled":
                assert "uninitialized" in read.output
                assert keep("initialize from MLB", invoke(
                    picks, "saver-state", "--init", "not_earned", "--season", YEAR)).exit_code == 0
                assert "not_earned" in keep("reread", invoke(picks, "saver-state", "--season", YEAR)).output
            else:
                assert "not_earned" in read.output
                if REV == "base":
                    assert keep("old unconditional initialize", invoke(
                        picks, "saver-state", "--init", "not_earned", "--season", YEAR)).exit_code == 1
            assert json.loads(legacy.read_text())["active_streak"] == 7
        rows[scenario] = changes

dump("checklist", rows)
