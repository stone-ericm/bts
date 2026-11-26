#!/usr/bin/env python3
"""Item 6: Probe every operator CLI command and sequence in INCIDENT.md and docs/ops/2027-season-start.md against a custom picks_dir."""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sys
import tempfile
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from click.testing import CliRunner
import bts.cli as cli_mod
from bts.cli import cli

SCRATCH_PARENT = REPO / ".codex-review/wr-d3b/tmp"


def main() -> None:
    SCRATCH_PARENT.mkdir(parents=True, exist_ok=True)
    runner = CliRunner()
    et = ZoneInfo("America/New_York")
    pinned = datetime(2027, 4, 2, 9, 30, 0, tzinfo=et)
    orig_now = cli_mod._now_et
    try:
        cli_mod._now_et = lambda: pinned
        with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
            root = Path(td)
            default_picks = root / "default_picks"
            custom_picks = root / "custom_picks"
            default_picks.mkdir()
            custom_picks.mkdir()
            # Plant a misleading saver flag in default_picks to verify --picks-dir isolates custom_picks!
            r_def = runner.invoke(cli, ["saver-state", "--init", "used", "--season", "2027", "--picks-dir", str(default_picks)])
            assert r_def.exit_code == 0, r_def.output

            # 1. B8a revoke on custom_picks
            r_rev = runner.invoke(cli, [
                "season-contract", "revoke", "--season", "2027",
                "--reason", "restored from the offseason snapshot",
                "--picks-dir", str(custom_picks),
            ])
            assert r_rev.exit_code == 0, r_rev.output

            # 2. status before dated observation -> live decision now: refused
            r_st1 = runner.invoke(cli, ["season-contract", "status", "--picks-dir", str(custom_picks)])
            assert r_st1.exit_code == 0 and "live decision now: refused" in r_st1.output, r_st1.output

            # 3. B8a set-contest-streak dated observation on custom_picks
            r_obs = runner.invoke(cli, [
                "set-contest-streak", "--date", "2027-04-02",
                "--streak", "0", "--best-streak", "0", "--saver", "not_earned",
                "--picks-dir", str(custom_picks),
            ])
            assert r_obs.exit_code == 0, r_obs.output

            # 4. INCIDENT.md §5 / B8a check: live decision now: accepted
            r_st2 = runner.invoke(cli, ["season-contract", "status", "--picks-dir", str(custom_picks)])
            assert r_st2.exit_code == 0 and "live decision now: accepted" in r_st2.output, r_st2.output

            # 5. B8a entry-intent check with both flags + dm + enter
            cfg = root / "orch.toml"
            cfg.write_text(
                f'[orchestrator]\npicks_dir = "{custom_picks}"\n'
                '[scheduler]\npick_delivery = "dm"\nentry_intent = "enter"\ncontest_state_required = true\n'
                '[health_checks]\ncontest_state_expected = true\n'
            )
            r_ei = runner.invoke(cli, ["entry-intent", "--config", str(cfg)])
            assert r_ei.exit_code == 0 and r_ei.output.strip() == "enter", r_ei.output

            # 6. deploy-canary on that exact prepared B8a config -> accepted, exit 0!
            r_can = runner.invoke(cli, ["deploy-canary", "--config", str(cfg)])
            assert r_can.exit_code == 0 and "Deploy canary: live contest state accepted (dated, streak 0)" in r_can.output

            # 7. B8a saver-state --show and --init with explicit --season and --picks-dir!
            r_sv_show = runner.invoke(cli, [
                "saver-state", "--show", "--season", "2027", "--picks-dir", str(custom_picks),
            ])
            assert r_sv_show.exit_code == 0 and "uninitialized" in r_sv_show.output, r_sv_show.output
            r_sv_init = runner.invoke(cli, [
                "saver-state", "--init", "not_earned", "--season", "2027", "--picks-dir", str(custom_picks),
            ])
            assert r_sv_init.exit_code == 0, r_sv_init.output
            r_sv_after = runner.invoke(cli, [
                "saver-state", "--show", "--season", "2027", "--picks-dir", str(custom_picks),
            ])
            assert r_sv_after.exit_code == 0 and "not_earned" in r_sv_after.output, r_sv_after.output

            print(json.dumps({
                "revoke_exit": r_rev.exit_code,
                "status_pre_obs": "live decision now: refused",
                "status_post_obs": "live decision now: accepted",
                "entry_intent": r_ei.output.strip(),
                "deploy_canary": r_can.output.strip(),
                "saver_show_custom_before_init": "uninitialized (isolated from default_picks=used)",
                "saver_show_custom_after_init": "not_earned",
                "status": "ALL OPERATOR CLI SEQUENCES VERIFIED",
            }, indent=2))
    finally:
        cli_mod._now_et = orig_now


if __name__ == "__main__":
    main()
