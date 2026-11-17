"""Adversarial Probe 3 (v4): Verify subworker comparison + run direct branch worker/audit probes."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

REVIEW_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REVIEW_DIR))

import probe_workers_cron_sched_orch_v3 as p3


def compare_clean(base: dict, branch: dict) -> dict:
    assert base["load_errors"] == branch["load_errors"], (base["load_errors"], branch["load_errors"])

    b0_po = base["priv_research_orchestrate"]
    b1_po = branch["priv_research_orchestrate"]
    assert b0_po["cli_exit"] == 0 and len(b0_po["cli_posts"]) == 1 and b0_po["cli_pick_saved"] is True
    assert b1_po["cli_exit"] == 2 and b1_po["cli_posts"] == [] and b1_po["cli_pick_saved"] is False
    assert b1_po["cli_stderr"].startswith("bts orchestrate: refusing to run: delivery 'public' requires contest state.")
    assert b0_po["direct_ok"] is True and b1_po["direct_ok"] is True
    assert b0_po["direct_posts"] == b1_po["direct_posts"] and len(b1_po["direct_posts"]) == 1

    warn_prefix = (
        "bts schedule: WARNING (a real run would refuse to start): "
        "scheduler delivery 'public' requires contest state. "
        "Optional reads can use a contest observation without the season-coverage check. "
        "Set contest_state_required = true under [scheduler] and "
        "contest_state_expected = true under [health_checks].\n"
    )
    for kind in ("game", "off"):
        d0 = base["schedule_dry_run"][kind]
        d1 = branch["schedule_dry_run"][kind]
        assert d1["stderr"] == warn_prefix + d0["stderr"], (kind, d0["stderr"], d1["stderr"])
        for k in ("exit", "stdout", "fetches", "ready_calls", "watchdog_calls", "sleeps", "hb", "picks_exists"):
            assert d1[k] == d0[k], (kind, k, d0[k], d1[k])

    c0 = base["cron"]
    c1 = branch["cron"]
    assert c0["off_show"]["rc"] == 0 and c0["off_install"]["rc"] == 0
    for act in ("show", "install"):
        r = c1[f"off_{act}"]
        assert r["rc"] == 1 and r["stdout"] == "" and r["crontab_calls"] == []
        err_lines = r["stderr"].splitlines()
        assert len(err_lines) == 2
        assert err_lines[0].startswith("entry-intent: scheduler delivery 'dm' and scheduler.entry_intent = 'enter' require contest state.")
        assert err_lines[1] == f"ERROR: refusing to {act}: no valid [scheduler].entry_intent in <HOME>/.bts-orchestrator.toml (see entry-intent above)"
        assert r["cron_after"] == "0 9 * * * echo foreign\n*/15 10-23 * * * uv run bts check-pick-entered # BTS-HETZNER\n"
    assert c1["off_remove"] == c0["off_remove"] and c1["off_remove"]["rc"] == 0
    for flag_kind in ("req", "exp"):
        assert c1[f"{flag_kind}_show"] == c0[f"{flag_kind}_show"]
        assert c1[f"{flag_kind}_install"] == c0[f"{flag_kind}_install"]
        assert c1[f"{flag_kind}_install"]["stdout"] == c0["off_install"]["stdout"]
        assert c1[f"{flag_kind}_install"]["cron_after"] == c0["off_install"]["cron_after"]

    assert branch["day_outcome"] == base["day_outcome"]
    return {
        "load_errors_identical": True,
        "priv_research_orchestrate_verified": True,
        "schedule_dry_run_game_and_off_day_verified": True,
        "cron_installer_show_install_remove_verified": True,
        "day_outcome_cases_identical": len(branch["day_outcome"]),
    }


def main() -> None:
    p3.install_audit_hooks()
    base_out = REVIEW_DIR / "details_base_v3.json"
    branch_out = REVIEW_DIR / "details_branch_v3.json"
    if not (base_out.exists() and branch_out.exists()):
        env = dict(os.environ, TZ="America/New_York", PYTHONDONTWRITEBYTECODE="1")
        for which, out_file in (("base", base_out), ("branch", branch_out)):
            subprocess.run([sys.executable, str(REVIEW_DIR / "probe_workers_cron_sched_orch_v3.py"), "--worker", which, str(out_file)], env=env, check=True)
    sub_cmp = compare_clean(json.loads(base_out.read_text()), json.loads(branch_out.read_text()))
    direct_rep = p3.run_branch_direct_adversarial_probes()
    full_summary = {"base_vs_branch_details": sub_cmp, "branch_direct_probes": direct_rep}
    (REVIEW_DIR / "details_summary.json").write_text(json.dumps(full_summary, indent=2))
    print(json.dumps(full_summary, indent=2))


if __name__ == "__main__":
    main()
