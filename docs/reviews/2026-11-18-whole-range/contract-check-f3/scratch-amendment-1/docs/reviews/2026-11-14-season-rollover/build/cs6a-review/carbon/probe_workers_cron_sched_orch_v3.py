"""Adversarial Probe 3 (v3): Real workers on uncovered contest state (Item 2),
`bts orchestrate` snapshot/mutation/errors (Item 4),
`check_entry_intent` + real `cron-setup-hetzner.sh` + `day_outcome` parity (Item 5),
`bts schedule` refusal audit + `--dry-run` base-vs-branch parity (Item 6),
and `config/orchestrator.example.toml` diff/parse check (Item 7).
"""
from __future__ import annotations

import contextlib
import copy
from datetime import datetime
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import tomllib
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[2]
REVIEW_DIR = Path(__file__).resolve().parent
BASE_SRC = REVIEW_DIR / "base_tree" / "src"
BRANCH_SRC = REPO / "src"
CRON_SCRIPT = REPO / "scripts" / "cron-setup-hetzner.sh"

ET = ZoneInfo("America/New_York")
DAY = "2027-07-15"
NOW = datetime(2027, 7, 15, 12, 0, 0, tzinfo=ET)

BLOCKED_AUDIT_EVENTS = frozenset({
    "socket.connect",
    "socket.connect_ex",
    "socket.getaddrinfo",
    "socket.gethostbyname",
    "socket.gethostbyname_ex",
    "socket.gethostbyaddr",
    "socket.getnameinfo",
    "socket.sendto",
    "urllib.Request",
})

AUDIT_FS_LOG: list[tuple[str, str]] = []
AUDIT_WATCH_PREFIXES: list[str] = []


def install_audit_hooks() -> None:
    def _hook(event: str, args: tuple) -> None:
        if event in BLOCKED_AUDIT_EVENTS:
            raise RuntimeError(f"Network/DNS forbidden by review audit hook: {event} {args!r}")
        if AUDIT_WATCH_PREFIXES and event in {
            "open", "os.stat", "os.listdir", "os.scandir", "os.mkdir", "os.rename", "os.remove", "os.unlink"
        }:
            target = args[0] if args else None
            if isinstance(target, (str, os.PathLike)):
                s = str(Path(target))
                for pfx in AUDIT_WATCH_PREFIXES:
                    if s == pfx or s.startswith(pfx + "/"):
                        AUDIT_FS_LOG.append((event, s))

    sys.addaudithook(_hook)


class FixedDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW.replace(tzinfo=None) if tz is None else NOW.astimezone(tz)


def enter_all(stack: contextlib.ExitStack, cms: list) -> None:
    for cm in cms:
        stack.enter_context(cm)


def write_uncovered_2026_state(picks_dir: Path) -> None:
    acc = picks_dir / "account_state"
    acc.mkdir(parents=True, exist_ok=True)
    (acc / "contest_streak.json").write_text(json.dumps({
        "schema_version": "bts_contest_streak_auto_v1",
        "active_streak": 7,
        "source": "mlb_bts_profile",
        "source_date": "2026-09-28",
        "recorded_at": "2026-09-28T16:00:00Z",
    }))
    (picks_dir / "streak.json").write_text(json.dumps({"streak": 3, "saver_available": True, "season": 2027}))


def run_subworker(which: str, out_file: Path) -> None:
    install_audit_hooks()
    target_src = BASE_SRC if which == "base" else BRANCH_SRC
    cleaned = [p for p in sys.path if Path(p if p else ".").resolve() != BRANCH_SRC.resolve()]
    sys.path[:] = [str(target_src)] + cleaned

    import bts
    assert Path(bts.__file__).resolve().is_relative_to(target_src.resolve())

    import pandas as pd
    from click.testing import CliRunner
    from bts import cli, contest_state, daily_decision, day_outcome, entry_receipt, heartbeat, orchestrator, picks, scheduler

    runner = CliRunner()
    out: dict = {}

    def make_df() -> pd.DataFrame:
        return pd.DataFrame([{
            "batter_name": "Jacob Wilson",
            "batter_id": 700363,
            "team": "ATH",
            "lineup": 1,
            "pitcher_name": "Jose Suarez",
            "pitcher_id": 660761,
            "p_hit_pa": 0.312,
            "p_game_hit": 0.812,
            "flags": "",
            "game_pk": 778899,
            "game_time": f"{DAY}T23:10:00Z",
        }])

    # 1. Orchestrate & schedule load errors comparison
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        missing = root / "does_not_exist.toml"
        bad_toml = root / "bad.toml"
        bad_toml.write_text("[scheduler\nx = 1\n")
        dir_cfg = root / "a_dir"
        dir_cfg.mkdir()
        unreadable = root / "unreadable.toml"
        unreadable.write_text("[scheduler]\ncontest_state_required = true\n")
        unreadable.chmod(0)

        load_errs = {}
        try:
            for label, p in [("missing", missing), ("bad_toml", bad_toml), ("dir", dir_cfg), ("unreadable", unreadable)]:
                r_o = runner.invoke(cli.cli, ["orchestrate", "--date", DAY, "--config", str(p)])
                r_s = runner.invoke(cli.cli, ["schedule", "--date", DAY, "--config", str(p)])
                norm = lambda s: s.replace(str(root), "<TMP>")
                load_errs[label] = {
                    "orch": [r_o.exit_code, type(r_o.exception).__name__ if r_o.exception else None, norm(r_o.stdout), norm(r_o.stderr)],
                    "sched": [r_s.exit_code, type(r_s.exception).__name__ if r_s.exception else None, norm(r_s.stdout), norm(r_s.stderr)],
                }
        finally:
            unreadable.chmod(0o600)
        out["load_errors"] = load_errs

    # 2. Private/research config through real `bts orchestrate` on uncovered 2026 state
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        picks_dir = root / "picks"
        write_uncovered_2026_state(picks_dir)
        cfg_file = root / "priv.toml"
        cfg_file.write_text(
            f'tiers = []\n[orchestrator]\npicks_dir = "{picks_dir}"\n'
            '[bluesky]\ndm_recipient = "op.bsky.social"\n'
            '[scheduler]\npick_delivery = "private"\nentry_intent = "research"\n'
        )
        posts, dms, posts_direct = [], [], []
        picks_dir2 = root / "picks_direct"
        write_uncovered_2026_state(picks_dir2)
        cfg_file2 = root / "priv_direct.toml"
        cfg_file2.write_text(
            f'tiers = []\n[orchestrator]\npicks_dir = "{picks_dir2}"\n'
            '[bluesky]\ndm_recipient = "op.bsky.social"\n'
            '[scheduler]\npick_delivery = "private"\nentry_intent = "research"\n'
        )
        with contextlib.ExitStack() as st:
            enter_all(st, [
                patch("datetime.datetime", FixedDateTime),
                patch.object(contest_state, "datetime", FixedDateTime),
                patch.object(picks, "datetime", FixedDateTime),
                patch.object(picks, "_git_head_sha", lambda *_a, **_k: "0" * 40),
                patch.object(orchestrator, "run_cascade", return_value=(make_df(), "local")),
                patch.object(picks, "get_game_statuses_detailed", return_value={778899: {"abstract": "P", "detailed": "Pre-Game"}}),
                patch("bts.strategy.get_game_statuses", return_value={778899: "P"}),
                patch("bts.posting.should_post_now", return_value=True),
                patch("bts.posting.post_to_bluesky", side_effect=lambda t: posts.append(t) or "at://pub/1"),
                patch("bts.dm.send_dm", side_effect=lambda r, t: dms.append((r, t)) or "dm-1"),
            ])
            r_priv_orch = runner.invoke(cli.cli, ["orchestrate", "--date", DAY, "--config", str(cfg_file)])
            with patch("bts.posting.post_to_bluesky", side_effect=lambda t: posts_direct.append(t) or "at://pub/2"):
                direct_ok = orchestrator.orchestrate(cfg_file2, DAY)

        out["priv_research_orchestrate"] = {
            "cli_exit": r_priv_orch.exit_code,
            "cli_stderr": r_priv_orch.stderr,
            "cli_posts": posts,
            "cli_dms": dms,
            "cli_pick_saved": (picks_dir / f"{DAY}.json").exists(),
            "direct_ok": direct_ok,
            "direct_posts": posts_direct,
            "direct_pick_saved": (picks_dir2 / f"{DAY}.json").exists(),
        }

    # 3. `bts schedule --dry-run` on game day and off day
    dry_results = {}
    for has_game in (True, False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            picks_dir = root / "picks"
            hb = root / "hb.json"
            cfg_file = root / "sched.toml"
            cfg_file.write_text(f'[orchestrator]\npicks_dir = "{picks_dir}"\nheartbeat_path = "{hb}"\n')
            game = {
                "gamePk": 778899,
                "gameDate": f"{DAY}T23:10:00Z",
                "teams": {"away": {"team": {"name": "ATH"}}, "home": {"team": {"name": "DET"}}},
                "status": {"abstractGameState": "Preview", "detailedState": "Scheduled"},
            }
            fetches, sleeps = [], []
            ready, watchdog = Mock(), Mock()
            with contextlib.ExitStack() as st:
                enter_all(st, [
                    patch.object(scheduler, "_now_et", lambda: NOW),
                    patch.object(scheduler, "datetime", FixedDateTime),
                    patch.object(heartbeat, "datetime", FixedDateTime),
                    patch.object(scheduler, "fetch_schedule", side_effect=lambda d: fetches.append(d) or ([game] if has_game else [])),
                    patch.object(scheduler, "notify_ready", ready),
                    patch.object(scheduler, "notify_watchdog", watchdog),
                    patch.object(scheduler, "_watchdog_ping_sleep", side_effect=lambda s: sleeps.append(s)),
                ])
                r_dry = runner.invoke(cli.cli, ["schedule", "--config", str(cfg_file), "--date", DAY, "--dry-run"])
            hb_data = json.loads(hb.read_text()) if hb.exists() else None
            if hb_data and "pid" in hb_data:
                hb_data["pid"] = 0
            dry_results["game" if has_game else "off"] = {
                "exit": r_dry.exit_code,
                "stdout": r_dry.stdout,
                "stderr": r_dry.stderr,
                "fetches": fetches,
                "ready_calls": ready.call_count,
                "watchdog_calls": watchdog.call_count,
                "sleeps": sleeps,
                "hb": hb_data,
                "picks_exists": picks_dir.exists(),
            }
    out["schedule_dry_run"] = dry_results

    # 4. Real `scripts/cron-setup-hetzner.sh` under throwaway HOME + shims
    def run_cron_box(toml_body: str, action: str, initial_cron: bytes) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            tpath = Path(tmp)
            home = tpath / "home"
            (home / "projects" / "bts").mkdir(parents=True)
            (home / "projects" / "bts" / ".env").write_text("")
            (home / ".local" / "bin").mkdir(parents=True)
            uv = home / ".local" / "bin" / "uv"
            uv.write_text(
                '#!/bin/sh\n'
                '[ "$1 $2 $3" = "run bts entry-intent" ] || { echo "uv shim: unexpected $*" >&2; exit 97; }\n'
                'shift 2\n'
                f'exec "{sys.executable}" -c "import sys; '
                f'sys.path[:] = [{str(target_src)!r}] + [p for p in sys.path if {str(BRANCH_SRC)!r} not in p]; '
                'from bts.cli import cli; cli()" "$@"\n'
            )
            uv.chmod(uv.stat().st_mode | stat.S_IXUSR)
            shims = tpath / "shims"
            shims.mkdir()
            cron_file = tpath / "crontab.txt"
            cron_file.write_bytes(initial_cron)
            ctab = shims / "crontab"
            ctab.write_text(
                f'#!/bin/sh\nf="{cron_file}"\necho "$1" >> "$f.calls"\n'
                'case "$1" in\n'
                '  -l) [ -f "$f" ] && cat "$f" || { echo "no crontab for user" >&2; exit 1; } ;;\n'
                '  -) cat > "$f.new" && mv "$f.new" "$f" ;;\n'
                '  *) exit 98 ;;\n'
                'esac\n'
            )
            ctab.chmod(ctab.stat().st_mode | stat.S_IXUSR)
            (home / ".bts-orchestrator.toml").write_text(toml_body)
            env = {
                "HOME": str(home),
                "PATH": f"{shims}:/usr/bin:/bin",
                "HEALTHCHECKS_PING_URL": "https://hc-ping.invalid/probe",
                "TZ": "America/New_York",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
            proc = subprocess.run(["bash", str(CRON_SCRIPT), action], env=env, capture_output=True, text=True, timeout=30)
            norm = lambda s: s.replace(str(home), "<HOME>")
            calls_file = Path(f"{cron_file}.calls")
            return {
                "rc": proc.returncode,
                "stdout": norm(proc.stdout),
                "stderr": norm(proc.stderr),
                "crontab_calls": calls_file.read_text().splitlines() if calls_file.exists() else [],
                "cron_after": norm(cron_file.read_text()),
            }

    init_cron = b"0 9 * * * echo foreign\n*/15 10-23 * * * uv run bts check-pick-entered # BTS-HETZNER\n"
    dm_enter_off = '[scheduler]\nentry_intent = "enter"\npick_delivery = "dm"\n'
    dm_enter_req = '[scheduler]\nentry_intent = "enter"\npick_delivery = "dm"\ncontest_state_required = true\n'
    dm_enter_exp = '[scheduler]\nentry_intent = "enter"\npick_delivery = "dm"\n[health_checks]\ncontest_state_expected = true\n'

    out["cron"] = {
        "off_show": run_cron_box(dm_enter_off, "show", init_cron),
        "off_install": run_cron_box(dm_enter_off, "install", init_cron),
        "off_remove": run_cron_box(dm_enter_off, "remove", init_cron),
        "req_show": run_cron_box(dm_enter_req, "show", b"0 9 * * * echo foreign\n"),
        "req_install": run_cron_box(dm_enter_req, "install", b"0 9 * * * echo foreign\n"),
        "exp_show": run_cron_box(dm_enter_exp, "show", b"0 9 * * * echo foreign\n"),
        "exp_install": run_cron_box(dm_enter_exp, "install", b"0 9 * * * echo foreign\n"),
    }

    # 5. `day_outcome.gather` and `bts day-outcome` across every DayOutcome state on live-gate-failing configs
    day_cases = {}
    configs_to_test = {
        "dm_enter_no_flags": '[scheduler]\nentry_intent = "enter"\npick_delivery = "dm"\n',
        "pub_enter_no_flags": '[scheduler]\nentry_intent = "enter"\npick_delivery = "public"\n',
        "priv_research_bad_health": 'health_checks = 1\n[scheduler]\nentry_intent = "research"\npick_delivery = "private"\n',
        "missing_intent_dm": '[scheduler]\npick_delivery = "dm"\n',
    }
    scenarios = [
        ("no_games", [], None, []),
        ("all_void", [{"status": {"detailedState": "Postponed"}}], None, []),
        ("sched_err", "RAISE", None, []),
        ("no_decision", [{"status": {"detailedState": "Final"}}], None, []),
        ("delib_skip", [{"status": {"detailedState": "Final"}}], ("skip", "mdp", False, None), []),
        ("bad_skip", [{"status": {"detailedState": "Final"}}], ("skip", "crash", False, None), []),
        ("unscoreable", [{"status": {"detailedState": "Final"}}], ("single", "mdp", False, "failed"), []),
        ("delivered_no_rcpt", [{"status": {"detailedState": "Final"}}], ("single", "mdp", True, "delivered"), []),
        ("delivered_entered", [{"status": {"detailedState": "Final"}}], ("single", "mdp", True, "delivered"), ["entered"]),
        ("delivered_absent", [{"status": {"detailedState": "Final"}}], ("single", "mdp", True, "delivered"), ["absent"]),
        ("private_locked", [{"status": {"detailedState": "Final"}}], ("single", "mdp", True, "private_locked"), []),
        ("contradiction", [], ("single", "mdp", True, "delivered"), []),
    ]
    for cname, cbody in configs_to_test.items():
        for sname, sched_spec, dec_spec, rcpt_spec in scenarios:
            with tempfile.TemporaryDirectory() as tmp:
                tpath = Path(tmp)
                pdir = tpath / "picks"
                pdir.mkdir()
                cfile = tpath / "c.toml"
                cfile.write_text(cbody)
                if dec_spec is not None:
                    act, src, sc_bool, dstat = dec_spec
                    daily_decision.write_decision(
                        DAY, pdir, action=act, source=src,
                        primary={"batter_id": 1, "batter_name": "Kwan", "p_game_hit": 0.8} if act != "skip" else None,
                        delivery_status=dstat, scoreable=sc_bool,
                    )
                fake_receipts = []
                if rcpt_spec:
                    _, raw = daily_decision.load_decision_bytes(DAY, pdir)
                    sha = entry_receipt._sha(raw)
                    rkind = rcpt_spec[0]
                    fake_receipts = [{
                        "schema": "bts_pick_entry_receipt_v1",
                        "outcome": "observed",
                        "et_date": DAY,
                        "season": 2027,
                        "selection": {"commit": {"decision_sha256": sha}},
                        "degraded": [],
                        "account": {"username": "stonehengee"},
                        "observation": {"before_cutoff": True, "response_completed_at": f"{DAY}T17:00:00-04:00"},
                        "verifier": {"reason": "match" if rkind == "entered" else "no_pick"},
                        "qualification": {"all_confirmed": rkind == "entered"},
                    }]

                def fake_sched(_d, spec=sched_spec):
                    if spec == "RAISE":
                        raise RuntimeError("schedule down")
                    return spec

                dms = []
                with contextlib.ExitStack() as st:
                    enter_all(st, [
                        patch.object(scheduler, "fetch_schedule", side_effect=fake_sched),
                        patch.object(entry_receipt, "discover", return_value=fake_receipts),
                        patch("bts.dm.send_dm", side_effect=lambda r, m: dms.append((r, m)) or "dm-ok"),
                    ])
                    g = day_outcome.gather(DAY, pdir, tomllib.loads(cbody), expected_username="stonehengee", fetch_schedule=fake_sched)
                    r_cli = runner.invoke(cli.cli, [
                        "day-outcome", "--date", DAY, "--config", str(cfile),
                        "--picks-dir", str(pdir), "--expected-username", "stonehengee",
                        "--dm-recipient", "op.bsky.social",
                    ])
                day_cases[f"{cname}__{sname}"] = {
                    "gather": g.as_dict(),
                    "cli_exit": r_cli.exit_code,
                    "cli_stdout": r_cli.stdout,
                    "cli_stderr": r_cli.stderr,
                    "dms": dms,
                }
    out["day_outcome"] = day_cases

    out_file.write_text(json.dumps(out, indent=2))


def run_branch_direct_adversarial_probes() -> dict:
    sys.path.insert(0, str(BRANCH_SRC))
    import pandas as pd
    from click.testing import CliRunner
    from bts import cli, contest_state, entry_intent, orchestrator, picks, scheduler

    runner = CliRunner()
    report: dict = {}

    def make_df() -> pd.DataFrame:
        return pd.DataFrame([{
            "batter_name": "Jacob Wilson",
            "batter_id": 700363,
            "team": "ATH",
            "lineup": 1,
            "pitcher_name": "Jose Suarez",
            "pitcher_id": 660761,
            "p_hit_pa": 0.312,
            "p_game_hit": 0.812,
            "flags": "",
            "game_pk": 778899,
            "game_time": f"{DAY}T23:10:00Z",
        }])

    # A. Item 2: Accepted outward configs on uncovered 2026 state -> zero picks delivered!
    accepted_outward_tomls = [
        ("dm_req_true", '[scheduler]\npick_delivery = "dm"\nentry_intent = "enter"\ncontest_state_required = true\n'),
        ("pub_exp_true", '[scheduler]\npick_delivery = "public"\nentry_intent = "enter"\n[health_checks]\ncontest_state_expected = true\n'),
        ("dm_req_str_false", '[scheduler]\npick_delivery = "bluesky-dm"\nentry_intent = "enter"\ncontest_state_required = "false"\n'),
        ("pub_exp_int_1", '[scheduler]\npick_delivery = "feed"\n[health_checks]\ncontest_state_expected = 1\n'),
        ("dm_req_nonempty_list", '[scheduler]\npick_delivery = "direct-message"\ncontest_state_required = [1]\n'),
        ("pub_exp_nonempty_tbl", '[scheduler]\nposting_mode = "post"\n[health_checks.contest_state_expected]\nenabled = false\n'),
    ]
    outward_results = {}
    for label, tbody in accepted_outward_tomls:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdir = root / "picks"
            write_uncovered_2026_state(pdir)
            hb = root / "hb.json"
            full_toml = (
                f'tiers = []\n[orchestrator]\npicks_dir = "{pdir}"\nheartbeat_path = "{hb}"\n'
                '[bluesky]\ndm_recipient = "op.bsky.social"\n' + tbody
            )
            cfile = root / "c.toml"
            cfile.write_text(full_toml)
            cfg = tomllib.loads(full_toml)
            assert entry_intent.live_gate_problem(cfg) is None, label
            assert entry_intent.live_gate_problem(cfg, delivery_override="public") is None, label

            pick_posts, pick_dms, alert_dms = [], [], []
            with contextlib.ExitStack() as stack:
                enter_all(stack, [
                    patch("datetime.datetime", FixedDateTime),
                    patch.object(contest_state, "datetime", FixedDateTime),
                    patch.object(picks, "datetime", FixedDateTime),
                    patch.object(scheduler, "_now_et", lambda: NOW),
                    patch.object(orchestrator, "run_cascade", return_value=(make_df(), "local")),
                    patch.object(scheduler, "check_confirmed_lineups", return_value={778899: {"away", "home"}}),
                    patch.object(picks, "get_game_statuses_detailed", return_value={778899: {"abstract": "P", "detailed": "Pre-Game"}}),
                    patch("bts.posting.should_post_now", return_value=True),
                    patch("bts.posting.post_to_bluesky", side_effect=lambda t: pick_posts.append(t) or "at://pub/x"),
                    patch("bts.dm.send_dm", side_effect=lambda r, t: pick_dms.append((r, t)) or "dm-x"),
                    patch("bts.health.alert.send_dm", side_effect=lambda r, t: alert_dms.append((r, t)) or "alert-x"),
                ])
                sc_res = scheduler.run_single_check([778899], set(), cfg, DAY, 0.03)
                assert sc_res["pick_result"] is None and sc_res["should_post"] is False
                assert pick_posts == [] and pick_dms == []
                assert len(alert_dms) == 1

                cached_daily = picks.DailyPick(
                    date=DAY, run_time=NOW.isoformat(),
                    pick=picks.Pick(
                        batter_name="Jacob Wilson", batter_id=700363, team="ATH",
                        lineup_position=1, pitcher_name="Jose Suarez", pitcher_id=660761,
                        p_game_hit=0.812, flags=[], projected_lineup=True,
                        game_pk=778899, game_time=f"{DAY}T23:10:00Z",
                    ),
                    double_down=None, runner_up=None,
                )
                st_obj = scheduler.SchedulerState(
                    date=DAY, schedule_fetched_at=NOW.isoformat(), games=[],
                    confirmed_game_pks=[], runs_completed=[], pick_locked=False,
                    pick_locked_at=None, result_status=None, next_wakeup=None,
                )
                with patch.object(scheduler, "_void_committed_games", return_value={}):
                    delivered = scheduler._deliver_and_lock_pick(cached_daily, cfg, pdir, st_obj, DAY, "test")
                assert delivered is False and st_obj.pick_locked is False
                assert pick_posts == [] and pick_dms == []

                orch_dms = []
                with patch("bts.dm.send_dm", side_effect=lambda r, t: orch_dms.append((r, t)) or "dm-o"):
                    r_orch = runner.invoke(cli.cli, ["orchestrate", "--date", DAY, "--config", str(cfile)])
                assert r_orch.exit_code == 1
                assert pick_posts == []
                assert not (pdir / f"{DAY}.json").exists()
                assert len(orch_dms) == 1 and "No pick made" in orch_dms[0][1]

            outward_results[label] = "blocked_cleanly_zero_picks_delivered"
    report["accepted_outward_on_uncovered_state"] = outward_results

    # B. Item 2: Accepted private/research configs through REAL `bts schedule` (`run_day`)
    accepted_private_tomls = [
        ("priv_research", '[scheduler]\npick_delivery = "private"\nentry_intent = "research"\n'),
        ("local_absent_intent", '[scheduler]\nposting_mode = "local"\n'),
        ("priv_mode_str_false", '[scheduler]\nprivate_mode = "false"\n'),
    ]
    priv_results = {}
    for label, tbody in accepted_private_tomls:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdir = root / "picks"
            write_uncovered_2026_state(pdir)
            hb = root / "hb.json"
            full_toml = (
                f'tiers = []\n[orchestrator]\npicks_dir = "{pdir}"\nheartbeat_path = "{hb}"\n'
                '[bluesky]\ndm_recipient = "op.bsky.social"\n' + tbody
            )
            cfile = root / "c.toml"
            cfile.write_text(full_toml)
            game = {
                "gamePk": 778899,
                "gameDate": f"{DAY}T16:30:00Z",  # 12:30 ET; NOW is 12:00 ET -> check at 11:45 ET (due!)
                "teams": {"away": {"team": {"name": "ATH"}}, "home": {"team": {"name": "DET"}}},
                "status": {"abstractGameState": "Preview", "detailedState": "Pre-Game"},
            }
            posts, dms = [], []
            with contextlib.ExitStack() as stack:
                enter_all(stack, [
                    patch("datetime.datetime", FixedDateTime),
                    patch.object(contest_state, "datetime", FixedDateTime),
                    patch.object(picks, "datetime", FixedDateTime),
                    patch.object(picks, "_git_head_sha", lambda *_a, **_k: "0" * 40),
                    patch.object(scheduler, "_now_et", lambda: NOW),
                    patch.object(scheduler, "datetime", FixedDateTime),
                    patch.object(scheduler, "fetch_schedule", return_value=[game]),
                    patch.object(scheduler, "notify_ready", lambda: None),
                    patch.object(scheduler, "notify_watchdog", lambda: None),
                    patch.object(scheduler, "_watchdog_ping_sleep", lambda _s: None),
                    patch.object(scheduler, "_idle_until_next_wakeup", lambda *_a, **_k: None),
                    patch.object(scheduler, "poll_and_grade_results", lambda *_a, **_k: None),
                    patch.object(scheduler, "_run_end_of_day_health_check", lambda *_a, **_k: None),
                    patch.object(scheduler, "_trigger_live_forward_capture_on_lock", lambda *_a, **_k: None),
                    patch.object(scheduler, "check_confirmed_lineups", return_value={778899: {"away", "home"}}),
                    patch.object(scheduler, "_void_committed_games", return_value={}),
                    patch.object(orchestrator, "run_cascade", return_value=(make_df(), "local")),
                    patch.object(picks, "get_game_statuses_detailed", return_value={778899: {"abstract": "P", "detailed": "Pre-Game"}}),
                    patch("bts.strategy.get_game_statuses", return_value={778899: "P"}),
                    patch("bts.posting.post_to_bluesky", side_effect=lambda t: posts.append(t) or "at://pub/x"),
                    patch("bts.dm.send_dm", side_effect=lambda r, t: dms.append((r, t)) or "dm-x"),
                ])
                r_sched = runner.invoke(cli.cli, ["schedule", "--config", str(cfile), "--date", DAY])
            assert r_sched.exit_code == 0, (label, r_sched.stderr, r_sched.exception)
            assert posts == [] and dms == []
            dec = json.loads((pdir / DAY / "decision.json").read_text())
            assert dec["delivery_status"] == "private_locked" and dec["scoreable"] is True
            saved_pick = json.loads((pdir / f"{DAY}.json").read_text())
            assert saved_pick["bluesky_posted"] is False and saved_pick["notification_sent"] is False
            priv_results[label] = "locked_privately_zero_dm_zero_post"
    report["accepted_private_through_real_schedule"] = priv_results

    # C. Item 4: `bts orchestrate` mutating config file on disk between check and run + open audit hook
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        pdir = root / "picks"
        evil_pdir = root / "evil_picks"
        acc = pdir / "account_state"
        acc.mkdir(parents=True)
        act_id = "f" * 32
        (acc / "season_contract.json").write_text(json.dumps({
            "schema_version": "bts_season_contract_v1", "season": 2027, "status": "automatic",
            "activation_id": act_id, "switched_at": NOW.isoformat(), "revoked_at": None, "note": None,
        }))
        (acc / "contest_streak.json").write_text(json.dumps({
            "schema_version": "bts_contest_streak_auto_v2", "season": 2027, "activation_id": act_id,
            "active_streak": 5, "best_streak": 5, "source": "mlb_bts_profile",
            "source_date": DAY, "recorded_at": NOW.isoformat(),
        }))
        cfile = root / "mutable.toml"
        cfile.write_text(
            f'tiers = []\n[orchestrator]\npicks_dir = "{pdir}"\n'
            '[bluesky]\ndm_recipient = "op.bsky.social"\n'
            '[scheduler]\ncontest_state_required = true\n'
        )

        real_load = orchestrator.load_config
        loaded_dicts = []
        core_received = []
        real_core = orchestrator.orchestrate

        def mutating_loader(path: Path) -> dict:
            d = real_load(path)
            loaded_dicts.append(d)
            cfile.write_text(
                f'[corrupt toml!\npicks_dir = "{evil_pdir}"\ncontest_state_required = false\n'
            )
            return d

        def spying_core(config_path: Path, date: str, *, config: dict | None = None) -> bool:
            core_received.append(config)
            return real_core(config_path, date, config=config)

        AUDIT_FS_LOG.clear()
        AUDIT_WATCH_PREFIXES[:] = [str(cfile)]
        try:
            with contextlib.ExitStack() as stack:
                enter_all(stack, [
                    patch("datetime.datetime", FixedDateTime),
                    patch.object(contest_state, "datetime", FixedDateTime),
                    patch.object(picks, "datetime", FixedDateTime),
                    patch.object(picks, "_git_head_sha", lambda *_a, **_k: "0" * 40),
                    patch.object(orchestrator, "load_config", side_effect=mutating_loader),
                    patch.object(orchestrator, "orchestrate", side_effect=spying_core),
                    patch.object(orchestrator, "run_cascade", return_value=(make_df(), "local")),
                    patch.object(picks, "get_game_statuses_detailed", return_value={778899: {"abstract": "P", "detailed": "Pre-Game"}}),
                    patch("bts.strategy.get_game_statuses", return_value={778899: "P"}),
                    patch("bts.posting.should_post_now", return_value=True),
                    patch("bts.posting.post_to_bluesky", return_value="at://pub/ok"),
                ])
                r_mut = runner.invoke(cli.cli, ["orchestrate", "--date", DAY, "--config", str(cfile)])
        finally:
            AUDIT_WATCH_PREFIXES.clear()

        open_events = [ev for ev in AUDIT_FS_LOG if ev[0] == "open"]
        assert r_mut.exit_code == 0, (r_mut.stderr, r_mut.exception)
        assert len(loaded_dicts) == 1 and len(core_received) == 1
        assert core_received[0] is loaded_dicts[0]
        assert len(open_events) == 2
        assert (pdir / f"{DAY}.json").exists() and not evil_pdir.exists()
        report["orchestrate_single_snapshot_mutation"] = {
            "exit": r_mut.exit_code,
            "same_dict_identity": core_received[0] is loaded_dicts[0],
            "cfile_open_events_count": len(open_events),
        }

    # D. Item 6: `bts schedule` refusal audit hook on picks_dir & heartbeat_path
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        pdir = root / "picks"
        pdir.mkdir()
        (pdir / "sentinel.json").write_text('{"untouched": true}')
        hb = root / "hb.json"
        cfile = root / "refuse.toml"
        cfile.write_text(
            f'[orchestrator]\npicks_dir = "{pdir}"\nheartbeat_path = "{hb}"\n'
            '[scheduler]\npick_delivery = "dm"\nentry_intent = "enter"\n'
        )
        before_tree = {str(p.relative_to(root)): p.read_bytes().hex() for p in root.rglob("*") if p.is_file()}
        AUDIT_FS_LOG.clear()
        AUDIT_WATCH_PREFIXES[:] = [str(pdir), str(hb)]
        today_et_spy = Mock(side_effect=AssertionError("_today_et called on refusal"))
        run_day_spy = Mock(side_effect=AssertionError("run_day called on refusal"))
        try:
            with (
                patch.object(cli, "_today_et", today_et_spy),
                patch.object(scheduler, "run_day", run_day_spy),
            ):
                r_ref = runner.invoke(cli.cli, ["schedule", "--config", str(cfile)])
        finally:
            AUDIT_WATCH_PREFIXES.clear()
        after_tree = {str(p.relative_to(root)): p.read_bytes().hex() for p in root.rglob("*") if p.is_file()}
        assert r_ref.exit_code == 2
        assert r_ref.stderr.startswith("bts schedule: refusing to start: ")
        assert AUDIT_FS_LOG == [], f"Unexpected FS access on refusal: {AUDIT_FS_LOG}"
        assert before_tree == after_tree
        assert today_et_spy.call_count == 0 and run_day_spy.call_count == 0
        report["schedule_refusal_audit"] = {
            "exit": r_ref.exit_code,
            "fs_events_on_picks_or_hb": AUDIT_FS_LOG,
            "tree_identical": before_tree == after_tree,
        }

    # E. Item 7: `config/orchestrator.example.toml` diff & parse verification
    base_ex = tomllib.loads((REVIEW_DIR / "base_tree" / "config" / "orchestrator.example.toml").read_text())
    branch_ex = tomllib.loads((REPO / "config" / "orchestrator.example.toml").read_text())
    assert branch_ex["scheduler"]["contest_state_required"] is True
    assert branch_ex["health_checks"]["contest_state_expected"] is True
    assert branch_ex["scheduler"]["entry_intent"] == "enter"
    assert entry_intent.live_gate_problem(branch_ex) is None
    assert entry_intent.check_entry_intent(branch_ex).problems == ()
    expected_from_base = copy.deepcopy(base_ex)
    expected_from_base["scheduler"]["contest_state_required"] = True
    expected_from_base["scheduler"]["entry_intent"] = "enter"
    expected_from_base["health_checks"]["contest_state_expected"] = True
    assert branch_ex == expected_from_base
    report["example_config_check"] = "parsed_ast_identical_except_3_specified_keys"

    return report


def compare_subworkers(base: dict, branch: dict) -> dict:
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
        assert c1[f"{flag_kind}_show"] == c0["off_show"]
        assert c1[f"{flag_kind}_install"] == c0["off_install"]

    assert branch["day_outcome"] == base["day_outcome"]

    return {
        "load_errors_identical": True,
        "priv_research_orchestrate_verified": True,
        "schedule_dry_run_game_and_off_day_verified": True,
        "cron_installer_show_install_remove_verified": True,
        "day_outcome_cases_identical": len(branch["day_outcome"]),
    }


def main() -> None:
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        run_subworker(sys.argv[2], Path(sys.argv[3]))
        return

    install_audit_hooks()
    base_out = REVIEW_DIR / "details_base_v3.json"
    branch_out = REVIEW_DIR / "details_branch_v3.json"
    env = dict(os.environ, TZ="America/New_York", PYTHONDONTWRITEBYTECODE="1")

    for which, out_file in (("base", base_out), ("branch", branch_out)):
        proc = subprocess.run(
            [sys.executable, str(Path(__file__)), "--worker", which, str(out_file)],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Subworker {which} failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}")

    sub_cmp = compare_subworkers(json.loads(base_out.read_text()), json.loads(branch_out.read_text()))
    direct_rep = run_branch_direct_adversarial_probes()
    full_summary = {"base_vs_branch_details": sub_cmp, "branch_direct_probes": direct_rep}
    (REVIEW_DIR / "details_summary.json").write_text(json.dumps(full_summary, indent=2))
    print(json.dumps(full_summary, indent=2))


if __name__ == "__main__":
    main()
