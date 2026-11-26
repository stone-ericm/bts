"""Probe 01: Diff scope, paired base-vs-candidate grid, and byte-for-byte cron parity."""
from __future__ import annotations

from contextlib import ExitStack
from datetime import date, datetime, timezone
import gzip
from hashlib import sha256
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
WORK_DIR = OUT_DIR / "_work_p01"
BASE_SHA = "1e69d0f0"
CAND_CODE_SHA = "717ac97f"
DAY = "2027-03-24"
PITCH = f"{DAY}T19:40:00-04:00"
XW = {1: 1010, 2: 1020, 3: 1030}
RIGHT_1 = {"id": 77, "roundId": 1, "feedId": 101}
WRONG_1 = {"id": 77, "roundId": 1, "feedId": 102}
RIGHT_2 = {"id": 78, "roundId": 1, "feedId": 201}
WRONG_2 = {"id": 78, "roundId": 1, "feedId": 202}


def _git(*args: str) -> str:
    res = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True, timeout=15)
    return res.stdout


def extract_base(dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["git", "archive", "--format=tar", BASE_SHA, "src", "scripts"],
        cwd=REPO,
        capture_output=True,
        check=True,
        timeout=15,
    )
    with tempfile.NamedTemporaryFile(dir=dest, suffix=".tar", delete=True) as tf:
        tf.write(proc.stdout)
        tf.flush()
        with tarfile.open(tf.name, "r") as tar:
            tar.extractall(dest, filter="data")
    return dest


def build_scenarios() -> list[dict]:
    scenarios: list[dict] = []

    # 1. Early exits and fetch failures across clean & wrong-game prior markers
    for prior_kind in ("none", "clean_alerted", "wg_unresolved", "confirmed_with_ref", "confirmed_no_ref"):
        for early in (
            "no_pick_file",
            "preview",
            "outside_early",
            "outside_at_cutoff",
            "identity_mismatch",
            "fail_profile",
            "fail_pending",
            "fail_rounds",
            "fail_crosswalk",
        ):
            scenarios.append({
                "group": "early_or_fetch",
                "prior": prior_kind,
                "early": early,
                "slot_mode": "single_right",
                "at": "18:30",
                "dm": "success",
                "delivery": "delivered",
                "elapsed": 0,
            })

    # 2. Clean-history sweep (no prior wrong-game observation) across all slot modes, prior markers, times, DMs, deliveries
    clean_priors = [
        "none",
        "stale_clean",
        "stale_wg",
        "corrupt_json",
        "legacy_alerted",
        "dm_failed_clean",
        "present_unverified_clean",
        "alerted_init",
        "alerted_init_t30",
        "alerted_all",
    ]
    slot_modes = [
        "no_pick",
        "mismatch_wrong_batter",
        "mismatch_missing_dd",
        "present_unverified",
        "single_right",
        "single_wrong_game",
        "single_no_capture",
        "single_null_feed",
        "single_conflicting_captures",
        "single_wrong_round",
        "single_ambiguous",
        "single_game_pk_none",
        "single_game_pk_true",
        "single_game_pk_false",
        "single_game_pk_str",
        "single_game_pk_float",
        "single_corrupt_capture",
        "double_both_right",
        "double_first_wrong",
        "double_second_wrong",
        "double_both_wrong",
        "double_one_confirmed_one_unverified",
        "double_shared_batter_diff_game",
        "double_shared_batter_same_game",
    ]
    for prior_kind in clean_priors:
        for slot_mode in slot_modes:
            for at, elapsed in (("18:30", 0), ("19:05", 0), ("19:20", 0), ("19:34", 61)):
                # Keep grid compact (~960 clean/A1(a) scenarios + ~300 A1(b)/delivery scenarios)
                if at in ("19:05", "19:20", "19:34") and prior_kind not in (
                    "none", "alerted_init", "alerted_init_t30", "dm_failed_clean"
                ):
                    continue
                scenarios.append({
                    "group": "clean_or_a1a",
                    "prior": prior_kind,
                    "early": None,
                    "slot_mode": slot_mode,
                    "at": at,
                    "dm": "success",
                    "delivery": "delivered",
                    "elapsed": elapsed,
                })

    # 3. DM outcomes & delivery states across representative clean and wrong-game states
    for prior_kind in ("none", "alerted_init", "dm_failed_clean", "wg_unresolved", "wg_delivered", "wg_dm_failed"):
        for slot_mode in (
            "no_pick",
            "mismatch_wrong_batter",
            "present_unverified",
            "single_right",
            "single_wrong_game",
            "single_null_feed",
            "single_conflicting_captures",
            "single_corrupt_capture",
            "double_both_right",
            "double_one_confirmed_one_unverified",
            "double_shared_batter_diff_game",
        ):
            for dm_mode in ("success", "failure", "none"):
                for delivery in ("delivered", "locked_unconfirmed"):
                    if delivery == "locked_unconfirmed" and dm_mode == "none":
                        continue
                    scenarios.append({
                        "group": "prior_dm_delivery",
                        "prior": prior_kind,
                        "early": None,
                        "slot_mode": slot_mode,
                        "at": "18:45",
                        "dm": dm_mode,
                        "delivery": delivery,
                        "elapsed": 0,
                    })

    # 4. Prior wrong-game marker representations (R1 three-way test) across times
    for prior_kind in (
        "wg_field_only_present_unverified",
        "wg_field_only_no_pick",
        "wg_reason_only_alerted",
        "wg_reason_only_dm_failed",
        "wg_tier_only_present_unverified",
        "wg_tier_and_init",
        "wg_tier_and_t30",
    ):
        for slot_mode in ("no_pick", "present_unverified", "single_right", "single_null_feed", "single_wrong_game"):
            for at in ("18:45", "19:05", "19:20"):
                scenarios.append({
                    "group": "r1_markers",
                    "prior": prior_kind,
                    "early": None,
                    "slot_mode": slot_mode,
                    "at": at,
                    "dm": "success",
                    "delivery": "delivered",
                    "elapsed": 0,
                })

    return scenarios


def _normalize(obj, picks_str: str):
    if isinstance(obj, dict):
        return {
            k: _normalize(v, picks_str)
            for k, v in obj.items()
            if k not in {
                "receipt", "attempt_id", "confirmed_by", "checked_at",
                "started_at", "published_at", "response_completed_at",
                "failed_at", "cutoff_at", "revision",
            }
        }
    if isinstance(obj, list):
        return [_normalize(v, picks_str) for v in obj]
    if isinstance(obj, str):
        return obj.replace(picks_str, "<PICKS>")
    return obj


def run_worker(scenarios_file: Path, out_file: Path) -> None:
    import bts
    from click.testing import CliRunner
    from bts import entry_receipt, picks
    from bts.cli import cli
    from bts.daily_decision import write_decision

    scenarios = json.loads(scenarios_file.read_text())
    runner = CliRunner()
    results = []

    def row(unit=77, player=1, number=1, rnd=1):
        return {"roundId": rnd, "unitId": unit, "playerId": player, "number": number, "result": None}

    with tempfile.TemporaryDirectory() as td:
        base_root = Path(td)
        for idx, sc in enumerate(scenarios):
            root = base_root / f"c{idx}"
            picks_dir = root / "picks"
            picks_dir.mkdir(parents=True)
            marker_path = root / "health_state" / "pick_entry_check.json"
            units_dir = entry_receipt.units_snapshot_dir(picks_dir)
            units_dir.mkdir(parents=True)

            early = sc["early"]
            slot_mode = sc["slot_mode"]
            delivery = None if early == "preview" else sc["delivery"]

            # Configure pick & captures
            game_pk = 101
            dd = None
            pending = [row()]
            crosswalk = XW

            if slot_mode == "no_pick":
                pending = []
                captures = [[RIGHT_1]]
            elif slot_mode == "mismatch_wrong_batter":
                pending = [row(player=3)]
                captures = [[RIGHT_1]]
            elif slot_mode == "mismatch_missing_dd":
                dd = (1020, 201, PITCH)
                pending = [row()]
                captures = [[RIGHT_1, RIGHT_2]]
            elif slot_mode == "present_unverified":
                pending = [row()]
                crosswalk = {}
                captures = [[WRONG_1]]
            elif slot_mode == "single_right":
                captures = [[RIGHT_1]]
            elif slot_mode == "single_wrong_game":
                captures = [[WRONG_1]]
            elif slot_mode == "single_no_capture":
                captures = []
            elif slot_mode == "single_null_feed":
                captures = [[{"id": 77, "roundId": 1, "feedId": None}]]
            elif slot_mode == "single_conflicting_captures":
                captures = [[WRONG_1], [RIGHT_1]]
            elif slot_mode == "single_wrong_round":
                captures = [[{"id": 77, "roundId": 2, "feedId": 102}]]
            elif slot_mode == "single_ambiguous":
                pending = [row(unit=77, number=1), row(unit=78, number=2)]
                captures = [[WRONG_1, WRONG_2]]
            elif slot_mode.startswith("single_game_pk_"):
                kind = slot_mode.removeprefix("single_game_pk_")
                game_pk = {"none": None, "true": True, "false": False, "str": "101", "float": 101.0}[kind]
                captures = [[WRONG_1]]
            elif slot_mode == "single_corrupt_capture":
                captures = ["CORRUPT"]
            elif slot_mode == "double_both_right":
                dd = (1020, 201, PITCH)
                pending = [row(77, 1, 1), row(78, 2, 2)]
                captures = [[RIGHT_1, RIGHT_2]]
            elif slot_mode == "double_first_wrong":
                dd = (1020, 201, PITCH)
                pending = [row(77, 1, 1), row(78, 2, 2)]
                captures = [[WRONG_1, RIGHT_2]]
            elif slot_mode == "double_second_wrong":
                dd = (1020, 201, PITCH)
                pending = [row(77, 1, 1), row(78, 2, 2)]
                captures = [[RIGHT_1, WRONG_2]]
            elif slot_mode == "double_both_wrong":
                dd = (1020, 201, PITCH)
                pending = [row(77, 1, 1), row(78, 2, 2)]
                captures = [[WRONG_1, WRONG_2]]
            elif slot_mode == "double_one_confirmed_one_unverified":
                dd = (1020, 201, PITCH)
                pending = [row(77, 1, 1), row(78, 2, 2)]
                captures = [[RIGHT_1, {"id": 78, "roundId": 1, "feedId": None}]]
            elif slot_mode == "double_shared_batter_diff_game":
                dd = (1010, 102, PITCH)
                pending = [row(77, 1, 1)]
                captures = [[WRONG_1]]
            elif slot_mode == "double_shared_batter_same_game":
                dd = (1010, 101, PITCH)
                pending = [row(77, 1, 1)]
                captures = [[WRONG_1]]
            else:
                raise AssertionError(f"unknown slot_mode {slot_mode}")

            if early != "no_pick_file":
                def make_slot(batter, gpk, gtime):
                    return picks.Pick(
                        batter_name=f"Batter {batter}", batter_id=batter, team="NYY",
                        lineup_position=1, pitcher_name="Pitcher", pitcher_id=500,
                        p_game_hit=0.8, flags=[], projected_lineup=False, game_pk=gpk, game_time=gtime,
                    )
                p1 = make_slot(1010, game_pk, PITCH)
                p2 = make_slot(*dd) if dd else None
                daily = picks.DailyPick(
                    DAY, f"{DAY}T16:00:00-04:00", p1, p2, None,
                    notification_sent=(delivery is not None),
                    notification_channel="bluesky_dm", notification_id="fixture",
                )
                picks.save_pick(daily, picks_dir)
                if delivery is not None:
                    write_decision(
                        DAY, picks_dir, action="double" if dd else "single", source="mdp",
                        delivery_status=delivery, scoreable=True,
                        primary={"batter_name": p1.batter_name},
                        double_down={"batter_name": p2.batter_name} if p2 else None,
                        now=datetime.fromisoformat(f"{DAY}T20:00:00+00:00"),
                    )

            for c_idx, cap in enumerate(captures):
                c_path = units_dir / f"20270324T2{c_idx}0000Z.json.gz"
                if cap == "CORRUPT":
                    c_path.write_bytes(b"not-a-gzip-stream")
                else:
                    c_path.write_bytes(gzip.compress(json.dumps({"units": cap}).encode(), mtime=0))

            # Seed prior marker
            pk = sc["prior"]
            prior_map = {
                "none": None,
                "clean_alerted": {"date": DAY, "status": "alerted", "reason": "no_pick", "escalations": ["initial"]},
                "stale_clean": {"date": "2027-03-23", "status": "alerted", "reason": "no_pick", "escalations": ["initial"]},
                "stale_wg": {"date": "2027-03-23", "status": "alerted", "reason": "wrong_game", "escalations": ["wrong_game"], "game_unresolved": True},
                "corrupt_json": "RAW:{bad json",
                "legacy_alerted": {"date": DAY, "status": "alerted", "reason": "no_pick"},
                "dm_failed_clean": {"date": DAY, "status": "dm_failed", "reason": "no_pick"},
                "present_unverified_clean": {"date": DAY, "status": "present_unverified", "reason": "present_unverified", "escalations": []},
                "alerted_init": {"date": DAY, "status": "alerted", "reason": "no_pick", "escalations": ["initial"]},
                "alerted_init_t30": {"date": DAY, "status": "alerted", "reason": "no_pick", "escalations": ["initial", "t30"]},
                "alerted_all": {"date": DAY, "status": "alerted", "reason": "no_pick", "escalations": ["initial", "t15", "t30"]},
                "confirmed_with_ref": {"date": DAY, "status": "confirmed", "reason": "match", "receipt": "deadbeef" * 4},
                "confirmed_no_ref": {"date": DAY, "status": "confirmed", "reason": "match"},
                "wg_unresolved": {"date": DAY, "status": "alerted", "reason": "no_pick", "escalations": ["initial"], "game_unresolved": True},
                "wg_delivered": {"date": DAY, "status": "alerted", "reason": "wrong_game", "escalations": ["wrong_game"], "game_unresolved": True},
                "wg_dm_failed": {"date": DAY, "status": "dm_failed", "reason": "wrong_game", "escalations": [], "game_unresolved": True},
                "wg_field_only_present_unverified": {"date": DAY, "status": "present_unverified", "reason": "present_unverified", "escalations": [], "game_unresolved": True},
                "wg_field_only_no_pick": {"date": DAY, "status": "alerted", "reason": "no_pick", "escalations": ["initial"], "game_unresolved": True},
                "wg_reason_only_alerted": {"date": DAY, "status": "alerted", "reason": "wrong_game"},
                "wg_reason_only_dm_failed": {"date": DAY, "status": "dm_failed", "reason": "wrong_game"},
                "wg_tier_only_present_unverified": {"date": DAY, "status": "present_unverified", "reason": "present_unverified", "escalations": ["wrong_game"]},
                "wg_tier_and_init": {"date": DAY, "status": "alerted", "reason": "no_pick", "escalations": ["initial", "wrong_game"]},
                "wg_tier_and_t30": {"date": DAY, "status": "alerted", "reason": "wrong_game", "escalations": ["t30", "wrong_game"], "game_unresolved": True},
            }
            prior_val = prior_map[pk]
            if prior_val is not None:
                marker_path.parent.mkdir(parents=True, exist_ok=True)
                if isinstance(prior_val, str) and prior_val.startswith("RAW:"):
                    marker_path.write_text(prior_val[4:])
                else:
                    marker_path.write_text(json.dumps(prior_val))

            at = sc["at"]
            if early == "outside_early":
                at = "12:00"
            elif early == "outside_at_cutoff":
                at = "19:35"

            sent_dms, attempted_dms = [], []
            dm_mode = sc["dm"]

            def send_dm(_, msg):
                attempted_dms.append(msg)
                if dm_mode == "failure":
                    raise RuntimeError("synthetic dm error")
                sent_dms.append(msg)

            fetch_fail = early.removeprefix("fail_") if (early and early.startswith("fail_")) else None

            def make_fetch(name, val):
                def _call(*a, **k):
                    if fetch_fail == name:
                        raise ValueError(f"synthetic {name} error")
                    return val
                return _call

            anchor = []
            elapsed = sc["elapsed"]

            def mono():
                if not anchor:
                    anchor.append(True)
                    return 1000.0
                return 1000.0 + elapsed

            uname = "someone_else" if early == "identity_mismatch" else "operator"
            session = SimpleNamespace(user_id="u1", username=uname, xsid="x1")
            patches = {
                "bts.leaderboard.auth.load_session_cookies": lambda: {},
                "bts.leaderboard.auth.extract_uid": lambda c: "u1",
                "bts.leaderboard.auth.fetch_login_session": lambda *a, **k: session,
                "bts.contest_fetch.fetch_profile": make_fetch("profile", {"predictions": []}),
                "bts.contest_fetch.fetch_pending_predictions": make_fetch("pending", pending),
                "bts.cli._fetch_rounds": make_fetch("rounds", {1: date.fromisoformat(DAY), 2: date(2027, 3, 25)}),
                "bts.cli._fetch_bts_to_mlb": make_fetch("crosswalk", crosswalk),
                "bts.dm.send_dm": send_dm,
                "bts.picks._git_head_sha": lambda *a, **k: "fixed-sha",
                "time.monotonic": mono,
            }
            args = [
                "check-pick-entered", "--picks-dir", str(picks_dir),
                "--now-et", f"{DAY}T{at}:00", "--expected-username", "operator",
            ]
            if dm_mode != "none":
                args += ["--dm-recipient", "captured.invalid"]

            with ExitStack() as stack:
                for target, fn in patches.items():
                    stack.enter_context(patch(target, fn))
                res = runner.invoke(cli, args)

            raw_marker = None
            if marker_path.exists():
                try:
                    raw_marker = json.loads(marker_path.read_text())
                except Exception:
                    raw_marker = {"_raw": marker_path.read_text()}
            recs = entry_receipt.discover(picks_dir, DAY)
            obs = _normalize({
                "exit": res.exit_code,
                "stdout": res.stdout,
                "stderr": res.stderr,
                "dm": sent_dms,
                "attempted_dm": attempted_dms,
                "marker": raw_marker,
                "receipts": recs,
            }, str(picks_dir))
            results.append(obs)

    out_file.write_text(json.dumps({
        "bts_file": str(Path(bts.__file__).resolve()),
        "results": results,
    }))


def check_cron_parity(base_dir: Path) -> dict:
    cand_script = REPO / "scripts" / "cron-setup-hetzner.sh"
    base_script = base_dir / "scripts" / "cron-setup-hetzner.sh"

    def run_installer(script: Path, intent_cfg: str, initial_cron: str | None, action: str) -> dict:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            home = root / "home"
            (home / "projects" / "bts").mkdir(parents=True)
            (home / "projects" / "bts" / ".env").write_text("")
            (home / ".local" / "bin").mkdir(parents=True)
            uv_shim = home / ".local" / "bin" / "uv"
            uv_shim.write_text(
                '#!/bin/sh\n[ "$1 $2 $3" = "run bts entry-intent" ] || exit 97\n'
                f'shift 2\nexec "{sys.executable}" -c "from bts.cli import cli; cli()" "$@"\n'
            )
            uv_shim.chmod(uv_shim.stat().st_mode | stat.S_IXUSR)

            shims = root / "shims"
            shims.mkdir()
            cron_file = root / "crontab.txt"
            if initial_cron is not None:
                cron_file.write_text(initial_cron)
            cron_shim = shims / "crontab"
            cron_shim.write_text(
                f'#!/bin/sh\nf="{cron_file}"\ncase "$1" in\n'
                '  -l) [ -f "$f" ] && cat "$f" || { echo "no crontab for user" >&2; exit 1; } ;;\n'
                '  -) cat > "$f.new" && mv "$f.new" "$f" ;;\n'
                '  *) exit 98 ;;\nesac\n'
            )
            cron_shim.chmod(cron_shim.stat().st_mode | stat.S_IXUSR)

            (home / ".bts-orchestrator.toml").write_text(intent_cfg)
            env = {
                "HOME": str(home),
                "PATH": f"{shims}:/usr/bin:/bin",
                "HEALTHCHECKS_PING_URL": "https://hc-ping.invalid/probe",
                "TZ": "America/New_York",
            }
            res = subprocess.run(["bash", str(script), action], env=env, capture_output=True, text=True, timeout=20)
            installed = cron_file.read_bytes() if cron_file.exists() else None
            # Normalize temporary home path so base and candidate outputs compare byte-for-byte
            norm_home = str(home).encode()
            if installed is not None:
                installed = installed.replace(norm_home, b"/HOME")
            return {
                "rc": res.returncode,
                "stdout": res.stdout.replace(str(home), "/HOME"),
                "stderr": res.stderr.replace(str(home), "/HOME"),
                "installed": installed.decode() if installed is not None else None,
            }

    research_cfg = '[scheduler]\nentry_intent = "research"\npick_delivery = "private"\n'
    enter_cfg = '[scheduler]\nentry_intent = "enter"\npick_delivery = "dm"\ncontest_state_required = true\n'
    bad_cfg = '[scheduler]\nentry_intent = "enter"\npick_delivery = "private"\n'
    seed = (
        "0 9 * * * echo foreign\n"
        "*/15 10-23 * * * unmarked_legacy check-pick-entered\n"
        "*/15 10-23 * * * old_marked check-pick-entered # BTS-HETZNER\n"
        "# */15 10-23 * * * old_commented check-pick-entered # BTS-HETZNER\n"
    )

    wrapper_insert = "/usr/bin/flock -n /home/bts/.check-pick-entered.lock /usr/bin/timeout --verbose -k 30 840 "

    # 1. Research: show and install must be 100% byte-for-byte identical
    for action in ("show", "install", "remove"):
        b_res = run_installer(base_script, research_cfg, seed, action)
        c_res = run_installer(cand_script, research_cfg, seed, action)
        assert b_res == c_res, f"research {action} differed: {b_res} vs {c_res}"

    # 2. Refusal: byte-for-byte identical
    for action in ("show", "install"):
        b_res = run_installer(base_script, bad_cfg, seed, action)
        c_res = run_installer(cand_script, bad_cfg, seed, action)
        assert b_res["rc"] == c_res["rc"] == 1
        assert b_res == c_res

    # 3. Enter: every line except check-pick-entered is byte-for-byte identical, and check-pick-entered differs only by wrapper_insert
    b_enter_inst = run_installer(base_script, enter_cfg, seed, "install")
    c_enter_inst = run_installer(cand_script, enter_cfg, seed, "install")
    assert b_enter_inst["rc"] == c_enter_inst["rc"] == 0

    b_lines = b_enter_inst["installed"].splitlines(keepends=True)
    c_lines = c_enter_inst["installed"].splitlines(keepends=True)
    assert len(b_lines) == len(c_lines)

    watchdog_diffs = 0
    cand_watchdog_line = None
    for bl, cl in zip(b_lines, c_lines):
        if "bts check-pick-entered" in cl:
            watchdog_diffs += 1
            cand_watchdog_line = cl.rstrip("\n")
            assert wrapper_insert in cl, cl
            assert cl.replace(wrapper_insert, "", 1) == bl, f"unexpected diff on watchdog line:\n{bl}\n{cl}"
        else:
            assert bl == cl, f"non-watchdog line differed:\n{bl}\n{cl}"
    assert watchdog_diffs == 1

    b_enter_show = run_installer(base_script, enter_cfg, None, "show")
    c_enter_show = run_installer(cand_script, enter_cfg, None, "show")
    assert c_enter_show["stdout"].replace(wrapper_insert, "", 1) == b_enter_show["stdout"]

    return {
        "research_byte_identical": True,
        "enter_other_lines_byte_identical": True,
        "enter_line_count": len(c_lines),
        "watchdog_line": cand_watchdog_line,
    }


def run_all() -> dict:
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    # Check git scope
    diff_code = _git("diff", "--name-status", BASE_SHA, CAND_CODE_SHA, "--", "src", "scripts", "docs/ops", "tests", ".github").splitlines()
    expected_diff = [
        "M\tdocs/ops/pick-entry-receipt-v1.md",
        "M\tscripts/cron-setup-hetzner.sh",
        "M\tsrc/bts/cli.py",
        "M\tsrc/bts/entry_receipt.py",
        "A\ttests/test_wr_f1_game_pk_types.py",
        "A\ttests/test_wr_f1_review.py",
        "A\ttests/test_wr_f1_wrong_game.py",
        "A\ttests/test_wr_f1b_lock.py",
    ]
    assert sorted(diff_code) == sorted(expected_diff), f"unexpected changed files: {diff_code}"
    f1b_src_diff = _git("diff", "15b27474", CAND_CODE_SHA, "--", "src", "docs/ops")
    assert f1b_src_diff == "", f"F1b unexpectedly changed src or docs/ops: {f1b_src_diff}"
    modified_existing_tests = _git("diff", "--diff-filter=M", BASE_SHA, "HEAD", "--", "tests")
    assert modified_existing_tests == "", f"existing tests modified: {modified_existing_tests}"

    base_dir = extract_base(WORK_DIR / "base_archive")
    cron_report = check_cron_parity(base_dir)

    scenarios = build_scenarios()
    sc_file = WORK_DIR / "scenarios.json"
    sc_file.write_text(json.dumps(scenarios))
    base_out = WORK_DIR / "base_out.json"
    cand_out = WORK_DIR / "cand_out.json"

    base_env = {**os.environ, "PYTHONPATH": str(base_dir / "src"), "TZ": "America/New_York"}
    cand_env = {**os.environ, "PYTHONPATH": str(REPO / "src"), "TZ": "America/New_York"}

    subprocess.run([sys.executable, __file__, "--worker", str(sc_file), str(base_out)], env=base_env, check=True, timeout=60)
    subprocess.run([sys.executable, __file__, "--worker", str(sc_file), str(cand_out)], env=cand_env, check=True, timeout=60)

    base_data = json.loads(base_out.read_text())
    cand_data = json.loads(cand_out.read_text())
    assert str((base_dir / "src").resolve()) in base_data["bts_file"], base_data["bts_file"]
    assert str((REPO / "src").resolve()) in cand_data["bts_file"], cand_data["bts_file"]
    assert base_data["bts_file"] != cand_data["bts_file"]

    counts = {
        "total": len(scenarios),
        "identical_outside_wg": 0,
        "a1a_wrong_game": 0,
        "a1b_prior_unresolved": 0,
        "r1_nonterminal_preserved": 0,
        "full_proof_after_wg": 0,
        "m10_raising_capture": 0,
        "early_exit_with_wg_marker": 0,
    }
    unexpected = []

    wg_priors = {
        "wg_unresolved", "wg_delivered", "wg_dm_failed",
        "wg_field_only_present_unverified", "wg_field_only_no_pick",
        "wg_reason_only_alerted", "wg_reason_only_dm_failed",
        "wg_tier_only_present_unverified", "wg_tier_and_init", "wg_tier_and_t30",
    }
    a1a_modes = {"single_wrong_game", "double_first_wrong", "double_second_wrong", "double_both_wrong"}

    for idx, (sc, b_obs, c_obs) in enumerate(zip(scenarios, base_data["results"], cand_data["results"])):
        has_prior_wg = sc["prior"] in wg_priors
        in_window_checked = sc["early"] is None and sc["prior"] not in ("confirmed_with_ref", "confirmed_no_ref")
        is_a1a = in_window_checked and sc["slot_mode"] in a1a_modes
        is_m10 = in_window_checked and sc["slot_mode"] == "single_corrupt_capture"

        if not has_prior_wg and not is_a1a and not is_m10:
            if b_obs != c_obs:
                unexpected.append({"idx": idx, "scenario": sc, "base": b_obs, "cand": c_obs})
            else:
                counts["identical_outside_wg"] += 1
            continue

        if not in_window_checked:
            # Early exit or fetch failure: marker is untouched on both sides!
            if b_obs != c_obs:
                unexpected.append({"idx": idx, "scenario": sc, "reason": "early_exit_changed", "base": b_obs, "cand": c_obs})
            else:
                counts["early_exit_with_wg_marker"] += 1
            continue

        if is_m10:
            counts["m10_raising_capture"] += 1
            assert c_obs["exit"] == 0 and c_obs["marker"]["status"] == "confirmed"
            assert c_obs["marker"]["reason"] == "match_game_unchecked"
            assert "game check unavailable (BadGzipFile); batter match only" in c_obs["stderr"]
            assert "game_unresolved" not in c_obs["marker"]
        elif is_a1a:
            counts["a1a_wrong_game"] += 1
            assert c_obs["exit"] == 1
            assert c_obs["marker"]["reason"] == "wrong_game"
            assert c_obs["marker"]["game_unresolved"] is True
            assert c_obs["marker"]["status"] in ("alerted", "dm_failed")
            # Receipt differs from base only in marker_status and exit_code
            b_rec = dict(b_obs["receipts"][0], marker_status=None, exit_code=None)
            c_rec = dict(c_obs["receipts"][0], marker_status=None, exit_code=None)
            assert b_rec == c_rec, f"receipt mismatch on A1(a): {b_rec} vs {c_rec}"
        elif has_prior_wg:
            if sc["slot_mode"] in ("single_right", "double_both_right"):
                counts["full_proof_after_wg"] += 1
                assert c_obs["exit"] == 0 and c_obs["marker"]["status"] == "confirmed"
                assert "game_unresolved" not in c_obs["marker"]
            elif sc["slot_mode"] in (
                "single_null_feed", "single_conflicting_captures", "single_no_capture",
                "single_wrong_round", "single_ambiguous", "double_one_confirmed_one_unverified",
                "double_shared_batter_diff_game", "double_shared_batter_same_game",
            ):
                counts["a1b_prior_unresolved"] += 1
                assert c_obs["exit"] == 1
                assert c_obs["marker"]["reason"] == "wrong_game"
                assert c_obs["marker"]["game_unresolved"] is True
                assert c_obs["marker"]["status"] != "confirmed"
            else:
                counts["r1_nonterminal_preserved"] += 1
                assert c_obs["marker"]["status"] != "confirmed"
                assert c_obs["marker"]["game_unresolved"] is True

    assert not unexpected, f"unexpected parity differences: {unexpected[:3]}"

    summary = {
        "base_bts_file": base_data["bts_file"],
        "cand_bts_file": cand_data["bts_file"],
        "diff_files": diff_code,
        "cron_parity": cron_report,
        "grid_counts": counts,
        "unexpected_count": len(unexpected),
    }
    (OUT_DIR / "log_01_scope_parity_and_cron.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def test_probe_01_scope_parity_and_cron():
    summary = run_all()
    assert summary["unexpected_count"] == 0


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        run_worker(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        res = run_all()
        print(json.dumps(res, indent=2))
