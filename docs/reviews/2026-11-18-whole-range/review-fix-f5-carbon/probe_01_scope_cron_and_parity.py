"""Probe 01: Diff scope, generated crontab byte-parity, and paired non-`--cron` CLI parity against base a4e8278d."""
from __future__ import annotations

import gzip
import io
import json
import os
import re
import shutil
import socket
import stat
import subprocess
import sys
import tarfile
from contextlib import ExitStack
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
BASE_REV = "a4e8278d"
PRE_CODE_REV = "833c2772"
F5_REV = "5a80f0cc"
HEAD_REV = "0f14c380"
DAY = "2027-03-24"
PREV_DAY = "2027-03-23"


def _git(*args: str, text: bool = True):
    return subprocess.run(
        ["git", "-C", str(REPO), *args],
        check=True,
        capture_output=True,
        text=text,
        timeout=20,
    )


def check_diff_scope() -> dict:
    diff_base = _git("diff", "--name-status", BASE_REV, F5_REV).stdout.strip().splitlines()
    non_review = [ln for ln in diff_base if "\tdocs/reviews/" not in ln]
    expected_code_files = [
        "M\tARCHITECTURE.md",
        "M\tdocs/ops/pick-entry-receipt-v1.md",
        "M\tscripts/cron-setup-hetzner.sh",
        "M\tsrc/bts/cli.py",
        "M\ttests/test_entry_intent.py",
        "M\ttests/test_wr_f1b_lock.py",
        "A\ttests/test_wr_f5_watchdog_window.py",
    ]
    assert sorted(non_review) == sorted(expected_code_files), non_review

    diff_code_only = _git("diff", "--name-status", PRE_CODE_REV, F5_REV).stdout.strip().splitlines()
    assert sorted(diff_code_only) == sorted(expected_code_files), diff_code_only

    diff_head = _git("diff", "--name-status", F5_REV, HEAD_REV).stdout.strip().splitlines()
    assert diff_head == ["A\tdocs/reviews/2026-11-18-whole-range/handover-fix-f5.md"], diff_head

    whitespace_check = _git("diff", "--check", BASE_REV, F5_REV).stdout
    assert whitespace_check == "", whitespace_check

    # Verify ONLY the two contract-listed assertions changed in existing test files.
    ti_diff = _git("diff", "-U0", BASE_REV, F5_REV, "--", "tests/test_entry_intent.py").stdout
    ti_changes = [ln for ln in ti_diff.splitlines() if ln.startswith(("-", "+")) and not ln.startswith(("---", "+++"))]
    assert ti_changes == [
        '-    assert len(entry) == 1 and entry[0].startswith("*/15 10-23 * * * ")',
        '+    assert len(entry) == 1 and entry[0].startswith("*/15 * * * * ")',
    ], ti_changes

    f1b_diff = _git("diff", "-U0", BASE_REV, F5_REV, "--", "tests/test_wr_f1b_lock.py").stdout
    f1b_changes = [ln for ln in f1b_diff.splitlines() if ln.startswith(("-", "+")) and not ln.startswith(("---", "+++"))]
    assert f1b_changes == [
        '-    assert schedule == ["*/15", "10-23", "*", "*", "*"]',
        '+    assert schedule == ["*/15", "*", "*", "*", "*"]',
    ], f1b_changes

    return {
        "non_review_changed_files": non_review,
        "code_commit_files": diff_code_only,
        "post_code_commit_files": diff_head,
        "entry_intent_test_delta": ti_changes,
        "f1b_lock_test_delta": f1b_changes,
    }


def _exe(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _run_installer_suite(script_path: Path, src_dir: Path, work_dir: Path) -> dict:
    results = {}

    def make_box(name: str):
        root = work_dir / name
        home = root / "home"
        (home / "projects" / "bts").mkdir(parents=True)
        (home / "projects" / "bts" / ".env").write_text("")
        (home / ".local" / "bin").mkdir(parents=True)
        _exe(
            home / ".local" / "bin" / "uv",
            '#!/bin/sh\n[ "$1 $2 $3" = "run bts entry-intent" ] || { echo "uv shim: unexpected $*" >&2; exit 97; }\n'
            f'shift 2\nPYTHONPATH="{src_dir}" exec "{sys.executable}" -c "from bts.cli import cli; cli()" "$@"\n',
        )
        shims = root / "shims"
        shims.mkdir()
        crontab = root / "crontab.txt"
        _exe(
            shims / "crontab",
            f'#!/bin/sh\nf="{crontab}"\necho "$1" >> "$f.calls"\ncase "$1" in\n'
            '  -l) [ -f "$f.unreadable" ] && { echo "crontab: cannot read" >&2; exit 1; }\n'
            '      [ -f "$f" ] && cat "$f" || { echo "no crontab for user" >&2; exit 1; } ;;\n'
            '  -) cat > "$f.new" && mv "$f.new" "$f" ;;\n  *) exit 98 ;;\nesac\n',
        )
        env = {
            "HOME": str(home),
            "PATH": f"{shims}:/usr/bin:/bin",
            "HEALTHCHECKS_PING_URL": "https://hc-ping.invalid/probe01",
            "TMPDIR": str(root / "tmp"),
        }
        (root / "tmp").mkdir()
        return root, home, shims, crontab, env

    configs = {
        "research_private": '[scheduler]\nentry_intent = "research"\npick_delivery = "private"\n',
        "research_local": '[scheduler]\nentry_intent = "research"\nposting_mode = "local"\n',
        "enter_dm": '[scheduler]\nentry_intent = "enter"\npick_delivery = "dm"\ncontest_state_required = true\n',
        "enter_bluesky": '[scheduler]\nentry_intent = "enter"\npick_delivery = "bluesky"\ncontest_state_required = true\n',
        "enter_health_gate": '[scheduler]\nentry_intent = "enter"\npick_delivery = "dm"\n[health_checks]\ncontest_state_expected = true\n',
    }
    for key, body in configs.items():
        root, home, _, crontab, env = make_box(key)
        (home / ".bts-orchestrator.toml").write_text(body)
        # Normalize HOME path in outputs so base and candidate work_dirs compare byte-for-byte
        norm = lambda s: s.replace(str(home), "$HOME_NORM")
        show_res = subprocess.run(["bash", str(script_path), "show"], env=env, capture_output=True, text=True, timeout=20)
        inst_res = subprocess.run(["bash", str(script_path), "install"], env=env, capture_output=True, text=True, timeout=20)
        installed_bytes = norm(crontab.read_text()) if crontab.exists() else None
        rem_res = subprocess.run(["bash", str(script_path), "remove"], env=env, capture_output=True, text=True, timeout=20)
        after_remove = norm(crontab.read_text()) if crontab.exists() else None
        results[key] = {
            "show": (show_res.returncode, norm(show_res.stdout), norm(show_res.stderr)),
            "install": (inst_res.returncode, norm(inst_res.stdout), norm(inst_res.stderr), installed_bytes),
            "remove": (rem_res.returncode, norm(rem_res.stdout), norm(rem_res.stderr), after_remove),
        }
    return results


def check_crontab_parity(base_dir: Path, work_dir: Path) -> dict:
    base_res = _run_installer_suite(base_dir / "scripts/cron-setup-hetzner.sh", base_dir / "src", work_dir / "cron_base")
    cand_res = _run_installer_suite(REPO / "scripts/cron-setup-hetzner.sh", REPO / "src", work_dir / "cron_cand")

    for key in ("research_private", "research_local"):
        assert base_res[key] == cand_res[key], f"Mismatch under {key}: {base_res[key]} != {cand_res[key]}"

    watchdog_diffs = {}
    for key in ("enter_dm", "enter_bluesky", "enter_health_gate"):
        assert base_res[key]["remove"] == cand_res[key]["remove"], key
        for kind, idx in (("show_stdout", 1), ("installed", 3)):
            b_text = base_res[key]["show"][idx] if kind == "show_stdout" else base_res[key]["install"][idx]
            c_text = cand_res[key]["show"][idx] if kind == "show_stdout" else cand_res[key]["install"][idx]
            b_lines = b_text.splitlines(keepends=True)
            c_lines = c_text.splitlines(keepends=True)
            assert len(b_lines) == len(c_lines), (key, kind, len(b_lines), len(c_lines))
            b_other = [ln for ln in b_lines if "bts check-pick-entered" not in ln]
            c_other = [ln for ln in c_lines if "bts check-pick-entered" not in ln]
            assert b_other == c_other, f"Non-watchdog line changed in {key}/{kind}"
            b_wd = [ln for ln in b_lines if "bts check-pick-entered" in ln]
            c_wd = [ln for ln in c_lines if "bts check-pick-entered" in ln]
            assert len(b_wd) == 1 and len(c_wd) == 1
            expected_wd = (
                b_wd[0]
                .replace("*/15 10-23 * * * ", "*/15 * * * * ", 1)
                .replace("bts check-pick-entered ", "bts check-pick-entered --cron ", 1)
            )
            assert c_wd[0] == expected_wd, f"Unexpected watchdog diff:\n  got: {c_wd[0]}\n  exp: {expected_wd}"
            watchdog_diffs[f"{key}:{kind}"] = {"base": b_wd[0].strip(), "cand": c_wd[0].strip()}

    return {
        "research_configs_byte_identical": ["research_private", "research_local"],
        "enter_configs_non_watchdog_byte_identical": ["enter_dm", "enter_bluesky", "enter_health_gate"],
        "enter_line_count": len(cand_res["enter_dm"]["install"][3].splitlines()),
        "watchdog_line": watchdog_diffs["enter_dm:installed"],
    }


def _worker_grid(work_root: Path) -> dict:
    """Executed inside an isolated child Python process against whatever `bts` is on PYTHONPATH."""
    import bts
    import bts.cli
    import bts.contest_fetch as _cf
    import httpx
    from click.testing import CliRunner
    from bts import entry_receipt, picks
    from bts.cli import cli
    from bts.daily_decision import write_decision
    from bts.leaderboard.auth import AuthError

    # Forbid any real network socket usage.
    def _no_net(*a, **kw):
        raise AssertionError(f"Unexpected network call: {a}")

    socket.socket.connect = _no_net  # type: ignore[assignment]
    socket.socket.connect_ex = _no_net  # type: ignore[assignment]
    socket.socket.sendto = _no_net  # type: ignore[assignment]
    socket.getaddrinfo = _no_net  # type: ignore[assignment]

    xw = {1: 1010, 2: 1020, 3: 1030}
    right_u1 = {"id": 77, "roundId": 1, "feedId": 101}
    right_u2 = {"id": 78, "roundId": 1, "feedId": 202}
    wrong_u1 = {"id": 77, "roundId": 1, "feedId": 102}

    def row(unit=77, player=1, number=1, rnd=1):
        return {"roundId": rnd, "unitId": unit, "playerId": player, "number": number, "result": None}

    def normalize_marker(m):
        if m is None:
            return None
        out = dict(m)
        if "receipt" in out and out["receipt"] is not None:
            assert isinstance(out["receipt"], str) and len(out["receipt"]) == 32
            out["receipt"] = "<UUID32>"
        return out

    def normalize_receipt(r, prior_ref):
        out = dict(r)
        assert isinstance(out["attempt_id"], str) and len(out["attempt_id"]) == 32
        out["attempt_id"] = "<UUID32>"
        out["producer"] = {**out["producer"], "revision": "<REV>"}
        if out.get("references") and out["references"].get("confirmed_by") is not None:
            if prior_ref == "prior-uuid-fixed":
                assert out["references"]["confirmed_by"] == "prior-uuid-fixed"
            else:
                out["references"] = {"confirmed_by": "<REF>"}
        return out

    cells = []
    # 1. Early & late local-gate matrix across pitches with and without base slots
    for now_hm in ("00:00", "03:00", "05:00", "05:15", "06:05", "08:50", "09:00", "09:45", "09:59",
                   "10:00", "10:30", "18:00", "19:05", "23:45"):
        for pitch_spec in ("none", "preview_06:10", "commit_06:10", "commit_10:05", "commit_10:06",
                           "commit_10:50", "commit_19:10", "double_09:35_19:10", "double_10:50_19:10",
                           "nextday_00:35"):
            cells.append({
                "id": f"gate:{now_hm}:{pitch_spec}",
                "now": f"{DAY}T{now_hm}:00",
                "pitch_spec": pitch_spec,
                "prior": None,
                "obs": "match_right",
                "dm_mode": "ok",
                "recipient": "op.bsky.social",
                "window_min": 75,
                "elapsed": 0,
            })

    # 2. Detailed observation, marker, failure, escalation, and window_min matrix at BOTH 09:45 (pre-10:00!)
    #    with a 10:50 pitch (which would trigger base_slot_available IF --cron leaked!) AND at 18:30 with 19:10 pitch.
    obs_modes = [
        "match_right",
        "match_wrong_game",
        "match_unit_unverified",
        "match_corrupt_capture",
        "double_match_right",
        "double_second_wrong_game",
        "present_unverified",
        "no_pick",
        "mismatch",
        "identity_mismatch",
        "auth_error",
        "http_503",
    ]
    priors = [
        None,
        {"date": DAY, "status": "confirmed", "receipt": "prior-uuid-fixed"},
        {"date": DAY, "status": "confirmed"},
        {"date": PREV_DAY, "status": "confirmed", "receipt": "old-day-uuid"},
        {"date": DAY, "status": "alerted", "reason": "no_pick", "escalations": ["initial"]},
        {"date": DAY, "status": "alerted", "reason": "wrong_game", "game_unresolved": True, "escalations": ["wrong_game"]},
        {"date": DAY, "status": "dm_failed", "reason": "no_pick"},
        {"date": DAY, "status": "present_unverified", "reason": "present_unverified", "escalations": []},
        "CORRUPT_JSON",
    ]
    for anchor_now, anchor_pitch in ((f"{DAY}T09:45:00", "commit_10:50"), (f"{DAY}T09:15:00", "commit_10:05"),
                                     (f"{DAY}T18:30:00", "commit_19:10")):
        for obs in obs_modes:
            for prior in priors:
                cells.append({
                    "id": f"obs:{anchor_now}:{anchor_pitch}:{obs}:{type(prior).__name__ if not isinstance(prior, dict) else prior.get('status')}_{prior.get('date') if isinstance(prior, dict) else prior}",
                    "now": anchor_now,
                    "pitch_spec": "double_10:50_19:10" if (obs.startswith("double") and "10:50" in anchor_pitch)
                                  else "double_09:35_19:10" if (obs.startswith("double") and "10:05" in anchor_pitch)
                                  else "double_19:10_21:10" if obs.startswith("double")
                                  else anchor_pitch,
                    "prior": prior,
                    "obs": obs,
                    "dm_mode": "ok",
                    "recipient": "op.bsky.social",
                    "window_min": 75,
                    "elapsed": 0,
                })

    # 3. DM failure, no recipient, locked_unconfirmed, elapsed past cutoff, and custom --window-min before 10:00
    for dm_mode in ("ok", "fail"):
        for recipient in ("op.bsky.social", None):
            for delivery in ("delivered", "locked_unconfirmed"):
                for elapsed in (0, 2600):  # 2600s = 43.33m > 40m to cutoff at 09:45 for 10:30 pitch
                    for win_min in (45, 75, 120):
                        cells.append({
                            "id": f"edge:{dm_mode}:{recipient}:{delivery}:{elapsed}:{win_min}",
                            "now": f"{DAY}T09:45:00",
                            "pitch_spec": "commit_10:30",
                            "delivery": delivery,
                            "prior": None,
                            "obs": "no_pick",
                            "dm_mode": dm_mode,
                            "recipient": recipient,
                            "window_min": win_min,
                            "elapsed": elapsed,
                        })

    runner = CliRunner()
    outputs = {}

    def make_slot(batter_id, game_pk, game_time_iso):
        return picks.Pick(
            batter_name=f"Batter {batter_id}",
            batter_id=batter_id,
            team="NYY",
            lineup_position=1,
            pitcher_name="Pitcher",
            pitcher_id=500,
            p_game_hit=0.80,
            flags=[],
            projected_lineup=False,
            game_pk=game_pk,
            game_time=game_time_iso,
        )

    for idx, cell in enumerate(cells):
        cdir = work_root / f"c_{idx}"
        pdir = cdir / "picks"
        pdir.mkdir(parents=True)
        marker_path = cdir / "health_state" / "pick_entry_check.json"
        units_dir = entry_receipt.units_snapshot_dir(pdir)
        units_dir.mkdir(parents=True)

        spec = cell["pitch_spec"]
        delivery = cell.get("delivery", "delivered")
        if spec != "none":
            committed = not spec.startswith("preview_")
            dd = None
            if spec in ("preview_06:10", "commit_06:10"):
                gt1 = f"{DAY}T06:10:00-04:00"
            elif spec == "commit_10:05":
                gt1 = f"{DAY}T10:05:00-04:00"
            elif spec == "commit_10:06":
                gt1 = f"{DAY}T10:06:00-04:00"
            elif spec == "commit_10:30":
                gt1 = f"{DAY}T10:30:00-04:00"
            elif spec == "commit_10:50":
                gt1 = f"{DAY}T10:50:00-04:00"
            elif spec == "commit_19:10":
                gt1 = f"{DAY}T19:10:00-04:00"
            elif spec == "nextday_00:35":
                gt1 = "2027-03-25T00:35:00-04:00"
            elif spec == "double_09:35_19:10":
                gt1 = f"{DAY}T19:10:00-04:00"
                dd = make_slot(1020, 202, f"{DAY}T09:35:00-04:00")
            elif spec == "double_10:50_19:10":
                gt1 = f"{DAY}T10:50:00-04:00"
                dd = make_slot(1020, 202, f"{DAY}T19:10:00-04:00")
            elif spec == "double_19:10_21:10":
                gt1 = f"{DAY}T19:10:00-04:00"
                dd = make_slot(1020, 202, f"{DAY}T21:10:00-04:00")
            else:
                raise ValueError(spec)

            p1 = make_slot(1010, 101, gt1)
            daily = picks.DailyPick(
                DAY, f"{DAY}T04:00:00-04:00", p1, dd, None,
                notification_sent=(committed and delivery == "delivered"),
                notification_channel="bluesky_dm" if committed else None,
                notification_id="msg-1" if (committed and delivery == "delivered") else None,
                delivery_attempted=committed,
            )
            picks.save_pick(daily, pdir)
            if committed:
                write_decision(
                    DAY, pdir,
                    action="double" if dd else "single",
                    source="mdp",
                    delivery_status=delivery,
                    scoreable=True,
                    primary={"batter_name": p1.batter_name},
                    double_down={"batter_name": dd.batter_name} if dd else None,
                    now=datetime.fromisoformat(f"{DAY}T04:00:00-04:00"),
                )

        prior = cell["prior"]
        prior_ref = None
        if prior == "CORRUPT_JSON":
            marker_path.parent.mkdir(parents=True, exist_ok=True)
            marker_path.write_text("{not-valid-json")
        elif isinstance(prior, dict):
            marker_path.parent.mkdir(parents=True, exist_ok=True)
            marker_path.write_text(json.dumps(prior))
            prior_ref = prior.get("receipt")

        obs = cell["obs"]
        if obs == "match_corrupt_capture":
            (units_dir / f"{DAY}T010000Z.json.gz").write_bytes(b"not-a-valid-gzip")
        elif obs in ("match_right", "no_pick", "mismatch", "present_unverified"):
            (units_dir / f"{DAY}T010000Z.json.gz").write_bytes(
                gzip.compress(json.dumps({"units": [right_u1, right_u2]}).encode(), mtime=0)
            )
        elif obs == "match_wrong_game":
            (units_dir / f"{DAY}T010000Z.json.gz").write_bytes(
                gzip.compress(json.dumps({"units": [wrong_u1, right_u2]}).encode(), mtime=0)
            )
        elif obs == "double_match_right":
            (units_dir / f"{DAY}T010000Z.json.gz").write_bytes(
                gzip.compress(json.dumps({"units": [right_u1, right_u2]}).encode(), mtime=0)
            )
        elif obs == "double_second_wrong_game":
            (units_dir / f"{DAY}T010000Z.json.gz").write_bytes(
                gzip.compress(json.dumps({"units": [right_u1, {**right_u2, "feedId": 203}]}).encode(), mtime=0)
            )
        # match_unit_unverified leaves units_dir empty

        calls = []
        dms = []
        mono_calls = []

        def fake_mono():
            if not mono_calls:
                mono_calls.append(1)
                return 1000.0
            return 1000.0 + cell["elapsed"]

        def fake_login(*a, **kw):
            calls.append("login")
            if obs == "auth_error":
                raise AuthError("synthetic auth failure")
            uname = "wrong_user" if obs == "identity_mismatch" else "stonehengee"
            return SimpleNamespace(user_id="u1", username=uname, xsid="x1")

        def fake_profile(*a, **kw):
            calls.append("profile")
            if obs == "http_503":
                req = httpx.Request("GET", "https://mlb.invalid/profile")
                resp = httpx.Response(503, request=req)
                raise httpx.HTTPStatusError("503", request=req, response=resp)
            return {"predictions": []}

        def fake_pending(*a, **kw):
            calls.append("pending")
            if obs in ("match_right", "match_wrong_game", "match_unit_unverified", "match_corrupt_capture"):
                return [row(77, 1, 1, 1)]
            if obs in ("double_match_right", "double_second_wrong_game"):
                return [row(77, 1, 1, 1), row(78, 2, 2, 1)]
            if obs == "present_unverified":
                return [row(77, 999, 1, 1)]
            if obs == "mismatch":
                return [row(77, 3, 1, 1)]
            return []

        def fake_rounds():
            calls.append("rounds")
            return {1: date.fromisoformat(DAY)}

        def fake_xw():
            calls.append("xw")
            return dict(xw)

        def fake_dm(rec, msg):
            calls.append("dm")
            if cell["dm_mode"] == "fail":
                raise RuntimeError("synthetic dm error")
            dms.append((rec, msg))

        argv = [
            "check-pick-entered",
            "--picks-dir", str(pdir),
            "--expected-username", "stonehengee",
            "--window-min", str(cell["window_min"]),
            "--now-et", cell["now"],
        ]
        if cell["recipient"]:
            argv.extend(["--dm-recipient", cell["recipient"]])

        with ExitStack() as stack:
            stack.enter_context(patch("bts.leaderboard.auth.load_session_cookies", lambda: {}))
            stack.enter_context(patch("bts.leaderboard.auth.extract_uid", lambda _: "u1"))
            stack.enter_context(patch("bts.leaderboard.auth.fetch_login_session", fake_login))
            stack.enter_context(patch("bts.contest_fetch.fetch_profile", fake_profile))
            stack.enter_context(patch("bts.contest_fetch.fetch_pending_predictions", fake_pending))
            stack.enter_context(patch("bts.cli._fetch_rounds", fake_rounds))
            stack.enter_context(patch("bts.cli._fetch_bts_to_mlb", fake_xw))
            stack.enter_context(patch("bts.dm.send_dm", fake_dm))
            stack.enter_context(patch("time.monotonic", fake_mono))
            res = runner.invoke(cli, argv)

        # Normalize temp path inside stdout/stderr/dms if quoted (e.g. locked_unconfirmed status hint)
        norm_p = lambda s: s.replace(str(pdir), "<PICKS_DIR>")
        marker_val = None
        if marker_path.exists():
            try:
                marker_val = normalize_marker(json.loads(marker_path.read_text()))
            except json.JSONDecodeError:
                marker_val = {"_raw_corrupt": marker_path.read_text()}

        recs = [normalize_receipt(r, prior_ref) for r in entry_receipt.discover(pdir, DAY)]
        outputs[cell["id"]] = {
            "exit_code": res.exit_code,
            "stdout": norm_p(res.stdout),
            "stderr": norm_p(res.stderr),
            "calls": calls,
            "dms": [(r, norm_p(m)) for r, m in dms],
            "marker": marker_val,
            "receipts": recs,
        }
        shutil.rmtree(cdir)

    return {
        "bts_file": str(Path(bts.__file__).resolve()),
        "cli_file": str(Path(bts.cli.__file__).resolve()),
        "cell_count": len(outputs),
        "outputs": outputs,
    }


def run_paired_parity(base_dir: Path, work_dir: Path) -> dict:
    out_base = work_dir / "base_grid.json"
    out_cand = work_dir / "cand_grid.json"

    for label, src_path, out_file in (
        ("base", base_dir / "src", out_base),
        ("cand", REPO / "src", out_cand),
    ):
        wdir = work_dir / f"run_{label}"
        wdir.mkdir(parents=True)
        env = dict(os.environ)
        env["PYTHONPATH"] = str(src_path)
        env["TZ"] = "America/New_York"
        env["TMPDIR"] = str(wdir)
        cmd = [
            sys.executable,
            "-c",
            (
                "import json, sys; from pathlib import Path; "
                f"sys.path.insert(0, {str(OUT_DIR)!r}); "
                "import probe_01_scope_cron_and_parity as p; "
                f"res = p._worker_grid(Path({str(wdir)!r})); "
                "print('BTS_FILE=' + res['bts_file']); "
                f"Path({str(out_file)!r}).write_text(json.dumps(res))"
            ),
        ]
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=60)
        assert proc.returncode == 0, f"{label} worker failed:\n{proc.stdout}\n{proc.stderr}"

    b_data = json.loads(out_base.read_text())
    c_data = json.loads(out_cand.read_text())
    assert b_data["bts_file"] != c_data["bts_file"]
    assert str(base_dir.resolve()) in b_data["bts_file"], b_data["bts_file"]
    assert str((REPO / "src").resolve()) in c_data["bts_file"], c_data["bts_file"]
    assert b_data["cell_count"] == c_data["cell_count"]

    mismatches = []
    for cid, b_val in b_data["outputs"].items():
        c_val = c_data["outputs"][cid]
        if b_val != c_val:
            mismatches.append({"cell": cid, "base": b_val, "cand": c_val})

    assert not mismatches, f"Found {len(mismatches)} non-cron parity mismatches: {mismatches[:3]}"
    return {
        "base_bts_file": b_data["bts_file"],
        "cand_bts_file": c_data["bts_file"],
        "paired_cells_checked": b_data["cell_count"],
        "mismatches": len(mismatches),
    }


def main() -> None:
    work = OUT_DIR / ".work_probe_01"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    try:
        scope = check_diff_scope()

        base_dir = work / "base_archive"
        base_dir.mkdir()
        archive = _git("archive", BASE_REV, "src", "scripts", text=False).stdout
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tf:
            tf.extractall(base_dir, filter="data")

        cron_parity = check_crontab_parity(base_dir, work)
        cli_parity = run_paired_parity(base_dir, work)

        report = {
            "status": "PASS",
            "scope": scope,
            "crontab_parity": cron_parity,
            "non_cron_cli_parity": cli_parity,
        }
        (OUT_DIR / "log_01_scope_cron_and_parity.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            f"PASS probe_01: base={cli_parity['base_bts_file']} cand={cli_parity['cand_bts_file']} "
            f"cells={cli_parity['paired_cells_checked']} mismatches={cli_parity['mismatches']}"
        )
    finally:
        if work.exists():
            shutil.rmtree(work)


if __name__ == "__main__":
    main()
