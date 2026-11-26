"""Establish item 1 (installer & CLI/health/morning parity vs base bf78b642) and item 5 (traffic, backups, audit)."""
from __future__ import annotations

from datetime import date, datetime
import gzip
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
SCRATCH = Path("/tmp/wr-f5b-scratch").resolve()
ET = ZoneInfo("America/New_York")
DAY = "2027-03-24"
UUID_RE = re.compile(r"\b[0-9a-f]{32}\b")


def check_installer(tmp: Path) -> dict:
    sys.path.insert(0, str(REPO))
    from tests import test_entry_intent

    enter_cfg = '[scheduler]\nentry_intent = "enter"\npick_delivery = "dm"\ncontest_state_required = true\n'
    res_cfg = '[scheduler]\nentry_intent = "research"\npick_delivery = "private"\n'
    old_line = f"*/15 10-23 * * * uv run {test_entry_intent.ENTRY_CMD} # BTS-HETZNER"
    legacy_unmarked = old_line.removesuffix(" # BTS-HETZNER")
    foreign = "17 4 * * * echo keep-foreign"

    for label, script in (
        ("base", SCRATCH / "base/scripts/cron-setup-hetzner.sh"),
        ("f5b", SCRATCH / "candidate/scripts/cron-setup-hetzner.sh"),
    ):
        chk = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True, timeout=5)
        assert chk.returncode == 0, (label, chk.stderr)

    comparisons = {}
    for cron_seed_name, cron_seed in (
        ("empty", None),
        ("populated", f"{foreign}\n{old_line}\n# {old_line}\n{legacy_unmarked}\n"),
    ):
        for intent, cfg_body in (("enter", enter_cfg), ("research", res_cfg)):
            outputs = {}
            for ver, sub in (("base", "base"), ("f5b", "candidate")):
                bdir = tmp / f"inst-{cron_seed_name}-{intent}-{ver}"
                bdir.mkdir()
                inst = test_entry_intent.box.__wrapped__(bdir)
                inst.config.write_text(cfg_body)
                if cron_seed is not None:
                    inst.cron.write_text(cron_seed)
                script = SCRATCH / sub / "scripts/cron-setup-hetzner.sh"
                with patch.object(test_entry_intent, "SCRIPT", script):
                    shown = inst.run("show")
                    assert shown.returncode == 0, shown.stderr
                    installed = inst.run("install")
                    assert installed.returncode == 0, installed.stderr
                    after_install = [ln.replace(str(bdir), "<BOX>") for ln in inst.lines()]
                    shown_norm = shown.stdout.replace(str(bdir), "<BOX>")
                    removed = inst.run("remove")
                    assert removed.returncode == 0, removed.stderr
                    after_remove = [ln.replace(str(bdir), "<BOX>") for ln in (inst.lines() or [])]
                outputs[ver] = {
                    "after_install": after_install,
                    "after_remove": after_remove,
                    "shown": shown_norm,
                }
            assert outputs["base"]["after_remove"] == outputs["f5b"]["after_remove"]
            if intent == "research":
                assert outputs["base"]["after_install"] == outputs["f5b"]["after_install"]
                assert outputs["base"]["shown"] == outputs["f5b"]["shown"]
            else:
                b_other = [ln for ln in outputs["base"]["after_install"] if "check-pick-entered" not in ln]
                f_other = [ln for ln in outputs["f5b"]["after_install"] if "check-pick-entered" not in ln]
                assert b_other == f_other
                (b_wd,) = [ln for ln in outputs["base"]["after_install"] if "check-pick-entered" in ln]
                (f_wd,) = [ln for ln in outputs["f5b"]["after_install"] if "check-pick-entered" in ln]
                assert b_wd.startswith("*/15 10-23 * * * ")
                assert f_wd.startswith("*/15 * * * * ")
                assert f_wd.replace("*/15 * * * * ", "*/15 10-23 * * * ", 1).replace(
                    "check-pick-entered --cron ", "check-pick-entered ", 1
                ) == b_wd
            comparisons[f"{cron_seed_name}_{intent}"] = "identical_except_watchdog_field_and_flag"

    return comparisons


def run_parity_worker(version: str) -> dict:
    sys.path.insert(0, str(REPO))
    import bts
    from click.testing import CliRunner
    from bts import daily_decision, day_outcome, entry_receipt, picks
    from bts.cli import cli
    from bts.data import backup
    from bts.health import pick_entry
    from tests.test_wr_f1_wrong_game import RIGHT, WRONG, row
    from tests.test_wr_f3_skip_state_review import evidence as f3_evidence

    bts_file = str(Path(bts.__file__).resolve())
    expected_root = SCRATCH / ("base" if version == "base" else "candidate") / "src"
    assert Path(bts_file).is_relative_to(expected_root), (bts_file, expected_root)

    def _norm(text: str, root: Path) -> str:
        text = text.replace(str(root), "<ROOT>")
        return UUID_RE.sub("<ID>", text)

    runner = CliRunner()
    traces = []

    def make_env(sub: Path):
        pdir = sub / "picks"
        pdir.mkdir(parents=True)
        cfg = sub / "config.toml"
        cfg.write_text('[scheduler]\nentry_intent = "enter"\npick_delivery = "dm"\ncontest_state_required = true\n')
        return pdir, cfg

    def save_committed(pdir: Path, pitch_hm: str, *, day: str = DAY, committed: bool = True, dd_hm: str | None = None):
        commit_iso = f"{day}T04:50:00-04:00"
        p1 = picks.Pick("Batter 1010", 1010, "NYY", 1, "P1", 501, 0.8, [], False, 101, f"{day}T{pitch_hm}:00-04:00")
        p2 = (
            picks.Pick("Batter 1020", 1020, "BOS", 2, "P2", 502, 0.78, [], False, 102, f"{day}T{dd_hm}:00-04:00")
            if dd_hm
            else None
        )
        dp = picks.DailyPick(
            day,
            commit_iso,
            p1,
            p2,
            None,
            notification_sent=committed,
            notification_channel="bluesky_dm" if committed else None,
            notification_id="m1" if committed else None,
            delivery_attempted=committed,
            delivered_at=commit_iso if committed else None,
        )
        picks.save_pick(dp, pdir)
        if committed:
            daily_decision.write_decision(
                day,
                pdir,
                action="double" if dd_hm else "single",
                source="mdp",
                delivery_status="delivered",
                scoreable=True,
                primary={"batter_name": "Batter 1010", "batter_id": 1010, "team": "NYY", "game_pk": 101, "p_game_hit": 0.8},
                double_down=(
                    {"batter_name": "Batter 1020", "batter_id": 1020, "team": "BOS", "game_pk": 102, "p_game_hit": 0.78}
                    if dd_hm
                    else None
                ),
                now=datetime.fromisoformat(commit_iso),
            )

    def put_units(pdir: Path, items: list[dict], stamp: str = "030000"):
        udir = entry_receipt.units_snapshot_dir(pdir)
        udir.mkdir(parents=True, exist_ok=True)
        (udir / f"{DAY}T{stamp}Z.json.gz").write_bytes(gzip.compress(json.dumps({"units": items}).encode(), mtime=0))

    def invoke_one(
        sub: Path,
        pdir: Path,
        at: str,
        *,
        use_cron_flag: bool = False,
        pending: list | None = None,
        crosswalk: dict | None = None,
        auth_mode: str = "ok",
        dm_fail: bool = False,
        elapsed: float = 0.0,
    ) -> dict:
        calls: list[str] = []
        dms: list[str] = []
        ticks = [0]

        def mono():
            ticks[0] += 1
            return 1000.0 if ticks[0] == 1 else 1000.0 + elapsed

        def send_dm(rec, msg):
            if dm_fail:
                raise OSError("simulated dm failure")
            dms.append(f"{rec}:{msg}")

        def login(*a, **k):
            calls.append("login")
            if auth_mode == "fail_login":
                raise ValueError("login boom")
            uname = "wrong_person" if auth_mode == "bad_user" else "stonehengee"
            return SimpleNamespace(user_id="u1", username=uname, xsid="x1")

        def prof(*a, **k):
            calls.append("profile")
            if auth_mode == "fail_profile":
                raise ValueError("profile boom")
            return {"predictions": []}

        cmd = [
            "check-pick-entered",
            "--picks-dir",
            str(pdir),
            "--expected-username",
            "stonehengee",
            "--dm-recipient",
            "op.bsky.social",
            "--now-et",
            f"{DAY}T{at}",
        ]
        if use_cron_flag and version == "f5":
            cmd.insert(1, "--cron")

        before_recs = set((entry_receipt.receipts_root(pdir) / DAY).glob("*.json")) if (entry_receipt.receipts_root(pdir) / DAY).exists() else set()
        with (
            patch("socket.getaddrinfo", side_effect=AssertionError("network!")),
            patch("bts.leaderboard.auth.load_session_cookies", lambda: calls.append("cookies") or {}),
            patch("bts.leaderboard.auth.extract_uid", lambda _: "u1"),
            patch("bts.leaderboard.auth.fetch_login_session", login),
            patch("bts.contest_fetch.fetch_profile", prof),
            patch("bts.contest_fetch.fetch_pending_predictions", lambda *a, **k: calls.append("pending") or (pending or [])),
            patch("bts.cli._fetch_rounds", lambda: calls.append("rounds") or {1: date.fromisoformat(DAY)}),
            patch("bts.cli._fetch_bts_to_mlb", lambda: calls.append("xw") or ({1: 1010, 2: 1020} if crosswalk is None else crosswalk)),
            patch("bts.dm.send_dm", send_dm),
            patch("bts.picks._git_head_sha", lambda *a, **k: "PINNED_SHA"),
            patch("time.monotonic", mono),
        ):
            res = runner.invoke(cli, cmd, catch_exceptions=True)

        after_recs = set((entry_receipt.receipts_root(pdir) / DAY).glob("*.json"))
        (new_rec_path,) = sorted(after_recs - before_recs)
        seal_bytes = new_rec_path.with_name(new_rec_path.name + ".sealed").read_bytes()
        rec_norm = _norm(new_rec_path.read_text(), sub)
        marker_path = pdir.parent / "health_state" / "pick_entry_check.json"
        marker_norm = _norm(marker_path.read_text(), sub) if marker_path.exists() else None
        early_dir = pdir.parent / "health_state" / "pick_entry_check.early"
        assert not early_dir.exists(), f"unexpected early dir created in parity run: {list(early_dir.glob('*'))}"

        rel_files = sorted(
            _norm(str(p.relative_to(sub)), sub)
            for p in sub.rglob("*")
            if p.is_file() and not p.name.startswith("lineup_evolution_")
        )
        return {
            "exit": res.exit_code,
            "exc": type(res.exception).__name__ if res.exception else None,
            "stdout": _norm(res.stdout, sub),
            "stderr": _norm(res.stderr, sub),
            "calls": calls,
            "dms": dms,
            "marker": marker_norm,
            "receipt": rec_norm,
            "seal": seal_bytes.decode("utf-8"),
            "files": rel_files,
        }

    with TemporaryDirectory() as tmp_str:
        root = Path(tmp_str)

        # Sequence 1: Manual runs BEFORE 10:00 ET (00:00, 05:15, 09:45) and AFTER 10:00 ET
        s1 = root / "s1_manual_early"
        pdir, cfg = make_env(s1)
        traces.append(("manual_0000_nopick", invoke_one(s1, pdir, "00:00:00")))
        save_committed(pdir, "06:10", committed=False)
        traces.append(("manual_0500_preview", invoke_one(s1, pdir, "05:00:00")))
        save_committed(pdir, "06:10", committed=True)
        put_units(pdir, [RIGHT])
        traces.append(("manual_0430_outside", invoke_one(s1, pdir, "04:30:00")))
        traces.append(("manual_0515_dmfail", invoke_one(s1, pdir, "05:15:00", pending=[], dm_fail=True)))
        traces.append(("manual_0530_warn", invoke_one(s1, pdir, "05:30:00", pending=[])))
        traces.append(("manual_0545_t15", invoke_one(s1, pdir, "05:45:00", pending=[])))
        traces.append(("manual_0555_exhausted", invoke_one(s1, pdir, "05:55:00", pending=[])))
        traces.append(("manual_0600_confirm_allclear", invoke_one(s1, pdir, "06:00:00", pending=[row()])))
        traces.append(("manual_0602_already_confirmed", invoke_one(s1, pdir, "06:02:00", pending=[row()])))
        traces.append(("manual_0605_exact_cutoff", invoke_one(s1, pdir, "06:05:00", pending=[row()])))

        # Sequence 2: --cron at >= 10:00 ET across auth failures, crosswalk gap, wrong-game, slow-fetch cutoff, EOD health & morning
        for mode_label, cron_flag in (("manual_post10", False), ("cron_post10", True)):
            s2 = root / f"s2_{mode_label}"
            pdir, cfg = make_env(s2)
            save_committed(pdir, "11:15", committed=True)
            put_units(pdir, [WRONG])
            traces.append((f"{mode_label}_1000_baduser", invoke_one(s2, pdir, "10:00:00", use_cron_flag=cron_flag, auth_mode="bad_user")))
            traces.append((f"{mode_label}_1005_failprof", invoke_one(s2, pdir, "10:05:00", use_cron_flag=cron_flag, auth_mode="fail_profile")))
            traces.append((f"{mode_label}_1015_xwgap", invoke_one(s2, pdir, "10:15:00", use_cron_flag=cron_flag, pending=[row()], crosswalk={})))
            traces.append((f"{mode_label}_1030_wronggame", invoke_one(s2, pdir, "10:30:00", use_cron_flag=cron_flag, pending=[row()])))
            traces.append((f"{mode_label}_1045_stillwrong", invoke_one(s2, pdir, "10:45:00", use_cron_flag=cron_flag, pending=[row()])))
            traces.append((f"{mode_label}_1100_slowcutoff", invoke_one(s2, pdir, "11:00:00", use_cron_flag=cron_flag, pending=[], elapsed=610.0)))

            # Check health & morning parity on s2
            h_alerts = [(a.level, a.message) for a in pick_entry.check(pdir, date.fromisoformat(DAY), datetime.fromisoformat(f"{DAY}T23:59:00-04:00"))]
            with patch("bts.scheduler.fetch_schedule", lambda _: [{"gamePk": 101}]):
                m_out = day_outcome.gather(DAY, pdir, {"scheduler": {"entry_intent": "enter", "pick_delivery": "dm"}}, expected_username="stonehengee").as_dict()
            traces.append((f"{mode_label}_health_and_morning", {"health": h_alerts, "morning": m_out}))

        # Sequence 3: F3 skip_state_review and F4 streak_problem parity with no early receipts
        s3 = root / "s3_f3_f4"
        pdir, cfg = make_env(s3)
        daily_decision.write_decision(
            DAY, pdir, action="skip", source="mdp", delivery_status="deferred", scoreable=False, streak=12, saver_available=False,
            now=datetime.fromisoformat(f"{DAY}T11:00:00-04:00"),
        )
        acct = pdir / "account_state"
        acct.mkdir()
        (acct / "contest_streak.manual.json").write_text(json.dumps(f3_evidence("current", day=DAY)))
        (pdir / "streak.json").write_text("[]")
        with patch("bts.scheduler.fetch_schedule", lambda _: [{"gamePk": 101}]):
            f3_f4_out = day_outcome.gather(DAY, pdir, {"scheduler": {"entry_intent": "enter", "pick_delivery": "dm"}}, expected_username="stonehengee").as_dict()
        traces.append(("f3_f4_parity", f3_f4_out))

        # Traffic & backup & audit readers check (candidate only)
        traffic_summary = {}
        if version == "f5":
            st = root / "traffic_96"
            pdir_t, _ = make_env(st)
            for idx in range(96):
                hh, mm = divmod(idx * 15, 60)
                r = invoke_one(st, pdir_t, f"{hh:02d}:{mm:02d}:00", use_cron_flag=True)
                assert r["calls"] == [] and r["dms"] == [] and r["marker"] is None
            assert len(entry_receipt.discover(pdir_t, DAY)) == 96
            traffic_summary["no_pick_96_slots"] = {"receipts": 96, "calls": 0, "dms": 0, "markers": 0}

            # Backup ops set includes data/health_state
            assert "data/health_state" in backup.BACKUP_SETS["ops"].paths
            traffic_summary["ops_backup_paths"] = list(backup.BACKUP_SETS["ops"].paths)

    return {"version": version, "bts_file": bts_file, "traces": traces, "traffic": traffic_summary}


def main() -> None:
    if len(sys.argv) == 2 and sys.argv[1] in ("base", "f5"):
        print(json.dumps(run_parity_worker(sys.argv[1])))
        return

    with TemporaryDirectory() as tmp_str:
        inst_cmp = check_installer(Path(tmp_str))

    outputs = {}
    for ver, sub in (("base", "base/src"), ("f5", "candidate/src")):
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), ver],
            cwd=str(REPO),
            env={**os.environ, "PYTHONPATH": f"{SCRATCH / sub}:{REPO}"},
            text=True,
            capture_output=True,
            timeout=30,
            check=True,
        )
        outputs[ver] = json.loads(proc.stdout.splitlines()[-1])
        print(ver, "->", outputs[ver]["bts_file"], "traces:", len(outputs[ver]["traces"]))

    assert len(outputs["base"]["traces"]) == len(outputs["f5"]["traces"])
    mismatches = []
    for (name_b, val_b), (name_f, val_f) in zip(outputs["base"]["traces"], outputs["f5"]["traces"], strict=True):
        assert name_b == name_f
        if val_b != val_f:
            mismatches.append((name_b, val_b, val_f))

    assert not mismatches, f"Parity mismatches found: {mismatches[:2]}"
    report = {
        "installer": inst_cmp,
        "base_bts_file": outputs["base"]["bts_file"],
        "f5b_bts_file": outputs["f5"]["bts_file"],
        "paired_trace_count": len(outputs["base"]["traces"]),
        "mismatches": len(mismatches),
        "traffic": outputs["f5"]["traffic"],
    }
    (OUT_DIR / "log_02_parity_cron_and_traffic.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PARITY OK:", json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
