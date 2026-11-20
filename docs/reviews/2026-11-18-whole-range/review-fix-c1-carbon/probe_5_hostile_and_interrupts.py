"""Item 5 probe: hostile inputs (FIFO, dir, symlinks, malformed, non-UTF-8, second manual read) bounded in historical mode, interrupts, and read-only contest files."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
import json
import logging
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch
from zoneinfo import ZoneInfo


HERE = Path(__file__).resolve().parent
ET = ZoneInfo("America/New_York")
D = date(2027, 3, 24)
MATCH_ID = "a" * 32


def at_et(day: date, hr: int = 0, mn: int = 0) -> datetime:
    return datetime.combine(day, time(hr, mn), ET)


def seed_valid_auto_world(picks: Path) -> dict[str, Path]:
    acct = picks / "account_state"
    acct.mkdir(parents=True, exist_ok=True)
    auto_p = acct / "contest_streak.json"
    sw_p = acct / "season_contract.json"
    sv_p = acct / "saver_state.json"
    cov_p = picks / f"{D.isoformat()}.json"
    cov_prev = picks / f"{(D - timedelta(days=1)).isoformat()}.json"

    auto_p.write_text(json.dumps({
        "schema_version": "bts_contest_streak_auto_v2",
        "season": D.year, "activation_id": MATCH_ID,
        "active_streak": 12, "best_streak": 12,
        "source_date": (D - timedelta(days=2)).isoformat(),
        "recorded_at": at_et(D, 20).isoformat(),
    }), encoding="utf-8")
    sw_p.write_text(json.dumps({
        "schema_version": "bts_season_contract_v1",
        "season": D.year, "activation_id": MATCH_ID,
        "status": "automatic", "switched_at": at_et(D, 8).isoformat(), "revoked_at": None,
    }), encoding="utf-8")
    sv_p.write_text(json.dumps({"season": D.year, "state": "active"}), encoding="utf-8")
    cov_prev.write_text('{"result":"hit"}', encoding="utf-8")
    cov_p.write_text('{"result":"hit"}', encoding="utf-8")
    return {
        "manual": acct / "contest_streak.manual.json",
        "automatic": auto_p,
        "switch": sw_p,
        "saver": sv_p,
        "coverage": cov_p,
    }


def child_run(slot: str, variant: str, scratch: Path) -> None:
    from bts.health import alert, contest_state, runner

    picks = scratch / "picks"
    paths = seed_valid_auto_world(picks)
    aux = scratch / "aux"
    aux.mkdir(parents=True, exist_ok=True)

    if slot == "masked_all":
        # Dated observation for D masks automatic, switch, and saver FIFOs!
        paths["manual"].write_text(json.dumps({
            "schema_version": "bts_contest_streak_manual_v3", "season": D.year,
            "effective_date": D.isoformat(), "active_streak": 4, "best_streak": 4,
            "saver_state": "not_earned", "recorded_at": at_et(D, 9).isoformat(),
            "expires_at": datetime.combine(D, time.max, ET).isoformat(), "source": "cli",
        }))
        (picks / f"{D.isoformat()}.json").unlink()
        (picks / f"{(D - timedelta(days=1)).isoformat()}.json").unlink()
        for k in ("automatic", "switch", "saver"):
            paths[k].unlink()
            os.mkfifo(paths[k])
        target = paths["automatic"]
    elif slot == "second_manual_fifo":
        # Manual starts absent during _select_contest_streak_state, then becomes a FIFO before the second manual read!
        target = paths["manual"]
        orig_select = contest_state._select_contest_streak_state

        def swap_to_fifo(*args, **kwargs):
            res = orig_select(*args, **kwargs)
            os.mkfifo(target)
            return res

        contest_state._select_contest_streak_state = swap_to_fifo
    else:
        target = paths[slot]
        valid_bytes = target.read_bytes() if target.exists() else json.dumps({
            "schema_version": "bts_contest_streak_manual_v3", "season": D.year,
            "effective_date": D.isoformat(), "active_streak": 4, "best_streak": 4,
            "saver_state": "not_earned", "recorded_at": at_et(D, 9).isoformat(),
            "expires_at": datetime.combine(D, time.max, ET).isoformat(), "source": "cli",
        }).encode("utf-8")
        target.unlink(missing_ok=True)

        if variant == "fifo":
            os.mkfifo(target)
        elif variant == "dir":
            target.mkdir()
        elif variant == "sym_reg":
            reg = aux / "target.json"
            reg.write_bytes(valid_bytes)
            target.symlink_to(reg)
        elif variant == "sym_fifo":
            fifo = aux / "target.fifo"
            os.mkfifo(fifo)
            target.symlink_to(fifo)
        elif variant == "sym_dir":
            d = aux / "target.dir"
            d.mkdir()
            target.symlink_to(d)
        elif variant == "sym_dangling":
            target.symlink_to(aux / "does_not_exist.json")
        elif variant == "sym_loop":
            loop_a = aux / "loop_a"
            loop_b = aux / "loop_b"
            loop_a.symlink_to(loop_b)
            loop_b.symlink_to(loop_a)
            target.symlink_to(loop_a)
        elif variant == "bad_json":
            target.write_text("{malformed", encoding="utf-8")
        elif variant == "bad_utf8":
            target.write_bytes(b"\xff\xfe\x00")
        else:
            raise ValueError(variant)

    opened = [0]
    orig_open = os.open

    def track_open(p, *a, **kw):
        if Path(p) == target:
            opened[0] += 1
            print("OPENED_TARGET", flush=True)
        return orig_open(p, *a, **kw)

    wall = at_et(D + timedelta(days=1), 2, 0)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return wall.astimezone(tz) if tz is not None else wall.replace(tzinfo=None)

    orig_safe = runner._safe_run
    with (
        patch.object(os, "open", track_open),
        patch.object(runner, "datetime", Clock),
        patch.object(contest_state, "datetime", Clock),
        patch.object(alert, "send_dm", return_value="ack"),
        patch.object(runner, "_safe_run", lambda n, fn: orig_safe(n, fn) if n == "contest_state" else []),
    ):
        alerts = runner.run_all_checks(
            picks, scratch / "models", None, today=D, contest_state_expected=True,
            thresholds_overrides={
                "warn_attention_state": scratch / ".h/w.json",
                "health_dm_delivery_status": scratch / ".h/d.json",
            },
        )
    reasons = [
        a.message.removeprefix(f"Prior slate {D.isoformat()} ET. ").removesuffix(" " + contest_state.HISTORICAL_POINTER)
        for a in alerts
    ]
    print(json.dumps({"opened": opened[0], "reasons": reasons}))


def expected_reasons_for(slot: str, variant: str) -> list[str]:
    lag_warn = "Coverage trails by more than one settled pick. The predictions array can trail the live counter."
    lag_info = "Coverage trails by one settled pick, the expected ledger lag."
    if variant == "sym_reg":
        return [] if slot == "manual" else [lag_warn]
    if variant in {"sym_dangling", "sym_loop"}:
        if slot == "manual":
            return [lag_warn]
        if slot == "automatic":
            return ["Contest streak observation is missing from account_state.", "There is no automatic observation for the switch activation."]
        if slot == "switch":
            return [lag_warn, "The season switch file is malformed or unreadable."]
        if slot == "saver":
            return [lag_warn, "The saver flag is uninitialized or unreadable. The saver counted as spent."]
        if slot == "coverage":
            return [lag_info]
    # Nonregular (fifo, dir, sym_fifo, sym_dir) or bad_json or bad_utf8:
    if slot == "manual":
        return ["The manual contest streak file is malformed or unreadable."]
    if slot == "automatic":
        return ["The automatic contest streak file is malformed or unreadable."]
    if slot == "switch":
        return [lag_warn, "The season switch file is malformed or unreadable."]
    if slot == "saver":
        return [lag_warn, "The saver flag is uninitialized or unreadable. The saver counted as spent."]
    if slot == "coverage":
        return [lag_info]
    raise AssertionError((slot, variant))


def verify_interrupts_and_readonly_contest_files(root: Path) -> None:
    from bts.health import alert, contest_state, runner
    import bts.util as util_mod

    picks = root / "ro_test" / "picks"
    seed_valid_auto_world(picks)
    # Snapshot every file in picks before running health checks
    def snap(p: Path) -> dict[str, bytes]:
        return {str(f.relative_to(p)): f.read_bytes() for f in sorted(p.rglob("*")) if f.is_file()}

    before = snap(picks)
    wall = at_et(D + timedelta(days=1), 2, 0)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return wall.astimezone(tz) if tz is not None else wall.replace(tzinfo=None)

    # 1. KeyboardInterrupt and SystemExit propagate out of _safe_run without writing contest files
    for exc_cls in (KeyboardInterrupt, SystemExit):
        with (
            patch.object(runner, "datetime", Clock),
            patch.object(util_mod, "os") as _unused,
        ):
            pass
        with (
            patch.object(runner, "datetime", Clock),
            patch.object(contest_state, "read_regular_file_text", side_effect=exc_cls()),
            patch.object(contest_state, "_select_contest_streak_state", side_effect=exc_cls()),
        ):
            try:
                runner.run_all_checks(picks, root / "models", "op.invalid", today=D, contest_state_expected=True)
            except exc_cls:
                pass
            else:
                raise AssertionError(f"{exc_cls} did not propagate")
        assert snap(picks) == before

    # 2. Full historical & same-day runs with read-only permissions on picks_dir & account_state -> 0 writes!
    acct = picks / "account_state"
    for f in picks.rglob("*"):
        if f.is_file():
            f.chmod(0o444)
    acct.chmod(0o555)
    picks.chmod(0o555)
    try:
        with (
            patch.object(runner, "datetime", Clock),
            patch.object(contest_state, "datetime", Clock),
            patch.object(alert, "send_dm", return_value="ack"),
        ):
            runner.run_all_checks(picks, root / "models", "op.invalid", today=D, contest_state_expected=True)
            runner.run_all_checks(picks, root / "models", "op.invalid", today=D + timedelta(days=1), contest_state_expected=True)
        assert snap(picks) == before
    finally:
        picks.chmod(0o755)
        acct.chmod(0o755)
        for f in picks.rglob("*"):
            if f.is_file():
                f.chmod(0o644)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        logging.disable(logging.CRITICAL)
        child_run(sys.argv[2], sys.argv[3], Path(sys.argv[4]))
        return

    signal.alarm(55)
    logging.disable(logging.CRITICAL)
    slots = ("manual", "automatic", "switch", "saver", "coverage")
    variants = ("fifo", "dir", "sym_reg", "sym_fifo", "sym_dir", "sym_dangling", "sym_loop", "bad_json", "bad_utf8")
    total = 0
    with TemporaryDirectory(dir=HERE) as tmp:
        root = Path(tmp)
        for slot in slots:
            for variant in variants:
                cdir = root / f"{slot}_{variant}"
                cdir.mkdir()
                proc = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), "child", slot, variant, str(cdir)],
                    capture_output=True, text=True, timeout=4, check=True,
                )
                payload = json.loads(proc.stdout.splitlines()[-1])
                exp_reasons = expected_reasons_for(slot, variant)
                assert payload["reasons"] == exp_reasons, (slot, variant, payload, exp_reasons)
                if variant in {"fifo", "sym_fifo", "sym_reg", "bad_json", "bad_utf8"}:
                    assert payload["opened"] >= 1, (slot, variant, payload)
                total += 1

        # Masked FIFOs + second manual read FIFO
        for special_slot, exp_open, exp_r in [
            ("masked_all", 0, []),
            ("second_manual_fifo", 1, [
                "Coverage trails by more than one settled pick. The predictions array can trail the live counter.",
                "A legacy or expired manual override file is present.",
            ]),
        ]:
            cdir = root / special_slot
            cdir.mkdir()
            proc = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "child", special_slot, "fifo", str(cdir)],
                capture_output=True, text=True, timeout=4, check=True,
            )
            payload = json.loads(proc.stdout.splitlines()[-1])
            assert payload["opened"] == exp_open and payload["reasons"] == exp_r, (special_slot, payload)
            total += 1

        verify_interrupts_and_readonly_contest_files(root)
        print(f"[PASS] Item 5: {total} bounded child runs (5 slots x 9 hostile variants + masked FIFOs + 2nd manual FIFO) + interrupts + read-only contest dir verified")
    signal.alarm(0)


if __name__ == "__main__":
    main()
