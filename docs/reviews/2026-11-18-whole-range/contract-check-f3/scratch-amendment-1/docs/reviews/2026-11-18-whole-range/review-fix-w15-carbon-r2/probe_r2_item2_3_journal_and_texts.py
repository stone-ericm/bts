"""Round-2 Probe 2 & 3: standalone unconfigured-logging journal verification, incident_key dedup matrix, boundary lengths, and operator text checks."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

R2_DIR = Path(__file__).resolve().parent
REPO_ROOT = R2_DIR.parents[2]
HEAD_SRC = REPO_ROOT / "src"

sys.path.insert(0, str(HEAD_SRC))

from bts.health import alert as alert_mod, attention as attention_mod, probable_pitcher_cache as h_ppc, runner
from bts.probable_pitcher_status import SCHEMA, SOURCE

DAY = date(2026, 12, 31)


def make_rec(day: date, reason: str, ident_char: str, preserved: str | None = None) -> tuple[str, dict]:
    ident = ident_char * 64
    inv = reason == "invalid_entries"
    row = {
        "schema_version": SCHEMA,
        "reason": reason,
        "recorded_at": f"{day.isoformat()}T14:00:00+00:00",
        "recorded_date_et": day.isoformat(),
        "cache_path": "data/models/probable_pitcher_lookup.json",
        "dropped": 2 if inv else None,
        "recovered_from_cache": 0 if inv else None,
        "rescanned": 1,
        "unusable_feed": 0 if inv else None,
        "without_feed": 1 if inv else None,
        "game_ids": [100, 101] if inv else [],
        "identity": ident,
        "preserved": preserved,
        "original_sha256": "f" * 64,
    }
    return f"{day.isoformat()}.{reason}.{ident[:12]}.json", row


def check_real_journal_logging_without_pytest(root: Path) -> None:
    print("=== R2-2a. Standalone child process (no pytest caplog, no basicConfig): where does log.warning actually go? ===")
    # Import bts.cli and bts.scheduler (exact bts-scheduler.service entry path!) and run run_all_checks!
    code = """
import json, logging, sys
from datetime import date
from pathlib import Path
from unittest.mock import patch
import bts.cli
import bts.scheduler
from bts.health import alert, runner

work = Path(sys.argv[1])
root_handlers = [type(h).__name__ for h in logging.getLogger().handlers]
bts_handlers = [type(h).__name__ for h in logging.getLogger("bts").handlers]
ppc_handlers = [type(h).__name__ for h in logging.getLogger("bts.health.probable_pitcher_cache").handlers]
last_resort = type(logging.lastResort).__name__ if logging.lastResort else None

orig_safe = runner._safe_run
with (
    patch.object(alert, "send_dm", return_value="msg-1"),
    patch.object(runner.realized_calibration, "_current_deploy_iso", return_value=None),
    patch.object(runner, "_safe_run", lambda name, fn: orig_safe(name, fn) if name == "probable_pitcher_cache" else []),
):
    alerts = runner.run_all_checks(work / "data/picks", work / "data/models", "eric.bsky.social", today=date(2026, 12, 31))

print(json.dumps({
    "root_handlers": root_handlers,
    "bts_handlers": bts_handlers,
    "ppc_handlers": ppc_handlers,
    "last_resort": last_resort,
    "alert_count": len(alerts),
    "alert_msg": alerts[0].message,
}))
"""
    w = root / "journal_probe"
    sdir = w / "data/health_state" / SOURCE
    (w / "data/picks").mkdir(parents=True)
    (w / "data/models").mkdir(parents=True)
    sdir.mkdir(parents=True)
    fn1, r1 = make_rec(DAY, "invalid_entries", "a", "probable_pitcher_lookup.20261231T140000Z.ffffffffffff.json")
    fn2, r2 = make_rec(DAY, "unreadable_cache", "b", None)
    (sdir / fn1).write_text(json.dumps(r1))
    (sdir / fn2).write_text(json.dumps(r2))
    (sdir / f"{DAY}.corrupt.json").write_text("{bad")

    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = str(HEAD_SRC)
    cp = subprocess.run([sys.executable, "-c", code, str(w)], env=env, capture_output=True, text=True, timeout=15)
    assert cp.returncode == 0, cp.stderr
    meta = json.loads(cp.stdout)
    stderr_lines = cp.stderr.strip().splitlines()
    print(f"  Logger state: { {k: meta[k] for k in ('root_handlers', 'bts_handlers', 'ppc_handlers', 'last_resort')} }")
    print(f"  Captured stderr lines ({len(stderr_lines)}):")
    for idx, ln in enumerate(stderr_lines, 1):
        print(f"    [{idx}] {ln[:120]}{'...' if len(ln) > 120 else ''}")

    assert meta["root_handlers"] == [] and meta["bts_handlers"] == [] and meta["ppc_handlers"] == []
    assert meta["last_resort"] == "_StderrHandler"
    # Verify all 4 lines hit stderr (which systemd StandardError=journal routes to journalctl --user -u bts-scheduler!):
    # 1) valid invalid_entries A7 message
    # 2) valid unreadable_cache A7 message
    # 3) status unreadable message for 2026-12-31.corrupt.json
    # 4) runner's log_alerts: [probable_pitcher_cache WARN] Probable-pitcher cache: 3 observations ...
    assert len(stderr_lines) == 4
    assert any("had 2 invalid entries" in ln and "`" not in ln for ln in stderr_lines)
    assert any("was unreadable at 2026-12-31T14:00:00+00:00" in ln and "`" not in ln for ln in stderr_lines)
    assert any(f"Probable-pitcher cache status unreadable at {sdir / f'{DAY}.corrupt.json'}." in ln for ln in stderr_lines)
    assert stderr_lines[-1].startswith("[probable_pitcher_cache WARN] Probable-pitcher cache: 3 observations (invalid entries, status unreadable, unreadable cache)")


def check_incident_key_dedup_matrix_and_length_bounds(root: Path) -> None:
    print("=== R2-2b & 3. Incident key dedup matrix (same-day change, D+1 repeat + streak label, status_unreadable->repaired) & 600-char boundary ===")
    w = root / "dedup_matrix"
    picks = w / "data/picks"
    models = w / "data/models"
    sdir = w / "data/health_state" / SOURCE
    picks.mkdir(parents=True)
    models.mkdir(parents=True)
    sdir.mkdir(parents=True)

    sent: list[str] = []
    def fake_send(rec: str, body: str) -> str:
        sent.append(body)
        return f"m-{len(sent)}"

    orig_safe = runner._safe_run
    with (
        patch.object(alert_mod, "send_dm", fake_send),
        patch.object(runner.realized_calibration, "_current_deploy_iso", return_value=None),
        patch.object(runner, "_safe_run", lambda name, fn: orig_safe(name, fn) if name == SOURCE else []),
    ):
        # 1. Start with a corrupt status file on D -> sends DM #1
        bad_file = sdir / f"{DAY}.invalid_entries.aaaaaaaaaaaa.json"
        bad_file.write_text("{corrupt")
        runner.run_all_checks(picks, models, "eric.bsky.social", today=DAY)
        assert len(sent) == 1 and "(status unreadable)" in sent[-1]

        # 2. Repeat run on D with same corrupt status file -> suppressed!
        runner.run_all_checks(picks, models, "eric.bsky.social", today=DAY)
        assert len(sent) == 1

        # 3. Replace corrupt file with valid record 'a'*64 on D -> key changes from filename to 64-hex identity -> sends DM #2!
        _, r_a = make_rec(DAY, "invalid_entries", "a", None)
        bad_file.write_text(json.dumps(r_a))
        runner.run_all_checks(picks, models, "eric.bsky.social", today=DAY)
        assert len(sent) == 2 and "1 observation (invalid entries)" in sent[-1]

        # 4. Add second valid record 'b'*64 on D -> sends DM #3!
        fn_b, r_b = make_rec(DAY, "unreadable_cache", "b", None)
        (sdir / fn_b).write_text(json.dumps(r_b))
        runner.run_all_checks(picks, models, "eric.bsky.social", today=DAY)
        assert len(sent) == 3 and "2 observations (invalid entries, unreadable cache)" in sent[-1]

        # 5. Next processed day D+1 (no new files): sends DM #4, suppressed on second D+1 run!
        runner.run_all_checks(picks, models, "eric.bsky.social", today=DAY + timedelta(days=1))
        assert len(sent) == 4
        print(f"  D+1 repeat DM tail: ...{sent[-1][-80:]!r}")
        runner.run_all_checks(picks, models, "eric.bsky.social", today=DAY + timedelta(days=1))
        assert len(sent) == 4

        # 6. Now add a recurrence of 'a'*64 dated D+1 (so D has 'a'*64 and D+1 ALSO has 'a'*64 -> ['a'*64, 'a'*64, 'b'*64]) -> sends DM #5!
        fn_a_d1, r_a_d1 = make_rec(DAY + timedelta(days=1), "invalid_entries", "a", None)
        (sdir / fn_a_d1).write_text(json.dumps(r_a_d1))
        runner.run_all_checks(picks, models, "eric.bsky.social", today=DAY + timedelta(days=1))
        assert len(sent) == 5 and "3 observations (invalid entries, unreadable cache)" in sent[-1]

    # Boundary check on MAX_ALERT_LENGTH == 600 in summarize():
    all_kinds = {"invalid entries", "unreadable cache", "status unreadable"}
    base_alert = h_ppc.summarize(Path("X"), DAY, ["a" * 64, "b" * 64], all_kinds)
    overhead = len(base_alert.message) - 1  # length without status_dir
    exact_600_dir = Path("p" * (600 - overhead))
    over_601_dir = Path("p" * (601 - overhead))
    a_600 = h_ppc.summarize(exact_600_dir, DAY, ["a" * 64, "b" * 64], all_kinds)
    a_601 = h_ppc.summarize(over_601_dir, DAY, ["a" * 64, "b" * 64], all_kinds)
    assert len(a_600.message) == 600 and str(exact_600_dir) in a_600.message
    assert len(a_601.message) <= 600 and "details in data/health_state/probable_pitcher_cache." in a_601.message
    default_a = h_ppc.summarize(Path("data/health_state/probable_pitcher_cache"), DAY, ["a" * 64, "b" * 64], all_kinds)
    print(f"  Default 3-kind summary length={len(default_a.message)}; exact-600 kept custom path; 601 fell back to {len(a_601.message)} chars.")
    print(f"  Rendered default summary:\n    {default_a.message}")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="r2_p23_") as tmp:
        root = Path(tmp)
        check_real_journal_logging_without_pytest(root)
        check_incident_key_dedup_matrix_and_length_bounds(root)
    print("ALL ROUND-2 PROBE 2 & 3 CHECKS PASSED")


if __name__ == "__main__":
    main()
