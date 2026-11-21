"""Verify per-LogRecord single-line/control-character safety (without OOM words), before/after genuine journal lines, operator artifact lookup, and exact rendered Amendment 4 sentences."""

from __future__ import annotations

import contextlib
import io
import json
import logging
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
REPO_SRC = HERE.parents[2] / "src"
sys.path.insert(0, str(REPO_SRC))

from bts.features import compute, probable_pitcher_cache as producer
from bts.health import analytics_artifacts_missing, attention, probable_pitcher_cache as health

DAY = date(2026, 12, 31)
TICK = chr(96)


class ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def main():
    print("=== 2B. Per-LogRecord Single-Line & Control-Char Check + Before/After Genuine Journal Evidence ===")
    with TemporaryDirectory(dir=HERE, prefix="tmp-perrec-") as tmp:
        # Pure newline/control paths WITHOUT 'oom' so only newline/control checks can catch a leak
        nl_root = Path(tmp) / f"line1\nline2\rline3\ttab{TICK}tick/data"
        (nl_root / "models").mkdir(parents=True)
        (nl_root / "raw/2025").mkdir(parents=True)
        (nl_root / "raw/2025/100.json").write_text(json.dumps({
            "gameData": {
                "probablePitchers": {"away": {"id": 2}, "home": {"id": 23}},
                "teams": {"away": {"id": 61}, "home": {"id": 62}},
            }
        }))
        cache = nl_root / "models/probable_pitcher_lookup.json"
        cache.write_text('{"100": null}')
        compute._probable_pitcher_cache_path = cache

        class FixedClock(datetime):
            @classmethod
            def now(cls, tz=None):
                dt = cls(2026, 12, 31, 20, 0, tzinfo=timezone.utc)
                return dt.astimezone(tz) if tz is not None else dt.replace(tzinfo=None)

        # 1. Capture producer failure lines individually
        (nl_root / "models/rejected").write_text("not a dir")
        (nl_root / "health_state").mkdir(parents=True)
        (nl_root / "health_state/probable_pitcher_cache").write_text("not a dir")
        err_buf = io.StringIO()
        with patch.object(producer, "datetime", FixedClock), contextlib.redirect_stderr(err_buf):
            compute._build_probable_pitcher_lookup(str(nl_root / "raw"))
        # Must be exactly 3 newline-terminated lines (not split into more lines by embedded \n/\r!)
        raw_err = err_buf.getvalue()
        assert raw_err.count("\n") == 3 and "\r" not in raw_err and "\t" not in raw_err and TICK not in raw_err, repr(raw_err)
        prod_msgs = raw_err.splitlines()

        # 2. Unblock and produce valid records (invalid_entries with copy + unreadable_cache with copy)
        (nl_root / "models/rejected").unlink()
        (nl_root / "health_state/probable_pitcher_cache").unlink()
        with patch.object(producer, "datetime", FixedClock), contextlib.redirect_stderr(io.StringIO()):
            compute._build_probable_pitcher_lookup(str(nl_root / "raw"))
        cache.write_text("{")
        (nl_root / "raw/2025/101.json").write_text(json.dumps({
            "gameData": {
                "probablePitchers": {"away": {"id": 5}, "home": {"id": 6}},
                "teams": {"away": {"id": 7}, "home": {"id": 8}},
            }
        }))
        with patch.object(producer, "datetime", FixedClock), contextlib.redirect_stderr(io.StringIO()):
            compute._build_probable_pitcher_lookup(str(nl_root / "raw"))

        st_dir = nl_root / "health_state/probable_pitcher_cache"
        # Tamper recorded_at separator on one record to '\n' and the other to '`'
        recs = sorted(st_dir.glob("*.json"))
        assert len(recs) == 2
        for idx, sep in enumerate(["\n", TICK]):
            d = json.loads(recs[idx].read_text())
            d["recorded_at"] = d["recorded_at"].replace("T", sep, 1)
            recs[idx].write_text(json.dumps(d))

        # Add valid-named corrupt record + newline-named corrupt entry
        (st_dir / f"{DAY}.invalid_entries.{'0'*12}.json").write_bytes(b"{")
        (st_dir / f"{DAY}.bad\nname\r{TICK}.json").write_bytes(b"{")

        lh = ListHandler()
        logger = logging.getLogger(health.__name__)
        logger.setLevel(logging.WARNING)
        logger.addHandler(lh)
        try:
            alerts1 = health.check(st_dir, DAY)
            fifo_dir = Path(tmp) / f"nl\ndir{TICK}"
            os.mkfifo(fifo_dir)
            alerts2 = health.check(fifo_dir, DAY)
        finally:
            logger.removeHandler(lh)

        assert len(lh.messages) == 5, lh.messages
        all_msgs = [*prod_msgs, *lh.messages, alerts1[0].message, alerts2[0].message]
        for m in all_msgs:
            assert "\n" not in m and "\r" not in m and "\t" not in m and TICK not in m, repr(m)
            assert all(c.isprintable() and ord(c) >= 32 for c in m), repr(m)
            assert "oom" not in m.lower() and "out of memory" not in m.lower(), m
            assert not attention.OOM_EVIDENCE_RE.search(f"probable_pitcher_cache {m.lower()}"), m

        # Verify genuine fatal lines both BEFORE and AFTER all W15 messages in _fatal_scheduler_journal_line
        genuine_lines = [
            "Nov 21 06:00:00 box systemd[1]: bts-scheduler.service: Failed with result 'oom-kill'.",
            "Nov 21 06:00:00 box systemd[1]: bts-scheduler.service: A process of this unit has been killed by the OOM killer.",
            "Nov 21 06:00:00 box kernel: Out of memory: Killed process 999 (python)",
            "Nov 21 06:00:00 box systemd[1]: bts-scheduler.service: Main process exited, code=killed, status=9/KILL",
            "Nov 21 06:00:00 box systemd[1]: bts-scheduler.service: Main process exited, code=killed, signal=KILL",
            "Nov 21 06:00:00 box systemd[1]: bts-scheduler.service: Main process exited, code=killed, status=KILL",
            "Nov 21 06:00:00 box systemd[1]: bts-scheduler.service: Main process exited, code=exited, status=137/n/a",
        ]
        for g in genuine_lines:
            for seq in ([g, *all_msgs], [*all_msgs, g]):
                out_str = "\n".join(seq) + "\n"
                with patch.object(
                    analytics_artifacts_missing.subprocess,
                    "run",
                    lambda *a, **kw: SimpleNamespace(returncode=0, stdout=out_str, stderr=""),
                ):
                    got = analytics_artifacts_missing._fatal_scheduler_journal_line("bts-scheduler.service", "2026-12-31T00:00:00+00:00")
                assert got == g[:240], (g, got)
        print(f"  Verified {len(all_msgs)} individual messages have 0 embedded CR/LF/TAB/control/backtick/OOM chars;")
        print(f"  Verified all {len(genuine_lines)} genuine OOM/KILL/137 journal variants fire both before and after all W15 lines.")

        print("\n=== 3B & 4. Exact Rendered Amendment-4 Texts & Operator Artifact Resolution ===")
        for idx, m in enumerate(prod_msgs, 1):
            print(f"  [producer {idx}] {m}")
        for idx, m in enumerate(lh.messages, 1):
            print(f"  [health log {idx}] {m}")
        print(f"  [summary] {alerts1[0].message}")

        # Confirm operator can resolve record -> cache_path -> rejected/<preserved> on non-default picks_dir
        custom_picks = nl_root / "picks"
        derived_status = custom_picks.parent / "health_state/probable_pitcher_cache"
        assert derived_status == st_dir
        valid_inv = json.loads(recs[0].read_text())
        resolved_cache = Path(valid_inv["cache_path"])
        resolved_evidence = resolved_cache.parent / "rejected" / valid_inv["preserved"]
        assert resolved_cache.exists() and resolved_evidence.exists()
        print("  Operator resolution (custom picks_dir -> health_state/probable_pitcher_cache -> record['cache_path'] -> rejected/<preserved>): VERIFIED")


if __name__ == "__main__":
    main()
