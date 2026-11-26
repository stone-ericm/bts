"""Round-2 Probe 1c: real send_dm 1000-grapheme transport replay (D/D+1, invalid+unreadable pair, 16 records) + shell 2>&- closed-fd-2 check."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
from datetime import date, datetime
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

R2_DIR = Path(__file__).resolve().parent
REPO_ROOT = R2_DIR.parents[2]
HEAD_SRC = REPO_ROOT / "src"

sys.path.insert(0, str(HEAD_SRC))

from bts import dm as dm_mod
from bts.features import compute, probable_pitcher_cache as ppc
from bts.health import alert as alert_mod, runner
from bts.probable_pitcher_status import SCHEMA, SOURCE


def check_r4_dm_transport(root: Path) -> None:
    print("=== R2-1c. Real send_dm + 1,000-grapheme chat cap transport replay ===")
    sent_wire: list[str] = []

    def fake_retry_urlopen(req: object, timeout: int = 15, idempotent: bool = True):
        url = req.full_url
        if url.endswith("/com.atproto.server.createSession"):
            return io.BytesIO(b'{"accessJwt": "fake-jwt"}')
        if "/com.atproto.identity.resolveHandle" in url:
            return io.BytesIO(b'{"did": "did:plc:testrecipient"}')
        if "/chat.bsky.convo.getConvoForMembers" in url:
            return io.BytesIO(b'{"convo": {"id": "convo-1"}}')
        if url.endswith("/chat.bsky.convo.sendMessage"):
            body = json.loads(req.data.decode("utf-8"))
            text = body["message"]["text"]
            if len(text) > 1000:
                raise HTTPError(url, 400, f"Message exceeds 1000-grapheme limit ({len(text)})", {}, io.BytesIO(b"too long"))
            sent_wire.append(text)
            return io.BytesIO(f'{{"id": "msg-{len(sent_wire)}"}}'.encode())
        raise AssertionError(f"Unexpected URL: {url}")

    def freeze_all(ts_iso: str):
        at = datetime.fromisoformat(ts_iso)
        class Clock(datetime):
            @classmethod
            def now(cls, tz=None):
                return at.astimezone(tz) if tz is not None else at.replace(tzinfo=None)
        ppc.datetime = Clock
        runner.datetime = Clock
        alert_mod.datetime = Clock

    orig_safe = runner._safe_run
    with (
        patch.object(dm_mod, "get_dm_password", return_value="fake-pw"),
        patch.object(dm_mod, "retry_urlopen", fake_retry_urlopen),
        patch.object(runner.realized_calibration, "_current_deploy_iso", return_value=None),
        patch.object(runner, "_safe_run", lambda name, fn: orig_safe(name, fn) if name == SOURCE else []),
    ):
        # 1. Persistent {"100": null} cache across D and D+1
        w1 = root / "persistent"
        cache1 = w1 / "data/models/probable_pitcher_lookup.json"
        raw1 = w1 / "data/raw"
        picks1 = w1 / "data/picks"
        cache1.parent.mkdir(parents=True)
        raw1.mkdir(parents=True)
        picks1.mkdir(parents=True)
        cache1.write_text('{"100": null}')

        with patch.object(compute, "_probable_pitcher_cache_path", cache1):
            freeze_all("2026-12-31T23:00:00+00:00")
            compute._build_probable_pitcher_lookup(str(raw1))
            a_d = runner.run_all_checks(picks1, cache1.parent, "eric.bsky.social", today=date(2026, 12, 31))
            assert len(a_d) == 1 and len(a_d[0].message) <= 600
            assert len(sent_wire) == 1 and len(sent_wire[-1]) < 1000

            freeze_all("2027-01-01T23:00:00+00:00")
            compute._build_probable_pitcher_lookup(str(raw1))
            a_d1 = runner.run_all_checks(picks1, cache1.parent, "eric.bsky.social", today=date(2027, 1, 1))
            assert len(a_d1) == 1 and len(a_d1[0].message) <= 600
            assert len(sent_wire) == 2 and len(sent_wire[-1]) < 1000
            print(f"  Persistent D/D+1 wire lengths: D={len(sent_wire[0])}, D+1={len(sent_wire[1])}")

        # 2. Same-day invalid_entries + unreadable_cache pair across D and D+1
        w2 = root / "pair"
        cache2 = w2 / "data/models/probable_pitcher_lookup.json"
        raw2 = w2 / "data/raw"
        picks2 = w2 / "data/picks"
        cache2.parent.mkdir(parents=True)
        raw2.mkdir(parents=True)
        picks2.mkdir(parents=True)
        with patch.object(compute, "_probable_pitcher_cache_path", cache2):
            freeze_all("2026-12-31T18:00:00+00:00")
            cache2.write_text('{"100": null}')
            compute._build_probable_pitcher_lookup(str(raw2))
            cache2.write_text("{corrupt")
            compute._build_probable_pitcher_lookup(str(raw2))
            a_p0 = runner.run_all_checks(picks2, cache2.parent, "eric.bsky.social", today=date(2026, 12, 31))
            assert len(a_p0) == 1 and len(a_p0[0].message) <= 600
            assert len(sent_wire) == 3 and len(sent_wire[-1]) < 1000

            freeze_all("2027-01-01T18:00:00+00:00")
            a_p1 = runner.run_all_checks(picks2, cache2.parent, "eric.bsky.social", today=date(2027, 1, 1))
            assert len(a_p1) == 1 and len(a_p1[0].message) <= 600
            assert len(sent_wire) == 4 and len(sent_wire[-1]) < 1000
            print(f"  Same-day invalid+unreadable pair wire lengths: D={len(sent_wire[2])}, D+1={len(sent_wire[3])}")

        # 3. 16 records (15 valid + 1 bad) through full runner + send_dm (both short relative cwd and >600 fallback)
        w3 = root / "sixteen"
        picks3 = w3 / "data/picks"
        models3 = w3 / "data/models"
        sdir3 = w3 / "data/health_state" / SOURCE
        picks3.mkdir(parents=True)
        models3.mkdir(parents=True)
        sdir3.mkdir(parents=True)
        for num in range(15):
            inv = bool(num % 2)
            ident = f"{num:012x}" + "c" * 52
            reason = "invalid_entries" if inv else "unreadable_cache"
            rec = {
                "schema_version": SCHEMA, "reason": reason,
                "recorded_at": "2026-12-31T14:00:00+00:00", "recorded_date_et": "2026-12-31",
                "cache_path": "data/models/probable_pitcher_lookup.json",
                "dropped": 1 if inv else None, "recovered_from_cache": 0 if inv else None,
                "rescanned": 0, "unusable_feed": 0 if inv else None, "without_feed": 1 if inv else None,
                "game_ids": [100] if inv else [], "identity": ident, "preserved": None, "original_sha256": "d" * 64,
            }
            (sdir3 / f"2026-12-31.{reason}.{ident[:12]}.json").write_text(json.dumps(rec))
        (sdir3 / "2026-12-31.corrupt.json").write_text("{")
        freeze_all("2026-12-31T23:00:00+00:00")
        a_16 = runner.run_all_checks(picks3, models3, "eric.bsky.social", today=date(2026, 12, 31))
        assert len(a_16) == 1 and len(a_16[0].message) <= 600
        assert len(sent_wire) == 5 and len(sent_wire[-1]) < 1000
        print(f"  16-record full-runner wire length: {len(sent_wire[4])} (alert len={len(a_16[0].message)})")
        assert all("`" not in m for m in sent_wire)


def check_closed_fd2_at_startup(root: Path) -> None:
    print("=== R2-1d. Check warn() when fd 2 is closed at process startup (2>&-, sys.stderr is None) ===")
    w = root / "fd2_closed"
    cache = w / "data/models/probable_pitcher_lookup.json"
    raw = w / "data/raw"
    cache.parent.mkdir(parents=True)
    raw.mkdir(parents=True)
    cache.write_text('{"100": null}')

    code = (
        "import json, sys; from pathlib import Path; "
        "from bts.features import compute; "
        "compute._probable_pitcher_cache_path = Path(sys.argv[1]); "
        "d = compute._build_probable_pitcher_lookup(sys.argv[2]); "
        "print(json.dumps({'stderr_is_none': sys.stderr is None, 'lookup': d}))"
    )
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = str(HEAD_SRC)
    cp = subprocess.run(
        ["/bin/sh", "-c", f'"{sys.executable}" -c "$1" "$2" "$3" 2>&-', "sh", code, str(cache), str(raw)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=10,
    )
    print(f"  2>&- returncode={cp.returncode}, stdout={cp.stdout!r}")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="r2_p1c_") as tmp:
        root = Path(tmp)
        check_r4_dm_transport(root)
        check_closed_fd2_at_startup(root)
    print("ALL ROUND-2 PROBE 1C CHECKS COMPLETED")


if __name__ == "__main__":
    main()
