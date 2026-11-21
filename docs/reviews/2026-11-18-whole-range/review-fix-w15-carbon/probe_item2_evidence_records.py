"""Probe family 2: evidence preservation, sha256 verification, A4 identity formula, coalescing, 64 KiB limit, and failure lines."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

REVIEW_DIR = Path(__file__).resolve().parent
REPO_ROOT = REVIEW_DIR.parents[1]
HEAD_SRC = REPO_ROOT / "src"

sys.path.insert(0, str(HEAD_SRC))

from bts.features import compute, probable_pitcher_cache as ppc
from bts.probable_pitcher_status import MAX_BYTES


VALID = {"away": 1, "home": 2, "away_tid": 51, "home_tid": 52}
VALID2 = {"away": 3, "home": 4, "away_tid": 61, "home_tid": 62}


def feed_json(entry: dict) -> str:
    return json.dumps({
        "gameData": {
            "probablePitchers": {"away": {"id": entry["away"]}, "home": {"id": entry["home"]}},
            "teams": {"away": {"id": entry["away_tid"]}, "home": {"id": entry["home_tid"]}},
        }
    })


def freeze(ts_iso: str):
    at = datetime.fromisoformat(ts_iso)
    class Frozen(datetime):
        @classmethod
        def now(cls, tz=None):
            return at.astimezone(tz) if tz is not None else at.replace(tzinfo=None)
    ppc.datetime = Frozen


def independent_a4_identity(
    *,
    reason: str,
    cache_path: str,
    dropped: int | None,
    recovered_from_cache: int | None,
    rescanned: int,
    unusable_feed: int | None,
    without_feed: int | None,
    original_sha256: str | None,
    preserved: str | None,
    outcomes: list,
) -> str:
    obj = {
        "reason": reason,
        "cache_path": cache_path,
        "dropped": dropped,
        "recovered_from_cache": recovered_from_cache,
        "rescanned": rescanned,
        "unusable_feed": unusable_feed,
        "without_feed": without_feed,
        "original_sha256": original_sha256,
        "preserved": preserved,
        "outcomes": outcomes,
    }
    raw = json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def run_child_build(cache_path: Path, raw_dir: Path, ts: str = "2027-01-01T04:59:00+00:00") -> subprocess.CompletedProcess:
    code = (
        "import json, sys; from datetime import datetime; from pathlib import Path; "
        "from bts.features import compute, probable_pitcher_cache as ppc; "
        "at = datetime.fromisoformat(sys.argv[3]); "
        "ppc.datetime = type('F', (datetime,), {'now': classmethod(lambda cls, tz=None: at.astimezone(tz) if tz else at.replace(tzinfo=None))}); "
        "compute._probable_pitcher_cache_path = Path(sys.argv[1]); "
        "res = compute._build_probable_pitcher_lookup(sys.argv[2]); "
        "print(json.dumps({str(k): v for k, v in res.items()}))"
    )
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = str(HEAD_SRC)
    return subprocess.run(
        [sys.executable, "-c", code, str(cache_path), str(raw_dir), ts],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )


def test_exact_bytes_and_reuse(root: Path) -> None:
    print("=== 2(a). Exact original bytes (including undecodable) & full-sha256 regular-file reuse ===")
    # 1. Undecodable UTF-16 / arbitrary binary bytes preserved byte-for-byte
    for idx, blob in enumerate([
        b"\xff\xfe\x00\x80invalid-utf8-\r\n-\r-\n",
        b'{"100": null, "unicode": "\\u00e9"}\r\n  ',
    ]):
        w = root / f"bytes_{idx}"
        cache = w / "data/models/probable_pitcher_lookup.json"
        raw = w / "data/raw"
        (raw / "2025").mkdir(parents=True)
        cache.parent.mkdir(parents=True)
        cache.write_bytes(blob)
        (raw / "2025/100.json").write_text(feed_json(VALID))
        cp = run_child_build(cache, raw)
        assert cp.returncode == 0, cp.stderr
        rej = list((cache.parent / "rejected").glob("*.json"))
        assert len(rej) == 1
        assert rej[0].read_bytes() == blob
        sha = hashlib.sha256(blob).hexdigest()
        assert rej[0].name == f"probable_pitcher_lookup.20270101T045900Z.{sha[:12]}.json"

    # 2. Reuse existing earlier-timestamped candidate when full sha256 matches, even if 12-hex suffix in earlier file differs or timestamp differs;
    #    and reject fake suffix match with wrong bytes, symlink to real bytes, FIFO, directory!
    blob = b'{"100": "bad_value"}'
    sha = hashlib.sha256(blob).hexdigest()

    # Valid reuse of lexically first regular file matching full sha256
    w_reuse = root / "reuse_valid"
    cache = w_reuse / "data/models/probable_pitcher_lookup.json"
    raw = w_reuse / "data/raw"
    (raw / "2025").mkdir(parents=True)
    rej_dir = cache.parent / "rejected"
    rej_dir.mkdir(parents=True)
    cache.write_bytes(blob)
    (raw / "2025/100.json").write_text(feed_json(VALID))
    # Put a fake earlier name with WRONG bytes, a symlink with RIGHT bytes, a FIFO, and a valid earlier regular copy!
    (rej_dir / f"probable_pitcher_lookup.20260101T000000Z.{sha[:12]}.json").write_bytes(b"wrong bytes")
    real_outside = w_reuse / "outside.json"
    real_outside.write_bytes(blob)
    (rej_dir / f"probable_pitcher_lookup.20260102T000000Z.{sha[:12]}.json").symlink_to(real_outside)
    os.mkfifo(rej_dir / f"probable_pitcher_lookup.20260103T000000Z.{sha[:12]}.json")
    earlier_good = rej_dir / f"probable_pitcher_lookup.20260104T000000Z.{sha[:12]}.json"
    earlier_good.write_bytes(blob)
    later_good = rej_dir / f"probable_pitcher_lookup.20260105T000000Z.{sha[:12]}.json"
    later_good.write_bytes(blob)

    cp = run_child_build(cache, raw)
    assert cp.returncode == 0, cp.stderr
    st = json.loads(next((w_reuse / "data/health_state/probable_pitcher_cache").glob("*.json")).read_text())
    assert st["preserved"] == earlier_good.name, st["preserved"]
    assert json.loads(cache.read_text()) == {"100": VALID}

    # 3. Occupied exact target (fake content, symlink to real content, FIFO, directory) with NO verified candidate -> preservation failure!
    for kind in ("wrong_bytes", "symlink_to_real", "fifo", "directory"):
        w_occ = root / f"occ_ev_{kind}"
        cache = w_occ / "data/models/probable_pitcher_lookup.json"
        raw = w_occ / "data/raw"
        (raw / "2025").mkdir(parents=True)
        rej_dir = cache.parent / "rejected"
        rej_dir.mkdir(parents=True)
        cache.write_bytes(blob)
        (raw / "2025/100.json").write_text(feed_json(VALID))
        dest = rej_dir / f"probable_pitcher_lookup.20270101T045900Z.{sha[:12]}.json"
        if kind == "wrong_bytes":
            dest.write_bytes(b"impostor")
        elif kind == "symlink_to_real":
            dest.symlink_to(cache)
        elif kind == "fifo":
            os.mkfifo(dest)
        elif kind == "directory":
            dest.mkdir()
        cp = run_child_build(cache, raw)
        assert cp.returncode == 0, cp.stderr
        assert json.loads(cp.stdout) == {"100": VALID}
        assert cache.read_bytes() == blob, f"active cache must stay untouched on {kind}"
        lines = cp.stderr.strip().splitlines()
        assert len(lines) == 2, lines
        assert lines[0].startswith(f"WARNING: probable-pitcher cache {cache}: dropped 1 invalid entry")
        assert lines[1].startswith(f"WARNING: could not preserve probable-pitcher cache evidence at {dest} (")
        st = json.loads(next((w_occ / "data/health_state/probable_pitcher_cache").glob("*.json")).read_text())
        assert st["preserved"] is None
    print("PASS: Exact original bytes, full-sha256 regular-file reuse, and occupied/special evidence failures verified.")


def test_a4_identity_and_coalescing_and_limits(root: Path) -> None:
    print("=== 2(b). Independent A4 identity verification, coalescing, occupied status, 64 KiB limit ===")
    # Rich multi-outcome cache: non-ASCII path character to test ensure_ascii=True, >10 games, all 4 outcome buckets
    w = root / "café_run"
    cache = w / "data/models/probable_pitcher_lookup.json"
    raw = w / "data/raw"
    (raw / "2025").mkdir(parents=True)
    cache.parent.mkdir(parents=True)
    # 100: 2 bad aliases + 1 valid alias -> [100, 2, 0, 0, 0, VALID]
    # 101: bad -> rescanned -> [101, 0, 1, 0, 0, VALID2]
    # 102: bad -> unusable feed -> [102, 0, 0, 1, 0, None]
    # 103..113 (11 games): bad -> without feed -> [g, 0, 0, 0, 1, None]
    raw_cache_str = (
        '{"00100": null, "0100": 5, "100": ' + json.dumps(VALID) + ', '
        '"101": false, "102": {"away": 1}, '
        + ", ".join(f'"{g}": []' for g in range(113, 102, -1))
        + "}"
    )
    cache.write_text(raw_cache_str)
    orig_bytes = cache.read_bytes()
    orig_sha = hashlib.sha256(orig_bytes).hexdigest()
    (raw / "2025/101.json").write_text(feed_json(VALID2))
    (raw / "2025/102.json").write_text("{corrupt")

    cp = run_child_build(cache, raw, "2027-01-01T04:59:00+00:00")
    assert cp.returncode == 0, cp.stderr
    rec_path = next((w / "data/health_state/probable_pitcher_cache").glob("*.json"))
    rec = json.loads(rec_path.read_text())

    expected_preserved = f"probable_pitcher_lookup.20270101T045900Z.{orig_sha[:12]}.json"
    expected_outcomes = [
        [100, 2, 0, 0, 0, VALID],
        [101, 0, 1, 0, 0, VALID2],
        [102, 0, 0, 1, 0, None],
        *[[g, 0, 0, 0, 1, None] for g in range(103, 114)],
    ]
    expected_id = independent_a4_identity(
        reason="invalid_entries",
        cache_path=str(cache),
        dropped=15,
        recovered_from_cache=2,
        rescanned=1,
        unusable_feed=1,
        without_feed=11,
        original_sha256=orig_sha,
        preserved=expected_preserved,
        outcomes=expected_outcomes,
    )
    assert rec["identity"] == expected_id, (rec["identity"], expected_id)
    assert rec_path.name == f"2026-12-31.invalid_entries.{expected_id[:12]}.json"
    assert rec["game_ids"] == list(range(100, 110))

    # Also verify A4 identity for unreadable_cache
    w_unr = root / "unreadable_a4"
    cache_u = w_unr / "data/models/probable_pitcher_lookup.json"
    raw_u = w_unr / "data/raw"
    (raw_u / "2025").mkdir(parents=True)
    cache_u.parent.mkdir(parents=True)
    cache_u.write_bytes(b"not json")
    u_sha = hashlib.sha256(b"not json").hexdigest()
    (raw_u / "2025/205.json").write_text(feed_json(VALID2))
    (raw_u / "2025/101.json").write_text(feed_json(VALID))
    cp_u = run_child_build(cache_u, raw_u, "2027-01-01T04:59:00+00:00")
    assert cp_u.returncode == 0, cp_u.stderr
    rec_u = json.loads(next((w_unr / "data/health_state/probable_pitcher_cache").glob("*.json")).read_text())
    exp_u_id = independent_a4_identity(
        reason="unreadable_cache",
        cache_path=str(cache_u),
        dropped=None,
        recovered_from_cache=None,
        rescanned=2,
        unusable_feed=None,
        without_feed=None,
        original_sha256=u_sha,
        preserved=f"probable_pitcher_lookup.20270101T045900Z.{u_sha[:12]}.json",
        outcomes=[[101, VALID], [205, VALID2]],
    )
    assert rec_u["identity"] == exp_u_id

    # Coalescing vs conflicting/malformed/special occupied status record
    # Restore original cache bytes and test:
    # (1) valid identical record with earlier timestamp -> coalesces, keeps earlier timestamp, allows cache rewrite!
    cache.write_bytes(orig_bytes)
    cp2 = run_child_build(cache, raw, "2027-01-01T04:59:45+00:00")
    assert cp2.returncode == 0 and len(cp2.stderr.splitlines()) == 1
    assert json.loads(rec_path.read_text())["recorded_at"] == "2027-01-01T04:59:00+00:00"
    assert "101" in json.loads(cache.read_text())

    # (2) conflicting valid record (same 12-hex prefix, different full identity!), malformed JSON, oversized >64KiB, symlink, FIFO, dir
    conflict_rec = dict(rec)
    conflict_rec["identity"] = expected_id[:12] + ("0" if expected_id[12] != "0" else "1") + expected_id[13:]
    for bad_kind in ("conflict_full_id", "conflict_count", "malformed", "oversized", "symlink", "fifo", "dir"):
        cache.write_bytes(orig_bytes)
        if rec_path.exists() or rec_path.is_symlink():
            if rec_path.is_dir() and not rec_path.is_symlink():
                rec_path.rmdir()
            else:
                rec_path.unlink()
        if bad_kind == "conflict_full_id":
            rec_path.write_text(json.dumps(conflict_rec))
        elif bad_kind == "conflict_count":
            alt = dict(rec, rescanned=2, without_feed=10)
            rec_path.write_text(json.dumps(alt))
        elif bad_kind == "malformed":
            rec_path.write_text("{bad")
        elif bad_kind == "oversized":
            rec_path.write_bytes(b" " * (MAX_BYTES + 1))
        elif bad_kind == "symlink":
            other = w / "valid_target.json"
            other.write_text(json.dumps(rec))
            rec_path.symlink_to(other)
        elif bad_kind == "fifo":
            os.mkfifo(rec_path)
        elif bad_kind == "dir":
            rec_path.mkdir()

        cp_bad = run_child_build(cache, raw, "2027-01-01T04:59:50+00:00")
        assert cp_bad.returncode == 0, (bad_kind, cp_bad.stderr)
        assert cache.read_bytes() == orig_bytes, f"cache must NOT be rewritten on status failure ({bad_kind})"
        err_lines = cp_bad.stderr.strip().splitlines()
        assert len(err_lines) == 2, (bad_kind, err_lines)
        assert err_lines[1].startswith(f"WARNING: could not write probable-pitcher cache status at {rec_path} (")
        if rec_path.is_dir() and not rec_path.is_symlink():
            rec_path.rmdir()
        else:
            rec_path.unlink()

    # (3) Producer 64 KiB limit check in publish()
    huge_row = dict(rec, cache_path="x" * (MAX_BYTES + 10))
    status_dest = w / "data/health_state/probable_pitcher_cache" / f"2026-12-31.invalid_entries.{expected_id[:12]}.json"
    try:
        ppc.publish(status_dest, huge_row)
    except ValueError as exc:
        assert "size limit" in str(exc)
    else:
        raise AssertionError("publish() failed to enforce 64 KiB limit")
    assert not status_dest.exists()
    print("PASS: A4 identity (invalid & unreadable, non-ASCII path), coalescing, conflicting/special status failure, and 64 KiB limit verified.")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="probe2_") as tmp:
        root = Path(tmp)
        test_exact_bytes_and_reuse(root)
        test_a4_identity_and_coalescing_and_limits(root)
    print("ALL PROBE 2 CHECKS PASSED")


if __name__ == "__main__":
    main()
