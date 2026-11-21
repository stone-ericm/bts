import contextlib
import hashlib
import io
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import bts
import probe_health
import probe_stderr


def producer_identity_and_failure():
    source = probe_health.HERE.parents[1] / "src"
    work = Path.cwd()
    cache = work / probe_stderr.QUOTED / "data/models/probable_pitcher_lookup.json"
    cache.parent.mkdir(parents=True)
    original = b'{"100":null}'
    cache.write_bytes(original)
    evidence = cache.parent / "rejected"
    evidence.write_bytes(b"obstruction")
    status = cache.parent.parent / "health_state/probable_pitcher_cache"
    status.parent.mkdir()
    status.write_bytes(b"obstruction")
    feed = work / "data/raw/2026/100.json"
    feed.parent.mkdir(parents=True)
    feed.write_text("{}")
    output = probe_stderr.invoke(source, work, "failed-evidence-and-status", "utf-8")
    lines = output.decode().splitlines()
    shown = probe_stderr.QUOTED.replace(chr(92), chr(92) * 2).replace(chr(96), chr(92) + "x60")
    assert len(lines) == 3 and all(shown in line and chr(96) not in line for line in lines)
    assert "could not preserve" in lines[1] and "could not write" in lines[2], lines
    assert cache.read_bytes() == original
    print("All three displayed producer paths escaped on real UTF-8 stderr:", repr(lines))
    evidence.unlink()
    status.unlink()
    raw_cache_path = str(cache.relative_to(work))
    probe_health.Clock.at = datetime.fromisoformat(probe_health.AT)
    with patch.object(probe_health.producer, "datetime", probe_health.Clock):
        with contextlib.redirect_stderr(io.StringIO()):
            lookup = probe_health.producer.build(Path(raw_cache_path), Path("data/raw"))
    record, = status.iterdir()
    row = json.loads(record.read_bytes())
    assert row["cache_path"] == raw_cache_path and (evidence / row["preserved"]).read_bytes() == original
    fields = ("reason", "cache_path", "original_sha256", "preserved",
              "dropped", "recovered_from_cache", "rescanned", "unusable_feed", "without_feed")
    payload = {field: row[field] for field in fields}
    payload["outcomes"] = [[100, 0, 1, 0, 0, lookup[100]]]
    digest = hashlib.sha256(json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert digest == row["identity"] and record.name.endswith(f".{digest[:12]}.json")
    (status / f"{probe_health.DAY}.bad{chr(96)}.json").write_text("{")
    with probe_health.captured() as (sent, rejected, logs):
        result, = probe_health.run(picks=(cache.parent.parent / "picks").relative_to(work))
    assert result.incident_key == probe_health.summary_key(probe_health.DAY, digest, f"{probe_health.DAY}.bad{chr(96)}.json")
    assert len(sent) == 1 and not rejected and chr(96) not in sent[0] + logs.getvalue()
    print("Raw cache_path, producer SHA-256, evidence file and raw basename in summary key", digest, result.incident_key)


def prior_journal():
    capture = probe_health.captured

    @contextlib.contextmanager
    def observe(*args, **kwargs):
        with capture(*args, **kwargs) as result:
            try:
                yield result
            finally:
                sent, blocked, _ = result
                print("Round 2 journal transport accepted", list(map(len, sent)), "rejected", list(map(len, blocked)))

    probe_health.captured = observe
    try:
        probe_health.journal_feedback(True)
    except AssertionError:
        receipt = json.loads(Path("replay-receipt.json").read_bytes())
        assert receipt["status"] == "sent" and receipt["critical_count"] == 1, receipt
        print("Round 2 sent the same false shadow CRITICAL to the transport below the cap")


def main():
    source = Path(sys.argv[1]).resolve()
    assert Path(bts.__file__).resolve() == source / "bts/__init__.py"
    probe = prior_journal if source == probe_health.HERE / "r2/src" else producer_identity_and_failure
    before = Path.cwd()
    with TemporaryDirectory(dir=probe_health.HERE, prefix="followup-") as work:
        os.chdir(work)
        try:
            probe()
        finally:
            os.chdir(before)


if __name__ == "__main__":
    main()
