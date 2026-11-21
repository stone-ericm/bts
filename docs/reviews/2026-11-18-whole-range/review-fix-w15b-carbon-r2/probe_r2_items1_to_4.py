"""Round-2 Probe for Items 1–4: Replay Codex r1 findings (probe_journal.py & timestamp/controls), hostile W15 text sweep, genuine journal evidence, and byte-identity vs 284dc97a."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import logging
import os
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from io import BytesIO, StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError

HERE = Path(__file__).resolve().parent
REPO_SRC = HERE.parents[2] / "src"
R1_W15B_SRC = HERE / "archives/w15b_r1/src"
BASE_SRC = HERE.parent / "archives/base/src"

TICK = chr(96)


CHILD = r"""
import contextlib, hashlib, io, json, logging, os, sys
from datetime import date, datetime, timedelta, timezone
from io import BytesIO, StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError

expected_src = Path(sys.argv[1]).resolve()
mode = sys.argv[2]
work = Path(sys.argv[3]).resolve()
os.chdir(work)

import bts
assert Path(bts.__file__).resolve() == expected_src / "bts/__init__.py", (bts.__file__, expected_src)

from bts import dm
from bts.daily_decision import DECISION_SCHEMA
from bts.features import compute, probable_pitcher_cache as producer
from bts.health import alert, analytics_artifacts_missing, attention, probable_pitcher_cache as health, runner

SOURCE = "probable_pitcher_cache"
TICK = chr(96)

class DynClock(datetime):
    at = datetime(2026, 7, 22, 23, 40, tzinfo=timezone(timedelta(hours=-4)))
    @classmethod
    def now(cls, tz=None):
        return cls.at.astimezone(tz) if tz is not None else cls.at.replace(tzinfo=None)

@contextlib.contextmanager
def captured_env(sources=(SOURCE,)):
    sent, rejected = [], []
    logs = StringIO()
    handler = logging.StreamHandler(logs)
    root = logging.getLogger()
    old_level = root.level
    root.setLevel(logging.WARNING)
    root.addHandler(handler)
    safe_run = runner._safe_run

    def api(request, **kwargs):
        url = request.full_url
        if url.endswith("createSession"):
            return BytesIO(b'{"accessJwt":"offline"}')
        if "getConvoForMembers" in url:
            return BytesIO(b'{"convo":{"id":"offline"}}')
        assert url.endswith("chat.bsky.convo.sendMessage")
        assert kwargs == dict(timeout=15, idempotent=False)
        body = json.loads(request.data)["message"]["text"]
        assert body.isascii() and "\r" not in body, repr(body)
        if len(body) > 1000:
            rejected.append(body)
            raise HTTPError(url, 400, "synthetic 1000-grapheme cap", {}, None)
        sent.append(body)
        return BytesIO(b'{"id":"offline"}')

    try:
        with (
            patch.object(runner, "_safe_run", lambda name, call: safe_run(name, call) if name in sources else []),
            patch.object(runner.realized_calibration, "_current_deploy_iso", return_value=None),
            patch.object(runner, "datetime", DynClock),
            patch.object(alert, "datetime", DynClock),
            patch.object(dm, "get_dm_password", return_value="offline"),
            patch.object(dm, "retry_urlopen", api),
        ):
            yield sent, rejected, logs
    finally:
        root.removeHandler(handler)
        root.setLevel(old_level)

if mode == "journal_replay":
    day = date(2026, 7, 22)
    picks = Path("data/picks")
    status_dir = Path("data/health_state") / SOURCE
    status_dir.mkdir(parents=True)
    (status_dir / f"{day}.oom-kill.json").write_bytes(b"{")
    # Also include a prefix-spoofed entry and a valid record under an oom cache_path!
    (status_dir / f"{day}.invalid_entries.{'a'*12}.json.oom-kill.json").write_bytes(b"{")

    state_dir = picks / str(day)
    state_dir.mkdir(parents=True)
    (state_dir / "decision.json").write_text(json.dumps(dict(
        schema_version=DECISION_SCHEMA, date=str(day), action="single",
        source="mdp", delivery_status="delivered", scoreable=True,
    )))
    (state_dir / "scheduler_state.json").write_text(json.dumps(dict(
        date=str(day), pick_locked=True, analytics_jobs={
            "shadow": {"status": "dispatched", "dispatched_at": "2026-07-22T21:10:00-04:00"},
        },
    )))
    # Note: probe_journal.py had only 2026-07-22.oom-kill.json first; let's match probe_journal.py EXACTLY first!
    (status_dir / f"{day}.invalid_entries.{'a'*12}.json.oom-kill.json").unlink()

    def run_step(journal_stdout, wall_iso, receipt_path):
        def fake_journal(command, **options):
            assert command[:4] == ["journalctl", "--user", "-u", "bts-scheduler.service"]
            assert command[command.index("--since") + 1] == "2026-07-22 21:10:00"
            assert options["timeout"] == 5
            return SimpleNamespace(returncode=0, stdout=journal_stdout)

        DynClock.at = datetime.fromisoformat(wall_iso)
        with captured_env(sources=(SOURCE, analytics_artifacts_missing.SOURCE)) as (sent, rejected, logs):
            with patch.object(analytics_artifacts_missing.subprocess, "run", fake_journal):
                alerts = runner.run_all_checks(
                    picks, picks.parent / "models", "did:plc:fake", today=day,
                    shadow_model_enabled=True, shadow_unit=None,
                    thresholds_overrides={"health_dm_delivery_status": receipt_path},
                )
        rec = json.loads(receipt_path.read_text())
        return {
            "alerts": [(a.level, a.source, a.message) for a in alerts],
            "sent_lens": list(map(len, sent)),
            "rejected_lens": list(map(len, rejected)),
            "receipt_status": rec["status"],
            "logs": logs.getvalue(),
        }

    receipt = Path("health-receipt.json")
    first = run_step("", "2026-07-22T23:40:00-04:00", receipt)
    w15_lines = [ln for ln in first["logs"].splitlines() if "Probable-pitcher cache" in ln]
    assert len(w15_lines) >= 1, first["logs"]
    journal_feed = "\n".join(f"Jul 22 23:40:00 synthetic bts[1]: {ln}" for ln in first["logs"].splitlines()) + "\n"
    second = run_step(journal_feed, "2026-07-22T23:50:00-04:00", receipt)

    # Also test genuine fatal controls placed AFTER all W15 journal lines
    genuine_controls = {}
    for g_name, g_line in [
        ("systemd_oom_kill", "Jul 22 23:30:00 synthetic systemd[2]: bts-scheduler.service: Failed with result 'oom-kill'."),
        ("systemd_oom_killer", "Jul 22 23:30:00 synthetic systemd[2]: bts-scheduler.service: A process of this unit has been killed by the OOM killer."),
        ("kernel_oom", "Jul 22 23:30:00 synthetic kernel: Out of memory: Killed process 444 (python)"),
        ("sigkill_9", "Jul 22 23:30:00 synthetic systemd[2]: bts-scheduler.service: Main process exited, code=killed, status=9/KILL"),
        ("status_137", "Jul 22 23:30:00 synthetic systemd[2]: bts-scheduler.service: Main process exited, code=exited, status=137/n/a"),
    ]:
        genuine_controls[g_name] = run_step(journal_feed + g_line + "\n", "2026-07-22T23:50:00-04:00", Path(f"rec-{g_name}.json"))

    print(json.dumps({"first": first, "second": second, "genuine_controls": genuine_controls}))

elif mode == "timestamp_and_proxy":
    day = date(2026, 12, 31)
    status_dir = Path("data/health_state") / SOURCE
    status_dir.mkdir(parents=True)
    identity = "a" * 64
    rec_path = status_dir / f"{day}.invalid_entries.{identity[:12]}.json"
    raw_ts = f"2027-01-01{TICK}04:59:00+00:00"
    raw_cache = f"vendor\nnext/oom{TICK}dir/models/cache.json"
    rec_path.write_text(json.dumps(dict(
        schema_version="bts_probable_pitcher_cache_status_v1",
        reason="invalid_entries",
        recorded_at=raw_ts,
        recorded_date_et=str(day),
        cache_path=raw_cache,
        dropped=1, recovered_from_cache=0, rescanned=0, unusable_feed=0, without_feed=1,
        game_ids=[100], identity=identity,
        preserved="probable_pitcher_lookup.20261231T140000Z.bbbbbbbbbbbb.json",
        original_sha256="b" * 64,
    )))
    DynClock.at = datetime(2027, 1, 1, 5, 1, tzinfo=timezone.utc)
    with captured_env() as (sent, rejected, logs):
        alerts = runner.run_all_checks(Path("data/picks"), Path("data/models"), "did:plc:fake", today=day)

    # StderrProxy without .errors attribute
    class StderrProxy:
        encoding = "utf-8"
        def __init__(self, stream):
            self.stream = stream
        def fileno(self):
            return self.stream.fileno()
        def flush(self):
            self.stream.flush()

    proxy_file = Path("proxy-stderr.bin")
    with proxy_file.open("wb") as handle:
        with contextlib.redirect_stderr(StderrProxy(handle)):
            producer.warn("unchanged ASCII wording")

    print(json.dumps({
        "logs": logs.getvalue(),
        "sent": sent,
        "incident_key": alerts[0].incident_key,
        "disk_recorded_at": json.loads(rec_path.read_text())["recorded_at"],
        "disk_cache_path": json.loads(rec_path.read_text())["cache_path"],
        "proxy_bytes": proxy_file.read_bytes().decode("utf-8"),
    }))

elif mode == "byte_identity":
    # Run rich producer + health scenarios and capture exact bytes of all files and all incident_keys
    DynClock.at = datetime(2026, 12, 31, 20, 0, tzinfo=timezone.utc)
    producer.datetime = DynClock
    out = {}

    # 1. Invalid entries with rewrite & evidence under custom hostile path
    root = Path(f"hostile_oom_{TICK}_root/data")
    (root / "models").mkdir(parents=True)
    (root / "raw/2025").mkdir(parents=True)
    (root / "raw/2025/100.json").write_text(json.dumps({
        "gameData": {
            "probablePitchers": {"away": {"id": 2}, "home": {"id": 23}},
            "teams": {"away": {"id": 61}, "home": {"id": 62}},
        }
    }))
    cache = root / "models/probable_pitcher_lookup.json"
    cache.write_bytes(b'{"100": null, "200": {"away": 1, "home": 22, "away_tid": 51, "home_tid": 52}}')
    compute._probable_pitcher_cache_path = cache
    with contextlib.redirect_stderr(io.StringIO()):
        lookup1 = compute._build_probable_pitcher_lookup(str(root / "raw"))

    # 2. Unreadable cache without feeds (no rewrite, second status record)
    cache2 = root / "models/probable_pitcher_lookup.json"
    rewritten_cache_bytes = cache2.read_bytes()
    cache2.write_bytes(b"{corrupt")
    with contextlib.redirect_stderr(io.StringIO()):
        lookup2 = compute._build_probable_pitcher_lookup(str(root / "missing_raw"))

    st_dir = root / "health_state" / SOURCE
    ev_dir = root / "models/rejected"
    # Add both a valid-named corrupt record and a badly-named corrupt record
    (st_dir / f"2026-12-30.invalid_entries.{'c'*12}.json").write_bytes(b"{")
    (st_dir / f"2026-12-31.oom{TICK}bad.json").write_bytes(b"{")

    alerts = health.check(st_dir, date(2026, 12, 31))

    # Unreadable FIFO status dir
    fifo_dir = Path(f"fifo_{TICK}_dir")
    os.mkfifo(fifo_dir)
    fifo_alerts = health.check(fifo_dir, date(2026, 12, 31))

    out["rewritten_cache_sha"] = hashlib.sha256(rewritten_cache_bytes).hexdigest()
    out["evidence_files"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ev_dir.iterdir())}
    out["status_files"] = {p.name: p.read_bytes().decode("utf-8") for p in sorted(st_dir.iterdir())}
    out["health_alert"] = (alerts[0].level, alerts[0].source, alerts[0].message, alerts[0].incident_key)
    out["fifo_alert"] = (fifo_alerts[0].level, fifo_alerts[0].source, fifo_alerts[0].message, fifo_alerts[0].incident_key)
    print(json.dumps(out))
"""


def run_child(src_dir: Path, mode: str) -> dict:
    with TemporaryDirectory(dir=HERE, prefix=f"tmp-{mode}-") as tmp:
        script = Path(tmp) / "child.py"
        script.write_text(CHILD, encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, str(script), str(src_dir), mode, tmp],
            env={**os.environ, "PYTHONPATH": str(src_dir), "PYTHONDONTWRITEBYTECODE": "1"},
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        return json.loads(proc.stdout)


def check_item1_and_2():
    print("=== 1A. Replay Codex probe_journal.py Sequence (284dc97a vs 0ce9e5ee) ===")
    old_j = run_child(R1_W15B_SRC, "journal_replay")
    new_j = run_child(REPO_SRC, "journal_replay")

    # First EOD: both send 741-char WARN digest, 0 CRITICALs
    assert old_j["first"]["sent_lens"] == [741] and new_j["first"]["sent_lens"] == [741]
    assert [lvl for lvl, _, _ in old_j["first"]["alerts"]] == ["WARN", "WARN"]
    assert [lvl for lvl, _, _ in new_j["first"]["alerts"]] == ["WARN", "WARN"]

    # Second EOD (restart reading journal from First EOD):
    # 284dc97a: false CRITICAL, 1009-grapheme rejection, receipt failed!
    assert [lvl for lvl, _, _ in old_j["second"]["alerts"]] == ["CRITICAL", "WARN"]
    assert old_j["second"]["sent_lens"] == [] and old_j["second"]["rejected_lens"] == [1009]
    assert old_j["second"]["receipt_status"] == "failed"

    # 0ce9e5ee: 0 CRITICALs, covered by receipt (0 rejections, receipt stays 'sent')!
    assert [lvl for lvl, _, _ in new_j["second"]["alerts"]] == ["WARN", "WARN"]
    assert new_j["second"]["rejected_lens"] == [] and new_j["second"]["receipt_status"] == "sent"
    print(
        f"  first-eod : 284dc97a sent={old_j['first']['sent_lens']} | 0ce9e5ee sent={new_j['first']['sent_lens']}\n"
        f"  eod-restart: 284dc97a alerts={[x[:2] for x in old_j['second']['alerts']]} rejected={old_j['second']['rejected_lens']} receipt={old_j['second']['receipt_status']} | "
        f"0ce9e5ee alerts={[x[:2] for x in new_j['second']['alerts']]} rejected={new_j['second']['rejected_lens']} receipt={new_j['second']['receipt_status']}"
    )

    # All 5 genuine fatal journal controls still fire beside W15 journal lines on 0ce9e5ee!
    for g_name, res in new_j["genuine_controls"].items():
        crits = [msg for lvl, src, msg in res["alerts"] if lvl == "CRITICAL" and src == "analytics_artifacts_missing"]
        assert len(crits) == 1 and "Probable-pitcher cache" not in crits[0], (g_name, res)
        assert len(res["sent_lens"]) == 1 and res["sent_lens"][0] < 1000 and res["receipt_status"] == "sent"
        print(f"  genuine control '{g_name}': CRITICAL fired & sent ({res['sent_lens'][0]} chars < 1000)")

    print("\n=== 1B. Replay Codex Finding 2 (Timestamp Backtick & Path Line Break) + StderrProxy ===")
    old_t = run_child(R1_W15B_SRC, "timestamp_and_proxy")
    new_t = run_child(REPO_SRC, "timestamp_and_proxy")

    assert TICK in old_t["logs"] and "vendor\nnext" in old_t["logs"] and "oom" in old_t["logs"].lower()
    assert old_t["proxy_bytes"] == ""

    assert TICK not in new_t["logs"] and "vendor" not in new_t["logs"] and "oom" not in new_t["logs"].lower()
    assert "Probable-pitcher cache at 2027-01-01T04:59:00+00:00 had 1 invalid entry" in new_t["logs"]
    assert "That run recorded an evidence copy named probable_pitcher_lookup.20261231T140000Z.bbbbbbbbbbbb.json in the rejected folder beside the cache." in new_t["logs"]
    assert new_t["incident_key"] == old_t["incident_key"]
    assert new_t["disk_recorded_at"] == f"2027-01-01{TICK}04:59:00+00:00"
    assert new_t["proxy_bytes"] == "WARNING: unchanged ASCII wording\n"
    print("  284dc97a leaked backtick, newline, and 'oom' into log and dropped StderrProxy output;")
    print("  0ce9e5ee normalized timestamp to 2027-01-01T04:59:00+00:00, omitted disk path, preserved raw record & key, and wrote to StderrProxy.")

    print("\n=== 2. Exhaustive Adversarial Sweep of Every W15 Text (Logged, Stderr, Sent) ===")
    sys.path.insert(0, str(REPO_SRC))
    from bts.features import compute, probable_pitcher_cache as producer
    from bts.health import analytics_artifacts_missing, attention, probable_pitcher_cache as health

    all_w15_lines = []
    with TemporaryDirectory(dir=HERE, prefix="tmp-sweep-") as tmp:
        # Hostile path containing oom, out of memory, room, backtick, CR, LF, TAB, ESC, DEL
        hostile_dir = Path(tmp) / f"oom_out of memory_room_{TICK}_\n_\r_\t_\x1b_\x7f/data"
        (hostile_dir / "models").mkdir(parents=True)
        (hostile_dir / "raw/2025").mkdir(parents=True)
        (hostile_dir / "raw/2025/100.json").write_text(json.dumps({
            "gameData": {
                "probablePitchers": {"away": {"id": 2}, "home": {"id": 23}},
                "teams": {"away": {"id": 61}, "home": {"id": 62}},
            }
        }))
        c = hostile_dir / "models/probable_pitcher_lookup.json"
        c.write_text('{"100": null}')
        compute._probable_pitcher_cache_path = c

        # 1. Producer dropped + preserve fail + status fail
        (hostile_dir / "models/rejected").write_text("block")
        (hostile_dir / "health_state").mkdir(parents=True)
        (hostile_dir / "health_state/probable_pitcher_cache").write_text("block")
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            compute._build_probable_pitcher_lookup(str(hostile_dir / "raw"))
        prod_lines = buf.getvalue().splitlines()
        assert len(prod_lines) == 3, prod_lines
        all_w15_lines.extend(prod_lines)

        # 2. Clean up blockers, produce real valid records (both invalid_entries w/ copy and unreadable_cache w/o copy)
        (hostile_dir / "models/rejected").unlink()
        (hostile_dir / "health_state/probable_pitcher_cache").unlink()
        with contextlib.redirect_stderr(io.StringIO()):
            compute._build_probable_pitcher_lookup(str(hostile_dir / "raw"))
        c.write_text("{")
        with contextlib.redirect_stderr(io.StringIO()):
            compute._build_probable_pitcher_lookup(str(hostile_dir / "missing_raw"))

        st_dir = hostile_dir / "health_state/probable_pitcher_cache"
        today_et = datetime.now(timezone.utc).astimezone(health.read_record.__globals__["ET"]).date()
        # Add hostile status entries: prefix-spoofed record name, control chars, OOM words, valid-pattern corrupt name
        for bad_suffix in [
            f"invalid_entries.{'a'*12}.json.oom-kill",
            "oom-kill",
            "out of memory",
            "room",
            f"bt{TICK}x",
            "line\nbreak",
            "cr\rbreak",
            "tab\tbreak",
            f"invalid_entries.{'f'*12}",
        ]:
            (st_dir / f"{today_et}.{bad_suffix}.json").write_bytes(b"{")
        # Also tamper recorded_at separator in existing valid record to newline
        valid_rec = next(p for p in st_dir.iterdir() if ".unreadable_cache." in p.name)
        rj = json.loads(valid_rec.read_text())
        rj["recorded_at"] = rj["recorded_at"].replace("T", "\n", 1)
        valid_rec.write_text(json.dumps(rj))

        logs = StringIO()
        h = logging.StreamHandler(logs)
        logger = logging.getLogger(health.__name__)
        logger.addHandler(h)
        try:
            alerts = health.check(st_dir, today_et)
            # Also unreadable status folder
            fifo_st = Path(tmp) / f"oom_{TICK}_\n_fifo"
            os.mkfifo(fifo_st)
            alerts_dir = health.check(fifo_st, today_et)
        finally:
            logger.removeHandler(h)

        # Split ONLY on the handler's trailing '\n' per record by checking individual record messages too!
        all_w15_lines.extend(logs.getvalue().splitlines())
        all_w15_lines.append(alerts[0].message)
        all_w15_lines.append(alerts_dir[0].message)
        all_w15_lines.append(f"[{alerts[0].source} {alerts[0].level}] {alerts[0].message}")

    for ln in all_w15_lines:
        low = ln.lower()
        assert "oom" not in low and "out of memory" not in low, ln
        assert not attention.OOM_EVIDENCE_RE.search(f"probable_pitcher_cache {low}"), ln
        assert TICK not in ln, ln
        assert "\n" not in ln and "\r" not in ln and "\t" not in ln, repr(ln)
        assert all(ch.isprintable() and ord(ch) >= 32 for ch in ln), repr(ln)
        assert "hostile" not in low and "tmp-sweep" not in low, ln

    # Verify real _fatal_scheduler_journal_line returns None on all W15 lines together
    joined_journal = "\n".join(all_w15_lines) + "\n"
    with patch.object(
        analytics_artifacts_missing.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(returncode=0, stdout=joined_journal, stderr=""),
    ):
        assert analytics_artifacts_missing._fatal_scheduler_journal_line("bts-scheduler.service", "2026-12-31T00:00:00+00:00") is None
    print(f"  Verified {len(all_w15_lines)} rendered W15 lines under hostile disk/path/timestamp inputs: 0 disk text, 0 OOM matches, 0 backticks, 0 control/newline chars.")


def check_item3_byte_identity():
    print("\n=== 3. Byte-Identity of Cache, Evidence, Records, Identities, and incident_keys (284dc97a vs 0ce9e5ee) ===")
    old_b = run_child(R1_W15B_SRC, "byte_identity")
    new_b = run_child(REPO_SRC, "byte_identity")
    assert old_b == new_b, (old_b, new_b)
    print("  100% byte-identical across 284dc97a and 0ce9e5ee:")
    print(f"    rewritten_cache_sha: {new_b['rewritten_cache_sha'][:16]}...")
    print(f"    evidence_files     : {list(new_b['evidence_files'].keys())}")
    print(f"    status_files       : {list(new_b['status_files'].keys())}")
    print(f"    health_incident_key: {new_b['health_alert'][3]}")
    print(f"    fifo_incident_key  : {new_b['fifo_alert'][3]}")


if __name__ == "__main__":
    check_item1_and_2()
    check_item3_byte_identity()
    print("\nALL ROUND-2 ITEMS 1-3 DIRECT PROBES COMPLETED.")
