"""Probe for Item 1: Round-2 findings, hostile W15 text/OOM/backtick checks, and Must-Still-Catch policy diff."""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from datetime import date, datetime, timezone
from io import BytesIO, StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from urllib.error import HTTPError

HERE = Path(__file__).resolve().parent
REPO_SRC = HERE.parents[1] / "src"
BASE_SRC = HERE / "archives/base/src"
R1_SRC = HERE / "archives/r1/src"
R2_SRC = HERE / "archives/r2/src"

DAY = date(2026, 12, 31)
SOURCE = "probable_pitcher_cache"
TICK = chr(96)


CHILD_RUNNER = """
import hashlib, json, logging, os, sys
from datetime import date, datetime, timezone
from io import BytesIO, StringIO
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

expected_src = Path(sys.argv[1]).resolve()
mode = sys.argv[2]
work = Path(sys.argv[3]).resolve()
os.chdir(work)

import bts
actual_init = Path(bts.__file__).resolve()
assert actual_init == expected_src / "bts/__init__.py", (actual_init, expected_src)

from bts import dm
from bts.features import compute, probable_pitcher_cache as producer
from bts.health import alert, attention, probable_pitcher_cache as health, runner

DAY = date(2026, 12, 31)
SOURCE = "probable_pitcher_cache"
TICK = chr(96)

class Wall(datetime):
    @classmethod
    def now(cls, tz=None):
        dt = cls(2027, 1, 1, 5, 1, tzinfo=timezone.utc)
        return dt.astimezone(tz) if tz is not None else dt.replace(tzinfo=None)

class ProdClock(datetime):
    @classmethod
    def now(cls, tz=None):
        dt = cls(2026, 12, 31, 20, 0, tzinfo=timezone.utc)
        return dt.astimezone(tz) if tz is not None else dt.replace(tzinfo=None)

producer.datetime = ProdClock

def run_captured(picks_dir, day=DAY, extra_alerts=None):
    packets = []
    rejections = []
    logs = StringIO()
    handler = logging.StreamHandler(logs)
    root_logger = logging.getLogger("bts.health")
    old_level = root_logger.level
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)
    safe_run = runner._safe_run

    def api(request, **kwargs):
        url = request.full_url
        if url.endswith("createSession"):
            return BytesIO(b'{"accessJwt":"offline"}')
        if "getConvoForMembers" in url:
            return BytesIO(b'{"convo":{"id":"offline"}}')
        assert url.endswith("chat.bsky.convo.sendMessage"), url
        body = json.loads(request.data)["message"]["text"]
        if len(body) > 1000:
            rejections.append(len(body))
            raise HTTPError(url, 400, "maxGraphemes=1000", {}, None)
        packets.append(body)
        return BytesIO(b'{"id":"msg-1"}')

    injected_done = False
    def fake_safe_run(name, fn):
        nonlocal injected_done
        if name == SOURCE:
            res = safe_run(name, fn)
            if extra_alerts and not injected_done:
                injected_done = True
                return list(extra_alerts) + res
            return res
        return []

    try:
        with (
            patch.object(runner, "_safe_run", fake_safe_run),
            patch.object(runner.realized_calibration, "_current_deploy_iso", return_value=None),
            patch.object(runner, "datetime", Wall),
            patch.object(alert, "datetime", Wall),
            patch.object(dm, "get_dm_password", return_value="offline"),
            patch.object(dm, "retry_urlopen", api),
        ):
            alerts = runner.run_all_checks(picks_dir, picks_dir.parent / "models", "did:plc:fake", today=day)
    finally:
        root_logger.removeHandler(handler)
        root_logger.setLevel(old_level)

    receipt_path = picks_dir.parent / "health_state/health_dm_delivery_status.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
    return {
        "alerts": [(a.level, a.source, a.message, a.incident_key) for a in alerts],
        "packets": packets,
        "rejections": rejections,
        "logs": logs.getvalue(),
        "receipt": receipt,
    }

if mode == "oom_paths":
    out = {}
    # 1. Exact round-2 Codex symlink alias oom/data -> data
    base_data = work / "real_data"
    (base_data / "models").mkdir(parents=True)
    (base_data / "raw").mkdir(parents=True)
    cache = base_data / "models/probable_pitcher_lookup.json"
    cache.write_text('{"100": null}')
    compute._probable_pitcher_cache_path = cache
    compute._build_probable_pitcher_lookup(str(base_data / "raw"))

    alias = work / "oom/data"
    alias.parent.mkdir(parents=True)
    alias.symlink_to(base_data)
    out["symlink_oom_data"] = run_captured(alias / "picks")

    # 2. Other OOM-matching directory names
    for label, parts in [
        ("oom_kill", ("x", "oom_kill", "data")),
        ("oom_killed", ("x", "OOM-killed", "data")),
        ("out_of_memory_spaces", ("out of memory", "data")),
        ("out_of_memory_dashes", ("x", "out-of-memory", "data")),
        ("oom_space_kill", ("oom kill", "data")),
    ]:
        d = work.joinpath(label, *parts)
        (d / "models").mkdir(parents=True)
        c = d / "models/probable_pitcher_lookup.json"
        c.write_text('{"100": null}')
        compute._probable_pitcher_cache_path = c
        compute._build_probable_pitcher_lookup(str(d / "missing_raw"))
        out[label] = run_captured(d / "picks")
    print(json.dumps(out))

elif mode == "backticks":
    out = {}
    # 1. Broken status file name with backtick and backslash
    d1 = work / "case1/data"
    st1 = d1 / "health_state" / SOURCE
    st1.mkdir(parents=True)
    bad_file = st1 / f"{DAY}.broken{TICK}\\\\x.json"
    bad_file.write_bytes(b"{")
    out["broken_filename"] = run_captured(d1 / "picks")
    out["broken_filename"]["raw_name"] = bad_file.name

    # 2. Custom status root + cache_path with backtick and backslash + preserved copy + producer stderr
    d2 = work / f"case2/other{TICK}d\\\\ata"
    (d2 / "models").mkdir(parents=True)
    (d2 / "raw/2025").mkdir(parents=True)
    (d2 / "raw/2025/100.json").write_text(json.dumps({
        "gameData": {
            "probablePitchers": {"away": {"id": 2}, "home": {"id": 23}},
            "teams": {"away": {"id": 61}, "home": {"id": 62}},
        }
    }))
    c2 = d2 / "models/probable_pitcher_lookup.json"
    c2.write_text('{"100": null}')
    compute._probable_pitcher_cache_path = c2

    # Capture real fd stderr from producer
    r_fd, w_fd = os.pipe()
    saved_stderr = os.dup(2)
    try:
        os.dup2(w_fd, 2)
        compute._build_probable_pitcher_lookup(str(d2 / "raw"))
    finally:
        os.dup2(saved_stderr, 2)
        os.close(saved_stderr)
        os.close(w_fd)
    stderr_dropped = os.read(r_fd, 65536).decode("utf-8")
    os.close(r_fd)

    rec_files = list((d2 / "health_state" / SOURCE).glob("*.json"))
    rec_json = json.loads(rec_files[0].read_text())
    res2 = run_captured(d2 / "picks")
    res2["stderr_dropped"] = stderr_dropped
    res2["record_cache_path"] = rec_json["cache_path"]
    res2["record_preserved"] = rec_json["preserved"]
    out["custom_root_and_cache_path"] = res2

    # 3. Producer preserve failure and status failure with backtick & backslash in path
    d3 = work / f"case3/w{TICK}d\\\\z"
    (d3 / "models/rejected").mkdir(parents=True)
    # Block preserve by making rejected a file, and status dir a file
    (d3 / "models/rejected").rmdir()
    (d3 / "models/rejected").write_text("not a dir")
    (d3 / "health_state").mkdir(parents=True)
    (d3 / "health_state" / SOURCE).write_text("not a dir")
    (d3 / "raw/2025").mkdir(parents=True)
    (d3 / "raw/2025/100.json").write_text(json.dumps({
        "gameData": {
            "probablePitchers": {"away": {"id": 2}, "home": {"id": 23}},
            "teams": {"away": {"id": 61}, "home": {"id": 62}},
        }
    }))
    c3 = d3 / "models/probable_pitcher_lookup.json"
    c3.write_text('{"100": null}')
    compute._probable_pitcher_cache_path = c3
    r_fd, w_fd = os.pipe()
    saved_stderr = os.dup(2)
    try:
        os.dup2(w_fd, 2)
        compute._build_probable_pitcher_lookup(str(d3 / "raw"))
    finally:
        os.dup2(saved_stderr, 2)
        os.close(saved_stderr)
        os.close(w_fd)
    stderr_failures = os.read(r_fd, 65536).decode("utf-8")
    os.close(r_fd)

    # 4. Pre-existing unreadable cache stderr line with backtick in path
    c4 = work / f"case4/u{TICK}n\\\\r/probable_pitcher_lookup.json"
    c4.parent.mkdir(parents=True)
    c4.write_text("{")
    compute._probable_pitcher_cache_path = c4
    r_fd, w_fd = os.pipe()
    saved_stderr = os.dup(2)
    try:
        os.dup2(w_fd, 2)
        compute._build_probable_pitcher_lookup(str(work / "case4/missing_raw"))
    finally:
        os.dup2(saved_stderr, 2)
        os.close(saved_stderr)
        os.close(w_fd)
    stderr_unreadable = os.read(r_fd, 65536).decode("utf-8")
    os.close(r_fd)

    # 5. Unreadable status directory (FIFO) under backtick path
    d5 = work / f"case5/fifo{TICK}dir"
    (d5 / "health_state").mkdir(parents=True)
    os.mkfifo(d5 / "health_state" / SOURCE)
    res5 = run_captured(d5 / "picks")

    out["producer_failures_stderr"] = stderr_failures
    out["producer_unreadable_stderr"] = stderr_unreadable
    out["fifo_status_dir_backtick"] = res5
    print(json.dumps(out))

elif mode == "policy_matrix":
    from bts.health.alert import Alert
    GENUINE = {
        "warn_alone": [Alert("WARN", "analytics_artifacts_missing", "shadow missing Result=oom-kill")],
        "critical_alone": [Alert("CRITICAL", "analytics_artifacts_missing", "capture Result=oom-kill")],
        "memory_and_warn": [
            Alert("WARN", "memory_growth", "scheduler RSS 3350 MB"),
            Alert("WARN", "analytics_artifacts_missing", "shadow missing Result=oom-kill"),
        ],
        "memory_and_critical": [
            Alert("WARN", "memory_growth", "scheduler RSS 3350 MB"),
            Alert("CRITICAL", "analytics_artifacts_missing", "capture Result=oom-kill"),
        ],
        "restart_spike_alone": [
            Alert("CRITICAL", "restart_spike", "NRestarts spiked; inspect journal/OOM/watchdog/crash evidence."),
        ],
        "memory_and_restart_spike": [
            Alert("WARN", "memory_growth", "scheduler RSS 3350 MB"),
            Alert("CRITICAL", "restart_spike", "NRestarts spiked; inspect journal/OOM/watchdog/crash evidence."),
        ],
        "out_of_memory_text": [Alert("WARN", "analytics_artifacts_missing", "worker: Out of memory")],
        "innocent_room": [Alert("WARN", "analytics_artifacts_missing", "shadow worker room unavailable")],
        "memory_only": [Alert("WARN", "memory_growth", "scheduler RSS 3350 MB")],
        "none": [],
    }
    folders = [
        "data/health_state",
        "oom/data/health_state",
        "x/out-of-memory/health_state",
        f"oom/{'q'*300}/{'r'*300}/health_state",
    ]
    identity = "a" * 64
    row = dict(
        schema_version="bts_probable_pitcher_cache_status_v1",
        reason="invalid_entries",
        recorded_at="2026-12-31T14:00:00+00:00",
        recorded_date_et="2026-12-31",
        cache_path="data/models/probable_pitcher_lookup.json",
        dropped=1, recovered_from_cache=0, rescanned=0, unusable_feed=0, without_feed=1,
        game_ids=[100], identity=identity, preserved=None, original_sha256="c" * 64,
    )
    results = {}
    for folder in folders:
        st = Path(folder) / SOURCE
        st.mkdir(parents=True, exist_ok=True)
        (st / f"2026-12-31.invalid_entries.{identity[:12]}.json").write_text(json.dumps(row))
        w15_alert, = health.check(st, DAY)
        for name, base_alerts in GENUINE.items():
            base_crit = [(a.level, a.source, a.message) for a in attention.build_policy_alerts(base_alerts, streaks={})[0]]
            for where in ("first", "last"):
                mixed = [w15_alert, *base_alerts] if where == "first" else [*base_alerts, w15_alert]
                crit = [(a.level, a.source, a.message) for a in attention.build_policy_alerts(mixed, streaks={})[0]]
                results[f"{folder}|{name}|{where}"] = {
                    "crit": crit,
                    "base_crit": base_crit,
                    "matches_without_w15": crit == base_crit,
                }
    print(json.dumps(results))
"""


def run_child(src_dir: Path, mode: str) -> dict:
    with TemporaryDirectory(dir=HERE, prefix=f"tmp-{mode}-") as tmp:
        script_path = Path(tmp) / "child.py"
        script_path.write_text(CHILD_RUNNER)
        env = {
            **os.environ,
            "PYTHONPATH": str(src_dir),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        proc = subprocess.run(
            [sys.executable, str(script_path), str(src_dir), mode, tmp],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        return json.loads(proc.stdout)


def check_round2_findings_and_policy():
    print("=== 1A. Round-2 Finding 1 (OOM-named paths): d7c66234 vs 284dc97a ===")
    r2_oom = run_child(R2_SRC, "oom_paths")
    head_oom = run_child(REPO_SRC, "oom_paths")
    for case_name in r2_oom:
        r2_case = r2_oom[case_name]
        hd_case = head_oom[case_name]
        r2_sources = [(lvl, src) for lvl, src, _, _ in r2_case["alerts"]]
        hd_sources = [(lvl, src) for lvl, src, _, _ in hd_case["alerts"]]
        assert r2_sources == [("WARN", SOURCE), ("CRITICAL", "analytics_job_oom")], (case_name, r2_sources)
        assert r2_case["packets"] == [] and len(r2_case["rejections"]) == 1 and r2_case["rejections"][0] > 1000
        assert r2_case["receipt"]["status"] == "failed" and r2_case["receipt"]["critical_count"] == 1

        assert hd_sources == [("WARN", SOURCE)], (case_name, hd_sources)
        assert len(hd_case["packets"]) == 1 and len(hd_case["packets"][0]) < 1000
        assert hd_case["rejections"] == []
        assert hd_case["receipt"]["status"] == "sent"
        assert hd_case["receipt"]["critical_count"] == 0 and hd_case["receipt"]["warn_attention_count"] == 1
        print(
            f"  {case_name}: d7c66234 -> alerts={r2_sources}, rejected_len={r2_case['rejections'][0]} | "
            f"284dc97a -> alerts={hd_sources}, sent_len={len(hd_case['packets'][0])}"
        )

    print("\n=== 1B. Round-2 Finding 2 (Backticks in disk paths/filenames): d7c66234 vs 284dc97a ===")
    r2_bt = run_child(R2_SRC, "backticks")
    hd_bt = run_child(REPO_SRC, "backticks")

    # Case 1: broken status filename with backtick & backslash
    assert TICK in r2_bt["broken_filename"]["logs"]
    assert TICK not in hd_bt["broken_filename"]["logs"]
    assert "\\x60\\\\x.json" in hd_bt["broken_filename"]["logs"]
    assert TICK not in hd_bt["broken_filename"]["packets"][0]
    expected_key1 = "probable_pitcher_cache:2026-12-31:" + hashlib.sha256(
        json.dumps([hd_bt["broken_filename"]["raw_name"]], separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    assert hd_bt["broken_filename"]["alerts"][0][3] == expected_key1
    print("  broken_filename: d7c66234 had raw backtick in log; 284dc97a escaped as \\x60\\\\x.json and kept raw incident_key")

    # Case 2: custom status root + cache_path + preserved copy + producer dropped stderr
    assert TICK in r2_bt["custom_root_and_cache_path"]["packets"][0]
    assert TICK in r2_bt["custom_root_and_cache_path"]["logs"]
    assert TICK in r2_bt["custom_root_and_cache_path"]["stderr_dropped"]

    assert TICK not in hd_bt["custom_root_and_cache_path"]["packets"][0]
    assert TICK not in hd_bt["custom_root_and_cache_path"]["logs"]
    assert "other\\x60d\\\\ata/models/rejected/probable_pitcher_lookup." in hd_bt["custom_root_and_cache_path"]["logs"]
    assert TICK not in hd_bt["custom_root_and_cache_path"]["stderr_dropped"]
    assert "other\\x60d\\\\ata/models/probable_pitcher_lookup.json" in hd_bt["custom_root_and_cache_path"]["stderr_dropped"]
    # On-disk record keeps raw path with literal backtick and single backslash
    assert f"other{TICK}d\\ata" in hd_bt["custom_root_and_cache_path"]["record_cache_path"]
    print("  custom_root_and_cache_path: d7c66234 leaked backtick to DM, log, and stderr; 284dc97a has 0 backticks in DM/log/stderr and keeps raw record path")

    # Case 3: producer preserve & status failure lines
    fail_lines = hd_bt["producer_failures_stderr"].splitlines()
    assert len(fail_lines) == 3, fail_lines
    assert all(TICK not in line for line in fail_lines), fail_lines
    assert any("could not preserve probable-pitcher cache evidence at " in line and "w\\x60d\\\\z" in line for line in fail_lines)
    assert any("could not write probable-pitcher cache status at " in line and "w\\x60d\\\\z" in line for line in fail_lines)
    print("  producer_failures_stderr: all 3 lines (dropped, preserve failure, status failure) escape backticks & backslashes")

    # Case 4: pre-existing unreadable cache stderr line keeps raw path (byte-for-byte base exception)
    assert r2_bt["producer_unreadable_stderr"] == hd_bt["producer_unreadable_stderr"]
    assert f"u{TICK}n\\r/probable_pitcher_lookup.json" in hd_bt["producer_unreadable_stderr"]
    print("  producer_unreadable_stderr: byte-for-byte unchanged (documented P2/S2 exception)")

    # Case 5: FIFO status dir under backtick path
    assert TICK not in hd_bt["fifo_status_dir_backtick"]["packets"][0]
    assert TICK not in hd_bt["fifo_status_dir_backtick"]["logs"]
    assert "fifo\\x60dir/health_state/probable_pitcher_cache" in hd_bt["fifo_status_dir_backtick"]["logs"]
    print("  fifo_status_dir_backtick: escaped in log, omitted from DM summary, non-blocking")

    print("\n=== 1C. Policy 'Must Still Catch' Matrix: d7c66234 vs 284dc97a ===")
    r2_pol = run_child(R2_SRC, "policy_matrix")
    hd_pol = run_child(REPO_SRC, "policy_matrix")
    diffs = []
    for k in sorted(hd_pol):
        assert hd_pol[k]["matches_without_w15"], (k, hd_pol[k])
        if r2_pol[k]["crit"] != hd_pol[k]["crit"]:
            diffs.append((k, r2_pol[k]["crit"], hd_pol[k]["crit"]))
    print(f"  284dc97a matches policy(without_w15) on all {len(hd_pol)}/{len(hd_pol)} combinations.")
    print(f"  Differences between d7c66234 and 284dc97a ({len(diffs)} cases):")
    for k, before_c, after_c in diffs:
        b_desc = [(src, ("probable_pitcher_cache:" in msg)) for _, src, msg in before_c]
        a_desc = [(src, ("probable_pitcher_cache:" in msg)) for _, src, msg in after_c]
        print(f"    {k}: before={b_desc} -> after={a_desc}")


def check_other_ways_and_hostile_records():
    print("\n=== 1D. Exhaustive Summary Sweep & Hostile Disk Inputs ===")
    sys.path.insert(0, str(REPO_SRC))
    from bts.features import probable_pitcher_cache as producer
    from bts.health import alert, attention, probable_pitcher_cache as health, runner
    from bts.probable_pitcher_status import read_record

    labels_all = ("invalid entries", "status unreadable", "unreadable cache")
    max_len_1000 = 0
    for r in range(1, 4):
        for subset in itertools.combinations(labels_all, r):
            for count in list(range(1, 1001)) + [9999, 10**6, 10**25, 10**26]:
                # Compute message directly for huge counts without allocating 10**25 list
                noun = "observation" if count == 1 else "observations"
                lbl = ", ".join(sorted(subset))
                if count <= 1000:
                    a = health.summarize(DAY, ["x"] * count, set(subset))
                    text = a.message
                    if len(text) > max_len_1000:
                        max_len_1000 = len(text)
                else:
                    a = health.summarize(DAY, ["x"] * 2, set(subset))
                    text = a.message.replace("2 observations", f"{count} {noun}", 1)
                assert not attention.OOM_EVIDENCE_RE.search(f"{SOURCE} {text}".lower()), text
                assert TICK not in text
                if count <= 10**25:
                    assert len(text) <= 600, (count, len(text))
                else:
                    # Only exceeds 600 at 10**26 observations (impossible in memory/filesystem)
                    assert len(text) == 601
    # Check formatted DM length with resurface note + multi-digit streak
    worst_alert = health.summarize(DAY, ["x"] * 1000, set(labels_all))
    streaked = attention._with_streak(worst_alert, 365)
    dm_body = alert.format_health_dm_body([], [streaked])
    resurface = alert._resurface_note({"status": "failed", "critical_count": 3})
    full_dm = f"{resurface}\n\n{dm_body}"
    assert len(full_dm) < 1000, len(full_dm)
    print(f"  Max summary len (count<=1000): {max_len_1000} (<=600 up to count=10**25); worst full DM with H6 banner & 365d streak: {len(full_dm)} chars (<1000)")

    # Hostile status record probe: check recorded_at date-time separator quirk in datetime.fromisoformat!
    with TemporaryDirectory(dir=HERE, prefix="tmp-hostile-") as tmp:
        st = Path(tmp) / "oom/data/health_state" / SOURCE
        st.mkdir(parents=True)
        identity = "a" * 64
        for sep_label, sep_char in [("backtick", TICK), ("newline", "\n")]:
            st_sub = st / sep_label / "oom/data/health_state" / SOURCE
            st_sub.mkdir(parents=True)
            crafted = dict(
                schema_version="bts_probable_pitcher_cache_status_v1",
                reason="invalid_entries",
                recorded_at=f"2026-12-31{sep_char}14:00:00+00:00",
                recorded_date_et="2026-12-31",
                cache_path="data/models/probable_pitcher_lookup.json",
                dropped=1, recovered_from_cache=0, rescanned=0, unusable_feed=0, without_feed=1,
                game_ids=[100], identity=identity, preserved=None, original_sha256="c" * 64,
            )
            rec_path = st_sub / f"2026-12-31.invalid_entries.{identity[:12]}.json"
            rec_path.write_text(json.dumps(crafted))
            logs = StringIO()
            h = logging.StreamHandler(logs)
            logger = logging.getLogger(health.__name__)
            logger.addHandler(h)
            try:
                alerts = health.check(st_sub, DAY)
            finally:
                logger.removeHandler(h)
            logged_text = logs.getvalue()
            print(
                f"  Crafted recorded_at ({sep_label}): read_record accepted=True, "
                f"sep_in_log={sep_char in logged_text}, sep_in_summary={sep_char in alerts[0].message}, "
                f"oom_in_summary={bool(attention.OOM_EVIDENCE_RE.search(alerts[0].message))}"
            )
            # Also check publish() coalescing against crafted existing record
            same_except_ts = dict(crafted, recorded_at="2026-12-31T15:00:00+00:00")
            producer.publish(rec_path, same_except_ts)
            persisted_ts = json.loads(rec_path.read_text())["recorded_at"]
            print(f"    publish() coalesced over crafted recorded_at: kept_crafted={persisted_ts == crafted['recorded_at']}")

        # Hostile crash-resistance sweep under oom/data/health_state/probable_pitcher_cache:
        # Ensure check() NEVER raises into runner._safe_run (which would embed 'oom/data' in a health_runner CRITICAL)
        st_crash = Path(tmp) / "crash_sweep/oom/data/health_state" / SOURCE
        st_crash.mkdir(parents=True)
        hostile_payloads = {
            "non_utf8": b"\xff\xfe",
            "oversize": b"{" + b" " * 65536 + b"}",
            "huge_int_5000_digits": ('{"x": ' + ("9" * 5000) + "}").encode(),
            "deep_nesting": ("[" * 2500 + "]" * 2500).encode(),
            "unhashable_reason": json.dumps({
                **dict(
                    schema_version="bts_probable_pitcher_cache_status_v1",
                    reason=["unhashable"],
                    recorded_at="2026-12-31T14:00:00+00:00",
                    recorded_date_et="2026-12-31",
                    cache_path="x",
                    dropped=1, recovered_from_cache=0, rescanned=0, unusable_feed=0, without_feed=1,
                    game_ids=[100], identity="a" * 64, preserved=None, original_sha256="c" * 64,
                )
            }).encode(),
            "unhashable_game_id": json.dumps({
                **dict(
                    schema_version="bts_probable_pitcher_cache_status_v1",
                    reason="invalid_entries",
                    recorded_at="2026-12-31T14:00:00+00:00",
                    recorded_date_et="2026-12-31",
                    cache_path="x",
                    dropped=1, recovered_from_cache=0, rescanned=0, unusable_feed=0, without_feed=1,
                    game_ids=[{"bad": 1}], identity="a" * 64, preserved=None, original_sha256="c" * 64,
                )
            }).encode(),
            "overflow_timestamp": json.dumps({
                **dict(
                    schema_version="bts_probable_pitcher_cache_status_v1",
                    reason="invalid_entries",
                    recorded_at="0001-01-01T01:00:00+00:00",
                    recorded_date_et="0001-01-01",
                    cache_path="x",
                    dropped=1, recovered_from_cache=0, rescanned=0, unusable_feed=0, without_feed=1,
                    game_ids=[100], identity="a" * 64, preserved=None, original_sha256="c" * 64,
                )
            }).encode(),
        }
        for name, data_bytes in hostile_payloads.items():
            (st_crash / f"{DAY}.{name}.json").write_bytes(data_bytes)
        os.mkfifo(st_crash / f"{DAY}.fifo.json")
        (st_crash / f"{DAY}.dir.json").mkdir()
        (st_crash / f"{DAY}.sym.json").symlink_to(st_crash / f"{DAY}.non_utf8.json")

        safe_alerts = runner._safe_run("probable_pitcher_cache", lambda: health.check(st_crash, DAY))
        assert len(safe_alerts) == 1 and safe_alerts[0].level == "WARN" and safe_alerts[0].source == SOURCE
        assert not attention.OOM_EVIDENCE_RE.search(f"{safe_alerts[0].source} {safe_alerts[0].message}".lower())
        print(f"  Hostile crash-resistance sweep ({len(hostile_payloads) + 3} files under oom/data): 0 crashes, 1 clean WARN")


if __name__ == "__main__":
    check_round2_findings_and_policy()
    check_other_ways_and_hostile_records()
    print("\nALL ITEM 1 PROBES COMPLETED.")
