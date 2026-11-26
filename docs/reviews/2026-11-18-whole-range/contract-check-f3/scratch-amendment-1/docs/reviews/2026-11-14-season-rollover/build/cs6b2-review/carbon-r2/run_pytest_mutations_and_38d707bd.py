"""Pytest Family A runner (invoked once via `~/bts-runs/yield_run.sh`):
1. Runs `tests/health/test_alert_status_cs6b2_review.py` (and `adopt/`) against `38d707bd`
   (`git archive 38d707bd src`) and verifies the lead's race cases fail cleanly via
   bounded `pytest.fail` (~3s per blocking case, no hang), then pass on `HEAD`.
2. Runs 10 targeted scratch-`src` mutations against `tests/health/test_alert_status_cs6b2_review.py`:
   - `drop_nonblock_read`, `drop_nonblock_write`
   - `drop_fstat_read`, `drop_fstat_write`
   - `truncate_before_check`, `drop_ftruncate`
   - `leak_fd_read`, `leak_fd_read_exc_only`, `leak_fd_write`, `leak_fd_write_exc_only`
   plus `o_trunc_in_open` against both `test_alert_status_cs6b2_review.py` and `adopt/`.
3. Verifies the `green_permission` and `test_one_status_snapshot` hooks in
   `tests/health/test_alert_status_cs6b2.py` deny the READ and allow the WRITE.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REVIEW_DIR = Path(__file__).resolve().parent


def run_pytest_with_src(
    src_dir: Path,
    targets: list[str],
    summary_file: Path,
    extra_args: list[str] | None = None,
) -> tuple[dict[str, object], float]:
    env = dict(os.environ)
    env["TZ"] = "America/New_York"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["BTS_SUMMARY_OUT"] = str(summary_file)
    existing_pp = env.get("PYTHONPATH", "")
    pp_parts = [str(src_dir), str(REVIEW_DIR)]
    if existing_pp:
        pp_parts.append(existing_pp)
    env["PYTHONPATH"] = os.pathsep.join(pp_parts)

    # Confirm bts.health.alert resolves to `src_dir`
    chk = subprocess.run(
        [sys.executable, "-c", "import bts.health.alert as a; print(a.__file__)"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    resolved = Path(chk.stdout.strip()).resolve()
    assert resolved == (src_dir / "bts" / "health" / "alert.py").resolve(), (
        f"unexpected alert.py path: {resolved} vs {src_dir}"
    )

    summary_file.unlink(missing_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        "-p",
        "review_pytest_plugin",
        "-q",
        *(extra_args or []),
        *targets,
    ]
    t0 = time.monotonic()
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    elapsed = round(time.monotonic() - t0, 2)
    assert summary_file.is_file(), (
        f"pytest did not write summary (rc={proc.returncode}):\n"
        f"STDOUT:\n{proc.stdout[-1500:]}\nSTDERR:\n{proc.stderr[-1500:]}"
    )
    data = json.loads(summary_file.read_text(encoding="utf-8"))
    assert data["state_leaks"] == [], f"process state leaked: {data['state_leaks']}"
    return data, elapsed


def extract_git_src(rev: str, dest_root: Path) -> Path:
    raw = subprocess.check_output(["git", "archive", rev, "src"], cwd=REPO_ROOT)
    with tarfile.open(fileobj=io.BytesIO(raw)) as tf:
        tf.extractall(path=dest_root, filter="data")
    return dest_root / "src"


def main() -> None:
    review_test = "tests/health/test_alert_status_cs6b2_review.py"
    adopt_test = ".codex-review/cs6b2-r2/adopt/test_alert_status_cs6b2_r2_adopt.py"
    authored_test = "tests/health/test_alert_status_cs6b2.py"

    with tempfile.TemporaryDirectory(dir=REVIEW_DIR, prefix="tmp_mut_") as tmp_s:
        tmp = Path(tmp_s)
        summary_json = tmp / "summary.json"

        # -----------------------------------------------------------------
        # 1. Lead's tests on 38d707bd (RED, clean bounded failures) vs HEAD (GREEN)
        # -----------------------------------------------------------------
        print("=== 1. Lead's review tests & adopt tests on 38d707bd vs HEAD ===", flush=True)
        r1_root = tmp / "r1_38d707bd"
        r1_root.mkdir()
        r1_src = extract_git_src("38d707bd", r1_root)

        r1_res, r1_sec = run_pytest_with_src(r1_src, [review_test], summary_json)
        r1_failed_ids = [item["nodeid"] for item in r1_res["failed_items"]]
        print(
            f"38d707bd ({review_test}): counts={r1_res['counts']} elapsed={r1_sec}s\n"
            f"  failed_ids={r1_failed_ids}",
            flush=True,
        )
        assert r1_res["counts"]["failed"] == 9, f"expected 9 failures on 38d707bd: {r1_res}"
        assert r1_res["counts"]["passed"] == 6

        r1_adopt_res, r1_adopt_sec = run_pytest_with_src(r1_src, [adopt_test], summary_json)
        print(
            f"38d707bd ({adopt_test}): counts={r1_adopt_res['counts']} elapsed={r1_adopt_sec}s",
            flush=True,
        )
        assert r1_adopt_res["counts"]["failed"] >= 8

        head_src = REPO_ROOT / "src"
        head_res, head_sec = run_pytest_with_src(head_src, [review_test, adopt_test], summary_json)
        print(
            f"HEAD ({review_test} + adopt): counts={head_res['counts']} elapsed={head_sec}s",
            flush=True,
        )
        assert head_res["counts"]["failed"] == 0 and head_res["exitstatus"] == 0

        # -----------------------------------------------------------------
        # 2. Verify green_permission & test_one_status_snapshot hooks in authored file
        # -----------------------------------------------------------------
        print("\n=== 2. Verify authored hooks in test_alert_status_cs6b2.py ===", flush=True)
        hook_res, hook_sec = run_pytest_with_src(
            head_src,
            [authored_test],
            summary_json,
            extra_args=["-k", "green_permission or test_one_status_snapshot"],
        )
        print(f"HEAD authored hooks: counts={hook_res['counts']} elapsed={hook_sec}s", flush=True)
        assert hook_res["counts"]["failed"] == 0
        # 3 lanes x 4 modes = 12 green_permission + 1 test_one_status_snapshot = 13 passed
        assert hook_res["counts"]["passed"] == 13

        # Prove directly that green_permission denies the READ (PermissionError) and allows the WRITE
        verify_hook_code = """
import json, os, tempfile
from pathlib import Path
from unittest.mock import Mock
from bts.health import alert

with tempfile.TemporaryDirectory() as d:
    status_path = Path(d) / "health_dm_delivery_status.json"
    status_path.write_bytes(b'{"status":"failed","critical_count":1}')
    real_os_open = os.open
    events = []
    def denied_read(path, flags, *args, **kwargs):
        if Path(path) == status_path and not flags & (os.O_WRONLY | os.O_RDWR):
            events.append(("READ_DENIED", flags))
            raise PermissionError("injected status read denial")
        if Path(path) == status_path:
            events.append(("WRITE_ALLOWED", flags))
        return real_os_open(path, flags, *args, **kwargs)
    os.open = denied_read
    send = Mock(return_value="ok")
    orig_send = alert.send_dm
    alert.send_dm = send
    try:
        res = alert.dispatch_dm_for_health_alerts([alert.Alert("CRITICAL", "probe", "m")], "offline.test", status_path=status_path)
    finally:
        os.open = real_os_open
        alert.send_dm = orig_send
    assert res is True
    assert [e[0] for e in events] == ["READ_DENIED", "WRITE_ALLOWED"], events
    assert "Previous health alert" not in send.call_args.args[1]
    assert json.loads(status_path.read_bytes())["status"] == "sent"
    print("HOOK_DIRECT_PROOF:", [e[0] for e in events])
"""
        hk_out = subprocess.check_output(
            [sys.executable, "-c", verify_hook_code],
            cwd=REPO_ROOT,
            env={**os.environ, "PYTHONPATH": str(head_src)},
            text=True,
        ).strip()
        print(hk_out, flush=True)

        # -----------------------------------------------------------------
        # 3. Scratch mutations on HEAD's src/bts/health/alert.py
        # -----------------------------------------------------------------
        print("\n=== 3. Scratch mutations against test_alert_status_cs6b2_review.py ===", flush=True)
        mut_src = tmp / "mut_src" / "src"
        shutil.copytree(head_src, mut_src)
        alert_file = mut_src / "bts" / "health" / "alert.py"
        clean_alert_text = alert_file.read_text(encoding="utf-8")

        mutations: list[tuple[str, str, str, bool]] = [
            (
                "drop_nonblock_read",
                "    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)",
                "    fd = os.open(path, os.O_RDONLY)",
                True,
            ),
            (
                "drop_nonblock_write",
                "    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_NONBLOCK, 0o666)",
                "    fd = os.open(path, os.O_WRONLY | os.O_CREAT, 0o666)",
                True,
            ),
            (
                "drop_fstat_read",
                '    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)\n'
                '    try:\n'
                '        if not stat.S_ISREG(os.fstat(fd).st_mode):\n'
                '            raise OSError("not a regular file")',
                '    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)\n'
                '    try:',
                True,
            ),
            (
                "drop_fstat_write",
                '    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_NONBLOCK, 0o666)\n'
                '    try:\n'
                '        if not stat.S_ISREG(os.fstat(fd).st_mode):\n'
                '            raise OSError("not a regular file")\n'
                '        os.ftruncate(fd, 0)',
                '    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_NONBLOCK, 0o666)\n'
                '    try:\n'
                '        os.ftruncate(fd, 0)',
                True,
            ),
            (
                "truncate_before_check",
                '    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_NONBLOCK, 0o666)\n'
                '    try:\n'
                '        if not stat.S_ISREG(os.fstat(fd).st_mode):\n'
                '            raise OSError("not a regular file")\n'
                '        os.ftruncate(fd, 0)',
                '    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_NONBLOCK, 0o666)\n'
                '    try:\n'
                '        os.ftruncate(fd, 0)\n'
                '        if not stat.S_ISREG(os.fstat(fd).st_mode):\n'
                '            raise OSError("not a regular file")',
                True,
            ),
            (
                "drop_ftruncate",
                '        os.ftruncate(fd, 0)\n'
                '        with open(fd, "w", closefd=False) as handle:',
                '        with open(fd, "w", closefd=False) as handle:',
                True,
            ),
            (
                "leak_fd_read",
                '        with open(fd, "r", closefd=False) as handle:\n'
                '            return handle.read()\n'
                '    finally:\n'
                '        os.close(fd)',
                '        with open(fd, "r", closefd=False) as handle:\n'
                '            return handle.read()\n'
                '    finally:\n'
                '        pass',
                True,
            ),
            (
                "leak_fd_read_exc_only",
                '        with open(fd, "r", closefd=False) as handle:\n'
                '            return handle.read()\n'
                '    finally:\n'
                '        os.close(fd)',
                '        with open(fd, "r", closefd=False) as handle:\n'
                '            _out = handle.read()\n'
                '        os.close(fd)\n'
                '        return _out\n'
                '    finally:\n'
                '        pass',
                True,
            ),
            (
                "leak_fd_write",
                '        with open(fd, "w", closefd=False) as handle:\n'
                '            handle.write(text)\n'
                '    finally:\n'
                '        os.close(fd)',
                '        with open(fd, "w", closefd=False) as handle:\n'
                '            handle.write(text)\n'
                '    finally:\n'
                '        pass',
                True,
            ),
            (
                "leak_fd_write_exc_only",
                '        with open(fd, "w", closefd=False) as handle:\n'
                '            handle.write(text)\n'
                '    finally:\n'
                '        os.close(fd)',
                '        with open(fd, "w", closefd=False) as handle:\n'
                '            handle.write(text)\n'
                '        os.close(fd)\n'
                '    finally:\n'
                '        pass',
                True,
            ),
            (
                "o_trunc_in_open_probe",
                '    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_NONBLOCK, 0o666)\n'
                '    try:\n'
                '        if not stat.S_ISREG(os.fstat(fd).st_mode):\n'
                '            raise OSError("not a regular file")\n'
                '        os.ftruncate(fd, 0)',
                '    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NONBLOCK, 0o666)\n'
                '    try:\n'
                '        if not stat.S_ISREG(os.fstat(fd).st_mode):\n'
                '            raise OSError("not a regular file")',
                False,
            ),
        ]

        for m_name, target, repl, must_kill_in_review in mutations:
            occ = clean_alert_text.count(target)
            assert occ == 1, f"mutation {m_name} target count={occ}"
            alert_file.write_text(clean_alert_text.replace(target, repl, 1), encoding="utf-8")

            m_res, m_sec = run_pytest_with_src(mut_src, [review_test], summary_json)
            f_ids = [it["nodeid"].split("::")[-1] for it in m_res["failed_items"]]
            print(
                f"MUTATION {m_name} vs review.py: failed={m_res['counts']['failed']} "
                f"passed={m_res['counts']['passed']} ({m_sec}s) -> {f_ids}",
                flush=True,
            )
            if must_kill_in_review:
                assert m_res["counts"]["failed"] >= 1, f"SURVIVED MUTATION: {m_name}"
            else:
                m_adopt, a_sec = run_pytest_with_src(mut_src, [adopt_test], summary_json)
                af_ids = [it["nodeid"].split("::")[-1] for it in m_adopt["failed_items"]]
                print(
                    f"MUTATION {m_name} vs adopt.py: failed={m_adopt['counts']['failed']} "
                    f"passed={m_adopt['counts']['passed']} ({a_sec}s) -> {af_ids}",
                    flush=True,
                )
                assert m_adopt["counts"]["failed"] >= 1

    print("\nPYTEST FAMILY A COMPLETE: ALL MUTATIONS & 38d707bd CHECKS VERIFIED", flush=True)


if __name__ == "__main__":
    main()
