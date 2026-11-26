"""Adversarial probe for Item 2: Round-2 HEAD (6ff1e87e) vs Base (88b69a16) equivalence.

Verifies identical _replay_season_streak returns/exceptions, CLI exits/exceptions,
streak.json bytes, pick bytes, saved/not_reached outputs/receipts, and round-2
escaped single-line stderr + raw receipt fields across:
  1. 630 histories of tests/test_wr_c3v_corpus.py
  2. 9 real-writer sequences of tests/test_wr_c3v_delivery.py
  3. 9 real-writer/symlink sequences of tests/test_wr_c3_r2_entered_pick_blockers.py
  4. Round 1's hostile probes (Codex hostile_probe + Carbon adversarial hunt + B1 stems)
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tarfile
import tempfile
from threading import Event, Thread
import urllib.error
from zoneinfo import ZoneInfo


ET = ZoneInfo("America/New_York")
PINNED_UTC = datetime(2027, 4, 20, 6, 0, 0, tzinfo=timezone.utc)
REVIEW_DIR = Path(__file__).resolve().parent
REPO_ROOT = REVIEW_DIR.parents[1]
HEAD_SRC = REPO_ROOT / "src"
BASE_REV = "88b69a16"


@contextmanager
def _alarm(seconds: int, *, as_keyboard_interrupt: bool = False):
    class _ProbeTimeout(KeyboardInterrupt if as_keyboard_interrupt else BaseException):
        pass

    prev = signal.signal(signal.SIGALRM, lambda *_a: (_ for _ in ()).throw(_ProbeTimeout("probe deadline")))
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev)


def _streak_state(root: Path):
    target = root / "streak.json"
    try:
        st = os.lstat(target)
    except FileNotFoundError:
        return {"kind": "missing"}
    if stat.S_ISLNK(st.st_mode):
        link = os.readlink(target)
        body = target.read_bytes().hex() if target.exists() and target.is_file() else None
        return {"kind": "symlink", "link": link, "hex": body}
    if stat.S_ISDIR(st.st_mode):
        return {"kind": "directory"}
    if stat.S_ISREG(st.st_mode):
        return {"kind": "file", "hex": target.read_bytes().hex()}
    return {"kind": f"mode:{st.st_mode}"}


def _pick_hashes(cli_picks: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not cli_picks.exists():
        return out
    for child in sorted(cli_picks.iterdir(), key=lambda p: p.name):
        if child.name == ".scoring.lock":
            continue
        st = os.lstat(child)
        if stat.S_ISLNK(st.st_mode):
            out[child.name] = f"symlink:{os.readlink(child)}"
        elif stat.S_ISFIFO(st.st_mode):
            out[child.name] = "fifo"
        elif stat.S_ISDIR(st.st_mode):
            out[child.name] = "dir"
        elif stat.S_ISREG(st.st_mode):
            try:
                out[child.name] = hashlib.sha256(child.read_bytes()).hexdigest()
            except OSError as exc:
                out[child.name] = f"err:{type(exc).__name__}"
    return out


@contextmanager
def _contended_lock_crossing(root: Path, start: datetime, after: datetime):
    acquired = Event()
    contender = Event()
    state = {"now": start}
    real_flock = fcntl.flock
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".scoring.lock"

    def _hold():
        with lock.open("w") as peer:
            real_flock(peer, fcntl.LOCK_EX)
            acquired.set()
            try:
                if contender.wait(3):
                    state["now"] = after
            finally:
                real_flock(peer, fcntl.LOCK_UN)

    def _observed_flock(fd, operation):
        if operation == fcntl.LOCK_EX:
            try:
                real_flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                contender.set()
            else:
                raise AssertionError("peer did not hold scoring lock")
        return real_flock(fd, operation)

    peer = Thread(target=_hold, daemon=True)
    peer.start()
    orig_flock = fcntl.flock
    try:
        assert acquired.wait(3)
        fcntl.flock = _observed_flock  # type: ignore[assignment]
        yield lambda: state["now"]
        assert contender.is_set() and state["now"] == after
    finally:
        fcntl.flock = orig_flock  # type: ignore[assignment]
        contender.set()
        peer.join(timeout=3)


def _worker_main(src_dir_str: str) -> int:
    src_dir = Path(src_dir_str).resolve()
    cleaned = [
        entry for entry in sys.path
        if Path(entry or ".").resolve() not in (REPO_ROOT, HEAD_SRC, REVIEW_DIR)
    ]
    sys.path[:] = [str(src_dir), *cleaned]

    def _no_net(*args, **kwargs):
        raise RuntimeError(f"Network blocked in worker: {args}")

    socket.socket.connect = _no_net  # type: ignore[assignment]
    socket.socket.connect_ex = _no_net  # type: ignore[assignment]
    socket.getaddrinfo = _no_net  # type: ignore[assignment]

    from click.testing import CliRunner
    from bts import cli as cli_mod
    from bts import picks as picks_mod
    from bts import receipt_io as rio_mod
    from bts import reconcile_receipt as rr_mod

    for mod in (picks_mod, cli_mod, rr_mod):
        assert Path(mod.__file__).resolve().is_relative_to(src_dir), mod.__file__

    class PinnedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return PINNED_UTC.astimezone(tz) if tz is not None else PINNED_UTC.replace(tzinfo=None)

    real_datetime = picks_mod.datetime
    real_git_sha = picks_mod._git_head_sha
    real_resolve = picks_mod.resolve_daily_slot_results
    real_save_pick = picks_mod.save_pick
    real_rr_class = rr_mod.ReconcileReceipt
    real_publish = rio_mod.publish
    real_read_text = Path.read_text

    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        spec = json.loads(raw_line)
        if spec.get("cmd") == "quit":
            break

        work_root = Path(spec["work_root"])
        fold_picks = work_root / "fold" / "picks"
        cli_picks = work_root / "cli" / "picks"
        applied_chmods: list[Path] = []

        try:
            fifo_rel = spec.get("mkfifo")
            if fifo_rel:
                for p in (fold_picks, cli_picks):
                    p.mkdir(parents=True, exist_ok=True)
                    os.mkfifo(p / fifo_rel)

            for item in spec.get("chmod", []):
                for p in (fold_picks, cli_picks):
                    t = p / item["rel"]
                    t.chmod(item["mode"])
                    applied_chmods.append(t)

            picks_mod.datetime = PinnedDatetime
            picks_mod._git_head_sha = lambda *_a, **_k: BASE_REV

            feed_mode = spec.get("result_feed")
            if feed_mode is None:
                picks_mod.resolve_daily_slot_results = lambda *_a, **_k: (_ for _ in ()).throw(
                    AssertionError("Unexpected resolve_daily_slot_results")
                )
            elif feed_mode == "raise_url_error":
                picks_mod.resolve_daily_slot_results = lambda *_a, **_k: (_ for _ in ()).throw(
                    urllib.error.URLError("synthetic feed down")
                )
            else:
                picks_mod.resolve_daily_slot_results = lambda *_a, _f=feed_mode, **_k: dict(_f)

            if spec.get("save_pick_error") == "disk_full":
                picks_mod.save_pick = lambda *_a, **_k: (_ for _ in ()).throw(OSError("disk full"))
            else:
                picks_mod.save_pick = real_save_pick

            if spec.get("publish_oserror"):
                rio_mod.publish = lambda *_a, **_k: (_ for _ in ()).throw(OSError("synthetic receipt io error"))
            else:
                rio_mod.publish = real_publish

            unlink_name = spec.get("concurrent_unlink_on_read")
            interrupt_name = spec.get("interrupt_on_read")
            if unlink_name:
                seen: set[str] = set()

                def _unlink_read(self_p: Path, *a, **k):
                    txt = real_read_text(self_p, *a, **k)
                    key = str(self_p)
                    if self_p.name == unlink_name and key not in seen:
                        seen.add(key)
                        self_p.unlink()
                    return txt

                Path.read_text = _unlink_read  # type: ignore[method-assign]
            elif interrupt_name:
                def _interrupt_read(self_p: Path, *a, **k):
                    if self_p.name == interrupt_name:
                        raise KeyboardInterrupt()
                    return real_read_text(self_p, *a, **k)

                Path.read_text = _interrupt_read  # type: ignore[method-assign]
            else:
                Path.read_text = real_read_text  # type: ignore[method-assign]

            clock_mode = spec.get("clock_mode", "fixed")
            start_now = datetime.fromisoformat(spec["clock_iso"])
            after_now = datetime.fromisoformat(spec["after_iso"]) if spec.get("after_iso") else start_now

            # 1. Direct _replay_season_streak check (skip on FIFO hang test)
            if spec.get("skip_direct_fold"):
                fold_out = {"status": "skipped"}
            else:
                fold_day = after_now.astimezone(ET).date() if clock_mode == "contended_flock" else start_now.astimezone(ET).date()
                with _alarm(10):
                    try:
                        val = picks_mod._replay_season_streak(fold_picks, fold_day.year, fold_day.isoformat())
                        fold_out = {"status": "ok", "value": list(val) if val is not None else None}
                    except BaseException as exc:
                        norm_args = [str(x).replace(str(work_root), "<work>") for x in exc.args]
                        fold_out = {"status": "exc", "type": type(exc).__name__, "args": norm_args}

            # 2. CLI reconcile check
            streak_before = _streak_state(cli_picks)
            current_now = start_now

            def _invoke_cli(clk):
                def _make_receipt(**kw):
                    rec = real_rr_class(clock=clk, lookback_days=kw["lookback_days"])
                    rec.run_id = "1" * 32
                    if clock_mode == "advance_after_publish":
                        orig_pub = rec.publish

                        def _pub_and_advance(pdir):
                            nonlocal current_now
                            out = orig_pub(pdir)
                            current_now = after_now
                            return out

                        rec.publish = _pub_and_advance  # type: ignore[method-assign]
                    return rec

                rr_mod.ReconcileReceipt = _make_receipt  # type: ignore[misc]
                return CliRunner().invoke(
                    cli_mod.cli,
                    ["reconcile", "--picks-dir", str(cli_picks), "--lookback", str(spec.get("lookback", 0))],
                )

            if clock_mode == "contended_flock":
                with _alarm(10), _contended_lock_crossing(cli_picks, start_now, after_now) as flock_clock:
                    res = _invoke_cli(flock_clock)
            elif fifo_rel:
                with _alarm(1, as_keyboard_interrupt=True):
                    res = _invoke_cli(lambda: current_now)
            else:
                with _alarm(10):
                    res = _invoke_cli(lambda: current_now)

            rec_root = rr_mod.receipts_root(cli_picks)
            rec_files = sorted(rec_root.glob("*/*.json")) if rec_root.exists() else []
            rec_hex = rec_files[0].read_bytes().hex() if len(rec_files) == 1 else None
            rec_json = json.loads(bytes.fromhex(rec_hex)) if rec_hex else None

            exc_sig = None
            if res.exception is not None:
                exc_sig = {
                    "type": type(res.exception).__name__,
                    "args": [str(x).replace(str(work_root), "<work>") for x in res.exception.args],
                }

            payload = {
                "fold": fold_out,
                "exit_code": res.exit_code,
                "exc": exc_sig,
                "stdout": res.stdout,
                "stderr": res.stderr,
                "streak_before": streak_before,
                "streak_after": _streak_state(cli_picks),
                "pick_hashes": _pick_hashes(cli_picks),
                "receipt_hex": rec_hex,
                "receipt_json": rec_json,
            }
        finally:
            picks_mod.datetime = real_datetime
            picks_mod._git_head_sha = real_git_sha
            picks_mod.resolve_daily_slot_results = real_resolve
            picks_mod.save_pick = real_save_pick
            rr_mod.ReconcileReceipt = real_rr_class  # type: ignore[misc]
            rio_mod.publish = real_publish
            Path.read_text = real_read_text  # type: ignore[method-assign]
            for p in reversed(applied_chmods):
                try:
                    if os.path.lexists(p):
                        p.chmod(0o700 if p.is_dir() else 0o600)
                except OSError:
                    pass

        sys.stdout.write(json.dumps(payload) + "\n")
        sys.stdout.flush()
    return 0


class WorkerPair:
    def __init__(self, base_src: Path, head_src: Path):
        env = dict(os.environ, TZ="America/New_York", PYTHONDONTWRITEBYTECODE="1")
        self.base = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--worker", str(base_src)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
        )
        self.head = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--worker", str(head_src)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
        )

    def compare(self, label: str, template_picks: Path, tmp_root: Path, **spec) -> tuple[dict, dict]:
        for tag in ("base", "head"):
            for sub in ("fold", "cli"):
                dst = tmp_root / tag / sub / "picks"
                dst.parent.mkdir(parents=True, exist_ok=True)
                if template_picks.exists():
                    shutil.copytree(template_picks, dst, symlinks=True)

        results = {}
        for tag, proc in (("base", self.base), ("head", self.head)):
            msg = dict(spec, work_root=str(tmp_root / tag))
            assert proc.stdin is not None and proc.stdout is not None
            proc.stdin.write(json.dumps(msg) + "\n")
            proc.stdin.flush()
            with _alarm(20):
                line = proc.stdout.readline()
            if not line:
                err = proc.stderr.read() if proc.stderr else ""
                raise RuntimeError(f"Worker {tag} died on {label}: {err}")
            results[tag] = json.loads(line)

        b, h = results["base"], results["head"]
        diffs = []
        for key in ("fold", "exit_code", "exc", "streak_before", "streak_after", "pick_hashes"):
            if b[key] != h[key]:
                diffs.append(f"{key} mismatch: base={b[key]} head={h[key]}")

        b_rec, h_rec = b["receipt_json"], h["receipt_json"]
        b_state = b_rec["replay"]["state"] if b_rec else "unpublished"
        if b_state == "unavailable" or "Season replay unavailable" in h["stderr"]:
            warnings = [ln for ln in h["stderr"].splitlines() if ln.startswith("Season replay unavailable for ")]
            if len(warnings) != 1:
                diffs.append(f"expected 1 warning line on stderr, got {len(warnings)}: {h['stderr']!r}")
            else:
                wline = warnings[0]
                if not all(0x20 <= ord(c) <= 0x7E for c in wline):
                    diffs.append(f"non-printable/non-ASCII in warning line: {wline!r}")
                if b_state == "unavailable":
                    rep = h_rec["replay"] if h_rec else {}
                    esc_date = rep.get("blocked_date", "").encode("unicode_escape").decode("ascii")
                    esc_reason = rep.get("reason", "").encode("unicode_escape").decode("ascii")
                    expected_w = f"Season replay unavailable for {esc_date} ({esc_reason}); streak retained, not recalculated."
                    if wline != expected_w:
                        diffs.append(f"warning mismatch: {wline!r} vs {expected_w!r}")
                    if b_rec != (h_rec | {"replay": {"state": "unavailable"}}):
                        diffs.append("receipt body differed outside replay blocker fields")
                stripped_err = h["stderr"].replace(wline + "\n", "", 1)
                if stripped_err != b["stderr"]:
                    diffs.append(f"remaining stderr differed: {stripped_err!r} vs {b['stderr']!r}")
            expected_out = b["stdout"].replace("Streak: ", "Streak retained: ").replace(
                "Streak recalculated: ", "Streak retained: "
            )
            if h["stdout"] != expected_out:
                diffs.append(f"stdout mismatch: {h['stdout']!r} vs {expected_out!r}")
            if "Streak recalculated" in h["stdout"]:
                diffs.append(f"blocked stdout said recalculated: {h['stdout']!r}")
        else:
            if b["stdout"] != h["stdout"]:
                diffs.append(f"stdout mismatch on {b_state}: {b['stdout']!r} vs {h['stdout']!r}")
            if b["stderr"] != h["stderr"]:
                diffs.append(f"stderr mismatch on {b_state}: {b['stderr']!r} vs {h['stderr']!r}")
            if b["receipt_hex"] != h["receipt_hex"]:
                diffs.append(f"receipt bytes mismatch on {b_state}")

        if diffs:
            raise AssertionError(f"[{label}] DIVERGENCE:\n  " + "\n  ".join(diffs))
        return b, h

    def close(self):
        for proc in (self.base, self.head):
            try:
                if proc.stdin:
                    proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                    proc.stdin.flush()
                proc.wait(timeout=5)
            except Exception:
                proc.kill()


def _extract_verified_base_src(dest_root: Path) -> Path:
    archive = subprocess.check_output(["git", "-C", str(REPO_ROOT), "archive", BASE_REV, "src"], timeout=15)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tf:
        tf.extractall(dest_root, filter="data")
    for rel in ("src/bts/cli.py", "src/bts/picks.py", "src/bts/reconcile_receipt.py"):
        expected = subprocess.check_output(["git", "-C", str(REPO_ROOT), "show", f"{BASE_REV}:{rel}"], timeout=10)
        assert (dest_root / rel).read_bytes() == expected, rel
    return dest_root / "src"


def _run_corpus_630(pair: WorkerPair) -> dict:
    from tests import test_wr_c3v_corpus as corpus

    counts = {"total": 0, "saved": 0, "unavailable": 0, "raised": 0}
    clock_iso = datetime(2026, 8, 20, 2, 0, tzinfo=ET).isoformat()
    with tempfile.TemporaryDirectory(dir=REVIEW_DIR) as tmp:
        tmp_path = Path(tmp)
        for ix in range(len(corpus.ORDINARY)):
            for case in corpus.cases():
                case_root = tmp_path / f"c_{counts['total']}"
                tpl = case_root / "tpl" / "picks"
                tpl.mkdir(parents=True)
                corpus.seed_streak(tpl, season=2026)
                corpus.build_history(tpl, case, ix)
                b, _ = pair.compare(
                    f"corpus:{case.name}[{ix}]", tpl, case_root / "run",
                    clock_iso=clock_iso, lookback=0,
                )
                counts["total"] += 1
                if b["exc"] is not None:
                    counts["raised"] += 1
                elif b["receipt_json"]["replay"]["state"] == "saved":
                    counts["saved"] += 1
                else:
                    counts["unavailable"] += 1
                shutil.rmtree(case_root)
    assert counts == {"total": 630, "saved": 462, "unavailable": 133, "raised": 35}, counts
    return counts


def _run_writers_18(pair: WorkerPair) -> dict:
    import pandas as pd
    import pytest
    from click.testing import CliRunner
    from bts import cli, contest_state, daily_decision, dm, entry_intent, orchestrator, picks, posting, scheduler, strategy
    from bts.model import predict
    from tests import test_wr_c3_r2_entered_pick_blockers as r2
    from tests import test_wr_c3v_delivery as deliv

    checked = []
    run_iso = datetime(2027, 4, 20, 2, 0, tzinfo=ET).isoformat()

    for scenario in (
        "lost_ack", "skip_conflict", "private_crash", "manual_run_crash",
        "lost-ack-eod", "private-preview-restart", "private-preview-quarantine",
        "scheduler-preview-ack", "orchestrator-post-interrupted",
    ):
        with tempfile.TemporaryDirectory(dir=REVIEW_DIR) as tmp, pytest.MonkeyPatch.context() as mp:
            mp.setattr(socket.socket, "connect", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("no net")))
            mp.setattr(socket, "getaddrinfo", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("no net")))
            root = Path(tmp) / "tpl" / "picks"
            for offset in range(9):
                deliv.plant(root, deliv.stagger("2027-04-01", offset), "hit", bluesky_posted=True)
            deliv.seed_streak(root, streak=9, saver=True)
            if scenario == "manual_run_crash":
                deliv.manual_run_commit(root)
            elif scenario in ("lost_ack", "skip_conflict", "private_crash"):
                deliv.scheduler_commit(root, scenario)
            elif scenario == "orchestrator-post-interrupted":
                deliv.orchestrator_commit(root)
            else:
                if scenario == "scheduler-preview-ack":
                    deliv.preview(root)
                elif scenario == "lost-ack-eod":
                    deliv.plant(root, deliv.ENTERED)
                deliv.scheduler_reviewer_commit(root, scenario)
            deliv.plant(root, deliv.DAY, bluesky_posted=True)
            mp.setattr(picks, "get_game_statuses_detailed", lambda *_a: {})
            mp.setattr(picks, "check_hit", lambda *_a, **_k: "hit")
            assert deliv.check_cli(root, deliv.DAY).exit_code == 0
            b, h = pair.compare(
                f"delivery:{scenario}", root, Path(tmp) / "run",
                clock_iso=run_iso, lookback=1, result_feed={"pick": "miss"},
            )
            assert b["fold"] == {"status": "ok", "value": None}
            assert h["receipt_json"]["replay"]["blocked_date"] == deliv.ENTERED
            checked.append(f"delivery:{scenario}")

    carbon_scenarios = [
        "lost_ack_then_fallback_skip_auto",
        "lost_ack_then_fallback_skip_dated",
        "manual_run_crash_with_daemon_eod_skip",
        "live_private_preview_lock_restart",
        "private_research_fresh_pick_restart",
        "live_private_preview_lock_quarantined",
        "orchestrate_not_live_post_crash",
        "preview_overwrites_unmarked_entered_pick",
    ]
    for idx, scenario in enumerate(carbon_scenarios, start=1):
        with tempfile.TemporaryDirectory(dir=REVIEW_DIR) as tmp, pytest.MonkeyPatch.context() as mp:
            mp.setattr(socket.socket, "connect", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("no net")))
            mp.setattr(socket, "getaddrinfo", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("no net")))
            work = Path(tmp) / "work"
            (work / "data").mkdir(parents=True)
            (work / "data" / "models").symlink_to(REPO_ROOT / "data" / "models")
            mp.chdir(work)
            mp.setattr(strategy, "_mdp_cache", None)
            root = work / "picks"
            r2._seed_nine_hits(picks, root)

            auto_obs = contest_state.DecisionStreakState(
                9, True, True, "contest", "fresh", 9, True,
                observation_kind="automatic", contest_recorded_at="2027-04-18T15:00:00+00:00",
            )
            dated_obs = contest_state.DecisionStreakState(
                9, True, True, "contest", "fresh", 9, True,
                observation_kind="dated",
                dated_identity={"effective_date": r2.ENTERED, "active_streak": 9, "best_streak": 9, "saver_state": "not_earned"},
            )
            model_obs = contest_state.DecisionStreakState(9, True, True, "model", "fresh", 0, True)
            hi_df = pd.DataFrame([{**r2._slot(r2.ENTERED, 101, 0.84), "lineup": 1}])
            lo_df = pd.DataFrame([{**r2._slot(r2.ENTERED, 101, 0.55), "lineup": 1}])
            detailed_pre = {101: {"abstract": "P", "detailed": "Scheduled"}}
            tiers = [{"name": "local", "type": "local"}]

            if scenario in ("lost_ack_then_fallback_skip_auto", "lost_ack_then_fallback_skip_dated"):
                obs = auto_obs if scenario.endswith("_auto") else dated_obs
                cfg = {
                    "orchestrator": {"picks_dir": str(root), "models_dir": str(REPO_ROOT / "data" / "models"),
                                     "heartbeat_path": str(work / ".hb")},
                    "tiers": tiers,
                    "scheduler": {"pick_delivery": "dm", "entry_intent": "enter", "contest_state_required": True,
                                  "live_forward_capture_on_lock": False},
                    "health_checks": {"contest_state_expected": True, "enabled": False},
                    "bluesky": {"dm_recipient": "operator.invalid"},
                }
                assert entry_intent.check_entry_intent(cfg, include_live_gate=True).ok
                st = scheduler.SchedulerState(r2.ENTERED, r2.DELIVERY_ET.isoformat(), [], [], [], False, None, None, None)
                with mp.context() as inner:
                    inner.setattr(scheduler, "_now_et", lambda: r2.DELIVERY_ET)
                    inner.setattr(scheduler, "count_new_confirmations", lambda *_a: 1)
                    inner.setattr(orchestrator, "run_cascade", lambda *_a, **_k: (hi_df, "local"))
                    inner.setattr(contest_state, "load_decision_streak_state", lambda *_a, **_k: obs)
                    inner.setattr(picks, "get_game_statuses_detailed", lambda *_a, **_k: detailed_pre)
                    inner.setattr(picks, "attach_provenance", lambda d, **_k: d)
                    inner.setattr(scheduler, "_void_committed_games", lambda *_a: {})
                    inner.setattr(dm, "send_dm", lambda *_a: (_ for _ in ()).throw(TimeoutError("lost ack")))
                    chk = scheduler.run_single_check(r2.ENTERED, [101], {(101, "away"), (101, "home")}, cfg, 0.03, set())
                    scheduler._deliver_and_lock_pick(chk["pick_result"].daily, cfg, root, st, r2.ENTERED, "lineup", selection=chk["selection"])
                with mp.context() as inner:
                    inner.setattr(scheduler, "_now_et", lambda: r2.FALLBACK_ET)
                    inner.setattr(orchestrator, "run_cascade", lambda *_a, **_k: (lo_df, "local"))
                    inner.setattr(contest_state, "load_decision_streak_state", lambda *_a, **_k: obs)
                    inner.setattr(picks, "get_game_statuses_detailed", lambda *_a, **_k: detailed_pre)
                    ref = scheduler._refresh_pick_at_fallback_decision(cfg, r2.ENTERED, picks.load_pick(r2.ENTERED, root), 0.03, set())
                    scheduler._capture_fallback_skip(st, ref)
                    scheduler.save_state(st, root)
                scheduler._write_endofday_skip(root, r2.ENTERED, st)
            elif scenario == "manual_run_crash_with_daemon_eod_skip":
                st = scheduler.SchedulerState(
                    r2.ENTERED, "2027-04-18T12:25:00-04:00", [], [], [], False, None, None, None,
                    final_skip_candidate={"source": "mdp", "primary": {"batter_id": 660271}},
                )
                scheduler.save_state(st, root)
                with mp.context() as inner:
                    inner.setattr(predict, "run_pipeline", lambda *_a, **_k: hi_df)
                    inner.setattr(contest_state, "load_decision_streak_state", lambda *_a, **_k: auto_obs)
                    inner.setattr(picks, "get_game_statuses_detailed", lambda *_a, **_k: detailed_pre)
                    inner.setattr(picks, "attach_provenance", lambda d, **_k: d)
                    inner.setattr(posting, "should_post_now", lambda *_a: True)
                    inner.setattr(posting, "post_to_bluesky", lambda _b: (_ for _ in ()).throw(KeyboardInterrupt()))
                    CliRunner().invoke(cli.cli, ["run", "--date", r2.ENTERED, "--picks-dir", str(root)])
                scheduler._write_endofday_skip(root, r2.ENTERED, st)
            elif scenario in ("live_private_preview_lock_restart", "live_private_preview_lock_quarantined"):
                with mp.context() as inner:
                    inner.setattr(predict, "run_pipeline", lambda *_a, **_k: hi_df)
                    inner.setattr(contest_state, "load_decision_streak_state", lambda *_a, **_k: model_obs)
                    inner.setattr(picks, "get_game_statuses_detailed", lambda *_a, **_k: detailed_pre)
                    inner.setattr(picks, "attach_provenance", lambda d, **_k: d)
                    CliRunner().invoke(cli.cli, ["preview", "--date", r2.ENTERED, "--picks-dir", str(root), "--models-dir", str(work / "m")])
                cfg = {
                    "orchestrator": {"picks_dir": str(root), "models_dir": str(work / "m"), "heartbeat_path": str(work / ".hb")},
                    "tiers": tiers,
                    "scheduler": {"pick_delivery": "private", "entry_intent": "research", "contest_state_required": True,
                                  "live_forward_capture_on_lock": True},
                    "health_checks": {"contest_state_expected": True, "enabled": False},
                }
                st = scheduler.SchedulerState(r2.ENTERED, r2.DELIVERY_ET.isoformat(), [], [], [], False, None, None, None)
                with mp.context() as inner:
                    inner.setattr(scheduler, "_now_et", lambda: r2.FALLBACK_ET)
                    inner.setattr(orchestrator, "run_cascade", lambda *_a, **_k: (None, None))
                    inner.setattr(scheduler, "_void_committed_games", lambda *_a: {})
                    inner.setattr(scheduler, "_trigger_live_forward_capture_on_lock", lambda *_a: (_ for _ in ()).throw(KeyboardInterrupt()))
                    ref = scheduler._refresh_pick_at_fallback_decision(cfg, r2.ENTERED, picks.load_pick(r2.ENTERED, root), 0.03, set())
                    try:
                        scheduler._deliver_and_lock_pick(ref.daily, cfg, root, st, r2.ENTERED, "final fallback", selection=ref.selection)
                    except KeyboardInterrupt:
                        pass
                if scenario == "live_private_preview_lock_restart":
                    sched_games = [{"gamePk": 101, "gameDate": f"{r2.ENTERED}T23:10:00Z",
                                    "teams": {"away": {"team": {"name": "LAD"}}, "home": {"team": {"name": "NYY"}}}}]
                    with mp.context() as inner:
                        inner.setattr(scheduler, "_now_et", lambda: datetime(2027, 4, 18, 18, 36, tzinfo=ET))
                        inner.setattr(scheduler, "fetch_schedule", lambda d: sched_games if d == r2.ENTERED else [])
                        inner.setattr(scheduler, "count_new_confirmations", lambda *_a: 0)
                        inner.setattr(picks, "get_game_statuses_detailed", lambda *_a, **_k: detailed_pre)
                        inner.setattr(orchestrator, "run_cascade", lambda *_a, **_k: (hi_df, "local"))
                        inner.setattr(contest_state, "load_decision_streak_state",
                                      lambda *_a, **_k: (_ for _ in ()).throw(contest_state.ContestStateError("stale")))
                        inner.setattr(scheduler, "_alert_contest_state_failure", lambda *_a: None)
                        inner.setattr(scheduler, "_maybe_alert_missed_pick", lambda *_a: None)
                        inner.setattr(scheduler, "_idle_until_next_wakeup", lambda *_a: None)
                        inner.setattr(scheduler, "notify_ready", lambda: None)
                        inner.setattr(scheduler, "notify_watchdog", lambda: None)
                        scheduler.run_day(r2.ENTERED, cfg)
                else:
                    (root / r2.ENTERED / "scheduler_state.json").write_text("{torn")
                    scheduler.load_state(r2.ENTERED, root)
            elif scenario == "private_research_fresh_pick_restart":
                cfg = {
                    "orchestrator": {"picks_dir": str(root), "models_dir": str(work / "m"), "heartbeat_path": str(work / ".hb")},
                    "tiers": tiers,
                    "scheduler": {"pick_delivery": "private", "entry_intent": "research", "contest_state_required": False,
                                  "live_forward_capture_on_lock": True},
                    "health_checks": {"contest_state_expected": False, "enabled": False},
                }
                st = scheduler.SchedulerState(r2.ENTERED, r2.DELIVERY_ET.isoformat(), [], [], [], False, None, None, None)
                with mp.context() as inner:
                    inner.setattr(scheduler, "_now_et", lambda: r2.DELIVERY_ET)
                    inner.setattr(scheduler, "count_new_confirmations", lambda *_a: 1)
                    inner.setattr(orchestrator, "run_cascade", lambda *_a, **_k: (hi_df, "local"))
                    inner.setattr(contest_state, "load_decision_streak_state", lambda *_a, **_k: model_obs)
                    inner.setattr(picks, "get_game_statuses_detailed", lambda *_a, **_k: detailed_pre)
                    inner.setattr(picks, "attach_provenance", lambda d, **_k: d)
                    inner.setattr(scheduler, "_void_committed_games", lambda *_a: {})
                    inner.setattr(scheduler, "_trigger_live_forward_capture_on_lock", lambda *_a: (_ for _ in ()).throw(KeyboardInterrupt()))
                    chk = scheduler.run_single_check(r2.ENTERED, [101], {(101, "away"), (101, "home")}, cfg, 0.03, set())
                    try:
                        scheduler._deliver_and_lock_pick(chk["pick_result"].daily, cfg, root, st, r2.ENTERED, "lineup", selection=chk["selection"])
                    except KeyboardInterrupt:
                        pass
                sched_games = [{"gamePk": 101, "gameDate": f"{r2.ENTERED}T23:10:00Z",
                                "teams": {"away": {"team": {"name": "LAD"}}, "home": {"team": {"name": "NYY"}}}}]
                with mp.context() as inner:
                    inner.setattr(scheduler, "_now_et", lambda: datetime(2027, 4, 18, 18, 20, tzinfo=ET))
                    inner.setattr(scheduler, "fetch_schedule", lambda _d: sched_games)
                    inner.setattr(scheduler, "_watchdog_ping_sleep", lambda _s: (_ for _ in ()).throw(SystemExit(143)))
                    inner.setattr(scheduler, "notify_ready", lambda: None)
                    inner.setattr(scheduler, "notify_watchdog", lambda: None)
                    try:
                        scheduler.run_day(r2.ENTERED, cfg)
                    except SystemExit:
                        pass
            elif scenario == "orchestrate_not_live_post_crash":
                cfg = {"orchestrator": {"picks_dir": str(root), "models_dir": str(work / "m")}, "tiers": tiers,
                       "bluesky": {"dm_recipient": "operator.invalid"}}
                with mp.context() as inner:
                    inner.setattr(orchestrator, "run_cascade", lambda *_a, **_k: (hi_df, "local"))
                    inner.setattr(contest_state, "load_decision_streak_state", lambda *_a, **_k: model_obs)
                    inner.setattr(picks, "get_game_statuses_detailed", lambda *_a, **_k: detailed_pre)
                    inner.setattr(picks, "attach_provenance", lambda d, **_k: d)
                    inner.setattr(posting, "should_post_now", lambda *_a: True)
                    inner.setattr(posting, "post_to_bluesky", lambda _b: (_ for _ in ()).throw(KeyboardInterrupt()))
                    try:
                        orchestrator.orchestrate(Path("unused.toml"), r2.ENTERED, config=cfg)
                    except KeyboardInterrupt:
                        pass
            elif scenario == "preview_overwrites_unmarked_entered_pick":
                daily = picks.parse_daily_pick(json.dumps(r2._pick_dict(
                    r2.ENTERED, chosen_on=picks.chosen_on_record(auto_obs, required=True)
                )))
                picks.save_pick(daily, root)
                with mp.context() as inner:
                    inner.setattr(predict, "run_pipeline", lambda *_a, **_k: hi_df)
                    inner.setattr(contest_state, "load_decision_streak_state", lambda *_a, **_k: model_obs)
                    inner.setattr(picks, "get_game_statuses_detailed", lambda *_a, **_k: detailed_pre)
                    inner.setattr(picks, "attach_provenance", lambda d, **_k: d)
                    CliRunner().invoke(cli.cli, ["preview", "--date", r2.ENTERED, "--picks-dir", str(root), "--models-dir", str(work / "m")])

            with mp.context() as inner:
                inner.setattr(cli, "_now_et", lambda: r2.RUN_ET)
                inner.setattr(picks, "get_game_statuses_detailed", lambda *_a, **_k: {})
                inner.setattr(picks, "resolve_daily_slot_results", lambda *_a, **_k: {"pick": "hit"})
                picks.save_pick(picks.parse_daily_pick(json.dumps(r2._pick_dict(r2.GRADED, None, bluesky_posted=True))), root)
                CliRunner().invoke(cli.cli, ["check-results", "--date", r2.GRADED, "--picks-dir", str(root)])

            b, h = pair.compare(
                f"carbon_r2:N{idx}_{scenario}", root, Path(tmp) / "run",
                clock_iso=run_iso, lookback=8, result_feed={"pick": "miss"},
            )
            assert b["fold"] == {"status": "ok", "value": None}
            assert h["receipt_json"]["replay"]["blocked_date"] == r2.ENTERED
            checked.append(f"N{idx}:{scenario}")

    with tempfile.TemporaryDirectory(dir=REVIEW_DIR) as tmp:
        ext = Path(tmp) / "external"
        daily_decision.write_decision(
            "2027-04-16", ext, action="skip", source="mdp", delivery_status="not_applicable", scoreable=False, now=r2.RUN_ET,
        )
        root = Path(tmp) / "tpl" / "picks"
        root.mkdir(parents=True)
        picks.save_pick(picks.parse_daily_pick(json.dumps(r2._pick_dict("2027-04-01", "hit", bluesky_posted=True))), root)
        obs = contest_state.DecisionStreakState(9, True, True, "contest", "fresh", 9, True, observation_kind="automatic")
        picks.save_pick(
            picks.parse_daily_pick(json.dumps(r2._pick_dict("2027-04-16", None, chosen_on=picks.chosen_on_record(obs, required=True)))),
            root,
        )
        (root / "2027-04-16").symlink_to(ext / "2027-04-16")
        picks.save_streak(9, root, saver_available=False, season=2027)
        b, h = pair.compare(
            "carbon_r2:b2_symlink", root, Path(tmp) / "run",
            clock_iso=run_iso, lookback=8, result_feed={"pick": "hit"},
        )
        assert b["fold"] == {"status": "ok", "value": None}
        assert h["receipt_json"]["replay"]["blocked_date"] == "2027-04-16"
        checked.append("b2:symlink")

    assert len(checked) == 18
    return {"checked": len(checked), "cases": checked}


def _form(path: Path, shape: tuple[str, object]) -> None:
    kind, value = shape
    if kind == "missing":
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if kind == "json":
        path.write_text(json.dumps(value))
    elif kind == "raw":
        assert isinstance(value, bytes)
        path.write_bytes(value)
    elif kind == "directory":
        path.mkdir()
    elif kind == "dangling":
        path.symlink_to("missing-target")
    elif kind == "symlink":
        target = path.with_name("indirect.data")
        target.write_text(json.dumps(value))
        path.symlink_to(target.name)
    else:
        raise AssertionError(kind)


def _run_hostile_probes(pair: WorkerPair) -> dict:
    from tests.test_wr_c3v_corpus import dump, pick_data

    run_iso = datetime(2027, 4, 20, 2, 0, tzinfo=ET).isoformat()
    today = "2027-04-20"
    past = "2027-04-16"
    yest = "2027-04-19"
    good = {"streak": 9, "season": 2027, "saver_available": False, "updated": "2027-01-01T06:00:00+00:00"}
    scalar = [("null", None), ("array", []), ("number", 3), ("float", 0.5), ("string", "hello"), ("boolean", True)]
    special = {
        "missing": ("missing", None), "malformed": ("raw", b"{"),
        "nonutf8": ("raw", b"\xff"), "blank": ("raw", b""),
        "directory": ("directory", None), "dangling": ("dangling", None),
    }

    total = 0
    cats: dict[str, int] = {}

    def _bump(cat: str):
        nonlocal total
        total += 1
        cats[cat] = cats.get(cat, 0) + 1

    with tempfile.TemporaryDirectory(dir=REVIEW_DIR) as tmp:
        bt = Path(tmp)

        # 1. Codex + Carbon pick shapes across past, today, yesterday(lookback=1), future
        pick_shapes = special | {k: ("json", v) for k, v in scalar} | {
            "empty": ("json", {}), "pending": ("json", {"result": "suspended"}),
            "hit_missing_slots": ("json", {"result": "hit"}), "miss": ("json", {"result": "miss"}),
            "void": ("json", {"result": "void"}), "result_array": ("json", {"result": []}),
            "result_dict": ("json", {"result": {}}), "result_number": ("json", {"result": 5}),
            "result_boolean": ("json", {"result": True}), "result_null": ("json", {"result": None}),
            "symlink_pending": ("symlink", {}),
            "bom_hit": ("raw", ("\ufeff" + json.dumps(pick_data(past, "hit"))).encode("utf-8")),
        }
        for day in (past, today, yest, "2027-04-21"):
            for label, spec in pick_shapes.items():
                cdir = bt / f"p_{total}"
                tpl = cdir / "tpl" / "picks"
                tpl.mkdir(parents=True)
                _form(tpl / "streak.json", ("json", good | {"applied_dates": [today]}))
                _form(tpl / f"{day}.json", spec)
                pair.compare(
                    f"pick:{day}:{label}", tpl, cdir / "run",
                    clock_iso=run_iso, lookback=1 if day == yest else 0,
                )
                shutil.rmtree(cdir)
                _bump("pick_shapes")

            # chmod 000 on each day
            cdir = bt / f"p_perm_{total}"
            tpl = cdir / "tpl" / "picks"
            tpl.mkdir(parents=True)
            _form(tpl / "streak.json", ("json", good | {"applied_dates": [today]}))
            _form(tpl / f"{day}.json", ("json", {}))
            pair.compare(
                f"pick:{day}:chmod000", tpl, cdir / "run",
                clock_iso=run_iso, lookback=1 if day == yest else 0,
                chmod=[{"rel": f"{day}.json", "mode": 0}],
            )
            shutil.rmtree(cdir)
            _bump("pick_shapes")

        # 2. Streak shapes across saved, past-blocker, today, blocked_with_correction
        marker_values = scalar + [
            ("object", {today: True}), ("plain_date", today),
            ("mixed", [None, 3, {}, today, today]), ("past", [past]), ("trailing_space", [today + " "]),
        ]
        streak_shapes = special | {k: ("json", v) for k, v in scalar} | {
            f"marker_{k}": ("json", good | {"applied_dates": v}) for k, v in marker_values
        } | {
            "object": ("json", good), "minimal": ("json", {}),
            "foreign": ("json", good | {"season": 2026, "applied_dates": [today]}),
            "float_tag": ("json", good | {"season": 2027.0, "applied_dates": [today]}),
            "text_tag": ("json", good | {"season": "2027", "applied_dates": [today]}),
            "tagless": ("json", {"applied_dates": [today], "streak": 11}),
            "bool_tag": ("json", good | {"season": True, "applied_dates": [today]}),
            "weird_streak": ("json", good | {"streak": [], "applied_dates": [today]}),
            "symlink_to_object": ("symlink", good | {"applied_dates": [today]}),
        }
        for label, spec in streak_shapes.items():
            for scen, day, pval in (("saved", past, {"result": "void"}), ("past-blocker", past, {}), ("today", today, {})):
                cdir = bt / f"st_{total}"
                tpl = cdir / "tpl" / "picks"
                tpl.mkdir(parents=True)
                _form(tpl / "streak.json", spec)
                _form(tpl / f"{day}.json", ("json", pval))
                pair.compare(f"streak:{scen}:{label}", tpl, cdir / "run", clock_iso=run_iso)
                shutil.rmtree(cdir)
                _bump("streak_shapes")

        # 3. Decision shapes across orphan, paired, terminal
        decision_shapes = special | {k: ("json", v) for k, v in scalar} | {
            "empty": ("json", {}), "skip": ("json", {"action": "skip", "scoreable": False}),
            "delivered": ("json", {"action": "single", "scoreable": True, "delivery_status": "delivered"}),
            "odd": ("json", {"schema_version": [], "date": False, "scoreable": "true"}),
            "symlink": ("symlink", {"action": "skip", "scoreable": False}),
        }
        for label, spec in decision_shapes.items():
            for scen, pval in (("orphan", None), ("paired", {}), ("terminal", {"result": "void"})):
                cdir = bt / f"dec_{total}"
                tpl = cdir / "tpl" / "picks"
                tpl.mkdir(parents=True)
                _form(tpl / "streak.json", ("json", good))
                if pval is not None:
                    _form(tpl / f"{past}.json", ("json", pval))
                _form(tpl / past / "decision.json", spec)
                pair.compare(f"decision:{scen}:{label}", tpl, cdir / "run", clock_iso=run_iso)
                shutil.rmtree(cdir)
                _bump("decision_shapes")

        # 4. Walk-order, contended flock midnight/New Year, publish OSError, interrupt, FIFO, lost-hit, Phase 1/2 errors, stems
        for label, second_body in (("first-of-mixed", "{"), ("first-of-pending", "{}"), ("first-before-exception", "[]")):
            cdir = bt / f"ord_{total}"
            tpl = cdir / "tpl" / "picks"
            tpl.mkdir(parents=True)
            _form(tpl / "streak.json", ("json", good))
            (tpl / f"{past}.json").write_text("{}")
            (tpl / "2027-04-17.json").write_text(second_body)
            (tpl / "2027-04-21.json").write_text("{")
            pair.compare(f"order:{label}", tpl, cdir / "run", clock_iso=run_iso)
            shutil.rmtree(cdir)
            _bump("order_and_faults")

        for label, before_s, after_s, day, exp_reason in (
            ("contended_midnight", "2027-04-19T23:59:00-04:00", "2027-04-20T00:01:00-04:00", "2027-04-19",
             "no final result on the pick"),
            ("contended_new_year", "2027-12-31T23:59:00-05:00", "2028-01-01T00:01:00-05:00", "2028-01-01",
             "today's result is applied to streak.json but not yet saved on the pick"),
        ):
            cdir = bt / f"flock_{total}"
            tpl = cdir / "tpl" / "picks"
            tpl.mkdir(parents=True)
            after_dt = datetime.fromisoformat(after_s)
            _form(tpl / "streak.json", ("json", {"streak": 9, "season": after_dt.year, "applied_dates": [day]}))
            (tpl / f"{day}.json").write_text("{}")
            if before_s[:4] != after_s[:4]:
                (tpl / "2027-12-31.json").write_text("{")
            _, h = pair.compare(
                f"clock:{label}", tpl, cdir / "run",
                clock_iso=before_s, after_iso=after_s, clock_mode="contended_flock",
            )
            assert h["receipt_json"]["replay"] == {"state": "unavailable", "blocked_date": day, "reason": exp_reason}
            shutil.rmtree(cdir)
            _bump("clock_crossings")

        for c_label, c_iso, a_iso, c_mode in (
            ("utc_ahead_dec31", "2028-01-01T01:00:00+00:00", None, "fixed"),
            ("post_publish_new_year", "2027-12-31T23:59:58-05:00", "2028-01-01T00:00:02-05:00", "advance_after_publish"),
        ):
            cdir = bt / f"clk2_{total}"
            tpl = cdir / "tpl" / "picks"
            tpl.mkdir(parents=True)
            _form(tpl / "streak.json", ("json", {"streak": 12, "saver_available": False, "season": 2027, "applied_dates": []}))
            dump(tpl / "2027-12-20.json", pick_data("2027-12-20", None))
            pair.compare(f"clock:{c_label}", tpl, cdir / "run", clock_iso=c_iso, after_iso=a_iso, clock_mode=c_mode)
            shutil.rmtree(cdir)
            _bump("clock_crossings")

        for p_label, body in (("publish-fails-blocked", "{}"), ("publish-fails-saved", '{"result": "void"}')):
            cdir = bt / f"pub_{total}"
            tpl = cdir / "tpl" / "picks"
            tpl.mkdir(parents=True)
            _form(tpl / "streak.json", ("json", good))
            (tpl / f"{past}.json").write_text(body)
            _, h = pair.compare(f"fault:{p_label}", tpl, cdir / "run", clock_iso=run_iso, publish_oserror=True)
            assert h["exit_code"] == 0 and h["receipt_json"] is None
            shutil.rmtree(cdir)
            _bump("order_and_faults")

        # Interrupt, FIFO, lost-hit, Phase 1 URLError, Phase 2 save_pick OSError, missing picks_dir
        cdir = bt / f"int_{total}"
        tpl = cdir / "tpl" / "picks"
        tpl.mkdir(parents=True)
        _form(tpl / "streak.json", ("json", good))
        (tpl / f"{past}.json").write_text("{}")
        _, h = pair.compare("fault:interrupt", tpl, cdir / "run", clock_iso=run_iso, interrupt_on_read=f"{past}.json")
        assert h["exit_code"] == 1 and h["receipt_json"]["replay"] == {"state": "not_reached"}
        shutil.rmtree(cdir)
        _bump("order_and_faults")

        cdir = bt / f"fifo_{total}"
        tpl = cdir / "tpl" / "picks"
        tpl.mkdir(parents=True)
        _form(tpl / "streak.json", ("json", good))
        _, h = pair.compare(
            "fault:fifo", tpl, cdir / "run",
            clock_iso=run_iso, mkfifo=f"{past}.json", skip_direct_fold=True,
        )
        assert h["exit_code"] == 1 and h["receipt_json"]["replay"] == {"state": "not_reached"}
        shutil.rmtree(cdir)
        _bump("order_and_faults")

        cdir = bt / f"lh_{total}"
        tpl = cdir / "tpl" / "picks"
        tpl.mkdir(parents=True)
        _form(tpl / "streak.json", ("json", good))
        dump(tpl / f"{past}.json", pick_data(past, "hit"))
        _, h = pair.compare(
            "fault:lost-hit", tpl, cdir / "run",
            clock_iso=run_iso, concurrent_unlink_on_read=f"{past}.json",
        )
        assert h["receipt_json"]["replay"] == {"state": "unavailable", "blocked_date": past, "reason": "hit pick could not be loaded"}
        shutil.rmtree(cdir)
        _bump("order_and_faults")

        for err_kind in ("phase1_url_error", "phase2_save_pick_oserror"):
            cdir = bt / f"err_{total}"
            tpl = cdir / "tpl" / "picks"
            tpl.mkdir(parents=True)
            _form(tpl / "streak.json", ("json", good))
            dump(tpl / f"{past}.json", pick_data(past, None))
            dump(tpl / f"{yest}.json", pick_data(yest, "hit", slot_results={"pick": "hit"}))
            _, h = pair.compare(
                f"fault:{err_kind}", tpl, cdir / "run",
                clock_iso=run_iso, lookback=2,
                result_feed="raise_url_error" if err_kind == "phase1_url_error" else {"pick": "miss"},
                save_pick_error="disk_full" if err_kind == "phase2_save_pick_oserror" else None,
            )
            assert h["receipt_json"]["replay"] == {"state": "not_reached"}
            shutil.rmtree(cdir)
            _bump("order_and_faults")

        cdir = bt / f"abs_{total}"
        pair.compare("fault:missing_picks_dir", cdir / "absent_tpl", cdir / "run", clock_iso=run_iso)
        shutil.rmtree(cdir)
        _bump("order_and_faults")

        # 5. Noncanonical stems (including B1 trailing LF and past non-ASCII digits)
        for s_label, stem, expected_blocked in (
            ("lf_stem", f"{past}\n", True),
            ("past_arabic_digits", "2027-03-\u0661\u0666", True),
            ("past_arabic_plus_lf", "2027-03-\u0661\u0666\n", True),
            ("past_osmanya_digits", "2027-03-\U000104a1\U000104a6", True),
            ("future_arabic_month", "2027-\u0660\u0664-\u0661\u0660", False),
            ("impossible_date", "2027-02-99", True),
        ):
            cdir = bt / f"stem_{total}"
            tpl = cdir / "tpl" / "picks"
            tpl.mkdir(parents=True)
            _form(tpl / "streak.json", ("json", good))
            (tpl / f"{stem}.json").write_text("{}")
            _, h = pair.compare(f"stem:{s_label}", tpl, cdir / "run", clock_iso=run_iso, lookback=8)
            if expected_blocked:
                assert h["receipt_json"]["replay"] == {
                    "state": "unavailable", "blocked_date": stem, "reason": "no final result on the pick",
                }
                assert len(h["stderr"].splitlines()) == 1
            else:
                assert h["receipt_json"]["replay"]["state"] == "saved"
            shutil.rmtree(cdir)
            _bump("noncanonical_stems")

    return {"total_hostile": total, "by_category": cats}


def main() -> int:
    sys.path.insert(0, str(REPO_ROOT))
    with tempfile.TemporaryDirectory(dir=REVIEW_DIR) as base_extract_dir:
        base_src = _extract_verified_base_src(Path(base_extract_dir))
        pair = WorkerPair(base_src, HEAD_SRC)
        try:
            corpus = _run_corpus_630(pair)
            writers = _run_writers_18(pair)
            hostile = _run_hostile_probes(pair)
        finally:
            pair.close()

    summary = {
        "status": "PASS",
        "divergences": 0,
        "corpus_630": corpus,
        "writers_18": writers,
        "hostile_probes": hostile,
        "grand_total_runs": corpus["total"] + writers["checked"] + hostile["total_hostile"],
    }
    out_path = REVIEW_DIR / "item2_results.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        raise SystemExit(_worker_main(sys.argv[2]))
    raise SystemExit(main())
