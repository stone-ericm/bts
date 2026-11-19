"""Adversarial review Item 1: Base (88b69a16) vs HEAD (cb3bc37c) equivalence probe.

Runs isolated worker subprocesses (one importing only archived 88b69a16 src/,
one importing only HEAD src/) side by side across:
  1. All 630 histories (90 shapes x 7 mixes) from the round-1 corpus
  2. All 9 real-writer sequences in test_wr_c3v_delivery.py (4 r1 + 5 r2 Codex)
  3. All 9 real-writer/symlink states in test_wr_c3_r2_entered_pick_blockers.py (N1..N8 + b2)
  4. Adversarial state hunt: missing/corrupt/non-object/foreign/float/bool streak.json,
     JSON array/null/int/float/bool/string/BOM/non-UTF8/truncated/chmod-000/dir/broken-symlink
     pick files, concurrent unlink before load_pick, unhashable result values, non-date and
     Unicode-digit filenames, clock crossings across midnight ET, New Year ET, UTC/ET skew,
     and post-lock New Year rollover, plus Phase 1/Phase 2 exceptions.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import urllib.error
from zoneinfo import ZoneInfo


ET = ZoneInfo("America/New_York")
PINNED_UTC = datetime(2027, 4, 20, 6, 0, 0, tzinfo=timezone.utc)
REVIEW_DIR = Path(__file__).resolve().parent
REPO_ROOT = REVIEW_DIR.parents[1]
BASE_SRC = REVIEW_DIR / "base_88b69a16" / "src"
HEAD_SRC = REPO_ROOT / "src"


@contextmanager
def _alarm(seconds: int = 20):
    class _Timeout(BaseException):
        pass

    prev = signal.signal(signal.SIGALRM, lambda *_a: (_ for _ in ()).throw(_Timeout("probe timeout")))
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev)


def _worker_main(src_dir_str: str) -> int:
    src_dir = Path(src_dir_str).resolve()
    # Strip repo root / existing src entries so only src_dir provides `bts`.
    cleaned = []
    for entry in sys.path:
        p = Path(entry or ".").resolve()
        if p in (REPO_ROOT, HEAD_SRC, REVIEW_DIR):
            continue
        cleaned.append(entry)
    sys.path[:] = [str(src_dir), *cleaned]

    # Block all network access in the worker process.
    def _no_net(*args, **kwargs):
        raise RuntimeError(f"Network blocked in worker: {args}")

    socket.socket.connect = _no_net  # type: ignore[assignment]
    socket.getaddrinfo = _no_net  # type: ignore[assignment]

    from click.testing import CliRunner
    from bts import cli as cli_mod
    from bts import picks as picks_mod
    from bts import reconcile_receipt as rr_mod

    assert Path(picks_mod.__file__).resolve().is_relative_to(src_dir), picks_mod.__file__
    assert Path(cli_mod.__file__).resolve().is_relative_to(src_dir), cli_mod.__file__
    assert Path(rr_mod.__file__).resolve().is_relative_to(src_dir), rr_mod.__file__

    class PinnedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return PINNED_UTC.astimezone(tz) if tz is not None else PINNED_UTC.replace(tzinfo=None)

    real_datetime = picks_mod.datetime
    real_git_sha = picks_mod._git_head_sha
    real_resolve = picks_mod.resolve_daily_slot_results
    real_save_pick = picks_mod.save_pick
    real_scoring_lock = picks_mod.scoring_lock
    real_rr_class = rr_mod.ReconcileReceipt
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
        chmod_entries = spec.get("chmod", [])
        applied_chmods: list[Path] = []

        try:
            with _alarm(15):
                for item in chmod_entries:
                    for base_p in (fold_picks, cli_picks):
                        target = base_p / item["rel"]
                        target.chmod(item["mode"])
                        applied_chmods.append(target)

                picks_mod.datetime = PinnedDatetime
                picks_mod._git_head_sha = lambda *_a, **_k: "88b69a16"

                feed_mode = spec.get("result_feed")
                if feed_mode is None:
                    def _unexpected_resolve(*_a, **_k):
                        raise AssertionError("Unexpected resolve_daily_slot_results call")
                    picks_mod.resolve_daily_slot_results = _unexpected_resolve
                elif feed_mode == "raise_url_error":
                    def _fail_resolve(*_a, **_k):
                        raise urllib.error.URLError("synthetic feed error")
                    picks_mod.resolve_daily_slot_results = _fail_resolve
                else:
                    picks_mod.resolve_daily_slot_results = lambda *_a, _f=feed_mode, **_k: dict(_f)

                if spec.get("save_pick_error") == "disk_full":
                    picks_mod.save_pick = lambda *_a, **_k: (_ for _ in ()).throw(OSError("disk full"))
                else:
                    picks_mod.save_pick = real_save_pick

                unlink_name = spec.get("concurrent_unlink_on_read")
                if unlink_name:
                    seen_unlink: set[str] = set()

                    def _unlink_after_first_read(self_path: Path, *a, **k):
                        txt = real_read_text(self_path, *a, **k)
                        key = str(self_path)
                        if self_path.name == unlink_name and key not in seen_unlink:
                            seen_unlink.add(key)
                            self_path.unlink()
                        return txt

                    Path.read_text = _unlink_after_first_read  # type: ignore[method-assign]
                else:
                    Path.read_text = real_read_text  # type: ignore[method-assign]

                clock_mode = spec.get("clock_mode", "fixed")
                current_now = datetime.fromisoformat(spec["clock_iso"])
                after_iso = spec.get("after_iso")
                after_now = datetime.fromisoformat(after_iso) if after_iso else current_now

                if clock_mode == "advance_on_lock":
                    @contextmanager
                    def _advancing_lock(d):
                        nonlocal current_now
                        with real_scoring_lock(d):
                            current_now = after_now
                            yield

                    picks_mod.scoring_lock = _advancing_lock
                else:
                    picks_mod.scoring_lock = real_scoring_lock

                # 1. Run _replay_season_streak directly on fold_picks
                fold_day = after_now.astimezone(ET).date() if clock_mode == "advance_on_lock" else current_now.astimezone(ET).date()
                try:
                    val = picks_mod._replay_season_streak(fold_picks, fold_day.year, fold_day.isoformat())
                    fold_out = {"status": "ok", "value": list(val) if val is not None else None}
                except BaseException as exc:
                    fold_out = {"status": "exc", "type": type(exc).__name__, "args": [str(x) for x in exc.args]}

                # Reset clock before CLI run
                current_now = datetime.fromisoformat(spec["clock_iso"])

                def _make_receipt(**kw):
                    rec = real_rr_class(clock=lambda: current_now, lookback_days=kw["lookback_days"])
                    rec.run_id = "a" * 32
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

                res = CliRunner().invoke(
                    cli_mod.cli,
                    ["reconcile", "--picks-dir", str(cli_picks), "--lookback", str(spec.get("lookback", 0))],
                )

                sp = cli_picks / "streak.json"
                streak_exists = sp.exists()
                streak_hex = sp.read_bytes().hex() if streak_exists and sp.is_file() else None

                pick_hashes = {}
                if cli_picks.exists():
                    for child in sorted(cli_picks.iterdir()):
                        if child.name == ".scoring.lock":
                            continue
                        if child.is_file() and not child.is_symlink():
                            try:
                                pick_hashes[child.name] = hashlib.sha256(child.read_bytes()).hexdigest()
                            except OSError as exc:
                                pick_hashes[child.name] = f"err:{type(exc).__name__}"

                rec_root = rr_mod.receipts_root(cli_picks)
                rec_files = sorted(rec_root.glob("*/*.json")) if rec_root.exists() else []
                rec_hex = rec_files[0].read_bytes().hex() if len(rec_files) == 1 else None
                rec_json = json.loads(bytes.fromhex(rec_hex)) if rec_hex else None

                exc_sig = None
                if res.exception is not None:
                    exc_sig = {"type": type(res.exception).__name__, "args": [str(x) for x in res.exception.args]}

                payload = {
                    "fold": fold_out,
                    "exit_code": res.exit_code,
                    "exc": exc_sig,
                    "stdout": res.stdout,
                    "stderr": res.stderr,
                    "streak_exists": streak_exists,
                    "streak_hex": streak_hex,
                    "pick_hashes": pick_hashes,
                    "receipt_hex": rec_hex,
                    "receipt_json": rec_json,
                }
        finally:
            picks_mod.datetime = real_datetime
            picks_mod._git_head_sha = real_git_sha
            picks_mod.resolve_daily_slot_results = real_resolve
            picks_mod.save_pick = real_save_pick
            picks_mod.scoring_lock = real_scoring_lock
            rr_mod.ReconcileReceipt = real_rr_class  # type: ignore[misc]
            Path.read_text = real_read_text  # type: ignore[method-assign]
            for p in reversed(applied_chmods):
                try:
                    if p.exists():
                        p.chmod(0o700 if p.is_dir() else 0o600)
                except OSError:
                    pass

        sys.stdout.write(json.dumps(payload) + "\n")
        sys.stdout.flush()
    return 0


class WorkerPair:
    def __init__(self):
        env = dict(os.environ, TZ="America/New_York", PYTHONDONTWRITEBYTECODE="1")
        self.base = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--worker", str(BASE_SRC)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
        )
        self.head = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--worker", str(HEAD_SRC)],
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
        if b["fold"] != h["fold"]:
            diffs.append(f"fold mismatch: base={b['fold']} head={h['fold']}")
        if b["exit_code"] != h["exit_code"]:
            diffs.append(f"exit_code mismatch: base={b['exit_code']} head={h['exit_code']}")
        if b["exc"] != h["exc"]:
            diffs.append(f"exc mismatch: base={b['exc']} head={h['exc']}")
        if b["streak_exists"] != h["streak_exists"]:
            diffs.append(f"streak_exists mismatch: base={b['streak_exists']} head={h['streak_exists']}")
        if b["streak_hex"] != h["streak_hex"]:
            diffs.append(f"streak.json bytes mismatch: base={b['streak_hex']} head={h['streak_hex']}")
        if b["pick_hashes"] != h["pick_hashes"]:
            diffs.append(f"pick_hashes mismatch: base={b['pick_hashes']} head={h['pick_hashes']}")

        b_rec = b["receipt_json"]
        h_rec = h["receipt_json"]
        if b_rec is not None and b_rec["replay"]["state"] in ("saved", "not_reached"):
            if b["stdout"] != h["stdout"]:
                diffs.append(f"stdout mismatch on {b_rec['replay']['state']}: {b['stdout']!r} vs {h['stdout']!r}")
            if b["stderr"] != h["stderr"]:
                diffs.append(f"stderr mismatch on {b_rec['replay']['state']}: {b['stderr']!r} vs {h['stderr']!r}")
            if b["receipt_hex"] != h["receipt_hex"]:
                diffs.append(f"receipt bytes mismatch on {b_rec['replay']['state']}")
        elif b_rec is not None and b_rec["replay"]["state"] == "unavailable":
            rep = h_rec["replay"] if h_rec else {}
            if rep.get("state") != "unavailable" or "blocked_date" not in rep or "reason" not in rep:
                diffs.append(f"head receipt missing blocker detail: {rep}")
            expected_err = (
                f"Season replay unavailable for {rep.get('blocked_date')} ({rep.get('reason')}); "
                "streak retained, not recalculated.\n"
            )
            if h["stderr"] != expected_err:
                diffs.append(f"head stderr mismatch: {h['stderr']!r} vs {expected_err!r}")
            if "Streak recalculated" in h["stdout"]:
                diffs.append(f"head printed recalculated on blocked replay: {h['stdout']!r}")
            if b["exc"] is None and "Streak retained:" not in h["stdout"]:
                diffs.append(f"head missing Streak retained on blocked replay: {h['stdout']!r}")

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


def verify_base_archive_and_oracle() -> dict:
    changed = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "diff", "--name-only", "88b69a16", "HEAD", "--", "src"],
        text=True, timeout=10,
    ).strip().splitlines()
    assert sorted(changed) == ["src/bts/cli.py", "src/bts/picks.py", "src/bts/reconcile_receipt.py"], changed

    for rel in ("src/bts/cli.py", "src/bts/picks.py", "src/bts/reconcile_receipt.py"):
        git_bytes = subprocess.check_output(["git", "-C", str(REPO_ROOT), "show", f"88b69a16:{rel}"], timeout=10)
        disk_bytes = (REVIEW_DIR / "base_88b69a16" / rel).read_bytes()
        assert git_bytes == disk_bytes, f"Archive mismatch on {rel}"

    # Verify corpus shape parity between c3/test_wr_c3_corpus.py and tests/test_wr_c3v_corpus.py
    sys.path.insert(0, str(REPO_ROOT))
    from tests import test_wr_c3v_corpus as new_corpus
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_old_c3_corpus", REVIEW_DIR / "c3" / "test_wr_c3_corpus.py"
    )
    assert spec and spec.loader
    # Read names directly from source AST so we don't import nonexistent test_wr_c3_reconcile
    import ast
    old_tree = ast.parse((REVIEW_DIR / "c3" / "test_wr_c3_corpus.py").read_text())
    old_names = [
        node.args[0].value
        for node in ast.walk(old_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Case"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    ]
    new_names = [c.name for c in new_corpus.cases()]
    assert old_names == new_names and len(new_names) == 90, (len(old_names), len(new_names))

    return {
        "changed_src_files": changed,
        "archive_verified": True,
        "corpus_shapes": len(new_names),
        "corpus_mixes": len(new_corpus.ORDINARY),
    }


def run_corpus_630(pair: WorkerPair) -> dict:
    from tests import test_wr_c3v_corpus as corpus

    counts = {"total": 0, "saved": 0, "unavailable": 0, "raised": 0}
    clock_iso = datetime(2026, 8, 20, 2, 0, tzinfo=ET).isoformat()
    with tempfile.TemporaryDirectory() as tmp:
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
    assert counts["total"] == 630
    return counts


def run_writer_sequences_18(pair: WorkerPair) -> dict:
    """Run all 9 sequences from test_wr_c3v_delivery.py + all 9 states from test_wr_c3_r2_entered_pick_blockers.py."""
    import pytest
    from bts import cli, contest_state, daily_decision, dm, entry_intent, orchestrator, picks, posting, scheduler, strategy
    from bts.model import predict
    from click.testing import CliRunner
    from tests import test_wr_c3v_delivery as deliv
    from tests import test_wr_c3_r2_entered_pick_blockers as r2

    import pandas as pd

    checked = []
    run_iso = datetime(2027, 4, 20, 2, 0, tzinfo=ET).isoformat()

    # Part A: 9 scenarios from test_wr_c3v_delivery.py
    r1_r2_codex = [
        "lost_ack", "skip_conflict", "private_crash", "manual_run_crash",
        "lost-ack-eod", "private-preview-restart", "private-preview-quarantine",
        "scheduler-preview-ack", "orchestrator-post-interrupted",
    ]
    for scenario in r1_r2_codex:
        with tempfile.TemporaryDirectory() as tmp, pytest.MonkeyPatch.context() as mp:
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
            scored = deliv.check_cli(root, deliv.DAY)
            assert scored.exit_code == 0

            b, h = pair.compare(
                f"delivery:{scenario}", root, Path(tmp) / "run",
                clock_iso=run_iso, lookback=1, result_feed={"pick": "miss"},
            )
            assert b["fold"] == {"status": "ok", "value": None}
            assert h["receipt_json"]["replay"] == {
                "state": "unavailable", "blocked_date": deliv.ENTERED, "reason": "no final result on the pick",
            }
            checked.append(f"delivery:{scenario}")

    # Part B: N1..N8 + b2 from test_wr_c3_r2_entered_pick_blockers.py
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
        with tempfile.TemporaryDirectory() as tmp, pytest.MonkeyPatch.context() as mp:
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
            assert h["receipt_json"]["replay"] == {
                "state": "unavailable", "blocked_date": r2.ENTERED, "reason": "no final result on the pick",
            }
            checked.append(f"N{idx}:{scenario}")

    # b2 symlinked decision directory
    with tempfile.TemporaryDirectory() as tmp:
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


def run_adversarial_hunt(pair: WorkerPair) -> dict:
    """Exhaustive hunt across weird pick files, streak.json states, markers, and clock boundaries."""
    from tests.test_wr_c3v_corpus import dump, pick_data

    run_iso = datetime(2027, 4, 20, 2, 0, tzinfo=ET).isoformat()
    today = "2027-04-20"
    past = "2027-04-16"
    yest = "2027-04-19"
    total = 0
    categories: dict[str, int] = {}

    def bump(cat: str):
        nonlocal total
        total += 1
        categories[cat] = categories.get(cat, 0) + 1

    with tempfile.TemporaryDirectory() as tmp:
        base_tmp = Path(tmp)

        # 1. Missing and corrupt streak.json x {completed replay, blocked replay, blocked+correction, completed+correction}
        streak_variants = [
            ("missing", None),
            ("corrupt_json", b"{corrupt"),
            ("empty_bytes", b""),
            ("invalid_utf8", b"\xff\xfe"),
            ("json_array", b"[]"),
            ("json_null", b"null"),
            ("json_number", b"42"),
            ("json_bool", b"true"),
            ("json_string", b'"hello"'),
            ("season_bool_true", json.dumps({"streak": 8, "saver_available": False, "season": True}).encode()),
            ("season_float", json.dumps({"streak": 8, "saver_available": False, "season": 2027.0}).encode()),
            ("season_str", json.dumps({"streak": 8, "saver_available": False, "season": "2027"}).encode()),
            ("season_foreign", json.dumps({"streak": 8, "saver_available": False, "season": 2026, "applied_dates": [today]}).encode()),
            ("season_tagless", json.dumps({"streak": 8, "saver_available": False, "applied_dates": [today]}).encode()),
            ("markers_non_list", json.dumps({"streak": 8, "season": 2027, "applied_dates": "not-a-list"}).encode()),
            ("markers_mixed_types", json.dumps({"streak": 8, "season": 2027, "applied_dates": [123, None, {}, today]}).encode()),
        ]
        for s_name, s_bytes in streak_variants:
            for mode in ("completed", "blocked_past", "blocked_today_applied", "blocked_with_correction"):
                cdir = base_tmp / f"s_{total}"
                tpl = cdir / "tpl" / "picks"
                tpl.mkdir(parents=True)
                if s_bytes is not None:
                    (tpl / "streak.json").write_bytes(s_bytes)
                dump(tpl / "2027-04-01.json", pick_data("2027-04-01", "hit"))
                if mode == "blocked_past":
                    dump(tpl / f"{past}.json", pick_data(past, None))
                elif mode == "blocked_today_applied":
                    dump(tpl / f"{today}.json", pick_data(today, None))
                elif mode == "blocked_with_correction":
                    dump(tpl / f"{past}.json", pick_data(past, None))
                    dump(tpl / f"{yest}.json", pick_data(yest, "hit", slot_results={"pick": "hit"}))
                pair.compare(
                    f"streak:{s_name}:{mode}", tpl, cdir / "run",
                    clock_iso=run_iso,
                    lookback=2 if mode == "blocked_with_correction" else 0,
                    result_feed={"pick": "miss"} if mode == "blocked_with_correction" else None,
                )
                shutil.rmtree(cdir)
                bump("streak_shapes")

        # 2. Pick file JSON types (array, null, int, float, bool, string, BOM, non-UTF8,chmod 000, dir, broken symlink)
        # tested at past, today, AND future (> today)
        pick_payloads = [
            ("array_empty", "[]"),
            ("array_obj", json.dumps([pick_data(past, "hit")])),
            ("null", "null"),
            ("int_zero", "0"),
            ("int_pos", "42"),
            ("float", "3.14"),
            ("bool_true", "true"),
            ("bool_false", "false"),
            ("string", '"hit"'),
            ("truncated", "{"),
            ("empty", ""),
            ("bom", "\ufeff" + json.dumps(pick_data(past, "hit"))),
            ("res_list", json.dumps({"result": []})),
            ("res_dict", json.dumps({"result": {}})),
            ("res_bool", json.dumps({"result": True})),
            ("res_int", json.dumps({"result": 1})),
            ("hit_missing_pick", json.dumps({"result": "hit"})),
            ("hit_null_pick", json.dumps({"date": past, "run_time": "t", "pick": None, "double_down": None, "runner_up": None, "result": "hit"})),
        ]
        for p_name, payload in pick_payloads:
            for target_day in (past, today, "2027-04-21"):
                cdir = base_tmp / f"p_{total}"
                tpl = cdir / "tpl" / "picks"
                tpl.mkdir(parents=True)
                dump(tpl / "streak.json", {"streak": 5, "saver_available": True, "season": 2027, "applied_dates": [today]})
                (tpl / f"{target_day}.json").write_text(payload)
                pair.compare(f"pick:{p_name}:{target_day}", tpl, cdir / "run", clock_iso=run_iso)
                shutil.rmtree(cdir)
                bump("pick_shapes")

        # Special filesystem pick shapes: chmod 000, directory, broken symlink, concurrent unlink on hit
        for special in ("chmod_000", "directory", "broken_symlink", "concurrent_unlink_hit", "impossible_iso_date", "unicode_digits"):
            cdir = base_tmp / f"fs_{total}"
            tpl = cdir / "tpl" / "picks"
            tpl.mkdir(parents=True)
            dump(tpl / "streak.json", {"streak": 5, "saver_available": True, "season": 2027, "applied_dates": []})
            extra_spec: dict = {}
            if special == "chmod_000":
                dump(tpl / f"{past}.json", pick_data(past, "hit"))
                extra_spec["chmod"] = [{"rel": f"{past}.json", "mode": 0}]
            elif special == "directory":
                (tpl / f"{past}.json").mkdir()
            elif special == "broken_symlink":
                (tpl / f"{past}.json").symlink_to(tpl / "nonexistent_target.json")
            elif special == "concurrent_unlink_hit":
                dump(tpl / f"{past}.json", pick_data(past, "hit"))
                extra_spec["concurrent_unlink_on_read"] = f"{past}.json"
            elif special == "impossible_iso_date":
                (tpl / "2027-02-99.json").write_text("{}")
            elif special == "unicode_digits":
                # Python re \d matches Arabic-Indic digits too; verify base and HEAD match identically!
                (tpl / "2027-٠٤-١٠.json").write_text("{}")
            b, h = pair.compare(f"special:{special}", tpl, cdir / "run", clock_iso=run_iso, **extra_spec)
            if special == "concurrent_unlink_hit":
                assert h["receipt_json"]["replay"] == {
                    "state": "unavailable", "blocked_date": past, "reason": "hit pick could not be loaded",
                }
            shutil.rmtree(cdir)
            bump("fs_special")

        # 3. Nonexistent picks_dir
        cdir = base_tmp / f"missing_dir_{total}"
        pair.compare("missing_picks_dir", cdir / "absent_tpl", cdir / "run", clock_iso=run_iso)
        shutil.rmtree(cdir)
        bump("fs_special")

        # 4. Clock crossings (midnight ET, New Year ET, UTC/ET skew, post-publish New Year rollover)
        clock_cases = [
            ("midnight_completed", "2027-04-19T23:59:58-04:00", "2027-04-20T00:00:02-04:00", "advance_on_lock", "hit", []),
            ("midnight_blocked", "2027-04-19T23:59:58-04:00", "2027-04-20T00:00:02-04:00", "advance_on_lock", None, []),
            ("new_year_completed", "2027-12-31T23:59:58-05:00", "2028-01-01T00:00:02-05:00", "advance_on_lock", "hit", []),
            ("new_year_blocked_applied", "2027-12-31T23:59:58-05:00", "2028-01-01T00:00:02-05:00", "advance_on_lock", None, ["2028-01-01"]),
            ("utc_ahead_of_et_dec31", "2028-01-01T01:00:00+00:00", None, "fixed", None, []),
            ("post_publish_new_year", "2027-12-31T23:59:58-05:00", "2028-01-01T00:00:02-05:00", "advance_after_publish", None, []),
        ]
        for c_name, start_s, after_s, c_mode, res_val, markers in clock_cases:
            cdir = base_tmp / f"clk_{total}"
            tpl = cdir / "tpl" / "picks"
            tpl.mkdir(parents=True)
            dump(tpl / "streak.json", {"streak": 12, "saver_available": False, "season": 2027, "applied_dates": markers})
            if "new_year" in c_name or "dec31" in c_name:
                if c_mode == "advance_on_lock":
                    (tpl / "2027-12-30.json").write_text("{corrupt_old_season")
                    if res_val is None:
                        dump(tpl / "2028-01-01.json", pick_data("2028-01-01", None))
                    else:
                        dump(tpl / "2028-01-01.json", pick_data("2028-01-01", res_val))
                else:
                    dump(tpl / "2027-12-20.json", pick_data("2027-12-20", None))
            else:
                dump(tpl / "2027-04-19.json", pick_data("2027-04-19", res_val))
            pair.compare(
                f"clock:{c_name}", tpl, cdir / "run",
                clock_iso=start_s, after_iso=after_s, clock_mode=c_mode,
            )
            shutil.rmtree(cdir)
            bump("clock_boundaries")

        # 5. Replay not reached due to Phase 1 URLError and Phase 2 save_pick OSError
        for err_kind in ("phase1_url_error", "phase2_save_pick_oserror"):
            cdir = base_tmp / f"err_{total}"
            tpl = cdir / "tpl" / "picks"
            tpl.mkdir(parents=True)
            dump(tpl / "streak.json", {"streak": 9, "saver_available": True, "season": 2027, "applied_dates": []})
            dump(tpl / f"{past}.json", pick_data(past, None))
            dump(tpl / f"{yest}.json", pick_data(yest, "hit", slot_results={"pick": "hit"}))
            b, h = pair.compare(
                f"not_reached:{err_kind}", tpl, cdir / "run",
                clock_iso=run_iso, lookback=2,
                result_feed="raise_url_error" if err_kind == "phase1_url_error" else {"pick": "miss"},
                save_pick_error="disk_full" if err_kind == "phase2_save_pick_oserror" else None,
            )
            assert b["receipt_json"]["replay"] == {"state": "not_reached"}
            assert h["receipt_json"]["replay"] == {"state": "not_reached"}
            shutil.rmtree(cdir)
            bump("not_reached")

    return {"total_adversarial_cases": total, "by_category": categories}


def main() -> int:
    oracle_info = verify_base_archive_and_oracle()
    pair = WorkerPair()
    try:
        corpus_info = run_corpus_630(pair)
        writers_info = run_writer_sequences_18(pair)
        hunt_info = run_adversarial_hunt(pair)
    finally:
        pair.close()

    summary = {
        "status": "PASS",
        "divergences": 0,
        "oracle": oracle_info,
        "corpus_630": corpus_info,
        "writers_18": writers_info,
        "adversarial_hunt": hunt_info,
    }
    out_path = REVIEW_DIR / "item1_equivalence_results.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        raise SystemExit(_worker_main(sys.argv[2]))
    raise SystemExit(main())
