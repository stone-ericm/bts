"""Run archived F5r1, F5r2, and contract-check-f5b scenarios against base (bf78b642) and candidate (e66cc74c)."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
SCRATCH = Path("/tmp/wr-f5b-scratch").resolve()
ARCHIVE = REPO / ".codex-review/f5-records/docs/reviews/2026-11-18-whole-range"


class BlockedError(RuntimeError):
    pass


@contextmanager
def time_bound(seconds: float = 2.0):
    def _expire(*_):
        raise BlockedError(f"blocked > {seconds}s")

    prev = signal.signal(signal.SIGALRM, _expire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prev)


def generate_cron_lines(tmp: Path) -> dict[str, list[str]]:
    sys.path.insert(0, str(REPO))
    from tests import test_entry_intent

    orig = (
        "2 9 * * * echo foreign\n"
        "0 10 * * * echo old # BTS-HETZNER\n"
        "# */15 10-23 * * * cd x && uv run bts check-pick-entered # BTS-HETZNER\n"
    )
    out: dict[str, list[str]] = {}
    for ver in ("base", "f5"):
        tree = SCRATCH / ("base" if ver == "base" else "candidate")
        script = tree / "scripts/cron-setup-hetzner.sh"
        for intent in ("enter", "research"):
            bdir = tmp / f"{ver}-{intent}"
            bdir.mkdir()
            inst = test_entry_intent.box.__wrapped__(bdir)
            inst.config.write_text(
                f'[scheduler]\nentry_intent = "{intent}"\n'
                f'pick_delivery = "{"dm" if intent == "enter" else "private"}"\n'
                + ('contest_state_required = true\n' if intent == "enter" else "")
            )
            inst.cron.write_text(orig)
            with patch.object(test_entry_intent, "SCRIPT", script):
                shown = inst.run("show")
                assert shown.returncode == 0, shown.stderr
                added = inst.run("install")
                assert added.returncode == 0, added.stderr
                lines = inst.lines()
                removed = inst.run("remove")
                assert removed.returncode == 0
                assert not any("# BTS-HETZNER" in ln for ln in (inst.lines() or []))
            out[f"{ver}_{intent}"] = lines
    return out


def run_worker(version: str, generated_path: Path) -> dict:
    sys.path[:0] = [
        str(ARCHIVE / "review-fix-f5-codex-r2"),
        str(ARCHIVE / "review-fix-f5-codex"),
    ]
    import bts
    import alert_probe
    import fallback_recovery
    import marker_health
    import morning_fifo
    import mutation_witness
    import probe_support
    import state_replay
    import state_sequences
    import support
    from bts import entry_receipt
    from bts.health import pick_entry
    from tests.test_wr_f1_wrong_game import DAY, RIGHT, WRONG, row

    bts_file = str(Path(bts.__file__).resolve())
    expected_root = (SCRATCH / ("base" if version == "base" else "candidate") / "src").resolve()
    assert Path(bts_file).is_relative_to(expected_root), (bts_file, expected_root)

    gen = json.loads(generated_path.read_text())

    def _load_line(ver: str) -> str:
        (line,) = [ln for ln in gen[f"{ver}_enter"] if " bts check-pick-entered " in ln]
        return line

    probe_support.load_line = support.load_line = alert_probe.load_line = _load_line
    alert_probe.ROOT = REPO
    support.source = state_sequences.source = marker_health.source = morning_fifo.source = lambda v: None

    summary: dict = {"version": version, "bts_file": bts_file, "checks": {}}

    with TemporaryDirectory() as tmp_str:
        root = Path(tmp_str)

        # 1. F5r1 alert_probe: closing_fixture + shared_window
        alert_probe.closing_fixture(root / "closing", version)
        summary["checks"]["closing_fixture"] = "pass"
        for variant in ("bind-wrong", "remove-entry", "late-commit"):
            alert_probe.shared_window(root / f"shared-{variant}", version, variant)
        summary["checks"]["shared_window"] = "pass"

        # 2. R1 (corrected pitch), R2 (replacement), R3 (spent early tiers + revised 10:06)
        for prev_alert, r_key in ((False, "R2_revised_replacement"), (True, "R3_revised_spent_tiers")):
            sub = root / r_key
            if version == "base":
                alert_probe.revised_window(sub, version, previous_alert=prev_alert)
                summary["checks"][r_key] = "base_pass"
            else:
                try:
                    alert_probe.revised_window(sub, version, previous_alert=prev_alert)
                except AssertionError:
                    w = probe_support.Replay(sub / f"revised-{prev_alert}", version)
                    recs = entry_receipt.discover(w.picks, DAY)
                    ten_rec = [r for r in recs if r["started_at"].startswith(f"{DAY}T10:00")][0]
                    assert ten_rec["outcome"] == "observed" and ten_rec["marker_status"] == "alerted"
                    morning = w.morning()
                    assert morning[1]["state"] == ("not_entered" if prev_alert else "entry_unverifiable")
                    summary["checks"][r_key] = f"fixed:{morning[1]['state']}"
                else:
                    raise AssertionError(f"{r_key} reproduced F5r1 bug!")

        if version == "base":
            alert_probe.corrected_pitch(root / "R1", version)
            summary["checks"]["R1_corrected_pitch"] = "base_pass"
        else:
            try:
                alert_probe.corrected_pitch(root / "R1", version)
            except AssertionError:
                w = probe_support.Replay(root / "R1/corrected-pitch", version)
                morning = w.morning()
                assert morning[1]["state"] == "entry_unverifiable" and len(morning[2]) == 1
                summary["checks"]["R1_corrected_pitch"] = "fixed:entry_unverifiable"
            else:
                raise AssertionError("R1 reproduced F5r1 bug!")

        # 3. R4: previous_date_health
        if version == "base":
            alert_probe.previous_date_health(root / "R4", version)
            summary["checks"]["R4_prior_date_health"] = "base_pass"
        else:
            try:
                alert_probe.previous_date_health(root / "R4", version)
            except AssertionError:
                w = probe_support.Replay(root / "R4/prior-health", version)
                yesterday = date(2027, 3, 23)
                alerts = pick_entry.check(w.picks, yesterday, datetime.fromisoformat(f"{DAY}T09:01:00+00:00"))
                assert [a.level for a in alerts] == ["CRITICAL"]
                assert w.read_marker()["date"] == yesterday.isoformat()
                summary["checks"]["R4_prior_date_health"] = "fixed:CRITICAL_preserved"
            else:
                raise AssertionError("R4 reproduced F5r1 bug!")

        # 4. R5 / R6: mutation_witness candidate mode
        if version == "f5":
            mutation_witness.replay("candidate")
            summary["checks"]["R6_mutation_witness"] = "pass"

        # 5. R7: state_sequences.manual_carries_tiers
        if version == "base":
            sent = state_sequences.manual_carries_tiers(root / "R7", version)
            assert sent and "1 min to submit" in sent[0]
            summary["checks"]["R7_manual_crosswalk_gap"] = "base_pass"
        else:
            try:
                state_sequences.manual_carries_tiers(root / "R7", version)
            except AssertionError:
                w = probe_support.Replay(root / "R7", version)
                assert w.read_marker()["status"] == "alerted" and "initial" in w.read_marker()["escalations"]
                summary["checks"]["R7_manual_crosswalk_gap"] = "fixed:10:00_warned"
            else:
                raise AssertionError("R7 reproduced F5r2 bug!")

        # 6. R8 & R13: surviving_mutations logic
        w8 = probe_support.Replay(root / "R8", version)
        d8 = w8.commit_at("10:05", at="09:40")
        if version == "f5":
            e8 = w8.invoke("09:45", pending=[row()])
            assert e8["receipt"]["early_cron_run"] is True and not w8.marker.exists()
        state_sequences.edit_pitch(w8, d8, "10:50")
        w8.capture([WRONG], f"{DAY}T140000Z")
        m8 = w8.invoke("10:00", pending=[row()], manual=True)
        assert "pending" in m8["calls"] and "entry is recorded in game 102" in m8["sent"][0]
        summary["checks"]["R8_manual_rechecks"] = "pass"

        if version == "f5":
            w13 = probe_support.Replay(root / "R13", version)
            w13.commit_at("10:05", at="09:00")
            w13.invoke("09:15", dm="failed")
            early_m = entry_receipt.read_early_marker(w13.picks, DAY)
            assert early_m["status"] == "dm_failed"
            w13.set_marker({"date": DAY, "status": "confirmed"})
            assert pick_entry.check(w13.picks, date.fromisoformat(DAY), datetime.fromisoformat(f"{DAY}T23:59:00-04:00")) == []
            summary["checks"]["R13_live_precedence"] = "pass"

        # 7. R9: state_replay.duplicate_clear
        sent_r9: list = []
        orig_display = state_replay.display

        def _capture_display(label, run):
            orig_display(label, run)
            if run is not None:
                sent_r9.append((label, run["sent"]))

        with patch.object(state_replay, "display", _capture_display):
            if version == "base":
                assert state_replay.duplicate_clear(root / "R9", version) == []
                summary["checks"]["R9_single_all_clear"] = "base_pass"
            else:
                try:
                    state_replay.duplicate_clear(root / "R9", version)
                except AssertionError:
                    all_dms = [m for _, s in sent_r9 for m in s]
                    assert len(all_dms) == 2
                    assert "entry is recorded in game 102" in all_dms[0]
                    assert all_dms[1].startswith("✅")
                    assert dict(sent_r9)["all-clear 10:00"] == []
                    summary["checks"]["R9_single_all_clear"] = "fixed:one_warn_one_clear_zero_at_1000"
                else:
                    raise AssertionError("R9 reproduced F5r2 duplicate all-clear!")

        # 8. F5r2 #3: slot_gained_then_lost
        if version == "base":
            m_sgtl = state_sequences.slot_gained_then_lost(root / "sgtl", version)
            assert m_sgtl[1]["state"] == "not_entered"
            summary["checks"]["F5r2_3_slot_gained_then_lost"] = "base_not_entered"
        else:
            try:
                state_sequences.slot_gained_then_lost(root / "sgtl", version)
            except AssertionError:
                w_sgtl = probe_support.Replay(root / "sgtl", version)
                m_sgtl = w_sgtl.morning()
                assert m_sgtl[0] == 1 and m_sgtl[1]["state"] == "not_entered" and len(m_sgtl[2]) == 1
                summary["checks"]["F5r2_3_slot_gained_then_lost"] = "fixed:not_entered"
            else:
                raise AssertionError("slot_gained_then_lost returned clean entered after 10:00 absence!")

        # 9. moving_double
        assert state_sequences.moving_double(root / "moving_double", version) == "entry_unverifiable"
        summary["checks"]["moving_double"] = "pass"

        # 10. R10: morning_fifo
        w10 = probe_support.Replay(root / "R10_shared", version)
        w10.commit_at("10:05", at="09:40")
        if version == "f5":
            r10 = w10.invoke("09:45")
            assert r10["receipt"]["early_cron_run"] is True
        else:
            w10.invoke("09:45", manual=True)
        fifo_pick = w10.picks / f"{DAY}.json"
        fifo_pick.unlink()
        os.mkfifo(fifo_pick)
        with time_bound(1.0):
            code10, body10, dms10 = w10.morning()
        expected10 = "entry_unverifiable" if version == "f5" else "not_entered"
        assert code10 == 1 and body10["state"] == expected10 and len(dms10) == 1
        summary["checks"]["R10_morning_fifo"] = f"pass:{expected10}"

        # 11. R11 & R12: fallback_recovery
        w11 = probe_support.Replay(root / "R11", version, fallback_recovery.CURRENT)
        w11.set_marker({})
        w11.marker.write_bytes(b'{"reason":"\xff"}')
        levels11, msgs11 = fallback_recovery.dispatch(w11, fallback_recovery.CURRENT)
        assert levels11 == ["CRITICAL"] and len(msgs11) == 1 and "UnicodeDecodeError" in msgs11[0]
        summary["checks"]["R11_live_invalid_utf8"] = "pass:CRITICAL"

        prev_d, cur_d = fallback_recovery.PREVIOUS, fallback_recovery.CURRENT
        old12 = probe_support.Replay(root / "R12", version, prev_d)
        old12.commit_at("19:10", at="18:00")
        assert state_sequences.cron(old12, version, "18:00", dm="failed")["marker"]["status"] == "dm_failed"
        cur12 = probe_support.Replay(root / "R12", version, cur_d)
        cur12.commit_at("06:10", at="04:55")
        state_sequences.cron(cur12, version, "05:00", pending=[row()])
        assert old12.invoke("18:15", manual=True, pending=[row()])["marker"]["status"] == "confirmed"
        state_sequences.cron(cur12, version, "05:15", pending=[row()])
        levels12, msgs12 = fallback_recovery.dispatch(old12, prev_d)
        assert levels12 == [] and msgs12 == []
        summary["checks"]["R12_year_boundary_recovery"] = "pass:quiet"

    return summary


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] in ("base", "f5"):
        res = run_worker(sys.argv[1], Path(sys.argv[2]))
        print(json.dumps(res))
        return

    with TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        gen = generate_cron_lines(tmp)
        gen_path = OUT_DIR / "generated_crons.json"
        gen_path.write_text(json.dumps(gen, indent=2) + "\n")

        results = {}
        for ver, sub in (("base", "base/src"), ("f5", "candidate/src")):
            proc = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), ver, str(gen_path)],
                cwd=str(REPO),
                env={**os.environ, "PYTHONPATH": f"{SCRATCH / sub}:{REPO}"},
                text=True,
                capture_output=True,
                timeout=30,
            )
            if proc.returncode != 0:
                sys.stderr.write(proc.stderr)
                raise SystemExit(proc.returncode)
            line = [ln for ln in proc.stdout.splitlines() if ln.startswith('{"version":')][-1]
            results[ver] = json.loads(line)
            print(ver, "->", results[ver]["bts_file"])
            for k, v in results[ver]["checks"].items():
                print(f"  {k}: {v}")

        (OUT_DIR / "log_01_archived_and_r_rows.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
