"""Item 3 & Item 6 probe: real run_all_checks delivery, acknowledgement receipts, dedup, near-limit digest, transport faults, aliases, and interrupts."""

from __future__ import annotations

import datetime as dt
import json
import os
import signal
import sys
import unicodedata
from collections import namedtuple
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest
from click.testing import CliRunner

import bts
from bts import cli, contest_state
from bts.health import alert, contest_state as health, runner

ET = ZoneInfo("America/New_York")
REAL_DT = dt.datetime
DAY = REAL_DT(2027, 7, 15, 22, 15, tzinfo=ET)
INCIDENT = "saver_state:dated_used_mismatch"
MAIN_STATUS = "health_dm_delivery_status.json"
MISMATCH_STATUS = MAIN_STATUS + ".dated_used_mismatch.json"


class DeliveryBlocked(BaseException):
    pass


@contextmanager
def step_deadline(seconds: int = 5):
    def _timeout(*_):
        raise DeliveryBlocked("delivery or receipt step blocked")

    prev = signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev)


def make_clock(fixed_now: dt.datetime):
    class Clock(REAL_DT):
        @classmethod
        def now(cls, tz=None):
            return fixed_now.astimezone(tz) if tz else fixed_now.replace(tzinfo=None)
    return Clock


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def cli_dated(picks: Path, now: dt.datetime, saver: str = "used", streak: int = 14) -> Path:
    clk = make_clock(now)
    with patch.object(dt, "datetime", clk), patch.object(cli, "datetime", clk):
        res = CliRunner().invoke(cli.cli, [
            "set-contest-streak",
            "--date", now.astimezone(ET).date().isoformat(),
            "--streak", str(streak),
            "--best-streak", str(max(streak, 15)),
            "--saver", saver,
            "--picks-dir", str(picks),
        ])
    assert res.exit_code == 0, res.output
    return picks / "account_state" / "contest_streak.manual.json"


def set_flag(picks: Path, state: str, season: int = 2027) -> Path:
    p = picks / "account_state" / "saver_state.json"
    write_json(p, {"season": season, "state": state})
    return p


def run_real_eod(
    monkeypatch,
    picks: Path,
    *,
    slate: dt.date,
    wall: dt.datetime,
    recipient: str | None = "ops.bsky.social",
    transport=None,
    overrides: dict | None = None,
    extra_alerts: list[alert.Alert] | None = None,
):
    """Execute real runner.run_all_checks with real _safe_run across local checks."""
    models = picks.parent / "models"
    models.mkdir(parents=True, exist_ok=True)
    if not (picks / "streak.json").exists():
        write_json(picks / "streak.json", {"season": slate.year, "streak": 6, "saver_available": True})
    (models / f"blend_{(slate + dt.timedelta(days=1)).isoformat()}.pkl").write_text("", encoding="utf-8")

    _Usage = namedtuple("_Usage", ["total", "used", "free"])
    clean_usage = _Usage(total=100 * 1024 ** 3, used=30 * 1024 ** 3, free=70 * 1024 ** 3)

    clk = make_clock(wall)
    sent = []

    def send_wrapper(to, body):
        sent.append((to, body))
        return transport(to, body) if transport is not None else f"msg-{len(sent)}"

    with monkeypatch.context() as mp:
        for mod in (dt, contest_state, alert, health, runner):
            mp.setattr(mod, "datetime", clk)
        mp.setattr(alert, "send_dm", send_wrapper)
        mp.setattr("bts.health.disk_fill.shutil.disk_usage", lambda _: clean_usage)
        mp.setattr(runner.realized_calibration, "_current_deploy_iso", lambda _: None)
        mp.setattr(runner.tail_policy, "check", lambda **_: [])
        mp.setattr(runner.park_drag_freshness, "check", lambda *a, **k: [])
        mp.setattr(runner.unit_drift, "check", lambda **_: [])
        mp.setattr(runner.backup_freshness, "check", lambda *a, **k: [])
        if extra_alerts is not None:
            mp.setattr(runner.calibration, "check", lambda *a, **k: list(extra_alerts))
        with step_deadline(5):
            alerts_out = runner.run_all_checks(
                picks_dir=picks,
                models_dir=models,
                dm_recipient=recipient,
                today=slate,
                thresholds_overrides=overrides,
                contest_state_expected=True,
                live_forward_capture_enabled=False,
                shadow_model_enabled=False,
            )
    return alerts_out, sent


def test_digest_first_unchanged_second_warn_only_and_h6_resurface(tmp_path, monkeypatch):
    picks = tmp_path / "data" / "picks"
    cli_dated(picks, DAY, saver="used", streak=14)
    set_flag(picks, "used", 2027)
    crit = alert.Alert("CRITICAL", "pick_entry", "Unentered pick")
    warn_other = alert.Alert("WARN", "backup_freshness", "Backup stale")

    base_alerts, base_sends = run_real_eod(
        monkeypatch, picks, slate=DAY.date(), wall=DAY, extra_alerts=[crit, warn_other],
    )
    assert len(base_sends) == 1
    hs = tmp_path / "data" / "health_state"
    base_main_bytes = (hs / MAIN_STATUS).read_bytes()
    assert not (hs / MISMATCH_STATUS).exists()

    for f in hs.iterdir():
        f.unlink()
    set_flag(picks, "active", 2027)
    cand_alerts, cand_sends = run_real_eod(
        monkeypatch, picks, slate=DAY.date(), wall=DAY, extra_alerts=[crit, warn_other],
    )
    assert len(cand_sends) == 2
    assert cand_sends[0] == base_sends[0]
    assert (hs / MAIN_STATUS).read_bytes() == base_main_bytes
    assert cand_sends[1][1].startswith("BTS health WARN attention:\n- [saver_state] Streak Saver on 2027-07-15 ET.")
    assert "CRITICAL" not in cand_sends[1][1]

    mm_rec = json.loads((hs / MISMATCH_STATUS).read_text())
    assert set(mm_rec.keys()) == {"schema_version", "status", "sent_day", "sent_sources", "message_id", "body_sha256"}
    assert mm_rec["status"] == "sent"
    assert mm_rec["message_id"] == "msg-2"
    assert mm_rec["sent_day"] == "2027-07-15"
    assert mm_rec["sent_sources"] == [f"WARN_ATTENTION:{INCIDENT}"]
    assert [a for a in cand_alerts if a.incident_key != INCIDENT] == base_alerts

    # Verify H6 resurface note still sends on first dispatch even when today ONLY has F2 mismatch:
    p_h6 = tmp_path / "h6" / "data" / "picks"
    cli_dated(p_h6, DAY, saver="used", streak=14)
    set_flag(p_h6, "active", 2027)
    write_json(tmp_path / "h6" / "data" / "health_state" / MAIN_STATUS, {
        "schema_version": alert.HEALTH_DM_STATUS_SCHEMA_VERSION,
        "status": "failed",
        "critical_count": 2,
    })
    _, h6_sends = run_real_eod(monkeypatch, p_h6, slate=DAY.date(), wall=DAY)
    assert len(h6_sends) == 2
    assert "Previous health alert DM FAILED to send (2 CRITICAL)" in h6_sends[0][1]
    assert "Streak Saver on 2027-07-15 ET." not in h6_sends[0][1]
    assert "Streak Saver on 2027-07-15 ET." in h6_sends[1][1]


def test_receipt_identity_strictness_and_alias_and_override_edges(tmp_path, monkeypatch):
    # 1. Strict 6-key / no-duplicate-member / nonblank message_id / body_sha256 check on receipt:
    picks = tmp_path / "strict" / "data" / "picks"
    cli_dated(picks, DAY, saver="used", streak=14)
    set_flag(picks, "active", 2027)
    _, sends0 = run_real_eod(monkeypatch, picks, slate=DAY.date(), wall=DAY)
    assert len(sends0) == 1
    rec_path = tmp_path / "strict" / "data" / "health_state" / MISMATCH_STATUS
    good_rec = json.loads(rec_path.read_text())

    # Identical rerun deduplicates:
    assert run_real_eod(monkeypatch, picks, slate=DAY.date(), wall=DAY)[1] == []

    bad_receipts = [
        json.dumps({**good_rec, "extra": 1}),
        json.dumps({**good_rec, "message_id": "   "}),
        json.dumps({**good_rec, "body_sha256": "0" * 64}),
        json.dumps({**good_rec, "sent_day": "2027-07-14"}),
        json.dumps({**good_rec, "sent_sources": ["WARN_ATTENTION:saver_state"]}),
        '{"schema_version":"bts_health_dm_delivery_status_v1","status":"sent","status":"sent",'
        f'"sent_day":"2027-07-15","sent_sources":["WARN_ATTENTION:{INCIDENT}"],'
        f'"message_id":"msg-1","body_sha256":"{good_rec["body_sha256"]}"}}',
    ]
    for raw_bad in bad_receipts:
        rec_path.write_text(raw_bad, encoding="utf-8")
        _, s = run_real_eod(monkeypatch, picks, slate=DAY.date(), wall=DAY)
        assert len(s) == 1, raw_bad

    # 2. Aliases & non-regular files at mismatch receipt (symlink/hardlink to saver_state, symlink to manual, FIFO, dir, symlinked parent, >255-byte derived name):
    crit = alert.Alert("CRITICAL", "offline-probe", "Critical present")
    for kind in ("symlink_flag", "hardlink_flag", "symlink_manual", "fifo", "dir", "symlink_parent", "enametoolong_derived"):
        sub = tmp_path / f"edge_{kind}" / "data"
        p = sub / "picks"
        manual = cli_dated(p, DAY, saver="used", streak=14)
        flag_f = set_flag(p, "active", 2027)
        flag_bytes = flag_f.read_bytes()
        manual_bytes = manual.read_bytes()
        overrides = None

        hs = sub / "health_state"
        mm = hs / MISMATCH_STATUS
        if kind == "symlink_parent":
            real_hs = tmp_path / f"real_hs_{kind}"
            real_hs.mkdir()
            sub.mkdir(parents=True, exist_ok=True)
            hs.symlink_to(real_hs)
        else:
            hs.mkdir(parents=True, exist_ok=True)
            if kind == "symlink_flag":
                mm.symlink_to(flag_f)
            elif kind == "hardlink_flag":
                os.link(flag_f, mm)
            elif kind == "symlink_manual":
                mm.symlink_to(manual)
            elif kind == "fifo":
                os.mkfifo(mm)
            elif kind == "dir":
                mm.mkdir()
            elif kind == "enametoolong_derived":
                long_main = hs / ("a" * 240 + ".json")
                overrides = {"health_dm_delivery_status": str(long_main)}

        _, sends = run_real_eod(monkeypatch, p, slate=DAY.date(), wall=DAY, overrides=overrides, extra_alerts=[crit])
        assert len(sends) == 2, kind
        assert flag_f.read_bytes() == flag_bytes
        assert manual.read_bytes() == manual_bytes
        if kind in ("symlink_flag", "hardlink_flag"):
            assert mm.samefile(flag_f)

    # 3. Override with no basename or raising with_name:
    class RaisingStatusPath:
        def __fspath__(self):
            return "/"

    for bad_ov in ("/", "//", "", ".", "/.", RaisingStatusPath()):
        p_ov = tmp_path / f"ov_{ id(bad_ov) }" / "data" / "picks"
        cli_dated(p_ov, DAY, saver="used", streak=14)
        set_flag(p_ov, "active", 2027)
        _, sends_ov = run_real_eod(
            monkeypatch, p_ov, slate=DAY.date(), wall=DAY,
            overrides={"health_dm_delivery_status": bad_ov if isinstance(bad_ov, str) else "/"},
            extra_alerts=[crit],
        )
        assert len(sends_ov) == 2
        default_mm = p_ov.parent / "health_state" / MISMATCH_STATUS
        assert json.loads(default_mm.read_text())["status"] == "sent"


def test_same_day_both_orders_consecutive_days_after_midnight_and_1000_grapheme_limit(tmp_path, monkeypatch):
    # 1. Both incident orders on same processed date D (live and historical):
    monkeypatch.chdir(tmp_path)
    for hist, wall in ((False, DAY), (True, DAY.replace(hour=0, minute=4) + dt.timedelta(days=1))):
        for first, second in (("not_earned", "used"), ("used", "not_earned")):
            p = Path(f"ord_{hist}_{first}") / "picks"
            set_flag(p, "active", 2027)
            for st in (first, second):
                cli_dated(p, DAY, saver=st, streak=14)
                raw, sends = run_real_eod(monkeypatch, p, slate=DAY.date(), wall=wall)
                assert len(sends) == 1
                assert (any(a.incident_key == INCIDENT for a in raw)) == (st == "used")
                assert run_real_eod(monkeypatch, p, slate=DAY.date(), wall=wall)[1] == []

    # 2. Consecutive days D..D+3 + attention label placement after midnight + independence from K1:
    p_seq = tmp_path / "seq" / "data" / "picks"
    set_flag(p_seq, "active", 2027)
    # Day 1 after midnight (historical):
    d1_wall = DAY.replace(hour=0, minute=6) + dt.timedelta(days=1)
    cli_dated(p_seq, DAY, saver="used", streak=14)
    _, s1 = run_real_eod(monkeypatch, p_seq, slate=DAY.date(), wall=d1_wall)
    assert len(s1) == 1
    b1 = s1[0][1]
    assert b1.count(health.HISTORICAL_POINTER) == 1
    assert b1.index("read active.") < b1.index(" (1st observed day)") < b1.index(health.HISTORICAL_POINTER)
    assert json.loads((p_seq.parent / "health_state" / MISMATCH_STATUS).read_text())["sent_day"] == "2027-07-15"

    # Days 2, 3, 4 live:
    for offset, expected_lbl in ((1, "(2nd consecutive day)"), (2, "(3rd consecutive day)"), (3, "(4th consecutive day)")):
        dn = DAY + dt.timedelta(days=offset)
        cli_dated(p_seq, dn, saver="used", streak=14)
        _, sn = run_real_eod(monkeypatch, p_seq, slate=dn.date(), wall=dn)
        assert len(sn) == 1 and sn[0][1].endswith(expected_lbl)

    # Day 5 switches to K1 (dated not_earned at streak 14) -> starts at (1st observed day), not 5th!
    d5 = DAY + dt.timedelta(days=4)
    cli_dated(p_seq, d5, saver="not_earned", streak=14)
    _, s5 = run_real_eod(monkeypatch, p_seq, slate=d5.date(), wall=d5)
    assert len(s5) == 1 and "(1st observed day)" in s5[0][1]

    # 3. Digest near 1,000-grapheme chat limit (using combining acute e\u0301 -> 999 graphemes, >1900 code points!):
    p_lim = tmp_path / "lim" / "data" / "picks"
    cli_dated(p_lim, DAY, saver="used", streak=14)
    set_flag(p_lim, "used", 2027)
    near_crit = alert.Alert("CRITICAL", "near_cap", "e\u0301" * 958)
    grapheme_lengths = []

    def strict_1000_grapheme_transport(_, body):
        g_len = sum(not unicodedata.combining(ch) for ch in body)
        grapheme_lengths.append(g_len)
        if g_len > 1000:
            raise ValueError(f"exceeds 1000 graphemes: {g_len}")
        return "ack-ok"

    _, ctrl_sends = run_real_eod(monkeypatch, p_lim, slate=DAY.date(), wall=DAY, transport=strict_1000_grapheme_transport, extra_alerts=[near_crit])
    ctrl_main_bytes = (p_lim.parent / "health_state" / MAIN_STATUS).read_bytes()
    (p_lim.parent / "health_state" / MAIN_STATUS).unlink()

    set_flag(p_lim, "active", 2027)
    _, lim_sends = run_real_eod(monkeypatch, p_lim, slate=DAY.date(), wall=DAY, transport=strict_1000_grapheme_transport, extra_alerts=[near_crit])
    assert len(lim_sends) == 2
    assert lim_sends[0] == ctrl_sends[0]
    assert (p_lim.parent / "health_state" / MAIN_STATUS).read_bytes() == ctrl_main_bytes
    assert 995 <= grapheme_lengths[0] == grapheme_lengths[1] <= 1000
    assert grapheme_lengths[1] + grapheme_lengths[2] > 1400


def test_transport_errors_empty_ids_and_interrupts_before_receipt_writes(tmp_path, monkeypatch):
    crit = alert.Alert("CRITICAL", "offline-probe", "Primary failure")

    # 1. Empty / whitespace / None message IDs: primary digest counts as sent, mismatch counts as failed and retries!
    for empty_val in ("", "   ", None):
        p = tmp_path / f"empty_{repr(empty_val)}" / "data" / "picks"
        cli_dated(p, DAY, saver="used", streak=14)
        set_flag(p, "active", 2027)
        _, sends1 = run_real_eod(monkeypatch, p, slate=DAY.date(), wall=DAY, transport=lambda _r, _b: empty_val, extra_alerts=[crit])
        assert len(sends1) == 2
        hs = p.parent / "health_state"
        assert json.loads((hs / MAIN_STATUS).read_text())["status"] == "sent"
        assert json.loads((hs / MISMATCH_STATUS).read_text())["status"] == "failed"

        # Rerun with valid transport ID: primary is deduplicated, ONLY mismatch retries and then deduplicates!
        _, sends2 = run_real_eod(monkeypatch, p, slate=DAY.date(), wall=DAY, extra_alerts=[crit])
        assert len(sends2) == 1 and "Streak Saver on 2027-07-15 ET." in sends2[0][1]
        assert json.loads((hs / MISMATCH_STATUS).read_text())["status"] == "sent"
        assert run_real_eod(monkeypatch, p, slate=DAY.date(), wall=DAY, extra_alerts=[crit])[1] == []

    # 2. Interrupt after 1st send before 1st receipt write (_persist_health_dm_delivery_status):
    p_int1 = tmp_path / "int1" / "data" / "picks"
    cli_dated(p_int1, DAY, saver="used", streak=14)
    flag1 = set_flag(p_int1, "active", 2027)
    f1_bytes = flag1.read_bytes()
    with monkeypatch.context() as mp:
        mp.setattr(alert, "_persist_health_dm_delivery_status", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt("cut-1")))
        with pytest.raises(KeyboardInterrupt):
            run_real_eod(monkeypatch, p_int1, slate=DAY.date(), wall=DAY, extra_alerts=[crit])
    assert flag1.read_bytes() == f1_bytes
    assert not (p_int1.parent / "health_state" / MAIN_STATUS).exists()
    assert not (p_int1.parent / "health_state" / MISMATCH_STATUS).exists()
    # Restart sends both, then deduplicates both:
    assert len(run_real_eod(monkeypatch, p_int1, slate=DAY.date(), wall=DAY, extra_alerts=[crit])[1]) == 2
    assert run_real_eod(monkeypatch, p_int1, slate=DAY.date(), wall=DAY, extra_alerts=[crit])[1] == []

    # 3. Interrupt after 2nd send before 2nd receipt replacement (_replace_receipt):
    p_int2 = tmp_path / "int2" / "data" / "picks"
    manual2 = cli_dated(p_int2, DAY, saver="used", streak=14)
    flag2 = set_flag(p_int2, "active", 2027)
    f2_bytes = flag2.read_bytes()
    m2_bytes = manual2.read_bytes()
    attempted_bodies = []
    with monkeypatch.context() as mp:
        mp.setattr(alert, "_replace_receipt", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt("cut-2")))
        with pytest.raises(KeyboardInterrupt):
            run_real_eod(
                monkeypatch, p_int2, slate=DAY.date(), wall=DAY,
                transport=lambda _r, b: attempted_bodies.append(b) or "ack-2",
                extra_alerts=[crit],
            )
    assert len(attempted_bodies) == 2
    assert flag2.read_bytes() == f2_bytes and manual2.read_bytes() == m2_bytes
    assert json.loads((p_int2.parent / "health_state" / MAIN_STATUS).read_text())["status"] == "sent"
    assert not (p_int2.parent / "health_state" / MISMATCH_STATUS).exists()
    # Restart deduplicates 1st digest, retries ONLY 2nd mismatch DM once, then deduplicates both:
    _, retry2 = run_real_eod(monkeypatch, p_int2, slate=DAY.date(), wall=DAY, extra_alerts=[crit])
    assert len(retry2) == 1 and "Streak Saver on 2027-07-15 ET." in retry2[0][1]
    assert json.loads((p_int2.parent / "health_state" / MISMATCH_STATUS).read_text())["status"] == "sent"
    assert run_real_eod(monkeypatch, p_int2, slate=DAY.date(), wall=DAY, extra_alerts=[crit])[1] == []


if __name__ == "__main__":
    print(f"bts.__file__ = {bts.__file__}")
    sys.exit(pytest.main([__file__, "-v", "-s", "-rA"]))
