"""Review probe 3: DM 1,000-grapheme transport, non-ASCII code point vs grapheme boundaries, and R64 operator-text states."""
from __future__ import annotations

import datetime as dt
import json
import tempfile
import unicodedata
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError
from unittest.mock import patch
from zoneinfo import ZoneInfo

from click.testing import CliRunner

from bts import day_outcome, dm, picks, scheduler
from bts.cli import cli
from bts.health import alert as health_alert, runner as health_runner, streak_validation
from tests.test_day_outcome import DATE, PICK, _bound_receipt, _box, _publish_receipt, _write_decision

ET = ZoneInfo("America/New_York")


class PinnedClock(dt.datetime):
    wall = dt.datetime(2027, 4, 3, 8, 30, tzinfo=ET)

    @classmethod
    def now(cls, tz=None):
        return cls.wall.astimezone(tz) if tz else cls.wall.replace(tzinfo=None)


def count_graphemes(text: str) -> int:
    """Count extended grapheme clusters for ASCII, CRLF, combining marks, skin tones, flags, and ZWJ emoji."""
    n = len(text)
    if n == 0:
        return 0
    clusters = 0
    i = 0
    ri_run = 0
    while i < n:
        ch = text[i]
        cp = ord(ch)
        if ch == "\r" and i + 1 < n and text[i + 1] == "\n":
            clusters += 1
            ri_run = 0
            i += 2
            continue
        if ch in ("\r", "\n"):
            clusters += 1
            ri_run = 0
            i += 1
            continue
        cat = unicodedata.category(ch)
        is_extend = (
            cat in ("Mn", "Mc", "Me")
            or ch in ("\u200d", "\ufe0e", "\ufe0f")
            or 0x1F3FB <= cp <= 0x1F3FF
        )
        prev_zwj = i > 0 and text[i - 1] == "\u200d"
        is_ri = 0x1F1E6 <= cp <= 0x1F1FF
        if prev_zwj:
            ri_run = 0
        elif is_extend and i > 0 and text[i - 1] not in ("\r", "\n"):
            pass
        elif is_ri:
            ri_run += 1
            if ri_run % 2 == 1:
                clusters += 1
        else:
            clusters += 1
            ri_run = 0
        i += 1
    return clusters


def grapheme_transport(delivered: list[str], rejected: list[tuple[int, int, int]]):
    def respond(req, **opts):
        route = req.full_url.rsplit("/", 1)[-1]
        if route == "chat.bsky.convo.sendMessage":
            assert opts["idempotent"] is False
            body = json.loads(req.data)["message"]["text"]
            g = count_graphemes(body)
            b = len(body.encode("utf-8"))
            c = len(body)
            if g > 1000 or b > 10000:
                rejected.append((c, g, b))
                raise HTTPError(req.full_url, 400, f"maxGraphemes=1000/maxLength=10000 (g={g}, b={b})", {}, None)
            delivered.append(body)
            return BytesIO(b'{"id":"chat-msg-1"}')
        if route == "com.atproto.server.createSession":
            return BytesIO(b'{"accessJwt":"jwt"}')
        if route.startswith("chat.bsky.convo.getConvoForMembers?"):
            return BytesIO(b'{"convo":{"id":"convo"}}')
        raise AssertionError(req.full_url)

    return respond


def run_cli(pdir: Path, cfg: Path, games: int = 1, *, date: str = DATE, wall: dt.datetime = PinnedClock.wall):
    delivered: list[str] = []
    rejected: list[tuple[int, int, int]] = []
    argv = [
        "day-outcome", "--date", date, "--config", str(cfg), "--picks-dir", str(pdir),
        "--expected-username", "stonehengee", "--dm-recipient", "did:plc:operator",
    ]
    with patch.object(PinnedClock, "wall", wall), \
         patch.object(dt, "datetime", PinnedClock), \
         patch.object(scheduler, "fetch_schedule", lambda _: [{"gamePk": 900}] * games), \
         patch.object(dm, "get_dm_password", lambda: "pw"), \
         patch.object(dm, "retry_urlopen", grapheme_transport(delivered, rejected)):
        res = CliRunner().invoke(cli, argv)
    return res, delivered, rejected


def test_exact_code_point_boundaries_clean_and_problem(tmp_path):
    # Problem state: not_entered. Fixed chars with "not a JSON object" = 605.
    for n_chars, expect_compact in [(394, False), (395, False), (396, True)]:
        pdir, cfg = _box(tmp_path / f"prob_{n_chars}")
        batter = "B" * n_chars
        _write_decision(pdir, primary={**PICK, "batter_name": batter})
        _publish_receipt(pdir, _bound_receipt(pdir, confirmed=False, reason="no_pick"))
        (pdir / "streak.json").write_bytes(b"[]")
        res, delivered, rejected = run_cli(pdir, cfg, games=1)
        assert res.exit_code == 1 and res.stderr == "" and rejected == []
        assert len(delivered) == 1
        msg = delivered[0]
        out = json.loads(res.stdout)
        assert out["pick"] == batter  # stdout always retains full pick
        assert msg.startswith(f"BTS day check {DATE}: NOT ENTERED. ")
        if expect_compact:
            assert len(msg) == 260
            assert "See day-outcome JSON in cron.log" in msg
            assert batter not in msg
        else:
            assert len(msg) == 605 + n_chars
            assert batter in msg
            assert "Also: The local model streak file (streak.json) needs review (not a JSON object)." in msg

    # Clean state: entered. Fixed chars with "not a JSON object" = 597.
    for n_chars, expect_compact in [(402, False), (403, False), (404, True)]:
        pdir, cfg = _box(tmp_path / f"clean_{n_chars}")
        batter = "C" * n_chars
        _write_decision(pdir, primary={**PICK, "batter_name": batter})
        _publish_receipt(pdir, _bound_receipt(pdir))
        (pdir / "streak.json").write_bytes(b"[]")
        res, delivered, rejected = run_cli(pdir, cfg, games=1)
        assert res.exit_code == 1 and res.stderr == "" and rejected == []
        assert len(delivered) == 1
        msg = delivered[0]
        out = json.loads(res.stdout)
        assert out["state"] == "entered" and out["ok"] is False and out["pick"] == batter
        assert msg.startswith(f"BTS day check {DATE}: MODEL STREAK FILE NEEDS REVIEW. ")
        if expect_compact:
            assert len(msg) == 279
            assert "See day-outcome JSON in cron.log" in msg
        else:
            assert len(msg) == 597 + n_chars
            assert batter in msg


def test_non_ascii_graphemes_never_lose_base_alert(tmp_path):
    # Sanity check our grapheme counter on multi-codepoint sequences
    assert count_graphemes("Jose\u0301") == 4 and len("Jose\u0301") == 5
    assert count_graphemes("👨\u200d👩\u200d👧\u200d👦") == 1 and len("👨\u200d👩\u200d👧\u200d👦") == 7
    assert count_graphemes("🇩🇴🇩🇴") == 2 and len("🇩🇴🇩🇴") == 4
    assert count_graphemes("\U0002000b") == 1 and len("\U0002000b".encode("utf-8")) == 4

    non_ascii_batters = [
        ("mlb_nfc", "José Ramírez"),
        ("mlb_nfd_short", "Jose\u0301 Rami\u0301rez + Ronald Acun\u0303a Jr."),
        # 200 NFD 'e\u0301' = 400 code points, 200 graphemes: base DM < 1000 graphemes; F4 = 1005 code points -> compact!
        ("nfd_200", "e\u0301" * 200),
        # 420 NFD 'e\u0301' = 840 code points, 420 graphemes: base DM = 973 code points, 553 graphemes (succeeds on base!);
        # F4 = 1445 code points -> compact DM (260 graphemes, succeeds!)
        ("nfd_420", "e\u0301" * 420),
        # 460 NFD 'e\u0301' = 920 code points, 460 graphemes: base DM = 1053 code points > 1000, BUT 593 graphemes <= 1000!
        # Base succeeds under 1000-grapheme transport; F4 sends compact DM and ALSO succeeds!
        ("nfd_460_over_1000cp_under_1000g", "e\u0301" * 460),
        # ZWJ family emoji x 80 = 560 code points, 80 graphemes, 2000 UTF-8 bytes
        ("zwj_80", "👨\u200d👩\u200d👧\u200d👦" * 80),
        # 4-byte UTF-8 character x 395 = 395 code points, 395 graphemes, 1580 UTF-8 bytes -> exact 1000 code points full DM!
        ("cjk_ext_395_exact_1000cp", "\U0002000b" * 395),
        # Regional indicator flags x 197 = 394 code points, 197 graphemes -> 999 code points full DM!
        ("flags_197", "🇩🇴" * 197),
    ]

    for label, batter in non_ascii_batters:
        for state in ("not_entered", "entered", "delivered_under_research", "commit_not_scoreable"):
            intent = "research" if state == "delivered_under_research" else "enter"
            delivery = "private" if intent == "research" else "dm"
            pdir, cfg = _box(tmp_path / f"{label}_{state}", intent=intent, delivery=delivery)
            scoreable = state != "commit_not_scoreable"
            d_status = "failed" if state == "commit_not_scoreable" else "delivered"
            _write_decision(pdir, primary={**PICK, "batter_name": batter}, scoreable=scoreable, delivery_status=d_status)
            if state == "not_entered":
                _publish_receipt(pdir, _bound_receipt(pdir, confirmed=False, reason="no_pick"))
            elif state == "entered":
                _publish_receipt(pdir, _bound_receipt(pdir))

            # First with good streak.json (base behaviour)
            (pdir / "streak.json").write_bytes(b"{}")
            res_good, deliv_good, rej_good = run_cli(pdir, cfg, games=1)
            assert rej_good == [], (label, state, rej_good)
            if state == "entered":
                assert res_good.exit_code == 0 and deliv_good == []
            else:
                assert res_good.exit_code == 1 and len(deliv_good) == 1

            # Now with broken streak.json (F4 behaviour): NO alert is ever rejected!
            (pdir / "streak.json").write_bytes(b"[]")
            res_bad, deliv_bad, rej_bad = run_cli(pdir, cfg, games=1)
            assert res_bad.exit_code == 1 and res_bad.stderr == "" and rej_bad == [], (label, state, rej_bad)
            assert len(deliv_bad) == 1
            expected_title = "MODEL STREAK FILE NEEDS REVIEW" if state == "entered" else state.replace("_", " ").upper()
            assert deliv_bad[0].startswith(f"BTS day check {DATE}: {expected_title}. ")
            assert count_graphemes(deliv_bad[0]) <= 1000
            assert len(deliv_bad[0].encode("utf-8")) <= 10000
            assert json.loads(res_bad.stdout)["pick"] == batter


def test_r64_four_worst_ordinary_states(tmp_path):
    # State A: Scorer failed on corrupt streak.json, leaving unresolved pick; reconcile warned and failed; 08:30 report
    pdir_a, cfg_a = _box(tmp_path / "r64_scorer")
    _write_decision(pdir_a)
    _publish_receipt(pdir_a, _bound_receipt(pdir_a))
    primary = picks.Pick(
        batter_name="Kwan", batter_id=1, team="CLE", lineup_position=1,
        pitcher_name="Cole", pitcher_id=2, p_game_hit=0.78, flags=[],
        projected_lineup=False, game_pk=900, game_time="2027-04-02T23:10:00Z",
    )
    picks.save_pick(
        picks.DailyPick(DATE, "2027-04-02T17:00:00-04:00", primary, None, None,
                        notification_sent=True, notification_id="dm-1", result=None),
        pdir_a,
    )
    (pdir_a / "streak.json").write_bytes(b"[]")
    with patch.object(PinnedClock, "wall", dt.datetime(2027, 4, 3, 2, 0, tzinfo=ET)), \
         patch.object(dt, "datetime", PinnedClock), \
         patch.object(picks, "resolve_daily_slot_results", return_value={"pick": "hit"}):
        rec_res = CliRunner().invoke(cli, ["reconcile", "--picks-dir", str(pdir_a)])
    assert rec_res.exit_code == 1 and isinstance(rec_res.exception, AttributeError)
    assert "Season replay unavailable for 2027-04-02 (no final result on the pick)" in rec_res.stderr

    res_a, deliv_a, rej_a = run_cli(pdir_a, cfg_a, games=1)
    assert res_a.exit_code == 1 and rej_a == [] and len(deliv_a) == 1
    assert "Preserve the files in place during triage." in deliv_a[0]
    assert "Follow B8b item 2 in docs/ops/2027-season-start.md before the next affected job." in deliv_a[0]
    assert "The 08:00 ET correction cutoff still applies." in deliv_a[0]

    # State B: Health CRITICAL ('streak_validation') the same night + 08:30 day-outcome DM: neither deduplicates the other
    pdir_b, cfg_b = _box(tmp_path / "r64_health")
    (pdir_b / "streak.json").write_bytes(b"[]")
    health_dms: list[str] = []
    health_rej: list[tuple[int, int, int]] = []
    alerts = health_runner._safe_run("streak_validation", lambda: streak_validation.check(pdir_b))
    assert len(alerts) == 1 and alerts[0].level == "CRITICAL"
    with patch.object(dm, "get_dm_password", lambda: "pw"), \
         patch.object(dm, "retry_urlopen", grapheme_transport(health_dms, health_rej)):
        health_alert.dispatch_dm_for_health_alerts(
            alerts,
            recipient="did:plc:operator",
            status_path=pdir_b.parent / "health_state" / "health_dm_delivery_status.json",
            today=dt.date(2027, 4, 2),
        )
    assert len(health_dms) == 1 and health_rej == []
    res_b, deliv_b, rej_b = run_cli(pdir_b, cfg_b, games=0)
    assert res_b.exit_code == 1 and len(deliv_b) == 1 and rej_b == []
    assert "and health alerts." in deliv_b[0]

    # State C: First day of a new season (2027-03-24 opener read on 2027-03-25 08:30 ET) with 2026 streak.json
    pdir_c, cfg_c = _box(tmp_path / "r64_opener")
    opener = "2027-03-24"
    opener_wall = dt.datetime(2027, 3, 25, 8, 30, tzinfo=ET)
    (pdir_c / "streak.json").write_bytes(b'{"season":2026,"streak":12,"saver_available":false}')
    res_c_good, deliv_c_good, _ = run_cli(pdir_c, cfg_c, games=0, date=opener, wall=opener_wall)
    assert res_c_good.exit_code == 0 and deliv_c_good == []
    (pdir_c / "streak.json").write_bytes(b'{"season":2026,"streak":"12"}')
    res_c_bad, deliv_c_bad, _ = run_cli(pdir_c, cfg_c, games=0, date=opener, wall=opener_wall)
    assert res_c_bad.exit_code == 1 and len(deliv_c_bad) == 1
    assert "Grading or replay may be affected." in deliv_c_bad[0]

    # State D: Read after 08:00 ET cutoff (e.g. 09:15 ET) after a problem day
    pdir_d, cfg_d = _box(tmp_path / "r64_after_cutoff")
    (pdir_d / "streak.json").write_bytes(b"{")
    late_wall = dt.datetime(2027, 4, 3, 9, 15, tzinfo=ET)
    res_d, deliv_d, _ = run_cli(pdir_d, cfg_d, games=1, date=DATE, wall=late_wall)
    assert res_d.exit_code == 1 and len(deliv_d) == 1
    assert deliv_d[0].startswith(f"BTS day check {DATE}: NO DECISION. ")
    assert "The 08:00 ET correction cutoff still applies." in deliv_d[0]


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        test_exact_code_point_boundaries_clean_and_problem(root / "b")
        test_non_ascii_graphemes_never_lose_base_alert(root / "u")
        test_r64_four_worst_ordinary_states(root / "r")
    print("PROBE 3 PASSED")
