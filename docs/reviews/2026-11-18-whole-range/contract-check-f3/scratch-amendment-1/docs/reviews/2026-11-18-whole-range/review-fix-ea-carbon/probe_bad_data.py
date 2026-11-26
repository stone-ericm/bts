"""Probe item 4: Bad data in scheduler_state.json, HTML/line-break escaping,
template 'below the' invariant, and bidirectional state compatibility with 63669c42.
"""
from __future__ import annotations

import contextlib
import datetime as dt_mod
import html
import json
import re
import socket
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

from bts import scheduler, web

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BASE_SRC = HERE / "base_63669c42" / "src"
ET = ZoneInfo("America/New_York")


def _no_net(*a, **k):
    raise OSError("network disabled")


def _check_banner_escaped(banner: str) -> None:
    if not banner:
        return
    prefix = '<div class="skip-banner"><strong>LAST RECORDED NO-PICK SUMMARY</strong><span>'
    suffix = "</span></div>"
    assert banner.startswith(prefix) and banner.endswith(suffix), banner
    inner = banner[len(prefix):-len(suffix)]
    assert "<" not in inner and ">" not in inner, f"Unescaped angle bracket in {inner!r}"
    # Every '&' in inner must start a valid entity produced by html.escape
    leftover = re.sub(r"&(amp|lt|gt|quot|#x27);", "", inner)
    assert "&" not in leftover, f"Unescaped ampersand in {inner!r}"


def _is_expected_k2(summary: object) -> bool:
    if not isinstance(summary, dict):
        return False
    raw_reason = summary.get("reason")
    if isinstance(raw_reason, str) and " ".join(raw_reason.splitlines()).strip():
        return False
    if summary.get("basis") == "no_decision":
        return False

    def valid_num(v):
        if type(v) not in (int, float):
            return None
        try:
            f = float(v)
        except OverflowError:
            return None
        return f if 0 <= f <= 1 else None

    p = valid_num(summary.get("best_p"))
    b = valid_num(summary.get("bar"))
    return p is not None and b is not None and p < b


def main() -> None:
    now = datetime(2027, 7, 15, 19, 0, tzinfo=ET)
    bad_numbers = [
        None, True, False, "", "0.75", [], {"a": 1},
        -1, -0.1, -1e-9, 1.0000001, 2, 100,
        1e308, 1.5e308, -1e308, 10**400, -(10**400),
        float("nan"), float("inf"), float("-inf"),
        0, 0.0, 0.75, 0.80, 1, 1.0,
    ]
    bad_strings = [
        None, 123, True, ["<script>"], {"k": "v"}, "", "   ",
        "\n", "\r\n", " \r\n\t\v\f\u2028\u2029 ",
        "<script>alert('xss')</script>&\"'",
        "Line1\nLine2\r\nLine3\u2028Line4",
        "Reason without dot", "Reason with dot.", "Bang!", "Question?",
    ]
    bad_streaks = [None, True, False, -5, -1, 0, 10, 56, 10**100, 3.5, "10", [10], {"s": 10}]
    bad_bases = [None, "decision", "no_decision", "other", 123, True, ["no_decision"]]
    bad_verified = [None, True, False, 1, 0, "true", ["x"]]

    summaries: list[object] = [
        None, {}, [], ["not", "dict"], "string_summary", 0, True,
        # Legacy shapes (no basis, no candidate_verified)
        {"best_batter": "Bo Bichette", "best_team": "TOR", "best_p": 0.75, "streak": 10},
        {"best_batter": "Bo Bichette", "best_team": "TOR", "best_p": 0.75, "bar": 0.80, "streak": 10},
        {"best_batter": "Bo Bichette", "best_team": "TOR", "best_p": 0.85, "bar": 0.80, "streak": 10},
        {"best_batter": "Bo Bichette", "best_team": "TOR", "best_p": 0.75, "streak": 0, "reason": "stop rule"},
    ]

    for num in bad_numbers:
        summaries.append({"best_batter": "<b>Bat</b>\n1", "best_team": "T&M", "best_p": num, "bar": 0.8, "streak": 8, "candidate_verified": True})
        summaries.append({"best_batter": "Bat", "best_team": "<T>", "best_p": 0.75, "bar": num, "streak": 8, "candidate_verified": False})
        summaries.append({"best_batter": "Bat", "best_team": "T", "best_p": num, "bar": num, "streak": 8})
    for s_val in bad_strings:
        summaries.append({"best_batter": s_val, "best_team": s_val, "best_p": 0.75, "bar": 0.80, "streak": 8, "reason": s_val})
    for st_val in bad_streaks:
        for b_val in bad_bases:
            for ver_val in bad_verified:
                summaries.append({
                    "best_batter": "<A>", "best_team": "<B>", "best_p": 0.75, "bar": 0.80,
                    "streak": st_val, "basis": b_val, "candidate_verified": ver_val,
                })

    k2_count = 0
    non_k2_count = 0
    with tempfile.TemporaryDirectory() as tmp:
        picks_dir = Path(tmp) / "picks"
        picks_dir.mkdir()
        date_dir = picks_dir / "2027-07-15"
        date_dir.mkdir()
        state_file = date_dir / "scheduler_state.json"

        with patch.object(socket.socket, "connect", _no_net), \
                patch.object(web, "PICKS_DIR", picks_dir), \
                patch.object(web, "HEALTH_STATE_DIR", Path(tmp) / "hs"), \
                patch.object(web, "VALIDATION_DIR", Path(tmp) / "val"), \
                patch.object(web, "fetch_bluesky_posts", lambda *a, **k: []):
            for idx, item in enumerate(summaries):
                payload = {
                    "date": "2027-07-15",
                    "schedule_fetched_at": now.isoformat(),
                    "games": [],
                    "confirmed_game_pks": [],
                    "runs_completed": [],
                    "pick_locked": False,
                    "pick_locked_at": None,
                    "result_status": None,
                    "next_wakeup": None,
                    "skip_summary": item,
                }
                state_file.write_text(json.dumps(payload, allow_nan=True))
                loaded_sched = scheduler.load_state("2027-07-15", picks_dir)
                assert loaded_sched is not None
                loaded_web = web.load_scheduler_state("2027-07-15")
                dec_sum = loaded_sched.skip_summary

                s_txt = scheduler.format_skip_summary(dec_sum)
                dm_txt = scheduler.format_skip_dm("2027-07-15", dec_sum)
                ban_txt = web.render_skip_banner(loaded_web.get("skip_summary"))

                assert "\n" not in s_txt and "\r" not in s_txt
                assert "\n" not in dm_txt and "\r" not in dm_txt
                assert "\n" not in ban_txt and "\r" not in ban_txt
                _check_banner_escaped(ban_txt)

                is_k2 = _is_expected_k2(dec_sum)
                if is_k2:
                    k2_count += 1
                    assert "below the" in s_txt and "below the" in dm_txt and "below the" in ban_txt
                else:
                    non_k2_count += 1
                    assert "below the" not in s_txt, (idx, dec_sum, s_txt)
                    assert "below the" not in dm_txt, (idx, dec_sum, dm_txt)
                    if ban_txt:
                        assert "below the" not in ban_txt, (idx, dec_sum, ban_txt)

                # Also exercise full web.render_page() on HTML/XSS/malformed cases
                if idx < 40:
                    with patch.object(dt_mod, "datetime", wraps=dt_mod.datetime) as m_dt:
                        m_dt.now.return_value = now
                        page = web.render_page()
                    assert "<script>alert('xss')</script>" not in page

        # Bidirectional state compatibility with 63669c42 (if extracted)
        assert BASE_SRC.exists(), "Run probe_words_only.py first or ensure base_63669c42 exists"
        compat_dir = Path(tmp) / "compat"
        compat_dir.mkdir()
        # 1. Base writes -> HEAD reads
        subprocess.run(
            [sys.executable, "-c",
             f"import sys; sys.path.insert(0, {str(BASE_SRC)!r}); "
             "from pathlib import Path; from bts import scheduler; "
             f"assert Path(scheduler.__file__).resolve().is_relative_to(Path({str(BASE_SRC)!r}).resolve()); "
             f"p = Path({str(compat_dir)!r}); "
             "st = scheduler.SchedulerState(date='2027-07-15', schedule_fetched_at='t', games=[], "
             "confirmed_game_pks=[], runs_completed=[], pick_locked=False, pick_locked_at=None, "
             "result_status=None, next_wakeup=None, "
             "skip_summary={'best_batter': 'BaseBat', 'best_team': 'CLE', 'best_p': 0.75, 'bar': 0.80, 'streak': 10}); "
             "scheduler.save_state(st, p)"],
            check=True, timeout=15,
        )
        head_loaded = scheduler.load_state("2027-07-15", compat_dir)
        assert head_loaded is not None and head_loaded.skip_summary["best_batter"] == "BaseBat"
        head_dm = scheduler.format_skip_dm("2027-07-15", head_loaded.skip_summary)
        assert "Top model prediction (eligibility unverified): BaseBat (CLE) 75.0%, below the streak-10 bar (80.0%) using unrounded values. Streak used at this check: 10." in head_dm

        # 2. HEAD writes (with basis and candidate_verified) -> Base reads
        head_loaded.skip_summary.update(basis="decision", candidate_verified=True)
        scheduler.save_state(head_loaded, compat_dir)
        res_base_read = subprocess.run(
            [sys.executable, "-c",
             f"import sys; sys.path.insert(0, {str(BASE_SRC)!r}); "
             "import json; from pathlib import Path; from bts import scheduler; "
             f"assert Path(scheduler.__file__).resolve().is_relative_to(Path({str(BASE_SRC)!r}).resolve()); "
             f"st = scheduler.load_state('2027-07-15', Path({str(compat_dir)!r})); "
             "print(json.dumps(st.skip_summary, sort_keys=True))"],
            check=True, capture_output=True, text=True, timeout=15,
        )
        assert json.loads(res_base_read.stdout) == head_loaded.skip_summary

        # 3. Corrupt JSON and directory path
        bad_dir = Path(tmp) / "corrupt"
        f_bad = bad_dir / "2027-07-15" / "scheduler_state.json"
        f_bad.parent.mkdir(parents=True)
        f_bad.write_text("{corrupt")
        with contextlib.redirect_stderr(io.StringIO() if "io" in globals() else open("/dev/null", "w")):
            assert scheduler.load_state("2027-07-15", bad_dir) is None
        assert list(f_bad.parent.glob("scheduler_state.json.corrupt-*"))
        with patch.object(web, "PICKS_DIR", bad_dir):
            f_bad.mkdir()
            assert web.load_scheduler_state("2027-07-15") == {}
            try:
                scheduler.load_state("2027-07-15", bad_dir)
                raise AssertionError("expected IsADirectoryError")
            except IsADirectoryError:
                pass

    summary_out = {
        "total_summaries_tested": len(summaries),
        "k2_cases": k2_count,
        "non_k2_cases": non_k2_count,
        "bidirectional_state_compat": True,
        "corrupt_and_directory_checked": True,
    }
    (HERE / "probe_bad_data_summary.json").write_text(json.dumps(summary_out, indent=2))
    print("probe_bad_data: ALL CHECKS PASSED:", json.dumps(summary_out))


if __name__ == "__main__":
    import io
    main()
