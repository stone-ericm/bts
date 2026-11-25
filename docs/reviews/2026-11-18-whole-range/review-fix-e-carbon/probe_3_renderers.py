"""Probe 3: Renderer resilience, HTML escaping, line-break normalization, `below the` invariant,
and cross-version `scheduler_state.json` compatibility between 63669c42 and HEAD.
"""
from __future__ import annotations

import html
import io
import json
import math
import os
import subprocess
import sys
import tempfile
from contextlib import redirect_stderr
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BASE_SRC = REPO / ".carbon-review/wr-e/base_63669c42/src"
sys.path.insert(0, str(REPO / "src"))

from bts import scheduler, web

LINE_BREAK_CHARS = ("\n", "\r", "\v", "\f", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029")


def classify_expected_case(summary: object) -> str:
    if not isinstance(summary, dict):
        return "K4"
    reason = summary.get("reason")
    if isinstance(reason, str) and reason:
        return "K1"
    if summary.get("basis") == "no_decision":
        return "K5"
    p = scheduler._skip_display_number(summary.get("best_p"))
    bar = scheduler._skip_display_number(summary.get("bar"))
    if p is not None and bar is not None:
        return "K2" if p < bar else "K3"
    return "K4"


def main() -> None:
    bad_numbers = [
        None, True, False, 0, -0.0, 0.75, 0.75004, 1.0, -0.5, -999,
        10**400, -10**400, 1e308, -1e308, sys.float_info.max, -sys.float_info.max,
        float("nan"), float("inf"), float("-inf"),
        "", "0.75", "nan", "inf", [], [0.75], {"p": 0.75},
    ]
    bad_strings = [
        None, "", "Shohei Ohtani", "<script>alert('xss')</script>&\"'",
        "Line1\nLine2\r\nLine3\rLine4\u2028Line5\u2029Line6\x0bLine7\x0cLine8",
        "\n", "\r\n", 123, True, ["nested"], {"k": "v"}, float("nan"),
    ]
    bad_streaks = [None, True, False, 0, 1, 56, -1, -100, 3.5, "8", "<img src=x>", [], {}, float("nan"), 999999]
    bad_reasons = [None, "", "Stopped", "Stopped.", "Stopped!", "Stopped?", "LineA\nLineB", "\n", "<b onmouseover=x>", 0, True, ["r"], {}]
    bad_bases = ["<absent>", None, "decision", "no_decision", "other", True, 1, []]
    bad_verified = ["<absent>", None, True, False, 1, 0, "true", []]

    # Non-dict summaries
    for non_dict in (None, {}, [], ["x"], "not a dict", 0, 42, True, False, float("nan")):
        s_txt = scheduler.format_skip_summary(non_dict)
        dm_txt = scheduler.format_skip_dm("2027-07-15", non_dict)
        banner = web.render_skip_banner(non_dict)
        assert banner == "", f"Expected empty banner for {non_dict!r}, got {banner!r}"
        assert "No probability comparison is available for this check." in s_txt
        assert "below the" not in dm_txt
        assert not any(ch in dm_txt for ch in LINE_BREAK_CHARS)

    # Exhaustive structured grid over hostile field combinations
    tested = 0
    k_counts = {"K1": 0, "K2": 0, "K3": 0, "K4": 0, "K5": 0}
    extreme_float_outputs = []

    for p in bad_numbers[::2] + [0.75, 0.7956, 0.80, 1e308, sys.float_info.max]:
        for bar in bad_numbers[::3] + [0.75, 0.7960, 0.80, 1.5e308, sys.float_info.max]:
            for st in bad_streaks[::2] + [0, 10, -1, True]:
                for rsn in (None, "", "Tail stop; no pick", "Done!", "Multi\nLine?"):
                    for basis in ("<absent>", "decision", "no_decision", 123):
                        for ver in ("<absent>", True, False, 1):
                            summary = {
                                "best_batter": "<script>alert(1)</script>\r\nO'Neil & \"Jr\"",
                                "best_team": "NY\nM<&>",
                                "best_p": p,
                                "bar": bar,
                                "streak": st,
                            }
                            if rsn is not None:
                                summary["reason"] = rsn
                            if basis != "<absent>":
                                summary["basis"] = basis
                            if ver != "<absent>":
                                summary["candidate_verified"] = ver

                            case = classify_expected_case(summary)
                            k_counts[case] += 1
                            tested += 1

                            desc = scheduler.format_skip_summary(summary)
                            dm = scheduler.format_skip_dm("2027-07-15", summary)
                            banner = web.render_skip_banner(summary)

                            # Single-line check
                            assert not any(ch in desc for ch in LINE_BREAK_CHARS), f"Line break in desc: {desc!r}"
                            assert not any(ch in dm for ch in LINE_BREAK_CHARS), f"Line break in dm: {dm!r}"

                            # HTML escaping check on banner
                            prefix = '<div class="skip-banner"><strong>NO PICK AT LAST CHECK</strong><span>'
                            suffix = "</span></div>"
                            assert banner.startswith(prefix) and banner.endswith(suffix)
                            inner = banner[len(prefix):-len(suffix)]
                            assert "<script>" not in inner and "<" not in inner and ">" not in inner
                            assert inner == html.escape(desc)

                            # `below the` template phrase check (inputs above do not contain "below the")
                            if case == "K2":
                                assert "below the " in desc and "below the " in dm and "below the " in banner
                            else:
                                assert "below the" not in desc, f"Unexpected 'below the' in {case}: {desc}"
                                assert "below the" not in dm, f"Unexpected 'below the' in {case}: {dm}"
                                assert "below the" not in banner, f"Unexpected 'below the' in {case}: {banner}"

                            if p in (1e308, sys.float_info.max) and bar in (1.5e308, sys.float_info.max) and len(extreme_float_outputs) < 4:
                                extreme_float_outputs.append((p, bar, case, desc))

    # Also test bad_strings for name/team/reason specifically
    for bs in bad_strings:
        for bt in bad_strings:
            for br in bad_reasons:
                summary = {"best_batter": bs, "best_team": bt, "best_p": 0.72, "bar": 0.80, "streak": 5, "reason": br}
                desc = scheduler.format_skip_summary(summary)
                dm = scheduler.format_skip_dm("2027-07-15", summary)
                banner = web.render_skip_banner(summary)
                assert not any(ch in dm for ch in LINE_BREAK_CHARS)
                inner = banner.split("<span>", 1)[1].rsplit("</span>", 1)[0]
                assert "<" not in inner and ">" not in inner
                case = classify_expected_case(summary)
                if case == "K2":
                    assert "below the" in desc
                else:
                    assert "below the" not in desc
                tested += 1

    # Test JSON round-trip from scheduler_state.json including NaN, Infinity, 1e308, nested objects, >4300-digit int
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        day = "2027-07-15"
        state_dir = tmp_path / day
        state_dir.mkdir()
        state_file = state_dir / "scheduler_state.json"

        raw_jsons = [
            '{"date":"2027-07-15","schedule_fetched_at":"t","games":[],"confirmed_game_pks":[],"runs_completed":[],"pick_locked":false,"pick_locked_at":null,"result_status":null,"next_wakeup":null,"skip_summary":{"best_batter":"<b>A</b>\\nB","best_team":{"x":1},"best_p":NaN,"bar":Infinity,"streak":true}}',
            '{"date":"2027-07-15","schedule_fetched_at":"t","games":[],"confirmed_game_pks":[],"runs_completed":[],"pick_locked":false,"pick_locked_at":null,"result_status":null,"next_wakeup":null,"skip_summary":{"best_batter":"X","best_team":"Y","best_p":1e308,"bar":1.5e308,"streak":5}}',
            # Legacy summary without basis or candidate_verified
            '{"date":"2027-07-15","schedule_fetched_at":"t","games":[],"confirmed_game_pks":[],"runs_completed":[],"pick_locked":false,"pick_locked_at":null,"result_status":null,"next_wakeup":null,"skip_summary":{"best_batter":"Legacy","best_team":"CLE","best_p":0.82,"bar":0.80,"streak":10}}',
        ]
        for idx, raw in enumerate(raw_jsons):
            state_file.write_text(raw)
            loaded = scheduler.load_state(day, tmp_path)
            assert loaded is not None
            dm = scheduler.format_skip_dm(day, loaded.skip_summary)
            banner = web.render_skip_banner(loaded.skip_summary)
            if idx == 2:
                # Legacy without basis and best_p >= bar -> K3, unverified!
                assert "Top model prediction (eligibility unverified): Legacy (CLE) 82.0%, at or above the streak-10 bar (80.0%). Streak used at this check: 10." in dm
                assert "below the" not in dm and "below the" not in banner

        # >4300-digit integer in scheduler_state.json -> load_state quarantines as corrupt ValueError; web.load_scheduler_state returns {}
        huge_digits = "9" * 5000
        state_file.write_text('{"date":"2027-07-15","skip_summary":{"streak":' + huge_digits + '}}')
        old_picks_dir = web.PICKS_DIR
        try:
            web.PICKS_DIR = tmp_path
            web_loaded = web.load_scheduler_state(day)
            assert web_loaded == {}
            assert web.render_skip_banner(web_loaded.get("skip_summary")) == ""
        finally:
            web.PICKS_DIR = old_picks_dir
        with redirect_stderr(io.StringIO()):
            assert scheduler.load_state(day, tmp_path) is None

        # Check in-memory 5000-digit int (1 << 17000) directly passed to format_skip_summary!
        in_mem_huge_streak = {"best_batter": "A", "best_team": "T", "best_p": 0.75, "streak": 1 << 17000}
        try:
            _ = scheduler.format_skip_summary(in_mem_huge_streak)
            in_mem_huge_streak_raised = None
        except Exception as exc:
            in_mem_huge_streak_raised = f"{type(exc).__name__}: {exc}"

    # Cross-version state compatibility via base_63669c42 subprocess
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        env_base = os.environ.copy()
        env_base["PYTHONPATH"] = f"{BASE_SRC}:{REPO}"
        # 1. Write state with base_63669c42
        subprocess.run(
            [str(REPO / ".venv/bin/python"), "-c",
             "import sys; from pathlib import Path; from bts import scheduler; "
             "p = Path(sys.argv[1]); "
             "st = scheduler.SchedulerState(date='2027-07-15', schedule_fetched_at='2027-07-15T10:00:00-04:00', "
             "games=[], confirmed_game_pks=[], runs_completed=[], pick_locked=False, pick_locked_at=None, "
             "result_status=None, next_wakeup=None, "
             "skip_summary={'best_batter':'BaseBatter','best_team':'NYY','best_p':0.77,'bar':0.80,'streak':12}); "
             "scheduler.save_state(st, p)",
             str(tmp_path)],
            env=env_base, cwd=REPO, check=True, timeout=15,
        )
        head_loaded = scheduler.load_state("2027-07-15", tmp_path)
        assert head_loaded is not None and "basis" not in head_loaded.skip_summary
        dm_from_base = scheduler.format_skip_dm("2027-07-15", head_loaded.skip_summary)
        banner_from_base = web.render_skip_banner(head_loaded.skip_summary)
        assert dm_from_base == (
            "BTS 2027-07-15: No pick at this check. Top model prediction (eligibility unverified): "
            "BaseBatter (NYY) 77.0%, below the streak-12 bar (80.0%) using unrounded values. "
            "Streak used at this check: 12."
        )
        assert "NO PICK AT LAST CHECK" in banner_from_base

        # 2. Write state with HEAD (includes basis and candidate_verified) and read with base_63669c42
        head_loaded.skip_summary = {
            "best_batter": "HeadBatter", "best_team": "CLE", "best_p": 0.75,
            "bar": 0.796, "streak": 10, "objective": "reach57",
            "basis": "decision", "candidate_verified": True,
        }
        scheduler.save_state(head_loaded, tmp_path)
        proc = subprocess.run(
            [str(REPO / ".venv/bin/python"), "-c",
             "import json, sys; from pathlib import Path; from bts import scheduler, web; "
             "p = Path(sys.argv[1]); st = scheduler.load_state('2027-07-15', p); "
             "web.PICKS_DIR = p; ws = web.load_scheduler_state('2027-07-15'); "
             "print(json.dumps({'loaded': st.skip_summary, 'dm': scheduler.format_skip_dm('2027-07-15', st.skip_summary), 'banner': web.render_skip_banner(ws['skip_summary'])}))",
             str(tmp_path)],
            env=env_base, cwd=REPO, capture_output=True, text=True, check=True, timeout=15,
        )
        base_readback = json.loads(proc.stdout)
        assert base_readback["loaded"]["basis"] == "decision"
        assert base_readback["loaded"]["candidate_verified"] is True

    print(f"Tested {tested} renderer combinations; case counts: {k_counts}")
    print("Extreme float 1e308 outputs:", extreme_float_outputs[:2])
    print("In-memory 1<<17000 streak result:", in_mem_huge_streak_raised)
    print("Cross-version state compatibility: PASS")


if __name__ == "__main__":
    main()
