"""Probe for Item 3 (The renderers cannot raise or mislead on bad data):
Tests format_skip_summary, format_skip_dm, journal line, and web.render_skip_banner over:
- Hostile decoded JSON types, NaN, +-inf, 1e308, 1.5e308, 10**400, -0.1, 1.0000001, 2, -0.0, 0, 1,
  4300-digit ints, booleans, nested dicts/lists, HTML tags/entities/quotes, Unicode line breaks,
  whitespace-only reason, missing/legacy keys, non-dict summaries
- Cross-version scheduler_state.json round-trips (base 63669c42 -> HEAD and HEAD -> base 63669c42),
  corrupt JSON quarantine, over-4300-digit JSON int quarantine, and directory-path read
- Asserts: zero exceptions on decoded summaries; single-line journal output; zero unescaped HTML in banner;
  'below the' appears ONLY in case K2.
"""
from __future__ import annotations

import itertools
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent

from bts import scheduler, web


_ALLOWED_ENTITIES = re.compile(r"&(?:amp|lt|gt|quot|#x27);")


def assert_banner_safe(banner: str) -> None:
    if banner == "":
        return
    prefix = '<div class="skip-banner"><strong>NO PICK AT LAST CHECK</strong><span>'
    suffix = "</span></div>"
    assert banner.startswith(prefix) and banner.endswith(suffix), banner
    inner = banner[len(prefix):-len(suffix)]
    assert "<" not in inner and ">" not in inner and '"' not in inner and "'" not in inner, inner
    stripped = _ALLOWED_ENTITIES.sub("", inner)
    assert "&" not in stripped, f"Unescaped & in banner: {inner}"


def assert_single_line(text: str) -> None:
    for ch in ("\n", "\r", "\v", "\f", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029"):
        assert ch not in text, f"Line break {ch!r} remained in rendered text: {text!r}"


def is_case_k2(summary: object) -> bool:
    if not isinstance(summary, dict):
        return False
    raw_reason = summary.get("reason")
    reason = " ".join(raw_reason.splitlines()).strip() if isinstance(raw_reason, str) else ""
    if reason:
        return False
    if summary.get("basis") == "no_decision":
        return False
    p = scheduler._skip_display_number(summary.get("best_p"))
    b = scheduler._skip_display_number(summary.get("bar"))
    return p is not None and b is not None and p < b


def load_base_scheduler() -> ModuleType:
    src = subprocess.run(
        ["git", "show", "63669c42:src/bts/scheduler.py"],
        cwd=REPO, check=True, capture_output=True, text=True, timeout=10,
    ).stdout
    mod = ModuleType("bts._base_63669c42_scheduler")
    exec(compile(src, "63669c42:src/bts/scheduler.py", "exec"), mod.__dict__)
    return mod


def main() -> None:
    names = [None, "", "<script>alert('x')</script>&\"'", "Line1\nLine2\r\nLine3\u2028Line4\u2029", ["list"], {"k": 1}, True, 42]
    teams = [None, "<NY&M>\"'", "A\r\nB", [], 0]
    nums = [
        None, True, False, "0.75", "", [], {"a": 1},
        float("nan"), float("inf"), float("-inf"),
        1e308, 1.5e308, -1e308, 10**400,
        -0.1, -1e-9, -0.0, 0, 0.0, 0.75, 0.7501, 1, 1.0, 1.0000001, 2,
    ]
    streaks = [None, True, False, -1, 0, 10, 10**100, "10", 5.5, []]
    reasons = [None, "", " \r\n\t\u2028 ", "Stopped", "Stopped.", "Stopped!", "Stopped?", "<bad & \"reason\">\nnext", ["r"], 99]
    bases = [None, "decision", "no_decision", "other", 123]
    verifieds = [None, True, False, 1, "true"]

    total = 0
    k2_count = 0
    # Exhaustive check over core combinations + targeted hostile sweep
    for b_p, bar_v, st_v, r_v, bas_v, ver_v in itertools.product(
        nums,
        [None, True, "0.8", float("nan"), float("inf"), 1e308, 1.5e308, -0.1, 0.0, 0.75, 0.80, 1.0, 1.0000001],
        [None, True, -1, 0, 8, "8", 10**200],
        [None, "", " \r\n\t ", "Tail stop", "Tail stop!", "<tag & \"q\">\nline2", []],
        [None, "decision", "no_decision", []],
        [None, True, False, 1],
    ):
        name_v = names[total % len(names)]
        team_v = teams[total % len(teams)]
        summary: dict = {"best_batter": name_v, "best_team": team_v, "best_p": b_p}
        if bar_v is not None or (total % 2 == 0):
            summary["bar"] = bar_v
        if st_v is not None or (total % 3 == 0):
            summary["streak"] = st_v
        if r_v is not None:
            summary["reason"] = r_v
        if bas_v is not None:
            summary["basis"] = bas_v
        if ver_v is not None:
            summary["candidate_verified"] = ver_v

        body = scheduler.format_skip_summary(summary)
        dm_txt = scheduler.format_skip_dm("2027-07-15", summary)
        journal_txt = f"  SKIP — {body}"
        banner_txt = web.render_skip_banner(summary)

        assert_single_line(body)
        assert_single_line(dm_txt)
        assert_single_line(journal_txt)
        assert_banner_safe(banner_txt)
        assert "inf%" not in body and "nan%" not in body.lower()

        k2 = is_case_k2(summary)
        if k2:
            k2_count += 1
            assert "below the" in body
        else:
            # None of our test name/team/reason values contain the substring 'below the'
            assert "below the" not in body and "below the" not in dm_txt and "below the" not in banner_txt, (summary, body)
        total += 1

    # Also test 4300-digit integer streak (the max JSON-decodable int in Python 3.12)
    max_json_int = int("9" * 4300)
    s_huge_streak = {"best_batter": "A", "best_team": "T", "best_p": 0.75, "bar": 0.80, "streak": max_json_int}
    body_huge = scheduler.format_skip_summary(s_huge_streak)
    assert "999999" in body_huge
    assert_banner_safe(web.render_skip_banner(s_huge_streak))

    # Non-dict / empty summaries
    for non_obj in (None, {}, [], "string", 0, 42, True, False):
        assert web.render_skip_banner(non_obj) == ""
        dm_no = scheduler.format_skip_dm("2027-07-15", non_obj)
        assert dm_no == (
            "BTS 2027-07-15: No pick at this check. "
            "Top model prediction (eligibility unverified): ? (?) ?. "
            "No probability comparison is available for this check."
        )
        assert "below the" not in dm_no

    # Cross-version scheduler_state.json round-trips & corrupt/directory/huge-int disk reads
    base_sch = load_base_scheduler()
    with tempfile.TemporaryDirectory() as tmp:
        tp = Path(tmp)
        old_PICKS = web.PICKS_DIR
        try:
            web.PICKS_DIR = tp
            # Base -> HEAD (legacy summary without basis / candidate_verified)
            legacy_sum = {"best_batter": "Bo <B>", "best_team": "NYM", "best_p": 0.75, "bar": 0.80, "streak": 10}
            b_state = base_sch.SchedulerState(date="2027-07-15", skip_summary=legacy_sum)
            state_file = base_sch.save_state(b_state, tp)
            h_loaded = scheduler.load_state("2027-07-15", tp)
            w_loaded = web.load_scheduler_state("2027-07-15")
            assert h_loaded.skip_summary == legacy_sum and w_loaded["skip_summary"] == legacy_sum
            b_html = web.render_skip_banner(w_loaded.get("skip_summary"))
            assert_banner_safe(b_html)
            assert "Bo &lt;B&gt;" in b_html and "below the streak-10 bar (80.0%) using unrounded values." in b_html

            # HEAD -> Base (summary with basis and candidate_verified)
            new_sum = legacy_sum | {"basis": "decision", "candidate_verified": True}
            h_loaded.skip_summary = new_sum
            scheduler.save_state(h_loaded, tp)
            b_reloaded = base_sch.load_state("2027-07-15", tp)
            assert b_reloaded.skip_summary == new_sum

            # NaN/Infinity/1e308 written to scheduler_state.json and read back
            state_file.write_text('{"date": "2027-07-15", "skip_summary": {"best_batter": "X", "best_team": "Y", "best_p": 1e308, "bar": 1.5e308, "streak": 5}}')
            w_1e308 = web.load_scheduler_state("2027-07-15")
            b_1e308 = web.render_skip_banner(w_1e308.get("skip_summary"))
            assert_banner_safe(b_1e308)
            assert "inf%" not in b_1e308 and "?." in b_1e308

            # >4300-digit integer in JSON file: load_state quarantines cleanly; web.load_scheduler_state returns {}
            state_file.write_text('{"date": "2027-07-15", "skip_summary": {"streak": ' + ("9" * 5000) + '}}')
            assert web.load_scheduler_state("2027-07-15") == {}
            assert scheduler.load_state("2027-07-15", tp) is None

            # Corrupt JSON -> quarantined
            state_file.write_text("{corrupt")
            assert web.load_scheduler_state("2027-07-15") == {}
            assert scheduler.load_state("2027-07-15", tp) is None

            # Directory at scheduler_state.json -> web returns {}, scheduler raises IsADirectoryError
            state_file.mkdir()
            assert web.load_scheduler_state("2027-07-15") == {}
            raised_dir = False
            try:
                scheduler.load_state("2027-07-15", tp)
            except IsADirectoryError:
                raised_dir = True
            assert raised_dir
        finally:
            web.PICKS_DIR = old_PICKS

    res = {
        "hostile_summaries_tested": total,
        "k2_cases": k2_count,
        "non_k2_cases": total - k2_count,
        "zero_exceptions_on_decoded_json": True,
        "all_banners_escaped": True,
        "below_the_strictly_k2_only": True,
        "base_head_roundtrip_ok": True,
    }
    out_path = OUT_DIR / "probe_3_results.json"
    out_path.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
