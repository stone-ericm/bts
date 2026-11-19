"""Adversarial probe for Item 1: B1 and stderr/receipt rendering across stems and reasons."""
from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import re
import signal
import tempfile
import unicodedata
from zoneinfo import ZoneInfo

from click.testing import CliRunner

from bts import cli, picks, reconcile_receipt


ET = ZoneInfo("America/New_York")
RUN = datetime(2027, 4, 20, 2, 0, tzinfo=ET)
TODAY = "2027-04-20"
PAST = "2027-04-16"

FIXED_REASONS = (
    "pick file unreadable or not valid JSON",
    "today's result is applied to streak.json but not yet saved on the pick",
    "no final result on the pick",
    "hit pick could not be loaded",
)


def _invoke_reconcile(picks_dir: Path, *, lookback: int = 0) -> tuple[object, list[dict]]:
    real_cls = reconcile_receipt.ReconcileReceipt
    orig_init = reconcile_receipt.ReconcileReceipt
    reconcile_receipt.ReconcileReceipt = lambda **kw: real_cls(clock=lambda: RUN, lookback_days=kw["lookback_days"])
    try:
        res = CliRunner().invoke(cli.cli, ["reconcile", "--picks-dir", str(picks_dir), "--lookback", str(lookback)])
    finally:
        reconcile_receipt.ReconcileReceipt = orig_init
    recs = reconcile_receipt.discover(picks_dir, TODAY)
    return res, recs


def check_regex_exhaustive() -> dict:
    nd_chars = []
    other_digit_matches = []
    suffix_matches = []
    for cp in range(0x110000):
        ch = chr(cp)
        if picks._ISO_DATE_RE.match(f"2027-04-1{ch}"):
            if unicodedata.category(ch) == "Nd":
                nd_chars.append(cp)
            else:
                other_digit_matches.append(cp)
        if picks._ISO_DATE_RE.match(f"2027-04-16{ch}"):
            suffix_matches.append((cp, repr(ch)))
    two_char_checks = {}
    for label, suf in [
        ("lf_lf", "\n\n"),
        ("cr_lf", "\r\n"),
        ("lf_cr", "\n\r"),
        ("cr", "\r"),
        ("ls", "\u2028"),
        ("ps", "\u2029"),
        ("nel", "\x85"),
        ("vt", "\x0b"),
        ("ff", "\x0c"),
        ("nul", "\x00"),
        ("backslash_n", r"\n"),
        ("surrogate_ff", "\udcff"),
    ]:
        two_char_checks[label] = bool(picks._ISO_DATE_RE.match(f"2027-04-16{suf}"))
    return {
        "nd_digit_codepoints_matched": len(nd_chars),
        "non_nd_digit_matches": other_digit_matches,
        "single_char_suffix_matches": suffix_matches,
        "extra_suffix_matches": two_char_checks,
    }


def check_fixed_reasons_and_checklist() -> dict:
    checklist = (
        Path(__file__).resolve().parents[2] / "docs" / "ops" / "2027-season-start.md"
    ).read_text()
    reason_results = {}
    for r in FIXED_REASONS:
        escaped = r.encode("unicode_escape").decode("ascii")
        reason_results[r] = {
            "escaped": escaped,
            "unchanged": escaped == r,
        }
    expected_quote = "Season replay unavailable for <date> (no final result on the pick); streak retained, not recalculated."
    return {
        "reasons": reason_results,
        "all_reasons_unchanged": all(v["unchanged"] for v in reason_results.values()),
        "checklist_contains_verbatim_quote": expected_quote in checklist,
    }


def check_real_filesystem_stems() -> dict:
    out = {}
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        # Test non-UTF-8 filename creation on this OS/filesystem
        raw_dir = base / "raw_fs"
        raw_dir.mkdir()
        try:
            raw_path = os.fsencode(str(raw_dir)) + b"/2027-04-16\xff.json"
            with open(raw_path, "wb") as fh:
                fh.write(b"{}")
            out["fs_allows_non_utf8"] = True
        except OSError as exc:
            out["fs_allows_non_utf8"] = False
            out["fs_non_utf8_error"] = f"{type(exc).__name__}: {exc}"

        stems = [
            ("ordinary", "2027-04-16", "{}"),
            ("trailing_lf_pending", "2027-04-16\n", "{}"),
            ("trailing_lf_corrupt", "2027-04-16\n", "{"),
            ("trailing_lf_lost_hit", "2027-04-16\n", '{"result": "hit"}'),
            ("arabic_indic_digits", "2027-03-\u0661\u0666", "{}"),
            ("devanagari_digits", "2027-03-\u0967\u096c", "{}"),
            ("fullwidth_digits", "2027-03-\uff11\uff16", "{}"),
            ("osmanya_smp_digits", "2027-03-\U000104a1\U000104a6", "{}"),
            ("arabic_plus_lf", "2027-03-\u0661\u0666\n", "{}"),
        ]
        for label, stem, payload in stems:
            pdir = base / label / "picks"
            pdir.mkdir(parents=True)
            (pdir / "streak.json").write_text(json.dumps({"streak": 9, "season": 2027}))
            (pdir / f"{stem}.json").write_text(payload)
            detail = picks._replay_season_streak_detail(pdir, 2027, TODAY)
            res, recs = _invoke_reconcile(pdir, lookback=8)
            stderr_lines = res.stderr.splitlines()
            out[label] = {
                "detail": detail,
                "exit_code": res.exit_code,
                "stderr": res.stderr,
                "stderr_line_count": len(stderr_lines),
                "stderr_newline_count": res.stderr.count("\n"),
                "EndsWithSingleLF": res.stderr.endswith("\n") and "\n" not in res.stderr[:-1],
                "stdout": res.stdout,
                "receipt_replay": recs[0]["replay"] if recs else None,
                "receipt_matches_raw_stem": (recs[0]["replay"]["blocked_date"] == stem) if recs else False,
            }

        # All four real reasons end-to-end
        real_reasons = {}
        for kind in ("corrupt", "today_applied", "pending", "lost_hit"):
            pdir = base / f"reason_{kind}" / "picks"
            pdir.mkdir(parents=True)
            (pdir / "streak.json").write_text(
                json.dumps({"streak": 5, "season": 2027, "applied_dates": [TODAY] if kind == "today_applied" else []})
            )
            day = TODAY if kind == "today_applied" else PAST
            body = {"corrupt": "{", "lost_hit": '{"result": "hit"}'}.get(kind, "{}")
            (pdir / f"{day}.json").write_text(body)
            res, recs = _invoke_reconcile(pdir, lookback=0)
            real_reasons[kind] = {
                "exit_code": res.exit_code,
                "stderr": res.stderr,
                "stderr_line_count": len(res.stderr.splitlines()),
                "receipt_replay": recs[0]["replay"],
            }
        out["all_four_real_reasons"] = real_reasons
    return out


def check_synthetic_hostile_strings() -> dict:
    cases = [
        ("lf", "2027-04-16\n", "no final result on the pick"),
        ("cr", "2027-04-16\r", "no final result on the pick"),
        ("crlf", "2027-04-16\r\n", "reason\r\nline2"),
        ("tab_vt_ff", "2027-04-16\t\x0b\x0c", "reason\t\x0b\x0c"),
        ("nul_bel_bs_esc_del", "2027-04-16\x00\x07\x08\x1b[31m\x7f", "r\x00\x1b[0m\x7f"),
        ("fs_gs_rs_us_nel_csi", "2027-04-16\x1c\x1d\x1e\x1f\x85\x9b", "r\x85\x9b"),
        ("unicode_ls_ps", "2027-04-16\u2028\u2029", "r\u2028\u2029"),
        ("backslash", r"2027-04-16\n\x00\\", r"reason\with\backslash"),
        ("quotes", "2027-04-16'\"`", "today's \"quoted\" reason"),
        ("non_ascii_digits", "2027-\u0660\u0664-\U000104a1\U000104a6", "non-ascii \u2014 emdash"),
        ("surrogate_escape", "2027-04-16\udcff\udc80", "surrogate \udcff reason"),
    ]
    out = {}
    orig_reconcile = picks.reconcile_results
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        for label, b_date, b_reason in cases:
            pdir = base / label / "picks"
            pdir.mkdir(parents=True)
            (pdir / "streak.json").write_text(json.dumps({"streak": 3, "season": 2027}))

            def _stub(*args, receipt=None, _d=b_date, _r=b_reason, **kwargs):
                receipt.replayed(None, blocked_date=_d, reason=_r)
                return []

            picks.reconcile_results = _stub
            try:
                res, recs = _invoke_reconcile(pdir, lookback=0)
            finally:
                picks.reconcile_results = orig_reconcile

            out[label] = {
                "exit_code": res.exit_code,
                "exception": repr(res.exception) if res.exception else None,
                "stderr": res.stderr,
                "stderr_splitlines_len": len(res.stderr.splitlines()),
                "only_printable_ascii_before_final_lf": (
                    res.stderr.endswith("\n")
                    and all(0x20 <= ord(c) <= 0x7E for c in res.stderr[:-1])
                ),
                "receipt_count": len(recs),
                "receipt_replay": recs[0]["replay"] if recs else None,
                "receipt_raw_preserved": (
                    recs[0]["replay"] == {"state": "unavailable", "blocked_date": b_date, "reason": b_reason}
                ) if recs else False,
            }
    return out


def main() -> None:
    signal.alarm(30)
    results = {
        "regex_exhaustive": check_regex_exhaustive(),
        "fixed_reasons_and_checklist": check_fixed_reasons_and_checklist(),
        "real_filesystem_stems": check_real_filesystem_stems(),
        "synthetic_hostile_strings": check_synthetic_hostile_strings(),
    }
    out_path = Path(__file__).with_name("item1_results.json")
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=True))
    print(f"Wrote {out_path}")
    print("all_reasons_unchanged:", results["fixed_reasons_and_checklist"]["all_reasons_unchanged"])
    print("checklist_verbatim:", results["fixed_reasons_and_checklist"]["checklist_contains_verbatim_quote"])
    print("single_char_suffix_matches:", results["regex_exhaustive"]["single_char_suffix_matches"])
    print("fs_allows_non_utf8:", results["real_filesystem_stems"]["fs_allows_non_utf8"])
    for k, v in results["real_filesystem_stems"].items():
        if isinstance(v, dict) and "stderr_line_count" in v:
            print(f"  real[{k}]: exit={v['exit_code']} lines={v['stderr_line_count']} raw_ok={v['receipt_matches_raw_stem']} stderr={v['stderr']!r}")
    for k, v in results["synthetic_hostile_strings"].items():
        print(f"  synth[{k}]: exit={v['exit_code']} lines={v['stderr_splitlines_len']} ascii_ok={v['only_printable_ascii_before_final_lf']} raw_ok={v['receipt_raw_preserved']} exc={v['exception']}")


if __name__ == "__main__":
    main()
