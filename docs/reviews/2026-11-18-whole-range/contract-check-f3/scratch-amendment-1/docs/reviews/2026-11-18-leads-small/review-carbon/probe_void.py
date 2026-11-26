"""Adversarial probe for parse_result_from_reply and feed regeneration (base df454725 vs HEAD b0c11135).

Checks:
1. Full format_result_reply round-trip across hit/miss/void and streaks 0..60 + surrounding whitespace.
2. Near-misses (prefix/suffix text, case variations, negative/float streaks, emoji + void text,
   multi-line text) and exhaustive scan of all 0x110000 Unicode codepoints for \\d vs int().
3. Full timeline + CLI `bts state regenerate` + `bts state verify` + `bts state export` across
   every void position (first day, first-and-only post-cutoff day, between hits, at streak 10-15
   before a miss both with prior feed hit and right after cutoff, last day, double-down void),
   comparing base vs HEAD and verifying BTS pass rules.
4. Downstream consumers of regenerated single/double-down void pick files: load_pick,
   streak_increment_for_resolved_hit, _replay_season_streak, export_initial_state, diff_pick_files,
   health.result_resolution, health.predicted_vs_realized, health.same_team_corr, model.calibrate.
"""
from __future__ import annotations

import importlib.util
import json
import re
import tempfile
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from click.testing import CliRunner

from bts.cli import cli
from bts.health import predicted_vs_realized, result_resolution, same_team_corr
from bts.model import calibrate
from bts.picks import _replay_season_streak, load_pick, streak_increment_for_resolved_hit
from bts.posting import format_post, format_result_reply, format_skip_post
from bts.state import regenerate as head_regen
from bts.state.export import UnresolvedPickError, export_initial_state
from bts.state.verify import diff_pick_files

REVIEW_DIR = Path(__file__).resolve().parent
BASE_REGEN_PATH = REVIEW_DIR / "base_ref" / "src" / "bts" / "state" / "regenerate.py"


def _load_base_regenerate():
    spec = importlib.util.spec_from_file_location("_leads_review_base_regenerate", BASE_REGEN_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    import sys as _sys
    _sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        _sys.modules.pop(spec.name, None)
    return mod


def _fake_bsky_entry(uri: str, created_at: str, text: str, is_reply: bool):
    record = SimpleNamespace(
        created_at=created_at,
        text=text,
        reply=SimpleNamespace(parent=uri) if is_reply else None,
    )
    return SimpleNamespace(post=SimpleNamespace(uri=uri, record=record))


class _FakeBskyClient:
    def __init__(self, entries: list):
        self._entries = list(reversed(entries))  # real Bluesky author feed returns newest-first

    def get_author_feed(self, actor: str, limit: int = 100, cursor=None):
        return SimpleNamespace(feed=self._entries, cursor=None)


def check_roundtrip_and_nearmisses(base_regen) -> dict:
    # 1. Every string format_result_reply produces for streaks 0..60
    for streak in range(61):
        h_hit = head_regen.parse_result_from_reply(format_result_reply("hit", streak))
        b_hit = base_regen.parse_result_from_reply(format_result_reply("hit", streak))
        assert (h_hit.is_result, h_hit.result, h_hit.streak_after) == (True, "hit", streak)
        assert (b_hit.is_result, b_hit.result, b_hit.streak_after) == (True, "hit", streak)

        h_miss = head_regen.parse_result_from_reply(format_result_reply("miss", streak))
        b_miss = base_regen.parse_result_from_reply(format_result_reply("miss", streak))
        assert (h_miss.is_result, h_miss.result, h_miss.streak_after) == (True, "miss", 0)
        assert (b_miss.is_result, b_miss.result, b_miss.streak_after) == (True, "miss", 0)

        h_void = head_regen.parse_result_from_reply(format_result_reply("void", streak))
        b_void = base_regen.parse_result_from_reply(format_result_reply("void", streak))
        assert (h_void.is_result, h_void.result, h_void.streak_after) == (True, "void", streak)
        assert (b_void.is_result, b_void.result, b_void.streak_after) == (False, None, None)

    # Surrounding whitespace allowed
    for ws_text, expected_n in (
        ("  Void. Streak: 0  ", 0),
        ("\n\tVoid. Streak: 15\r\n", 15),
        (" Void. Streak: 57\n", 57),
    ):
        p = head_regen.parse_result_from_reply(ws_text)
        assert (p.is_result, p.result, p.streak_after) == (True, "void", expected_n)

    # 2. Near-misses without emoji -> must stay is_result=False (matching base)
    non_result_near_misses = [
        "Note: Void. Streak: 12",
        "Avoid. Streak: 12",
        "Void. Streak: 12.",
        "Void. Streak: 12 (rain)",
        "void. streak: 12",
        "VOID. Streak: 12",
        "Void. streak: 12",
        "void. Streak: 12",
        "Void. Streak: -1",
        "Void. Streak: +12",
        "Void. Streak: 12.0",
        "Void. Streak: ",
        "Void.  Streak: 12",
        "Void. Streak:12",
        "Void Streak: 12",
        "Void. Streak: 12\nExtra line",
        "First line\nVoid. Streak: 12",
        "Void.\nStreak: 12",
        "",
        "   ",
    ]
    for txt in non_result_near_misses:
        hp = head_regen.parse_result_from_reply(txt)
        bp = base_regen.parse_result_from_reply(txt)
        assert not hp.is_result and hp.result is None and hp.streak_after is None, f"false positive on {txt!r}"
        assert (hp.is_result, hp.result, hp.streak_after) == (bp.is_result, bp.result, bp.streak_after)

    # Emoji near-misses -> must parse identically to base (never become result="void")
    emoji_near_misses = [
        "Void. Streak: 12 \u2705",
        "\u2705 Void. Streak: 12",
        "Void. Streak: 12 \u274c",
        "\u274c Void. Streak: 12",
        "Void. Streak: 12 \u2705 \u274c",
    ]
    emoji_outcomes = {}
    for txt in emoji_near_misses:
        hp = head_regen.parse_result_from_reply(txt)
        bp = base_regen.parse_result_from_reply(txt)
        assert hp.result != "void"
        assert (hp.is_result, hp.result, hp.streak_after) == (bp.is_result, bp.result, bp.streak_after)
        emoji_outcomes[txt] = (hp.is_result, hp.result, hp.streak_after)

    # 3. Exhaustive Unicode \d vs int() scan over all 1,114,112 Unicode codepoints
    digit_re = re.compile(r"\d")
    matched_cps = []
    int_failures = []
    for cp in range(0x110000):
        ch = chr(cp)
        if digit_re.fullmatch(ch):
            matched_cps.append(cp)
            try:
                v = int(ch)
                assert isinstance(v, int) and 0 <= v <= 9
            except Exception as exc:
                int_failures.append((cp, repr(exc)))
    assert len(int_failures) == 0

    non_ascii_sample = head_regen.parse_result_from_reply("Void. Streak: \u0661\u0662")  # Arabic-Indic 12
    assert (
        non_ascii_sample.is_result is True
        and non_ascii_sample.result == "void"
        and non_ascii_sample.streak_after == 12
        and type(non_ascii_sample.streak_after) is int
    )

    return {
        "roundtrip_streaks_0_60": "ok",
        "non_result_near_misses_checked": len(non_result_near_misses),
        "emoji_near_misses": emoji_outcomes,
        "unicode_digit_codepoints_matched": len(matched_cps),
        "unicode_digit_int_failures": len(int_failures),
        "non_ascii_arabic_indic_12": (non_ascii_sample.result, non_ascii_sample.streak_after),
    }


def _build_scenario_entries(days: list[dict]) -> list:
    entries = []
    for idx, d in enumerate(days, start=1):
        day_str = d["date"]
        if d.get("kind") == "skip":
            text = format_skip_post("Skip Batter", "NYM", 0.75, streak=d["streak_before"])
            entries.append(_fake_bsky_entry(f"at://did:plc:bot/post/s{idx}", f"{day_str}T15:00:00Z", text, False))
            continue
        if d.get("double_down"):
            text = format_post(
                d["batter"], d["team"], "Pitcher A", 0.82, streak=d["streak_before"],
                double=d["dd_batter"], double_p_game=0.80, double_team=d["dd_team"], double_pitcher="Pitcher B",
            )
        else:
            text = format_post(d["batter"], d["team"], "Pitcher A", 0.82, streak=d["streak_before"])
        entries.append(_fake_bsky_entry(f"at://did:plc:bot/post/p{idx}", f"{day_str}T15:00:00Z", text, False))
        if d.get("result") is not None:
            rtext = format_result_reply(d["result"], d["streak_after"])
            entries.append(_fake_bsky_entry(f"at://did:plc:bot/post/r{idx}", f"{day_str}T23:00:00Z", rtext, True))
    return entries


def check_positions_and_cli(base_regen) -> dict:
    scenarios = {
        "first_day_before_hit": {
            "snapshot_streak": 0,
            "snapshot_saver": True,
            "days": [
                {"date": "2026-05-01", "batter": "Juan Soto", "team": "NYM", "streak_before": 0, "result": "void", "streak_after": 0},
                {"date": "2026-05-02", "batter": "Bobby Witt", "team": "KC", "streak_before": 0, "result": "hit", "streak_after": 1},
            ],
            "expected_head": {"results": {"2026-05-01": "void", "2026-05-02": "hit"}, "streak": 1, "saver": True},
            "expected_base": {"results": {"2026-05-01": None, "2026-05-02": "hit"}, "streak": 1, "saver": True},
        },
        "first_and_only_post_cutoff_void": {
            "snapshot_streak": 8,
            "snapshot_saver": True,
            "days": [
                {"date": "2026-05-01", "batter": "Juan Soto", "team": "NYM", "streak_before": 8, "result": "void", "streak_after": 8},
            ],
            "expected_head": {"results": {"2026-05-01": "void"}, "streak": 8, "saver": True},
            "expected_base": {"results": {"2026-05-01": None}, "streak": 0, "saver": True},
        },
        "between_hits": {
            "snapshot_streak": 0,
            "snapshot_saver": True,
            "days": [
                {"date": "2026-05-01", "batter": "A", "team": "NYM", "streak_before": 3, "result": "hit", "streak_after": 4},
                {"date": "2026-05-02", "batter": "B", "team": "NYM", "streak_before": 4, "result": "void", "streak_after": 4},
                {"date": "2026-05-03", "batter": "C", "team": "NYM", "streak_before": 4, "result": "hit", "streak_after": 5},
            ],
            "expected_head": {"results": {"2026-05-01": "hit", "2026-05-02": "void", "2026-05-03": "hit"}, "streak": 5, "saver": True},
            "expected_base": {"results": {"2026-05-01": "hit", "2026-05-02": None, "2026-05-03": "hit"}, "streak": 5, "saver": True},
        },
        "at_12_with_prior_feed_hit_before_miss": {
            "snapshot_streak": 11,
            "snapshot_saver": True,
            "days": [
                {"date": "2026-05-01", "batter": "A", "team": "NYM", "streak_before": 11, "result": "hit", "streak_after": 12},
                {"date": "2026-05-02", "batter": "B", "team": "NYM", "streak_before": 12, "result": "void", "streak_after": 12},
                {"date": "2026-05-03", "batter": "C", "team": "NYM", "streak_before": 12, "result": "miss", "streak_after": 0},
            ],
            "expected_head": {"results": {"2026-05-01": "hit", "2026-05-02": "void", "2026-05-03": "miss"}, "streak": 0, "saver": False},
            "expected_base": {"results": {"2026-05-01": "hit", "2026-05-02": None, "2026-05-03": "miss"}, "streak": 0, "saver": False},
        },
        "at_12_first_post_cutoff_void_before_miss": {
            "snapshot_streak": 12,
            "snapshot_saver": True,
            "days": [
                {"date": "2026-05-01", "batter": "B", "team": "NYM", "streak_before": 12, "result": "void", "streak_after": 12},
                {"date": "2026-05-02", "batter": "C", "team": "NYM", "streak_before": 12, "result": "miss", "streak_after": 0},
            ],
            # Notice: before this fix, Day 1's void had streak_after=None, so Day 2's miss saw streak_before=0
            # and WRONGLY left saver=True! With the fix, Day 1 carries streak_after=12 and saver becomes False!
            "expected_head": {"results": {"2026-05-01": "void", "2026-05-02": "miss"}, "streak": 0, "saver": False},
            "expected_base": {"results": {"2026-05-01": None, "2026-05-02": "miss"}, "streak": 0, "saver": True},
        },
        "last_day_after_hit": {
            "snapshot_streak": 6,
            "snapshot_saver": True,
            "days": [
                {"date": "2026-05-01", "batter": "A", "team": "NYM", "streak_before": 6, "result": "hit", "streak_after": 7},
                {"date": "2026-05-02", "batter": "B", "team": "NYM", "streak_before": 7, "result": "void", "streak_after": 7},
            ],
            "expected_head": {"results": {"2026-05-01": "hit", "2026-05-02": "void"}, "streak": 7, "saver": True},
            "expected_base": {"results": {"2026-05-01": "hit", "2026-05-02": None}, "streak": 7, "saver": True},
        },
        "double_down_void_at_12": {
            "snapshot_streak": 11,
            "snapshot_saver": True,
            "days": [
                {"date": "2026-05-01", "batter": "A", "team": "NYM", "streak_before": 11, "result": "hit", "streak_after": 12},
                {
                    "date": "2026-05-02", "batter": "Jose Altuve", "team": "HOU",
                    "double_down": True, "dd_batter": "Kyle Tucker", "dd_team": "HOU",
                    "streak_before": 12, "result": "void", "streak_after": 12,
                },
            ],
            "expected_head": {"results": {"2026-05-01": "hit", "2026-05-02": "void"}, "streak": 12, "saver": True},
            "expected_base": {"results": {"2026-05-01": "hit", "2026-05-02": None}, "streak": 12, "saver": True},
        },
    }

    runner = CliRunner()
    summary = {}

    with tempfile.TemporaryDirectory(dir=REVIEW_DIR) as td:
        root = Path(td)
        for sname, spec in scenarios.items():
            sdir = root / sname
            sdir.mkdir()
            snap_path = sdir / "snapshot.json"
            snap_path.write_text(json.dumps({
                "version": 1,
                "exported_at": "2026-05-01T00:00:00Z",
                "cutoff_date": "2026-04-30",
                "streak_at_cutoff": spec["snapshot_streak"],
                "saver_available": spec["snapshot_saver"],
                "historical_picks": [],
            }))
            entries = _build_scenario_entries(spec["days"])

            # Run base regenerate
            base_out = sdir / "base_picks"
            with patch.object(base_regen, "_bluesky_client", return_value=_FakeBskyClient(entries)):
                base_regen.regenerate(snap_path, "beatthestreakbot.bsky.social", base_out)

            # Run HEAD CLI `bts state regenerate`
            head_out = sdir / "head_picks"
            with patch.object(head_regen, "_bluesky_client", return_value=_FakeBskyClient(entries)):
                cli_regen = runner.invoke(cli, [
                    "state", "regenerate",
                    "--snapshot", str(snap_path),
                    "--handle", "beatthestreakbot.bsky.social",
                    "--out-picks-dir", str(head_out),
                ])
            assert cli_regen.exit_code == 0, cli_regen.output

            for label, out_dir, exp in (("base", base_out, spec["expected_base"]), ("head", head_out, spec["expected_head"])):
                got_results = {
                    d["date"]: json.loads((out_dir / f"{d['date']}.json").read_text())["result"]
                    for d in spec["days"]
                }
                got_streak = json.loads((out_dir / "streak.json").read_text())
                assert got_results == exp["results"], f"{sname}/{label}: {got_results} != {exp['results']}"
                assert got_streak["streak"] == exp["streak"], f"{sname}/{label}: streak {got_streak}"
                assert got_streak["saver_available"] is exp["saver"], f"{sname}/{label}: saver {got_streak}"

            # Build authoritative "live" dir matching truth and run CLI `bts state verify`
            live_dir = sdir / "live"
            live_dir.mkdir()
            for p in head_out.iterdir():
                (live_dir / p.name).write_bytes(p.read_bytes())

            with patch.object(head_regen, "_bluesky_client", return_value=_FakeBskyClient(entries)):
                verify_res = runner.invoke(cli, [
                    "state", "verify",
                    "--live-dir", str(live_dir),
                    "--snapshot", str(snap_path),
                    "--handle", "beatthestreakbot.bsky.social",
                ])
            assert verify_res.exit_code == 0 and "Drift check CLEAN" in verify_res.output

            # Compare live_dir against base_out via diff_pick_files -> always drifted before!
            base_drift = diff_pick_files(live_dir, base_out)
            assert not base_drift.is_clean

            # Export check: base_out raises UnresolvedPickError, head_out exports cleanly!
            base_export_failed = False
            try:
                export_initial_state(base_out, sdir / "base_export.json")
            except UnresolvedPickError:
                base_export_failed = True
            assert base_export_failed
            exported = export_initial_state(head_out, sdir / "head_export.json")
            assert len(exported["historical_picks"]) == len(spec["days"])

            # Consumer checks on double_down_void_at_12 and between_hits
            if sname in ("double_down_void_at_12", "between_hits"):
                for d in spec["days"]:
                    loaded = load_pick(d["date"], head_out)
                    assert loaded is not None and loaded.result == d["result"]
                    if loaded.result == "void":
                        assert streak_increment_for_resolved_hit(loaded) == 0
                if sname == "double_down_void_at_12":
                    dd_loaded = load_pick("2026-05-02", head_out)
                    assert dd_loaded.double_down is not None and dd_loaded.double_down.batter_name == "Kyle Tucker"

                # _replay_season_streak succeeds on head_out, returns None on base_out (unresolved null)
                assert _replay_season_streak(base_out, 2026, "2026-05-10") is None
                replayed = _replay_season_streak(head_out, 2026, "2026-05-10")
                assert replayed is not None

                # Health & calibration consumers do not raise or false-alert on head_out
                assert result_resolution.check(head_out, today=date(2026, 5, 10)) == []
                assert len(result_resolution.check(base_out, today=date(2026, 5, 10))) >= 1
                predicted_vs_realized.check(head_out, today=date(2026, 5, 10))
                same_team_corr.check(head_out, today=date(2026, 5, 10))
                empty_pa = pd.DataFrame(columns=["batter_id", "date", "is_hit", "is_resumed_portion"])
                calibrate.fit_calibrator_from_picks(head_out, empty_pa, today=date(2026, 5, 10))

            summary[sname] = {
                "base_results": spec["expected_base"]["results"],
                "base_streak_saver": (spec["expected_base"]["streak"], spec["expected_base"]["saver"]),
                "head_results": spec["expected_head"]["results"],
                "head_streak_saver": (spec["expected_head"]["streak"], spec["expected_head"]["saver"]),
                "base_drift_issues": base_drift.issues,
                "head_verify_clean": True,
                "head_export_ok": True,
            }

    return summary


def main() -> None:
    base_regen = _load_base_regenerate()
    report = {
        "roundtrip_and_nearmisses": check_roundtrip_and_nearmisses(base_regen),
        "positions_and_consumers": check_positions_and_cli(base_regen),
    }
    out_path = REVIEW_DIR / "probe_void_results.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
