"""Probe family 3 & 4: exhaustive A6 record validation, processed-day window, real attention+dispatcher DM flow, and character-for-character A7/stderr/CLAUDE.md checks."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

REVIEW_DIR = Path(__file__).resolve().parent
REPO_ROOT = REVIEW_DIR.parents[1]
HEAD_SRC = REPO_ROOT / "src"

sys.path.insert(0, str(HEAD_SRC))

from bts.health import alert as alert_mod, attention as attention_mod, probable_pitcher_cache as h_ppc, runner
from bts.probable_pitcher_status import MAX_BYTES, SCHEMA, SOURCE, ET


DAY = date(2026, 12, 31)
ID_A = "0123456789abcdef" * 4
ID_B = "fedcba9876543210" * 4
SHA_ORIG = "a1b2c3d4e5f60718" * 4
PRESERVED_NAME = "probable_pitcher_lookup.20261231T140000Z.a1b2c3d4e5f6.json"


def base_record(day: date = DAY, reason: str = "invalid_entries") -> tuple[str, dict]:
    ident = ID_A if reason == "invalid_entries" else ID_B
    inv = reason == "invalid_entries"
    rec = {
        "schema_version": SCHEMA,
        "reason": reason,
        "recorded_at": f"{day.isoformat()}T14:00:00+00:00",
        "recorded_date_et": day.isoformat(),
        "cache_path": "data/models/probable_pitcher_lookup.json",
        "dropped": 2 if inv else None,
        "recovered_from_cache": 0 if inv else None,
        "rescanned": 1,
        "unusable_feed": 0 if inv else None,
        "without_feed": 1 if inv else None,
        "game_ids": [100, 101] if inv else [],
        "identity": ident,
        "preserved": PRESERVED_NAME,
        "original_sha256": SHA_ORIG,
    }
    return f"{day.isoformat()}.{reason}.{ident[:12]}.json", rec


def test_a6_validation(root: Path) -> None:
    print("=== 3(a). Exhaustive A6 record validation (accepted vs rejected) ===")
    # 1. Valid envelopes that MUST be accepted
    valid_variants = []

    # invalid_entries singular
    n1, r1 = base_record()
    r1.update(dropped=1, rescanned=0, without_feed=1, game_ids=[100], preserved=None)
    valid_variants.append(("inv_singular_null_preserved", n1, json.dumps(r1).encode()))

    # invalid_entries multiple aliases of single game (dropped=3, len(game_ids)=1 <= min(10, 3))
    n2, r2 = base_record()
    r2.update(dropped=3, recovered_from_cache=3, rescanned=0, without_feed=0, game_ids=[100])
    valid_variants.append(("inv_alias_3_drops_1_id", n2, json.dumps(r2).encode()))

    # invalid_entries 15 drops with 10 sampled IDs
    n3, r3 = base_record()
    r3.update(dropped=15, without_feed=14, rescanned=1, game_ids=list(range(100, 110)))
    valid_variants.append(("inv_15_drops_10_ids", n3, json.dumps(r3).encode()))

    # exact 64 KiB (MAX_BYTES) valid JSON (padded with trailing spaces)
    n4, r4 = base_record()
    compact = json.dumps(r4).encode("utf-8")
    padded_64k = compact + b" " * (MAX_BYTES - len(compact))
    assert len(padded_64k) == MAX_BYTES
    valid_variants.append(("exact_64kib", n4, padded_64k))

    # unreadable_cache with original_sha256=None, preserved=None, rescanned=0
    n5, r5 = base_record(reason="unreadable_cache")
    r5.update(rescanned=0, preserved=None, original_sha256=None)
    valid_variants.append(("unreadable_nulls_0_games", n5, json.dumps(r5).encode()))

    # unreadable_cache with rescanned=1 and preserved copy
    n6, r6 = base_record(reason="unreadable_cache")
    r6.update(rescanned=1)
    valid_variants.append(("unreadable_1_game_preserved", n6, json.dumps(r6).encode()))

    for label, fname, payload in valid_variants:
        sdir = root / f"valid_{label}"
        sdir.mkdir()
        (sdir / fname).write_bytes(payload)
        alerts = h_ppc.check(sdir, DAY)
        assert len(alerts) == 1, (label, alerts)
        assert alerts[0].level == "WARN" and alerts[0].source == SOURCE
        assert ":status_unreadable:" not in alerts[0].incident_key, (label, alerts[0])
        expected_id = json.loads(payload)["identity"]
        assert alerts[0].incident_key == f"{SOURCE}:{DAY.isoformat()}:{expected_id}"

    # 2. Invalid envelopes / files that MUST produce status_unreadable WARN
    base_name, base_row = base_record()
    bad_mutations = [
        ("missing_field", base_name, {k: v for k, v in base_row.items() if k != "preserved"}),
        ("extra_field", base_name, {**base_row, "extra": 1}),
        ("bad_schema", base_name, {**base_row, "schema_version": "v0"}),
        ("bad_reason_str", base_name, {**base_row, "reason": "other"}),
        ("bad_reason_list", base_name, {**base_row, "reason": []}),
        ("bad_reason_null", base_name, {**base_row, "reason": None}),
        ("upper_identity", f"{DAY}.invalid_entries.{ID_A[:12].upper()}.json", {**base_row, "identity": ID_A.upper()}),
        ("short_identity", base_name, {**base_row, "identity": ID_A[:63]}),
        ("prefix_mismatch", base_name, {**base_row, "identity": "f" * 12 + ID_A[12:]}),
        ("date_mismatch_name", f"{DAY - timedelta(days=1)}.invalid_entries.{ID_A[:12]}.json", base_row),
        ("empty_cache_path", base_name, {**base_row, "cache_path": ""}),
        ("non_str_cache_path", base_name, {**base_row, "cache_path": 123}),
        ("preserved_traversal", base_name, {**base_row, "preserved": f"../{PRESERVED_NAME}"}),
        ("preserved_subdir", base_name, {**base_row, "preserved": f"sub/{PRESERVED_NAME}"}),
        ("preserved_newline", base_name, {**base_row, "preserved": f"{PRESERVED_NAME}\n"}),
        ("preserved_upper_hex", base_name, {**base_row, "preserved": "probable_pitcher_lookup.20261231T140000Z.A1B2C3D4E5F6.json"}),
        ("preserved_bad_prefix", base_name, {**base_row, "preserved": "other.20261231T140000Z.a1b2c3d4e5f6.json"}),
        ("sha_upper", base_name, {**base_row, "original_sha256": SHA_ORIG.upper()}),
        ("sha_short", base_name, {**base_row, "original_sha256": "abc"}),
        ("ts_z_suffix", base_name, {**base_row, "recorded_at": "2026-12-31T14:00:00Z"}),
        ("ts_non_utc_offset", base_name, {**base_row, "recorded_at": "2026-12-31T09:00:00-05:00"}),
        ("ts_minyear_overflow", base_name, {**base_row, "recorded_at": "0001-01-01T01:00:00+00:00"}),
        ("ts_et_disagree", base_name, {**base_row, "recorded_at": "2027-01-01T05:00:00+00:00"}),
        ("ids_not_list", base_name, {**base_row, "game_ids": "100"}),
        ("ids_empty_on_invalid", base_name, {**base_row, "game_ids": []}),
        ("ids_11", base_name, {**base_row, "dropped": 11, "without_feed": 10, "game_ids": list(range(100, 111))}),
        ("ids_more_than_dropped", base_name, {**base_row, "dropped": 1, "rescanned": 1, "without_feed": 0, "game_ids": [100, 101]}),
        ("ids_unsorted", base_name, {**base_row, "game_ids": [101, 100]}),
        ("ids_duplicate", base_name, {**base_row, "game_ids": [100, 100]}),
        ("ids_bool", base_name, {**base_row, "dropped": 1, "rescanned": 1, "without_feed": 0, "game_ids": [True]}),
        ("ids_float", base_name, {**base_row, "dropped": 1, "rescanned": 1, "without_feed": 0, "game_ids": [100.0]}),
        ("ids_unhashable", base_name, {**base_row, "dropped": 1, "rescanned": 1, "without_feed": 0, "game_ids": [[100]]}),
        ("dropped_zero", base_name, {**base_row, "dropped": 0, "rescanned": 0, "without_feed": 0}),
        ("counts_sum_mismatch", base_name, {**base_row, "without_feed": 2}),
        ("count_bool", base_name, {**base_row, "rescanned": True}),
        ("count_negative", base_name, {**base_row, "recovered_from_cache": -1, "without_feed": 2}),
        ("rescanned_bool_unreadable", f"{DAY}.unreadable_cache.{ID_B[:12]}.json", {**base_record(reason="unreadable_cache")[1], "rescanned": False}),
        ("ids_nonempty_unreadable", f"{DAY}.unreadable_cache.{ID_B[:12]}.json", {**base_record(reason="unreadable_cache")[1], "game_ids": [100]}),
        ("nonnull_dropped_unreadable", f"{DAY}.unreadable_cache.{ID_B[:12]}.json", {**base_record(reason="unreadable_cache")[1], "dropped": 0}),
    ]

    for label, fname, mutated in bad_mutations:
        sdir = root / f"bad_{label}"
        sdir.mkdir()
        p = sdir / fname
        p.write_text(json.dumps(mutated))
        alerts = h_ppc.check(sdir, DAY)
        assert len(alerts) == 1, (label, alerts)
        assert alerts[0].incident_key == f"{SOURCE}:status_unreadable:{fname}", (label, alerts[0])

    # Raw byte / special-file bad status entries (64 KiB + 1, deep nesting, non-UTF-8, FIFO, symlink, dir)
    raw_bad = [
        ("over_64kib", padded_64k + b" "),
        ("deep_recursion", b"[" * 5000 + b"]" * 5000),
        ("bad_utf8", b"\x80\xff"),
        ("top_list", b"[]"),
    ]
    for label, payload in raw_bad:
        sdir = root / f"rawbad_{label}"
        sdir.mkdir()
        p = sdir / base_name
        p.write_bytes(payload)
        alerts = h_ppc.check(sdir, DAY)
        assert len(alerts) == 1 and alerts[0].incident_key == f"{SOURCE}:status_unreadable:{base_name}", label

    print(f"PASS: {len(valid_variants)} valid variants accepted and {len(bad_mutations) + len(raw_bad)} invalid variants warned status_unreadable.")


def test_window_runner_and_dm_dispatch(root: Path) -> None:
    print("=== 3(b). Date window, today=None ET sampling, D/D+1 DM & suppression ===")
    work = root / "dispatch_flow"
    picks_dir = work / "data/picks"
    models_dir = work / "data/models"
    status_dir = work / "data/health_state" / SOURCE
    picks_dir.mkdir(parents=True)
    models_dir.mkdir(parents=True)
    status_dir.mkdir(parents=True)

    # Put records for D-2, D-1, D, D+1
    for off in (-2, -1, 0, 1):
        d = DAY + timedelta(days=off)
        fn, r = base_record(d, "invalid_entries")
        # distinct identity per day for clarity
        ident = f"{off + 5:01x}" * 64
        r["identity"] = ident
        fn = f"{d.isoformat()}.invalid_entries.{ident[:12]}.json"
        (status_dir / fn).write_text(json.dumps(r))

    sent_dms: list[tuple[str, str]] = []
    def fake_send_dm(recipient: str, body: str) -> str:
        sent_dms.append((recipient, body))
        return f"msg-{len(sent_dms)}"

    # Freeze wall clock to 2027-01-01T05:15:00+00:00 (00:15 ET on D+1 = 2027-01-01!)
    wall_after_midnight = datetime(2027, 1, 1, 5, 15, tzinfo=timezone.utc)
    class WallClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return wall_after_midnight.astimezone(tz) if tz is not None else wall_after_midnight.replace(tzinfo=None)

    orig_safe = runner._safe_run
    with (
        patch.object(runner, "datetime", WallClock),
        patch.object(alert_mod, "datetime", WallClock),
        patch.object(alert_mod, "send_dm", fake_send_dm),
        patch.object(runner.realized_calibration, "_current_deploy_iso", return_value=None),
        patch.object(runner, "_safe_run", lambda name, fn: orig_safe(name, fn) if name == SOURCE else []),
    ):
        # 1. Late D run after midnight (wall is D+1 00:15 ET, processed today=D): sees D-1 and D ONLY (not D-2, not D+1!)
        alerts_d1 = runner.run_all_checks(picks_dir, models_dir, "eric.bsky.social", today=DAY)
        keys_d1 = [a.incident_key for a in alerts_d1]
        assert keys_d1 == [
            f"{SOURCE}:2026-12-30:{'4'*64}",
            f"{SOURCE}:2026-12-31:{'5'*64}",
        ], keys_d1
        assert len(sent_dms) == 1
        assert sent_dms[0][0] == "eric.bsky.social"
        assert "(1st observed day)" in sent_dms[0][1]

        # 2. Second run on processed day D -> suppressed by D's receipt!
        alerts_d2 = runner.run_all_checks(picks_dir, models_dir, "eric.bsky.social", today=DAY)
        assert len(alerts_d2) == 2
        assert len(sent_dms) == 1, "same-day repeat digest should be suppressed"

        # 3. Run with today=None when wall clock is 2027-01-01T05:15:00+00:00 (D+1 in ET!):
        #    processed_date is sampled as 2027-01-01 (D+1), NOT suppressed by D's receipt,
        #    sees D (streak=2 -> '2nd consecutive day') and D+1 (streak=1 -> '1st observed day')!
        alerts_next = runner.run_all_checks(picks_dir, models_dir, "eric.bsky.social", today=None)
        keys_next = [a.incident_key for a in alerts_next]
        assert keys_next == [
            f"{SOURCE}:2026-12-31:{'5'*64}",
            f"{SOURCE}:2027-01-01:{'6'*64}",
        ], keys_next
        assert len(sent_dms) == 2
        dm_d_plus_1 = sent_dms[1][1]
        assert "2026-12-31T14:00:00+00:00" in dm_d_plus_1 and "(2nd consecutive day)" in dm_d_plus_1
        assert "2027-01-01T14:00:00+00:00" in dm_d_plus_1 and "(1st observed day)" in dm_d_plus_1

        # 4. Repeat run on D+1 -> suppressed; then add a NEW observation on D+1 -> sends digest repeating covered + new!
        runner.run_all_checks(picks_dir, models_dir, "eric.bsky.social", today=None)
        assert len(sent_dms) == 2
        fn_u, r_u = base_record(DAY + timedelta(days=1), "unreadable_cache")
        (status_dir / fn_u).write_text(json.dumps(r_u))
        runner.run_all_checks(picks_dir, models_dir, "eric.bsky.social", today=None)
        assert len(sent_dms) == 3
        assert sent_dms[2][1].count(f"- [{SOURCE}] ") == 3

        # 5. Run on D+3 (2027-01-03) -> silent, no DM!
        alerts_d3 = runner.run_all_checks(picks_dir, models_dir, "eric.bsky.social", today=DAY + timedelta(days=3))
        assert alerts_d3 == [] and len(sent_dms) == 3
    print("PASS: Processed date window, post-midnight D vs D+1 isolation, today=None ET sampling, streaks, and DM suppression verified.")


def test_item4_texts_and_formatter(root: Path) -> None:
    print("=== 4. Render every operator text through format_health_dm_body & compare character-for-character against A7 ===")
    _, inv_pres = base_record(DAY, "invalid_entries")
    inv_pres.update(dropped=1, recovered_from_cache=0, rescanned=1, unusable_feed=0, without_feed=0, game_ids=[100])

    _, inv_null = base_record(DAY, "invalid_entries")
    inv_null.update(dropped=4, recovered_from_cache=1, rescanned=1, unusable_feed=1, without_feed=1, game_ids=[100, 101], preserved=None)

    _, unr_pres = base_record(DAY, "unreadable_cache")
    unr_pres.update(rescanned=1)

    _, unr_null = base_record(DAY, "unreadable_cache")
    unr_null.update(rescanned=0, preserved=None)

    a_inv_pres = alert_mod.Alert("WARN", SOURCE, h_ppc.message(inv_pres), incident_key="k1")
    a_inv_null = alert_mod.Alert("WARN", SOURCE, h_ppc.message(inv_null), incident_key="k2")
    a_unr_pres = alert_mod.Alert("WARN", SOURCE, h_ppc.message(unr_pres), incident_key="k3")
    a_unr_null = alert_mod.Alert("WARN", SOURCE, h_ppc.message(unr_null), incident_key="k4")
    a_bad_stat = h_ppc.unreadable(Path("data/health_state/probable_pitcher_cache/2026-12-31.bad.json"))

    _, att = attention_mod.apply_warn_attention_policy(
        [a_inv_pres, a_inv_null, a_unr_pres, a_unr_null, a_bad_stat],
        state_path=root / "att.json",
        today=DAY,
    )
    rendered_digest = alert_mod.format_health_dm_body([], att)
    assert rendered_digest is not None
    print("--- Rendered DM Digest ---")
    print(rendered_digest)
    print("--- End Digest ---")

    # Check exact character-for-character comparison against P6 + A7
    # Note the backtick detail on preserved path!
    expected_inv_null = (
        "Probable-pitcher cache at 2026-12-31T14:00:00+00:00 had 4 invalid entries (first game IDs 100, 101). "
        "Of those entries, 1 had a valid cache alias and 1 had a usable raw feed. "
        "No usable lookup was found for 1 with unusable or unverified feeds and 1 with missing feeds. "
        "This observation names no evidence copy. That run left the active cache unchanged. "
        "This describes the cache at the recorded time; health may show this observation again on the next processed day. "
        "Keep the active cache, raw feeds, any named evidence copy and other available copies intact for review. "
        "Send this alert promptly to the supervised BTS lead Claude session (bts-lead), or to Eric if the session is unavailable. "
        "Ask them to review the latest preview and any pending pick before the next scheduled selection. "
        "If a pick was already sent or entered, report that and preserve its records."
    )
    assert a_inv_null.message == expected_inv_null

    expected_unr_null = (
        "Probable-pitcher cache was unreadable at 2026-12-31T14:00:00+00:00. "
        "That run started from an empty lookup and built usable entries from raw feeds for 0 games. "
        "Keep the active cache, raw feeds, any named evidence copy and other available copies intact for review. "
        "Other games may have been missing from that run's lookup. "
        "This observation names no evidence copy. That run left the active cache unchanged. "
        "Send this alert immediately to the supervised BTS lead Claude session (bts-lead), or to Eric if the session is unavailable. "
        "Ask them to preserve and verify any usable cache copy in the current or previous R2 manifest before the next successful sync. "
        "An unreferenced content-addressed cache object can be pruned once it is at least seven days old."
    )
    assert a_unr_null.message == expected_unr_null

    expected_bad_stat = (
        "Probable-pitcher cache status unreadable at data/health_state/probable_pitcher_cache/2026-12-31.bad.json. "
        "Send this alert to the supervised BTS lead Claude session (bts-lead), or to Eric if the session is unavailable."
    )
    assert a_bad_stat.message == expected_bad_stat

    # Check preserved branch exact characters
    assert f"That run recorded an evidence copy at `data/models/rejected/{PRESERVED_NAME}`." in a_inv_pres.message
    assert f"That run recorded an evidence copy at `data/models/rejected/{PRESERVED_NAME}`." in a_unr_pres.message
    assert "for 1 game. " in a_unr_pres.message
    print("PASS: All operator alert variants rendered through format_health_dm_body and checked character-for-character.")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="probe34_") as tmp:
        root = Path(tmp)
        test_a6_validation(root)
        test_window_runner_and_dm_dispatch(root)
        test_item4_texts_and_formatter(root)
    print("ALL PROBE 3 & 4 CHECKS PASSED")


if __name__ == "__main__":
    main()
