from contextlib import contextmanager
from pathlib import Path
import shutil

from review_runner import BASE, CHANGED_TESTS, REVIEW, ROOT, archive, run

CONTEST = "tests/test_contest_dated_cs4a.py"
FRESHNESS = "tests/health/test_park_drag_freshness.py"
PRODUCER = "tests/test_park_drag_producer.py"
CRON = "tests/test_entry_intent.py"
ARTIFACT = "tests/experiment/test_artifacts.py"
CAPTURE = "tests/scripts/test_live_forward_capture_once.py"
SHADOW = "tests/test_shadow_eval.py"
POLICY = "tests/test_skip_policy_shadow.py"
F4 = (CONTEST, "-k", "opening_eod or automatic_coverage or future_settlement")
F5 = (FRESHNESS,)
F5_PRODUCER = (PRODUCER, "-k", "empty_reseed or refresh_end_to_end")
F6 = (CRON + "::test_enter_install_adds_entry_cron_and_fetch_recipient",)
F7_ARTIFACT = (ARTIFACT,)
F7 = (CAPTURE,)
F8 = (SHADOW, POLICY)


@contextmanager
def patch_file(path, old, new):
    before = path.read_text()
    assert before.count(old) == 1, (path, old, before.count(old))
    path.write_text(before.replace(old, new))
    try:
        yield
    finally:
        path.write_text(before)


def stage_tests(path):
    for relative in ("pyproject.toml", "tests/conftest.py", "tests/__init__.py",
                     "tests/experiment/conftest.py", *CHANGED_TESTS):
        source = ROOT / relative
        if source.exists():
            dest = path / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)


def wrapped(path, args):
    return run(path, "pytest", "-o", "addopts=", *args, cwd=path)


def mutations():
    stage = REVIEW / "mutant"
    archive("HEAD", stage, "src", "scripts")
    stage_tests(stage)
    stage_tests(BASE)
    red_cases = [
        ("F4", F4),
        ("F5", (*F5, PRODUCER + "::TestRefresh::test_empty_reseed_through_first_game")),
        ("F6", F6 + ("tests/test_contest_fetch.py", "-k", "fetch_recipient or failure_without_recipient")),
        ("F7", (ARTIFACT + "::test_verify_candidate_artifact_pair_accepts_live_preoutcome_artifact",
                CAPTURE + "::test_failed_verification_retries_only_before_outcomes")),
        ("F8", (*F8, "-k", "defaults_to_et_season_and_allows_historical_read")),
    ]
    for name, args in red_cases:
        print(f"BASE {name}", flush=True)
        assert wrapped(BASE, args) == 1, name

    f4 = "src/bts/health/contest_state.py"
    f5 = "src/bts/health/park_drag_freshness.py"
    producer = "src/bts/features/park_drag_producer.py"
    f6 = "scripts/cron-setup-hetzner.sh"
    manifest = "src/bts/experiment/artifacts.py"
    capture = "scripts/live_forward_capture_once.py"
    shadow = "src/bts/shadow_eval.py"
    policy = "src/bts/skip_policy_shadow.py"
    cli = "src/bts/cli.py"
    specs = [
        ("F4-auto-file", f4, 'state.path.name == "contest_streak.json"', "True", F4),
        ("F4-timestamp-parsed", f4, "and recorded is not None", "and True", F4),
        ("F4-timestamp-year", f4, "and recorded.astimezone(et).year == local_now.year", "and True", F4),
        ("F4-no-future-timestamp", f4, "timedelta(0) <= now - recorded <= timedelta(hours=36)",
         "now - recorded <= timedelta(hours=36)", F4),
        ("F4-36h", f4, "timedelta(0) <= now - recorded <= timedelta(hours=36)",
         "timedelta(0) <= now - recorded", F4),
        ("F4-has-settlement", f4, "and latest is not None", "and True", F4),
        ("F4-settlement-year", f4, "and previous_year_end < latest <= local_now.date()",
         "and latest <= local_now.date()", F4),
        ("F4-no-future-settlement", f4, "and previous_year_end < latest <= local_now.date()",
         "and previous_year_end < latest", F4),
        ("F4-min-settlements", f4, "and 1 <= resolved_pick_settlement_gap",
         "and 0 <= resolved_pick_settlement_gap", F4),
        ("F4-max-settlements", f4, "<= EXPECTED_OVERNIGHT_LAG_STEPS + 1",
         "<= 9999", F4),
        ("F5-prior-year", f5, "if source_date.year < today.year:", "if True:", F5),
        ("F5-unconfigured-year", f5, "if opener is not None:",
         'if True:\n                    opener = opener or f"{today.year}-03-24"', F5),
        ("F5-opener-lag", f5, "gap_days = max(0, (today - date.fromisoformat(opener)).days)", "pass", F5),
        ("F5-clamp", f5, "max(0, (today - date.fromisoformat(opener)).days)",
         "(today - date.fromisoformat(opener)).days", F5),
        ("F5-empty-year", producer, 'today.year if store.empty else int(store["game_date"].dt.year.max())',
         'int(store["game_date"].dt.year.max())', F5_PRODUCER),
        ("F5-empty-window", producer, "if not store.empty:\n            window_start", "if True:\n            window_start",
         F5_PRODUCER),
        ("F5-empty-ingest", producer, "fetched if store.empty else pd.concat", "pd.concat", F5_PRODUCER),
        ("F5-empty-physics", producer, "pd.DataFrame() if store.empty else compute_game_level",
         "compute_game_level", F5_PRODUCER),
        ("F5-empty-output", producer, "if store.empty and game_level.empty:", "if False:", F5_PRODUCER),
        ("F6-research-recipient", f6, 'FETCH_RECIPIENT=""', 'FETCH_RECIPIENT=" --dm-recipient stonehengee.bsky.social"', F6),
        ("F7-write-effective-count", manifest, '        "effective_top_n": effective_top_n,\n', "", F7_ARTIFACT),
        ("F7-use-effective-count", manifest, "            row_top_n = effective", "            pass", F7_ARTIFACT),
        ("F7-live-kind", manifest, 'if run_kind in {"live_forward_preoutcome", "live_forward_resolved"} and "effective_top_n" in manifest:',
         'if "effective_top_n" in manifest:', F7_ARTIFACT),
        ("F7-effective-type", manifest, "type(effective) is int and type(requested) is int",
         "isinstance(effective, (int, float)) and type(requested) is int", F7_ARTIFACT),
        ("F7-effective-positive", manifest, "0 < effective <= requested", "effective <= requested", F7_ARTIFACT),
        ("F7-effective-cap", manifest, "0 < effective <= requested", "0 < effective", F7_ARTIFACT),
        ("F7-retry", capture, "if stale_snapshot or retry_failed:", "if stale_snapshot:", F7),
        ("F7-verified-protection", capture, 'verification.get("ok") is False and not previously_verified',
         'verification.get("ok") is False', F7),
        ("F7-same-snapshot", capture, 'snapshot_state.get("stale_pick_snapshot") is False\n                        and verification',
         'True\n                        and verification', F7),
        ("F7-recorded-failure", capture, 'verification.get("ok") is False and not previously_verified',
         'not previously_verified', F7),
        ("F7-publication-check", capture, 'if snapshot_state.get("retry_failed_verification"):\n',
         "if False:\n", F7),
        ("F7-publication-result", capture, "not pick_is_unresolved(latest_pick) or n_pa_rows",
         "n_pa_rows", F7),
        ("F7-publication-PA", capture, "not pick_is_unresolved(latest_pick) or n_pa_rows",
         "not pick_is_unresolved(latest_pick)", F7),
        ("F7-publication-pick-hash", capture, 'or file_sha256(pick_path) != snapshot_state["current_pick_sha256"]',
         "or False", F7),
        ("F8-shadow-glob", shadow, 'glob(f"{season}-*.shadow.json")', 'glob("*.shadow.json")', F8),
        ("F8-shadow-wrong-ET-season", shadow, 'datetime.now(ZoneInfo("America/New_York")).year',
         "2026", F8),
        ("F8-policy-glob", policy, 'glob(f"{season}-*.policy_shadow.json")', 'glob("*.policy_shadow.json")', F8),
        ("F8-contradiction-scope", policy, 'if day.startswith(f"{season}-")', "if True", F8),
        ("F8-check-results-date", cli, "build_shadow_cycle_status(picks_path, season=season_of_date(date))",
         "build_shadow_cycle_status(picks_path)", F8),
        ("F8-policy-CLI-date", cli, "season=season if season is not None else season_of_date(date),", "season=season,", F8),
    ]
    survived = []
    killed = []
    for name, path, old, new, args in specs:
        print(f"MUTANT {name}", flush=True)
        with patch_file(stage / path, old, new):
            code = wrapped(stage, args)
        assert code in (0, 1), (name, code)
        (survived if code == 0 else killed).append(name)
        print(f"RESULT {name}: {'survived' if code == 0 else 'killed'}", flush=True)
    print(f"Killed {len(killed)}, survived {len(survived)}: {', '.join(survived)}", flush=True)
    return 0
