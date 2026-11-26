"""Rerun the applicable first-review survivors on a scratch copy of HEAD."""
from dataclasses import dataclass
import json
from pathlib import Path
import shutil

from review_runner import REVIEW, ROOT, archive, run

CONTEST_TESTS = ("tests/test_contest_dated_cs4a.py",)
PARK_TESTS = ("tests/health/test_park_drag_freshness.py",)
PRODUCER_TESTS = ("tests/test_park_drag_producer.py",)
CAPTURE_TESTS = ("tests/experiment/test_artifacts.py",)
SHADOW_TESTS = ("tests/test_shadow_eval.py", "tests/test_skip_policy_shadow.py")
CONTEST = "src/bts/health/contest_state.py"
PARK = "src/bts/health/park_drag_freshness.py"
PRODUCER = "src/bts/features/park_drag_producer.py"
CAPTURE = "src/bts/experiment/artifacts.py"
SHADOW = "src/bts/shadow_eval.py"
POLICY = "src/bts/skip_policy_shadow.py"
CLI = "src/bts/cli.py"


@dataclass(frozen=True)
class Mutation:
    name: str
    path: str
    old: str
    new: str
    tests: tuple[str, ...]


SPECS = (
    Mutation("F4-auto-file", CONTEST, 'state.path.name == "contest_streak.json"', "True", CONTEST_TESTS),
    Mutation("F4-timestamp-parsed", CONTEST, "and recorded is not None", "and True", CONTEST_TESTS),
    # The original explicit latest-is-not-None guard is now implicit in the season count.
    Mutation("F4-has-settlement", CONTEST, "and latest <= today",
             "and (latest is None or latest <= today)", CONTEST_TESTS),
    Mutation("F4-settlement-year", CONTEST, " if day.year == today.year", "", CONTEST_TESTS),
    Mutation("F4-min-settlements", CONTEST, "and 1 <= len(season_settlements)",
             "and 0 <= len(season_settlements)", CONTEST_TESTS),
    Mutation("F5-prior-year", PARK, "if source_date.year == today.year - 1:", "if True:", PARK_TESTS),
    Mutation("F5-clamp", PARK, "max(0, (today - date.fromisoformat(opener)).days)",
             "(today - date.fromisoformat(opener)).days", PARK_TESTS),
    Mutation("F5-empty-physics", PRODUCER, "pd.DataFrame() if store.empty else compute_game_level",
             "compute_game_level", PRODUCER_TESTS),
    Mutation("F5-empty-output", PRODUCER, "if store.empty and game_level.empty:", "if False:", PRODUCER_TESTS),
    Mutation("F7-live-kind", CAPTURE,
             'if run_kind in {"live_forward_preoutcome", "live_forward_resolved"} and "effective_top_n" in manifest:',
             'if "effective_top_n" in manifest:', CAPTURE_TESTS),
    Mutation("F7-effective-type", CAPTURE, "type(effective) is int and type(requested) is int",
             "isinstance(effective, (int, float)) and type(requested) is int", CAPTURE_TESTS),
    Mutation("F7-effective-positive", CAPTURE, "0 < effective <= requested",
             "effective <= requested", CAPTURE_TESTS),
    Mutation("F7-effective-cap", CAPTURE, "0 < effective <= requested", "0 < effective", CAPTURE_TESTS),
    Mutation("F8-shadow-wrong-ET-season", SHADOW, 'datetime.now(ZoneInfo("America/New_York")).year',
             "2026", SHADOW_TESTS),
    Mutation("F8-contradiction-scope", POLICY,
             "contradicted = set(find_aged_contradictions(picks_dir, now=now))",
             'contradicted = {day for day in find_aged_contradictions(picks_dir, now=now) if day.startswith(f"{season}-")}',
             SHADOW_TESTS),
    Mutation("F8-check-results-date", CLI, "build_shadow_cycle_status(picks_path, season=season_of_date(date))",
             "build_shadow_cycle_status(picks_path)", SHADOW_TESTS),
    Mutation("F8-policy-CLI-date", CLI, "season=season if season is not None else season_of_date(date),",
             "season=season,", SHADOW_TESTS),
)

EXTRAS = (
    Mutation("F4-remove-time-bound", CONTEST, "and min(season_settlements) >= today - timedelta(days=1)",
             "and True", CONTEST_TESTS),
    Mutation("F4-remove-future-guard", CONTEST, "and latest <= today", "and True", CONTEST_TESTS),
    Mutation("F4-remove-max-settlements", CONTEST, "len(season_settlements) <= EXPECTED_OVERNIGHT_LAG_STEPS + 1",
             "len(season_settlements) <= 9999", CONTEST_TESTS),
    Mutation("F7-bool-effective", CAPTURE, "type(effective) is int and type(requested) is int",
             "isinstance(effective, int) and type(requested) is int", CAPTURE_TESTS),
    Mutation("F7-negative-effective", CAPTURE, "0 < effective <= requested",
             "effective != 0 and effective <= requested", CAPTURE_TESTS),
)


def main():
    stage = REVIEW / "mutant"
    archive("HEAD", stage, "src", "scripts")
    tests = tuple(dict.fromkeys(test for entry in SPECS for test in entry.tests))
    for relative in ("pyproject.toml", "tests/__init__.py", "tests/conftest.py",
                     "tests/experiment/conftest.py", *tests):
        source = ROOT / relative
        if source.exists():
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)

    print("Unmutated control", flush=True)
    assert run(stage, "pytest", *tests) == 0
    results = {}
    for entry in (*SPECS, *EXTRAS):
        print(f"MUTANT {entry.name}", flush=True)
        path = stage / entry.path
        before = path.read_text()
        assert before.count(entry.old) == 1, (path, entry.old, before.count(entry.old))
        after = before.replace(entry.old, entry.new)
        compile(after, str(path), "exec")
        path.write_text(after)
        try:
            code = run(stage, "pytest", *entry.tests)
        finally:
            path.write_text(before)
        assert code in (0, 1), (entry.name, code)
        results[entry.name] = "caught" if code == 1 else "survived"
        print(f"RESULT {entry.name}: {results[entry.name]}", flush=True)
    (REVIEW / "mutations.json").write_text(json.dumps(results, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
