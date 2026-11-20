"""Prepare independent, single-fault source trees for the C1 tests."""

from pathlib import Path
import shutil


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNNER = "bts/health/runner.py"
CONTEST = "bts/health/contest_state.py"
SEASON = "bts/season_contract.py"
MUTANTS = {
    "wall-endpoint": (RUNNER, (
        ("if today is not None and today < datetime.now(timezone.utc).astimezone(ET).date():",
         "wall_now = datetime.now(timezone.utc) if today is not None else None\n"
         "    if today is not None and today < wall_now.astimezone(ET).date():"),
        ("datetime.combine(today, time.max, ET).astimezone(timezone.utc)", "wall_now"),
    )),
    "inclusive-mode": (RUNNER, (("today < datetime.now", "today <= datetime.now"),)),
    "no-mode": (RUNNER, (("            \"historical_day\": today,\n", ""),)),
    "old-absent-switch-copy": (SEASON, (
        ('elif state.state in {"absent", "other_season", "revoked"}:\n        if historical:',
         'elif state.state in {"absent", "other_season", "revoked"}:\n        if historical and state.state != "absent":'),
    )),
    "wall-saver-season": (CONTEST, (
        ('season = now.astimezone(ZoneInfo("America/New_York")).year',
         'season = (datetime.now(timezone.utc) if historical else now).astimezone(ZoneInfo("America/New_York")).year'),
    )),
    "wall-switch-season": (SEASON, (
        ("inspection = now.astimezone(ET).year if season is None else season",
         "inspection = (datetime.now(timezone.utc) if historical else now).astimezone(ET).year if season is None else season"),
    )),
    "no-warn-pointer": (RUNNER, (("+ separator + label + pointer,", "+ separator + label,"),)),
    "blocking-manual-reread": (CONTEST, (
        ("raw = read_regular_file_text(manual_path) if historical else manual_path.read_text()",
         "raw = manual_path.read_text()"),
    )),
    "diagnostic-as-crash": (CONTEST, (
        ("if historical and not isinstance(exc, _HistoricalContestStateError):", "if historical:"),
    )),
}


def main():
    for mutant, (relative_path, replacements) in MUTANTS.items():
        dst = HERE / "mutants" / mutant / "src"
        shutil.copytree(ROOT / "src", dst, ignore=shutil.ignore_patterns("__pycache__"))
        file = dst / relative_path
        content = file.read_text()
        for original, changed in replacements:
            assert content.count(original) == 1, (mutant, original)
            content = content.replace(original, changed, 1)
        compile(content, str(file), "exec")
        file.write_text(content)
        print(f"{mutant}: {relative_path}")


if __name__ == "__main__":
    main()
