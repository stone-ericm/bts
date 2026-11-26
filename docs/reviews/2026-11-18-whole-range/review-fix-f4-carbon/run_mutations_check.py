"""Standalone mutation driver: applies each required scratch change on a scratch copy outside .carbon-review/wr-f4-r1/."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRATCH_PARENT = Path("/Users/eric/.gemini/antigravity-cli/brain/14e767f2-bdf2-400a-af04-e12ff1bff447/scratch")

RUNNER_SNIPPET = r"""
import os
import sys
from pathlib import Path
import pytest

scratch_src = Path(os.environ["SCRATCH_SRC"]).resolve()
repo_root = Path(os.environ["REPO_ROOT"]).resolve()
test_files = os.environ["TEST_FILES"].split(":")

sys.meta_path = [
    f for f in sys.meta_path
    if "editable" not in type(f).__name__.lower() and "bts" not in repr(f).lower()
]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(scratch_src))

import bts
actual = Path(bts.__file__).resolve()
expected = (scratch_src / "bts" / "__init__.py").resolve()
assert actual == expected, f"Expected {expected}, got {actual}"
print(f"bts.__file__={actual}")
sys.exit(pytest.main(["-q", "--tb=no", *test_files]))
"""


def replace_exact(path: Path, old: str, new: str) -> None:
    content = path.read_text(encoding="utf-8")
    assert old in content, f"Target snippet not found in {path}:\n{old}"
    assert content.count(old) == 1, f"Target snippet appeared multiple times in {path}"
    path.write_text(content.replace(old, new, 1), encoding="utf-8")


MUTATIONS = [
    (
        "1a_read_before_classification_order_only",
        "src/bts/day_outcome.py",
        (
            "    void_games = 0\n"
            "    try:\n"
            "        schedule = fetch_schedule(date)\n"
            "        games = len(schedule)\n"
            "        void_games = sum(_is_void_detailed_state((g.get(\"status\") or {}).get(\"detailedState\")) for g in schedule)\n"
            "    except Exception:\n"
            "        games = None\n"
            "    decision, raw = load_decision_bytes(date, picks_dir)\n"
            "    intent = check_entry_intent(config, include_live_gate=False)\n"
            "    outcome = classify_day(date=date, games=games, void_games=void_games,\n"
            "                           decision=decision, decision_sha256=entry_receipt._sha(raw),\n"
            "                           receipts=entry_receipt.discover(picks_dir, date),\n"
            "                           intent=intent.intent if intent.ok else None, expected_username=expected_username)\n"
            "    if outcome.state == \"deliberate_skip\":\n"
            "        outcome = _skip_review(date, picks_dir, decision, intent.intent) or outcome\n"
            "    return replace(outcome, streak_problem=_streak_problem(Path(picks_dir) / \"streak.json\"))"
        ),
        (
            "    problem = _streak_problem(Path(picks_dir) / \"streak.json\")\n"
            "    void_games = 0\n"
            "    try:\n"
            "        schedule = fetch_schedule(date)\n"
            "        games = len(schedule)\n"
            "        void_games = sum(_is_void_detailed_state((g.get(\"status\") or {}).get(\"detailedState\")) for g in schedule)\n"
            "    except Exception:\n"
            "        games = None\n"
            "    decision, raw = load_decision_bytes(date, picks_dir)\n"
            "    intent = check_entry_intent(config, include_live_gate=False)\n"
            "    outcome = classify_day(date=date, games=games, void_games=void_games,\n"
            "                           decision=decision, decision_sha256=entry_receipt._sha(raw),\n"
            "                           receipts=entry_receipt.discover(picks_dir, date),\n"
            "                           intent=intent.intent if intent.ok else None, expected_username=expected_username)\n"
            "    if outcome.state == \"deliberate_skip\":\n"
            "        outcome = _skip_review(date, picks_dir, decision, intent.intent) or outcome\n"
            "    return replace(outcome, streak_problem=problem)"
        ),
    ),
    (
        "1b_read_and_return_before_classification",
        "src/bts/day_outcome.py",
        "    void_games = 0\n",
        (
            "    early = _streak_problem(Path(picks_dir) / \"streak.json\")\n"
            "    if early is not None:\n"
            "        return DayOutcome(date=date, state=\"entered\", detail=\"\", streak_problem=early)\n"
            "    void_games = 0\n"
        ),
    ),
    (
        "2_missing_file_is_problem",
        "src/bts/day_outcome.py",
        "    except FileNotFoundError:\n        return None",
        "    except FileNotFoundError:\n        return \"cannot be read (FileNotFoundError)\"",
    ),
    (
        "3_accept_bool_streak",
        "src/bts/day_outcome.py",
        "    if type(streak) is not int or streak < 0:",
        "    if not isinstance(streak, int) or streak < 0:",
    ),
    (
        "4a_drop_1mib_bound_arg",
        "src/bts/day_outcome.py",
        "        text = read_regular_file_text(path, max_bytes=STREAK_MAX_BYTES)",
        "        text = read_regular_file_text(path)",
    ),
    (
        "4b_drop_1mib_limit_check_only",
        "src/bts/util.py",
        (
            "            while len(data) <= max_bytes:\n"
            "                chunk = os.read(fd, max_bytes + 1 - len(data))\n"
            "                if not chunk:\n"
            "                    break\n"
            "                data.extend(chunk)\n"
            "            if len(data) > max_bytes:\n"
            "                raise FileReadLimitError(f\"{path}: exceeds read limit\")\n"
            "            return data.decode(\"utf-8\")"
        ),
        (
            "            with io.open(fd, \"r\", encoding=\"utf-8\", closefd=False) as handle:\n"
            "                return handle.read()"
        ),
    ),
    (
        "5_drop_compact_dm",
        "src/bts/day_outcome.py",
        (
            "        if len(body) > DM_MAX_CODE_POINTS:\n"
            "            return (\n"
            "                f\"{head}. Local model streak file needs review ({self.streak_problem}). Preserve the files. \"\n"
            "                \"See day-outcome JSON in cron.log and B8b item 2 in docs/ops/2027-season-start.md before the \"\n"
            "                \"next affected job. Escalate through Eric if needed.\"\n"
            "            )\n"
        ),
        "",
    ),
    (
        "6_drop_recursion_error",
        "src/bts/day_outcome.py",
        "    except (ValueError, RecursionError) as exc:",
        "    except ValueError as exc:",
    ),
    (
        "7_clean_day_head_uses_state_name",
        "src/bts/day_outcome.py",
        (
            "        if self.streak_problem is not None and self.state in OK_STATES:\n"
            "            title = \"MODEL STREAK FILE NEEDS REVIEW\"\n"
        ),
        "",
    ),
    (
        "8_compact_boundary_off_by_one_ge",
        "src/bts/day_outcome.py",
        "        if len(body) > DM_MAX_CODE_POINTS:",
        "        if len(body) >= DM_MAX_CODE_POINTS:",
    ),
]


def run_pytest_against(scratch_src: Path, test_files: list[str]) -> tuple[int, str]:
    env = os.environ.copy()
    env["SCRATCH_SRC"] = str(scratch_src)
    env["REPO_ROOT"] = str(REPO_ROOT)
    env["TEST_FILES"] = ":".join(test_files)
    env["TZ"] = "America/New_York"
    proc = subprocess.run(
        [sys.executable, "-c", RUNNER_SNIPPET],
        env=env,
        capture_output=True,
        text=True,
        timeout=45,
    )
    summary = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    tail = summary[-1] if summary else ""
    return proc.returncode, tail


def main() -> None:
    SCRATCH_PARENT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(SCRATCH_PARENT), prefix="mut_") as base_tmp:
        root = Path(base_tmp)
        for name, rel_file, old, new in MUTATIONS:
            mut_dir = root / name / "src"
            shutil.copytree(REPO_ROOT / "src", mut_dir)
            replace_exact(mut_dir / rel_file.removeprefix("src/"), old, new)
            rc_author, tail_author = run_pytest_against(mut_dir, ["tests/test_wr_f4_streak_file.py"])
            rc_probes, tail_probes = run_pytest_against(
                mut_dir,
                [
                    ".carbon-review/wr-f4-r1/probe_matrix_and_standing.py",
                    ".carbon-review/wr-f4-r1/probe_dm_and_text.py",
                ],
            )
            status_author = "KILLED" if rc_author != 0 else "SURVIVED"
            status_probes = "KILLED" if rc_probes != 0 else "SURVIVED"
            print(
                f"{name}: author_tests={status_author} ({tail_author}) | "
                f"review_probes={status_probes} ({tail_probes})",
                flush=True,
            )


if __name__ == "__main__":
    main()
