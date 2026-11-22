"""Prepare a disposable source copy to verify the suggested core repairs."""

from pathlib import Path
import shutil


HERE = Path(__file__).resolve().parent
PATCHED = HERE / "patched" / "src" / "bts"
TARGET = HERE / "scratch" / "repair" / "src" / "bts"


def replace_once(text, old, new):
    assert text.count(old) == 1
    return text.replace(old, new)


source = (PATCHED / "scheduler.py").read_text()
source = replace_once(source,
    "        state.committed_pick_written = True\n"
    "        save_state(state, picks_dir)\n\n"
    "    for run_idx, run_info in enumerate(runs):",
    "        state.committed_pick_written = True\n"
    "        save_state(state, picks_dir)\n"
    "        if record.get('delivery_status') == 'locked_unconfirmed':\n"
    "            _alert_delivery_outcome_unknown(config, picks_dir, date)\n\n"
    "    for run_idx, run_info in enumerate(runs):")
source = replace_once(source,
    "if current_record is not None and current is not None and current.date == date:",
    "if current is not None and current.date == date:")
source = replace_once(source,
    "                                daily, config, picks_dir, state, date, \"fallback\", defer_void_games)\n"
    "                            continue\n"
    "                    if plan.action == \"defer\":",
    "                                daily, config, picks_dir, state, date, \"fallback\", defer_void_games)\n"
    "                            continue\n"
    "                        if plan.action == \"defer\":\n"
    "                            plan = plan_fallback_action(now=now, **planner_inputs)\n"
    "                    if plan.action == \"defer\":")
source = replace_once(source,
    "        daily.automatic_reading = prior_reading\n"
    "        save_pick(daily, picks_dir)\n"
    "        if now >= cutoff:\n",
    "        daily.automatic_reading = prior_reading\n"
    "        try:\n"
    "            save_pick(daily, picks_dir)\n"
    "        except OSError:\n"
    "            save_pick(daily, picks_dir)\n"
    "        if now >= cutoff:\n")
shutil.copytree(PATCHED, TARGET, dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
compile(source, str(TARGET / "scheduler.py"), "exec")
(TARGET / "scheduler.py").write_text(source)
