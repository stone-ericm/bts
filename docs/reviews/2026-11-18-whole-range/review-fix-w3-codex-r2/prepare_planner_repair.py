"""Prepare a scratch repair to verify timing and preservation of the void observation."""

from pathlib import Path
import shutil


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[2] / "src" / "bts"
TARGET = HERE / "planner-repair" / "src" / "bts"
code = (SOURCE / "scheduler.py").read_text()


def edit(old, new):
    global code
    assert code.count(old) == 1
    code = code.replace(old, new)


edit('                        if polled_for_defer and now < cutoff:\n',
     '                        if (polled_for_defer and now < cutoff and not (defer_void_games\n'
     '                                and len(defer_void_games) == len(_committed_pick_game_pks(daily)))):\n')
edit('                            continue\n'
     '                    if plan.action == "defer":\n'
     '                        archive = _defer_pick_at_fallback',
     '                            continue\n'
     '                        if polled_for_defer:\n'
     '                            plan = plan_fallback_action(now=now, **planner_inputs)\n'
     '                            state.fallback_refreshes[-1].update(action=plan.action, reason=plan.reason)\n'
     '                    if plan.action == "defer":\n'
     '                        archive = _defer_pick_at_fallback')
compile(code, str(TARGET / "scheduler.py"), "exec")
shutil.copytree(SOURCE, TARGET, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
(TARGET / "scheduler.py").write_text(code)
