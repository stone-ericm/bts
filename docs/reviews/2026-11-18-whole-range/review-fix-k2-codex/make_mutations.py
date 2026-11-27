import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BOUNDARY_CALL = """    try:
        _season_boundary_check(date, games, runs, config, picks_dir)
    except Exception as error:
        _report_season_boundary_failure(error)

"""
MORNING_CALL = """    try:
        if (delivery_mode in {"dm", "public"} and _contest_state_required(config)
"""


def replace(text, old, new):
    assert text.count(old) == 1, (old, text.count(old))
    return text.replace(old, new)


def move_warning(text):
    text = replace(text, BOUNDARY_CALL, "")
    return replace(text, MORNING_CALL, BOUNDARY_CALL + MORNING_CALL)


def remove_deadline(text):
    start = text.index("    worker = Thread(target=dispatch, daemon=True)")
    end = text.index("\n\n\ndef run_day", start)
    return text[:start] + "    dispatch()" + text[end:]


def one_edit(old, new):
    return lambda text: replace(text, old, new)


MUTATIONS = {
    "unconditional-floor": ("strategy", one_edit("    days = max(_days_remaining(date), min_days)",
                                                 "    days = max(_days_remaining(date), min_days, 1)")),
    "no-early-floor": ("strategy", one_edit("_regime(None, streak, date, min_days=1)[0]",
                                            "_regime(None, streak, date)[0]")),
    "no-context-floor": ("strategy", one_edit("        min_days=1,\n    )\n    decision =",
                                              "    )\n    decision =")),
    "warning-first": ("scheduler", move_warning),
    "no-official-date": ("scheduler", one_edit(' or game.get("officialDate") != slate_date', "")),
    "synchronous-send": ("scheduler", remove_deadline),
    "plain-dispatch": ("scheduler", one_edit("now_et_date=slate_day, require_message_id=True",
                                             "now_et_date=slate_day")),
}

if __name__ == "__main__":
    root = Path(sys.argv[1])
    for name, (module, transform) in MUTATIONS.items():
        target = root / name / "src"
        shutil.copytree(ROOT / "src", target, ignore=shutil.ignore_patterns("__pycache__"))
        path = target / "bts" / f"{module}.py"
        path.write_text(transform(path.read_text()))
        print(name, path)
