"""Create one change at a time in an archived source copy."""
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
MUTATIONS = {
    "integer": [(' and type(slot["game_pk"]) is int', "")],
    "shared": [('if len({slot["batter_id"] for slot in slots}) != len(slots):', "if False:")],
    "case-b": [('if not affected and prior_wrong_game and not qualification["all_confirmed"]:', "if False:"),
               ("            if prior_wrong_game:\n                affected", "            if False:\n                affected")],
    "initial": [('            consumed.add("wrong_game")', '            consumed.update(("wrong_game", "initial"))')],
    "any-leg": [('not qualification["all_confirmed"]:',
                 'not any(result["state"] == "confirmed" for result in qualification["slots"]):')],
    "texts": [('        elif tier == "wrong_game":\n            msg =',
               '        elif tier != "wrong_game":\n            msg =')],
}


def create(name):
    target = HERE / f"scratch-{name}" / "src"
    shutil.copytree(HERE / "candidate" / "src", target)
    path = target / "bts" / "cli.py"
    text = path.read_text()
    for original, changed in MUTATIONS[name]:
        if text.count(original) != 1:
            raise ValueError(f"Expected one occurrence of {original!r}")
        text = text.replace(original, changed, 1)
    path.write_text(text)
    print(path)


if __name__ == "__main__":
    create(sys.argv[1])
