"""Build a scratch-only fix for an unconsumed wrong-game observation."""
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent


def main():
    src = HERE / "scratch-observation-fix" / "src"
    shutil.copytree(HERE / "candidate" / "src", src)
    cli = src / "bts" / "cli.py"
    content = cli.read_text()
    mark = '        marker["receipt"] = rec.attempt_id'
    assert content.count(mark) == 1
    content = content.replace(mark, '''        if (marker["status"] != "confirmed" and prior_wrong_game and marker["reason"] != "wrong_game"
                and "wrong_game" not in marker.get("escalations", ())):
            marker["wrong_game_observed"] = True
''' + mark)
    history = '    prior_wrong_game = prior.get("reason") == "wrong_game" or "wrong_game" in prior_escalations'
    assert content.count(history) == 1
    content = content.replace(history, '''    prior_wrong_game = (prior.get("reason") == "wrong_game" or "wrong_game" in prior_escalations
                        or prior.get("wrong_game_observed") is True)''')
    fallback = 'was_alerted and prior.get("reason") != "wrong_game" else []'
    assert content.count(fallback) == 1
    content = content.replace(fallback, 'was_alerted and prior.get("reason") != "wrong_game"\n'
                              '                                  and not prior.get("wrong_game_observed") else []')
    cli.write_text(content)
    print(cli)


if __name__ == "__main__":
    main()
