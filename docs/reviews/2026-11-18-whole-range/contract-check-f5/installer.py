"""Generate both crontabs and stage the one-field scratch change."""
from pathlib import Path
import json
import shutil

from tests.test_entry_intent import box, ENTRY_CMD
import tests.test_entry_intent as test_entry_intent

here = Path(__file__).resolve().parent
repo = here.parents[1]
scratch = here / "scratch"
scratch.mkdir(exist_ok=True)
shutil.copytree(repo / "tests", scratch / "tests", dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("__pycache__"))
shutil.copyfile(repo / "pyproject.toml", scratch / "pyproject.toml")
(scratch / "scripts").mkdir(exist_ok=True)
old = (repo / "scripts/cron-setup-hetzner.sh").read_text()
assert old.count("*/15 10-23 * * * $PREFIX") == 1
new_installer = scratch / "scripts/cron-setup-hetzner.sh"
new_installer.write_text(old.replace("*/15 10-23 * * * $PREFIX", "*/15 * * * * $PREFIX"))

def generate(script, label, intent="enter"):
    root = here / "generated" / label
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    fake = box.__wrapped__(root)
    test_entry_intent.SCRIPT = script
    settings = {'enter': 'pick_delivery = "dm"\ncontest_state_required = true',
                'research': 'pick_delivery = "private"'}
    fake.config.write_text(f'[scheduler]\nentry_intent = "{intent}"\n{settings[intent]}\n')
    old_line = f"*/15 10-23 * * * uv run {ENTRY_CMD} # BTS-HETZNER"
    fake.cron.write_text(f"0 9 * * * echo foreign\n{old_line}\n# {old_line}\n")
    shown = fake.run("show")
    assert shown.returncode == 0, shown.stderr
    proposed = shown.stdout.split("Would install:\n", 1)[1].splitlines()
    installed = fake.run("install")
    assert installed.returncode == 0, installed.stderr
    assert fake.lines() == ["0 9 * * * echo foreign", *proposed]
    assert fake.run("remove").returncode == 0
    assert fake.lines() == ["0 9 * * * echo foreign"]
    return proposed

scripts = {"base": repo / "scripts/cron-setup-hetzner.sh", "f5": new_installer}
results = {key: generate(script, key) for key, script in scripts.items()}
normalize = lambda lines, label: [line.replace(str(here / "generated" / label / "home"), "<HOME>") for line in lines]
base, f5 = [normalize(results[label], label) for label in scripts]
assert [line.replace("*/15 10-23 * * * ", "*/15 * * * * ") if ENTRY_CMD in line else line for line in base] == f5
base_research, f5_research = [normalize(generate(script, key + "-research", "research"), key + "-research")
                              for key, script in scripts.items()]
assert base_research == f5_research
(here / "generated-crons.json").write_text(json.dumps(results, indent=2) + "\n")
for key in scripts:
    print(key, next(line for line in results[key] if ENTRY_CMD in line))
print("show/install/remove, marked active and commented replacement, foreign lines, all other lines and research parity passed")
