from pathlib import Path
from shutil import copy2, copytree, ignore_patterns


root = Path(__file__).resolve().parents[2]
scratch = Path(__file__).resolve().parent / "scratch"
scratch.mkdir()
for directory in ("src", "tests", "docs", "config", "data", "scripts", "deploy", ".github"):
    copytree(root / directory, scratch / directory, ignore=ignore_patterns("__pycache__"))
for filename in ("pyproject.toml", "CLAUDE.md", "ARCHITECTURE.md", "README.md", "INCIDENT.md", "Dockerfile", ".gitignore", "uv.lock"):
    copy2(root / filename, scratch / filename)

path = scratch / "src/bts/health/contest_state.py"
base = path.read_text()
marker = '        return alerts\n\n    # The flag belongs to the caller'
block = '''        if state.saver_state == "used":
            from bts.saver_state import _read_saver_state, load_saver_state

            season = now.astimezone(ZoneInfo("America/New_York")).year
            saver = (_read_saver_state(picks_dir, season=season, historical=True)
                     if historical else load_saver_state(picks_dir, season=season))
            if saver.state in {"active", "not_earned"}:
                if historical:
                    message = ("The dated observation says the saver was used while the saver flag does not. "
                               "The saver flag may still count it as in hand.")
                else:
                    directory = shlex.quote(str(picks_dir))
                    message = (
                        "Streak Saver: today's dated observation says MLB used the saver, but the saver "
                        f"flag for season {season} reads {saver.state}. When no dated observation is in effect, "
                        "the live decision reads the flag and counts this saver as still in hand. Mark it used: "
                        f"`bts saver-state --use --season {season} --picks-dir {directory}`. Run it in {season} ET "
                        "away from midnight. It changes only the flag; the dated observation stays as it is."
                    )
                alerts.append(Alert(level="WARN", source=SAVER_SOURCE, message=message))
'''
assert base.count(marker) == 1
path.write_text(base.replace(marker, block + marker))
print(path)
