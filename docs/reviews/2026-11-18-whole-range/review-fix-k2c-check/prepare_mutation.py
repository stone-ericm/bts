import sys
from pathlib import Path

target = Path(sys.argv[1]).resolve()
assert target.is_relative_to(Path("/private/tmp")) and target.name == "scheduler.py"
original = Path("src/bts/scheduler.py").read_text()
call = (
    "    try:\n"
    "        _season_boundary_check(date, games, runs, config, picks_dir)\n"
    "    except Exception as error:\n"
    "        _report_season_boundary_failure(error)\n\n"
)
anchors = {
    "before-check": "    # 4. Main loop",
    "before-final-fallback": "    # 5. Fallback",
}
anchor = anchors[sys.argv[2]]
assert original.count(call) == original.count(anchor) == 1
assert call + "    # 7. Next-day lookahead" in original
mutated = original.replace(call, "").replace(anchor, call + anchor)
target.write_text(mutated)
print(sys.argv[2], target)
