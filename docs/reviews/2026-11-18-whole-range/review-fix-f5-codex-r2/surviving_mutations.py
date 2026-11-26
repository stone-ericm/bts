"""Exercise the states that two surviving scratch changes would alter."""
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from marker_health import CURRENT, PREVIOUS, Replay, dated, health
from state_sequences import DAY, WRONG, edit_pitch, row, source


if __name__ == "__main__":
    version = sys.argv[1]
    source(version)
    with TemporaryDirectory() as directory:
        root = Path(directory)
        world = Replay(root / "manual", version)
        daily = world.commit_at("10:05", at="09:40")
        if version == "f5":
            early = world.invoke("09:45", pending=[row()])
            assert early["marker"]["status"] == "confirmed" and early["marker"]["early_cron_run"] is True
        edit_pitch(world, daily, "10:50")
        world.capture([WRONG], f"{DAY}T140000Z")
        manual = world.invoke("10:00", pending=[row()], manual=True)
        if version == "f5":
            assert manual["receipt"]["outcome"] == "already_confirmed" and not manual["calls"] and not manual["sent"]
        else:
            assert "pending" in manual["calls"] and "entry is recorded in game 102" in manual["sent"][0]
        print("10:00 manual", version, manual["receipt"]["outcome"], manual["sent"])
        world = Replay(root / "copy", day=CURRENT)
        world.set_marker({"date": PREVIOUS, "status": "confirmed"})
        dated(world).write_text(f'{{"date":"{PREVIOUS}","status":"dm_failed"}}')
        assert health(world) == []
        print("current live confirmed marker took precedence over same-date dm_failed copy")
