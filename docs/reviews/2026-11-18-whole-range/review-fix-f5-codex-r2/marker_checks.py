"""Check dated marker publication and Click's receipt from an interrupted write."""
from pathlib import Path
from tempfile import TemporaryDirectory

from marker_health import (
    CURRENT, NEXT, PREVIOUS, RAW, Replay, dated, health, health_inputs, row, source, write_inputs,
)


def rollover(root):
    world = Replay(root, day=CURRENT)
    world.commit_at("06:10", at="04:55")
    world.set_marker({})
    world.marker.write_bytes(RAW)
    assert health(world)[0][0] == "CRITICAL"
    first = world.invoke("05:00", auth="profile")
    assert not dated(world).exists() and world.marker.read_bytes() == RAW and first["calls"]

    def interrupt(path, value):
        assert path == world.marker and value["date"] == CURRENT
        assert dated(world).read_bytes() == RAW and path.read_bytes() == RAW
        raise KeyboardInterrupt

    result = world.invoke("05:15", pending=[row()], patches={"bts.cli._atomic_write_json": interrupt})
    assert result["receipt"]["detail"] == "raised KeyboardInterrupt", result
    assert dated(world).read_bytes() == world.marker.read_bytes() == RAW
    inode = dated(world).stat().st_ino
    restart = Replay(root, day=CURRENT)
    result = restart.invoke("05:30", dm="failed")
    assert result["marker"]["status"] == "dm_failed" and result["marker"]["date"] == CURRENT
    assert dated(world).stat().st_ino == inode and dated(world).read_bytes() == RAW
    second_raw = world.marker.read_bytes()
    assert health(world)[0][0] == health(world, CURRENT)[0][0] == "CRITICAL"
    third = Replay(root, day=NEXT)
    third.commit_at("06:10", at="04:55")
    result = third.invoke("05:00", pending=[row()])
    assert result["marker"]["date"] == NEXT and result["marker"]["status"] == "confirmed"
    assert dated(world, CURRENT).read_bytes() == second_raw and dated(world).read_bytes() == RAW
    assert [health(world, day)[0][0] for day in (PREVIOUS, CURRENT)] == ["CRITICAL", "CRITICAL"]
    print("exact bytes; auth failure unchanged; interrupt before marker; existing copy inode kept; two CRITICALs")


if __name__ == "__main__":
    source("f5")
    with TemporaryDirectory() as directory:
        root = Path(directory)
        rollover(root / "rollover")
        health_inputs(root / "health")
        write_inputs(root / "inputs")
