"""atomic_write_text: a crash mid-write must never truncate the prior good file.

Bare path.write_text() can leave a torn/truncated file if the process is killed
mid-write (deploy restarts are routine). A torn pick/streak JSON then crashes
every loader -> a silent crash-loop the heartbeat monitor can't see (audit D1).
"""
import json
import pytest

from bts.util import atomic_write_text


def test_roundtrip_and_no_tmp_left(tmp_path):
    p = tmp_path / "x.json"
    atomic_write_text(p, json.dumps({"a": 1}))
    assert json.loads(p.read_text()) == {"a": 1}
    assert not list(tmp_path.glob("*.tmp"))


def test_failure_preserves_existing_file(tmp_path, monkeypatch):
    p = tmp_path / "x.json"
    atomic_write_text(p, json.dumps({"v": 1}))
    good = p.read_text()

    # Simulate a crash at the atomic-commit step.
    monkeypatch.setattr(
        "bts.util.os.replace",
        lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")),
    )
    with pytest.raises(OSError):
        atomic_write_text(p, json.dumps({"v": 2}))

    assert p.read_text() == good            # original intact, not truncated
    assert not list(tmp_path.glob("*.tmp"))  # temp file cleaned up
