"""bts.receipt_io: durable publication and tombstone discovery for producer receipts (watchdog P1/P2)."""
import os

import pytest

import bts.receipt_io as rio


def test_a_published_receipt_is_discoverable(tmp_path):
    rio.publish(tmp_path / "a" / "b" / "r.json", b"{}\n")
    assert rio.discover(tmp_path / "a" / "b") == [tmp_path / "a" / "b" / "r.json"]


def test_a_tombstone_after_a_failed_withdrawal_is_itself_synced(tmp_path, monkeypatch):
    """Producer review r2 C6: the post-rename directory sync fails and the final file cannot be removed. The
    tombstone is written, fsynced, and its directory entry synced too; discovery returns nothing."""
    (tmp_path / "d").mkdir()
    synced, state = [], {"replaced": False}
    real_replace, real_dirsync = os.replace, rio._fsync_dir

    def replace(a, b):
        real_replace(a, b)
        state["replaced"] = True

    def dirsync(p):
        if state["replaced"] and not state.get("failed_once"):
            state["failed_once"] = True
            raise OSError("dirsync")
        synced.append(rio.Path(p))
        real_dirsync(p)
    monkeypatch.setattr(os, "replace", replace)
    monkeypatch.setattr(rio, "_fsync_dir", dirsync)
    monkeypatch.setattr(type(tmp_path), "unlink", lambda self, missing_ok=False: (_ for _ in ()).throw(OSError("rm"))
                        if self.suffix == ".json" else os.unlink(self))
    with pytest.raises(OSError):
        rio.publish(tmp_path / "d" / "r.json", b"{}\n")
    assert (tmp_path / "d" / "r.json.failed").exists() and rio.discover(tmp_path / "d") == []
    assert tmp_path / "d" in synced                                          # synced after the tombstone write



def test_an_unsealed_receipt_is_never_discovered(tmp_path):
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "r.json").write_bytes(b"{}\n")                    # complete but never sealed
    assert rio.discover(tmp_path / "d") == []


def test_the_double_refusal_leaves_no_discoverable_receipt(tmp_path, monkeypatch):
    """Producer review r2 C6: the directory fsync, the removal and the tombstone write all fail. The receipt file
    survives, but it is never sealed, so it never counts."""
    (tmp_path / "d").mkdir()
    state = {"replaced": False}
    real_replace, real_open = os.replace, open

    def replace(a, b):
        real_replace(a, b)
        state["replaced"] = True
    monkeypatch.setattr(os, "replace", replace)
    monkeypatch.setattr(rio, "_fsync_dir", lambda p: (_ for _ in ()).throw(OSError("dirsync")) if state["replaced"] else None)
    monkeypatch.setattr(type(tmp_path), "unlink", lambda self, missing_ok=False: (_ for _ in ()).throw(OSError("rm"))
                        if self.suffix == ".json" else os.unlink(self))
    monkeypatch.setattr("builtins.open", lambda f, m="r", *a, **k: (_ for _ in ()).throw(OSError("tombstone"))
                        if str(f).endswith(".failed") else real_open(f, m, *a, **k))
    with pytest.raises(OSError):
        rio.publish(tmp_path / "d" / "r.json", b"{}\n")
    assert (tmp_path / "d" / "r.json").exists() and rio.discover(tmp_path / "d") == []


def test_a_seal_failure_withdraws_the_receipt(tmp_path, monkeypatch):
    (tmp_path / "d").mkdir()
    real_open = open
    monkeypatch.setattr("builtins.open", lambda f, m="r", *a, **k: (_ for _ in ()).throw(OSError("seal"))
                        if ".sealed" in str(f) else real_open(f, m, *a, **k))
    with pytest.raises(OSError):
        rio.publish(tmp_path / "d" / "r.json", b"{}\n")
    assert rio.discover(tmp_path / "d") == [] and not (tmp_path / "d" / "r.json").exists()


def test_the_seal_is_written_only_after_the_receipt_is_durable(tmp_path, monkeypatch):
    (tmp_path / "d").mkdir()
    order = []
    real_replace, real_dirsync = os.replace, rio._fsync_dir
    monkeypatch.setattr(os, "replace", lambda a, b: (order.append(("rename", rio.Path(b).name)), real_replace(a, b))[1])
    monkeypatch.setattr(rio, "_fsync_dir", lambda p: (order.append(("dirsync",)), real_dirsync(p))[1])
    rio.publish(tmp_path / "d" / "r.json", b"{}\n")
    assert order == [("rename", "r.json"), ("dirsync",), ("rename", "r.json.sealed"), ("dirsync",)]
    assert rio.discover(tmp_path / "d") == [tmp_path / "d" / "r.json"]
