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
