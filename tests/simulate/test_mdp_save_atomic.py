"""MDPSolution.save() must write atomically.

A reader (`load_policy` / strategy._load_mdp) must never see a half-written `.npz`,
even if the write is interrupted (killed process, disk full) mid-`np.savez_compressed`.
The save writes to a temp file in the destination directory, then `os.replace`s it
into place, so the target always points at a complete file (old or new).
"""
import os
import stat

import numpy as np
import pytest

from bts.simulate.mdp import MDPSolution, load_policy
from bts.simulate.quality_bins import QualityBins


def _tiny_solution(fill: int = 1) -> MDPSolution:
    """A minimal valid MDPSolution. save() only reads policy_table, boundaries,
    season_length, optimal_p57 (value_table is not persisted)."""
    qb = QualityBins(bins=[], boundaries=[0.5, 0.6, 0.7, 0.8])
    return MDPSolution(
        optimal_p57=0.0333,
        value_table=np.zeros((58, 11, 2, 5)),
        policy_table=np.full((57, 11, 2, 5), fill, dtype=int),
        quality_bins=qb,
        season_length=10,
    )


def test_save_roundtrips_through_load_policy(tmp_path):
    out = tmp_path / "policy.npz"
    sol = _tiny_solution()
    sol.save(out)

    table, boundaries, season_length = load_policy(out)
    np.testing.assert_array_equal(table, sol.policy_table)
    assert boundaries == [0.5, 0.6, 0.7, 0.8]
    assert season_length == 10


def test_save_leaves_no_temp_files_on_success(tmp_path):
    out = tmp_path / "policy.npz"
    _tiny_solution().save(out)
    assert [p.name for p in tmp_path.iterdir()] == ["policy.npz"]


def test_interrupted_write_does_not_corrupt_existing_target(tmp_path, monkeypatch):
    """A write that emits a partial file then dies must not damage the prior file.

    The simulated `savez_compressed` writes garbage to whatever it is handed, then
    raises. Direct-to-target (non-atomic) saving corrupts `out`; atomic saving writes
    the garbage to a temp file, so the real target survives. This is the red→green
    test: it fails on the original direct write.
    """
    out = tmp_path / "policy.npz"
    _tiny_solution(fill=1).save(out)          # establish a valid file (all "single")
    original = out.read_bytes()

    def partial_then_fail(file, **kwargs):
        # `file` is a file object (atomic temp) or a path (direct write) -- corrupt
        # whichever it is, then die mid-write.
        if hasattr(file, "write"):
            file.write(b"PK\x03\x04 truncated npz")
        else:
            with open(file, "wb") as fh:
                fh.write(b"PK\x03\x04 truncated npz")
        raise RuntimeError("killed mid-write")

    monkeypatch.setattr("bts.simulate.mdp.np.savez_compressed", partial_then_fail)
    with pytest.raises(RuntimeError):
        _tiny_solution(fill=2).save(out)

    # the prior valid file is untouched (not truncated/clobbered)...
    assert out.read_bytes() == original
    table, _, _ = load_policy(out)
    np.testing.assert_array_equal(table, np.full((57, 11, 2, 5), 1, dtype=int))
    # ...and no partial temp file is left behind
    assert [p.name for p in tmp_path.iterdir()] == ["policy.npz"]


def test_save_preserves_existing_target_mode(tmp_path):
    """An atomic save must not silently tighten perms: regenerating an existing 0644
    policy keeps it 0644 (mkstemp creates the temp 0600)."""
    out = tmp_path / "policy.npz"
    _tiny_solution().save(out)
    os.chmod(out, 0o644)
    _tiny_solution(fill=2).save(out)
    assert stat.S_IMODE(out.stat().st_mode) == 0o644


def test_fresh_save_uses_umask_default_not_0600(tmp_path):
    """A brand-new policy file gets the umask-respecting mode a normal write would,
    not mkstemp's restrictive 0600 (which would crash a different-user reader since
    strategy._load_mdp does not catch PermissionError)."""
    out = tmp_path / "policy.npz"
    _tiny_solution().save(out)
    umask = os.umask(0)
    os.umask(umask)
    assert stat.S_IMODE(out.stat().st_mode) == (0o666 & ~umask)
