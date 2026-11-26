"""/api/audit-progress must reject path traversal in dir/seeds_file (audit G).

`dir` was joined unsanitized under data/{provider}_results and `seeds_file`
allowed absolute paths — a reachable caller could probe for boxes.json anywhere
(and trigger the root-SSH fan-out) or stat arbitrary absolute paths. Confine
both to their roots.
"""
import json

from bts.web import audit_progress_response


def _setup_root(tmp_path):
    d = tmp_path / "data" / "vultr_results" / "aud"
    d.mkdir(parents=True)
    (d / "boxes.json").write_text(json.dumps([]))
    return tmp_path


def test_valid_dir_invokes_scanner(tmp_path):
    root = _setup_root(tmp_path)
    status, _ = audit_progress_response(
        {"provider": "vultr", "dir": "aud"},
        project_root=root, scan=lambda *a, **k: {"ok": True},
    )
    assert status == 200


def test_dir_traversal_rejected(tmp_path):
    root = _setup_root(tmp_path)
    called = []
    status, _ = audit_progress_response(
        {"provider": "vultr", "dir": "../../../../etc"},
        project_root=root, scan=lambda *a, **k: called.append(1) or {},
    )
    assert status == 400
    assert not called  # scanner (root-SSH fan-out) never invoked


def test_absolute_seeds_file_outside_root_rejected(tmp_path):
    root = _setup_root(tmp_path)
    outside = str(tmp_path.parent)  # absolute path outside the project root
    status, _ = audit_progress_response(
        {"provider": "vultr", "dir": "aud", "seeds_file": outside},
        project_root=root, scan=lambda *a, **k: {},
    )
    assert status == 400
