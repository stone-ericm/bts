"""Check source substitution and receipt replay using the draft's acceptance predicates."""

import hashlib
import json
import runpy
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
base = runpy.run_path(str(HERE / "capture_probe.py"))["exercise"].__globals__
artifacts, capture, pd = (base[name] for name in ("artifacts", "capture", "pd"))
DAY, COMMIT, CANDIDATE, VARIANTS = (base[name] for name in ("DAY", "COMMIT", "CANDIDATE", "VARIANTS"))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def snapshot(directory):
    manifest = (directory / "manifest.json").read_bytes()
    parsed = json.loads(manifest)
    paths = {variant: parsed["profile_paths"][variant][DAY] for variant in VARIANTS}
    profiles = {path: (directory / path).read_bytes() for path in paths.values()}
    return parsed, paths, {"manifest.json": manifest} | profiles


def draft_receipt_match(source, receipts, target, *, expected=10):
    """Evaluate the stated identity, byte, profile hash and exact-count predicates."""
    manifest, paths, files = snapshot(source)
    manifest_hash = digest(files["manifest.json"])
    path = receipts / f"{DAY}.{manifest_hash}.json"
    if not path.is_file():
        return False
    receipt = json.loads(path.read_bytes())
    target_manifest, target_paths, _ = snapshot(target)
    effective_top_n = receipt["effective_top_n"]
    return (
        receipt["schema"] == "bts_thin_capture_receipt_v1"
        and type(effective_top_n) is type(receipt["top_n"]) is int
        and 1 <= effective_top_n < receipt["top_n"] == expected
        and all(receipt[key] == manifest[key] == target_manifest[key]
                for key in ("date", "candidate_name", "git_commit", "top_n"))
        and receipt["manifest_sha256"] == manifest_hash
        and all(receipt["profile_sha256"][variant] == digest(files[paths[variant]])
                for variant in VARIANTS)
        and all(pd.read_parquet(target / target_paths[variant]).groupby("date").size()
                .eq(effective_top_n).all() for variant in VARIANTS)
    )


def base_failures(directory, *, resolved=False):
    report = artifacts.verify_candidate_artifact_pair(
        artifact_dir=directory, expected_date=DAY, expected_candidate=CANDIDATE,
        expected_git_commit=COMMIT, expected_top_n=10, require_live_preoutcome=not resolved,
        require_production_pick_snapshot=True,
        expected_run_kind="live_forward_resolved" if resolved else "live_forward_preoutcome",
    )
    return [check["name"] for check in report["checks"] if check["status"] == "fail"]


def run(root):
    config, pick, live = base["fixture"](root, "capture")
    receipts = live.parent.with_name(live.parent.name + "_thin_receipts")
    receipts.mkdir()
    rc, status = capture.capture_once(config)
    assert (rc, status["status"]) == (1, "failed_verify")
    thin_manifest, paths, thin_files = snapshot(live)
    thin_hash = digest(thin_files["manifest.json"])
    receipt = {key: thin_manifest[key] for key in ("date", "candidate_name", "git_commit", "top_n")}
    receipt.update(schema="bts_thin_capture_receipt_v1", effective_top_n=9, manifest_sha256=thin_hash,
                   profile_sha256={variant: digest(thin_files[paths[variant]]) for variant in VARIANTS})
    receipt_path = receipts / f"{DAY}.{thin_hash}.json"
    with receipt_path.open("x") as file:
        json.dump(receipt, file)
    old_receipt_bytes = receipt_path.read_bytes()

    base["SCORES"] = (10, 10)
    overwrite = base["dataclasses"].replace(config, overwrite=True)
    rc, status = capture.capture_once(overwrite)
    assert (rc, status["status"]) == (0, "exported_verified")
    full_manifest, _, full_files = snapshot(live)
    assert not draft_receipt_match(live, receipts, live)

    base["add_outcomes"](config)
    resolved = root / "resolved"
    artifacts.resolve_live_candidate_artifact_pair(
        artifact_dir=live, output_dir=resolved, data_dir=config.production_root / config.data_dir,
    )
    assert base_failures(resolved, resolved=True) == []

    base["rewrite"](live, 9, update_manifest=True)
    rewritten_full_matches = draft_receipt_match(live, receipts, live)
    assert not rewritten_full_matches
    for path, content in thin_files.items():
        (live / path).write_bytes(content)
    replay_matches = draft_receipt_match(live, receipts, live)
    replay_failures = base_failures(live)
    assert replay_matches and len(replay_failures) == 2

    another_root = root / "copied" / DAY
    for path, content in thin_files.items():
        target = another_root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    cross_root_matches = draft_receipt_match(another_root, receipts, another_root)
    assert cross_root_matches

    base["rewrite"](resolved, 9, update_manifest=True)
    resolved_path = resolved / "manifest.json"
    amended = base["read"](resolved_path)
    amended["outcome_status_counts"]["resolved"] = 18
    base["write"](resolved_path, amended)
    assert full_manifest["row_counts"]["production"][DAY] == 10
    resolved_failures = base_failures(resolved, resolved=True)
    assert len(resolved_failures) == 2 and all("top_n_rows_per_date" in name for name in resolved_failures)
    resolved_matches = draft_receipt_match(live, receipts, resolved)
    assert resolved_matches
    assert receipt_path.read_bytes() == old_receipt_bytes

    for path, content in full_files.items():
        (live / path).write_bytes(content)
    assert not draft_receipt_match(live, receipts, resolved)
    return {"plain_full_rewrite_matches_receipt": rewritten_full_matches,
            "restored_thin_after_full_overwrite_matches_receipt": replay_matches,
            "restored_thin_base_failures": replay_failures,
            "copied_thin_in_another_root_matches_shared_receipt": cross_root_matches,
            "shortened_full_resolution_matches_replacement_source_receipt": resolved_matches,
            "shortened_full_resolution_base_failures": resolved_failures,
            "existing_receipt_bytes_unchanged": True}


if __name__ == "__main__":
    with (base["patch"].object(base["predict"], "run_pipeline", base["predictions"]),
          base["patch"].object(artifacts, "current_git_commit", lambda: COMMIT),
          base["patch"].object(artifacts, "utc_timestamp", lambda: base["NOW"]),
          base["patch"].object(capture, "git_head", lambda _: COMMIT),
          base["patch"].object(capture, "run_bts", base["transport"])):
        with tempfile.TemporaryDirectory(prefix="receipt-", dir=HERE) as temp:
            result = run(Path(temp))
        result["transports"] = base["CALLS"]
        print(json.dumps(result, indent=2))
