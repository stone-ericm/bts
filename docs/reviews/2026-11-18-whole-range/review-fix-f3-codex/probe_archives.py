"""Exercise archive precedence, equal instants, restart and the aggregate read size."""
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from review_support import ARCHIVE, CURRENT, DATE, SWITCH, command, dated, identify_package, put, setup, state, switch


def arbitration(root):
    earlier_matching = dated(streak=30, stamp=f"{DATE}T09:00:00-04:00")
    a = dated(streak=1, stamp=f"{DATE}T10:00:00-04:00")
    b = dated(streak=2, stamp=f"{DATE}T15:00:00+01:00")
    newer = dated(streak=3, stamp=f"{DATE}T13:01:00-01:00")
    valid_switch = (SWITCH, switch())
    current_match = (CURRENT, dated(streak=30, stamp=f"{DATE}T06:00:00-04:00"))
    cases = {
        "tied_disagreement_without_revoke": ([(ARCHIVE, a), ("b", b)], ()),
        "tied_disagreement_reverse_list": ([("b", b), (ARCHIVE, a)], ()),
        "tied_disagreement_with_revoke": ([(ARCHIVE, a), ("b", b), valid_switch], ("revocation",)),
        "tied_equal_policy_inputs": ([(ARCHIVE, a), ("b", a | {"recorded_at": b["recorded_at"]})], ("streak 1",)),
        "lexically_earlier_but_later_instant": ([(ARCHIVE, b), ("b", newer)], ("streak 3",)),
        "old_match_suppresses_only_revoke": ([(ARCHIVE, earlier_matching), ("b", newer), valid_switch], ("streak 3",)),
        "old_match_suppresses_revoke_with_different_current": (
            [(CURRENT, a), (ARCHIVE, earlier_matching), ("b", newer), valid_switch], ("streak 1",)),
        "valid_current_match_beats_later_archive": ([current_match, (ARCHIVE, newer), valid_switch], ()),
        "research_never_adds_revoke": ([(ARCHIVE, newer), valid_switch], ("streak 3",)),
        "same_streak_different_policy_saver": (
            [(ARCHIVE, earlier_matching | {"saver_state": "active"}), valid_switch], ("streak 30", "revocation")),
    }
    for label, (files, reasons) in cases.items():
        intent = "research" if label.startswith("research") else "enter"
        cfg, picks, account = setup(root / label, intent=intent)
        for name, data in files:
            path = account / ("contest_streak.manual.revoked-b.json" if name == "b" else name)
            put(path, data)
        result = command(cfg, picks)
        assert state(result) == ("skip_state_review" if reasons else "deliberate_skip"), (label, result)
        assert result["exit"] == int(bool(reasons)) and len(result["sends"]) == int(bool(reasons))
        # The skip itself says "streak 30". Inspect only the reason sentences.
        detail = json.loads(result["stdout"])["detail"]
        body = detail.partition("unavailable. ")[2]
        expected_reason_text = {
            "streak 1": "records streak 1 and", "streak 3": "records streak 3 and",
            "streak 30": "records streak 30 and policy saver availability available",
            "revocation": "records a revocation on",
        }
        assert all((fragment in body) == (key in reasons) for key, fragment in expected_reason_text.items()), (label, result)
        print(json.dumps({"case": label, **result}))


def read_counts(root):
    cfg, picks, account = setup(root)
    paths = [account / f"contest_streak.manual.revoked-20260402T1{i:05}Z.json" for i in range(80)]
    for path in paths:
        put(path, dated("2026-04-02") | {"padding": "x" * 900})
    huge = account / ARCHIVE
    with huge.open("wb") as handle:
        handle.write(json.dumps(dated()).encode())
        handle.truncate(8 * 1024 * 1024)
    sizes = {}
    fds = {}
    evidence = {huge, *paths}
    original_open, original_read = os.open, os.read

    def capture_open(path, *args, **kwargs):
        fd = original_open(path, *args, **kwargs)
        if Path(path) in evidence:
            fds[fd] = Path(path)
        else:
            fds.pop(fd, None)
        return fd

    def capture_read(fd, amount):
        chunk = original_read(fd, amount)
        if fd in fds:
            path = fds[fd]
            sizes[path] = sizes.get(path, 0) + len(chunk)
        return chunk

    with patch.object(os, "open", capture_open), patch.object(os, "read", capture_read):
        result = command(cfg, picks)
    assert state(result) == "deliberate_skip" and sizes[huge] == 65537
    assert len(sizes) == 81 and sum(sizes.values()) > 65537
    print(f"Only {sizes[huge]} bytes read of an {huge.stat().st_size}-byte file. "
          f"{len(sizes)} archives opened, {sum(sizes.values())} total bytes.")

    put(huge, dated())
    assert state(command(cfg, picks)) == "skip_state_review"
    huge.unlink()
    assert state(command(cfg, picks)) == "deliberate_skip"
    print("A new CLI invocation stopped warning after the differing archive was removed.")


if __name__ == "__main__":
    identify_package()
    with TemporaryDirectory() as temp:
        root = Path(temp)
        arbitration(root / "ordering")
        read_counts(root / "bounded_reads")
