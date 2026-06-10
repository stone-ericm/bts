"""safe_filename_component: untrusted leaderboard usernames -> safe filenames."""
from bts.leaderboard.storage import safe_filename_component


def test_plain_username_unchanged():
    assert safe_filename_component("tombrady12") == "tombrady12"
    assert safe_filename_component("a-b_c.d") == "a-b_c.d"


def test_path_traversal_neutralized():
    assert "/" not in safe_filename_component("../../etc/passwd")
    assert safe_filename_component("..") == "_"
    assert safe_filename_component(".") == "_"
    assert safe_filename_component("") == "_"


def test_slashes_and_specials_stripped():
    assert safe_filename_component("a/b\\c") == "a_b_c"
    # dots are safe WITHIN a name; only the separators are neutralized, so the
    # result is a single traversal-free path component.
    out = safe_filename_component("../leaderboard_snapshots/2026-06-09")
    assert out == ".._leaderboard_snapshots_2026-06-09"
    assert "/" not in out and out not in (".", "..")
