"""Exercise state written by both releases and malformed decoded fields."""
import contextlib
import html
import importlib.util
import io
import sys
import tempfile
from pathlib import Path

from fixtures import HERE, fixture
from pytest import MonkeyPatch
from bts import scheduler, web

old = fixture("probe_rendering")
spec = importlib.util.spec_from_file_location("base_scheduler", HERE / "base/src/bts/scheduler.py")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)


def run(root, mp):
    mp.setattr(web, "PICKS_DIR", root)
    good = {"best_batter": "A", "best_team": "B", "best_p": .79, "bar": .79, "streak": 10}
    base.save_state(old.state(base, good), root)
    assert scheduler.load_state(old.DAY, root).skip_summary == good
    assert "eligibility unverified" in web.render_skip_banner(web.load_scheduler_state(old.DAY)["skip_summary"])
    new = good | {"basis": "decision", "candidate_verified": True}
    path = scheduler.save_state(old.state(scheduler, new), root)
    assert base.load_state(old.DAY, root).skip_summary == new
    print("Round trips passed both ways; legacy comparison correctly labels eligibility.")

    cases = [(None, False), ({}, False), ("bad", False), ([], False), (True, False)]
    for value in ("0.9", True, None, {}, [-1], float("nan"), float("inf"), 10**400,
                  1e308, 1.5e308, -.1, 1.0000001, 2):
        cases.append((new | {"best_p": value, "bar": value}, False))
    cases += [
        (new | {"reason": " \r\n "}, False),
        (new | {"reason": "<img src=x>\u2028Line?", "best_batter": "<b>A</b>\r\nB",
                "best_team": "\"\n&", "basis": "no_decision"}, False),
        (new | {"best_batter": ["x"], "best_team": {"y": 1}, "reason": ["z"], "streak": {"x": 1}}, False),
        (new | {"best_p": 0, "bar": 1}, True),
        (new | {"best_p": 1.0, "bar": 0.0}, False),
        (new | {"best_p": .75, "bar": .80, "streak": None}, True),
        (new | {"best_p": .75, "bar": .80, "basis": "no_decision"}, False),
    ]
    for summary, k2 in cases:
        scheduler.save_state(old.state(scheduler, summary), root)
        loaded = scheduler.load_state(old.DAY, root).skip_summary
        body = scheduler.format_skip_summary(loaded)
        assert scheduler.format_skip_dm(old.DAY, loaded) == f"BTS {old.DAY}: No pick at this check. {body}"
        assert ("below the" in body) is k2 and "inf%" not in body and "Reason: ." not in body
        assert not any(char in body for char in "\r\n\x85\u2028")
        banner = web.render_skip_banner(web.load_scheduler_state(old.DAY)["skip_summary"])
        assert (f"<span>{html.escape(body)}</span>" in banner) if isinstance(summary, dict) and summary else (not banner)
        print(repr(summary), body)
    path.write_text("{")
    with contextlib.redirect_stderr(io.StringIO()) as errors:
        assert scheduler.load_state(old.DAY, root) is None
    assert "quarantined" in errors.getvalue()
    path.mkdir()
    assert web.load_scheduler_state(old.DAY) == {}
    try:
        scheduler.load_state(old.DAY, root)
    except IsADirectoryError:
        print("Directory path raises in scheduler loader before renderer; dashboard returns empty state.")
    else:
        raise AssertionError("expected directory read error")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(dir=HERE) as temporary, MonkeyPatch.context() as mp:
        run(Path(temporary), mp)
