"""Exercise real formatter, author-feed parsing, timeline, state CLI and saved picks."""

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from click.testing import CliRunner

from bts.cli import cli
from bts.picks import active_streak_results, effective_daily_result, load_pick
from bts.posting import format_post, format_result_reply, format_skip_post
from bts.state import regenerate


HERE = Path(__file__).resolve().parent
PREFIX = "at://did:plc:fixture/app.bsky.feed.post/"


def reply_parsing(version):
    has_void = version == "head"
    for result in ("hit", "miss", "void"):
        for streak in range(58):
            text = format_result_reply(result, streak)
            parsed = regenerate.parse_result_from_reply(text)
            expected = (result, 0 if result == "miss" else streak)
            if result == "void" and not has_void:
                expected = (None, None)
            assert (parsed.result, parsed.streak_after) == expected
            assert parsed.is_result == (expected[0] is not None)

    examples = {
        "  \tVoid. Streak: 12\n\n": ("void", 12),
        "re: Void. Streak: 12": (None, None),
        "Void. Streak: 12 extra": (None, None),
        "void. Streak: 12": (None, None),
        "Void. streak: 12": (None, None),
        "Void. Streak: -1": (None, None),
        "Void. Streak: 12 ✅": ("hit", 12),
        "Void. Streak: 12 ❌": ("miss", 12),
        "Void.\nStreak: 12": (None, None),
        "Void. Streak: 12\nOther line": (None, None),
        "Void. Streak: ١٢": ("void", 12),
        "Void. Streak: 𝟙２": ("void", 12),
        "Void. Streak: 0012": ("void", 12),
        "Void. Streak: ¹²": (None, None),
        "Void. Streak:  12": (None, None),
        format_skip_post("Skip Batter", "NYM", 0.7, 3): (None, None),
    }
    for text, expectation in examples.items():
        if version == "base" and expectation[0] == "void":
            expectation = (None, None)
        parsed = regenerate.parse_result_from_reply(text)
        assert (parsed.result, parsed.streak_after) == expectation
        assert parsed.is_result == (expectation[0] is not None)
        print("parser", repr(text), expectation)


def entry(day, hour, key, text, parent=None):
    reply = None
    if parent:
        ref = SimpleNamespace(uri=f"{PREFIX}pick-{parent}", cid="test-cid")
        reply = SimpleNamespace(parent=ref, root=ref)
    record = SimpleNamespace(created_at=f"2026-05-{day:02d}T{hour:02d}:00:00Z", text=text, reply=reply)
    return SimpleNamespace(post=SimpleNamespace(uri=PREFIX + key, record=record))


def batter(day):
    return f"Batter {day}"


def pick(day, streak=0, double=False):
    kwargs = dict(double="Second Batter", double_p_game=0.8, double_team="LAD",
                  double_pitcher="Second Pitcher") if double else {}
    text = format_post(batter(day), "NYM", "First Pitcher", 0.82, streak, **kwargs)
    return entry(day, 15, f"pick-{day}", text)


def reply(day, result, streak, *, parent=None, hour=23):
    return entry(day, hour, f"reply-{day}-{result}", format_result_reply(result, streak), parent or day)


class FakeClient:
    def __init__(self, posts):
        self.posts = list(reversed(posts))

    def get_author_feed(self, *, actor, limit, cursor):
        assert actor == "offline.example" and limit == 100 and cursor is None
        return SimpleNamespace(feed=self.posts, cursor=None)


def payload(result, streak, saver):
    return result, {"streak": streak, "saver_available": saver}


def write_json(path, data):
    path.write_text(json.dumps(data))


def summarize(case, version, scratch, posts, days, base, head, truth, *, snap=None, double_day=None):
    expected = base if version == "base" else head
    directory = scratch / case
    directory.mkdir()
    snapshot = directory / "snapshot.json"
    out = directory / "output"
    write_json(snapshot, {"cutoff_date": "2026-04-30", "streak_at_cutoff": 0,
                          "saver_available": True, "historical_picks": []} | (snap or {}))

    live = directory / "live"
    live.mkdir()
    for day, result in zip(days, truth[0], strict=True):
        write_json(live / f"2026-05-{day:02d}.json", {"result": result, "pick": {"batter_name": batter(day)}})
    write_json(live / "streak.json", truth[1])

    runner = CliRunner()
    common = ["--snapshot", str(snapshot), "--handle", "offline.example"]
    with patch.object(regenerate, "_bluesky_client", return_value=FakeClient(posts)):
        fetched = regenerate.fetch_bluesky_posts("offline.example", "2026-04-30")
        parsed = []
        for post in fetched:
            if not post.is_reply:
                picked = regenerate.parse_pick_from_post(post.text)
                if not picked:
                    continue
                picked.uri, picked.created_at = post.uri, post.created_at
                post = picked
            parsed.append(post)
        timeline = regenerate.reconstruct_pick_timeline(parsed)
        assert [record.result for record in timeline.pick_records] == expected[0]
        if days:
            assert (timeline.final_streak, timeline.saver_available_at_end) == (
                expected[1]["streak"], expected[1]["saver_available"])
        if case == "late-void" and version == "head":
            assert [record.streak_after for record in timeline.pick_records] == [10, 9]

        rebuilt = runner.invoke(cli, ["state", "regenerate", *common, "--out-picks-dir", str(out)])
        assert rebuilt.exit_code == 0, (rebuilt.output, rebuilt.exception)
        saved = [load_pick(f"2026-05-{day:02d}", out) for day in days]
        assert [item.result for item in saved] == expected[0]
        assert json.loads((out / "streak.json").read_bytes()) == expected[1]
        if double_day:
            dd = saved[days.index(double_day)].double_down
            assert dd is not None and dd.batter_name == "Second Batter" and dd.team == "LAD"
        verified = runner.invoke(cli, ["state", "verify", *common, "--live-dir", str(live)])
        assert verified.exit_code == (0 if expected == truth else 1), (verified.output, verified.exception)
        if expected == truth:
            assert "Drift check CLEAN" in verified.stdout and not verified.stderr
        else:
            assert "Drift detected" in verified.stderr and not verified.stdout
            if case == "late-void" and version == "head":
                assert "live=hit, regen=void" in verified.stderr
                assert "streak mismatch: live=10, regen=9" in verified.stderr
        target = directory / "export.json"
        exported = runner.invoke(cli, ["state", "export", "--picks-dir", str(out), "--to", str(target)])
        assert exported.exit_code == (2 if None in expected[0] else 0), (exported.output, exported.exception)
        if None in expected[0]:
            assert "still unresolved" in exported.stderr and not target.exists()
        else:
            records = json.loads(target.read_bytes())["historical_picks"]
            assert [record["result"] for record in records] == expected[0]

    print(case, "results", expected[0], expected[1], f"verify={verified.exit_code}", f"export={exported.exit_code}")
    for line in verified.stderr.splitlines()[1:]:
        print(line)


def timeline_cases(version, scratch):
    def check(case, posts, days, base, head, truth=None, **kwargs):
        summarize(case, version, scratch, posts, days, base, head, head if truth is None else truth, **kwargs)

    check("first", [pick(1, 7), reply(1, "void", 7)], [1],
          payload([None], 0, True), payload(["void"], 7, True), snap={"streak_at_cutoff": 7})
    check("between-hits", [pick(1, 11), reply(1, "hit", 12), pick(2, 12), reply(2, "void", 12),
                           entry(2, 23, "comment", "Thanks for the pick", 2),
                           entry(3, 12, "skip", format_skip_post("Skip Batter", "NYM", 0.7, 12)),
                           pick(4, 12), reply(4, "hit", 13)], [1, 2, 4],
          payload(["hit", None, "hit"], 13, True), payload(["hit", "void", "hit"], 13, True))
    for streak in range(10, 16):
        check(f"first-void-before-miss-{streak}", [pick(1, streak), reply(1, "void", streak),
                                                   pick(2, streak), reply(2, "miss", 0)], [1, 2],
              payload([None, "miss"], 0, True), payload(["void", "miss"], 0, False),
              snap={"streak_at_cutoff": streak})
    check("last-after-spent", [pick(1, 11), reply(1, "hit", 12), pick(2, 12), reply(2, "miss", 0),
                               pick(3), reply(3, "void", 0)], [1, 2, 3],
          payload(["hit", "miss", None], 0, False), payload(["hit", "miss", "void"], 0, False))

    slots = {"pick": "void", "double_down": "void"}
    assert active_streak_results(slots) == [] and effective_daily_result(slots) == "void"
    assert effective_daily_result({"pick": "hit", "double_down": "void"}) == "hit"
    check("double-pass", [pick(1, 12, double=True), reply(1, effective_daily_result(slots), 12)], [1],
          payload([None], 0, True), payload(["void"], 12, True), double_day=1)
    check("spent-in-snapshot", [pick(1), reply(1, "void", 0)], [1],
          payload([None], 0, True), payload(["void"], 0, True), payload(["void"], 0, False),
          snap={"saver_available": False})
    check("empty", [entry(1, 12, "skip", format_skip_post("Skip Batter", "NYM", 0.7, 12))], [],
          payload([], 12, False), payload([], 12, False), snap={"streak_at_cutoff": 12, "saver_available": False})
    check("late-void", [pick(1, 9), pick(2, 9), reply(2, "void", 9, parent=1, hour=16),
                        reply(2, "hit", 10)], [1, 2],
          payload([None, "hit"], 10, True), payload(["hit", "void"], 9, True),
          payload(["void", "hit"], 10, True), snap={"streak_at_cutoff": 9})


if __name__ == "__main__":
    _, version = sys.argv
    expected_src = HERE / "base" / "src" if version == "base" else HERE.parents[1] / "src"
    print("source", regenerate.__file__)
    assert Path(regenerate.__file__).is_relative_to(expected_src)
    reply_parsing(version)
    with TemporaryDirectory(prefix="void-probe-", dir=HERE) as directory:
        timeline_cases(version, Path(directory))
