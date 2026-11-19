import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from click.testing import CliRunner

from bts.cli import cli
from bts.posting import format_result_reply
from bts.state import regenerate as regen


def post(day, hour, text, *, reply=False):
    return regen.ParsedPost(f"at://synthetic/{day}/{hour}", f"{day}T{hour}:00:00Z", text, reply)


def main(root):
    snapshot = root / "snapshot.json"
    snapshot.write_text(json.dumps({"cutoff_date": "2027-06-05", "streak_at_cutoff": 12,
                                    "saver_available": False}))
    text = format_result_reply("void", 12)
    assert text == "Void. Streak: 12"
    assert not regen.parse_result_from_reply(text).is_result
    lead = post("2027-06-06", "16", "Today's pick: Example Batter (NYM)\n\nStreak: 12")
    posts = [lead, post("2027-06-06", "22", text, reply=True)]
    out = root / "regenerated"
    with patch.object(regen, "fetch_bluesky_posts", return_value=posts):
        regen.regenerate(snapshot, "synthetic.invalid", out)
    assert json.loads((out / "streak.json").read_text()) == {"streak": 0, "saver_available": True}
    assert json.loads((out / "2027-06-06.json").read_text())["result"] is None
    print("formatter -> regenerate: a void-only post-cutoff timeline writes null result, streak 0, saver true")

    # Exercise the CLI and the feed parser separately, including a prior hit at
    # 10 and a later genuine miss. A void neither consumes nor replenishes a saver.
    def entry(day, hour, text, reply=False):
        record = SimpleNamespace(created_at=f"{day}T{hour}:00:00Z", text=text,
                                 reply=object() if reply else None)
        return SimpleNamespace(post=SimpleNamespace(uri=f"at://synthetic/{day}/{hour}", record=record))

    feed = [
        entry("2027-06-07", "22", "Void. Streak: 10", True),
        entry("2027-06-06", "22", "✅ Streak: 10", True),
        entry("2027-06-07", "16", "Today's pick: Second Batter (NYM)\n\nStreak: 10"),
        entry("2027-06-06", "16", "Today's pick: First Batter (NYM)\n\nStreak: 9"),
    ]
    client = SimpleNamespace(get_author_feed=lambda **_: SimpleNamespace(feed=feed, cursor=None))
    args = ["state", "regenerate", "--snapshot", str(snapshot), "--out-picks-dir", str(out)]
    with patch.object(regen, "_bluesky_client", return_value=client):
        result = CliRunner().invoke(cli, args, catch_exceptions=False)
        assert result.exit_code == 0, result.output
        assert json.loads((out / "streak.json").read_text()) == {"streak": 10, "saver_available": True}
        assert json.loads((out / "2027-06-07.json").read_text())["result"] is None
        feed.extend([
            entry("2027-06-08", "16", "Today's pick: Third Batter (NYM)\n\nStreak: 10"),
            entry("2027-06-08", "22", "❌ Streak reset to 0", True),
        ])
        result = CliRunner().invoke(cli, args, catch_exceptions=False)
        assert result.exit_code == 0, result.output
        assert [json.loads((out / f"2027-06-0{day}.json").read_text())["result"]
                for day in (6, 7, 8)] == ["hit", None, "miss"]
        assert json.loads((out / "streak.json").read_text()) == {"streak": 0, "saver_available": False}
    print("CLI + fake feed: preceding hit carries streak 10; void remains null; later miss stays on its day "
          "and consumes the inferred saver")


if __name__ == "__main__":
    with TemporaryDirectory() as tmp, patch("socket.socket.connect", side_effect=AssertionError("network")):
        main(Path(tmp))
