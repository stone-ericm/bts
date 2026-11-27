from pathlib import Path
from tempfile import TemporaryDirectory

from probe_startup import DAY, HAS_WARNING, Startup, game
from bts import picks
from bts.picks import DailyPick, Pick


def already_sent(root):
    start = Startup(root / "sent", mode="public", required=True)
    picks_dir = Path(start.config["orchestrator"]["picks_dir"])
    daily = DailyPick(
        date=DAY, run_time=f"{DAY}T15:00:00+00:00",
        pick=Pick(batter_name="Already communicated", batter_id=11, team="AWY",
                  lineup_position=1, pitcher_name="Starter", pitcher_id=22,
                  p_game_hit=.82, flags=[], projected_lineup=False, game_pk=301,
                  game_time=f"{DAY}T23:05:00Z"),
        double_down=None, runner_up=None, bluesky_posted=True,
        bluesky_uri="at://offline/prior-commit",
    )
    path = picks.save_pick(daily, picks_dir)
    previous = path.read_bytes()
    _, journal = start.run([game()], at="15:54:55")
    assert path.read_bytes() == previous
    assert len(start.bodies) == int(HAS_WARNING), (start.bodies, journal)
    if HAS_WARNING:
        assert "Keep any pick already sent or entered as recorded." in start.bodies[0]
        assert "Check the last regular-season game date on MLB's updated schedule" in start.bodies[0]
        print(start.bodies[0])


def final_and_past(root):
    day = "2028-09-29"
    start = Startup(root / "final", mode="private")
    completed = game(day, status={"abstractGameCode": "F", "detailedState": "Final"})
    _, journal = start.run([completed], at="09:00:00", day=day)
    assert len(start.bodies) == int(HAS_WARNING)
    if HAS_WARNING:
        assert "1 eligible regular-season game" in journal
        body = start.bodies[0]
        assert "The fallback exclusive boundary for the unconfigured year 2028 is 2028-09-28" in body
        assert "a fresh pick selection with an available predicted candidate" in body
        assert len(body) < 1000
        print(body)
    late = Startup(root / "past")
    checks, journal = late.run([game(day)], day=day, at="2028-09-30T00:01:00")
    assert not late.bodies
    assert checks == [("2028-09-30T00:01:00-04:00", [301])]
    if HAS_WARNING:
        assert "first lineup check is due within 10 minutes" in journal
        print(journal)


if __name__ == "__main__":
    with TemporaryDirectory() as directory:
        root = Path(directory)
        already_sent(root)
        final_and_past(root)
