"""Render the proposed terminal suffix in real WARN/CRITICAL DM envelopes."""

from dataclasses import replace
from datetime import timedelta
import logging
import os
from pathlib import Path
import signal
import socket
from tempfile import TemporaryDirectory
from unittest.mock import patch

from final_copy import POINTER
import probe


def render(picks, day, wall, reasons):
    from bts.health.alert import format_health_dm_body
    from bts.health.attention import build_policy_alerts

    base, sent = probe.real_runner(picks, day, wall)
    reference, _ = probe.real_runner(picks, day, probe.end(day))
    historical = [replace(alert, message=f"Prior slate {day} ET. {reason} {POINTER}")
                  for alert, reason in zip(reference, reasons, strict=True)]
    _, attention = build_policy_alerts(historical, streaks={})
    corrected_attention = []
    for item in attention:
        before, marker = item.message.split(POINTER, maxsplit=1)
        assert marker == " (1st observed day)"
        corrected_attention.append(replace(item, message=f"{before.rstrip()}{marker} {POINTER}"))
    dm = format_health_dm_body([item for item in historical if item.level == "CRITICAL"], corrected_attention)
    assert dm
    assert all(line.endswith(POINTER) for line in dm.splitlines() if line.startswith("- ["))
    print("Base:", [(item.level, item.source, item.message) for item in base])
    print("Candidate:", dm, sep="\n")
    return base, sent


def run(root):
    day = probe.YEAR_END
    following = day + timedelta(days=1)
    picks = root / "year_end_dated_saver" / "picks"
    probe.dated(picks, day, streak=12)
    probe.auto(picks, following, source_date=day.isoformat())
    probe.switch(picks, following)
    probe.saver_flag(picks, following)
    base, _ = render(picks, day, probe.at(following, 0, 15), [
        "The saver state in the dated observation contradicts its streak. The saver counted as spent.",
    ])
    assert not base

    day = probe.D
    following = day + timedelta(days=1)
    picks = root / "late_replay_after_cutoff" / "picks"
    probe.auto(picks)
    probe.switch(picks, status="revoked")
    probe.saver_flag(picks, state="not_earned")
    from bts.picks import DailyPick, Pick, save_pick, submission_cutoff_et

    game = probe.at(following, 14)
    pick = Pick(batter_name="Synthetic", batter_id=1, team="CLE", lineup_position=1,
                pitcher_name="Synthetic Pitcher", pitcher_id=2, p_game_hit=0.6, flags=[],
                projected_lineup=False, game_pk=3, game_time=game.isoformat())
    run_time = game - timedelta(hours=1)
    daily = DailyPick(date=following.isoformat(), run_time=run_time.isoformat(), pick=pick,
                      double_down=None, runner_up=None, delivered_at=run_time.isoformat())
    path = save_pick(daily, picks)
    snapshot = path.read_bytes()
    wall = probe.at(following, 14, 1)
    assert wall > submission_cutoff_et(daily)
    render(picks, day, wall, [
        "The saver flag contradicts the recorded streak. The saver counted as spent.",
        "The season switch is revoked.",
    ])
    assert path.read_bytes() == snapshot

    for label in ("confirmed_no_prior_entry", "prior_entry_uncertain"):
        picks = root / label / "picks"
        probe.switch(picks)
        render(picks, day, probe.at(following, 2), [
            "Contest streak observation is missing from account_state.",
            "There is no automatic observation for the switch activation.",
        ])
        # Contest-file states cannot distinguish these two MLB-side fixtures.
        # The copy works in both because its only verification is in MLB.


if __name__ == "__main__":
    signal.alarm(50)
    logging.disable(logging.CRITICAL)
    with TemporaryDirectory(dir=probe.HERE) as scratch:
        scratch = Path(scratch)
        os.chdir(scratch)
        with patch.object(socket, "socket", probe.deny_network), patch.object(socket, "create_connection", probe.deny_network):
            run(scratch)
    signal.alarm(0)
