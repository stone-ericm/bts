"""Reproduce opening-window faults and exercise delivery recovery controls."""

import argparse
from contextlib import redirect_stderr
from datetime import timedelta
from io import BytesIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType
from unittest.mock import patch

from opening_rehearsal import DAYS, HERE, DM, Season, at, read_json, stop_scheduler, write_json


def picks_sent(season):
    return [message for message in season.messages if message["body"].startswith("Today's pick")]


def finish(season, day, outcome_exit):
    tomorrow = (at(day, "08:30") + timedelta(days=1))
    season.clock.move(tomorrow)
    return json.loads(season.outcome(day, expected=outcome_exit))


def correction_between_reads(season, correct_first=False):
    """The clock is fixed while one MLB status request and one CLI write overlap."""
    from bts import scheduler

    day = DAYS[1]
    season.clock.move(at(day, "09:30"))
    season.dated(day, 1 if correct_first else 46,
                 saver="not_earned" if correct_first else "used")
    original = scheduler._delivery_game_statuses
    corrections = []

    def during_status_request(date):
        if not correct_first and not corrections:
            corrections.append(season.snapshot(day)["pick"]["chosen_on"])
            season.dated(day, 1)
        return original(date)

    season.clock.move(at(day, "10:00"))
    season.clock.when(at(day, "15:55"), lambda: season.enter(day))
    with patch.object(scheduler, "_delivery_game_statuses", during_status_request):
        season.start(day)
    original_choice = corrections[0] if corrections else None
    if not correct_first:
        assert original_choice["streak"] == 46 and original_choice["saver_available"] is False
    return {"initial_choice": original_choice, "messages": season.messages,
            "snapshot": season.snapshot(day), "next_morning": finish(season, day, 0)}


def transport_retry(season):
    """Use real DM request construction, with a captured HTTP response lost after acceptance."""
    from bts import dm

    day = DAYS[0]
    season.clock.move(at(day, "09:30"))
    season.dated(day, 0)
    sent = []
    original_transport = vars(dm).get("send_dm")
    # Season.run_offline patches the symbol; the unpatched callable is supplied by main.
    assert original_transport == season.dm

    def fake_http(request, *, timeout, idempotent=True):
        route = request.full_url.rsplit("/", 1)[1].split("?", 1)[0]
        if route == "com.atproto.server.createSession":
            response = {"accessJwt": "offline"}
        elif route == "com.atproto.identity.resolveHandle":
            assert request.full_url.endswith(DM)
            response = {"did": "did:plc:synthetic"}
        elif route == "chat.bsky.convo.getConvoForMembers":
            response = {"convo": {"id": "synthetic"}}
        else:
            assert route == "chat.bsky.convo.sendMessage" and idempotent is False
            data = json.loads(request.data)
            assert data["convoId"] == "synthetic"
            wire_id = f"wire-{len(sent) + 1}"
            sent.append({"at": season.clock.now.isoformat(), "recipient": DM,
                         "body": data["message"]["text"], "id": wire_id})
            if len(sent) == 1:
                class LostResponse:
                    def read(self):
                        raise TimeoutError("synthetic timeout reading an accepted DM response")
                return LostResponse()
            response = {"id": wire_id}
        return BytesIO(json.dumps(response).encode())

    season.clock.move(at(day, "10:00"))
    season.clock.when(at(day, "18:50"), lambda: season.enter(day, expected=None))
    with patch.object(dm, "send_dm", season.real_transport), \
            patch.object(dm, "retry_urlopen", fake_http), \
            patch.object(dm, "get_dm_password", lambda: "offline-password"):
        season.start(day)
    return {"accepted_by_transport": sent, "messages": season.messages,
            "snapshot": season.snapshot(day), "journal": season.journal.getvalue()}


def restart_at_send(season, phase):
    corrected = phase == "corrected_after_marker"
    day = DAYS[1] if corrected else DAYS[0]
    season.clock.move(at(day, "09:30"))
    season.dated(day, 1 if corrected else 0)
    season.clock.move(at(day, "10:00"))
    boundary = []
    if phase == "after_checkpoint":
        season.clock.when(at(day, "16:20"), stop_scheduler)
        season.clock.when(at(day, "18:50"), lambda: season.enter(day))
    else:
        def capture_boundary():
            boundary.append(season.snapshot(day))
            stop_scheduler()

        if phase == "after_remote_accept":
            season.cut_after_transport = capture_boundary
        if corrected:
            season.clock.when(at(day, "15:25:30"), lambda: season.dated(day, 46, saver="used"))
        season.clock.when(at(day, "15:40" if corrected else "18:30"),
                          lambda: verify_unconfirmed_entry(season))

    original = season.record_transport

    def transport(channel, recipient, body):
        if phase in {"before_remote_accept", "corrected_after_marker"} and body.startswith("Today's pick") and not boundary:
            capture_boundary()
        return original(channel, recipient, body)

    with patch.object(season, "record_transport", transport):
        season.start(day)
    assert len(season.restarts) == 1
    if phase == "after_checkpoint":
        assert len(picks_sent(season)) == 1
        assert season.snapshot(day)["decision"]["delivery_status"] == "delivered"
    else:
        assert boundary[0]["pick"]["delivery_attempted"] is True
        assert boundary[0]["pick"]["notification_sent"] is False
        assert len(picks_sent(season)) == int(phase == "after_remote_accept")
        assert season.snapshot(day)["decision"]["delivery_status"] == "locked_unconfirmed"
    outcome = finish(season, day, 0 if phase == "after_checkpoint" else 1)
    return {"boundary": boundary, "restarts": season.restarts, "messages": season.messages,
            "snapshot": season.snapshot(day), "next_morning": outcome}


def verify_unconfirmed_entry(season):
    season.command("check-pick-entered", "--expected-username", "synthetic_account",
                   "--dm-recipient", DM, expected=None)


def preview_control(season):
    day = DAYS[0]
    season.clock.move(at(day, "03:00"))
    season.projected_on[day] = "16:00"
    predict = ModuleType("bts.model.predict")
    predict.run_pipeline = lambda date, *a, **kw: season.mark_predictions(date)
    predict.load_blend = lambda path: None
    with patch.dict(sys.modules, {"bts.model.predict": predict}):
        season.command("preview", "--date", day, contains="PROJECTED")
    preview = season.snapshot(day)["pick"]
    assert preview["chosen_on"]["streak"] == 17
    assert preview["chosen_on"]["required"] is False
    season.dated(day, 0)
    season.cascade_outages = {(day, time) for time in ("16:05", "17:05", "18:20", "18:30")}
    season.clock.move(at(day, "10:00"))
    season.start(day)
    result = season.snapshot(day)
    assert not picks_sent(season) and result["decision"] is None
    assert any("chosen without the live check" in m["body"] for m in season.messages)
    outcome = finish(season, day, 1)
    return {"preview": preview, "snapshot": result, "messages": season.messages,
            "next_morning": outcome}


def corrected_cached_pick(season):
    day = DAYS[0]
    season.clock.move(at(day, "09:30"))
    season.dated(day, 11, saver="used")
    season.first_lineups[day] = "18:24"
    season.projected_on[day] = "18:24"
    season.cascade_outages = {(day, time) for time in ("17:05", "18:20", "18:30")}
    season.clock.move(at(day, "10:00"))
    season.clock.when(at(day, "18:23"), lambda: season.dated(day, 0))
    season.start(day)
    snap = season.snapshot(day)
    assert not picks_sent(season) and snap["decision"] is None
    assert snap["pick"]["chosen_on"]["streak"] == 11
    assert any("was not delivered because it was chosen on" in m["body"] for m in season.messages)
    assert any("NO PICK DELIVERED" in m["body"] for m in season.messages)
    return {"snapshot": snap, "messages": season.messages, "next_morning": finish(season, day, 1)}


def cutover_with_masked_file(season):
    import httpx

    day = DAYS[1]
    season.clock.move(at(day, "09:30"))
    season.dated(day, 1)
    season.command("season-contract", "enable", "--season", 2027)
    season.fetch_failure = httpx.ConnectError("captured outage")
    season.fetch(expected=2, contains="upstream/network")
    before = season.status(contains="live decision now: accepted")
    assert "automatic: parsed schema='bts_contest_streak_auto_v1' streak=17" in before
    season.clock.move(at(DAYS[2], "10:00"))
    after = season.status(contains="live decision now: refused")
    season.start(DAYS[2])
    assert not picks_sent(season)
    assert any("Morning check for 2027-03-26" in m["body"] for m in season.messages)
    return {"before": before, "after": after, "messages": season.messages,
            "next_morning": finish(season, DAYS[2], 1)}


def ambiguous_saver_instructions(season, expect_fixed):
    day = DAYS[0]
    season.clock.move(at(day, "09:30"))
    season.command("season-contract", "enable", "--season", 2027)
    season.streak = 11  # Synthetic stale profile counter; MLB UI actually shows 0.
    season.fetch(contains="wrote")
    season.command("saver-state", "--init", "not_earned", "--season", 2027)
    season.clock.move(at(day, "10:00"))
    season.clock.when(at(day, "18:50"), lambda: season.enter(day))
    season.start(day)
    warn = [m["body"] for m in season.messages if "contradicts streak 11" in m["body"]]
    assert len(warn) == 1
    refused = None
    if expect_fixed:
        assert f"set-contest-streak --date {day}" in warn[0] and "--saver" in warn[0]
    else:
        assert "with bts set-contest-streak" in warn[0]
        season.command("set-contest-streak", "--streak", 0)
        refused = season.status(contains="legacy manual override is no longer accepted")
    season.dated(day, 0)
    recovered = season.status(contains="live decision now: accepted")
    return {"warning": warn[0], "generic_command_status": refused,
            "dated_command_status": recovered, "snapshot": season.snapshot(day),
            "messages": season.messages}


def first_lineups_after_last_check(season):
    day = DAYS[0]
    season.clock.move(at(day, "09:30"))
    season.dated(day, 0)
    season.first_lineups[day] = "18:24"
    season.clock.move(at(day, "10:00"))
    season.start(day)
    snap = season.snapshot(day)
    assert not picks_sent(season) and snap["pick"] is None and snap["decision"] is None
    assert not any("NO PICK DELIVERED" in m["body"] for m in season.messages)
    return {"snapshot": snap, "messages": season.messages,
            "next_morning": finish(season, day, 1)}


def stale_warning(season, at_start):
    day = DAYS[0]
    if at_start:
        season.clock.move(at("2027-03-23", "15:00"))
    season.command("season-contract", "enable", "--season", 2027)
    # Use an empty account history to exercise the accepted coverage alert, too.
    season.profile_predictions.clear()
    season.fetch(contains="wrote")
    season.command("saver-state", "--init", "not_earned", "--season", 2027)
    season.clock.move(at(day, "10:00"))
    season.clock.when(at(day, "12:00"), stop_scheduler)
    season.clock.when(at(day, "18:50"), lambda: season.enter(day))
    season.start(day)
    snap = season.snapshot(day)
    warnings = [m for m in season.messages if "stale at startup" in m["body"]
                or "automatic contest reading flagged stale" in m["body"]]
    assert len(warnings) == 1 and warnings[0]["at"][11:16] == ("10:00" if at_start else "16:05")
    assert len(picks_sent(season)) == 1 and snap["pick"]["automatic_reading"]["stale"] is True
    assert any("no source_date" in m["body"] or "undated" in m["body"] for m in season.messages)
    return {"messages": season.messages, "snapshot": snap, "next_morning": finish(season, day, 0)}


def collect(probe, *args):
    from bts.dm import send_dm

    with TemporaryDirectory(prefix=f"wr-{probe.__name__}-") as tmp:
        season = Season(Path(tmp))
        if probe == stale_warning and args[0] is True:
            season.clock.now = at("2027-03-23", "14:30")
        season.real_transport = send_dm
        season.old_seed()
        try:
            with season.run_offline(), redirect_stderr(season.journal):
                season.restore()
                return probe(season, *args)
        finally:
            suffix = f"-{args[0]}" if args else ""
            (HERE / f"{probe.__name__}{suffix}.journal.log").write_text(season.journal.getvalue())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect-fixed", action="store_true",
                        help="Assert delivery invariants rather than current failure signatures")
    args = parser.parse_args()
    results = {
        "late_correction": collect(correction_between_reads),
        "early_correction_control": collect(correction_between_reads, True),
        "transport_retry": collect(transport_retry),
        "crash_before_accept": collect(restart_at_send, "before_remote_accept"),
        "crash_after_accept": collect(restart_at_send, "after_remote_accept"),
        "crash_then_corrected": collect(restart_at_send, "corrected_after_marker"),
        "restart_after_checkpoint": collect(restart_at_send, "after_checkpoint"),
        "cached_preview": collect(preview_control),
        "corrected_cached_pick": collect(corrected_cached_pick),
        "masked_cutover": collect(cutover_with_masked_file),
        "saver_instructions": collect(ambiguous_saver_instructions, args.expect_fixed),
        "late_first_lineups": collect(first_lineups_after_last_check),
        "stale_at_start": collect(stale_warning, True),
        "stale_at_delivery": collect(stale_warning, False),
    }
    write_json(HERE / "opening_edges.json", results)
    late = results["late_correction"]["snapshot"]["pick"]
    control = results["early_correction_control"]["snapshot"]["pick"]
    assert control["double_down"] and control["chosen_on"]["streak"] == 1
    accepted = results["transport_retry"]["accepted_by_transport"]
    nags = [m for m in results["crash_then_corrected"]["messages"] if "Fix it now!" in m["body"]]
    if args.expect_fixed:
        assert late["double_down"] and late["chosen_on"]["streak"] == 1, late
        assert len(accepted) == 1, accepted
        assert not nags, nags
    else:
        assert late["double_down"] is None and late["chosen_on"]["streak"] == 46
        assert "Streak: 1" in results["late_correction"]["messages"][0]["body"]
        assert len(accepted) == 2 and accepted[0]["body"] == accepted[1]["body"]
        assert len(nags) == 1 and "Batter 27250 + DD Batter 27251" in nags[0]["body"]
    print(f"{len(results)} edge probes exercised")


if __name__ == "__main__":
    main()
