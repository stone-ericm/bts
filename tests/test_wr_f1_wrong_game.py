"""File-backed regression cases for the entry watchdog game check."""
from contextlib import ExitStack
from datetime import date, datetime, timezone
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from bts import entry_receipt, picks
from bts.cli import cli
from bts.daily_decision import write_decision
from bts.health import pick_entry

DAY = "2027-03-24"
XW = {1: 1010, 2: 1020, 3: 1030}
RIGHT = {"id": 77, "roundId": 1, "feedId": 101}
WRONG = {**RIGHT, "feedId": 102}


def row(unit=77, player=1, number=1, rnd=1):
    return {"roundId": rnd, "unitId": unit, "playerId": player, "number": number, "result": None}


class World:
    def __init__(self, root, day=DAY):
        self.root, self.day = root, day
        self.picks = root / "picks"
        self.picks.mkdir(parents=True, exist_ok=True)
        self.marker = root / "health_state" / "pick_entry_check.json"
        self.units = entry_receipt.units_snapshot_dir(self.picks)
        self.units.mkdir(parents=True, exist_ok=True)
        self.config = root / "config.toml"
        self.config.write_text('[scheduler]\nentry_intent = "enter"\npick_delivery = "dm"\n')
        self.runner = CliRunner()

    def commit(self, *, game=101, dd=None, pitch=None, delivery="delivered", name="Batter 1010"):
        pitch = pitch if pitch is not None else f"{self.day}T19:10:00-04:00"

        def slot(batter, game_pk, game_time):
            return picks.Pick(batter_name=f"Batter {batter}", batter_id=batter, team="NYY",
                              lineup_position=1, pitcher_name="Pitcher", pitcher_id=500,
                              p_game_hit=0.8, flags=[], projected_lineup=False, game_pk=game_pk,
                              game_time=game_time)

        primary = slot(1010, game, pitch)
        primary.batter_name = name
        second = slot(*dd) if dd else None
        daily = picks.DailyPick(self.day, f"{self.day}T16:00:00-04:00", primary, second, None,
                                notification_sent=True, notification_channel="bluesky_dm", notification_id="fixture")
        if delivery is None:
            daily.notification_sent = False
        picks.save_pick(daily, self.picks)
        if delivery is not None:
            decision = write_decision(self.day, self.picks, action="double" if dd else "single", source="mdp",
                                      delivery_status=delivery, scoreable=True,
                                      primary={"batter_name": primary.batter_name},
                                      double_down={"batter_name": second.batter_name} if second else None,
                                      now=datetime.fromisoformat(f"{self.day}T20:00:00+00:00"))
            assert decision is not None
        return daily

    def capture(self, units, stamp=None):
        stamp = stamp or f"{self.day}T210000Z"
        path = self.units / f"{stamp}.json.gz"
        path.write_bytes(gzip.compress(json.dumps({"units": units}).encode(), mtime=0))
        return path

    def read_marker(self):
        try:
            return json.loads(self.marker.read_text())
        except (OSError, ValueError):
            return None

    def set_marker(self, marker):
        self.marker.parent.mkdir(exist_ok=True)
        self.marker.write_text(json.dumps(marker))

    def check(self, at, pending=(), *, profile=None, crosswalk=None, rounds=None, dm="success", elapsed=0,
              more_patches=(), fetch_failure=None):
        messages, attempted, fetches = [], [], []

        def send(recipient, message):
            attempted.append(message)
            if dm == "failure":
                raise RuntimeError("synthetic send failure")
            messages.append(message)

        def fetch(name, data):
            def call(*args, **kwargs):
                fetches.append(name)
                if fetch_failure == name:
                    raise ValueError("synthetic fetch failure")
                return data
            return call

        def stable_mono():
            if not anchor:
                anchor.append(True)
                return 1000
            return 1000 + elapsed

        session = SimpleNamespace(user_id="fixture", username="operator", xsid="fixture")
        leaves = {
            "bts.leaderboard.auth.load_session_cookies": lambda: {},
            "bts.leaderboard.auth.extract_uid": lambda c: "fixture",
            "bts.leaderboard.auth.fetch_login_session": lambda *a, **k: session,
            "bts.contest_fetch.fetch_profile": fetch("profile", profile or {"predictions": []}),
            "bts.contest_fetch.fetch_pending_predictions": fetch("pending", list(pending)),
            "bts.cli._fetch_rounds": fetch("rounds", rounds if rounds is not None else {1: date.fromisoformat(self.day)}),
            "bts.cli._fetch_bts_to_mlb": fetch("crosswalk", crosswalk if crosswalk is not None else XW),
            "bts.dm.send_dm": send,
            "bts.picks._git_head_sha": lambda *a, **k: "fixture-fixed-head",
            "time.monotonic": stable_mono,
        }
        anchor = []
        args = ["check-pick-entered", "--picks-dir", str(self.picks), "--now-et", f"{self.day}T{at}:00",
                "--expected-username", "operator"]
        if dm != "none":
            args += ["--dm-recipient", "captured.invalid"]
        with ExitStack() as stack:
            for target, leaf in (*leaves.items(), *more_patches):
                stack.enter_context(patch(target, leaf))
            result = self.runner.invoke(cli, args)
        receipts = entry_receipt.discover(self.picks, self.day)
        return {"exit": result.exit_code, "exception": type(result.exception).__name__ if result.exception else None,
                "stdout": result.stdout, "stderr": result.stderr, "marker": self.read_marker(),
                "dm": messages, "attempted_dm": attempted, "fetches": fetches, "receipts": receipts}

    def morning(self):
        sent = []
        with patch("bts.scheduler.fetch_schedule", lambda _: [{"gamePk": 101}, {"gamePk": 102}]), \
             patch("bts.dm.send_dm", lambda _, msg: sent.append(msg)):
            result = self.runner.invoke(cli, ["day-outcome", "--date", self.day, "--config", str(self.config),
                                             "--picks-dir", str(self.picks), "--expected-username", "operator",
                                             "--dm-recipient", "captured.invalid"])
        assert result.exit_code in (0, 1), result.output
        return {"exit": result.exit_code, "outcome": json.loads(result.stdout), "dm": sent}

    def health(self):
        return [{"level": a.level, "message": a.message}
                for a in pick_entry.check(self.picks, date.fromisoformat(self.day),
                                          datetime(2030, 1, 1, tzinfo=timezone.utc))]


FIRST = ("⚠️ BTS entry game needs verification. {legs}. Check the delivered pick and the game shown in "
         "MLB for each listed leg. Change only a leg whose correct game you can verify and that MLB still "
         "lets you edit. Keep any correct leg as entered.")
REMINDER = ("⏰ BTS entry game verification is still needed. {legs}. "
            "Check the delivered pick against the game shown in MLB.")
LEG = "pick Batter 1010: recommended game 101 (first pitch 7:40 PM ET), the entry is recorded in game 102"
PITCH = f"{DAY}T19:40:00-04:00"
MATCH = [row()]
FETCHES = ["profile", "pending", "rounds", "crosswalk"]


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


@pytest.fixture
def wrong(world):
    world.commit(pitch=PITCH)
    world.capture([WRONG])
    return world


def assert_wrong(result, *, escalation, status="alerted"):
    assert result["exit"] == 1
    marker = result["marker"]
    assert (marker["status"], marker["reason"], marker["escalations"]) == (status, "wrong_game", escalation)
    receipt = next(item for item in result["receipts"] if item["attempt_id"] == marker["receipt"])
    assert receipt["verifier"]["ok"] is True
    assert receipt["verifier"]["reason"] == "match"
    assert receipt["verifier"]["game_qualified"] is False
    assert receipt["marker_status"] == status
    assert receipt["exit_code"] == 1


def test_m1_missing_then_wrong_game(wrong):
    assert wrong.check("18:30")["marker"]["escalations"] == ["initial"]
    result = wrong.check("18:45", MATCH)
    assert_wrong(result, escalation=["initial", "wrong_game"])
    assert result["dm"] == [FIRST.format(legs=LEG)]
    again = wrong.check("19:00", MATCH)
    assert_wrong(again, escalation=["initial", "wrong_game"])
    assert again["fetches"] == FETCHES
    assert again["dm"] == []
    assert again["stdout"] == (
        f"check-pick-entered: {DAY} game verification unresolved (pick); re-verifying each run until cutoff\n"
    )


def test_m2_first_observation_is_wrong_game(wrong):
    result = wrong.check("18:30", MATCH)
    assert_wrong(result, escalation=["wrong_game"])
    assert result["dm"] == [FIRST.format(legs=LEG)]


def test_m3_only_wrong_double_down_leg_is_named(world):
    world.commit(pitch=PITCH, dd=(1020, 201, PITCH))
    world.capture([RIGHT, {"id": 78, "feedId": 202, "roundId": 1}])
    result = world.check("18:30", [row(), row(78, 2, 2)])
    assert_wrong(result, escalation=["wrong_game"])
    assert result["dm"] == [FIRST.format(
        legs="double-down Batter 1020: recommended game 201 (first pitch 7:40 PM ET), "
             "the entry is recorded in game 202")]


def test_m4_reminders_share_cutoff_tiers(wrong):
    wrong.check("18:30", MATCH)
    for at, repeat, escalation in (("19:05", "19:06", ["t30", "wrong_game"]),
                                   ("19:20", "19:21", ["t15", "t30", "wrong_game"])):
        result = wrong.check(at, MATCH)
        assert_wrong(result, escalation=escalation)
        assert result["dm"] == [REMINDER.format(legs=LEG)]
        assert wrong.check(repeat, MATCH)["dm"] == []


@pytest.mark.parametrize("already_alerted", [False, True])
def test_m5_failed_dm_retries(wrong, already_alerted):
    prior = ["initial"] if already_alerted else []
    if already_alerted:
        wrong.check("18:30")
    result = wrong.check("18:45", MATCH, dm="failure")
    assert_wrong(result, escalation=prior, status="alerted" if already_alerted else "dm_failed")
    assert result["attempted_dm"] == [FIRST.format(legs=LEG)]
    assert result["dm"] == []
    assert result["stderr"] == "check-pick-entered: DM failed: synthetic send failure\n"
    retry = wrong.check("18:46", MATCH)
    assert_wrong(retry, escalation=[*prior, "wrong_game"])
    assert retry["dm"] == result["attempted_dm"]


def test_m6_locked_unconfirmed_text(wrong):
    from shlex import quote

    wrong.commit(pitch=PITCH, delivery="locked_unconfirmed")
    result = wrong.check("18:30", MATCH)
    assert_wrong(result, escalation=["wrong_game"])
    assert result["dm"] == [
        f"⚠️ Delivery of this pick was NOT confirmed. The scheduler tried to send Batter 1010. "
        f"MLB game evidence needs verification. {LEG}. Check your DMs and the MLB entry before making any change. "
        f"Run `bts season-contract status --picks-dir {quote(str(wrong.picks))}`."
    ]


def test_m7_cutoff_passes_during_fetch(wrong):
    result = wrong.check("19:34", MATCH, elapsed=61)
    assert_wrong(result, escalation=[])
    assert result["dm"] == []
    assert result["stdout"] == (
        "check-pick-entered: submission cutoff passed during this check; skipping the nag (entry can no longer change)\n"
    )
    assert result["marker"]["game_unresolved"] is True


def test_m8_removal_still_gets_initial_alert(wrong):
    assert wrong.check("18:30", MATCH)["dm"] == [FIRST.format(legs=LEG)]
    result = wrong.check("18:45")
    assert result["marker"]["escalations"] == ["initial", "wrong_game"]
    assert result["dm"] == [
        "⚠️ BTS pick NOT entered in MLB app: Batter 1010 — first pitch 7:40 PM ET (50 min to submit). Fix it now!"
    ]


def test_m9_lost_binding_does_not_clear_warning(wrong):
    wrong.check("18:30", MATCH)
    wrong.capture([RIGHT], f"{DAY}T220000Z")
    result = wrong.check("19:05", MATCH)
    assert_wrong(result, escalation=["t30", "wrong_game"])
    assert result["dm"] == [REMINDER.format(legs=LEG.split(", the entry")[0])]


@pytest.mark.parametrize("prior", [None, "no_pick", "wrong_game"])
def test_m10_raising_read_confirms_batter_only(wrong, prior):
    if prior == "no_pick":
        wrong.check("18:30")
    elif prior == "wrong_game":
        wrong.check("18:30", MATCH)
    wrong.capture([]).write_bytes(b"not gzip")
    result = wrong.check("18:45", MATCH)
    assert result["exit"] == 0
    assert result["marker"]["status"] == "confirmed"
    assert result["marker"]["reason"] == "match_game_unchecked"
    assert result["stdout"].endswith(
        f"check-pick-entered: {DAY} batter entered; the game could not be checked (match_game_unchecked)\n"
    )
    assert result["stderr"] == (
        "check-pick-entered: game check unavailable (BadGzipFile); batter match only\n"
        "check-pick-entered: entry receipt unavailable (BadGzipFile)\n"
    )
    assert result["dm"] == (
        [f"✅ BTS pick entry: the batter is entered for {DAY}; the game could not be checked."] if prior else []
    )
    assert len(result["receipts"]) == int(prior is not None)
    again = wrong.check("18:46", MATCH)
    assert again["stdout"] == f"check-pick-entered: already confirmed for {DAY}\n"
    assert again["fetches"] == []


def archive(value):
    return gzip.compress(json.dumps(value).encode())


@pytest.mark.parametrize(("payload", "error"), [
    (b"plain", "BadGzipFile"),
    (archive({"units": [RIGHT]})[:15], "EOFError"),
    (gzip.compress(b"{"), "JSONDecodeError"),
    (gzip.compress(b"\xff"), "UnicodeDecodeError"),
    (archive([]), "AttributeError"),
    (archive(None), "AttributeError"),
    (archive({"units": 1}), "TypeError"),
    (None, "IsADirectoryError"),
    (archive({}), None),
    (archive({"units": None}), None),
    (archive({"units": False}), None),
    (archive({"units": {"id": 77}}), None),
    (archive({"units": "x"}), None),
    (archive({"units": [False, None, 42, "x", [], {"id": "77"}]}), None),
    (archive({"units": [WRONG, None, {"id": "77", "feedId": 101, "roundId": 1}]}), "wrong_game"),
])
def test_capture_shapes_use_existing_reader(wrong, payload, error):
    path = wrong.capture([])
    if payload is None:
        path.unlink()
        path.mkdir()
    else:
        path.write_bytes(payload)
    result = wrong.check("18:30", MATCH)
    if error == "wrong_game":
        assert_wrong(result, escalation=["wrong_game"])
    elif error:
        assert result["exit"] == 0
        assert result["marker"]["reason"] == "match_game_unchecked"
        assert result["stderr"] == (
            f"check-pick-entered: game check unavailable ({error}); batter match only\n"
            f"check-pick-entered: entry receipt unavailable ({error})\n"
        )
        assert not result["receipts"]
    else:
        assert result["exit"] == 0
        assert result["marker"]["reason"] == "match"
        assert result["stderr"] == ""
        assert result["receipts"][0]["qualification"]["slots"][0]["state"] == "unit_unverified"


@pytest.mark.parametrize("fault", [KeyboardInterrupt, SystemExit])
def test_interrupt_during_game_read(wrong, fault):
    original = entry_receipt.load_units
    calls = []

    def interrupt_once(*args):
        calls.append(args)
        if len(calls) == 1:
            raise fault(17)
        return original(*args)

    result = wrong.check("18:30", MATCH, more_patches=[("bts.entry_receipt.load_units", interrupt_once)])
    assert len(calls) == 2
    assert result["marker"] is None
    assert result["attempted_dm"] == []
    receipt, = result["receipts"]
    if fault is KeyboardInterrupt:
        assert receipt["detail"] == "raised KeyboardInterrupt"
        assert receipt["exit_code"] is None
    else:
        assert receipt["exit_code"] == 17


def test_unreadable_capture_uses_game_unchecked_outcome(wrong):
    path = wrong.capture([WRONG])
    original = Path.read_bytes

    def deny_capture(candidate):
        if candidate == path:
            raise PermissionError
        return original(candidate)

    result = wrong.check("18:30", MATCH, more_patches=[("pathlib.Path.read_bytes", deny_capture)])
    assert result["exit"] == 0
    assert result["marker"]["reason"] == "match_game_unchecked"
    assert result["stderr"] == (
        "check-pick-entered: game check unavailable (PermissionError); batter match only\n"
        "check-pick-entered: entry receipt unavailable (PermissionError)\n"
    )
    assert not result["receipts"]


def test_duplicate_batter_skips_new_read_but_keeps_existing_warning(wrong):
    wrong.check("18:30", MATCH)
    wrong.commit(pitch=PITCH, dd=(1010, 202, PITCH))
    original = entry_receipt.load_units
    calls = []

    def observe(*args):
        calls.append(args)
        return original(*args)

    result = wrong.check("19:05", MATCH, more_patches=[("bts.entry_receipt.load_units", observe)])
    assert len(calls) == 1  # Publication only.
    assert_wrong(result, escalation=["t30", "wrong_game"])
    assert result["dm"] == [REMINDER.format(
        legs="pick Batter 1010: recommended game 101 (first pitch 7:40 PM ET); "
             "double-down Batter 1010: recommended game 202 (first pitch 7:40 PM ET)")]


@pytest.mark.parametrize("match", [False, True])
def test_fifo_capture_blocks_before_match_marker_only(wrong, match):
    path = wrong.capture([])
    path.unlink()
    os.mkfifo(path)
    child_code = (
        "import runpy, sys; from pathlib import Path; "
        "ns = runpy.run_path(sys.argv[1]); world = ns['World'](Path(sys.argv[2])); "
        "world.check('18:30', ns['MATCH'] if sys.argv[3] == 'match' else ())"
    )
    with pytest.raises(subprocess.TimeoutExpired):
        subprocess.run([sys.executable, "-c", child_code, __file__, str(wrong.root),
                        "match" if match else "no_pick"], timeout=20, capture_output=True, check=True)
    marker = wrong.read_marker()
    if match:
        assert marker is None
    else:
        assert (marker["status"], marker["reason"]) == ("alerted", "no_pick")


def test_interrupted_marker_write_leaves_tier_retryable(wrong):
    wrong.check("18:30")
    old_marker = wrong.read_marker()
    accepted = []

    def interrupt(*args):
        raise KeyboardInterrupt

    result = wrong.check("18:45", MATCH, more_patches=[
        ("bts.dm.send_dm", lambda recipient, message: accepted.append(message)),
        ("bts.cli._atomic_write_json", interrupt),
    ])
    assert result["marker"] == old_marker
    assert accepted == [FIRST.format(legs=LEG)]
    assert wrong.check("18:46", MATCH)["dm"] == accepted


def test_midnight_observations_use_start_date_and_year(tmp_path):
    first = World(tmp_path, "2027-12-31")
    second = World(tmp_path, "2028-01-01")
    for world in (first, second):
        world.commit(pitch="2028-01-01T00:50:00-05:00")
    first.capture([WRONG], "20271231T235900Z")
    second.capture([{**RIGHT, "roundId": 2}], "20280101T045000Z")
    rounds = {1: date.fromisoformat(first.day), 2: date.fromisoformat(second.day)}
    prior = first.check("23:59", MATCH, rounds=rounds, elapsed=120)
    assert_wrong(prior, escalation=["wrong_game"])
    assert prior["receipts"][0]["season"] == 2027
    result = second.check("00:05", [*MATCH, row(rnd=2)], rounds=rounds)
    assert result["marker"]["date"] == second.day
    assert result["marker"]["status"] == "confirmed"
    assert result["receipts"][0]["qualification"]["all_confirmed"]


def test_n10_full_proof_clears_warning(wrong):
    wrong.check("18:30", MATCH)
    wrong.capture([RIGHT])
    result = wrong.check("18:45", MATCH)
    assert result["exit"] == 0
    assert result["marker"]["status"] == "confirmed"
    assert result["marker"]["escalations"] == ["wrong_game"]
    assert result["dm"] == [f"✅ BTS pick entry confirmed for {DAY} (match)."]
    assert wrong.health() == []


def test_n11_unresolved_wrong_game_health_and_morning(wrong):
    wrong.check("18:30", MATCH)
    health, = wrong.health()
    assert health["level"] == "WARN"
    assert "last status=alerted, reason=wrong_game" in health["message"]
    assert wrong.morning()["outcome"]["state"] == "entry_unverifiable"


@pytest.mark.parametrize("name", [None, ""])
def test_n13_missing_name_uses_batter_id(wrong, name):
    wrong.commit(pitch=PITCH, name=name)
    result = wrong.check("18:30", MATCH)
    assert_wrong(result, escalation=["wrong_game"])
    assert result["dm"] == [FIRST.format(legs=LEG.replace("Batter 1010", "batter 1010"))]


def comparison_case(world, case):
    if case == "N7-no-file":
        return [world.check("18:30")]
    dd = (1020, 201, PITCH) if case in {"N2-missing-leg", "N9", "N12-repeat"} else None
    if case == "N12-repeat":
        dd = (1010, 202, PITCH)
    delivery = "locked_unconfirmed" if case.startswith("N8") else None if case == "N7-preview" else "delivered"
    world.commit(game=None if case == "N12-null" else 101, pitch=PITCH, dd=dd, delivery=delivery)
    if case in {"N3", "N9", "N11"}:
        world.capture([RIGHT, {"id": 78, "feedId": 201, "roundId": 1}])
    elif case == "N4-absent-unit":
        world.capture([{**RIGHT, "id": 79}])
    elif case == "N4-null-feed":
        world.capture([{**RIGHT, "feedId": None}])
    elif case == "N4-wrong-round":
        world.capture([{**WRONG, "roundId": 2}])
    elif case in {"N4-ambiguous", "N12-null", "N12-repeat"}:
        world.capture([WRONG])
    if case.startswith("N1-"):
        prior = case.removeprefix("N1-")
        prior_status = {"legacy": "alerted", "stale": "alerted", "dm_failed": "dm_failed",
                        "present": "present_unverified", "alerted": "alerted"}
        if prior in prior_status:
            world.set_marker({
                "date": "2027-03-23" if prior == "stale" else DAY,
                "status": prior_status[prior],
                "reason": "no_pick",
                **({"escalations": ["initial", "t30"]} if prior == "alerted" else {}),
            })
        return [world.check(at, dm=dm, elapsed=elapsed) for at, dm, elapsed in (
            ("18:30", "failure", 0), ("18:45", "success", 0), ("19:00", "success", 0),
            ("19:05", "success", 0), ("19:20", "success", 0), ("19:34", "success", 61),
        )]
    if case in {"N2-wrong-batter", "N8-mismatch"}:
        return [world.check("18:30", [row(player=2)])]
    if case == "N8-no-pick":
        return [world.check("18:30")]
    if case == "N3":
        return [world.check("18:30"), world.check("18:45", MATCH), world.check("19:00", MATCH)]
    if case == "N4-ambiguous":
        return [world.check("18:30", [row(), row(78, number=2)])]
    if case == "N5":
        return [world.check("18:30", MATCH, crosswalk={}), world.check("18:45", MATCH, crosswalk={})]
    if case == "N6-identity":
        session = SimpleNamespace(user_id="fixture", username="other", xsid="fixture")
        return [world.check("18:30", MATCH, more_patches=[
            ("bts.leaderboard.auth.fetch_login_session", lambda *a, **k: session),
        ])]
    if case.startswith("N6-"):
        return [world.check("18:30", MATCH, fetch_failure=case.removeprefix("N6-"))]
    if case == "N7-outside":
        return [world.check("12:00", MATCH)]
    if case == "N7-confirmed":
        world.set_marker({"date": DAY, "status": "confirmed", "reason": "match"})
    rows = [row(), row(78, 2, 2)] if case == "N9" else MATCH
    runs = [world.check("18:30", rows)]
    if case == "N11":
        runs.extend([world.morning(), world.health()])
    return runs


def normalized(value, picks):
    if isinstance(value, dict):
        return {key: normalized(item, picks) for key, item in value.items()
                if key not in {"receipt", "attempt_id", "confirmed_by", "checked_at", "started_at", "published_at",
                               "response_completed_at", "failed_at", "cutoff_at"}}
    if isinstance(value, list):
        return [normalized(item, picks) for item in value]
    if isinstance(value, str):
        return value.replace(str(picks), "<picks>")
    return value


# SHA-256 of the normalized watchdog, day-outcome and health observations on base 006fafef.
BASE_COMPARISON = {
    "N1-none": "01773cb95f6cd91c22fb970afbc3cd9b33d387d0b55be7876e299bebe9cfc4b8",
    "N1-legacy": "e31b3a7706195e4147441091a024d5a2af52856e9a62702b83078596074daf26",
    "N1-stale": "01773cb95f6cd91c22fb970afbc3cd9b33d387d0b55be7876e299bebe9cfc4b8",
    "N1-dm_failed": "01773cb95f6cd91c22fb970afbc3cd9b33d387d0b55be7876e299bebe9cfc4b8",
    "N1-present": "01773cb95f6cd91c22fb970afbc3cd9b33d387d0b55be7876e299bebe9cfc4b8",
    "N1-alerted": "002d0f95ac067595fb38762ef527410fc534968374396159e78f93ca4c27478b",
    "N2-wrong-batter": "53eecb1a6566cdee0dd3bec6090efb96ed02ceb652aabc1ec717c206ce6092b9",
    "N2-missing-leg": "e76f593c5af90173b17d3130f67af85782109996f0f2ac3bfe05089fe5c86e6d",
    "N3": "99223cb72cc2d137e6889cba0c05787cb257d57e5ab04bf19b67bf1fbe2ac3dc",
    "N4-no-capture": "e3d8134fef9c65474684f391bf49c67eb8a40d299861914e451bd207f887df76",
    "N4-absent-unit": "41d49c69cccf60ed9a17fb1d2cf1af89b406cab02388f7e27e53e03776920423",
    "N4-null-feed": "cf4ffa81cb3e6fd00520d55e5de2038cb41385c11a1e646926eea30753a2a936",
    "N4-wrong-round": "9f07d323298d430c7281d1ae31decc5d5287f1f3a5edfade3c0dc0b9c23424b8",
    "N4-ambiguous": "cb32d39c5b3ec89a1512f40ab5c81943762ec895e6cd3067395a2b3f4c9c9123",
    "N5": "d9d82d1268de78bafd4e7c4851e2fd6c46d7a5fa272518c38ace9599d3bbacee",
    "N6-profile": "bda001d84e28d106874456291ead2ea683e108577d215451c78ce443c8a62c8d",
    "N6-pending": "a776df3c655974d2531aac0867497a9ba50b4cfdfa91e486fde6268fe31fe378",
    "N6-rounds": "e84c570b27e599170fcaf137deedf5e262f19d5d0582af210a98187d0b7c7ffb",
    "N6-crosswalk": "f9f834ef9e2dd618a0b06d2a864ebcb5fdcc7391efe931d42fc6568c1062a404",
    "N6-identity": "ef822bc52b9209032b41d19f2e667db12c4decd482488193ec725e5e6ef2eeb2",
    "N7-no-file": "e690ccd314e5a987d119b5e503ebbd92801bcac4c854cf657c0a6cf1397031cf",
    "N7-preview": "1df781f7987203825836ca84287ede5a7febef52c4a63063add31023b90a9bcf",
    "N7-outside": "e08b4ebae88d650d206e43f7cff26a5ab6eb393c76ef8718fcd2929e48f0e76c",
    "N7-confirmed": "4d78d6bf8a7a9b6acb1fe5fcdb5aeb75fc38268a3f1482b8503002573d77dafe",
    "N8-no-pick": "441024b6973be5d3dc016911dff97823549d4ce36d3b92c36098c861ebf379bd",
    "N8-mismatch": "0db0fc2da4c07602510acf9f47fb413e7220d68c1981b6cc636af8d96da441c6",
    "N9": "41206e1c153c89a909fac996dcd8c36bc769a0b70ecb3a72565c974f87ce14de",
    "N11": "db2500e5089685ebaf952de6fc2031052eac3e01a7d77ca4a26f2cbeee746ebc",
    "N12-null": "37b8cc74c0b846915db0ad954bbbc2d2d5a4fdebd6f4e6284aa71ad4b32c7522",
    "N12-repeat": "c2f15489b6b3ea75969fe6617fe5d142ca1917fb8aff1c9c52c493686c2446d2",
}

@pytest.mark.parametrize("case", BASE_COMPARISON)
def test_unchanged_comparison_grid(world, case):
    from hashlib import sha256

    runs = comparison_case(world, case)
    morning = world.morning()
    morning_reports = [runs[1], morning] if case == "N11" else [morning]
    for report in morning_reports:
        assert report["outcome"].pop("streak_problem") is None
    actual = normalized([runs, [morning, world.health()]], world.picks)
    digest = sha256(json.dumps(actual, sort_keys=True).encode()).hexdigest()
    assert digest == BASE_COMPARISON[case], json.dumps(actual, indent=2, sort_keys=True)
