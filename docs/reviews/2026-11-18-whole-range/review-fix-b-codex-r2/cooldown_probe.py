"""Exercise both real fetch lanes with independently seeded cooldown files."""
from datetime import timedelta
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from probe_support import NOW, REVIEW, REV, dump, fetch, offline, put, snapshot, switch
from bts.leaderboard import auth

NOW_UTC = NOW.isoformat()
STAMPS = {
    "inside": "2027-07-15T10:30:00-05:00",
    "outside": "2027-07-15T02:59:59-07:00",
    "six-exact": (NOW - timedelta(hours=6)).isoformat(),
    "year-edge": "9999-12-31T23:59:59-01:00",
    "naive": "2027-07-15T15:00:00",
    "trailing-Z": "2027-07-15T15:00:00Z",
    "colonless-offset": "2027-07-15T15:00:00+0000",
    "date-only": "2027-07-15",
    "garbage": "not a timestamp",
    "number": 42,
    "Z-separator-Z": "2027-07-15Z15:00:00Z",
    "Z-separator-offset": "2027-07-15Z15:00:00-01:00",
    "Z-separator-naive": "2027-07-15Z15:00:00",
    "future-canonical": "2027-07-16T16:00:00Z",
    "future-Z-separator": "2027-07-16Z16:00:00Z",
    "year-edge-Z-separator": "9999-12-31Z23:59:59-01:00",
}

rows = {}
with TemporaryDirectory(dir=REVIEW, prefix="cooldown-fixtures-") as directory:
    root = Path(directory)

    def case(label, prior, good, now=NOW):
        picks = root / label / "picks"
        put(picks, "season_contract.json", switch(now))
        state = picks.parent / "health_state/contest_streak_fetch_status.json"
        state.parent.mkdir()
        state.write_bytes(prior)
        before = snapshot(picks)
        with offline(now) as (send, profile):
            profile.return_value = dict(activeStreak=7, seasonBestStreak=9,
                                        predictions=[dict(roundId=1, result="hit")])
            if not good:
                with patch.object(auth, "fetch_login_session", side_effect=auth.AuthError("offline")):
                    result = fetch(picks)
            else:
                result = fetch(picks)
        data = json.loads(state.read_bytes())
        account = picks / "account_state"
        contents = dict(legacy=data.get("last_alert_at", "<absent>") if isinstance(data, dict) else None,
                        category=data.get("last_alert_at_by_category", "<absent>") if isinstance(data, dict) else None)
        changed = state.read_bytes() != prior
        if not good:
            assert before == snapshot(picks), label
        if REV == "head":
            assert result.exit_code == (0 if good else 2) and changed, (label, result.output)
            if good:
                assert (account / "contest_streak.json").is_file(), label
                assert (account / "contest_ledger.jsonl").is_file(), label
                assert json.loads((account / "saver_state.json").read_bytes())["state"] == "not_earned", label
        return dict(exit=result.exit_code, dm=send.call_count, changed=changed,
                    exception=type(result.exception).__name__ if result.exception else None,
                    exception_text=str(result.exception) if result.exit_code == 1 else None,
                    status_nonobject=data if not isinstance(data, dict) else "<object>", **contents)

    for tag, stamp in STAMPS.items():
        rows[tag] = {}
        for layout, seed in (("legacy", {"last_alert_at": stamp}),
                             ("category", {"last_alert_at_by_category": {"actionable": stamp}})):
            prior = json.dumps(seed).encode()
            rows[tag][layout] = {
                phase: case(f"{tag}-{layout}-{phase}", prior, good)
                for phase, good in (("failure", False), ("success", True))
            }
    for tag, prior in (("list", b"[]"), ("null", b"null")):
        rows[tag] = {phase: case(f"{tag}-{phase}", prior, good)
                     for phase, good in (("failure", False), ("success", True))}

    tag = "future-Z-separator"
    seed = json.dumps({"last_alert_at": STAMPS[tag]}).encode()
    for hours in (7, 30):
        rows[f"{tag}-after-{hours}h"] = case(f"{tag}-after-{hours}h", seed, False,
                                            now=NOW + timedelta(hours=hours))

dump("cooldown", rows)
