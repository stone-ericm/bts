"""Repro of blind B2: length of the live contest-state refusal DM as the scheduler would send it on the box."""
import json, sys, tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo
sys.path.insert(0, "/Users/eric/projects/bts-work/wr2-walk")
from scripts.audit.season_walk.walk import offline, _invoke, _set_clock
from bts.contest_state import ContestStateError, load_decision_streak_state
from bts.health import alert as alert_mod
from bts import scheduler
ET = ZoneInfo("America/New_York")
BOX = "/home/bts/projects/bts/data/picks"
out = {}
for case in ("absent", "revoked", "enabled-no-fetch"):
    with tempfile.TemporaryDirectory() as tmp:
        picks = Path(tmp) / "data" / "picks"; picks.mkdir(parents=True)
        with offline(datetime(2027, 3, 24, 8, 0, tzinfo=ET)):
            if case in ("revoked", "enabled-no-fetch"):
                print(_invoke("season-contract", "enable", "--season", 2027, "--picks-dir", picks)[0], file=sys.stderr)
            if case == "revoked":
                print(_invoke("season-contract", "revoke", "--season", 2027, "--reason", "restore", "--picks-dir", picks)[0], file=sys.stderr)
            now = datetime(2027, 3, 24, 16, 1, tzinfo=ET); _set_clock(now)
            try:
                load_decision_streak_state(picks, require_contest_state=True, now=now); err = None
            except ContestStateError as e:
                err = e
            sent = []
            cfg = {"bluesky": {"dm_recipient": "op.invalid"}, "orchestrator": {"picks_dir": str(picks)}}
            with patch.object(alert_mod, "send_dm", lambda r, t: sent.append(t) or "id1"):
                scheduler._alert_contest_state_failure(cfg, ContestStateError(str(err).replace(str(picks), BOX)))
                scheduler._alert_contest_state_failure(cfg, ContestStateError("relative " + str(err).replace(str(picks), "data/picks")))
        out[case] = {"box_path_dm_chars": len(sent[0]) if sent else None,
                     "relative_path_dm_chars": len(sent[1]) if len(sent) > 1 else "deduped (same source, same day)",
                     "head": sent[0][:140] if sent else None}
print(json.dumps(out, indent=1))
