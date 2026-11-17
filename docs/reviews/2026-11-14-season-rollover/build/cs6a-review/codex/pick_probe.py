"""Compare the three bts run read sites using real state files and captured I/O."""

import local_guard
from contextlib import ExitStack
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo
import json
import sys

from click.testing import CliRunner

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REV = sys.argv[1]
SRC = {"base": HERE / "base", "repaired": HERE / "testbed"}.get(REV, ROOT) / "src"
sys.path[:0] = [str(SRC), str(ROOT)]
from bts import cli, contest_state, picks, strategy
from bts.model import predict
from tests.test_cli_integration import _mock_predictions, _sample_daily, _sample_pick
from tests.test_season_contract_cs4b2 import dated

DAY = "2027-07-15"
ET = ZoneInfo("America/New_York")
NOW = datetime(2027, 7, 15, 12, tzinfo=ET)
ID = "a" * 32
STATES = ("none", "auto_v1", "legacy", "dated", "covered", "corrupt")
PATHS = ("skip", "locked", "already-posted", "single", "double", "empty", "pipeline-error",
         "unexpected-pipeline-error", "dry-run", "delayed-single", "no-valid-picks")


class Clock(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW.replace(tzinfo=None) if tz is None else NOW.astimezone(tz)


def put(path, value):
    path.write_text(json.dumps(value))


def state_on_disk(root, variant):
    account = root / "account_state"
    account.mkdir(parents=True, exist_ok=True)
    automatic = account / "contest_streak.json"
    switch = account / "season_contract.json"
    if variant == "none":
        return
    if variant == "legacy":
        put(account / "contest_streak.manual.json", dict(
            active_streak=5, source="manual_screenshot", source_date=DAY,
            recorded_at=NOW.isoformat(), override_expires_at="2027-07-16T13:00:00Z"))
    elif variant == "dated":
        put(account / "contest_streak.manual.json", dated(active_streak=7, best_streak=9))
    elif variant == "auto_v1":
        put(automatic, dict(schema_version="bts_contest_streak_auto_v1", active_streak=7,
                            source="mlb_bts_profile", source_date="2026-07-15",
                            recorded_at="2026-07-15T16:00:00Z"))
    else:
        put(automatic, dict(schema_version="bts_contest_streak_auto_v2", active_streak=7,
                            best_streak=9, source="mlb_bts_profile", source_date=DAY,
                            recorded_at=NOW.isoformat(), season=2027, activation_id=ID))
        if variant == "corrupt":
            switch.write_text("{corrupt")
        else:
            put(switch, dict(schema_version="bts_season_contract_v1", season=2027, status="automatic",
                             activation_id=ID, switched_at=NOW.isoformat(), revoked_at=None, note=None))


def tree(root):
    return {str(p.relative_to(root)): p.read_bytes().hex()
            for p in sorted(root.rglob("*")) if p.is_file()}


def pick_run(root, path, state, *, revoke_at_read=False):
    root.mkdir(parents=True)
    state_on_disk(root, state)
    events = []
    daily = _sample_daily(date=DAY, run_time=NOW.astimezone(timezone.utc).isoformat(),
                          bluesky_posted=path == "already-posted",
                          double_down=_sample_pick(batter_name="Second") if path == "double" else None)
    if path in ("locked", "already-posted"):
        with patch.object(picks, "datetime", Clock):
            picks.save_pick(daily, root)
    frame = _mock_predictions()
    if path == "no-valid-picks":
        frame.loc[:, "p_game_hit"] = float("nan")
    load_real = contest_state.load_decision_streak_state
    save_real = picks.save_pick
    initial = tree(root)
    seen = 0

    def load(directory, **kwargs):
        nonlocal seen
        seen += 1
        events.append(("read", kwargs.get("require_contest_state", False)))
        if revoke_at_read and seen == 2:
            (directory / "account_state" / "season_contract.json").unlink()
        return load_real(directory, **kwargs)

    def predict_transport(*args, **kwargs):
        events.append(("prediction", args[0]))
        if path == "pipeline-error":
            raise RuntimeError("captured pipeline error")
        if path == "unexpected-pipeline-error":
            raise ValueError("captured value error")
        return frame.iloc[:0] if path == "empty" else frame

    def schedule_transport(date):
        events.append(("schedule", date))
        return {}

    def select(*args, **kwargs):
        events.append(("select", kwargs["streak"], kwargs["saver_available"]))
        if path in ("skip", "no-valid-picks"):
            return SimpleNamespace(pick_result=None, decision=None)
        locked = path in ("locked", "already-posted")
        chosen = picks.load_pick(DAY, root) if locked else daily
        return SimpleNamespace(pick_result=strategy.PickResult(chosen, locked=locked), decision=None)

    def post(text):
        events.append(("post", text))
        return "at://capture/pick"

    def save(chosen, directory):
        events.append(("save", chosen.bluesky_posted))
        return save_real(chosen, directory)

    with ExitStack() as ctx:
        for name, stub in {
            "datetime.datetime": Clock, "bts.contest_state.datetime": Clock, "bts.picks.datetime": Clock,
            "bts.contest_state.load_decision_streak_state": load, "bts.model.predict.run_pipeline": predict_transport,
            "bts.picks.get_game_statuses_detailed": schedule_transport, "bts.strategy.select_pick": select,
            "bts.posting.post_to_bluesky": post, "bts.picks.save_pick": save,
            "bts.posting.should_post_now": lambda *_: path != "delayed-single",
            "bts.dm.send_dm": lambda *args: events.append(("dm", *args)),
            "bts.picks.attach_provenance": lambda *_args, **_kwargs: None,
        }.items():
            ctx.enter_context(patch(name, stub))
        args = ["run", "--date", DAY, "--picks-dir", str(root), "--models-dir", str(root / "models")]
        if path == "dry-run":
            args.append("--dry-run")
        result = CliRunner().invoke(cli.cli, args)
    data = {key: value for key, value in tree(root).items() if initial.get(key) != value}
    return dict(exit=result.exit_code, stdout=result.stdout.replace(str(root), "<picks>"),
                stderr=result.stderr.replace(str(root), "<picks>"),
                exception=type(result.exception).__name__, events=events, files=data)


def main():
    with TemporaryDirectory(dir=HERE) as folder:
        root = Path(folder)
        outcomes = {}
        for path, state in product(PATHS, STATES):
            outcomes[f"{path}/{state}"] = pick_run(root / path / state, path, state)
        for path in ("locked", "single", "double"):
            outcomes[f"{path}/mid-run-revoke"] = pick_run(root / path / "revoke", path, "covered",
                                                         revoke_at_read=True)
    assert not local_guard.network, local_guard.network
    (HERE / f"pick-{REV}.json").write_text(json.dumps(outcomes, indent=2) + "\n")
    for key, value in outcomes.items():
        print(key, value["exit"], [e[0] for e in value["events"]], list(value["files"]),
              value["stderr"].strip())


if __name__ == "__main__":
    main()
