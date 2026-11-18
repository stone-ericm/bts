"""Run the uncovered preview case against an archived source tree and HEAD."""

import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

from offline import forbid_network

REVIEW = Path(__file__).resolve().parent
ROOT = REVIEW.parent.parent
base = sys.argv[1] == "base"
sys.path[:0] = [str(REVIEW / "base" / "src" if base else ROOT / "src"), str(ROOT)]
forbid_network()

from pytest import MonkeyPatch

from bts import contest_state, dm, orchestrator, picks, posting, scheduler
from bts.daily_decision import load_decision
from bts.health import alert
from tests.test_pick_chosen_on_cs5a import DAY, account_file, set_automatic, set_dated
from tests.test_saved_pick_paths_cs5a import invoke, scheduler_state, world


def run(channel, root, monkeypatch):
    posting_window = posting.should_post_now
    config, directory, _, statuses = world.__wrapped__(root, monkeypatch)
    monkeypatch.setattr(posting, "should_post_now", posting_window)
    monkeypatch.setattr(posting, "_now_et", scheduler._now_et)
    set_automatic(directory, streak=0, covered=False)
    auto = directory / "account_state" / "contest_streak.json"
    automatic = json.loads(auto.read_text())
    automatic["source"] = "mlb_bts_profile"
    auto.write_text(json.dumps(automatic))
    account_file(directory, "saver_state.json", {"season": 2027, "state": "not_earned"})
    preview_state = contest_state.load_decision_streak_state(directory)
    assert (preview_state.streak, preview_state.saver_available) == (0, True)
    assert invoke("preview", config, root).exit_code == 0
    preview = picks.load_pick(DAY, directory)
    assert preview.double_down is not None and preview.pick.batter_id != preview.double_down.batter_id
    if not base:
        assert preview.chosen_on["kind"] == "automatic" and preview.chosen_on["streak"] == 0
        assert preview.chosen_on["required"] is False and preview.chosen_on["saver_available"] is True
    set_dated(directory, streak=56, best=56, saver="used")
    state = contest_state.load_decision_streak_state(directory, require_contest_state=True)
    assert (state.streak, state.saver_available) == (56, False)
    pick_file = directory / f"{DAY}.json"
    before = pick_file.read_bytes()
    if channel == "scheduler":
        loop_state = scheduler_state()
        state_path = scheduler.save_state(loop_state, directory)
        state_before = state_path.read_bytes()
        frame, tier = orchestrator.run_cascade.return_value
        orchestrator.run_cascade.side_effect = RuntimeError("captured prediction failure")
        refresh = scheduler._refresh_pick_at_fallback_decision(config, DAY, preview, 0.03)
        assert refresh.daily is preview and refresh.selection is None
        trace = io.StringIO()
        with contextlib.redirect_stderr(trace):
            delivered = scheduler._deliver_and_lock_pick(
                refresh.daily, config, directory, loop_state, DAY, "fallback", selection=refresh.selection)
        assert (delivered, loop_state.pick_locked, dm.send_dm.called) == (base, base, base)
        if base:
            sent = dm.send_dm.call_args.args[1]
            assert "Streak: 56" in sent and "Steven Kwan" in sent and "Juan Soto" in sent
            assert load_decision(DAY, directory)["delivery_status"] == "delivered"
        else:
            assert state_path.read_bytes() == state_before and pick_file.read_bytes() == before
            assert load_decision(DAY, directory) is None
            assert alert.send_dm.call_count == 1 and not posting.post_to_bluesky.called
            message = alert.send_dm.call_args.args[1]
            assert "chosen without the live check" in message and "active streak 56" in message
            assert "pick delivery blocked" in trace.getvalue()
            orchestrator.run_cascade.side_effect = None
            orchestrator.run_cascade.return_value = (frame.assign(p_game_hit=[0.99, 0.81]), tier)
            _, healthy_at_56, _ = orchestrator.run_and_pick(config, DAY)
            assert healthy_at_56.decision.action == "skip" and healthy_at_56.pick_result is None
            # Use another dated reading where the bundled policy can choose a single.
            set_dated(directory, streak=30, best=30, saver="used")
            refresh = scheduler._refresh_pick_at_fallback_decision(config, DAY, preview, 0.03)
            assert refresh.selection.decision is not None and refresh.daily.double_down is None
            assert scheduler._deliver_and_lock_pick(
                refresh.daily, config, directory, loop_state, DAY, "fallback", selection=refresh.selection)
            assert dm.send_dm.call_count == 1 and "Streak: 30" in dm.send_dm.call_args.args[1]
            assert picks.load_pick(DAY, directory).notification_sent and loop_state.pick_locked
            assert load_decision(DAY, directory)["delivery_status"] == "delivered"
        print(channel, "first send:", delivered, "recovery checked:", not base)
        return

    # Unlike scheduler fallback, these commands exit on a pipeline error.
    # Warmup leaves the preview enterable while excluding it from new choices.
    from bts.model import predict

    flaky_transport = predict.run_pipeline if channel == "run" else orchestrator.run_cascade
    flaky_transport.side_effect = RuntimeError("captured prediction failure") if channel == "run" else None
    routine_frame, tier = orchestrator.run_cascade.return_value
    if channel == "orchestrate":
        flaky_transport.return_value = (None, None)
    invoke(channel, config, root)
    assert not posting.post_to_bluesky.called and pick_file.read_bytes() == before
    flaky_transport.side_effect = None
    if channel == "orchestrate":
        flaky_transport.return_value = (routine_frame, tier)
    statuses.update(dict.fromkeys(statuses, {"abstract": "P", "detailed": "Warmup", "code": "PW"}))
    result = invoke(channel, config, root)
    assert result.exit_code == (0 if base else 1), result.output
    assert posting.post_to_bluesky.call_count == (1 if base else 0)
    cached = picks.load_pick(DAY, directory)
    assert cached.bluesky_posted is base and not cached.delivery_attempted
    assert load_decision(DAY, directory) is None
    if base:
        text = posting.post_to_bluesky.call_args.args[0]
        assert "Streak: 56" in text and "Steven Kwan" in text and "Juan Soto" in text
    else:
        assert cached.chosen_on == preview.chosen_on and not cached.bluesky_uri
        assert f"bts {channel}: no pick: Saved pick for {DAY} was not delivered" in result.stderr
        set_dated(directory, streak=30, best=30, saver="used")
        statuses.update(dict.fromkeys(statuses, {"abstract": "P", "detailed": "Scheduled", "code": "S"}))
        predict.run_pipeline.return_value = routine_frame.assign(p_game_hit=[0.99, 0.81])
        orchestrator.run_cascade.return_value = (predict.run_pipeline.return_value, tier)
        result = invoke(channel, config, root)
        assert result.exit_code == 0, result.output
        assert posting.post_to_bluesky.call_count == 1
        chosen = picks.load_pick(DAY, directory)
        assert chosen.bluesky_posted and not chosen.double_down
        assert chosen.chosen_on["required"] and chosen.chosen_on["streak"] == 30
    print(channel, "reused double posted:", base, "recovery checked:", not base)


if __name__ == "__main__":
    assert str(picks.__file__).startswith(str(REVIEW / "base" if base else ROOT / "src"))
    for channel in ("scheduler", "run", "orchestrate"):
        with tempfile.TemporaryDirectory(dir=REVIEW) as temp, MonkeyPatch.context() as patch:
            run(channel, Path(temp), patch)
