"""Exercise composed E2/E3 and two real checks sharing one dashboard state."""
import re
import socket
import sys
import tempfile
from datetime import timedelta
from pathlib import Path

from fixtures import HERE, REVIEWS, fixture
from pytest import MonkeyPatch
from tests.test_saved_pick_paths_cs5a import prepare_run_day

ops = fixture("probe_operators")
archived = fixture("probe_text", "contract-check-e")
ops.CONTRACT = REVIEWS / "kickoff-fix-e.md"


def expected(label, **values):
    lines = ops.CONTRACT.read_text().splitlines()
    template, = [line.split("`")[1] for line in lines if line.startswith(f"- **{label}:** `")]
    if label == "In the morning":
        amendment = next(line for line in lines if "**Codex B1 (the morning warning).**" in line)
        old, new = re.findall(r"`([^`]+)`", amendment)[-2:]
        assert template.count(old) == 1
        template = template.replace(old, new)
    return template.format(**values)


def banner(root, mp):
    config, directory, clock, _ = ops.world.__wrapped__(root, mp)
    config["scheduler"]["entry_intent"] = "enter"
    ops.set_automatic(directory, streak=15)
    frame, tier = ops.orchestrator.run_cascade.return_value
    ops.orchestrator.run_cascade.side_effect = [(frame.assign(p_game_hit=[.72, .71]), tier), (None, None)]
    prepare_run_day(mp)
    mp.setattr(ops.scheduler, "notify_ready", lambda: None)
    mp.setattr(ops.scheduler, "notify_watchdog", lambda: None)
    mp.setattr(ops.scheduler, "_watchdog_ping_sleep",
               lambda seconds: clock.__setitem__(0, clock[0] + timedelta(seconds=seconds + 1)))
    ops.scheduler.run_day(date=ops.DAY, config=config)
    state = ops.scheduler.load_state(ops.DAY, directory)
    assert len(state.runs_completed) == ops.orchestrator.run_cascade.call_count == 2
    assert state.skip_summary["best_batter"] == "Steven Kwan" and state.skip_summary["candidate_verified"]
    mp.setattr(ops.web, "PICKS_DIR", directory)
    mp.setattr(ops.web, "datetime", ops.contest_state.datetime)
    mp.setattr(ops.web, "fetch_bluesky_posts", lambda *a, **k: [])
    page = ops.web.render_page()
    match = re.search(r'<div class="skip-banner">.*?</div>', page)
    assert match and "NO PICK AT LAST CHECK" in match[0] and "Candidate used in the decision: Steven Kwan" in match[0]
    print("Game-day checks:", state.runs_completed)
    print("Second cascade returned no predictions. Page:", match[0])


ops.exact_body = expected
with tempfile.TemporaryDirectory(dir=HERE) as name, MonkeyPatch.context() as mp:
    mp.setattr(socket.socket, "connect", archived.offline)
    mp.setattr(socket, "create_connection", archived.offline)
    mp.setattr(socket, "getaddrinfo", archived.offline)
    action = {"warnings": ops.warnings_and_refusals, "banner": banner}[sys.argv[1]]
    action(Path(name), mp)
