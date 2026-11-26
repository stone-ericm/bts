"""Independent assertions over the real day loop using the reviewed HTTP/clock fixture."""
import io
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import pytest
import bts.scheduler as sch
import tests.test_scheduler_last_attempt as harness
from tests.conftest import sample_game_feed
from tests.test_cold_start_integration import DATE, _prepare_history

HERE = Path(__file__).resolve().parent
sample = sample_game_feed.__wrapped__()
out = {}

with tempfile.TemporaryDirectory(dir=HERE) as directory:
    root = Path(directory)
    history = root / "history"
    _prepare_history(history, sample)

    def run(name, *, undo_postponement=False, doubleheader=False, **kwargs):
        work = root / name
        work.mkdir()
        with pytest.MonkeyPatch.context() as mp:
            # Keep provenance real. Only environment setup/cache reset is shared.
            def isolate(patches):
                patches.setenv("LOKY_MAX_CPU_COUNT", "1")
                patches.setenv("BTS_USE_CALIBRATION", "0")
                patches.delenv("BTS_REFRESH_ALWAYS", raising=False)
                patches.setattr("bts.strategy._mdp_cache", None)
            mp.setattr(harness, "_isolate", isolate)
            real_fake_http = harness._fake_http
            def install_http(patches, network):
                def wrapped(url, **kw):
                    response = network(url, **kw)
                    target = getattr(url, "full_url", url)
                    if "/schedule" not in target:
                        return response
                    body = json.loads(response.read())
                    for entry in body.get("dates", []):
                        games = entry.get("games", [])
                        if undo_postponement and sch._now_et().strftime("%H:%M") < "18:34":
                            for game in games:
                                game["status"].update(abstractGameCode="F", detailedState="Postponed")
                        if doubleheader and len(games) > 1:
                            games[1]["teams"] = games[0]["teams"]
                    return io.BytesIO(json.dumps(body).encode())
                real_fake_http(patches, wrapped)
            mp.setattr(harness, "_fake_http", install_http)
            drive = harness.drive.__wrapped__(work, mp, sample, history)
            day = drive(**kwargs)
            entry = {"runs": [r["time"][11:16] for r in day.state.runs_completed],
                     "locked": day.state.pick_locked, "decision": day.decision,
                     "cascades": [t.strftime("%H:%M") for t in day.cascades],
                     "sent": [(t.isoformat(), args[1]) for t, args in day.sent],
                     "status_reads": [t.strftime("%H:%M") for t in day.status_reads],
                     "fallbacks": day.state.fallback_refreshes, "refusals": day.state.delivery_refusals}
            out[name] = entry
            return entry

    d = run("reuse_due", starts=("19:10", "19:10"), posted_at={9001: "18:25"},
            projected_index=1, outage_until="18:20", cascade_min=17,
            scheduler={"cascade_budget_min": 19})
    assert d["locked"] and d["cascades"][-1] == "18:35" and len(d["cascades"]) == 4
    assert d["sent"][0][0][11:16] == "18:52" and d["fallbacks"][-1]["reused_check_cascade"]
    d = run("reuse_later_waits", starts=("19:10", "20:10"), posted_at={9002: "19:00"},
            projected_index=1, outage_until="18:20", cascade_min=17,
            scheduler={"lineup_check_offset_min": 180, "cascade_budget_min": 19})
    assert d["locked"] and d["sent"][0][0][11:16] == "19:52"
    assert d["fallbacks"][-1]["reused_check_cascade"] is False
    d = run("planned_unchanged", starts=("19:10", "19:10"), posted_at={9001: "18:25"},
            projected_index=1, outage_until="18:20", cascade_min=17,
            scheduler={"lineup_check_offset_min": 30, "cascade_budget_min": 19})
    assert not d["locked"] and d["cascades"][-2:] == ["18:40", "18:57"]
    assert d["refusals"][0]["at"][11:16] == "19:14"
    d = run("void_to_playable", undo_postponement=True)
    assert d["runs"] == ["16:10", "17:10", "18:10", "18:35"] and len(d["sent"]) == 1
    d = run("doubleheader_same_pair", starts=("19:10", "19:20"), doubleheader=True)
    assert d["runs"][-2:] == ["18:35", "18:45"] and len(d["sent"]) == 1
    d = run("void_during_attempt", void_after={9001: ("18:36", "Postponed")}, cascade_min=1)
    assert d["runs"][-1] == "18:36" and d["sent"] == []
    d = run("status_outage", fail_status_read=True)
    assert d["runs"][-1] == "18:35" and len(d["sent"]) == 1
    d = run("continue_without_result", starts=("19:10", "19:20", "19:30"),
            early_sleep_targets=("18:10", "18:30", "18:35"))
    assert d["runs"] == ["18:45", "18:55"] and len(d["sent"]) == 1

(HERE / "r2-scheduler-results.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps({k: {"runs": v["runs"], "cascades": v["cascades"], "locked": v["locked"]}
                  for k, v in out.items()}, indent=2))
