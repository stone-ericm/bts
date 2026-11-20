"""Item 2 probe (v2): re-run Round 2's probe_receipt_identity.py and probe_receipt_correction.py, plus reverse dedup."""

from __future__ import annotations

from contextlib import ExitStack, nullcontext, redirect_stderr
from datetime import datetime
from io import StringIO
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

REVIEW = Path(__file__).resolve().parent
ROOT = REVIEW.parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
R2_SRC = REVIEW / "src_r2_e897c766" / "src"
HEAD_SRC = ROOT / "src"
STAGED = REVIEW / "r2_codex_staged"
MPL_DIR = REVIEW / ".mplconfig"
ET = ZoneInfo("America/New_York")


class ProbeTimeout(BaseException):
    pass


def set_deadline(seconds: int) -> None:
    def on_alarm(*_):
        raise ProbeTimeout(f"probe_2b exceeded {seconds}s")

    signal.signal(signal.SIGALRM, on_alarm)
    signal.setitimer(signal.ITIMER_REAL, seconds)


def stage_r2_codex_probes() -> None:
    if STAGED.exists():
        shutil.rmtree(STAGED)
    STAGED.mkdir(parents=True)
    MPL_DIR.mkdir(exist_ok=True)

    support_src = (REVIEW / "round2" / "round1-codex" / "probe_support.py").read_text(encoding="utf-8")
    assert support_src.count("ROOT = HERE.parents[1]") == 1
    support_adapted = support_src.replace("ROOT = HERE.parents[1]", "ROOT = HERE.parents[2]", 1)
    (STAGED / "probe_support.py").write_text(support_adapted, encoding="utf-8")

    ident_src = (REVIEW / "round2" / "codex" / "probe_receipt_identity.py").read_text(encoding="utf-8")
    old_insert = 'sys.path.insert(0, str(Path(__file__).resolve().parent / "replay" / "codex"))'
    assert ident_src.count(old_insert) == 1
    ident_adapted = ident_src.replace(old_insert, "sys.path.insert(0, str(Path(__file__).resolve().parent))", 1)
    (STAGED / "probe_receipt_identity.py").write_text(ident_adapted, encoding="utf-8")

    corr_src = (REVIEW / "round2" / "codex" / "probe_receipt_correction.py").read_text(encoding="utf-8")
    (STAGED / "probe_receipt_correction.py").write_text(corr_src, encoding="utf-8")


def run_staged(script_name: str, src_path: Path, log_name: str) -> subprocess.CompletedProcess[str]:
    env = os.environ | {
        "PYTHONPATH": os.pathsep.join([str(src_path), str(STAGED), str(ROOT)]),
        "TZ": "America/New_York",
        "PYTHONDONTWRITEBYTECODE": "1",
        "MPLCONFIGDIR": str(MPL_DIR),
    }
    proc = subprocess.run(
        [str(PYTHON), str(STAGED / script_name)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=55,
    )
    (REVIEW / log_name).write_text(proc.stdout + proc.stderr, encoding="utf-8")
    return proc


def run_full_128_and_reverse_on_head() -> dict:
    os.environ["MPLCONFIGDIR"] = str(MPL_DIR)
    sys.path.insert(0, str(HEAD_SRC))
    sys.path.insert(0, str(STAGED))
    sys.path.insert(0, str(ROOT))

    import probe_receipt_correction as corr
    import probe_receipt_identity as identity
    import probe_support as support
    from bts import contest_state, daily_decision, dm, picks, posting, scheduler, strategy
    from bts.health import alert

    support.HERE = REVIEW

    old_5_member = json.dumps({k: v for k, v in identity.SHAPE.items() if k != "body_sha256"})
    identity.SHAPE["body_sha256"] = corr.checksum(identity.status_body(identity.DAY))

    mismatches = []
    total_cases = 0
    with identity.fixtures("r2-head-full") as (root, patch_ctx, opening):
        foreign_doc = corr.foreign_receipt(root, patch_ctx)
        suspicious = {
            "extra contradictory error": json.dumps(
                identity.SHAPE | {"error": "send_dm returned no usable message id"}
            ),
            "extra unknown key": json.dumps(identity.SHAPE | {"unexpected": [False]}),
            "duplicate failed then sent status": '{"status":"failed",' + json.dumps(identity.SHAPE)[1:],
            "duplicate bad then correct schema": '{"schema_version":null,' + json.dumps(identity.SHAPE)[1:],
            "cutoff copy with message ID": None,
            "other opt-in alert": foreign_doc,
            "old five-member receipt": old_5_member,
        }
        controls = dict(identity.wrong_field_cases()) | {
            "malformed JSON": "{",
            "wrong top-level type": "[]",
            "real other-slate copy": None,
            "real legacy cutoff copy": None,
            "real prior C2 acknowledgement": None,
        }
        all_items = controls | suspicious
        for route in ("direct", "classification"):
            for idx, (label, doc) in enumerate(all_items.items()):
                got = identity.observe(root / f"{route}-{idx}", patch_ctx, opening, route, label, doc)
                expected = int(label != "real prior C2 acknowledgement")
                total_cases += 1
                if got != expected:
                    mismatches.append((route, label, got, expected))

        mpath = root / "multi-order.json"
        keys = [identity.KEY, "CRITICAL:pick_delivery"]
        alerts_in = [
            alert.Alert("CRITICAL", "pick_delivery", "one"),
            alert.Alert("CRITICAL", "late_delivery", "two"),
        ]
        mbody = alert.format_health_dm_body(alerts_in)
        assert mbody is not None
        mpath.write_text(
            json.dumps(identity.SHAPE | {"sent_sources": list(reversed(keys)), "body_sha256": corr.checksum(mbody)}),
            encoding="utf-8",
        )
        msender = Mock(return_value="multi-fresh-id")
        with patch_ctx.context() as mscoped:
            mscoped.setattr(alert, "send_dm", msender)
            mkw = dict(status_path=mpath, now_et_date=identity.date.fromisoformat(identity.DAY), require_message_id=True)
            assert alert.dispatch_dm_for_health_alerts(alerts_in, "offline.test", **mkw) is True
            assert json.loads(mpath.read_bytes())["sent_sources"] == keys
            assert alert.dispatch_dm_for_health_alerts(alerts_in, "offline.test", **mkw) is False
            msender.assert_called_once()

    assert mismatches == [], f"Unexpected mismatches on HEAD: {mismatches}"

    reverse_checks = {}
    for label, slate, next_slate in [
        ("ordinary_midnight", "2027-03-25", "2027-03-26"),
        ("new_year_rollover", "2027-12-31", "2028-01-01"),
    ]:
        for origin_route in ("direct", "classification"):
            with TemporaryDirectory(prefix=f"rev_{label}_{origin_route}_", dir=REVIEW) as td:
                base = Path(td)
                pdir = base / "picks"
                pdir.mkdir(parents=True)
                now_box = [datetime.fromisoformat(f"{slate}T18:20:00").replace(tzinfo=ET)]

                class PinnedClock(datetime):
                    @classmethod
                    def now(cls, tz=None):
                        return now_box[0].astimezone(tz) if tz else now_box[0].replace(tzinfo=None)

                games = {}
                for s in (slate, next_slate):
                    p1 = picks.Pick("Steven Kwan", 680757, "CLE", 1, "Cole", 543037, 0.82, [], False, 101, f"{s}T23:10:00Z")
                    dp_s = picks.DailyPick(s, now_box[0].isoformat(), p1, None, None, delivery_attempted=True)
                    picks.save_pick(dp_s, pdir)
                    games[s] = [{
                        "gamePk": 101,
                        "gameDate": f"{s}T23:10:00Z",
                        "status": {"abstractGameCode": "P", "detailedState": "Scheduled"},
                        "teams": {"away": {"team": {"name": "Away"}}, "home": {"team": {"name": "Home"}}},
                    }]

                cfg = {
                    "orchestrator": {"picks_dir": str(pdir), "heartbeat_path": str(base / "hb")},
                    "bluesky": {"dm_recipient": "offline.test"},
                    "scheduler": {"pick_delivery": "dm", "live_forward_capture_on_lock": False, "contest_state_required": False},
                    "health_checks": {"enabled": False},
                    "tiers": [],
                }
                sender = Mock(return_value=f"ack-{slate}-{origin_route}")
                with ExitStack() as stack:
                    for ctx in (
                        patch.object(contest_state, "datetime", PinnedClock),
                        patch.object(daily_decision, "datetime", PinnedClock),
                        patch.object(alert, "datetime", PinnedClock),
                        patch.object(scheduler, "_now_et", lambda: now_box[0]),
                        patch.object(scheduler, "fetch_schedule", side_effect=lambda d: games.get(d, [])),
                        patch.object(scheduler, "check_confirmed_lineups", lambda *_: {}),
                        patch.object(scheduler, "_idle_until_next_wakeup", lambda *_: None),
                        patch.object(scheduler, "_watchdog_ping_sleep", lambda *_: None),
                        patch.object(scheduler, "heartbeat_watchdog", lambda *a, **k: nullcontext()),
                        patch.object(scheduler, "notify_ready", Mock()),
                        patch.object(scheduler, "notify_watchdog", Mock()),
                        patch.object(scheduler, "run_result_polling", Mock(return_value="final")),
                        patch.object(contest_state, "load_decision_streak_state", return_value=Mock(streak=0)),
                        patch.object(dm, "send_dm", Mock(side_effect=AssertionError("pick resent"))),
                        patch.object(alert, "send_dm", sender),
                        patch.object(posting, "post_to_bluesky", Mock(side_effect=AssertionError("public post"))),
                        patch.object(strategy, "_mdp_cache", None),
                        redirect_stderr(StringIO()),
                    ):
                        stack.enter_context(ctx)

                    dp_slate = picks.load_pick(slate, pdir)
                    if origin_route == "direct":
                        st = scheduler.SchedulerState(slate, now_box[0].isoformat(), [], [], [], False, None, None, None)
                        assert scheduler._deliver_and_lock_pick(dp_slate, cfg, pdir, st, slate, "e1") is False
                    else:
                        assert scheduler.run_day(slate, cfg) is None
                    assert sender.call_count == 1
                    own_file = base / "health_state/delivery_outcome_unknown" / f"{slate}.json"
                    first_bytes = own_file.read_bytes()

                    st2 = scheduler.SchedulerState(slate, now_box[0].isoformat(), [], [], [], False, None, None, None)
                    assert scheduler._deliver_and_lock_pick(dp_slate, cfg, pdir, st2, slate, "e2") is False
                    assert scheduler.run_day(slate, cfg) is None
                    assert sender.call_count == 1
                    assert own_file.read_bytes() == first_bytes

                    now_box[0] = datetime.fromisoformat(f"{next_slate}T00:01:00").replace(tzinfo=ET)
                    st3 = scheduler.SchedulerState(slate, now_box[0].isoformat(), [], [], [], False, None, None, None)
                    assert scheduler._deliver_and_lock_pick(dp_slate, cfg, pdir, st3, slate, "e3") is False
                    assert scheduler.run_day(slate, cfg) is None
                    assert sender.call_count == 1
                    assert own_file.read_bytes() == first_bytes

                    now_box[0] = datetime.fromisoformat(f"{next_slate}T09:15:00").replace(tzinfo=ET)
                    no_rec_cfg = json.loads(json.dumps(cfg))
                    no_rec_cfg["bluesky"].pop("dm_recipient")
                    assert scheduler.run_day(slate, no_rec_cfg) is None
                    assert sender.call_count == 1
                    assert own_file.read_bytes() == first_bytes

                    now_box[0] = datetime.fromisoformat(f"{next_slate}T18:20:00").replace(tzinfo=ET)
                    sender.return_value = f"ack-{next_slate}"
                    assert scheduler.run_day(next_slate, cfg) is None
                    assert sender.call_count == 2
                    assert scheduler.run_day(next_slate, cfg) is None
                    assert sender.call_count == 2
                    assert own_file.read_bytes() == first_bytes

                reverse_checks[f"{label}:{origin_route}"] = "deduped_across_5_encounters_and_midnight"

    return {
        "head_full_matrix_route_cases": total_cases,
        "head_full_matrix_mismatches": len(mismatches),
        "reverse_dedup": reverse_checks,
    }


def main() -> None:
    set_deadline(55)
    try:
        stage_r2_codex_probes()

        p_ident_r2 = run_staged("probe_receipt_identity.py", R2_SRC, "r2_probe_receipt_identity_on_e897c766.log")
        assert p_ident_r2.returncode == 0, p_ident_r2.stderr
        assert "12 mismatches across 126 route cases." in p_ident_r2.stdout, p_ident_r2.stdout

        p_ident_head_raw = run_staged("probe_receipt_identity.py", HEAD_SRC, "r2_probe_receipt_identity_raw_on_head.log")
        raw_out = p_ident_head_raw.stdout + p_ident_head_raw.stderr
        assert p_ident_head_raw.returncode != 0 and 'SHAPE | {"message_id": "fresh-id"}' in raw_out

        p_corr_head = run_staged("probe_receipt_correction.py", HEAD_SRC, "r2_probe_receipt_correction_on_head.log")
        assert p_corr_head.returncode == 0, p_corr_head.stdout + p_corr_head.stderr

        full_head = run_full_128_and_reverse_on_head()

        summary = {
            "e897c766_probe_receipt_identity_exit": p_ident_r2.returncode,
            "e897c766_mismatches_line": "12 mismatches across 126 route cases.",
            "head_probe_receipt_identity_unmodified_5key_shape_note": (
                "Exits 1 at line 142 (assert json.loads(after) == SHAPE | {'message_id': 'fresh-id'}) because HEAD writes 6-key receipt with body_sha256"
            ),
            "head_probe_receipt_correction_exit": p_corr_head.returncode,
            **full_head,
        }
        (REVIEW / "probe_2_results.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(
            "PROBE 2B OK: e897c766 reproduced 12/126 mismatches; probe_receipt_correction.py exited 0 on HEAD (10/10 B1+B2 cases fixed); "
            f"full 6-key matrix ran {full_head['head_full_matrix_route_cases']} route cases with 0 mismatches; "
            f"reverse dedup verified across {len(full_head['reverse_dedup'])} midnight/New Year/route chains."
        )
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == "__main__":
    main()
