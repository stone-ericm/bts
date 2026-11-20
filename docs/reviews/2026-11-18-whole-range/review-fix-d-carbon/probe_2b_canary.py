#!/usr/bin/env python3
"""Item 2 probe (2b): bts deploy-canary read-only guarantee, imports, config validation, and pinned ET boundaries."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime
from importlib import import_module
from pathlib import Path
from zoneinfo import ZoneInfo

from click.testing import CliRunner

from bts import contest_state, orchestrator, scheduler
from bts.cli import cli
from tests.test_contest_dated_cs4a import observation, write


def snapshot_tree(root: Path) -> dict[str, tuple[str, bytes | str]]:
    snap = {}
    for p in sorted(root.rglob("*")):
        rel = str(p.relative_to(root))
        st = p.lstat()
        if p.is_symlink():
            snap[rel] = ("symlink", os.readlink(p))
        elif p.is_dir():
            snap[rel] = ("dir", oct(st.st_mode & 0o777))
        elif p.is_file():
            snap[rel] = ("file", hashlib.sha256(p.read_bytes()).hexdigest())
        else:
            snap[rel] = ("other", oct(st.st_mode))
    return snap


def main() -> int:
    work_dir = Path(tempfile.mkdtemp(prefix="p2b_", dir=Path(__file__).resolve().parent))
    runner = CliRunner()
    results = {}
    try:
        orig_run_day = scheduler.run_day
        orig_run_and_pick = orchestrator.run_and_pick
        scheduler.run_day = lambda *a, **kw: (_ for _ in ()).throw(AssertionError("run_day called!"))
        orchestrator.run_and_pick = lambda *a, **kw: (_ for _ in ()).throw(AssertionError("run_and_pick called!"))

        try:
            cfg_root = work_dir / "configs"
            cfg_root.mkdir()
            picks_ok = cfg_root / "picks"
            picks_ok.mkdir()
            (picks_ok / "sentinel.json").write_bytes(b'{"keep":1}\n')
            et = ZoneInfo("America/New_York")
            pinned_now = datetime(2027, 7, 15, 14, 0, 0, tzinfo=et)
            write(picks_ok, observation("2027-07-15", recorded_at="2027-07-15T11:00:00-04:00"))

            cli_mod = import_module("bts.cli")
            orig_now_et = cli_mod._now_et
            cli_mod._now_et = lambda: pinned_now

            bad_configs = {
                "missing_file": None,
                "config_is_dir": "DIR",
                "invalid_toml": "[scheduler\nentry_intent = 'research'\n",
                "missing_orchestrator": '[scheduler]\nentry_intent = "research"\npick_delivery = "private"\n',
                "orchestrator_not_table": 'orchestrator = 123\n[scheduler]\nentry_intent = "research"\npick_delivery = "private"\n',
                "picks_dir_int": '[orchestrator]\npicks_dir = 123\n[scheduler]\nentry_intent = "research"\npick_delivery = "private"\n',
                "picks_dir_missing": f'[orchestrator]\npicks_dir = "{cfg_root / "nope"}"\n[scheduler]\nentry_intent = "research"\npick_delivery = "private"\n',
                "picks_dir_is_file": f'[orchestrator]\npicks_dir = "{picks_ok / "sentinel.json"}"\n[scheduler]\nentry_intent = "research"\npick_delivery = "private"\n',
                "private_missing_intent": f'[orchestrator]\npicks_dir = "{picks_ok}"\n[scheduler]\npick_delivery = "private"\n',
                "private_enter_mismatch": f'[orchestrator]\npicks_dir = "{picks_ok}"\n[scheduler]\nentry_intent = "enter"\npick_delivery = "private"\ncontest_state_required = true\n',
                "dm_research_mismatch": f'[orchestrator]\npicks_dir = "{picks_ok}"\n[scheduler]\nentry_intent = "research"\npick_delivery = "dm"\ncontest_state_required = true\n',
                "public_research_mismatch": f'[orchestrator]\npicks_dir = "{picks_ok}"\n[scheduler]\nentry_intent = "research"\npick_delivery = "public"\ncontest_state_required = true\n',
                "default_public_research_mismatch": f'[orchestrator]\npicks_dir = "{picks_ok}"\n[scheduler]\nentry_intent = "research"\n',
                "dm_missing_contest_gate": f'[orchestrator]\npicks_dir = "{picks_ok}"\n[scheduler]\nentry_intent = "enter"\npick_delivery = "dm"\n',
                "public_missing_contest_gate": f'[orchestrator]\npicks_dir = "{picks_ok}"\n[scheduler]\nentry_intent = "enter"\npick_delivery = "public"\n',
                "legacy_shadow_mode": f'[orchestrator]\npicks_dir = "{picks_ok}"\n[scheduler]\nentry_intent = "research"\nshadow_mode = true\n',
                "unknown_delivery": f'[orchestrator]\npicks_dir = "{picks_ok}"\n[scheduler]\nentry_intent = "enter"\npick_delivery = "pigeon"\ncontest_state_required = true\n',
                "research_bad_health_table": f'health_checks = "bad"\n[orchestrator]\npicks_dir = "{picks_ok}"\n[scheduler]\nentry_intent = "research"\npick_delivery = "private"\n',
                "live_bad_health_table": f'health_checks = 99\n[orchestrator]\npicks_dir = "{picks_ok}"\n[scheduler]\nentry_intent = "enter"\npick_delivery = "dm"\ncontest_state_required = true\n',
                "bool_intent": f'[orchestrator]\npicks_dir = "{picks_ok}"\n[scheduler]\nentry_intent = true\npick_delivery = "private"\n',
            }

            bad_outcomes = {}
            for name, text in bad_configs.items():
                if text is None:
                    target_cfg = cfg_root / "does_not_exist.toml"
                elif text == "DIR":
                    target_cfg = cfg_root / "dir_cfg"
                    target_cfg.mkdir(exist_ok=True)
                else:
                    target_cfg = cfg_root / f"{name}.toml"
                    target_cfg.write_text(text)
                before = snapshot_tree(cfg_root)
                res = runner.invoke(cli, ["deploy-canary", "--config", str(target_cfg)])
                after = snapshot_tree(cfg_root)
                assert before == after, f"Tree mutated on {name}!"
                assert res.exit_code == 1, f"{name} unexpectedly passed: {res.output}"
                assert "configuration refused" in res.output, (name, res.output)
                bad_outcomes[name] = res.output.strip()
            results["bad_configs_all_refused_count"] = len(bad_outcomes)

            res_root = work_dir / "research"
            res_picks = res_root / "picks"
            res_picks.mkdir(parents=True)
            res_cfg = res_root / "cfg.toml"
            res_cfg.write_text(
                f'[orchestrator]\npicks_dir = "{res_picks}"\n'
                '[scheduler]\nentry_intent = "research"\npick_delivery = "private"\n'
            )
            orig_loader = contest_state.load_decision_streak_state
            loader_calls = []

            def spy_loader(*a, **kw):
                loader_calls.append((a, kw))
                return orig_loader(*a, **kw)

            contest_state.load_decision_streak_state = spy_loader
            try:
                before = snapshot_tree(res_root)
                res = runner.invoke(cli, ["deploy-canary", "--config", str(res_cfg)])
                after = snapshot_tree(res_root)
                assert before == after
                assert res.exit_code == 0 and "research; contest state skipped" in res.output
                assert loader_calls == []
                results["research_skips_loader"] = True

                unusable_cases = {}
                for mode_name, delivery_line in [
                    ("dm", 'pick_delivery = "dm"\n'),
                    ("public", 'pick_delivery = "public"\n'),
                    ("bluesky_alias", 'pick_delivery = "bluesky"\n'),
                    ("implicit_public", ""),
                ]:
                    u_dir = work_dir / f"unusable_{mode_name}"
                    u_picks = u_dir / "picks"
                    u_picks.mkdir(parents=True)
                    u_cfg = u_dir / "cfg.toml"
                    u_cfg.write_text(
                        f'[orchestrator]\npicks_dir = "{u_picks}"\n'
                        f'[scheduler]\nentry_intent = "enter"\n{delivery_line}contest_state_required = true\n'
                        '[health_checks]\ncontest_state_expected = true\n'
                    )
                    before = snapshot_tree(u_dir)
                    r_empty = runner.invoke(cli, ["deploy-canary", "--config", str(u_cfg)])
                    assert snapshot_tree(u_dir) == before
                    assert r_empty.exit_code == 1 and "contest state refused" in r_empty.output

                    write(u_picks, observation("2027-07-14", recorded_at="2027-07-14T11:00:00-04:00"))
                    before = snapshot_tree(u_dir)
                    r_stale = runner.invoke(cli, ["deploy-canary", "--config", str(u_cfg)])
                    assert snapshot_tree(u_dir) == before
                    assert r_stale.exit_code == 1 and "contest state refused" in r_stale.output

                    (u_picks / "account_state" / "dated_contest_observation.json").write_text("{corrupt")
                    before = snapshot_tree(u_dir)
                    r_corrupt = runner.invoke(cli, ["deploy-canary", "--config", str(u_cfg)])
                    assert snapshot_tree(u_dir) == before
                    assert r_corrupt.exit_code == 1
                    unusable_cases[mode_name] = [r_empty.exit_code, r_stale.exit_code, r_corrupt.exit_code]
                results["live_unusable_all_refused"] = unusable_cases

                boundaries = {}
                for label, day, pre_iso, post_iso in [
                    ("midseason_midnight", "2027-07-15", "2027-07-15T23:59:59-04:00", "2027-07-16T00:00:01-04:00"),
                    ("year_end_midnight", "2027-12-31", "2027-12-31T23:59:59-05:00", "2028-01-01T00:00:01-05:00"),
                ]:
                    b_dir = work_dir / f"boundary_{label}"
                    b_picks = b_dir / "picks"
                    write(b_picks, observation(day, recorded_at=pre_iso[:10] + "T12:00:00" + pre_iso[-6:]))
                    b_cfg = b_dir / "cfg.toml"
                    b_cfg.write_text(
                        f'[orchestrator]\npicks_dir = "{b_picks}"\n'
                        '[scheduler]\nentry_intent = "enter"\npick_delivery = "dm"\ncontest_state_required = true\n'
                        '[health_checks]\ncontest_state_expected = true\n'
                    )
                    before = snapshot_tree(b_dir)
                    cli_mod._now_et = lambda p=pre_iso: datetime.fromisoformat(p)
                    r_pre = runner.invoke(cli, ["deploy-canary", "--config", str(b_cfg)])
                    cli_mod._now_et = lambda p=post_iso: datetime.fromisoformat(p)
                    r_post = runner.invoke(cli, ["deploy-canary", "--config", str(b_cfg)])
                    assert snapshot_tree(b_dir) == before
                    assert r_pre.exit_code == 0 and "live contest state accepted (dated, streak 12)" in r_pre.output
                    assert r_post.exit_code == 1 and "contest state refused" in r_post.output
                    boundaries[label] = {"23_59_59": r_pre.output.strip(), "00_00_01": r_post.output.strip()}
                results["pinned_boundaries"] = boundaries

                adv_dir = work_dir / "adv_dm_missing_recipient"
                adv_picks = adv_dir / "picks"
                write(adv_picks, observation("2027-07-15", recorded_at="2027-07-15T11:00:00-04:00"))
                adv_cfg = adv_dir / "cfg.toml"
                adv_cfg.write_text(
                    f'[orchestrator]\npicks_dir = "{adv_picks}"\n'
                    '[scheduler]\nentry_intent = "enter"\npick_delivery = "dm"\n'
                    'contest_state_required = true\ncascade_budget_min = "not-a-float"\n'
                    '[health_checks]\ncontest_state_expected = true\n'
                )
                cli_mod._now_et = lambda: pinned_now
                r_adv = runner.invoke(cli, ["deploy-canary", "--config", str(adv_cfg)])

                saved_posting = sys.modules.pop("bts.posting", None)
                try:
                    runner.invoke(cli, ["deploy-canary", "--config", str(adv_cfg)])
                    posting_imported_in_dm = "bts.posting" in sys.modules
                finally:
                    if saved_posting is not None:
                        sys.modules["bts.posting"] = saved_posting

                results["adversarial_notes"] = {
                    "dm_missing_recipient_and_bad_float_exit_code": r_adv.exit_code,
                    "dm_missing_recipient_output": r_adv.output.strip(),
                    "bts_posting_imported_in_dm_canary": posting_imported_in_dm,
                    "scheduler_imported": "bts.scheduler" in sys.modules,
                    "model_predict_imported": "bts.model.predict" in sys.modules,
                    "dm_imported": "bts.dm" in sys.modules,
                }
            finally:
                contest_state.load_decision_streak_state = orig_loader
                cli_mod._now_et = orig_now_et
        finally:
            scheduler.run_day = orig_run_day
            orchestrator.run_and_pick = orig_run_and_pick

        out_path = Path(__file__).resolve().parent / "probe_2b_canary.json"
        out_path.write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(results, indent=2))
        return 0
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
