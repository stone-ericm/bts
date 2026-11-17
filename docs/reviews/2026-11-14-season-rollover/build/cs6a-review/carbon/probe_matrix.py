"""Adversarial Probe 1: Independent specification check + Base (09d9b51f) vs Branch (cad7c5ec) CLI matrix.

Installs a Python audit hook blocking all DNS/socket/network operations.
"""
from __future__ import annotations

import builtins
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import tomllib
from unittest.mock import Mock, patch

REPO = Path(__file__).resolve().parents[2]
REVIEW_DIR = Path(__file__).resolve().parent
BASE_SRC = REVIEW_DIR / "base_tree" / "src"
BRANCH_SRC = REPO / "src"

BLOCKED_AUDIT_EVENTS = frozenset({
    "socket.connect",
    "socket.connect_ex",
    "socket.getaddrinfo",
    "socket.gethostbyname",
    "socket.gethostbyname_ex",
    "socket.gethostbyaddr",
    "socket.getnameinfo",
    "socket.sendto",
    "urllib.Request",
})


def install_network_audit_hook() -> None:
    def _hook(event: str, args: tuple) -> None:
        if event in BLOCKED_AUDIT_EVENTS:
            raise RuntimeError(f"Network/DNS forbidden by review audit hook: {event} {args!r}")

    sys.addaudithook(_hook)


# ---------------------------------------------------------------------------
# Independent specification (derived strictly from kickoff-cs6a.md + scheduler rules,
# WITHOUT calling bts.entry_intent or bts.scheduler)
# ---------------------------------------------------------------------------

SPEC_FIX = (
    "Optional reads can use a contest observation without the season-coverage check. "
    "Set contest_state_required = true under [scheduler] and "
    "contest_state_expected = true under [health_checks]."
)

PUBLIC_ALIASES = {"public", "bluesky", "feed", "post"}
DM_ALIASES = {"dm", "bluesky_dm", "direct_message"}
PRIVATE_ALIASES = {"private", "none", "off", "local"}


def spec_scheduler_delivery(config: dict) -> tuple[str | None, str | None]:
    """Return (effective_mode, error_message) independently of bts.scheduler."""
    sched = config.get("scheduler", {})
    raw = sched.get("pick_delivery", sched.get("posting_mode"))
    if raw is None and "shadow_mode" in sched and not sched.get("private_mode", False):
        return None, "legacy scheduler.shadow_mode is not supported: set scheduler.pick_delivery (public, dm or private)"
    if raw is None:
        return ("private" if sched.get("private_mode", False) else "public"), None
    norm = str(raw).strip().lower().replace("-", "_")
    if norm in PUBLIC_ALIASES:
        return "public", None
    if norm in DM_ALIASES:
        return "dm", None
    if norm in PRIVATE_ALIASES:
        return "private", None
    return None, "scheduler.pick_delivery must be one of: public, dm, private"


def spec_contest_state_required(config: dict) -> bool:
    sched = config.get("scheduler", {})
    health = config.get("health_checks", {})
    return bool(sched.get("contest_state_required", False) or health.get("contest_state_expected", False))


def spec_live_gate_problem(config: dict, *, delivery_override: str | None = None) -> str | None:
    if "scheduler" in config and not isinstance(config["scheduler"], dict):
        return "[scheduler] is not a table"
    if "health_checks" in config and not isinstance(config["health_checks"], dict):
        return "[health_checks] is not a table"
    if spec_contest_state_required(config):
        return None

    if delivery_override is not None:
        delivery = delivery_override
    else:
        delivery, _err = spec_scheduler_delivery(config)

    conditions: list[str] = []
    if delivery in ("dm", "public"):
        label = "scheduler delivery" if delivery_override is None else "delivery"
        conditions.append(f"{label} {delivery!r}")
    sched = config.get("scheduler", {})
    intent = sched.get("entry_intent")
    if isinstance(intent, str) and intent == "enter":
        conditions.append("scheduler.entry_intent = 'enter'")
    if not conditions:
        return None
    verb = "requires" if len(conditions) == 1 else "require"
    return f"{' and '.join(conditions)} {verb} contest state. {SPEC_FIX}"


def spec_check_entry_intent_problems(config: dict, *, include_live_gate: bool = True) -> tuple[str | None, str | None, tuple[str, ...]]:
    sched = config.get("scheduler", {})
    if not isinstance(sched, dict):
        return None, None, ("[scheduler] is not a table",)
    problems: list[str] = []
    raw_intent = sched.get("entry_intent")
    intent = raw_intent if isinstance(raw_intent, str) and raw_intent in ("enter", "research") else None
    if "entry_intent" not in sched:
        problems.append('scheduler.entry_intent is missing: set it to "enter" or "research" (there is no default)')
    elif intent is None:
        problems.append(f'scheduler.entry_intent must be exactly "enter" or "research", got {raw_intent!r}')

    delivery, deliv_err = spec_scheduler_delivery(config)
    if deliv_err is not None:
        problems.append(f"the scheduler refuses this delivery config: {deliv_err}")

    expected_map = {"enter": ("dm", "public"), "research": ("private",)}
    if intent is not None and delivery is not None and delivery not in expected_map[intent]:
        problems.append(
            f"entry_intent {intent!r} expects delivery {' or '.join(expected_map[intent])}, "
            f"but the scheduler would deliver {delivery!r}"
        )
    if include_live_gate:
        gate_prob = spec_live_gate_problem(config)
        if gate_prob is not None:
            problems.append(gate_prob)
    return intent, delivery, tuple(problems)


def run_exhaustive_in_memory_spec_check() -> dict:
    """Search for ANY counterexample across >500k configurations in memory (branch code)."""
    from bts import entry_intent, scheduler

    delivery_dicts: list[dict] = [
        {},
        # public spellings & case/whitespace
        {"pick_delivery": "public"},
        {"pick_delivery": "bluesky"},
        {"pick_delivery": "feed"},
        {"pick_delivery": "post"},
        {"pick_delivery": "  pUbLiC\t\n"},
        {"pick_delivery": " BLUESKY "},
        {"posting_mode": "feed"},
        {"posting_mode": " POST "},
        # dm spellings & hyphens/case/whitespace
        {"pick_delivery": "dm"},
        {"pick_delivery": "bluesky_dm"},
        {"pick_delivery": "direct_message"},
        {"pick_delivery": "bluesky-dm"},
        {"pick_delivery": "direct-message"},
        {"pick_delivery": "  Direct-Message  "},
        {"posting_mode": "bluesky-dm"},
        {"posting_mode": "DIRECT_MESSAGE"},
        # private spellings
        {"pick_delivery": "private"},
        {"pick_delivery": "none"},
        {"pick_delivery": "off"},
        {"pick_delivery": "local"},
        {"pick_delivery": "  LoCaL  "},
        {"posting_mode": "off"},
        {"private_mode": True},
        {"private_mode": "false"},  # truthy non-bool!
        {"private_mode": 1},
        {"private_mode": False},    # falsy -> defaults to public!
        {"private_mode": 0},
        {"private_mode": ""},
        # key conflicts & precedence
        {"pick_delivery": "private", "posting_mode": "dm"},
        {"pick_delivery": "dm", "posting_mode": "private", "private_mode": True},
        {"pick_delivery": "public", "private_mode": True, "shadow_mode": True},
        {"posting_mode": "dm", "private_mode": True},
        {"pick_delivery": None, "posting_mode": "dm"},  # Python-only None -> falls through to default public!
        {"pick_delivery": None, "posting_mode": "dm", "private_mode": True},  # Python-only None + private_mode -> private!
        # refused / invalid spellings
        {"shadow_mode": True},
        {"shadow_mode": False},
        {"shadow_mode": False, "private_mode": True},  # private_mode True suppresses shadow_mode error -> private!
        {"pick_delivery": "invalid", "posting_mode": "dm"},
        {"pick_delivery": ""},
        {"pick_delivery": "direct message"},  # internal space -> refused
        {"pick_delivery": "bluesky dm"},
        {"pick_delivery": True},
        {"pick_delivery": False},
        {"pick_delivery": 0},
        {"pick_delivery": 1},
        {"pick_delivery": []},
        {"pick_delivery": {}},
        {"posting_mode": 42},
    ]

    intent_variants: list[dict] = [
        {},
        {"entry_intent": "enter"},
        {"entry_intent": "research"},
        {"entry_intent": "Enter"},
        {"entry_intent": "ENTER"},
        {"entry_intent": " enter "},
        {"entry_intent": "research "},
        {"entry_intent": "other"},
        {"entry_intent": ""},
        {"entry_intent": 0},
        {"entry_intent": 1},
        {"entry_intent": True},
        {"entry_intent": False},
        {"entry_intent": []},
        {"entry_intent": {}},
        {"entry_intent": None},
    ]

    flag_variants: list[tuple[str, object]] = [
        ("absent", None),
        ("true", True),
        ("false", False),
        ("zero", 0),
        ("one", 1),
        ("empty_str", ""),
        ("str_false", "false"),
        ("empty_list", []),
        ("nonempty_list", [1]),
        ("empty_dict", {}),
        ("nonempty_dict", {"nested": 1}),
    ]

    overrides = (None, "public", "dm", "private", "bluesky", "", "invalid")

    # Guard against any file or clock access during predicate checks
    forbidden = Mock(side_effect=AssertionError("live_gate_problem / check_entry_intent touched file or clock"))
    gate_evals = 0
    intent_evals = 0
    counterexamples: list[dict] = []

    with (
        patch.object(builtins, "open", forbidden),
        patch.object(Path, "open", forbidden),
        patch.object(Path, "stat", forbidden),
        patch.object(scheduler, "_now_et", forbidden),
        patch.object(time, "time", forbidden),
    ):
        # 1. Non-table / missing / empty section combinations
        section_shapes = [
            ("absent", None),
            ("empty", {}),
            ("not_table_int", 1),
            ("not_table_str", "enter"),
            ("not_table_bool", False),
            ("not_table_list", []),
            ("not_table_none", None),
            ("present_req_true", {"contest_state_required": True, "entry_intent": "enter", "pick_delivery": "dm"}),
            ("present_req_false", {"contest_state_required": False, "entry_intent": "research", "pick_delivery": "private"}),
        ]
        health_shapes = [
            ("absent", None),
            ("empty", {}),
            ("not_table_int", 2),
            ("not_table_str", "true"),
            ("not_table_bool", True),
            ("not_table_list", [1]),
            ("not_table_none", None),
            ("present_exp_true", {"contest_state_expected": True}),
            ("present_exp_false", {"contest_state_expected": False}),
        ]
        for (sname, sval), (hname, hval) in itertools.product(section_shapes, health_shapes):
            cfg: dict = {}
            if sname != "absent":
                cfg["scheduler"] = sval
            if hname != "absent":
                cfg["health_checks"] = hval
            for ov in overrides:
                got = entry_intent.live_gate_problem(cfg, delivery_override=ov)
                exp = spec_live_gate_problem(cfg, delivery_override=ov)
                gate_evals += 1
                if got != exp and len(counterexamples) < 20:
                    counterexamples.append({"kind": "shape_gate", "cfg": repr(cfg), "ov": ov, "got": got, "exp": exp})
            for inc in (True, False):
                got_ic = entry_intent.check_entry_intent(cfg, include_live_gate=inc)
                exp_ic = spec_check_entry_intent_problems(cfg, include_live_gate=inc)
                intent_evals += 1
                if (got_ic.intent, got_ic.delivery, got_ic.problems) != exp_ic and len(counterexamples) < 20:
                    counterexamples.append({
                        "kind": "shape_intent", "cfg": repr(cfg), "inc": inc,
                        "got": (got_ic.intent, got_ic.delivery, got_ic.problems), "exp": exp_ic,
                    })

        # 2. Full well-shaped Cartesian product
        for ddict, ivar, (rname, rval), (hname, hval) in itertools.product(
            delivery_dicts, intent_variants, flag_variants, flag_variants
        ):
            sched = dict(ddict)
            sched.update(ivar)
            if rname != "absent":
                sched["contest_state_required"] = rval
            cfg = {"scheduler": sched}
            if hname != "absent":
                cfg["health_checks"] = {"contest_state_expected": hval}
            else:
                cfg["health_checks"] = {}

            for ov in overrides:
                got = entry_intent.live_gate_problem(cfg, delivery_override=ov)
                exp = spec_live_gate_problem(cfg, delivery_override=ov)
                gate_evals += 1
                if got != exp and len(counterexamples) < 20:
                    counterexamples.append({"kind": "product_gate", "cfg": repr(cfg), "ov": ov, "got": got, "exp": exp})

            for inc in (True, False):
                got_ic = entry_intent.check_entry_intent(cfg, include_live_gate=inc)
                exp_ic = spec_check_entry_intent_problems(cfg, include_live_gate=inc)
                intent_evals += 1
                if (got_ic.intent, got_ic.delivery, got_ic.problems) != exp_ic and len(counterexamples) < 20:
                    counterexamples.append({
                        "kind": "product_intent", "cfg": repr(cfg), "inc": inc,
                        "got": (got_ic.intent, got_ic.delivery, got_ic.problems), "exp": exp_ic,
                    })

    assert forbidden.call_count == 0
    return {
        "gate_evaluations": gate_evals,
        "check_entry_intent_evaluations": intent_evals,
        "counterexamples": counterexamples,
    }


# ---------------------------------------------------------------------------
# Real TOML + CliRunner Base vs Branch Matrix
# ---------------------------------------------------------------------------

def build_toml_cases() -> list[tuple[str, str]]:
    """Build deterministic (case_id, toml_text) covering TOML-representable matrix."""
    sched_deliveries = [
        ("deliv_absent", ""),
        ("deliv_public", 'pick_delivery = "public"\n'),
        ("deliv_bluesky", 'pick_delivery = "bluesky"\n'),
        ("deliv_feed_ws", 'pick_delivery = "  FeEd "\n'),
        ("deliv_dm", 'pick_delivery = "dm"\n'),
        ("deliv_bsky_dm_hyphen", 'pick_delivery = "bluesky-dm"\n'),
        ("deliv_direct_msg", 'pick_delivery = " DIRECT-MESSAGE "\n'),
        ("deliv_private", 'pick_delivery = "private"\n'),
        ("deliv_off", 'pick_delivery = "off"\n'),
        ("deliv_posting_dm", 'posting_mode = "bluesky-dm"\n'),
        ("deliv_posting_local", 'posting_mode = "local"\n'),
        ("deliv_priv_true", "private_mode = true\n"),
        ("deliv_priv_str_false", 'private_mode = "false"\n'),
        ("deliv_priv_false", "private_mode = false\n"),
        ("deliv_conflict_priv_over_dm", 'pick_delivery = "private"\nposting_mode = "dm"\n'),
        ("deliv_conflict_dm_over_priv", 'pick_delivery = "dm"\nprivate_mode = true\nshadow_mode = true\n'),
        ("deliv_shadow_false", "shadow_mode = false\n"),
        ("deliv_invalid_over_dm", 'pick_delivery = "invalid"\nposting_mode = "dm"\n'),
        ("deliv_int", "pick_delivery = 7\n"),
    ]
    intents = [
        ("int_absent", ""),
        ("int_enter", 'entry_intent = "enter"\n'),
        ("int_research", 'entry_intent = "research"\n'),
        ("int_case_Enter", 'entry_intent = "Enter"\n'),
        ("int_non_str", "entry_intent = 1\n"),
    ]
    flags = [
        ("absent", None),
        ("true", "true"),
        ("false", "false"),
        ("zero", "0"),
        ("one", "1"),
        ("empty_str", '""'),
        ("str_false", '"false"'),
        ("empty_arr", "[]"),
        ("table_empty", "{}"),
        ("table_nonempty", "{ enabled = 1 }"),
    ]

    cases: list[tuple[str, str]] = []
    # 1. Section shape combinations (absent, empty, non-table)
    sched_malformed = [
        ("s_absent", None),
        ("s_empty", "[scheduler]\n"),
        ("s_int", "scheduler = 1\n"),
        ("s_arr", "scheduler = []\n"),
        ("s_req_true", '[scheduler]\ncontest_state_required = true\nentry_intent = "enter"\npick_delivery = "dm"\n'),
        ("s_priv_res", '[scheduler]\nentry_intent = "research"\npick_delivery = "private"\n'),
    ]
    health_malformed = [
        ("h_absent", None),
        ("h_empty", "[health_checks]\n"),
        ("h_bool", "health_checks = false\n"),
        ("h_arr", "health_checks = []\n"),
        ("h_exp_true", "[health_checks]\ncontest_state_expected = true\n"),
    ]
    for (sn, sb), (hn, hb) in itertools.product(sched_malformed, health_malformed):
        # Top-level scalar keys in TOML must precede [table] headers!
        top_lines = []
        tbl_lines = []
        for piece in (sb, hb):
            if piece is None:
                continue
            if piece.startswith("["):
                tbl_lines.append(piece)
            else:
                top_lines.append(piece)
        cases.append((f"shape__{sn}__{hn}", "".join(top_lines + tbl_lines)))

    # 2. Focused cross-product over deliveries × intents × req_flag × exp_flag
    # Use all 19 deliveries × 5 intents × representative 6 req_flags × 4 exp_flags = 2,280 real CLI TOML cases
    req_subset = [f for f in flags if f[0] in ("absent", "true", "false", "zero", "str_false", "table_nonempty")]
    exp_subset = [f for f in flags if f[0] in ("absent", "true", "false", "one")]
    for (dn, db), (in_, ib), (rn, rv), (en, ev) in itertools.product(
        sched_deliveries, intents, req_subset, exp_subset
    ):
        lines = ["[scheduler]\n", db, ib]
        if rv is not None:
            lines.append(f"contest_state_required = {rv}\n")
        if ev is not None:
            lines.append("[health_checks]\n")
            lines.append(f"contest_state_expected = {ev}\n")
        cases.append((f"grid__{dn}__{in_}__req_{rn}__exp_{en}", "".join(lines)))

    return cases


def run_worker_mode(which: str, out_path: Path) -> None:
    install_network_audit_hook()
    target_src = BASE_SRC if which == "base" else BRANCH_SRC
    # Remove any repo src path before inserting target_src at index 0
    cleaned = [
        p for p in sys.path
        if Path(p if p else ".").resolve() != BRANCH_SRC.resolve()
    ]
    sys.path[:] = [str(target_src)] + cleaned

    import bts
    actual_bts = Path(bts.__file__).resolve()
    assert actual_bts.is_relative_to(target_src.resolve()), f"Wrong bts imported: {actual_bts} vs {target_src}"

    from click.testing import CliRunner
    from bts import cli, entry_intent, orchestrator, scheduler

    has_live_gate = hasattr(entry_intent, "live_gate_problem")
    runner = CliRunner()
    cases = build_toml_cases()
    results: dict[str, dict] = {}

    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "test.toml"
        for cid, body in cases:
            cfg_path.write_text(body)
            parsed = tomllib.loads(body)

            ic = entry_intent.check_entry_intent(parsed)
            gate_def = entry_intent.live_gate_problem(parsed) if has_live_gate else None
            gate_pub = entry_intent.live_gate_problem(parsed, delivery_override="public") if has_live_gate else None

            r_ei = runner.invoke(cli.cli, ["entry-intent", "--config", str(cfg_path)])

            sched_calls = []
            with patch.object(scheduler, "run_day", side_effect=lambda **kw: sched_calls.append(kw)):
                r_sched = runner.invoke(cli.cli, ["schedule", "--config", str(cfg_path), "--date", "2027-07-15"])

            dry_calls = []
            with patch.object(scheduler, "run_day", side_effect=lambda **kw: dry_calls.append(kw)):
                r_dry = runner.invoke(cli.cli, ["schedule", "--config", str(cfg_path), "--date", "2027-07-15", "--dry-run"])

            orch_calls = []
            def _orch_spy(*args, **kwargs):
                orch_calls.append((args, kwargs))
                return True

            with patch.object(orchestrator, "orchestrate", side_effect=_orch_spy):
                r_orch = runner.invoke(cli.cli, ["orchestrate", "--config", str(cfg_path), "--date", "2027-07-15"])

            results[cid] = {
                "toml": body,
                "ic": [ic.intent, ic.delivery, list(ic.problems)],
                "gate_def": gate_def,
                "gate_pub": gate_pub,
                "ei": [r_ei.exit_code, r_ei.stdout, r_ei.stderr],
                "sched": [r_sched.exit_code, r_sched.stdout, r_sched.stderr, len(sched_calls)],
                "dry": [r_dry.exit_code, r_dry.stdout, r_dry.stderr, len(dry_calls)],
                "orch": [r_orch.exit_code, r_orch.stdout, r_orch.stderr, len(orch_calls)],
            }

    out_path.write_text(json.dumps(results))


def compare_base_and_branch(base_data: dict[str, dict], branch_data: dict[str, dict]) -> dict:
    assert set(base_data.keys()) == set(branch_data.keys())
    unexplained: list[dict] = []
    counts = {
        "total_toml_cases": len(base_data),
        "gate_def_triggered": 0,
        "gate_pub_triggered": 0,
        "check_entry_intent_changed": 0,
        "ei_cli_changed": 0,
        "ei_former_pass_now_fail": 0,
        "sched_changed": 0,
        "dry_changed": 0,
        "orch_changed": 0,
    }

    for cid, b0 in base_data.items():
        b1 = branch_data[cid]
        cfg = tomllib.loads(b0["toml"])
        exp_gate_def = spec_live_gate_problem(cfg)
        exp_gate_pub = spec_live_gate_problem(cfg, delivery_override="public")

        if b1["gate_def"] != exp_gate_def:
            unexplained.append({"cid": cid, "where": "gate_def", "got": b1["gate_def"], "exp": exp_gate_def})
        if b1["gate_pub"] != exp_gate_pub:
            unexplained.append({"cid": cid, "where": "gate_pub", "got": b1["gate_pub"], "exp": exp_gate_pub})

        if exp_gate_def is not None:
            counts["gate_def_triggered"] += 1
        if exp_gate_pub is not None:
            counts["gate_pub_triggered"] += 1

        # Row 2: check_entry_intent & bts entry-intent
        sched_is_table = isinstance(cfg.get("scheduler", {}), dict)
        if not sched_is_table:
            exp_ic = b0["ic"]
        else:
            exp_probs = list(b0["ic"][2]) + ([exp_gate_def] if exp_gate_def is not None else [])
            exp_ic = [b0["ic"][0], b0["ic"][1], exp_probs]
        if b1["ic"] != exp_ic:
            unexplained.append({"cid": cid, "where": "check_entry_intent", "base": b0["ic"], "branch": b1["ic"], "exp": exp_ic})
        if b1["ic"] != b0["ic"]:
            counts["check_entry_intent_changed"] += 1

        if exp_ic[2]:
            exp_ei = [1, "", "".join(f"entry-intent: {p}\n" for p in exp_ic[2])]
        else:
            exp_ei = [0, f"{exp_ic[0]}\n", ""]
        if b1["ei"] != exp_ei:
            unexplained.append({"cid": cid, "where": "bts entry-intent", "base": b0["ei"], "branch": b1["ei"], "exp": exp_ei})
        if b1["ei"] != b0["ei"]:
            counts["ei_cli_changed"] += 1
            if b0["ei"][0] == 0 and b1["ei"][0] == 1:
                counts["ei_former_pass_now_fail"] += 1

        # Row 3: bts schedule (non-dry)
        if exp_gate_def is not None:
            exp_sched = [2, "", f"bts schedule: refusing to start: {exp_gate_def}\n", 0]
        else:
            exp_sched = b0["sched"]
        if b1["sched"] != exp_sched:
            unexplained.append({"cid": cid, "where": "bts schedule", "base": b0["sched"], "branch": b1["sched"], "exp": exp_sched})
        if b1["sched"] != b0["sched"]:
            counts["sched_changed"] += 1

        # Row 4: bts schedule --dry-run
        if exp_gate_def is not None:
            exp_dry = [
                b0["dry"][0],
                b0["dry"][1],
                f"bts schedule: WARNING (a real run would refuse to start): {exp_gate_def}\n" + b0["dry"][2],
                b0["dry"][3],
            ]
        else:
            exp_dry = b0["dry"]
        if b1["dry"] != exp_dry:
            unexplained.append({"cid": cid, "where": "bts schedule --dry-run", "base": b0["dry"], "branch": b1["dry"], "exp": exp_dry})
        if b1["dry"] != b0["dry"]:
            counts["dry_changed"] += 1

        # Row 5: bts orchestrate
        if exp_gate_pub is not None:
            exp_orch = [2, "", f"bts orchestrate: refusing to run: {exp_gate_pub}\n", 0]
        else:
            exp_orch = b0["orch"]
        if b1["orch"] != exp_orch:
            unexplained.append({"cid": cid, "where": "bts orchestrate", "base": b0["orch"], "branch": b1["orch"], "exp": exp_orch})
        if b1["orch"] != b0["orch"]:
            counts["orch_changed"] += 1

    counts["unexplained_differences"] = unexplained
    return counts


def main() -> None:
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        run_worker_mode(sys.argv[2], Path(sys.argv[3]))
        return

    install_network_audit_hook()
    sys.path.insert(0, str(BRANCH_SRC))
    t0 = time.monotonic()
    in_mem = run_exhaustive_in_memory_spec_check()
    t1 = time.monotonic()

    base_out = REVIEW_DIR / "matrix_base_raw.json"
    branch_out = REVIEW_DIR / "matrix_branch_raw.json"
    env = dict(os.environ, TZ="America/New_York", PYTHONDONTWRITEBYTECODE="1")

    for which, out_file in (("base", base_out), ("branch", branch_out)):
        proc = subprocess.run(
            [sys.executable, str(Path(__file__)), "--worker", which, str(out_file)],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Worker {which} failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}")

    comparison = compare_base_and_branch(
        json.loads(base_out.read_text()),
        json.loads(branch_out.read_text()),
    )
    t2 = time.monotonic()

    summary = {
        "in_memory_spec_check": in_mem,
        "in_memory_elapsed_sec": round(t1 - t0, 2),
        "cli_toml_comparison": comparison,
        "cli_toml_elapsed_sec": round(t2 - t1, 2),
    }
    summary_path = REVIEW_DIR / "matrix_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    assert not in_mem["counterexamples"], f"Found counterexamples: {in_mem['counterexamples']}"
    assert not comparison["unexplained_differences"], f"Found unexplained diffs: {comparison['unexplained_differences']}"


if __name__ == "__main__":
    main()
