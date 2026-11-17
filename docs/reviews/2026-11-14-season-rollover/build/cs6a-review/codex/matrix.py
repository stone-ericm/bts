"""Exercise the real TOML loader and CLI against an independent launch oracle."""

import local_guard
from collections import Counter
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import sys
import tomllib

from click.testing import CliRunner

HERE = Path(__file__).resolve().parent
REV = sys.argv[1]
sys.path.insert(0, str((HERE / "base" if REV == "base" else HERE.parents[1]) / "src"))
from bts import cli, entry_intent, orchestrator, scheduler

runner = CliRunner()
DATE = "2027-07-15"
MISSING = object()
FLAG = (MISSING, True, False, 0, 1, "", "false", [], {}, {"present": 0})
INTENT = (MISSING, "enter", "research", "play", "ENTER", "Research", " enter ", "",
          False, 1, [], {})
SHAPES = ("present", "absent", "empty", "scalar", "array")
MODES = [
    ({}, "public"),
    *[({"pick_delivery": alias}, mode) for mode, aliases in (
        ("public", ("public", "bluesky", "feed", "post", " PuBLic ")),
        ("dm", ("dm", "bluesky_dm", "direct_message", "bluesky-dm", "direct-message",
                "  DiRect-Message \t")),
        ("private", ("private", "none", "off", "local", " LoCaL ")),
        (None, ("carrier-pigeon", "", 3, False, [], {}, 3.5)),
    ) for alias in aliases],
    *[({"posting_mode": alias}, mode) for alias, mode in (
        ("feed", "public"), ("bluesky-dm", "dm"), ("local", "private"), (False, None)
    )],
    ({"private_mode": False}, "public"),
    ({"private_mode": True}, "private"),
    ({"private_mode": "false"}, "private"),
    ({"private_mode": []}, "public"),
    ({"shadow_mode": False}, None),
    ({"shadow_mode": True}, None),
    ({"shadow_mode": []}, None),
    ({"shadow_mode": False, "private_mode": True}, "private"),
    ({"shadow_mode": False, "private_mode": 0}, None),
    ({"shadow_mode": False, "posting_mode": "dm"}, "dm"),
    ({"pick_delivery": "feed", "posting_mode": "dm", "private_mode": True}, "public"),
    ({"pick_delivery": "off", "posting_mode": "dm", "private_mode": False}, "private"),
    ({"pick_delivery": "dm", "posting_mode": "private", "shadow_mode": False,
      "private_mode": True}, "dm"),
    ({"pick_delivery": 0, "posting_mode": "dm", "private_mode": True}, None),
]
OVERRIDES = (None, "public", "dm", "private", "bluesky", " PUBLIC ", "", False, [], {})
SUFFIX = (
    "Optional reads can use a contest observation without the season-coverage check. "
    "Set contest_state_required = true under [scheduler] and "
    "contest_state_expected = true under [health_checks]."
)


def toml_value(value):
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{key} = {toml_value(val)}" for key, val in value.items()) + " }"
    return json.dumps(value)


def sections(shape, value, *, malformed):
    return {"present": value, "absent": MISSING, "empty": {}, "scalar": malformed,
            "array": []}[shape]


def make_cases():
    shapes = product(SHAPES, repeat=2)
    emitted = set()
    for ss, hs in shapes:
        scheds = product(MODES, INTENT, FLAG) if ss == "present" else [(({}, "public"), MISSING, MISSING)]
        for (fields, delivery), intent, required in scheds:
            sched = dict(fields)
            if intent is not MISSING:
                sched["entry_intent"] = intent
            if required is not MISSING:
                sched["contest_state_required"] = required
            sched = sections(ss, sched, malformed="not a table")
            for expected in FLAG if hs == "present" else (MISSING,):
                health = {} if expected is MISSING else {"contest_state_expected": expected}
                health = sections(hs, health, malformed=3)
                body = "".join(
                    f"{key} = {toml_value(value)}\n"
                    for key, value in (("scheduler", sched), ("health_checks", health))
                    if value is not MISSING
                )
                if body in emitted:
                    continue
                emitted.add(body)
                yield body, delivery, intent


def gate(config, delivery, intent, override=None):
    bad = next((f"[{k}] is not a table" for k in ("scheduler", "health_checks")
                if k in config and type(config[k]) is not dict), None)
    if bad:
        return bad
    if (config.get("scheduler", {}).get("contest_state_required") or
            config.get("health_checks", {}).get("contest_state_expected")):
        return None
    mode = delivery if override is None else override
    conditions = []
    if mode in ("public", "dm"):
        conditions.append(f"{'scheduler delivery' if override is None else 'delivery'} {mode!r}")
    if intent == "enter":
        conditions.append("scheduler.entry_intent = 'enter'")
    if conditions:
        return f"{' and '.join(conditions)} {'requires' if len(conditions) == 1 else 'require'} contest state. {SUFFIX}"
    return None


def old_problems(config, delivery, raw):
    sched = config.get("scheduler", {})
    if type(sched) is not dict:
        return ("[scheduler] is not a table",)
    problems = []
    if raw is MISSING:
        problems.append('scheduler.entry_intent is missing: set it to "enter" or "research" (there is no default)')
    elif raw not in ("enter", "research"):
        problems.append(f'scheduler.entry_intent must be exactly "enter" or "research", got {raw!r}')
    if delivery is None:
        if "shadow_mode" in sched and not sched.get("private_mode") and not (
            "pick_delivery" in sched or "posting_mode" in sched
        ):
            problems.append("the scheduler refuses this delivery config: legacy scheduler.shadow_mode is not supported: "
                            "set scheduler.pick_delivery (public, dm or private)")
        else:
            problems.append("the scheduler refuses this delivery config: scheduler.pick_delivery must be one of: public, dm, private")
    elif raw == "enter" and delivery == "private":
        problems.append("entry_intent 'enter' expects delivery dm or public, but the scheduler would deliver 'private'")
    elif raw == "research" and delivery != "private":
        problems.append(f"entry_intent 'research' expects delivery private, but the scheduler would deliver {delivery!r}")
    return tuple(problems)


calls = []
scheduler.run_day = lambda **kwargs: calls.append("dry" if kwargs["dry_run"] else "live")


def work(*args, **kwargs):
    calls.append("orchestrate")
    return True


orchestrator.orchestrate = work


def invoke(args, config_path):
    calls.clear()
    command, *remaining = args
    result = runner.invoke(cli.cli, [command, "--config", str(config_path), *remaining])
    return result.exit_code, result.stdout, result.stderr, list(calls)


def main():
    counts = Counter()
    defects = []
    dest = HERE / f"matrix-{REV}.jsonl"
    with TemporaryDirectory(dir=HERE) as td, dest.open("w") as log:
        cfg_path = Path(td) / "input.toml"
        for body, mode, raw in make_cases():
            cfg_path.write_text(body)
            config = tomllib.loads(body)
            gate_schedule = gate(config, mode, raw)
            gate_orch = gate(config, mode, raw, "public")
            old = old_problems(config, mode, raw)
            new = old if old == ("[scheduler] is not a table",) else old + ((gate_schedule,) if gate_schedule else ())
            problems = new if REV == "head" else old
            observations = []
            for cmd, expected in (
                (("schedule", "--date", DATE),
                 (2, "", f"bts schedule: refusing to start: {gate_schedule}\n", [])
                 if REV == "head" and gate_schedule else (0, "", "", ["live"])),
                (("schedule", "--date", DATE, "--dry-run"),
                 (0, "", f"bts schedule: WARNING (a real run would refuse to start): {gate_schedule}\n"
                  if REV == "head" and gate_schedule else "", ["dry"])),
                (("orchestrate", "--date", DATE),
                 (2, "", f"bts orchestrate: refusing to run: {gate_orch}\n", [])
                 if REV == "head" and gate_orch else (0, "", "", ["orchestrate"])),
                (("entry-intent",),
                 (1, "", "".join(f"entry-intent: {p}\n" for p in problems), [])
                 if problems else (0, f"{raw}\n", "", [])),
            ):
                actual = invoke(cmd, cfg_path)
                observations.append(actual)
                if actual != expected:
                    defects.append((body, cmd, actual, expected))
            resolved = entry_intent.check_entry_intent(config)
            if resolved.problems != problems:
                defects.append((body, "problems", resolved.problems, problems))
            if REV == "head":
                if entry_intent.check_entry_intent(config, include_live_gate=False).problems != old:
                    defects.append((body, "opt-out"))
                for override in OVERRIDES:
                    expected = gate(config, mode, raw, override)
                    observed = entry_intent.live_gate_problem(config, delivery_override=override)
                    counts["predicate_calls"] += 1
                    if observed != expected:
                        defects.append((body, override, observed, expected))
            counts["files"] += 1
            counts["schedule changed"] += bool(gate_schedule)
            counts["orchestrate changed"] += bool(gate_orch)
            counts["entry-intent changed"] += old != new
            counts[f"accepted schedule {mode}"] += not gate_schedule
            log.write(json.dumps([body, observations]) + "\n")
    print(REV, dict(counts), f"mismatches={len(defects)}", "network attempts", local_guard.network)
    for defect in defects[:20]:
        print(repr(defect))
    return int(bool(defects or local_guard.network))


if __name__ == "__main__":
    raise SystemExit(main())
