"""Declared contest-entry intent, `[scheduler].entry_intent` (C1 rank-2 watchdog plan P4; registration R4).

The key is exactly "enter" or "research". There is no default and nothing is inferred: a missing, malformed or
unsupported value is a problem, never a fallback.
- `enter` expects recommendation delivery by DM or the public feed, plus an active entry checker
  (`check-pick-entered` in cron).
- `research` expects private delivery and no entry checker.

The delivery mode is the scheduler's own resolution (`_pick_delivery_mode`: its key precedence, aliases and the
checklist C2 refusal), so this check cannot disagree with what the scheduler would actually do. It only reports; it
changes nothing. `cron-setup-hetzner.sh` installs the entry cron only for an agreeing `enter`, and refuses to install
anything on a problem.
"""
from dataclasses import dataclass

INTENTS = ("enter", "research")
EXPECTED_DELIVERY = {"enter": ("dm", "public"), "research": ("private",)}


@dataclass(frozen=True)
class IntentCheck:
    intent: str | None          # the declared intent when it is exactly one of INTENTS, else None
    delivery: str | None        # the scheduler's effective delivery mode, None when the scheduler refuses the config
    problems: tuple[str, ...]   # intent, delivery and, when enabled, live startup problems in order

    @property
    def ok(self) -> bool:
        return not self.problems


def live_gate_problem(config: dict, *, delivery_override: str | None = None) -> str | None:
    """Report unsafe contest-state configuration for a new command invocation."""
    from bts.scheduler import _contest_state_required, _pick_delivery_mode

    sched = config.get("scheduler", {})
    if not isinstance(sched, dict):
        return "[scheduler] is not a table"
    if not isinstance(config.get("health_checks", {}), dict):
        return "[health_checks] is not a table"
    if _contest_state_required(config):
        return None

    delivery = delivery_override
    if delivery is None:
        try:
            delivery = _pick_delivery_mode(config)
        except ValueError:
            delivery = None

    conditions = []
    if delivery in ("dm", "public"):
        label = "scheduler delivery" if delivery_override is None else "delivery"
        conditions.append(f"{label} {delivery!r}")
    intent = sched.get("entry_intent")
    if isinstance(intent, str) and intent == "enter":
        conditions.append("scheduler.entry_intent = 'enter'")
    if not conditions:
        return None

    verb = "requires" if len(conditions) == 1 else "require"
    return (
        f"{' and '.join(conditions)} {verb} contest state. "
        "Optional reads can use a contest observation without the season-coverage check. "
        "Set contest_state_required = true under [scheduler] and "
        "contest_state_expected = true under [health_checks]."
    )


def check_entry_intent(config: dict, *, include_live_gate: bool = True) -> IntentCheck:
    from bts.scheduler import _pick_delivery_mode

    sched = config.get("scheduler", {})
    if not isinstance(sched, dict):
        return IntentCheck(None, None, ("[scheduler] is not a table",))

    problems = []
    raw = sched.get("entry_intent")
    intent = raw if isinstance(raw, str) and raw in INTENTS else None
    if "entry_intent" not in sched:
        problems.append('scheduler.entry_intent is missing: set it to "enter" or "research" (there is no default)')
    elif intent is None:
        problems.append(f'scheduler.entry_intent must be exactly "enter" or "research", got {raw!r}')

    try:
        delivery = _pick_delivery_mode(config)
    except ValueError as exc:
        delivery = None
        problems.append(f"the scheduler refuses this delivery config: {exc}")

    if intent is not None and delivery is not None and delivery not in EXPECTED_DELIVERY[intent]:
        problems.append(f"entry_intent {intent!r} expects delivery {' or '.join(EXPECTED_DELIVERY[intent])}, "
                        f"but the scheduler would deliver {delivery!r}")
    if include_live_gate:
        problem = live_gate_problem(config)
        if problem is not None:
            problems.append(problem)
    return IntentCheck(intent, delivery, tuple(problems))
