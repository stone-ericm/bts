"""Watchdog notifications (registration R5, §4.3; W0 reviews r1 B3–B5, B7 and r2 N1–N4).

**Targets and episodes:** a target is (check, incident, ET date, selection). An alerting result (fault or checker
failure) opens an episode; a change of alerting state within an open episode is an escalation.

**Recovery is computed from the whole observation (r2 N2):** results are grouped by (check, ET date, selection).
- If any result in a group alerts, nothing in that group recovers.
- Otherwise, a `verified` result closes its open targets: an `incident`-carrying verified result closes only that
  incident's target; a verified result with no incident is an all-clear for the group.
- The recovery notice is built from the **recovered target's** own identity.

**Checker execution (r2 N1):** a checker failure targets (name, `checker:<name>`, date, no selection). A later
successful execution of the same registered name (`executed_ok`) closes every open checker-failure target of that
name, whatever its business result.

**Order:** every notice carries a persistent sequence number (`seq`), and flushes go in that causal order.

**Delivery (r1 B4, B7; r2 N4):**
- `enqueue` always persists notices, even with no transport (queue-only).
- `flush` claims due notices under the short state lock. Each claim has a unique token and a UTC lease, and sends
  happen outside the lock.
- A completion is recorded only if the notice still carries that token. Only a returned non-empty message id marks
  a notice `sent`.
- A clock read earlier than a claim's start counts the lease as expired (a duplicate is possible; exactly-once
  delivery is not claimed).
- The flush is bounded by `max_sends` and by an **elapsed monotonic** budget, checked before each send. Neither
  bound interrupts an in-flight send; the whole-job deadline is a deploy-gate item.

**Validation (r1 B5, r2 N3):** every container, field, type, timestamp, state, cross-reference and recomputed key
is validated on load. Anything invalid raises `NotifyStateError` and leaves the file untouched.
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from datetime import date, datetime, timedelta, timezone

from bts.watchdog.result import ALERTING, CheckResult, Status

LEASE = timedelta(minutes=10)
BUDGET_S = 90.0
MAX_SENDS = 20
STATE = ("notify", "state.json")
LOCK = ("notify", ".lock")
DELIVERY = ("pending", "sending", "sent")
ALERT_STATES = tuple(s.value for s in ALERTING)
NOTICE_STATES = ALERT_STATES + ("recovered",)


class NotifyStateError(RuntimeError):
    pass


def _h(*parts) -> str:
    return hashlib.sha256(json.dumps(parts).encode()).hexdigest()


def target_key(check: str, incident: str | None, et_date: str, selection: str | None) -> str:
    return _h("target", check, incident, et_date, selection)


def notice_key(target: str, episode: int, state: str) -> str:
    return _h("notice", target, episode, state)


def _utc(t: datetime) -> datetime:
    if t.tzinfo is None:
        raise ValueError("timezone-aware time required")
    return t.astimezone(timezone.utc)


def _parse_utc(s) -> datetime:
    if not isinstance(s, str):
        raise ValueError("not a timestamp string")
    t = datetime.fromisoformat(s)
    if t.tzinfo is None:
        raise ValueError("naive time")
    return t.astimezone(timezone.utc)


def _int(v, lo=0) -> bool:
    return type(v) is int and v >= lo


def _opt_str(v) -> bool:
    return v is None or (isinstance(v, str) and v != "")


def _bad(msg):
    raise NotifyStateError(msg)


def _validate(st) -> dict:
    """Every container, field and relationship; raises NotifyStateError on anything else."""
    try:
        if not isinstance(st, dict) or set(st) != {"version", "seq", "targets", "notices"} or st["version"] != 1:
            _bad("not a version-1 {seq, targets, notices} object")
        if not _int(st["seq"]) or not isinstance(st["targets"], dict) or not isinstance(st["notices"], dict):
            _bad("bad seq or containers")
        for k, t in st["targets"].items():
            need = {"check", "incident", "et_date", "selection", "episode", "open", "state", "opened_at"}
            if not isinstance(t, dict) or not need <= set(t) or not set(t) <= need | {"closed_at"}:
                _bad(f"target {k}: fields")
            if not (isinstance(t["check"], str) and t["check"] and _opt_str(t["incident"])
                    and _opt_str(t["selection"]) and isinstance(t["et_date"], str) and _int(t["episode"], 1)
                    and isinstance(t["open"], bool) and t["state"] in NOTICE_STATES):
                _bad(f"target {k}: types")
            date.fromisoformat(t["et_date"])
            _parse_utc(t["opened_at"])
            if t["open"] != (t["state"] in ALERT_STATES):
                _bad(f"target {k}: open/state inconsistent")
            if (not t["open"]) != ("closed_at" in t):
                _bad(f"target {k}: closed_at inconsistent")
            if "closed_at" in t:
                _parse_utc(t["closed_at"])
            if target_key(t["check"], t["incident"], t["et_date"], t["selection"]) != k:
                _bad(f"target {k}: key does not match its identity")
        seqs = set()
        for k, n in st["notices"].items():
            need = {"target", "episode", "state", "text", "status", "attempts", "queued_at", "seq", "message_id",
                    "recipient", "last_error"}
            claim = {"claim_token", "claimed_at", "lease_until"}
            if not isinstance(n, dict) or not need <= set(n) or not set(n) <= need | claim | {"last_attempt_at"}:
                _bad(f"notice {k}: fields")
            t = st["targets"].get(n["target"])
            if t is None:
                _bad(f"notice {k}: unknown target")
            if not (_int(n["episode"], 1) and n["episode"] <= t["episode"] and n["state"] in NOTICE_STATES
                    and isinstance(n["text"], str) and n["text"] and n["status"] in DELIVERY and _int(n["attempts"])
                    and _int(n["seq"], 1) and _opt_str(n["recipient"]) and _opt_str(n["last_error"])):
                _bad(f"notice {k}: types")
            if n["seq"] in seqs or n["seq"] > st["seq"]:
                _bad(f"notice {k}: seq")
            seqs.add(n["seq"])
            _parse_utc(n["queued_at"])
            if "last_attempt_at" in n:
                _parse_utc(n["last_attempt_at"])
            if notice_key(n["target"], n["episode"], n["state"]) != k:
                _bad(f"notice {k}: key does not match its identity")
            if (n["status"] == "sent") != (isinstance(n["message_id"], str) and n["message_id"] != ""):
                _bad(f"notice {k}: sent/message_id inconsistent")
            if n["status"] == "sending":
                if set(n) & claim != claim or not (isinstance(n["claim_token"], str) and n["claim_token"]):
                    _bad(f"notice {k}: claim")
                _parse_utc(n["claimed_at"])
                _parse_utc(n["lease_until"])
            elif set(n) & claim:
                _bad(f"notice {k}: claim fields outside sending")
    except NotifyStateError:
        raise
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise NotifyStateError(f"malformed notification state: {type(exc).__name__}: {exc}") from None
    return st


def empty_state() -> dict:
    return {"version": 1, "seq": 0, "targets": {}, "notices": {}}


def load_state(root) -> dict:
    raw = root.read_bytes(STATE)
    if raw is None:
        return empty_state()
    try:
        st = json.loads(raw)
    except ValueError as exc:
        raise NotifyStateError(f"notification state is unreadable: {exc}") from exc
    return _validate(st)


def save_state(root, st: dict) -> None:
    root.write_atomic(STATE, (json.dumps(_validate(st), indent=1, sort_keys=True) + "\n").encode())


def state_lock(root):
    return root.lock(LOCK, blocking=True)


def _text(t: dict, kind: str, episode: int, state: str, detail: str) -> str:
    tag = f" [{t['incident']}]" if t["incident"] else ""
    sel = f" ({t['selection']})" if t["selection"] else ""
    head = {"open": state, "escalation": f"escalated to {state}", "recovered": "RECOVERED"}[kind]
    return f"BTS watchdog {head}{tag}: {t['check']} on {t['et_date']}{sel}, episode {episode}: {detail}"[:1000]


class Notifier:
    def __init__(self, root, *, recipient: str | None, send, clock, lease: timedelta = LEASE,
                 budget_s: float = BUDGET_S, max_sends: int = MAX_SENDS, monotonic=time.monotonic):
        self.root, self.recipient, self.send, self.clock = root, recipient, send, clock
        self.lease, self.budget_s, self.max_sends, self.monotonic = lease, budget_s, max_sends, monotonic

    # ---- episodes -------------------------------------------------------------------------------------------------
    def enqueue(self, results, executed_ok=()) -> int:
        """Apply one observation (all results of a run) and the names of checks that executed successfully."""
        new = 0
        with state_lock(self.root):
            st = load_state(self.root)
            now = _utc(self.clock.now()).isoformat()
            groups: dict = {}
            for r in results:
                groups.setdefault((r.check, r.et_date, r.selection), []).append(r)
            for (check, et_date, selection), rs in groups.items():
                alerting = [r for r in rs if r.status in ALERTING]
                for r in alerting:
                    new += self._alert(st, r, now)
                verified = [r for r in rs if r.status is Status.VERIFIED]
                if verified and not alerting:
                    incidents = {r.incident for r in verified}
                    all_clear = None in incidents
                    for tk, t in st["targets"].items():
                        if (t["open"] and t["check"] == check and t["et_date"] == et_date
                                and t["selection"] == selection and (all_clear or t["incident"] in incidents)):
                            new += self._recover(st, tk, t, now, "verified")
            for name in executed_ok:
                for tk, t in st["targets"].items():
                    if t["open"] and t["check"] == name and t["incident"] == f"checker:{name}":
                        new += self._recover(st, tk, t, now, "the check executes again")
            save_state(self.root, st)
        return new

    def _alert(self, st, r: CheckResult, now) -> int:
        tk = target_key(r.check, r.incident, r.et_date, r.selection)
        t = st["targets"].get(tk)
        if t is None or not t["open"]:
            ep = (t["episode"] + 1) if t else 1
            t = {"check": r.check, "incident": r.incident, "et_date": r.et_date, "selection": r.selection,
                 "episode": ep, "open": True, "state": r.status.value, "opened_at": now}
            st["targets"][tk] = t
            return self._queue(st, tk, ep, r.status.value, _text(t, "open", ep, r.status.value, r.detail), now)
        if t["state"] != r.status.value:
            t["state"] = r.status.value
            return self._queue(st, tk, t["episode"], r.status.value,
                               _text(t, "escalation", t["episode"], r.status.value, r.detail), now)
        return 0

    def _recover(self, st, tk, t, now, why) -> int:
        t.update(open=False, state="recovered", closed_at=now)
        return self._queue(st, tk, t["episode"], "recovered", _text(t, "recovered", t["episode"], "recovered", why), now)

    @staticmethod
    def _queue(st, tk, episode, state, text, now) -> int:
        nk = notice_key(tk, episode, state)
        if nk in st["notices"]:
            return 0
        st["seq"] += 1
        st["notices"][nk] = {"target": tk, "episode": episode, "state": state, "text": text, "status": "pending",
                             "attempts": 0, "queued_at": now, "seq": st["seq"], "message_id": None,
                             "recipient": None, "last_error": None}
        return 1

    # ---- delivery -------------------------------------------------------------------------------------------------
    @staticmethod
    def _due(n, now: datetime) -> bool:
        if n["status"] == "pending":
            return True
        if n["status"] != "sending":
            return False
        claimed, until = _parse_utc(n["claimed_at"]), _parse_utc(n["lease_until"])
        return now >= until or now < claimed          # expired, or the clock rolled back past the claim

    def flush(self) -> dict:
        """Send due notices in causal order (bounded). Returns {"sent", "failed", "skipped_budget"}."""
        report = {"sent": 0, "failed": 0, "skipped_budget": 0}
        if self.send is None or not self.recipient:
            return report                                # queue-only: notices stay pending
        with state_lock(self.root):
            st = load_state(self.root)
            now = _utc(self.clock.now())
            due = sorted((k for k, n in st["notices"].items() if self._due(n, now)),
                         key=lambda k: st["notices"][k]["seq"])[: self.max_sends]
            claims = {}
            for k in due:
                token = uuid.uuid4().hex
                st["notices"][k].update(status="sending", claim_token=token, claimed_at=now.isoformat(),
                                        lease_until=(now + self.lease).isoformat())
                claims[k] = (token, st["notices"][k]["text"])
            if claims:
                save_state(self.root, st)
        started = self.monotonic()
        outcomes = {}
        for k, (token, text) in claims.items():          # network, outside the lock
            if self.monotonic() - started > self.budget_s:
                outcomes[k] = (token, None, "budget exhausted before send", False)
                report["skipped_budget"] += 1
                continue
            try:
                mid = self.send(self.recipient, text)
            except Exception as exc:  # noqa: BLE001 - a failed send stays pending
                outcomes[k] = (token, None, type(exc).__name__, True)
            else:
                ok = isinstance(mid, str) and mid != ""
                outcomes[k] = (token, mid if ok else None, None if ok else "no message id", True)
        if not outcomes:
            return report
        with state_lock(self.root):
            st = load_state(self.root)
            done = _utc(self.clock.now()).isoformat()
            for k, (token, mid, err, attempted) in outcomes.items():
                n = st["notices"].get(k)
                if n is None or n.get("claim_token") != token:
                    continue                             # a newer claim owns this notice: never overwrite it
                for f in ("claim_token", "claimed_at", "lease_until"):
                    n.pop(f, None)
                if attempted:
                    n["attempts"] += 1
                    n["last_attempt_at"] = done
                if mid:
                    n.update(status="sent", message_id=mid, recipient=self.recipient, last_error=None)
                    report["sent"] += 1
                else:
                    n.update(status="pending", last_error=err)
                    report["failed"] += attempted
            save_state(self.root, st)
        return report
