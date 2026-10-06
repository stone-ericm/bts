"""Watchdog notifications (registration R5, §4.3; W0 reviews r1 B3–B5, B7 and r2 N1–N4).

**Targets and episodes:** a target is (check, incident, ET date, selection). An alerting result (fault or checker
failure) opens an episode; a change of alerting state within an open episode is an escalation.

**Recovery is computed from the whole observation (r2 N2):** results are grouped by (check, ET date, selection).
- If any result in a group alerts, nothing in that group recovers.
- Otherwise, a `verified` result closes its open targets: an `incident`-carrying verified result closes only that
  incident's target; a verified result with no incident is an all-clear for the group.
- The recovery notice is built from the **recovered target's** own identity.
- Business recovery never touches a checker target (incident `checker:…`; r3 R3-1).

**Checker execution (r2 N1, r3 R3-1):** a checker failure targets (`<job>/<check_id>`, `checker:<job>/<check_id>`,
date, no selection). A later successful execution of the same registered invocation (`executed_ok`) closes every open
checker-failure target of that invocation, whatever its business result.

**Order (r3 R3-2):** every notice carries a persistent sequence number (`seq`). A target's unsent notices form a queue
in `seq` order, and a notice is delivered only after every earlier notice of its target was confirmed:
- a flush claims, per target, the longest due prefix of that queue, so nothing is claimed behind a live claim. A
  chain is always claimed whole, so all of a target's `sending` notices belong to one claim and share one lease
  (a partial re-claim of a stalled flusher's expired chain would leave its tail on the stale claim);
- it sends each target's claimed chain in order, and at the first failure, missing message id, exhausted budget or
  reached `max_sends` it releases the rest of that chain unattempted. Other targets continue.
- **Progress across targets (C2 r1 B1):** a flush claims the chains of at most `max_sends` targets, least-tried
  head first (then oldest), and `max_sends` bounds actual send attempts, not claimed notices. A target whose head
  fails therefore uses one attempt, and independent targets use the rest; a repeatedly failing head yields to fresh
  alerts.

**Delivery (r1 B4, B7; r2 N4):**
- `enqueue` always persists notices, even with no transport (queue-only).
- `flush` claims due notices under the short state lock. Each claim has a unique token and a UTC lease, and sends
  happen outside the lock.
- A completion is recorded only if the notice still carries that token. Only a returned non-empty message id marks
  a notice `sent`.
- A clock read earlier than a claim's start counts the lease as expired (a duplicate is possible; exactly-once
  delivery is not claimed).
- The flush is bounded by `max_sends` send attempts and by an **elapsed monotonic** budget, both checked before
  each send. Neither bound interrupts an in-flight send; the whole-job deadline is a deploy-gate item.

**Validation (r1 B5, r2 N3, r3 R3-3):** every container, field, type, timestamp, state, cross-reference and recomputed
key is validated on load, and so is the lifecycle the protocol implies (it never prunes a notice):
- the notice `seq` values are exactly 1 … `seq`;
- each target's notices cover its episodes 1 … `episode`, in `seq` order across episodes;
- each episode opens with an alerting notice, and a `recovered` notice, if any, is its last;
- every earlier episode was recovered, and the current one was recovered exactly when the target is closed;
- an open target has a notice for its current state;
- by `seq`, a target's notices are `sent`, then `sending`, then `pending` (delivery never overtook a predecessor);
- a target's `sending` notices share one claim time and one lease (one claim; C2 r1 B3);
- delivery metadata is consistent (C2 r1 B2): an attempt count above zero exactly when `last_attempt_at` is
  recorded; a `sent` notice has an attempt, its recipient and message id, and no error; an unsent notice has no
  recipient and no message id.

Anything invalid raises `NotifyStateError` and leaves the file untouched.
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from datetime import date, datetime, timedelta, timezone

from bts.watchdog.result import ALERTING, CHECKER_PREFIX, CheckResult, Status

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
            if (n["attempts"] > 0) != ("last_attempt_at" in n):
                _bad(f"notice {k}: attempts and last_attempt_at inconsistent")
            if n["status"] == "sent":
                if n["attempts"] < 1 or not n["recipient"] or n["last_error"] is not None:
                    _bad(f"notice {k}: a sent notice without its completed attempt, recipient, or with an error")
            elif n["recipient"] is not None:
                _bad(f"notice {k}: an unsent notice with a recipient")
            if n["status"] == "sending":
                if set(n) & claim != claim or not (isinstance(n["claim_token"], str) and n["claim_token"]):
                    _bad(f"notice {k}: claim")
                _parse_utc(n["claimed_at"])
                _parse_utc(n["lease_until"])
            elif set(n) & claim:
                _bad(f"notice {k}: claim fields outside sending")
        _validate_lifecycle(st, seqs)
    except NotifyStateError:
        raise
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise NotifyStateError(f"malformed notification state: {type(exc).__name__}: {exc}") from None
    return st


_RANK = {"sent": 0, "sending": 1, "pending": 2}


def _validate_lifecycle(st, seqs) -> None:
    """r3 R3-3: the reverse direction. Every target's episodes and states have the notices the protocol queued."""
    if sorted(seqs) != list(range(1, st["seq"] + 1)):
        _bad("notice seq values are not exactly 1..seq")
    by_target: dict = {}
    for n in st["notices"].values():
        by_target.setdefault(n["target"], []).append(n)
    for tk, t in st["targets"].items():
        ns = sorted(by_target.get(tk, []), key=lambda n: n["seq"])
        eps = [n["episode"] for n in ns]
        if eps != sorted(eps):
            _bad(f"target {tk}: episodes out of seq order")
        if sorted(set(eps)) != list(range(1, t["episode"] + 1)):
            _bad(f"target {tk}: notices do not cover episodes 1..{t['episode']}")
        for e in range(1, t["episode"] + 1):
            states = [n["state"] for n in ns if n["episode"] == e]
            if states[0] not in ALERT_STATES:
                _bad(f"target {tk}: episode {e} does not open with an alert")
            if "recovered" in states[:-1]:
                _bad(f"target {tk}: episode {e} has a notice after its recovery")
            recovered = states[-1] == "recovered"
            if e < t["episode"] and not recovered:
                _bad(f"target {tk}: episode {e} was never recovered")
            if e == t["episode"] and recovered == t["open"]:
                _bad(f"target {tk}: the current episode's recovery does not match the target")
            if e == t["episode"] and t["open"] and t["state"] not in states:
                _bad(f"target {tk}: no notice for its current state {t['state']}")
        ranks = [_RANK[n["status"]] for n in ns]
        if ranks != sorted(ranks):
            _bad(f"target {tk}: a notice was delivered or claimed ahead of an earlier one")
        claims = {(_parse_utc(n["claimed_at"]), _parse_utc(n["lease_until"])) for n in ns if n["status"] == "sending"}
        if len(claims) > 1:
            _bad(f"target {tk}: its sending notices are split across claims")


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


def _is_checker(t: dict) -> bool:
    return isinstance(t["incident"], str) and t["incident"].startswith(CHECKER_PREFIX)


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
                                and t["selection"] == selection and not _is_checker(t)
                                and (all_clear or t["incident"] in incidents)):
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

    def _claimable(self, st, now: datetime) -> list:
        """Whole chains, one per target, for at most `max_sends` targets (r3 R3-2; C2 r1 B1).

        A chain is the longest due prefix of a target's unsent notices in `seq` order, so a target whose head is
        live-claimed contributes nothing. Chains are ordered least-tried head first, then oldest head, so a head
        that keeps failing yields to fresh alerts. Each claimed target needs at least one attempt, so claiming more
        than `max_sends` targets could never be sent in this flush."""
        queues: dict = {}
        for k, n in st["notices"].items():
            if n["status"] != "sent":
                queues.setdefault(n["target"], []).append(k)
        chains = []
        for ks in queues.values():
            chain = []
            for k in sorted(ks, key=lambda k: st["notices"][k]["seq"]):
                if not self._due(st["notices"][k], now):
                    break
                chain.append(k)
            if chain:
                chains.append(chain)
        chains.sort(key=lambda c: (st["notices"][c[0]]["attempts"], st["notices"][c[0]]["seq"]))
        return chains[: self.max_sends]

    def flush(self) -> dict:
        """Send due notices, each target's in order (bounded). Returns {"sent", "failed", "skipped_budget", "held"}."""
        report = {"sent": 0, "failed": 0, "skipped_budget": 0, "held": 0}
        if self.send is None or not self.recipient:
            return report                                # queue-only: notices stay pending
        with state_lock(self.root):
            st = load_state(self.root)
            now = _utc(self.clock.now())
            claims = {}
            for k in (k for chain in self._claimable(st, now) for k in chain):
                token = uuid.uuid4().hex
                st["notices"][k].update(status="sending", claim_token=token, claimed_at=now.isoformat(),
                                        lease_until=(now + self.lease).isoformat())
                claims[k] = (token, st["notices"][k]["text"], st["notices"][k]["target"])
            if claims:
                save_state(self.root, st)
        started = self.monotonic()
        outcomes, blocked, attempts = {}, set(), 0
        for k, (token, text, tk) in claims.items():      # network, outside the lock; chain by chain, each in seq order
            if tk in blocked:                            # an earlier notice of this target was not delivered
                outcomes[k] = (token, None, "an earlier notice of this target was not delivered", False)
                report["held"] += 1
                continue
            if attempts >= self.max_sends:
                outcomes[k] = (token, None, "max_sends reached before send", False)
                report["skipped_budget"] += 1
                blocked.add(tk)
                continue
            if self.monotonic() - started > self.budget_s:
                outcomes[k] = (token, None, "budget exhausted before send", False)
                report["skipped_budget"] += 1
                blocked.add(tk)
                continue
            attempts += 1
            try:
                mid = self.send(self.recipient, text)
            except Exception as exc:  # noqa: BLE001 - a failed send stays pending
                outcomes[k] = (token, None, type(exc).__name__, True)
                blocked.add(tk)
            else:
                ok = isinstance(mid, str) and mid != ""
                outcomes[k] = (token, mid if ok else None, None if ok else "no message id", True)
                if not ok:
                    blocked.add(tk)
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
