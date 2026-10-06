"""Watchdog notifications (registration R5, §4.3; W0 review r1 B3, B4, B5, B7).

**Episodes (B3):** results are grouped by incident target, (check, incident, ET date, selection).
- **Opening and escalation:** an alerting result (fault or checker failure) opens an episode with a fresh number.
  A change of alerting state within an open episode is an escalation.
- **Recovery:** a later `verified` result for the same (check, ET date, selection) closes every open episode of that
  target and queues a recovery notice.
- **Recurrence:** a fault after recovery opens episode n+1.

Each notice's dedup key binds (target, episode, state), so repeated polling of one episode stays silent across
restarts. `pending` and `unverifiable` change no episode.

**Delivery (B4, B7):**
- **Queueing:** `enqueue` always persists notices, even with no transport (`--no-send`) and no recipient.
- **Claims:** `flush` claims due notices under the short state lock. Each claim carries a unique token and a UTC
  lease, and the send happens outside the lock.
- **Completion:** a result is recorded only if the notice still carries that claim's token, so a stale sender can
  never overwrite a newer outcome. Only a returned non-empty message id marks a notice `sent`; failures and
  id-less returns stay `pending`.
- **Clock rollback:** if the clock reads earlier than a claim's start, the lease counts as expired, so a duplicate
  retry is possible (documented; exactly-once delivery is not claimed).
- **Bounds:** at most `max_sends` sends per flush, and no new send starts after `budget` has elapsed.

**Validation (B5):** every stored entry is validated on load. A malformed or unknown state raises `NotifyStateError`
and leaves the file untouched.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timedelta, timezone

from bts.watchdog.result import ALERTING, CheckResult, Status

LEASE = timedelta(minutes=10)
BUDGET = timedelta(seconds=90)
MAX_SENDS = 20
STATE = ("notify", "state.json")
LOCK = ("notify", ".lock")
NOTICE_STATUSES = {"pending", "sending", "sent"}


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
    t = datetime.fromisoformat(s)
    if t.tzinfo is None:
        raise ValueError("naive time")
    return t.astimezone(timezone.utc)


def _validate(st) -> dict:
    if not isinstance(st, dict) or set(st) != {"version", "targets", "notices"} or st["version"] != 1:
        raise NotifyStateError("notification state is not a version-1 {targets, notices} object")
    for k, t in st["targets"].items():
        ok = (isinstance(t, dict) and isinstance(t.get("episode"), int) and t["episode"] >= 1
              and isinstance(t.get("open"), bool) and t.get("state") in {s.value for s in ALERTING} | {"recovered"}
              and isinstance(t.get("check"), str) and isinstance(t.get("et_date"), str))
        if not ok:
            raise NotifyStateError(f"malformed target {k}")
    for k, n in st["notices"].items():
        ok = (isinstance(n, dict) and n.get("status") in NOTICE_STATUSES and isinstance(n.get("text"), str)
              and n.get("target") in st["targets"] and isinstance(n.get("attempts"), int))
        if ok and n["status"] == "sending":
            try:
                _parse_utc(n["claimed_at"])
                _parse_utc(n["lease_until"])
                ok = isinstance(n.get("claim_token"), str) and n["claim_token"] != ""
            except (KeyError, TypeError, ValueError):
                ok = False
        if ok and n["status"] == "sent":
            ok = isinstance(n.get("message_id"), str) and n["message_id"] != ""
        if not ok:
            raise NotifyStateError(f"malformed notice {k}")
    return st


def load_state(root) -> dict:
    raw = root.read_bytes(STATE)
    if raw is None:
        return {"version": 1, "targets": {}, "notices": {}}
    try:
        st = json.loads(raw)
    except ValueError as exc:
        raise NotifyStateError(f"notification state is unreadable: {exc}") from exc
    return _validate(st)


def save_state(root, st: dict) -> None:
    root.write_atomic(STATE, (json.dumps(_validate(st), indent=1, sort_keys=True) + "\n").encode())


def state_lock(root):
    return root.lock(LOCK, blocking=True)


def notice_text(r: CheckResult, kind: str, episode: int) -> str:
    tag = f" [{r.incident}]" if r.incident else ""
    sel = f" ({r.selection})" if r.selection else ""
    head = {"open": f"{r.status.value}", "escalation": f"escalated to {r.status.value}", "recovered": "RECOVERED"}[kind]
    return f"BTS watchdog {head}{tag}: {r.check} on {r.et_date}{sel}, episode {episode}: {r.detail}"[:1000]


class Notifier:
    def __init__(self, root, *, recipient: str | None, send, clock, lease: timedelta = LEASE,
                 budget: timedelta = BUDGET, max_sends: int = MAX_SENDS):
        self.root, self.recipient, self.send, self.clock = root, recipient, send, clock
        self.lease, self.budget, self.max_sends = lease, budget, max_sends

    # ---- episodes -------------------------------------------------------------------------------------------------
    def enqueue(self, results) -> int:
        """Apply results to the episodes and queue the resulting notices. Returns how many notices were new."""
        new = 0
        with state_lock(self.root):
            st = load_state(self.root)
            now = _utc(self.clock.now()).isoformat()
            for r in results:
                if r.status in ALERTING:
                    tk = target_key(r.check, r.incident, r.et_date, r.selection)
                    t = st["targets"].get(tk)
                    if t is None or not t["open"]:
                        ep = (t["episode"] + 1) if t else 1
                        st["targets"][tk] = {"check": r.check, "incident": r.incident, "et_date": r.et_date,
                                             "selection": r.selection, "episode": ep, "open": True,
                                             "state": r.status.value, "opened_at": now}
                        new += self._queue(st, tk, ep, r.status.value, notice_text(r, "open", ep), now)
                    elif t["state"] != r.status.value:
                        t["state"] = r.status.value
                        new += self._queue(st, tk, t["episode"], r.status.value,
                                           notice_text(r, "escalation", t["episode"]), now)
                elif r.status is Status.VERIFIED:
                    for tk, t in st["targets"].items():
                        if (t["open"] and t["check"] == r.check and t["et_date"] == r.et_date
                                and t["selection"] == r.selection):
                            t.update(open=False, state="recovered", closed_at=now)
                            new += self._queue(st, tk, t["episode"], "recovered",
                                               notice_text(r, "recovered", t["episode"]), now)
            save_state(self.root, st)
        return new

    @staticmethod
    def _queue(st, tk, episode, state, text, now) -> int:
        nk = notice_key(tk, episode, state)
        if nk in st["notices"]:
            return 0
        st["notices"][nk] = {"target": tk, "episode": episode, "state": state, "text": text, "status": "pending",
                             "attempts": 0, "queued_at": now, "message_id": None, "recipient": None,
                             "last_error": None}
        return 1

    # ---- delivery -------------------------------------------------------------------------------------------------
    def _due(self, n, now: datetime) -> bool:
        if n["status"] == "pending":
            return True
        if n["status"] != "sending":
            return False
        claimed, until = _parse_utc(n["claimed_at"]), _parse_utc(n["lease_until"])
        return now >= until or now < claimed          # expired, or the clock rolled back past the claim

    def flush(self) -> dict:
        """Send due notices (bounded). Returns {"sent": n, "failed": n, "skipped_budget": n}."""
        report = {"sent": 0, "failed": 0, "skipped_budget": 0}
        if self.send is None or not self.recipient:
            return report                                # queue-only (B7): notices stay pending
        with state_lock(self.root):
            st = load_state(self.root)
            now = _utc(self.clock.now())
            due = sorted((k for k, n in st["notices"].items() if self._due(n, now)),
                         key=lambda k: st["notices"][k]["queued_at"])[: self.max_sends]
            claims = {}
            for k in due:
                token = uuid.uuid4().hex
                st["notices"][k].update(status="sending", claim_token=token, claimed_at=now.isoformat(),
                                        lease_until=(now + self.lease).isoformat())
                claims[k] = (token, st["notices"][k]["text"])
            if claims:
                save_state(self.root, st)
        start = now
        outcomes = {}
        for k, (token, text) in claims.items():          # network, outside the lock
            if _utc(self.clock.now()) - start > self.budget:
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
