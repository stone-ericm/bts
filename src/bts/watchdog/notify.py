"""Watchdog notifications (registration R5).

- **Dedup:** a key binds the incident, the affected ET date, the selection and the escalation/recovery state. It is
  kept in `data/watchdog/notify/state.json`, so it survives restarts.
- **A short owned critical section:** `state_lock` (flock on `notify/.lock`) covers only reading and writing that
  file. The send itself happens outside it.
- **Claims:** a flush claims due entries under the lock with a lease (`sending`, `lease_until`). A concurrent flush
  skips a live lease; an expired lease (a crashed sender) is claimed again.
- **Confirmation:** only a returned non-empty message id records a send as `sent`. A failure, or a return without
  an id, leaves it `pending` and retryable. A crash after the remote accepted a message can cause a duplicate on
  retry; exactly-once delivery is not claimed.
"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
from datetime import timedelta

from bts.watchdog.result import ALERTING, CheckResult

LEASE = timedelta(minutes=10)


class NotifyStateError(RuntimeError):
    pass


def dedup_key(incident: str, et_date: str, selection: str | None, state: str) -> str:
    return hashlib.sha256(json.dumps([incident, et_date, selection, state]).encode()).hexdigest()


def _state_path(root):
    return root.child("notify", "state.json")


@contextlib.contextmanager
def state_lock(root):
    root.ensure_dir("notify")
    with open(root.child("notify", ".lock"), "a") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def load_state(root) -> dict:
    p = _state_path(root)
    if not p.exists():
        return {}
    try:
        st = json.loads(p.read_text())
    except ValueError as exc:
        # never silently reset: a lost state would drop pending alerts
        raise NotifyStateError(f"{p} is unreadable: {exc}") from exc
    if not isinstance(st, dict):
        raise NotifyStateError(f"{p} is not an object")
    return st


def save_state(root, st: dict) -> None:
    root.write_atomic(_state_path(root), (json.dumps(st, indent=1, sort_keys=True) + "\n").encode())


def alert_text(r: CheckResult) -> str:
    tag = f" [{r.incident}]" if r.incident else ""
    return f"BTS watchdog {r.status.value}{tag}: {r.check} on {r.et_date}: {r.detail}"[:1000]


class Notifier:
    def __init__(self, root, *, recipient: str, send, clock, lease: timedelta = LEASE):
        self.root, self.recipient, self.send, self.clock, self.lease = root, recipient, send, clock, lease

    def enqueue(self, results) -> int:
        """Record new alerting results as pending (deduplicated). Returns how many were new."""
        alerts = [r for r in results if r.status in ALERTING]
        if not alerts:
            return 0
        now = self.clock.now().isoformat()
        new = 0
        with state_lock(self.root):
            st = load_state(self.root)
            for r in alerts:
                incident = r.incident or r.check
                key = dedup_key(incident, r.et_date, r.selection, r.status.value)
                if key in st:
                    continue
                st[key] = {"incident": incident, "check": r.check, "et_date": r.et_date, "selection": r.selection,
                           "state": r.status.value, "text": alert_text(r), "recipient": self.recipient,
                           "status": "pending", "attempts": 0, "first_seen": now, "message_id": None}
                new += 1
            save_state(self.root, st)
        return new

    def flush(self) -> None:
        now = self.clock.now()
        with state_lock(self.root):                         # claim, briefly
            st = load_state(self.root)
            claim = [k for k, e in st.items()
                     if e.get("status") == "pending"
                     or (e.get("status") == "sending" and e.get("lease_until", "") <= now.isoformat())]
            for k in claim:
                st[k].update(status="sending", lease_until=(now + self.lease).isoformat(), lease_owner=os.getpid())
            if claim:
                save_state(self.root, st)
            texts = {k: (st[k]["recipient"], st[k]["text"]) for k in claim}
        outcomes = {}
        for k, (recipient, text) in texts.items():              # the network, outside the lock
            try:
                mid = self.send(recipient, text)
            except Exception as exc:  # noqa: BLE001 - a failed send stays pending
                outcomes[k] = (None, type(exc).__name__)
            else:
                ok = isinstance(mid, str) and mid != ""
                outcomes[k] = (mid if ok else None, None if ok else "no message id")
        if not outcomes:
            return
        with state_lock(self.root):                         # record, briefly
            st = load_state(self.root)
            for k, (mid, err) in outcomes.items():
                e = st.get(k)
                if e is None:
                    continue
                e["attempts"] = e.get("attempts", 0) + 1
                e["last_attempt_at"] = now.isoformat()
                e.pop("lease_until", None)
                e.pop("lease_owner", None)
                if mid:
                    e.update(status="sent", message_id=mid, last_error=None)
                else:
                    e.update(status="pending", last_error=err)
            save_state(self.root, st)
