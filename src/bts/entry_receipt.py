"""The pick-entry receipt (C1 rank-2 watchdog plan P1; registration R4 and its producer receipt contract).

`bts check-pick-entered` publishes one receipt per run, whatever the outcome. The schema, the paths and the status
meanings are documented in `docs/ops/pick-entry-receipt-v1.md`. The receipt only records; the run's DMs, marker
statuses, exit codes and authenticated requests are unchanged. A failed publication is reported on stderr and is
unavailable evidence, never success. Cookies and tokens are never stored.

**Outcomes:**
- **No attempt:** `no_pick_file`, `not_committed`, `outside_window`.
- **No fetch:** `already_confirmed`. It references the receipt that confirmed the entry and claims no new observation.
- **Attempted:** `fetch_failed`, `identity_mismatch` and `observed`. Only `observed` is a successful account
  observation. `incomplete` means the run raised before setting an outcome.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

SCHEMA = "bts_pick_entry_receipt_v1"
REPO = Path(__file__).resolve().parents[2]


def receipts_root(picks_dir: Path) -> Path:
    return Path(picks_dir).parent / "health_state" / "pick_entry_receipts"


def payload_sha256(obj) -> str:
    """sha256 of a payload's canonical JSON (sorted keys, compact), binding what the verifier consumed."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _file_sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def _write_durable(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def selection_identity(daily, date: str, picks_dir: Path) -> dict:
    """The committed selection the run checks against (expected identity): each slot's role, batter and game, plus
    the delivery and commit references and the pick file's sha256."""
    from bts.daily_decision import decision_path, load_decision

    slots = [{"role": role, "batter_id": p.batter_id, "batter_name": p.batter_name, "game_pk": p.game_pk,
              "game_time": p.game_time}
             for role, p in (("pick", daily.pick), ("double_down", daily.double_down)) if p is not None]
    rec = load_decision(date, picks_dir)
    return {"slots": slots,
            "delivery": {"notification_sent": daily.notification_sent, "notification_channel": daily.notification_channel,
                         "notification_id": daily.notification_id, "bluesky_posted": daily.bluesky_posted,
                         "bluesky_uri": daily.bluesky_uri, "delivered_at": daily.delivered_at},
            "commit": {"decision_sha256": _file_sha256(decision_path(date, picks_dir)),
                       "delivery_status": rec.get("delivery_status") if rec else None,
                       "scoreable": rec.get("scoreable") if rec else None},
            "pick_file_sha256": _file_sha256(Path(picks_dir) / f"{date}.json")}


def rows_for_date(profile: dict, pending: list, rounds: dict, target) -> dict:
    """The observed source rows for the target date: profile predictions and pending rows whose round maps to it."""
    def on_target(round_id):
        try:
            return round_id is not None and rounds.get(int(round_id)) == target
        except (TypeError, ValueError):
            return False
    return {"profile": [p for p in profile.get("predictions", []) if on_target(p.get("roundId"))],
            "pending": [r for r in pending if on_target(r.get("roundId"))]}


@dataclass
class EntryReceipt:
    et_date: str
    started_at: datetime
    clock: object                                   # () -> tz-aware datetime
    expected_username: str | None = None
    attempt_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    outcome: str = "incomplete"
    detail: str | None = None
    selection: dict | None = None
    cutoff_at: datetime | None = None
    minutes_to_pitch: float | None = None
    account: dict | None = None
    observation: dict | None = None
    verifier: dict | None = None
    references: dict | None = None
    error: dict | None = None
    failed_at: datetime | None = None
    marker_status: str | None = None
    exit_code: int | None = 0
    fetch_mono: float | None = None                 # the run's own fetch-start monotonic reading (test clock anchor)

    def set_account(self, user_id, username) -> None:
        self.account = {"expected_username": self.expected_username, "user_id": user_id, "username": username}

    def observed(self, *, profile, pending, rounds, crosswalk, target, ok, reason, required_mlb_ids) -> None:
        """A successful account observation, timestamped when the responses completed."""
        from bts.contest_fetch import entered_bts_player_ids

        done = self.clock()
        entered = entered_bts_player_ids(profile, pending, rounds, target)
        self.outcome = "observed"
        self.observation = {
            "response_completed_at": done.isoformat(),
            "before_cutoff": (done < self.cutoff_at) if self.cutoff_at is not None else None,
            "sources": {"profile_sha256": payload_sha256(profile), "pending_sha256": payload_sha256(pending),
                        "rounds_sha256": payload_sha256({str(k): v for k, v in rounds.items()}),
                        "crosswalk_sha256": payload_sha256({str(k): v for k, v in crosswalk.items()})},
            "rows": rows_for_date(profile, pending, rounds, target),
            "entered_bts_ids": sorted(entered),
            "resolved_mlb_ids": sorted({crosswalk[b] for b in entered if b in crosswalk})}
        self.verifier = {"ok": bool(ok), "reason": reason, "required_mlb_ids": sorted(required_mlb_ids)}

    def failed(self, exc: BaseException) -> None:
        self.outcome = "fetch_failed"
        self.failed_at = self.clock()
        status = getattr(getattr(exc, "response", None), "status_code", None)
        # The type (and an HTTP status) only: messages can carry request URLs and headers.
        self.error = {"type": type(exc).__name__, "http_status": status if isinstance(status, int) else None}

    def record(self) -> dict:
        iso = lambda t: t.isoformat() if t is not None else None  # noqa: E731
        from bts.picks import _git_head_sha
        return {"schema": SCHEMA, "attempt_id": self.attempt_id,
                "producer": {"command": "check-pick-entered", "revision": _git_head_sha(REPO)},
                "et_date": self.et_date, "season": int(self.et_date[:4]),
                "started_at": iso(self.started_at), "outcome": self.outcome, "detail": self.detail,
                "selection": self.selection, "cutoff_at": iso(self.cutoff_at),
                "minutes_to_pitch": self.minutes_to_pitch, "account": self.account,
                "observation": self.observation, "verifier": self.verifier, "references": self.references,
                "error": self.error, "failed_at": iso(self.failed_at), "marker_status": self.marker_status,
                "exit_code": self.exit_code, "published_at": iso(self.clock())}

    def publish(self, picks_dir: Path) -> Path | None:
        """Atomic, durable, one file per attempt. Any failure is reported and returns None: the receipt is then
        unavailable evidence, and the run's behaviour is unaffected."""
        try:
            rec = self.record()
            stamp = self.started_at.strftime("%Y%m%dT%H%M%S%f")
            path = receipts_root(picks_dir) / self.et_date / f"{stamp}-{self.attempt_id}.json"
            _write_durable(path, (json.dumps(rec, indent=1, sort_keys=True, default=str) + "\n").encode())
            return path
        except Exception as exc:  # noqa: BLE001 - publication must never change the run's outcome
            print(f"check-pick-entered: entry receipt unavailable ({type(exc).__name__}: {exc})", file=sys.stderr)
            return None
