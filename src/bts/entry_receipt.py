"""The pick-entry receipt (C1 rank-2 watchdog plan P1; registration R4 and its producer receipt contract).

`bts check-pick-entered` publishes one receipt per run, whatever the outcome. The schema, paths, states and consumer
rules are in `docs/ops/pick-entry-receipt-v1.md`.

**Behaviour neutral (producer review r1 C5):** during the run, the hooks below only store references or a clock
reading, each behind a guard. All extraction, hashing, qualification and serialization happens in `publish`, after
the run's decisions, DMs and marker writes, and inside its own protection. A failure there leaves **no** discoverable
receipt (unavailable evidence, `bts.receipt_io`); it never changes the run.

**Bound evidence:**
- **Selection (C2):** the exact pick-file and decision bytes the run parsed and gated on. Each was read once; the
  path is never re-read later.
- **Response time (C4):** captured the moment the last fetch returned, before verification.
- **Rows (C3):** only a typed allowlist of fields. No response dictionary is copied, so cookies, session ids and
  tokens cannot ride along.
- **Game qualification (C1):** each slot's observed row is bound to its game through the locally captured
  `units.json` snapshot (`unitId` -> `feedId`, written by the `capture-static` cron; no new request). A missing,
  ambiguous or contradicting mapping is explicitly unverified. The legacy batter-only verifier result is kept
  separately and is not game-qualified.
"""
from __future__ import annotations

import functools
import gzip
import hashlib
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path

from bts import receipt_io

SCHEMA = "bts_pick_entry_receipt_v1"
REPO = Path(__file__).resolve().parents[2]
ROW_FIELDS = {"roundId": int, "unitId": int, "playerId": int, "number": int, "result": str}


def receipts_root(picks_dir: Path) -> Path:
    return Path(picks_dir).parent / "health_state" / "pick_entry_receipts"


def units_snapshot_dir(picks_dir: Path) -> Path:
    return Path(picks_dir).parent / "leaderboard" / "static_snapshots" / "units"


def discover(picks_dir: Path, et_date: str) -> list[dict]:
    """Published receipts for one date (a failed publication is never returned)."""
    return [json.loads(p.read_text()) for p in receipt_io.discover(receipts_root(picks_dir) / et_date)]


def _sha(b: bytes | None) -> str | None:
    return hashlib.sha256(b).hexdigest() if b is not None else None


def payload_sha256(obj) -> str:
    """sha256 of a payload's canonical JSON (sorted keys, compact): binds what the verifier consumed."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _typed(v, kind):
    return v if (v is None or (type(v) is kind)) else None


def clean_row(row: dict, round_id=None) -> dict:
    """The allowlisted, typed fields of one entry row. Anything else, and any mistyped value, is dropped and named."""
    src = dict(row) if isinstance(row, dict) else {}
    if round_id is not None:
        src["roundId"] = round_id
    out = {k: _typed(src.get(k), kind) for k, kind in ROW_FIELDS.items()}
    bad = sorted(k for k, kind in ROW_FIELDS.items() if src.get(k) is not None and out[k] is None)
    if bad:
        out["mistyped"] = bad
    return out


def target_rows(profile: dict, pending: list, rounds: dict, target) -> dict:
    """The target date's rows, allowlisted: flat pending rows, and profile slots flattened under their round."""
    def on_target(round_id):
        try:
            return round_id is not None and rounds.get(int(round_id)) == target
        except (TypeError, ValueError):
            return False
    prof = []
    for p in (profile or {}).get("predictions", []) or []:
        if isinstance(p, dict) and on_target(p.get("roundId")):
            for rp in p.get("roundPredictions", []) or []:
                prof.append(clean_row(rp, round_id=p.get("roundId")))
    pend = [clean_row(r) for r in (pending or []) if isinstance(r, dict) and on_target(r.get("roundId"))]
    return {"profile": prof, "pending": pend}


def load_units(picks_dir: Path) -> tuple[dict | None, dict | None]:
    """The latest captured units.json: ({unit_id: (feedId, roundId)}, source identity). A unit listed twice with
    different identities maps to None (ambiguous). Returns (None, None) when there is no readable capture."""
    d = units_snapshot_dir(picks_dir)
    snaps = sorted(p for p in d.glob("*.json.gz")) if d.is_dir() else []
    if not snaps:
        return None, None
    path = snaps[-1]
    raw = path.read_bytes()
    body = json.loads(gzip.decompress(raw))
    units: dict = {}
    for u in body.get("units", []) or []:
        if not isinstance(u, dict) or type(u.get("id")) is not int:
            continue
        ident = (_typed(u.get("feedId"), int), _typed(u.get("roundId"), int))
        units[u["id"]] = ident if units.get(u["id"], ident) == ident else None
    return units, {"path": str(path.relative_to(Path(picks_dir).parent)), "sha256": _sha(raw),
                   "captured_utc": path.name.split(".")[0]}


def qualify(slots: list[dict], rows: dict, crosswalk: dict, units: dict | None) -> dict:
    """Per selection slot: is its batter's target-date row bound to its game? States: confirmed, missing,
    player_unverified, ambiguous, unit_unverified, wrong_round, wrong_game."""
    all_rows = rows["profile"] + rows["pending"]
    unresolved = any(r["playerId"] is None or r["playerId"] not in crosswalk for r in all_rows)
    out = []
    for s in slots:
        hits = {(r["roundId"], r["unitId"], r["playerId"], r["number"]) for r in all_rows
                if r["playerId"] is not None and crosswalk.get(r["playerId"]) == s["batter_id"]}
        row = None
        if not hits:
            state = "player_unverified" if unresolved else "missing"
        elif len(hits) > 1:
            state = "ambiguous"
        else:
            row = dict(zip(("roundId", "unitId", "playerId", "number"), next(iter(hits))))
            ident = units.get(row["unitId"]) if units is not None else None
            if ident is None or ident[0] is None:
                state = "unit_unverified"
            elif ident[1] != row["roundId"]:
                state = "wrong_round"
            elif ident[0] != s["game_pk"]:
                state = "wrong_game"
            else:
                state = "confirmed"
        out.append({"role": s["role"], "batter_id": s["batter_id"], "game_pk": s["game_pk"], "row": row,
                    "state": state})
    return {"slots": out, "all_confirmed": bool(out) and all(x["state"] == "confirmed" for x in out)}


def _serialize(rec: dict) -> bytes:
    return (json.dumps(rec, indent=1, sort_keys=True, default=str) + "\n").encode()


def _guarded(fn):
    @functools.wraps(fn)
    def wrapper(self, *a, **k):
        try:
            return fn(self, *a, **k)
        except Exception as exc:  # noqa: BLE001 - a hook failure never reaches the run
            self.degraded.append(f"{fn.__name__}:{type(exc).__name__}")
            return None
    return wrapper


class EntryReceipt:
    def __init__(self, *, et_date: str, started_at: datetime, clock, expected_username: str | None, picks_dir):
        self.et_date, self.started_at, self.clock = et_date, started_at, clock
        self.expected_username = expected_username
        self.picks_dir = Path(picks_dir)
        self.attempt_id = uuid.uuid4().hex
        self.outcome, self.detail, self.exit_code = "incomplete", None, 0
        self.fetch_mono = None                    # the run's own fetch-start monotonic reading (test clock anchor)
        self.degraded: list[str] = []
        # stored references only; everything derived is computed in publish()
        self._daily = self._pick_raw = self._dec = self._dec_raw = None
        self._first_pitch = self._cutoff_min = self._minutes = None
        self._account = None
        self._sources = None
        self._responses_done = None
        self._verdict = None
        self._error = self._failed_at = None
        self._references = None
        self._marker_status = None

    # ---- hooks (references and clock readings only) ---------------------------------------------------------------
    @_guarded
    def selection(self, daily, pick_raw: bytes | None, dec, dec_raw: bytes | None) -> None:
        self._daily, self._pick_raw, self._dec, self._dec_raw = daily, pick_raw, dec, dec_raw

    @_guarded
    def window(self, first_pitch: datetime, cutoff_min: int, minutes_to_pitch: float) -> None:
        self._first_pitch, self._cutoff_min, self._minutes = first_pitch, cutoff_min, minutes_to_pitch

    @_guarded
    def set_outcome(self, outcome: str) -> None:
        self.outcome = outcome

    @_guarded
    def confirmed_by(self, attempt_id) -> None:
        self.outcome, self._references = "already_confirmed", {"confirmed_by": attempt_id}

    @_guarded
    def set_account(self, user_id, username) -> None:
        self._account = (user_id, username)

    @_guarded
    def responses_done(self, profile, pending, rounds, crosswalk) -> None:
        """Called the moment the last fetch returns: the actual response-completion time (C4)."""
        self._responses_done = self.clock()
        self._sources = (profile, pending, rounds, crosswalk)

    @_guarded
    def verified(self, ok, reason, required_mlb_ids, target) -> None:
        self.outcome = "observed"
        self._verdict = (ok, reason, required_mlb_ids, target)

    @_guarded
    def failed(self, exc: BaseException) -> None:
        self.outcome = "fetch_failed"
        self._failed_at = self.clock()
        status = getattr(getattr(exc, "response", None), "status_code", None)
        # The type and an HTTP status only: messages can carry request URLs and headers.
        self._error = (type(exc).__name__, status if type(status) is int else None)

    @_guarded
    def marker(self, status) -> None:
        self._marker_status = status

    # ---- the record (publish time) --------------------------------------------------------------------------------
    def _selection(self) -> dict | None:
        if self._daily is None:
            return None
        d = self._daily
        slots = [{"role": role, "batter_id": p.batter_id, "batter_name": p.batter_name, "game_pk": p.game_pk,
                  "game_time": p.game_time}
                 for role, p in (("pick", d.pick), ("double_down", d.double_down)) if p is not None]
        dec = self._dec or {}
        return {"slots": slots,
                "delivery": {"notification_sent": d.notification_sent, "notification_channel": d.notification_channel,
                             "notification_id": d.notification_id, "bluesky_posted": d.bluesky_posted,
                             "bluesky_uri": d.bluesky_uri, "delivered_at": d.delivered_at},
                "commit": {"decision_sha256": _sha(self._dec_raw), "decision_valid": self._dec is not None,
                           "delivery_status": dec.get("delivery_status"), "scoreable": dec.get("scoreable")},
                "pick_file_sha256": _sha(self._pick_raw)}

    def record(self) -> dict:
        from datetime import timedelta
        from bts.picks import _git_head_sha
        iso = lambda t: t.isoformat() if t is not None else None  # noqa: E731
        selection = self._selection()
        cutoff = (self._first_pitch - timedelta(minutes=self._cutoff_min)) if self._first_pitch is not None else None
        account = None
        if self._account is not None:
            account = {"expected_username": self.expected_username, "user_id": self._account[0],
                       "username": self._account[1]}
        observation = verifier = qualification = None
        if self.outcome == "observed" and self._sources is not None and self._verdict is not None:
            profile, pending, rounds, crosswalk = self._sources
            ok, reason, required, target = self._verdict
            rows = target_rows(profile, pending, rounds, target)
            units, units_source = load_units(self.picks_dir)
            entered = [r["playerId"] for r in rows["profile"] + rows["pending"] if r["playerId"] is not None]
            done = self._responses_done
            observation = {
                "response_completed_at": iso(done),
                "before_cutoff": (done < cutoff) if (done is not None and cutoff is not None) else None,
                "sources": {"profile_sha256": payload_sha256(profile), "pending_sha256": payload_sha256(pending),
                            "rounds_sha256": payload_sha256({str(k): v for k, v in rounds.items()}),
                            "crosswalk_sha256": payload_sha256({str(k): v for k, v in crosswalk.items()}),
                            "units": units_source},
                "rows": rows, "entered_bts_ids": sorted(set(entered)),
                "resolved_mlb_ids": sorted({crosswalk[b] for b in entered if b in crosswalk})}
            verifier = {"ok": bool(ok), "reason": reason, "required_mlb_ids": sorted(required),
                        "game_qualified": False}
            qualification = qualify(selection["slots"] if selection else [], rows, crosswalk, units)
        error = {"type": self._error[0], "http_status": self._error[1]} if self._error else None
        return {"schema": SCHEMA, "attempt_id": self.attempt_id,
                "producer": {"command": "check-pick-entered", "revision": _git_head_sha(REPO)},
                "et_date": self.et_date, "season": int(self.et_date[:4]), "started_at": iso(self.started_at),
                "outcome": self.outcome, "detail": self.detail, "selection": selection, "cutoff_at": iso(cutoff),
                "minutes_to_pitch": round(self._minutes, 3) if self._minutes is not None else None,
                "account": account, "observation": observation, "verifier": verifier, "qualification": qualification,
                "references": self._references, "error": error, "failed_at": iso(self._failed_at),
                "marker_status": self._marker_status, "exit_code": self.exit_code, "degraded": self.degraded,
                "published_at": iso(self.clock())}

    def publish(self) -> Path | None:
        """Build, then publish durably. Any failure leaves no discoverable receipt and is reported on stderr."""
        try:
            data = _serialize(self.record())
            stamp = self.started_at.strftime("%Y%m%dT%H%M%S%f")
            path = receipts_root(self.picks_dir) / self.et_date / f"{stamp}-{self.attempt_id}.json"
            receipt_io.publish(path, data)
            return path
        except Exception as exc:  # noqa: BLE001 - publication must never change the run's outcome
            print(f"check-pick-entered: entry receipt unavailable ({type(exc).__name__})", file=sys.stderr)
            return None
