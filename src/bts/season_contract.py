"""Season-switch snapshots and offline inspection; no decision enforcement yet."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from bts.contest_state import _UNREADABLE

SCHEMA = "bts_season_contract_v1"
AUTO_SCHEMA = "bts_contest_streak_auto_v2"
ET = ZoneInfo("America/New_York")


def valid_activation_id(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{32}", value) is not None


@dataclass(frozen=True)
class SwitchState:
    state: str
    snapshot: dict | None = None
    check: int | None = None
    error: str | None = None


class InvalidContract(ValueError):
    def __init__(self, check: int, text: str):
        self.check = check
        super().__init__(text)


def validate_contract(data: object) -> dict:
    """Validate all eight checks in order before inspecting the season."""
    def require(ok: bool, check: int, text: str) -> None:
        if not ok:
            raise InvalidContract(check, text)

    require(isinstance(data, dict), 1, "file must be a readable UTF-8 JSON object")
    require(data.get("schema_version") == SCHEMA, 2, f"schema_version must be {SCHEMA}")
    require(type(data.get("season")) is int, 3, "season must be an integer")
    require(data.get("status") in ("automatic", "revoked"), 4, "status must be automatic or revoked")
    require(valid_activation_id(data.get("activation_id")), 5,
            "activation_id must be 32 lowercase hexadecimal characters")
    for number, field, required_status in [(6, "switched_at", "automatic"), (7, "revoked_at", "revoked")]:
        value = data.get(field)
        valid = field in data
        if value is None:
            valid = valid and data["status"] != required_status
        elif isinstance(value, str):
            try:
                parsed = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
                valid = valid and parsed.tzinfo is not None
            except ValueError:
                valid = False
        else:
            valid = False
        require(valid, number, f"{field} must be present, null or a timezone-aware ISO timestamp; "
                f"non-null when status is {required_status}")
    require(data.get("note") is None or isinstance(data["note"], str), 8, "note must be null or a string")
    return data


def switch_state(picks_dir: Path, now: datetime, *, season: int | None = None) -> SwitchState:
    """Read and validate one snapshot. Only lstat FileNotFoundError is absent."""
    path = Path(picks_dir) / "account_state/season_contract.json"
    try:
        path.lstat()
    except FileNotFoundError:
        return SwitchState("absent")
    except _UNREADABLE as exc:
        return SwitchState("corrupt", check=1, error=f"{path}: {exc}")
    try:
        data = validate_contract(json.loads(path.read_text(encoding="utf-8")))
    except InvalidContract as exc:
        return SwitchState("corrupt", check=exc.check, error=f"{path}: {exc}")
    except _UNREADABLE as exc:
        return SwitchState("corrupt", check=1, error=f"{path}: {exc}")
    inspection = now.astimezone(ET).year if season is None else season
    return SwitchState("other_season" if data["season"] != inspection else data["status"], data)


def switch_lines(state: SwitchState) -> list[str]:
    lines = [f"switch: {state.state}"]
    if state.state == "corrupt":
        lines.append(f"  check {state.check}: {state.error}")
    elif state.snapshot is not None:
        lines.extend(f"  {key}: {state.snapshot.get(key)}" for key in
                     ["season", "status", "activation_id", "switched_at", "revoked_at", "note"])
    return lines


def _present(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def inspection_lines(picks: Path, now: datetime, state: SwitchState, *, season: int) -> list[str]:
    """Independent read-only sections. A failed section never suppresses its neighbours."""
    from bts.contest_state import _parse_state_file, ContestStateError, dated_observation_in_effect
    from bts.saver_state import load_saver_state

    today = now.astimezone(ET).date()
    lines = [f"ET date: {today}; year: {today.year}; inspected season: {season}", *switch_lines(state)]
    directory = picks / "account_state"
    for label, filename in [("manual", "contest_streak.manual.json"), ("automatic", "contest_streak.json")]:
        path = directory / filename
        try:
            if not _present(path):
                lines.append(f"{label}: absent")
                continue
            parsed = _parse_state_file(path, require_dated=False)
            if label == "manual":
                if parsed.effective_date is None:
                    lines.append(f"manual: legacy streak={parsed.streak} expiry={parsed.override_expires_at}")
                else:
                    lines.append(f"manual: dated date={parsed.effective_date} streak={parsed.streak} "
                                 f"best={parsed.best_streak} saver={parsed.saver_state} "
                                 f"in effect={dated_observation_in_effect(parsed, now)}")
            else:
                raw = json.loads(path.read_text(encoding="utf-8"))
                schema = ("missing" if "schema_version" not in raw else
                          "null" if raw["schema_version"] is None else repr(raw["schema_version"]))
                if raw.get("schema_version") not in ("bts_contest_streak_auto_v1", AUTO_SCHEMA, None):
                    schema = f"unknown ({schema})"
                lines.append(f"automatic: parsed schema={schema} streak={parsed.streak} best={parsed.best_streak} "
                             f"source date={parsed.source_date} recorded time={parsed.recorded_at}")
                if parsed.schema_version == AUTO_SCHEMA:
                    equal = (parsed.activation_id == state.snapshot["activation_id"]
                             if state.snapshot is not None else "unavailable")
                    lines.extend([f"  season: {parsed.season}; activation_id: {parsed.activation_id}",
                                  f"  identifier matches switch: {equal} (comparison only)"])
        except (ContestStateError, *_UNREADABLE) as exc:
            adjective = "invalid" if label == "manual" else "refused"
            lines.append(f"{label}: {adjective}; unavailable: {exc}")

    path = directory / "saver_state.json"
    try:
        if not _present(path):
            lines.append(f"saver: unavailable (absent); reads uninitialized for season {today.year}")
        else:
            # Decode first so all _UNREADABLE failures get a diagnostic, even those
            # the existing fail-closed saver loader deliberately does not catch.
            raw = json.loads(path.read_text(encoding="utf-8"))
            flag = load_saver_state(picks, season=today.year)
            if flag.state != "uninitialized":
                lines.append(f"saver: {flag.state}; season: {flag.season}")
            else:
                valid = (isinstance(raw, dict) and isinstance(raw.get("state"), str)
                         and raw["state"] in {"not_earned", "active", "used"}
                         and isinstance(raw.get("season"), int))
                reason = "another season" if valid and raw["season"] != today.year else "invalid"
                lines.append(f"saver: unavailable ({reason}); season: {flag.season}; reads uninitialized")
    except _UNREADABLE as exc:
        lines.append(f"saver: unavailable (invalid: {exc}); reads uninitialized for season {today.year}")

    path = directory / "contest_ledger.jsonl"
    try:
        if not _present(path):
            lines.append("ledger: absent; rows: 0; bad lines: 0")
        else:
            rows, bad = [], 0
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("not an object")
                    rows.append(row)
                except (ValueError, RecursionError):
                    bad += 1
            lines.append(f"ledger: readable; rows: {len(rows)}; bad lines: {bad}")
            keys = ["recorded_at", "active_streak", "best_streak", "source_date", "season", "switched",
                    "activation_id", "settled_by_year", "settled_unmapped", "unsettled"]
            lines.extend("  " + "; ".join(f"{key}={row.get(key, '—')}" for key in keys) for row in rows[-10:])
    except _UNREADABLE as exc:
        lines.append(f"ledger: unreadable; unavailable: {exc}; counts unavailable")
    return lines
