"""O7 BTS static captures (rounds, players, units) and acquired MLB schedule responses (spec §4, §6). Item
locators are positional, so a capture that lists an id twice still yields distinct occurrences. Each row keeps
the RAW values of every field it normalizes (`record_raw_json`), so a wrong-typed value survives in the
occurrence table; the rest of an item (e.g. unit lineups) stays in the sealed bundle, addressable by
(source_path, locator)."""
from __future__ import annotations

import re

from ..ids import (Parsed, canonical_json, is_int, joined, load_json_bytes, obs_id, quarantine, sha256_hex,
                   stamp_to_utc, take, typed)

_SCHEDULE_PATH = re.compile(r"^schedules/(\d{4}-\d{2}-\d{2})\.json$")


def _items(rel_path: str, data: bytes, key: str):
    try:
        doc = load_json_bytes(data)
    except ValueError as exc:
        return None, Parsed([], [quarantine(rel_path, "file", str(exc))])
    if not isinstance(doc, dict) or not isinstance(doc.get(key), list):
        return None, Parsed([], [quarantine(rel_path, "file", f"missing_{key}_list", raw=doc)])
    return doc[key], None


def _raw(item, keys: tuple[str, ...]):
    return {k: item[k] for k in keys if k in item} if isinstance(item, dict) else item


def _row(kind: str, rel_path: str, locator: str, content: str, raw, **fields) -> dict:
    return {"obs_id": obs_id(rel_path, locator, content), "locator": locator, "source_kind": kind,
            "source_path": rel_path, "content_sha256": content, **fields, "record_raw_json": canonical_json(raw)}


ROUND_KEYS, PLAYER_KEYS, UNIT_KEYS = ("id", "date", "status"), ("id", "feedId", "squadId", "name"), (
    "id", "feedId", "roundId", "status")
GAME_KEYS = ("gamePk", "gameNumber", "officialDate", "status", "teams")


def parse_rounds(rel_path: str, data: bytes) -> Parsed:
    items, bad = _items(rel_path, data, "rounds")
    if bad is not None:
        return bad
    content, out = sha256_hex(data), Parsed()
    for i, r in enumerate(items):
        loc = f"item={i}"
        if not isinstance(r, dict) or not is_int(r.get("id")) or not isinstance(r.get("date"), str):
            out.quarantined.append(quarantine(rel_path, loc, "round_missing_id_or_date", raw=_raw(r, ROUND_KEYS)))
            continue
        status, mismatch = typed(r.get("status"), "str")
        out.rows.append(_row("rounds", rel_path, loc, content, _raw(r, ROUND_KEYS), round_id=r["id"],
                             round_date=r["date"][:10], status=status,
                             type_mismatch_fields="status" if mismatch else None))
    return out


def parse_players(rel_path: str, data: bytes) -> Parsed:
    items, bad = _items(rel_path, data, "players")
    if bad is not None:
        return bad
    content, out = sha256_hex(data), Parsed()
    for i, p in enumerate(items):
        loc = f"item={i}"
        if not isinstance(p, dict) or not is_int(p.get("id")):
            out.quarantined.append(quarantine(rel_path, loc, "player_missing_id", raw=_raw(p, PLAYER_KEYS)))
            continue
        values, absent, mismatch = take(p, {"feedId": "int", "squadId": "int", "name": "str"})
        out.rows.append(_row("players", rel_path, loc, content, _raw(p, PLAYER_KEYS), player_id=p["id"],
                             feed_id=values["feedId"],
                             squad_id=values["squadId"], name=values["name"], absent_fields=joined(absent),
                             type_mismatch_fields=joined(mismatch)))
    return out


def parse_units(rel_path: str, data: bytes) -> Parsed:
    items, bad = _items(rel_path, data, "units")
    if bad is not None:
        return bad
    content, out = sha256_hex(data), Parsed()
    captured_at = stamp_to_utc(rel_path.rsplit("/", 1)[-1])
    for i, u in enumerate(items):
        loc = f"item={i}"
        if not isinstance(u, dict) or not is_int(u.get("id")):
            out.quarantined.append(quarantine(rel_path, loc, "unit_missing_id", raw=_raw(u, UNIT_KEYS)))
            continue
        values, absent, mismatch = take(u, {"feedId": "int", "roundId": "int", "status": "str"})
        out.rows.append(_row("units", rel_path, loc, content, _raw(u, UNIT_KEYS), unit_id=u["id"],
                             feed_id=values["feedId"],
                             round_id=values["roundId"], status=values["status"], captured_at=captured_at,
                             absent_fields=joined(absent), type_mismatch_fields=joined(mismatch)))
    return out


def _abbreviation(teams, side: str) -> str | None:
    entry = teams.get(side) if isinstance(teams, dict) else None
    team = entry.get("team") if isinstance(entry, dict) else None
    abbr = team.get("abbreviation") if isinstance(team, dict) else None
    return abbr if isinstance(abbr, str) and abbr else None


def parse_schedule(rel_path: str, data: bytes) -> Parsed:
    """Every listed game, whatever its status. A date entry without games, or a game without a gamePk or
    both team abbreviations, is quarantined — the compiler then treats that date's schedule as incomplete
    and allows no inference on it (Codex plan r1 #7)."""
    m = _SCHEDULE_PATH.match(rel_path)
    if not m:
        return Parsed([], [quarantine(rel_path, "file", "unexpected_schedule_path")])
    items, bad = _items(rel_path, data, "dates")
    if bad is not None:
        return bad
    content, out = sha256_hex(data), Parsed()
    for i, day in enumerate(items):
        games = day.get("games") if isinstance(day, dict) else None
        if not isinstance(games, list) or not games:
            out.quarantined.append(quarantine(rel_path, f"date={i}", "date_without_games_list",
                                              raw=_raw(day, ("date", "games"))))
            continue
        if day.get("date") != m.group(1):   # Codex code r1 #5: only the requested date's games are evidence for it
            out.quarantined.append(quarantine(rel_path, f"date={i}", "date_entry_not_query_date",
                                              raw=_raw(day, ("date",))))
            continue
        for j, g in enumerate(games):
            loc = f"date={i}/game={j}"
            if not isinstance(g, dict) or not is_int(g.get("gamePk")):
                out.quarantined.append(quarantine(rel_path, loc, "game_missing_gamePk", raw=_raw(g, GAME_KEYS)))
                continue
            away, home = _abbreviation(g.get("teams"), "away"), _abbreviation(g.get("teams"), "home")
            if away is None or home is None:
                out.quarantined.append(quarantine(rel_path, loc, "game_missing_team_abbreviation",
                                                  raw=_raw(g, GAME_KEYS)))
                continue
            status = g.get("status") if isinstance(g.get("status"), dict) else {}
            out.rows.append(_row("schedule", rel_path, loc, content, _raw(g, GAME_KEYS), query_date=m.group(1),
                                 game_pk=g["gamePk"],
                                 away_abbr=away, home_abbr=home,
                                 coded_state=typed(status.get("codedGameState"), "str")[0],
                                 detailed_state=typed(status.get("detailedState"), "str")[0],
                                 official_date=typed(g.get("officialDate"), "str")[0],
                                 game_number=typed(g.get("gameNumber"), "int")[0]))
    return out
