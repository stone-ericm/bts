"""Load and repair the probable-pitcher lookup from available raw feeds."""

import hashlib
import io
import json
import os
import stat
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from bts.probable_pitcher_status import (
    COUNTS, ET, EVIDENCE, MAX_BYTES, SCHEMA, SOURCE, NonRegularFile,
    directory_exists, display, open_regular, read_record,
)


MEMBERS = {"away", "home", "away_tid", "home_tid"}


def valid(entry):
    return (isinstance(entry, dict) and entry.keys() == MEMBERS
            and all(value is None or type(value) is int for value in entry.values()))


def warn(message):
    try:
        stream = sys.stderr
        line = f"WARNING: {message}"
        try:
            fd = stream.fileno()
        except (AttributeError, OSError, ValueError):
            fd = -1
        if fd < 0:
            print(line, file=stream, flush=True)
        else:
            stream.flush()
            os.write(fd, f"{line}\n".encode(stream.encoding or "utf-8", stream.errors or "backslashreplace"))
    except Exception:
        pass


def decode(raw):
    # Match Path.read_text's locale encoding and universal newline handling.
    with io.TextIOWrapper(io.BytesIO(raw)) as stream:
        return json.load(stream)


def read_cache(path):
    original = None
    try:
        path.lstat()
    except FileNotFoundError:
        return [], None, False, False
    except OSError as error:
        warn(f"unreadable probable-pitcher cache {path} ({type(error).__name__}); rebuilding it from raw feeds")
        return [], None, True, False

    try:
        with open_regular(path, follow_symlinks=True) as stream:
            original = stream.read()
        decoded = decode(original)
        return [(int(key), entry) for key, entry in decoded.items()], original, False, False
    except Exception as error:
        special = isinstance(error, NonRegularFile)
        error_name = "OSError" if special else type(error).__name__
        if original is None and isinstance(error, OSError):
            try:
                mode = path.stat().st_mode
            except OSError:
                pass
            else:
                special = not stat.S_ISREG(mode)
                if stat.S_ISDIR(mode):
                    error_name = "IsADirectoryError"
        warn(f"unreadable probable-pitcher cache {path} ({error_name}); rebuilding it from raw feeds")
        return [], original, True, special


def load_entries(members):
    # Keep the position of the first integer alias even if that alias is bad.
    lookup = dict.fromkeys(game for game, _ in members)
    invalid = Counter()
    for game, entry in members:
        if valid(entry):
            lookup[game] = entry
        else:
            invalid[game] += 1
    return {game: entry for game, entry in lookup.items() if entry is not None}, invalid


def scan(raw, lookup):
    scanned = {}
    rejected = set()
    incomplete = False
    try:
        seasons = sorted(raw.iterdir())
    except (FileNotFoundError, NotADirectoryError):
        return scanned, rejected, incomplete
    except OSError:
        return scanned, rejected, True

    for season in seasons:
        try:
            if not stat.S_ISDIR(season.stat().st_mode):
                continue
        except FileNotFoundError:
            continue
        except OSError:
            incomplete = True
            continue
        try:
            files = sorted(season.iterdir())
        except OSError:
            incomplete = True
            continue
        for path in files:
            if path.suffix != ".json" or not path.stem.isdigit():
                continue
            try:
                game = int(path.stem)
            except ValueError:
                continue
            if game in lookup or game in scanned:
                continue
            try:
                with open_regular(path, follow_symlinks=True) as stream:
                    data = decode(stream.read()).get("gameData", {})
                teams = data.get("teams", {})
                pitchers = data.get("probablePitchers", {})
                entry = {
                    "away": pitchers.get("away", {}).get("id"),
                    "home": pitchers.get("home", {}).get("id"),
                    "away_tid": teams.get("away", {}).get("id"),
                    "home_tid": teams.get("home", {}).get("id"),
                }
            except Exception:
                rejected.add(game)
                continue
            if valid(entry):
                scanned[game] = entry
            else:
                rejected.add(game)
    return scanned, rejected, incomplete


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def canonical(payload):
    return json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")


def ensure_directory(path):
    if not directory_exists(path):
        path.mkdir(exist_ok=True)
        directory_exists(path)


def write_artifact(path, payload):
    temp = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temp = Path(stream.name)
            stream.write(payload)
        os.replace(temp, path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def preserve(folder, target, original, original_sha):
    if original is None:
        raise OSError("original cache bytes unavailable")
    ensure_directory(folder)
    for candidate in sorted(folder.iterdir()):
        if not EVIDENCE.fullmatch(candidate.name):
            continue
        try:
            with open_regular(candidate) as stream:
                candidate_sha = hashlib.file_digest(stream, "sha256").hexdigest()
        except OSError:
            continue
        if candidate_sha == original_sha:
            return candidate.name
    try:
        target.lstat()
    except FileNotFoundError:
        write_artifact(target, original)
        return target.name
    raise FileExistsError(target)


def publish(path, row):
    ensure_directory(path.parent.parent)
    ensure_directory(path.parent)
    payload = canonical(row)
    if len(payload) > MAX_BYTES:
        raise ValueError("status exceeds size limit")
    try:
        path.lstat()
    except FileNotFoundError:
        write_artifact(path, payload)
        return
    existing = read_record(path)
    if any(existing[key] != value for key, value in row.items() if key != "recorded_at"):
        raise FileExistsError(path)


def drop_outcomes(lookup, invalid, scanned, rejected, incomplete):
    outcomes = []
    totals = [0, 0, 0, 0]
    for game, count in sorted(invalid.items()):
        if game in lookup:
            index = 0
        elif game in scanned:
            index = 1
        elif incomplete or game in rejected:
            index = 2
        else:
            index = 3
        counts = [0, 0, 0, 0]
        counts[index] = count
        totals[index] += count
        outcomes.append([game, *counts, lookup.get(game, scanned.get(game))])
    return outcomes, [sum(totals), *totals]


def report(cache, original, special, reason, scanned, outcomes, counts):
    at = datetime.now(timezone.utc)
    day = at.astimezone(ET).date().isoformat()
    original_sha = digest(original) if original is not None else None
    preserved = None
    preserved_ok = True
    if scanned and not special:
        folder = cache.parent / "rejected"
        target = folder
        if original_sha is not None:
            target = folder / f"probable_pitcher_lookup.{at:%Y%m%dT%H%M%SZ}.{original_sha[:12]}.json"
        try:
            preserved = preserve(folder, target, original, original_sha)
        except Exception as error:
            preserved_ok = False
            warn(f"could not preserve probable-pitcher cache evidence at {display(target)} ({type(error).__name__})")

    observation = dict(
        reason=reason, cache_path=str(cache), original_sha256=original_sha, preserved=preserved,
        **dict(zip(COUNTS, counts, strict=True)),
    )
    identity = digest(canonical(dict(observation, outcomes=outcomes)))
    row = dict(
        observation, schema_version=SCHEMA, recorded_at=at.isoformat(), recorded_date_et=day,
        game_ids=[outcome[0] for outcome in outcomes[:10]] if reason == "invalid_entries" else [],
        identity=identity,
    )
    target = cache.parent.parent / "health_state" / SOURCE / f"{day}.{reason}.{identity[:12]}.json"
    try:
        publish(target, row)
    except Exception as error:
        warn(f"could not write probable-pitcher cache status at {display(target)} ({type(error).__name__})")
        return False
    return preserved_ok


def build(cache, raw):
    members, original, unreadable, special = read_cache(cache)
    lookup, invalid = load_entries(members)
    scanned, rejected, incomplete = scan(Path(raw), lookup)
    can_rewrite = True
    if unreadable:
        outcomes = [[game, entry] for game, entry in sorted(scanned.items())]
        counts = [None, None, len(scanned), None, None]
        can_rewrite = report(cache, original, special, "unreadable_cache", scanned, outcomes, counts)
    elif invalid:
        outcomes, counts = drop_outcomes(lookup, invalid, scanned, rejected, incomplete)
        dropped, aliases, rescanned, unusable, missing = counts
        noun = "entry" if dropped == 1 else "entries"
        sample = ", ".join(str(outcome[0]) for outcome in outcomes[:10])
        warn(
            f"probable-pitcher cache {display(cache)}: dropped {dropped} invalid {noun} "
            f"({aliases} recovered from valid cache entries, {rescanned} rescanned from raw feeds, "
            f"{unusable} with unusable feeds, {missing} without feeds); first game ids: {sample}"
        )
        can_rewrite = report(cache, original, special, "invalid_entries", scanned, outcomes, counts)

    lookup.update(scanned)
    if scanned and can_rewrite and not special:
        cache.parent.mkdir(parents=True, exist_ok=True)
        temp = cache.with_name(f".{cache.name}.{os.getpid()}.tmp")
        try:
            temp.write_text(json.dumps({str(game): entry for game, entry in lookup.items()}))
            os.replace(temp, cache)
        finally:
            temp.unlink(missing_ok=True)
    return lookup
