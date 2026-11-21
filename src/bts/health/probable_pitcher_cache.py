"""Read recent probable-pitcher cache observations for end-of-day health."""

import hashlib
import re
import json
import logging
from datetime import date, datetime, timedelta
from fnmatch import fnmatchcase
from pathlib import Path

from bts.health.alert import Alert
from bts.probable_pitcher_status import SOURCE, directory_exists, read_record


ROUTE = "to the supervised BTS lead Claude session (bts-lead), or to Eric if the session is unavailable."
KEEP = "Keep the active cache, raw feeds, any named evidence copy and other available copies intact for review."
KINDS = {"unreadable_cache": "unreadable cache", "invalid_entries": "invalid entries"}
STATUS_UNREADABLE = "status unreadable"
log = logging.getLogger(__name__)


RECORD_NAME = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}\.(?:invalid_entries|unreadable_cache)\.[0-9a-f]{12}\.json")
FOLDER = "the status folder (health_state/probable_pitcher_cache beside the picks directory)"


# W15b round 1 (amendment 4): no logged or sent W15 text carries text read from disk or configuration
# (paths, file names, raw field values). The scheduler journal is scanned for OOM words by the shadow
# check and alert text by the attention policy; fixed words, numbers and our own strict names cannot match.
def log_unreadable(what):
    log.warning("Probable-pitcher cache status unreadable: %s. Send this alert %s", what, ROUTE)


def entry_label(name, day):
    if RECORD_NAME.fullmatch(name):
        return f"record {name}"
    return f"an entry for {day} whose name is not a status record name"


def message(row):
    at = datetime.fromisoformat(row["recorded_at"]).isoformat()
    if row["preserved"] is None:
        evidence = "This observation names no evidence copy. That run left the active cache unchanged."
    else:
        evidence = (f"That run recorded an evidence copy named {row['preserved']} "
                    "in the rejected folder beside the cache.")

    rescanned = row["rescanned"]
    if row["reason"] == "unreadable_cache":
        noun = "game" if rescanned == 1 else "games"
        return (
            f"Probable-pitcher cache was unreadable at {at}. "
            f"That run started from an empty lookup and built usable entries from raw feeds for {rescanned} {noun}. "
            f"{KEEP} Other games may have been missing from that run's lookup. "
            f"{evidence} Send this alert immediately {ROUTE} "
            "Ask them to preserve and verify any usable cache copy in the current or previous R2 manifest "
            "before the next successful sync. "
            "An unreferenced content-addressed cache object can be pruned once it is at least seven days old."
        )

    dropped = row["dropped"]
    noun = "entry" if dropped == 1 else "entries"
    ids = ", ".join(map(str, row["game_ids"]))
    return (
        f"Probable-pitcher cache at {at} had {dropped} invalid {noun} (first game IDs {ids}). "
        f"Of those entries, {row['recovered_from_cache']} had a valid cache alias "
        f"and {rescanned} had a usable raw feed. "
        f"No usable lookup was found for {row['unusable_feed']} with unusable or unverified feeds "
        f"and {row['without_feed']} with missing feeds. "
        f"{evidence} This describes the cache at the recorded time; health may show this observation "
        f"again on the next processed day. {KEEP} Send this alert promptly {ROUTE} "
        "Ask them to review the latest preview and any pending pick before the next scheduled selection. "
        "If a pick was already sent or entered, report that and preserve its records."
    )


def summarize(today, identities, kinds):
    # Fixed words, a count and fixed labels only: no path or other text read from disk, so the
    # summary can never be OOM evidence for the attention policy (W15b) and stays under 600 characters.
    count = len(identities)
    noun = "observation" if count == 1 else "observations"
    labels = ", ".join(sorted(kinds))
    text = (
        f"Probable-pitcher cache: {count} {noun} ({labels}) recorded today or yesterday; "
        "details in this health run's log and in health_state/probable_pitcher_cache beside the picks directory. "
        "Keep the active cache, raw feeds, evidence copies and other copies intact. "
        "Send this alert promptly to the supervised BTS lead Claude session (bts-lead), "
        "or to Eric if it is unavailable, to review the latest preview and any pending pick before "
        "the next selection; for an unreadable cache, to secure a usable R2 copy before the next successful sync."
    )
    payload = json.dumps(sorted(identities), separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:16]
    return Alert("WARN", SOURCE, text, incident_key=f"{SOURCE}:{today}:{digest}")


def check(status_dir: Path, today: date) -> list[Alert]:
    try:
        if not directory_exists(status_dir):
            return []
        paths = sorted(status_dir.iterdir())
    except OSError:
        log_unreadable(FOLDER)
        return [summarize(today, [status_dir.name], {STATUS_UNREADABLE})]

    days = (str(today), str(today - timedelta(days=1)))
    identities = []
    kinds = set()
    for path in paths:
        day = next((day for day in days if fnmatchcase(path.name, f"{day}.*.json")), None)
        if day is None:
            continue
        try:
            row = read_record(path)
        except (OSError, ValueError, OverflowError, RecursionError):
            log_unreadable(entry_label(path.name, day))
            identities.append(path.name)
            kinds.add(STATUS_UNREADABLE)
            continue
        log.warning("%s", message(row))
        identities.append(row["identity"])
        kinds.add(KINDS[row["reason"]])
    return [summarize(today, identities, kinds)] if identities else []
