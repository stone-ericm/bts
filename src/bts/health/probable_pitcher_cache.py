"""Read recent probable-pitcher cache observations for end-of-day health."""

from datetime import date, timedelta
from fnmatch import fnmatchcase
from pathlib import Path

from bts.health.alert import Alert
from bts.probable_pitcher_status import SOURCE, directory_exists, read_record


ROUTE = "to the supervised BTS lead Claude session (bts-lead), or to Eric if the session is unavailable."
KEEP = "Keep the active cache, raw feeds, any named evidence copy and other available copies intact for review."


def unreadable(path):
    return Alert(
        "WARN", SOURCE,
        f"Probable-pitcher cache status unreadable at {path}. Send this alert {ROUTE}",
        incident_key=f"{SOURCE}:status_unreadable:{path.name}",
    )


def message(row):
    at = row["recorded_at"]
    if row["preserved"] is None:
        evidence = "This observation names no evidence copy. That run left the active cache unchanged."
    else:
        copy = Path(row["cache_path"]).parent / "rejected" / row["preserved"]
        evidence = f"That run recorded an evidence copy at `{copy}`."

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


def check(status_dir: Path, today: date) -> list[Alert]:
    try:
        if not directory_exists(status_dir):
            return []
        paths = sorted(status_dir.iterdir())
    except OSError:
        return [unreadable(status_dir)]

    patterns = (f"{today}.*.json", f"{today - timedelta(days=1)}.*.json")
    alerts = []
    for path in paths:
        if not any(fnmatchcase(path.name, pattern) for pattern in patterns):
            continue
        try:
            row = read_record(path)
        except (OSError, ValueError, OverflowError, RecursionError):
            alerts.append(unreadable(path))
            continue
        alerts.append(Alert(
            "WARN", SOURCE, message(row),
            incident_key=f"{SOURCE}:{row['recorded_date_et']}:{row['identity']}",
        ))
    return alerts
