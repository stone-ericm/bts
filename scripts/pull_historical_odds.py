"""Pull historical batter_hits props from The Odds API.

Uses per-game timestamps (commence_time - 2 hours) for consistent offset.
Saves raw JSON per date to data/external/odds/v2/{date}.json.
Resumable — skips dates that already have data.

Usage:
    UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/pull_historical_odds.py
"""
import json
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import HTTPError

API_BASE = "https://api.the-odds-api.com/v4"
SPORT = "baseball_mlb"
MARKET = "batter_hits"
OFFSET_HOURS = 2  # Pull odds this many hours before game start

# MLB regular season date ranges — batter_hits props available from Sept 2023
SEASON_RANGES = {
    2023: ("2023-09-01", "2023-10-01"),
    2024: ("2024-03-20", "2024-09-29"),
    2025: ("2025-03-20", "2025-09-28"),
}


def get_api_key():
    return subprocess.run(
        ["security", "find-generic-password", "-a", "claude-cli", "-s", "odds-api-key", "-w"],
        capture_output=True, text=True,
    ).stdout.strip()


def pull_date(date_str: str, api_key: str) -> dict:
    """Pull batter_hits props for all games on a date using per-game timestamps."""

    # First get the events to find commence times
    # Use midday as the discovery timestamp
    disc_ts = f"{date_str}T17:00:00Z"
    events_url = (
        f"{API_BASE}/historical/sports/{SPORT}/events"
        f"?apiKey={api_key}&date={disc_ts}"
    )
    req = Request(events_url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        resp = urlopen(req, timeout=30)
        remaining = resp.headers.get("x-requests-remaining", "?")
        events_data = json.loads(resp.read())
    except HTTPError as e:
        if e.code == 422:
            return {"date": date_str, "error": "no_data", "events": []}
        raise

    events = events_data.get("data", [])
    if not events:
        return {"date": date_str, "events": [], "credits_remaining": remaining}

    # Filter to events on THIS date only (the API can return next-day games)
    day_events = []
    for e in events:
        ct = e.get("commence_time", "")
        if ct.startswith(date_str):
            day_events.append(e)
        else:
            # Check if it's a late-night game that spills into next day UTC
            # but is still on the same calendar day ET (UTC-4)
            try:
                utc = datetime.fromisoformat(ct.replace("Z", "+00:00"))
                et = utc - timedelta(hours=4)
                if et.strftime("%Y-%m-%d") == date_str:
                    day_events.append(e)
            except:
                pass

    # Pull odds per game at (commence_time - OFFSET_HOURS)
    results = []
    for event in day_events:
        eid = event["id"]
        ct = event.get("commence_time", "")

        try:
            commence = datetime.fromisoformat(ct.replace("Z", "+00:00"))
            snapshot = commence - timedelta(hours=OFFSET_HOURS)
            snapshot_str = snapshot.strftime("%Y-%m-%dT%H:%M:%SZ")
        except:
            continue

        props_url = (
            f"{API_BASE}/historical/sports/{SPORT}/events/{eid}/odds"
            f"?apiKey={api_key}&regions=us&markets={MARKET}"
            f"&oddsFormat=american&date={snapshot_str}"
        )
        req = Request(props_url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            resp = urlopen(req, timeout=30)
            remaining = resp.headers.get("x-requests-remaining", "?")
            props_data = json.loads(resp.read())
            results.append({
                "event_id": eid,
                "away_team": event.get("away_team"),
                "home_team": event.get("home_team"),
                "commence_time": ct,
                "snapshot_time": snapshot_str,
                "offset_hours": OFFSET_HOURS,
                "props": props_data.get("data", {}),
            })
        except HTTPError as e:
            if e.code == 422:
                results.append({
                    "event_id": eid,
                    "away_team": event.get("away_team"),
                    "home_team": event.get("home_team"),
                    "commence_time": ct,
                    "error": "no_props",
                })
            else:
                results.append({
                    "event_id": eid,
                    "error": str(e),
                })

        time.sleep(0.1)

    return {
        "date": date_str,
        "offset_hours": OFFSET_HOURS,
        "n_events": len(day_events),
        "n_with_props": sum(1 for r in results if "props" in r),
        "events": results,
        "credits_remaining": remaining,
    }


def main():
    api_key = get_api_key()
    output_dir = Path("data/external/odds/v2")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_dates = []
    for season, (start, end) in SEASON_RANGES.items():
        current = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")
        while current <= end_dt:
            all_dates.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)

    print(f"Total dates: {len(all_dates)}")

    existing = {p.stem for p in output_dir.glob("*.json")}
    to_pull = [d for d in all_dates if d not in existing]
    print(f"Already downloaded: {len(existing)}")
    print(f"Remaining: {len(to_pull)}")

    if not to_pull:
        print("All dates already downloaded!")
        return

    pulled = 0
    errors = 0
    for i, date_str in enumerate(to_pull):
        try:
            result = pull_date(date_str, api_key)
            out_path = output_dir / f"{date_str}.json"
            out_path.write_text(json.dumps(result, indent=2))

            n_props = result.get("n_with_props", 0)
            remaining = result.get("credits_remaining", "?")
            pulled += 1

            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [{i+1}/{len(to_pull)}] {date_str}: {n_props} games with props "
                      f"(credits remaining: {remaining})", flush=True)

        except Exception as e:
            errors += 1
            print(f"  ERROR {date_str}: {e}", flush=True)
            out_path = output_dir / f"{date_str}.json"
            out_path.write_text(json.dumps({"date": date_str, "error": str(e)}))

            if "401" in str(e) or "403" in str(e):
                print("Auth error — stopping.", flush=True)
                break
            time.sleep(1)

        time.sleep(0.2)

    print(f"\nDone. Pulled: {pulled}, Errors: {errors}")
    print(f"Data saved to {output_dir}/")


if __name__ == "__main__":
    main()
