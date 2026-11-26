"""Discover BTS API endpoints by replaying a logged-in browser session.

Reads session cookies from the platform-native keychain (created by
`scripts/capture_bts_cookies.py`), launches a headless Chromium with
those cookies pre-set, navigates the BTS app, clicks through the
leaderboard tabs and a top user's drilldown, and records every XHR/fetch
network request with method, URL, status, and a truncated response preview.

Output: writes a JSON report to `--out` (default: /tmp/bts_endpoints.json)
listing all discovered endpoints. The report is read by humans to identify
which URLs to bake into `src/bts/leaderboard/endpoints.py`.

Usage:
    UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/discover_bts_endpoints.py
    # optional: --headed for visible browser, --out /tmp/foo.json
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright


KEYCHAIN_ACCOUNT = "claude-cli"
DEFAULT_NAME = "mlb-bts-session-cookies"
BTS_URL = "https://www.mlb.com/apps/beat-the-streak/game"


def load_cookies(name: str) -> list[dict]:
    """Load cookies from platform-native keychain and return Playwright cookie list."""
    if sys.platform == "darwin":
        out = subprocess.check_output(
            ["security", "find-generic-password", "-a", KEYCHAIN_ACCOUNT, "-s", name, "-w"]
        )
        raw = out.decode().strip()
        # `security -w` may emit hex when it thinks the password is binary
        # (any byte with the high bit set triggers this). Detect + decode.
        if re.fullmatch(r"[0-9a-fA-F]+", raw) and len(raw) % 2 == 0:
            raw = bytes.fromhex(raw).decode("utf-8")
    elif sys.platform.startswith("linux"):
        raw = subprocess.check_output(["pass", "show", name]).decode().strip()
    else:
        raise RuntimeError(f"unknown platform: {sys.platform}")
    cookies = json.loads(raw)
    # Playwright's add_cookies expects sameSite to be one of: 'Strict', 'Lax', 'None'
    # Some MLB cookies use 'no_restriction' or '' — normalize.
    for c in cookies:
        ss = c.get("sameSite")
        if ss in (None, "", "no_restriction", "unspecified"):
            c["sameSite"] = "Lax"
        elif ss.lower() == "lax":
            c["sameSite"] = "Lax"
        elif ss.lower() == "strict":
            c["sameSite"] = "Strict"
        elif ss.lower() == "none":
            c["sameSite"] = "None"
    return cookies


def is_interesting(url: str) -> bool:
    """Heuristic: URLs that look like they might be data APIs we care about."""
    # Skip image/font/static asset patterns
    skip_patterns = [
        r"\.png", r"\.jpg", r"\.jpeg", r"\.gif", r"\.svg", r"\.webp",
        r"\.woff", r"\.woff2", r"\.ttf", r"\.eot",
        r"\.css", r"\.js$", r"\.js\?",
        r"google-analytics", r"googletag", r"doubleclick",
        r"newrelic", r"sentry\.io", r"chartbeat",
        r"intercom", r"segment\.com", r"branch\.io",
    ]
    for p in skip_patterns:
        if re.search(p, url, re.IGNORECASE):
            return False
    # Want anything API-shaped
    return any(k in url.lower() for k in [
        "api", "/bts", "leaderboard", "streak", "pick", "user", "stats",
        "graphql", "feed", "json",
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=DEFAULT_NAME)
    ap.add_argument("--out", default="/tmp/bts_endpoints.json")
    ap.add_argument("--headed", action="store_true",
                    help="show browser window (default: headless)")
    ap.add_argument("--wait-seconds", type=int, default=8,
                    help="how long to dwell on each tab to let it load (default: 8)")
    args = ap.parse_args()

    cookies = load_cookies(args.name)
    print(f"loaded {len(cookies)} cookies from keychain", file=sys.stderr)

    requests_log: list[dict] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not args.headed)
        context = browser.new_context()
        context.add_cookies(cookies)
        page = context.new_page()

        # Listen for all responses; capture interesting ones
        def on_response(response):
            try:
                url = response.url
                if not is_interesting(url):
                    return
                req = response.request
                entry = {
                    "method": req.method,
                    "url": url,
                    "status": response.status,
                    "content_type": response.headers.get("content-type", ""),
                    "request_headers": {
                        k: v for k, v in req.headers.items()
                        if k.lower() in ("authorization", "x-api-key", "accept")
                    },
                }
                # Try to capture a body preview (only for JSON-ish responses)
                ct = entry["content_type"].lower()
                if "json" in ct and response.status == 200:
                    try:
                        body = response.text()
                        entry["body_full"] = body
                        entry["body_len"] = len(body)
                    except Exception as e:
                        entry["body_error"] = str(e)
                requests_log.append(entry)
            except Exception as e:
                # Don't let listener errors abort the run
                requests_log.append({"listener_error": str(e), "url": getattr(response, "url", "?")})

        page.on("response", on_response)

        # Navigate and explore
        print(f"navigating to {BTS_URL}...", file=sys.stderr)
        page.goto(BTS_URL, wait_until="domcontentloaded", timeout=60_000)
        page.wait_for_timeout(args.wait_seconds * 1000)

        # Try to click the Leaderboard tab
        for selector in [
            "text=Leaderboard",
            "[data-testid*='leaderboard']",
            "button:has-text('Leaderboard')",
            "a:has-text('Leaderboard')",
        ]:
            try:
                page.click(selector, timeout=3_000)
                print(f"clicked Leaderboard via {selector!r}", file=sys.stderr)
                page.wait_for_timeout(args.wait_seconds * 1000)
                break
            except Exception:
                continue

        # Try clicking each leaderboard tab to capture each tab's API call
        for tab_label in ("Active Streak", "All Season", "All Time", "Yesterday"):
            for selector in [
                f"text={tab_label}",
                f"button:has-text('{tab_label}')",
                f"[role='tab']:has-text('{tab_label}')",
            ]:
                try:
                    page.click(selector, timeout=3_000)
                    print(f"clicked tab {tab_label!r} via {selector!r}", file=sys.stderr)
                    page.wait_for_timeout(4_000)
                    break
                except Exception:
                    continue

        # Make sure we're back on Active Streak before drilldown
        try:
            page.click("text=Active Streak", timeout=3_000)
            page.wait_for_timeout(3_000)
        except Exception:
            pass

        # Try to drill into rank-1 user. The leaderboard is a list of clickable
        # rows; the chevron arrow on the right indicates clickability. Try by
        # the actual top-1 username from our earlier capture (tombrady12) AND
        # several generic patterns.
        clicked = False
        drilldown_selectors = [
            "text=tombrady12",
            "[role='link']:has-text('tombrady12')",
            "button:has-text('tombrady12')",
            ".leaderboard-row >> nth=0",
            "[class*='leaderboard'] [class*='row'] >> nth=0",
            "[class*='rank-row'] >> nth=0",
            # JS-driven: click the first element under the leaderboard scroll container
            "[class*='leaderboard'] >> button >> nth=0",
            # As a last resort, any element on the page that contains '35' (top streak)
            # combined with a usernamesque sibling
            "li:has-text('tombrady12')",
        ]
        for selector in drilldown_selectors:
            try:
                page.click(selector, timeout=3_000)
                print(f"drilled in via {selector!r}", file=sys.stderr)
                page.wait_for_timeout(args.wait_seconds * 1000)
                clicked = True
                break
            except Exception:
                continue
        if not clicked:
            # Fallback: use JS to find a leaderboard row by text content and click it
            try:
                page.evaluate("""
                  (() => {
                    const candidates = [...document.querySelectorAll('*')]
                      .filter(el => el.textContent && el.textContent.trim() === 'tombrady12');
                    for (const c of candidates) {
                      // Walk up to find a clickable ancestor
                      let n = c;
                      for (let i = 0; i < 8 && n; i++) {
                        if (n.tagName === 'BUTTON' || n.tagName === 'A' || n.onclick || n.getAttribute('role') === 'button') {
                          n.click();
                          return n.tagName + '#' + (n.id || '') + '.' + (n.className || '');
                        }
                        n = n.parentElement;
                      }
                    }
                    return null;
                  })()
                """)
                print("fallback JS click attempted", file=sys.stderr)
                page.wait_for_timeout(args.wait_seconds * 1000)
            except Exception as e:
                print(f"fallback JS click failed: {e}", file=sys.stderr)

        # One more dwell to catch any late requests
        page.wait_for_timeout(3_000)

        browser.close()

    # Sort by URL host+path for readability
    requests_log.sort(key=lambda r: r.get("url", ""))

    out = Path(args.out)
    out.write_text(json.dumps(requests_log, indent=2, default=str))
    print(f"\nwrote {len(requests_log)} interesting requests to {out}", file=sys.stderr)

    # Print a compact summary to stdout
    seen = set()
    print("\n=== UNIQUE URLs (status / method / url) ===")
    for entry in requests_log:
        url = entry.get("url", "?")
        # Strip query string for dedup
        base = url.split("?")[0]
        if base in seen:
            continue
        seen.add(base)
        print(f"  {entry.get('status', '?'):>3} {entry.get('method', '?'):>4} {url}")


if __name__ == "__main__":
    main()
