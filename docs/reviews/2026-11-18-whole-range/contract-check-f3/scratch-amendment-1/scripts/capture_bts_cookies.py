"""One-time interactive cookie capture for MLB.com BTS.

Opens Playwright Chromium, lets the user log in interactively, then stores
the resulting session cookies in the platform-native keychain:
  - macOS: Keychain via `security add-generic-password -a claude-cli -s <name>`
  - Linux: GNU `pass` insert
On any other platform, falls back to printing JSON to stdout.

Usage:
    UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/capture_bts_cookies.py
    # optional: --name to override the keychain entry name (default: mlb-bts-session-cookies)
    # optional: --print also writes JSON to stdout regardless of platform

Login manually in the browser window that opens. Once you can see
the leaderboard at https://www.mlb.com/apps/beat-the-streak/game,
press Enter in the terminal. Cookies are stored, then the browser closes.
"""
import argparse
import json
import subprocess
import sys

from playwright.sync_api import sync_playwright


DEFAULT_NAME = "mlb-bts-session-cookies"
KEYCHAIN_ACCOUNT = "claude-cli"


def store_cookies(cookies: list[dict], name: str) -> None:
    """Store cookies (as a JSON list) in the platform-native keychain."""
    payload = json.dumps(cookies, indent=2)
    if sys.platform == "darwin":
        # Delete any existing entry so add-generic-password doesn't conflict.
        subprocess.run(
            ["security", "delete-generic-password", "-a", KEYCHAIN_ACCOUNT, "-s", name],
            capture_output=True,
        )
        subprocess.run(
            ["security", "add-generic-password",
             "-a", KEYCHAIN_ACCOUNT, "-s", name, "-w", payload],
            check=True,
        )
        print(f"Saved {len(cookies)} cookies to macOS Keychain "
              f"(account={KEYCHAIN_ACCOUNT}, service={name})", file=sys.stderr)
    elif sys.platform.startswith("linux"):
        subprocess.run(["pass", "insert", "-m", name],
                       input=payload.encode(), check=True)
        print(f"Saved {len(cookies)} cookies to pass:{name}", file=sys.stderr)
    else:
        print(f"unknown platform {sys.platform}; printing cookies to stdout", file=sys.stderr)
        print(payload)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=DEFAULT_NAME,
                    help="keychain entry name (default: %(default)s)")
    ap.add_argument("--print", action="store_true", dest="also_print",
                    help="also print JSON to stdout regardless of platform")
    args = ap.parse_args()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        page.goto("https://www.mlb.com/apps/beat-the-streak/game")
        print("Browser opened. Log in to MLB.com, navigate to the leaderboard, "
              "then press Enter here...", file=sys.stderr)
        input()
        cookies = context.cookies()
        try:
            store_cookies(cookies, name=args.name)
        finally:
            if args.also_print:
                print(json.dumps(cookies, indent=2))
            browser.close()


if __name__ == "__main__":
    main()
