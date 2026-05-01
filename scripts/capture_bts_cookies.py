"""One-time interactive cookie capture for MLB.com BTS.

Opens Playwright Chromium, lets the user log in interactively, then
serializes session cookies to stdout (or to `pass` if PASS_STORE_KEY
env var is set).

Usage:
    UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/capture_bts_cookies.py

Login manually in the browser window that opens. Once you can see
the leaderboard at https://www.mlb.com/apps/beat-the-streak/game,
press Enter in the terminal. Cookies are extracted and printed as JSON.
"""
import json
import os
import subprocess
import sys

from playwright.sync_api import sync_playwright


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        page.goto("https://www.mlb.com/apps/beat-the-streak/game")
        print("Browser opened. Log in to MLB.com, navigate to the leaderboard, then press Enter here...", file=sys.stderr)
        input()
        cookies = context.cookies()
        cookie_json = json.dumps(cookies, indent=2)

        pass_key = os.environ.get("PASS_STORE_KEY")
        if pass_key:
            subprocess.run(["pass", "insert", "-m", pass_key], input=cookie_json.encode(), check=True)
            print(f"Saved {len(cookies)} cookies to pass:{pass_key}", file=sys.stderr)
        else:
            print(cookie_json)
        browser.close()


if __name__ == "__main__":
    main()
