"""Simple BTS dashboard — LAN-only web frontend.

Shows past picks, current streak, today's pick, and Bluesky posts.
Reads from data/picks/*.json (created by bts run).

Usage:
    UV_CACHE_DIR=/tmp/uv-cache uv run python -m bts.web
    # Serves on http://0.0.0.0:3003
"""

import json
from datetime import datetime, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.request import urlopen, Request


PICKS_DIR = Path("data/picks")
PORT = 3003


def load_all_picks():
    """Load all pick files, sorted by date descending."""
    picks = []
    for f in sorted(PICKS_DIR.glob("*.json"), reverse=True):
        if f.stem in ("streak", "automation"):
            continue
        try:
            data = json.loads(f.read_text())
            picks.append(data)
        except:
            pass
    return picks


def load_streak():
    streak_path = PICKS_DIR / "streak.json"
    if streak_path.exists():
        return json.loads(streak_path.read_text()).get("streak", 0)
    return 0


def fetch_bluesky_posts(limit=5):
    """Fetch recent posts from @beatthestreakbot.bsky.social."""
    try:
        url = ("https://public.api.bsky.app/xrpc/app.bsky.feed.getAuthorFeed"
               "?actor=beatthestreakbot.bsky.social&limit=" + str(limit))
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        data = json.loads(urlopen(req, timeout=10).read())
        posts = []
        for item in data.get("feed", []):
            post = item.get("post", {})
            record = post.get("record", {})
            posts.append({
                "text": record.get("text", ""),
                "created_at": record.get("createdAt", ""),
                "uri": post.get("uri", ""),
            })
        return posts
    except:
        return []


def render_page():
    picks = load_all_picks()
    streak = load_streak()
    posts = fetch_bluesky_posts()
    today = datetime.now().strftime("%Y-%m-%d")

    today_pick = None
    for p in picks:
        if p.get("date") == today:
            today_pick = p
            break

    # Build pick rows
    pick_rows = ""
    for p in picks[:30]:
        pick = p.get("pick", {})
        date = p.get("date", "?")
        name = pick.get("batter_name", "?")
        team = pick.get("team", "?")
        pitcher = pick.get("pitcher_name", "?")
        pct = pick.get("p_game_hit", 0)
        flags = pick.get("flags", [])
        flags_str = ", ".join(flags) if isinstance(flags, list) else str(flags)
        projected = "projected" in flags_str.lower()

        double = p.get("double_down")
        double_str = ""
        if double:
            double_str = f'<span class="double">+ {double.get("batter_name", "?")} ({double.get("p_game_hit", 0):.1%})</span>'

        row_class = "today" if date == today else ""
        pick_rows += f"""
        <tr class="{row_class}">
            <td>{date}</td>
            <td><strong>{name}</strong> <span class="team">({team})</span></td>
            <td>vs {pitcher}</td>
            <td>{pct:.1%}</td>
            <td>{double_str}</td>
            <td class="flags">{flags_str}</td>
        </tr>"""

    # Build Bluesky posts
    posts_html = ""
    for post in posts:
        created = post["created_at"][:10] if post["created_at"] else ""
        text = post["text"].replace("\n", "<br>")
        posts_html += f"""
        <div class="post">
            <div class="post-date">{created}</div>
            <div class="post-text">{text}</div>
        </div>"""

    # Today's pick hero
    hero = ""
    if today_pick:
        tp = today_pick["pick"]
        hero = f"""
        <div class="hero">
            <div class="hero-label">TODAY'S PICK</div>
            <div class="hero-name">{tp.get('batter_name', '?')}</div>
            <div class="hero-detail">{tp.get('team', '?')} vs {tp.get('pitcher_name', '?')}</div>
            <div class="hero-pct">{tp.get('p_game_hit', 0):.1%}</div>
        </div>"""
    else:
        hero = """
        <div class="hero">
            <div class="hero-label">TODAY'S PICK</div>
            <div class="hero-name">Pending...</div>
            <div class="hero-detail">Next run will generate today's pick</div>
        </div>"""

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Beat the Streak Bot</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, system-ui, sans-serif; background: #0a0a0a; color: #e0e0e0; }}
        .container {{ max-width: 900px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #4ade80; font-size: 1.5em; margin-bottom: 5px; }}
        h2 {{ color: #888; font-size: 1.1em; margin: 30px 0 10px; }}
        .subtitle {{ color: #666; font-size: 0.9em; margin-bottom: 20px; }}

        .streak-box {{ display: inline-block; background: #1a1a2e; border: 1px solid #4ade80;
                       border-radius: 8px; padding: 15px 30px; margin: 15px 0; }}
        .streak-label {{ color: #888; font-size: 0.8em; text-transform: uppercase; }}
        .streak-number {{ color: #4ade80; font-size: 3em; font-weight: bold; }}

        .hero {{ background: #1a1a2e; border-radius: 8px; padding: 20px; margin: 15px 0;
                 border-left: 4px solid #4ade80; }}
        .hero-label {{ color: #4ade80; font-size: 0.75em; text-transform: uppercase; letter-spacing: 2px; }}
        .hero-name {{ font-size: 1.8em; font-weight: bold; margin: 5px 0; }}
        .hero-detail {{ color: #888; }}
        .hero-pct {{ color: #4ade80; font-size: 1.5em; font-weight: bold; margin-top: 5px; }}

        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th {{ text-align: left; color: #888; font-size: 0.8em; text-transform: uppercase;
              padding: 8px; border-bottom: 1px solid #333; }}
        td {{ padding: 8px; border-bottom: 1px solid #1a1a1a; font-size: 0.9em; }}
        tr.today {{ background: #1a2e1a; }}
        .team {{ color: #888; }}
        .double {{ color: #f59e0b; }}
        .flags {{ color: #666; font-size: 0.8em; }}

        .posts {{ margin-top: 10px; }}
        .post {{ background: #1a1a2e; border-radius: 8px; padding: 15px; margin: 8px 0; }}
        .post-date {{ color: #666; font-size: 0.8em; }}
        .post-text {{ margin-top: 5px; line-height: 1.4; }}

        .footer {{ color: #444; font-size: 0.8em; margin-top: 30px; text-align: center; }}
        a {{ color: #4ade80; text-decoration: none; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Beat the Streak Bot</h1>
        <div class="subtitle">
            14 features · 12-model blend · 86.9% P@1 ·
            <a href="https://bsky.app/profile/beatthestreakbot.bsky.social">@beatthestreakbot</a> ·
            <a href="https://github.com/stone-ericm/bts">GitHub</a>
        </div>

        <div class="streak-box">
            <div class="streak-label">Current Streak</div>
            <div class="streak-number">{streak}</div>
        </div>

        {hero}

        <h2>Pick History</h2>
        <table>
            <tr><th>Date</th><th>Batter</th><th>Matchup</th><th>P(Hit)</th><th>Double</th><th>Flags</th></tr>
            {pick_rows}
        </table>

        <h2>Bluesky Posts</h2>
        <div class="posts">
            {posts_html if posts_html else '<div class="post">No posts loaded</div>'}
        </div>

        <div class="footer">
            Updated {datetime.now().strftime('%Y-%m-%d %H:%M ET')} · LAN only
        </div>
    </div>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(render_page().encode())

    def log_message(self, format, *args):
        pass  # Suppress logs


def main():
    server = HTTPServer(("0.0.0.0", PORT), Handler)
    print(f"BTS Dashboard running at http://0.0.0.0:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
