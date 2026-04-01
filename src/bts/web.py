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

# MLB team abbreviation -> team ID (for logo URLs)
TEAM_IDS = {
    "ATH": 133, "ATL": 144, "AZ": 109, "BAL": 110, "BOS": 111,
    "CHC": 112, "CIN": 113, "CLE": 114, "COL": 115, "CWS": 145,
    "DET": 116, "HOU": 117, "KC": 118, "LAA": 108, "LAD": 119,
    "MIA": 146, "MIL": 158, "MIN": 142, "NYM": 121, "NYY": 147,
    "PHI": 143, "PIT": 134, "SD": 135, "SEA": 136, "SF": 137,
    "STL": 138, "TB": 139, "TEX": 140, "TOR": 141, "WSH": 120,
}


def team_logo_url(abbrev, size=40):
    tid = TEAM_IDS.get(abbrev, "")
    if tid:
        return f"https://www.mlbstatic.com/team-logos/{tid}.svg"
    return ""


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


def _at_uri_to_web_url(at_uri, handle="beatthestreakbot.bsky.social"):
    """Convert at://did:plc:xxx/app.bsky.feed.post/rkey to bsky.app URL."""
    parts = at_uri.split("/")
    if len(parts) >= 5:
        rkey = parts[-1]
        return f"https://bsky.app/profile/{handle}/post/{rkey}"
    return ""


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
            at_uri = post.get("uri", "")
            posts.append({
                "text": record.get("text", ""),
                "created_at": record.get("createdAt", ""),
                "uri": at_uri,
                "web_url": _at_uri_to_web_url(at_uri),
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

        logo = team_logo_url(team)
        logo_img = f'<img src="{logo}" class="team-logo" alt="{team}">' if logo else ""

        double = p.get("double_down")
        double_str = ""
        if double:
            d_team = double.get("team", "?")
            d_logo = team_logo_url(d_team)
            d_logo_img = f'<img src="{d_logo}" class="team-logo-sm" alt="{d_team}">' if d_logo else ""
            double_str = f'{d_logo_img} <span class="double">{double.get("batter_name", "?")} ({double.get("p_game_hit", 0):.1%})</span>'

        row_class = "today" if date == today else ""
        pick_rows += f"""
        <tr class="{row_class}">
            <td class="date-cell">{date}</td>
            <td class="batter-cell">{logo_img} <strong>{name}</strong></td>
            <td class="matchup-cell">vs {pitcher}</td>
            <td class="pct-cell">{pct:.1%}</td>
            <td class="double-cell">{double_str}</td>
            <td class="flags-cell">{flags_str}</td>
        </tr>"""

    # Build Bluesky posts as official embeds
    posts_html = ""
    for post in posts:
        web_url = post.get("web_url", "")
        if web_url:
            posts_html += f"""
        <div class="post-embed">
            <blockquote class="bluesky-embed" data-bluesky-uri="{post['uri']}"
                data-bluesky-cid="">
                <p>{post['text']}</p>
                &mdash; Beat the Streak Bot
                (<a href="{web_url}">link</a>)
            </blockquote>
        </div>"""
        else:
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
        t_logo = team_logo_url(tp.get("team", ""), size=72)
        t_logo_img = f'<img src="{t_logo}" class="hero-logo" alt="{tp.get("team", "")}">' if t_logo else ""
        hero = f"""
        <div class="hero">
            <div class="hero-left">
                {t_logo_img}
            </div>
            <div class="hero-right">
                <div class="hero-label">TODAY'S PICK</div>
                <div class="hero-name">{tp.get('batter_name', '?')}</div>
                <div class="hero-detail">{tp.get('team', '?')} vs {tp.get('pitcher_name', '?')}</div>
            </div>
            <div class="hero-pct">{tp.get('p_game_hit', 0):.1%}</div>
        </div>"""
    else:
        hero = """
        <div class="hero">
            <div class="hero-right">
                <div class="hero-label">TODAY'S PICK</div>
                <div class="hero-name">Coming at 11 AM</div>
                <div class="hero-detail">Pi5 orchestrator runs at 11am, 4pm, and 7:30pm ET</div>
            </div>
        </div>"""

    # MLB logo SVG (silhouette batter)
    mlb_logo = '<img src="https://www.mlbstatic.com/team-logos/league-on-dark/1.svg" class="mlb-logo" alt="MLB">'

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Beat the Streak Bot</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Inter', -apple-system, system-ui, sans-serif;
                background: #f0f2f5; color: #1a1a2e; }}

        .topbar {{ background: #041E42; border-bottom: 3px solid #D50032;
                   padding: 8px 20px; display: flex; align-items: center; gap: 12px; }}
        .mlb-logo {{ height: 32px; opacity: 0.8; }}
        .topbar-title {{ font-size: 0.85em; color: #a0b0cc; font-weight: 500; letter-spacing: 0.5px; }}

        .container {{ max-width: 960px; margin: 0 auto; padding: 20px; }}

        .header {{ display: flex; align-items: center; justify-content: space-between;
                   margin-bottom: 20px; }}
        .header-left h1 {{ color: #041E42; font-size: 1.6em; font-weight: 800; }}
        .header-left h1 span {{ color: #D50032; }}
        .subtitle {{ color: #666; font-size: 0.85em; margin-top: 4px; }}
        .subtitle a {{ color: #002D72; text-decoration: none; font-weight: 500; }}
        .subtitle a:hover {{ text-decoration: underline; }}

        .streak-box {{ background: #041E42; border: 2px solid #D50032;
                       border-radius: 12px; padding: 15px 30px; text-align: center; }}
        .streak-label {{ color: #8899bb; font-size: 0.7em; text-transform: uppercase;
                         letter-spacing: 2px; font-weight: 600; }}
        .streak-number {{ color: #fff; font-size: 3.2em; font-weight: 800;
                          line-height: 1.1; }}
        .streak-sub {{ color: #D50032; font-size: 0.7em; text-transform: uppercase;
                       letter-spacing: 1px; font-weight: 600; }}

        .hero {{ background: #fff; border-radius: 12px; padding: 24px; margin: 20px 0;
                 border: 1px solid #ddd; border-left: 4px solid #D50032;
                 display: flex; align-items: center; gap: 20px;
                 box-shadow: 0 2px 8px rgba(0,0,0,0.06); }}
        .hero-left {{ flex-shrink: 0; }}
        .hero-logo {{ width: 72px; height: 72px; }}
        .hero-right {{ flex: 1; }}
        .hero-label {{ color: #D50032; font-size: 0.7em; text-transform: uppercase;
                       letter-spacing: 3px; font-weight: 700; }}
        .hero-name {{ font-size: 1.8em; font-weight: 800; color: #041E42; margin: 4px 0; }}
        .hero-detail {{ color: #666; font-size: 0.95em; }}
        .hero-pct {{ font-size: 2.2em; font-weight: 800; color: #D50032;
                     flex-shrink: 0; }}

        .section-header {{ color: #041E42; font-size: 0.75em; text-transform: uppercase;
                           letter-spacing: 2px; font-weight: 700; margin: 28px 0 12px;
                           padding-bottom: 8px; border-bottom: 2px solid #ddd; }}

        table {{ width: 100%; border-collapse: collapse; background: #fff;
                 border-radius: 8px; overflow: hidden;
                 box-shadow: 0 2px 8px rgba(0,0,0,0.06); }}
        th {{ text-align: left; color: #041E42; font-size: 0.7em; text-transform: uppercase;
              letter-spacing: 1px; font-weight: 700; padding: 10px 8px;
              background: #f8f9fa; border-bottom: 2px solid #ddd; }}
        td {{ padding: 10px 8px; border-bottom: 1px solid #eee; font-size: 0.9em; }}
        tr:hover {{ background: #f0f4ff; }}
        tr.today {{ background: #e8f5e9; }}
        tr.today:hover {{ background: #dcedc8; }}

        table {{ table-layout: fixed; }}
        col.col-date {{ width: 100px; }}
        col.col-batter {{ width: 28%; }}
        col.col-matchup {{ width: 22%; }}
        col.col-pct {{ width: 70px; }}
        col.col-double {{ width: 22%; }}
        col.col-flags {{ width: 80px; }}

        .team-logo {{ width: 24px; height: 24px; vertical-align: middle; margin-right: 6px; }}
        .team-logo-sm {{ width: 18px; height: 18px; vertical-align: middle; margin-right: 4px; }}
        .batter-cell strong {{ color: #041E42; }}
        .batter-cell, .matchup-cell, .double-cell {{ overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
        .matchup-cell {{ color: #666; }}
        .pct-cell {{ color: #002D72; font-weight: 600; font-variant-numeric: tabular-nums; }}
        .double {{ color: #D50032; font-weight: 600; }}
        .flags-cell {{ color: #999; font-size: 0.8em; }}
        .date-cell {{ color: #888; font-variant-numeric: tabular-nums; }}

        .posts {{ margin-top: 8px; }}
        .post-embed {{ margin: 12px 0; }}
        .post-embed blockquote {{ max-width: 600px; }}
        .post {{ background: #fff; border-radius: 10px; padding: 16px; margin: 8px 0;
                 border: 1px solid #ddd; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }}
        .post-date {{ color: #888; font-size: 0.75em; font-weight: 600; }}
        .post-text {{ margin-top: 6px; line-height: 1.5; color: #333; }}

        .footer {{ color: #999; font-size: 0.75em; margin-top: 30px; text-align: center;
                   padding-top: 20px; border-top: 1px solid #ddd; }}

        @media (max-width: 640px) {{
            .hero {{ flex-direction: column; text-align: center; }}
            .hero-pct {{ margin-top: 10px; }}
            .header {{ flex-direction: column; gap: 12px; }}
            .flags-cell, .double-cell {{ display: none; }}
        }}
    </style>
</head>
<body>
    <div class="topbar">
        {mlb_logo}
        <span class="topbar-title">BEAT THE STREAK</span>
    </div>

    <div class="container">
        <div class="header">
            <div class="header-left">
                <h1>Beat the <span>Streak</span> Bot</h1>
                <div class="subtitle">
                    14 features · 12-model blend · 86.9% P@1 ·
                    <a href="https://bsky.app/profile/beatthestreakbot.bsky.social">@beatthestreakbot</a> ·
                    <a href="https://github.com/stone-ericm/bts">GitHub</a>
                </div>
            </div>
            <div class="streak-box">
                <div class="streak-label">Current Streak</div>
                <div class="streak-number">{streak}</div>
                <div class="streak-sub">Consecutive Hits</div>
            </div>
        </div>

        {hero}

        <div class="section-header">Pick History</div>
        <table>
            <colgroup>
                <col class="col-date"><col class="col-batter"><col class="col-matchup">
                <col class="col-pct"><col class="col-double"><col class="col-flags">
            </colgroup>
            <tr><th>Date</th><th>Batter</th><th>Matchup</th><th>P(Hit)</th><th>Double</th><th>Flags</th></tr>
            {pick_rows}
        </table>

        <div class="section-header">Bluesky Posts</div>
        <div class="posts">
            {posts_html if posts_html else '<div class="post">No posts loaded</div>'}
        </div>

        <div class="footer">
            Updated {datetime.now().strftime('%Y-%m-%d %H:%M ET')} · LAN only · Not affiliated with MLB
        </div>
    </div>
    <script src="https://embed.bsky.app/static/embed.js" async charset="utf-8"></script>
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
