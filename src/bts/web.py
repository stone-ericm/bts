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


from zoneinfo import ZoneInfo

PICKS_DIR = Path("data/picks")
PORT = 3003
ET = ZoneInfo("America/New_York")

# MLB team abbreviation -> team ID (for logo URLs)
TEAM_IDS = {
    "ATH": 133, "ATL": 144, "AZ": 109, "BAL": 110, "BOS": 111,
    "CHC": 112, "CIN": 113, "CLE": 114, "COL": 115, "CWS": 145,
    "DET": 116, "HOU": 117, "KC": 118, "LAA": 108, "LAD": 119,
    "MIA": 146, "MIL": 158, "MIN": 142, "NYM": 121, "NYY": 147,
    "PHI": 143, "PIT": 134, "SD": 135, "SEA": 136, "SF": 137,
    "STL": 138, "TB": 139, "TEX": 140, "TOR": 141, "WSH": 120,
}


def _format_game_time(iso_utc: str) -> str:
    """Convert ISO UTC game time to '7:10 PM ET' format."""
    if not iso_utc:
        return ""
    try:
        dt = datetime.fromisoformat(iso_utc).astimezone(ET)
        return dt.strftime("%-I:%M %p ET")
    except (ValueError, TypeError):
        return ""


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

        result = p.get("result")
        if result == "hit":
            result_html = '<span class="result-hit">&#10003;</span>'
        elif result == "miss":
            result_html = '<span class="result-miss">&#10007;</span>'
        else:
            result_html = '<span class="result-pending">&ndash;</span>'

        logo = team_logo_url(team)
        logo_img = f'<img src="{logo}" class="team-logo" alt="{team}">' if logo else ""

        double = p.get("double_down")

        if date == today:
            row_class = "today"
        elif result == "hit":
            row_class = "row-hit"
        elif result == "miss":
            row_class = "row-miss"
        else:
            row_class = ""
        pick_rows += f"""
        <tr class="{row_class}">
            <td class="result-cell">{result_html}</td>
            <td class="date-cell">{date}</td>
            <td class="batter-cell">{logo_img} <strong>{name}</strong></td>
            <td class="matchup-cell">vs {pitcher}</td>
            <td class="pct-cell">{pct:.1%}</td>
            <td class="flags-cell">{flags_str}</td>
        </tr>"""

        if double:
            d_name = double.get("batter_name", "?")
            d_team = double.get("team", "?")
            d_pitcher = double.get("pitcher_name", "?")
            d_pct = double.get("p_game_hit", 0)
            d_flags = double.get("flags", [])
            d_flags_str = ", ".join(d_flags) if isinstance(d_flags, list) else str(d_flags)
            d_logo = team_logo_url(d_team)
            d_logo_img = f'<img src="{d_logo}" class="team-logo" alt="{d_team}">' if d_logo else ""
            pick_rows += f"""
        <tr class="{row_class} double-row">
            <td class="result-cell"><span class="double-plus">+</span></td>
            <td class="date-cell"></td>
            <td class="batter-cell">{d_logo_img} <strong>{d_name}</strong></td>
            <td class="matchup-cell">vs {d_pitcher}</td>
            <td class="pct-cell">{d_pct:.1%}</td>
            <td class="flags-cell">{d_flags_str}</td>
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
        dd = today_pick.get("double_down")
        t_logo = team_logo_url(tp.get("team", ""), size=72)
        t_logo_img = f'<img src="{t_logo}" class="hero-logo" alt="{tp.get("team", "")}">' if t_logo else ""
        t_time = _format_game_time(tp.get("game_time", ""))
        is_locked = today_pick.get("bluesky_posted", False)
        lock_badge = '<span class="lock-badge locked">LOCKED</span>' if is_locked else '<span class="lock-badge pending">PENDING</span>'
        label = ("TODAY'S PICKS" if dd else "TODAY'S PICK") + f" {lock_badge}"
        if dd:
            d_logo = team_logo_url(dd.get("team", ""), size=72)
            d_logo_img = f'<img src="{d_logo}" class="hero-logo" alt="{dd.get("team", "")}">' if d_logo else ""
            d_time = _format_game_time(dd.get("game_time", ""))
            p_both = tp.get('p_game_hit', 0) * dd.get('p_game_hit', 0)
            hero = f"""
        <div class="hero">
            <div class="hero-left">
                {t_logo_img}
            </div>
            <div class="hero-right">
                <div class="hero-label">{label}</div>
                <div class="hero-name">{tp.get('batter_name', '?')}</div>
                <div class="hero-detail">{tp.get('team', '?')} vs {tp.get('pitcher_name', '?')} · {t_time}</div>
            </div>
            <div class="hero-pct">{tp.get('p_game_hit', 0):.1%}</div>
        </div>
        <div class="hero hero-double">
            <div class="hero-left">
                {d_logo_img}
            </div>
            <div class="hero-right">
                <div class="hero-label">DOUBLE DOWN · P(BOTH) {p_both:.1%}</div>
                <div class="hero-name">{dd.get('batter_name', '?')}</div>
                <div class="hero-detail">{dd.get('team', '?')} vs {dd.get('pitcher_name', '?')} · {d_time}</div>
            </div>
            <div class="hero-pct">{dd.get('p_game_hit', 0):.1%}</div>
        </div>"""
        else:
            hero = f"""
        <div class="hero">
            <div class="hero-left">
                {t_logo_img}
            </div>
            <div class="hero-right">
                <div class="hero-label">{label}</div>
                <div class="hero-name">{tp.get('batter_name', '?')}</div>
                <div class="hero-detail">{tp.get('team', '?')} vs {tp.get('pitcher_name', '?')} · {t_time}</div>
            </div>
            <div class="hero-pct">{tp.get('p_game_hit', 0):.1%}</div>
        </div>"""
    else:
        hero = """
        <div class="hero">
            <div class="hero-right">
                <div class="hero-label">TODAY'S PICK</div>
                <div class="hero-name">Waiting for lineups</div>
                <div class="hero-detail">Scheduler checks lineups 45 min before each game</div>
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
        .hero-double {{ margin-top: -8px; border-top: none; border-left-color: #041E42; }}
        .lock-badge {{ font-size: 0.75em; padding: 2px 8px; border-radius: 4px;
                       letter-spacing: 1px; vertical-align: middle; margin-left: 8px; }}
        .lock-badge.locked {{ background: #2e7d32; color: #fff; }}
        .lock-badge.pending {{ background: #f57c00; color: #fff; }}

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
        tr.today {{ background: #fff8e1; }}
        tr.today:hover {{ background: #fff3c4; }}
        tr.row-hit {{ background: #e8f5e9; }}
        tr.row-hit:hover {{ background: #c8e6c9; }}
        tr.row-miss {{ background: #fce4ec; }}
        tr.row-miss:hover {{ background: #f8bbd0; }}

        table {{ table-layout: fixed; }}
        col.col-result {{ width: 36px; }}
        col.col-date {{ width: 95px; }}
        col.col-batter {{ width: 30%; }}
        col.col-matchup {{ width: 28%; }}
        col.col-pct {{ width: 65px; }}
        col.col-flags {{ width: 85px; }}

        .team-logo {{ width: 24px; height: 24px; vertical-align: middle; margin-right: 6px; }}
        .batter-cell strong {{ color: #041E42; }}
        .batter-cell, .matchup-cell {{ overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
        .matchup-cell {{ color: #666; }}
        .pct-cell {{ color: #002D72; font-weight: 600; font-variant-numeric: tabular-nums; }}
        .double-row td {{ border-top: none; padding-top: 2px; }}
        .double-plus {{ color: #D50032; font-weight: 800; font-size: 1.2em; }}
        .flags-cell {{ color: #999; font-size: 0.8em; }}
        .date-cell {{ color: #888; font-variant-numeric: tabular-nums; }}
        .result-cell {{ text-align: center; font-size: 1.1em; }}
        .result-hit {{ color: #2e7d32; font-weight: 800; }}
        .result-miss {{ color: #D50032; font-weight: 800; }}
        .result-pending {{ color: #ccc; }}

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
            .flags-cell {{ display: none; }}
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
                    <a href="https://www.mlb.com/apps/beat-the-streak/game">Play BTS</a> ·
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
                <col class="col-result"><col class="col-date"><col class="col-batter">
                <col class="col-matchup"><col class="col-pct"><col class="col-flags">
            </colgroup>
            <tr><th></th><th>Date</th><th>Batter</th><th>Matchup</th><th>P(Hit)</th><th>Flags</th></tr>
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
