"""Simple BTS dashboard — LAN-only web frontend.

Shows past picks, current streak, today's pick, and Bluesky posts.
Reads from data/picks/*.json (created by bts run).

Usage:
    UV_CACHE_DIR=/tmp/uv-cache uv run python -m bts.web
    # Serves on http://0.0.0.0:3003
"""

import json
import os
from datetime import datetime, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.request import urlopen, Request


from zoneinfo import ZoneInfo

from bts.heartbeat import read_heartbeat, is_heartbeat_fresh

PICKS_DIR = Path("data/picks")
HEARTBEAT_PATH = Path(os.environ.get("BTS_HEARTBEAT_PATH", "data/.heartbeat"))
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
        if f.stem in ("streak", "automation") or f.name.endswith(".shadow.json"):
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


def _ordinal(n: int) -> str:
    """Convert integer to ordinal: 1 -> '1st', 2 -> '2nd', etc."""
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    return f"{n}{['th', 'st', 'nd', 'rd'][n % 10] if n % 10 < 4 else 'th'}"


def _render_pitch_grid(pitches: list[dict]) -> str:
    """Render a 2-column numbered pitch grid as HTML."""
    if not pitches:
        return ""
    items = ""
    last_idx = len(pitches) - 1
    for i, p in enumerate(pitches):
        call = p.get("call", "?")
        is_strike = p.get("is_strike", False)
        num = p.get("number", i + 1)
        is_last = i == last_idx
        is_foul = call in ("F", "T")  # foul, foul tip
        in_play = call in ("X", "D")

        if is_strike or in_play:
            color = "#c41e3a"
        else:
            color = "#aaa"

        style = f"color:{color};"
        border_style = "border:1px solid #c41e3a;" if is_foul else "border:1px solid transparent;"
        weight = "font-weight:700;" if (is_last or in_play) else ""
        transform = "display:inline-block;transform:scaleX(-1);" if call == "Ч" else ""

        items += (
            f'<span style="font-size:9px;{style}{border_style}{weight}{transform}'
            f'padding:1px 2px;border-radius:2px;white-space:nowrap;">'
            f'{num}:{call}</span>'
        )

    # Wrap in a small flex container (2-col grid via wrapping)
    return (
        '<div style="display:flex;flex-wrap:wrap;gap:1px 3px;'
        'justify-content:flex-end;max-width:72px;">'
        + items
        + "</div>"
    )


def _render_diamond(pa: dict) -> str:
    """Render a 36x36 SVG baseball diamond for a completed PA."""
    # Coordinate mapping: MLB coordX/coordY are 0-250, map to SVG 4-36
    def _map(v, lo=4, hi=36):
        if v is None:
            return None
        return lo + (v / 250.0) * (hi - lo)

    hit_traj = pa.get("hit_trajectory") or {}
    coord_x = hit_traj.get("x")
    coord_y = hit_traj.get("y")
    is_hit = pa.get("is_hit", False)
    event_type = pa.get("event_type", "")

    # Base positions in SVG space (diamond rotated 45°)
    # Home=bottom, 1B=right, 2B=top, 3B=left (centered in 40px viewBox)
    cx, cy = 20, 20  # center
    r = 11  # radius from center to bases
    bases = {
        "1B": (cx + r, cy),
        "2B": (cx, cy - r),
        "3B": (cx - r, cy),
        "home": (cx, cy + r),
    }

    # Determine which bases are occupied after this PA
    runner_end_positions = set()
    for rm in pa.get("runners", []):
        end = rm.get("end")
        is_out = rm.get("is_out", False)
        if end and not is_out:
            runner_end_positions.add(end)
        # Scoring runner: end can be None when runner scored (common on HRs)
        if not is_out and end is None and rm.get("start") is not None:
            runner_end_positions.add("score")

    # Did the batter reach base?
    batter_reached = is_hit or event_type in (
        "walk", "hit_by_pitch", "intent_walk", "catcher_interf",
        "field_error",
    )

    # Draw diamond outline
    pts = " ".join(
        f"{bases[b][0]},{bases[b][1]}"
        for b in ("home", "1B", "2B", "3B")
    )
    svg = (
        f'<svg width="36" height="36" viewBox="4 4 32 32" '
        f'xmlns="http://www.w3.org/2000/svg" style="display:block;">'
        # Background
        f'<rect x="4" y="4" width="32" height="32" fill="none"/>'
        # Diamond outline
        f'<polygon points="{pts}" fill="none" stroke="#ccc" stroke-width="1"/>'
    )

    # Basepath lines for runner advancement
    base_order = ["home", "1B", "2B", "3B", "home"]
    if batter_reached:
        # Draw path segments for each base the runners reached
        reached = set()
        if batter_reached:
            reached.add("1B")
        if event_type == "home_run":
            reached.update(["1B", "2B", "3B"])
        for pos in runner_end_positions:
            if pos == "2B":
                reached.update(["1B", "2B"])
            elif pos == "3B":
                reached.update(["1B", "2B", "3B"])
            elif pos == "score":
                reached.update(["1B", "2B", "3B"])

        for i in range(len(base_order) - 1):
            b1, b2 = base_order[i], base_order[i + 1]
            if b1 in reached or b2 == "home":
                x1, y1 = bases[b1]
                x2, y2 = bases[b2]
                if b1 in reached:
                    svg += (
                        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                        f'stroke="#16a34a" stroke-width="1.5" opacity="0.5"/>'
                    )

    # Draw base squares (rotated 45° — diamonds)
    base_size = 3
    for base_name, (bx, by) in bases.items():
        if base_name == "home":
            filled = not batter_reached
            fill = "#999" if filled else "none"
            stroke = "#999"
            # Draw home plate as a small polygon
            svg += (
                f'<polygon points="{bx},{by-3} {bx+2.5},{by-1} {bx+2.5},{by+2} '
                f'{bx-2.5},{by+2} {bx-2.5},{by-1}" '
                f'fill="{fill}" stroke="{stroke}" stroke-width="0.8"/>'
            )
        else:
            is_occupied = base_name in runner_end_positions
            if is_occupied and is_hit:
                fill = "#16a34a"
            elif is_occupied:
                fill = "#333"
            else:
                fill = "none"
            # Rotated square (diamond shape)
            svg += (
                f'<rect x="{bx - base_size}" y="{by - base_size}" '
                f'width="{base_size*2}" height="{base_size*2}" '
                f'fill="{fill}" stroke="#999" stroke-width="0.8" '
                f'transform="rotate(45,{bx},{by})"/>'
            )

    # Trajectory line from home to hit coordinates
    if coord_x is not None and coord_y is not None:
        # MLB coordY increases downward from top of image, origin near plate
        # coordY=200 ≈ home plate area, coordY=0 ≈ outfield
        # Map: x 0-250 → SVG 4-36; y 0-250 → SVG 36-4 (inverted)
        tx = 4 + (coord_x / 250.0) * 32
        ty = 36 - (coord_y / 250.0) * 32
        color = "#16a34a" if is_hit else "#c41e3a"
        hx, hy = bases["home"]
        svg += (
            f'<line x1="{hx:.1f}" y1="{hy:.1f}" x2="{tx:.1f}" y2="{ty:.1f}" '
            f'stroke="{color}" stroke-width="1" stroke-dasharray="2,1" opacity="0.8"/>'
        )
        # Hit marker dot
        svg += (
            f'<circle cx="{tx:.1f}" cy="{ty:.1f}" r="1.5" '
            f'fill="{color}" opacity="0.9"/>'
        )

    svg += "</svg>"
    return svg


def _render_pa_cell(pa: dict | None, estimated_inning: str = "") -> str:
    """Render a single plate appearance as a <td> element."""
    if pa is None:
        # Upcoming PA placeholder
        style = (
            "border:1px dashed #ccc;color:#bbb;font-size:10px;"
            "vertical-align:top;padding:4px;width:100px;min-width:100px;"
            "text-align:center;"
        )
        inner = ""
        if estimated_inning:
            inner = (
                f'<div style="font-size:9px;color:#bbb;margin-top:4px;">'
                f'{estimated_inning}</div>'
            )
        return f'<td style="{style}">{inner}</td>'

    # In-progress PA — batter currently at bat
    if pa.get("in_progress"):
        pitches = pa.get("pitches", [])
        pitch_grid_html = _render_pitch_grid(pitches)
        count = f"{sum(1 for p in pitches if not p.get('is_strike'))}-{sum(1 for p in pitches if p.get('is_strike'))}"
        td_style = (
            "border:2px solid #f59e0b;vertical-align:top;padding:4px;"
            "width:100px;min-width:100px;background:rgba(245,158,11,0.06);"
            "position:relative;animation:pulse-border 2s ease-in-out infinite;"
        )
        inner = f"""<div style="display:flex;justify-content:space-between;align-items:flex-start;">
    <div><span style="font-size:13px;font-weight:700;color:#f59e0b;">AB</span>
    <div style="font-size:9px;color:#999;margin-top:1px;">{count}</div></div>
    <div>{pitch_grid_html}</div>
</div>"""
        return f'<td style="{td_style}">{inner}</td>'

    is_hit = pa.get("is_hit", False)
    result = pa.get("result", "?")
    out_number = pa.get("out_number")
    rbi = pa.get("rbi", 0)
    pitches = pa.get("pitches", [])

    bg = "rgba(34,197,94,0.08)" if is_hit else "transparent"
    result_color = "#16a34a" if is_hit else "#333"

    td_style = (
        f"border:1px solid #eee;vertical-align:top;padding:4px;"
        f"width:100px;min-width:100px;background:{bg};"
        f"position:relative;"
    )

    # Backwards К for called third strike
    is_backwards_k = result == "\u042f"  # Cyrillic Я used as backwards K
    result_display = result
    result_transform = ""
    if is_backwards_k:
        result_display = "K"
        result_transform = "display:inline-block;transform:scaleX(-1);"

    result_html = (
        f'<span style="font-size:13px;font-weight:700;color:{result_color};'
        f'{result_transform}">{result_display}</span>'
    )

    pitch_grid_html = _render_pitch_grid(pitches)
    diamond_html = _render_diamond(pa)

    # Out number: circled in MLB red
    out_html = ""
    if out_number is not None:
        out_html = (
            f'<span style="display:inline-flex;align-items:center;justify-content:center;'
            f'width:14px;height:14px;border-radius:50%;border:1.5px solid #c41e3a;'
            f'color:#c41e3a;font-size:9px;font-weight:700;">{out_number}</span>'
        )

    # RBI dots
    rbi_html = ""
    if rbi and rbi > 0:
        dots = "".join(
            f'<span style="display:inline-block;width:6px;height:6px;'
            f'border-radius:50%;background:#16a34a;margin-right:1px;"></span>'
            for _ in range(min(rbi, 4))
        )
        rbi_html = f'<div style="margin-top:2px;">{dots}</div>'

    # Layout: top row (result left, pitch grid right), bottom row (out+rbi left, diamond right)
    inner = f"""<div style="display:flex;justify-content:space-between;align-items:flex-start;gap:2px;">
    <div>{result_html}</div>
    <div>{pitch_grid_html}</div>
</div>
<div style="display:flex;justify-content:space-between;align-items:flex-end;margin-top:2px;">
    <div>
        {out_html}
        {rbi_html}
    </div>
    <div>{diamond_html}</div>
</div>"""

    return f'<td style="{td_style}">{inner}</td>'


def render_scorecard_section(scorecard_data: dict | None) -> str:
    """Render the full live scorecard HTML section."""
    if not scorecard_data:
        return ""

    game_status = scorecard_data.get("game_status")
    if game_status not in ("L", "F"):
        return ""

    inning = scorecard_data.get("inning", "")
    away_team = scorecard_data.get("away_team", "")
    home_team = scorecard_data.get("home_team", "")
    score = scorecard_data.get("score", {})
    batters = scorecard_data.get("batters", [])

    away_runs = score.get("away", 0)
    home_runs = score.get("home", 0)
    # score_label is set by merge_scorecards() when picks are in different games
    score_str = scorecard_data.get("score_label") or f"{away_team} {away_runs} – {home_runs} {home_team}"

    live_badge = ""
    if game_status == "L":
        live_badge = (
            '<span style="background:#c41e3a;color:#fff;font-size:9px;'
            'font-weight:700;padding:2px 6px;border-radius:3px;'
            'letter-spacing:1px;margin-right:8px;vertical-align:middle;">'
            'LIVE</span>'
        )
    elif game_status == "F":
        live_badge = (
            '<span style="background:#666;color:#fff;font-size:9px;'
            'font-weight:700;padding:2px 6px;border-radius:3px;'
            'letter-spacing:1px;margin-right:8px;vertical-align:middle;">'
            'FINAL</span>'
        )

    # score_label already includes per-game innings for double-downs
    has_score_label = scorecard_data.get("score_label") is not None
    inning_display = f" &middot; {inning}" if inning and game_status == "L" and not has_score_label else ""

    # Build table header: # | BATTERS | POS | up to 5 PA columns
    max_pas = max((len(b.get("pas", [])) for b in batters), default=0)
    num_pa_cols = max(5, max_pas)
    pa_headers = "".join(
        f'<th style="width:100px;text-align:center;padding:6px 4px;">PA {i+1}</th>'
        for i in range(num_pa_cols)
    )

    # Sticky column styles for horizontal scroll
    sticky_num = "position:sticky;left:0;z-index:2;background:#f8f9fa;"
    sticky_name = "position:sticky;left:28px;z-index:2;background:#f8f9fa;"
    sticky_pos = "position:sticky;left:148px;z-index:2;background:#f8f9fa;border-right:2px solid #ddd;"
    sticky_num_td = "position:sticky;left:0;z-index:1;background:#fff;"
    sticky_name_td = "position:sticky;left:28px;z-index:1;background:#fff;"
    sticky_pos_td = "position:sticky;left:148px;z-index:1;background:#fff;border-right:2px solid #ddd;"

    # Build batter rows
    batter_rows_html = ""
    for batter in batters:
        name = batter.get("name", "")
        position = batter.get("position", "")
        lineup_pos = batter.get("lineup_position", "")
        slash = batter.get("slash_line", "")
        pas = batter.get("pas", [])

        row_cells = ""
        for i in range(num_pa_cols):
            if i < len(pas):
                row_cells += _render_pa_cell(pas[i])
            else:
                # Estimate which inning this PA would occur in.
                # Each batter gets ~1 PA per 9 batters through the order.
                # First upcoming PA: current completed PAs + 1 → inning ≈ (pa_num) * 2 - 1
                pa_num = i + 1
                est_inning = pa_num * 2 - 1 if pa_num <= 5 else pa_num * 2
                est = f"~{_ordinal(est_inning)}" if i == len(pas) else ""
                row_cells += _render_pa_cell(None, estimated_inning=est)

        slash_html = (
            f'<div style="font-size:9px;color:#888;margin-top:1px;">{slash}</div>'
            if slash else ""
        )

        batter_rows_html += f"""<tr style="border-bottom:1px solid #eee;">
    <td style="padding:6px 8px;color:#888;font-size:11px;text-align:center;width:28px;{sticky_num_td}">{lineup_pos}</td>
    <td style="padding:6px 8px;min-width:120px;{sticky_name_td}">
        <div style="font-size:13px;font-weight:600;color:#041E42;">{name}</div>
        {slash_html}
    </td>
    <td style="padding:6px 8px;color:#888;font-size:11px;text-align:center;width:40px;{sticky_pos_td}">{position}</td>
    {row_cells}
</tr>"""

    # BTS status banner
    all_hits = all(any(pa.get("is_hit") for pa in b.get("pas", [])) for b in batters) if batters else False
    any_hits = any(any(pa.get("is_hit") for pa in b.get("pas", [])) for b in batters)
    has_pas = any(b.get("pas") for b in batters)

    if all_hits:
        banner_bg = "#d4edda"
        banner_color = "#155724"
        banner_text = "HIT! BTS pick successful" + (" — both batters!" if len(batters) > 1 else "")
    elif game_status == "F" and not all_hits:
        banner_bg = "#f8d7da"
        banner_color = "#721c24"
        banner_text = "Final — pick missed"
    elif any_hits and game_status == "L":
        banner_bg = "#fff3cd"
        banner_color = "#856404"
        banner_text = "Hit recorded — waiting on remaining batter"
    elif has_pas:
        banner_bg = "#e2e3e5"
        banner_color = "#495057"
        banner_text = "Game in progress — no hits yet"
    else:
        banner_bg = "#e2e3e5"
        banner_color = "#495057"
        banner_text = "Waiting for first plate appearance"

    banner_html = (
        f'<div style="margin-top:8px;padding:8px 12px;border-radius:6px;'
        f'background:{banner_bg};color:{banner_color};font-size:12px;font-weight:600;">'
        f'{banner_text}</div>'
    )

    table_html = f"""<div style="overflow-x:auto;">
<table style="width:100%;border-collapse:collapse;background:#fff;border-radius:8px;
overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.06);table-layout:auto;">
    <colgroup>
        <col style="width:28px">
        <col style="min-width:120px">
        <col style="width:40px">
    </colgroup>
    <thead>
        <tr style="background:#f8f9fa;border-bottom:2px solid #ddd;">
            <th style="padding:6px 8px;text-align:center;font-size:10px;color:#041E42;
                text-transform:uppercase;letter-spacing:1px;{sticky_num}">#</th>
            <th style="padding:6px 8px;text-align:left;font-size:10px;color:#041E42;
                text-transform:uppercase;letter-spacing:1px;{sticky_name}">Batter</th>
            <th style="padding:6px 8px;text-align:center;font-size:10px;color:#041E42;
                text-transform:uppercase;letter-spacing:1px;{sticky_pos}">Pos</th>
            {pa_headers}
        </tr>
    </thead>
    <tbody>
        {batter_rows_html}
    </tbody>
</table>
</div>"""

    return f"""<div id="scorecard">
{table_html}
{banner_html}
</div>"""


def _render_game_tags(scorecards: list[dict | None]) -> str:
    """Render compact game status tags between hero picks and scorecard."""
    tags = []
    for sc in scorecards:
        if sc is None:
            continue
        away = sc.get("away_team", "?")
        home = sc.get("home_team", "?")
        score = sc.get("score", {})
        status = sc.get("game_status", "P")
        inning = sc.get("inning", "")

        away_r = score.get("away", 0)
        home_r = score.get("home", 0)
        score_str = f"{away} {away_r} – {home_r} {home}"

        if status == "F":
            badge = ('<span style="background:#666;color:#fff;font-size:9px;'
                     'font-weight:700;padding:2px 5px;border-radius:3px;'
                     'letter-spacing:.5px;">FINAL</span>')
        elif status == "L":
            badge = (f'<span style="background:#c41e3a;color:#fff;font-size:9px;'
                     f'font-weight:700;padding:2px 5px;border-radius:3px;'
                     f'letter-spacing:.5px;">{inning}</span>')
        else:
            badge = ('<span style="background:#e0e0e0;color:#666;font-size:9px;'
                     'font-weight:700;padding:2px 5px;border-radius:3px;'
                     'letter-spacing:.5px;">PRE</span>')

        tags.append(
            f'<div style="display:inline-flex;align-items:center;gap:6px;'
            f'padding:4px 10px;background:#fff;border-radius:6px;'
            f'font-size:13px;font-weight:600;box-shadow:0 1px 2px rgba(0,0,0,.06);">'
            f'{badge} {score_str}</div>'
        )

    if not tags:
        return ""
    return (
        '<div style="display:flex;gap:8px;justify-content:center;'
        'margin:8px 0 16px;">' + "".join(tags) + '</div>'
    )


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
        result = p.get("result")
        if result == "hit":
            result_html = '<span class="result-hit">&#10003;</span>'
        elif result == "miss":
            result_html = '<span class="result-miss">&#10007;</span>'
        else:
            result_html = '<span class="result-pending">&ndash;</span>'

        logo = team_logo_url(team)
        logo_img = f'<img src="{logo}" class="team-logo" alt="{team}">' if logo else ""
        p_team = pick.get("pitcher_team", "")
        p_logo = team_logo_url(p_team) if p_team else ""
        p_logo_img = f'<img src="{p_logo}" class="team-logo" alt="{p_team}"> ' if p_logo else ""

        # Lineup status emoji
        if result in ("suspended", "unresolved"):
            lineup_icon = '<span title="Void">&#9888;&#65039;</span>'
        elif pick.get("projected_lineup"):
            lineup_icon = '<span title="Projected">&#128203;</span>'
        else:
            lineup_icon = '<span title="Confirmed">&#9989;</span>'

        # Non-lineup flags (IL return, opener, debut pitcher)
        flags = pick.get("flags", [])
        if isinstance(flags, str):
            flags = [f.strip() for f in flags.split(",") if f.strip()]
        other_flags = [f for f in flags if "PROJECTED" not in f]
        notes_html = f'<span class="notes-dot" data-tip="{", ".join(other_flags)}">&#9679;</span>' if other_flags else ""

        double = p.get("double_down")

        if result == "hit":
            row_class = "row-hit"
        elif result == "miss":
            row_class = "row-miss"
        elif date == today:
            row_class = "today"
        else:
            row_class = ""
        pick_rows += f"""
        <tr class="{row_class}">
            <td class="result-cell">{result_html}</td>
            <td class="date-cell">{date}</td>
            <td class="batter-cell">{logo_img} <strong>{name}</strong></td>
            <td class="matchup-cell">vs {p_logo_img}{pitcher}</td>
            <td class="pct-cell">{pct:.1%}</td>
            <td class="lineup-cell">{lineup_icon} {notes_html}</td>
        </tr>"""

        if double:
            d_name = double.get("batter_name", "?")
            d_team = double.get("team", "?")
            d_pitcher = double.get("pitcher_name", "?")
            d_pct = double.get("p_game_hit", 0)
            d_logo = team_logo_url(d_team)
            d_logo_img = f'<img src="{d_logo}" class="team-logo" alt="{d_team}">' if d_logo else ""
            dp_team = double.get("pitcher_team", "")
            dp_logo = team_logo_url(dp_team) if dp_team else ""
            dp_logo_img = f'<img src="{dp_logo}" class="team-logo" alt="{dp_team}"> ' if dp_logo else ""
            if result in ("suspended", "unresolved"):
                d_lineup_icon = '<span title="Void">&#9888;&#65039;</span>'
            elif double.get("projected_lineup"):
                d_lineup_icon = '<span title="Projected">&#128203;</span>'
            else:
                d_lineup_icon = '<span title="Confirmed">&#9989;</span>'
            d_flags = double.get("flags", [])
            if isinstance(d_flags, str):
                d_flags = [f.strip() for f in d_flags.split(",") if f.strip()]
            d_other_flags = [f for f in d_flags if "PROJECTED" not in f]
            d_notes_html = f'<span class="notes-dot" data-tip="{", ".join(d_other_flags)}">&#9679;</span>' if d_other_flags else ""
            pick_rows += f"""
        <tr class="{row_class}">
            <td class="result-cell"><span class="double-plus">+</span></td>
            <td class="date-cell"></td>
            <td class="batter-cell">{d_logo_img} <strong>{d_name}</strong></td>
            <td class="matchup-cell">vs {dp_logo_img}{d_pitcher}</td>
            <td class="pct-cell">{d_pct:.1%}</td>
            <td class="lineup-cell">{d_lineup_icon} {d_notes_html}</td>
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

    # Shadow model indicator
    shadow_html = ""
    shadow_files = sorted(PICKS_DIR.glob("*.shadow.json"))
    if shadow_files:
        s_total = 0
        s_agrees = 0
        prod_hits = 0
        prod_resolved = 0
        shadow_hits = 0
        shadow_resolved = 0
        s_disagrees_detail = []
        for sf in shadow_files:
            date_str = sf.name.replace(".shadow.json", "")
            prod_file = PICKS_DIR / f"{date_str}.json"
            if not prod_file.exists():
                continue
            try:
                sp = json.loads(sf.read_text())
                pp = json.loads(prod_file.read_text())
                s_total += 1
                sname = sp["pick"]["batter_name"]
                pname = pp["pick"]["batter_name"]
                prod_result = pp.get("result")
                shadow_result = sp.get("result")
                if sname == pname:
                    s_agrees += 1
                else:
                    s_disagrees_detail.append((date_str, pname, sname, prod_result, shadow_result))
                # Track P@1 for both models
                if prod_result in ("hit", "miss"):
                    prod_resolved += 1
                    if prod_result == "hit":
                        prod_hits += 1
                if shadow_result in ("hit", "miss"):
                    shadow_resolved += 1
                    if shadow_result == "hit":
                        shadow_hits += 1
            except Exception:
                continue
        if s_total > 0:
            pct = s_agrees / s_total * 100
            days_left = max(0, 30 - s_total)
            dot = "&#9679;"

            # Performance comparison
            perf_str = ""
            if prod_resolved > 0 and shadow_resolved > 0:
                prod_p1 = prod_hits / prod_resolved * 100
                shadow_p1 = shadow_hits / shadow_resolved * 100
                diff = shadow_p1 - prod_p1
                if abs(diff) < 1:
                    perf_str = f' · P@1: shadow {shadow_p1:.0f}% = prod {prod_p1:.0f}%'
                    dot_color = "#2d6a4f"
                elif diff > 0:
                    perf_str = f' · P@1: shadow {shadow_p1:.0f}% &gt; prod {prod_p1:.0f}% (+{diff:.0f}pp)'
                    dot_color = "#2d6a4f"
                else:
                    perf_str = f' · P@1: shadow {shadow_p1:.0f}% &lt; prod {prod_p1:.0f}% ({diff:.0f}pp)'
                    dot_color = "#c41e3a"
            elif pct >= 80:
                dot_color = "#2d6a4f"
            elif pct >= 60:
                dot_color = "#e9c46a"
            else:
                dot_color = "#c41e3a"

            # Disagreement head-to-head
            disagree_str = ""
            resolved_disagrees = [(d, pn, sn, pr, sr) for d, pn, sn, pr, sr in s_disagrees_detail
                                  if pr in ("hit", "miss") and sr in ("hit", "miss")]
            if resolved_disagrees:
                prod_wins = sum(1 for _, _, _, pr, sr in resolved_disagrees if pr == "hit" and sr != "hit")
                shadow_wins = sum(1 for _, _, _, pr, sr in resolved_disagrees if sr == "hit" and pr != "hit")
                both_hit = sum(1 for _, _, _, pr, sr in resolved_disagrees if pr == "hit" and sr == "hit")
                disagree_str = f' · Splits: shadow {shadow_wins}–{prod_wins} prod'
                if both_hit:
                    disagree_str += f' ({both_hit} both hit)'
            elif s_disagrees_detail:
                n_pending = len(s_disagrees_detail) - len(resolved_disagrees)
                if n_pending:
                    disagree_str = f' · {n_pending} split{"s" if n_pending > 1 else ""} pending'

            shadow_html = (
                f'<div style="margin:12px 0;padding:8px 14px;background:#f8f9fa;'
                f'border-radius:6px;font-size:11px;color:#666;display:flex;'
                f'align-items:center;gap:6px;flex-wrap:wrap;">'
                f'<span style="color:{dot_color};font-size:8px;">{dot}</span>'
                f'<span style="font-weight:600;color:#444;">Shadow Model</span>'
                f' {s_agrees}/{s_total} agree ({pct:.0f}%) · {days_left}d to eval'
                f'{perf_str}{disagree_str}'
                f'</div>'
            )

    # Today's pick hero
    hero = ""
    if today_pick:
        tp = today_pick["pick"]
        dd = today_pick.get("double_down")
        t_logo = team_logo_url(tp.get("team", ""), size=72)
        t_logo_img = f'<img src="{t_logo}" class="hero-logo" alt="{tp.get("team", "")}">' if t_logo else ""
        t_time = _format_game_time(tp.get("game_time", ""))
        # Pick is locked if posted, result is in, or game has started
        is_locked = today_pick.get("bluesky_posted", False) or today_pick.get("result") is not None
        if not is_locked:
            game_time_str = tp.get("game_time", "")
            if game_time_str:
                try:
                    game_dt = datetime.fromisoformat(game_time_str.replace("Z", "+00:00"))
                    is_locked = datetime.now(game_dt.tzinfo) > game_dt
                except (ValueError, TypeError):
                    pass
        if today_pick.get("result") == "hit":
            lock_badge = '<span class="lock-badge locked" style="background:#2d6a4f;">HIT &#10003;</span>'
        elif today_pick.get("result") == "miss":
            lock_badge = '<span class="lock-badge locked" style="background:#c41e3a;">MISS &#10007;</span>'
        elif is_locked:
            lock_badge = '<span class="lock-badge locked">LOCKED</span>'
        else:
            lock_badge = '<span class="lock-badge pending">PENDING</span>'
        # Pick lock time: 5 min before earliest picked game
        lock_time_html = ""
        if not is_locked and not today_pick.get("result"):
            game_times = [tp.get("game_time", "")]
            if dd:
                game_times.append(dd.get("game_time", ""))
            earliest = None
            for gt in game_times:
                if not gt:
                    continue
                try:
                    gdt = datetime.fromisoformat(gt.replace("Z", "+00:00"))
                    if earliest is None or gdt < earliest:
                        earliest = gdt
                except (ValueError, TypeError):
                    pass
            if earliest:
                lock_dt = earliest - timedelta(minutes=5)
                lock_str = lock_dt.strftime("%-I:%M %p ET").replace(" 0", " ")
                lock_time_html = (
                    f'<span style="font-size:11px;color:#888;font-weight:400;'
                    f'margin-left:auto;">Pick Lock: {lock_str}</span>'
                )

        label = "TODAY'S PICKS" if dd else "TODAY'S PICK"
        if dd:
            d_logo = team_logo_url(dd.get("team", ""), size=72)
            d_logo_img = f'<img src="{d_logo}" class="hero-logo" alt="{dd.get("team", "")}">' if d_logo else ""
            d_time = _format_game_time(dd.get("game_time", ""))
            p_both = tp.get('p_game_hit', 0) * dd.get('p_game_hit', 0)
            hero = f"""
        <div class="hero-status" style="display:flex;align-items:center;">{label} {lock_badge}{lock_time_html}</div>
        <div class="hero">
            <div class="hero-left">
                {t_logo_img}
            </div>
            <div class="hero-right">
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
        <div class="hero-status" style="display:flex;align-items:center;">{label} {lock_badge}{lock_time_html}</div>
        <div class="hero">
            <div class="hero-left">
                {t_logo_img}
            </div>
            <div class="hero-right">
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

    # Live scorecard (between hero and pick history)
    scorecard_html = ""
    game_tags_html = ""
    if today_pick:
        tp = today_pick["pick"]
        primary_game_pk = tp.get("game_pk")
        if primary_game_pk:
            dd = today_pick.get("double_down")
            dd_game_pk = dd.get("game_pk") if dd else None

            from bts.scorecard import fetch_live_scorecard, merge_scorecards

            if dd_game_pk and dd_game_pk != primary_game_pk:
                # Double-down is in a different game — fetch both and merge
                primary_ids = {tp.get("batter_id")}
                primary_ids.discard(None)
                dd_ids = {dd.get("batter_id")}
                dd_ids.discard(None)
                sc1 = fetch_live_scorecard(primary_game_pk, primary_ids)
                sc2 = fetch_live_scorecard(dd_game_pk, dd_ids)
                scorecard_data = merge_scorecards(sc1, sc2)
                game_tags_html = _render_game_tags([sc1, sc2])
            else:
                # Same game (or no double-down) — fetch once with both batter IDs
                batter_ids = {tp.get("batter_id")}
                if dd:
                    batter_ids.add(dd.get("batter_id"))
                batter_ids.discard(None)
                scorecard_data = fetch_live_scorecard(primary_game_pk, batter_ids)
                game_tags_html = _render_game_tags([scorecard_data])

            scorecard_html = render_scorecard_section(scorecard_data)

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
        .hero-status {{ color: #D50032; font-size: 0.7em; text-transform: uppercase;
                       letter-spacing: 3px; font-weight: 700; margin: 20px 0 8px; }}
        .lock-badge {{ font-size: 0.75em; padding: 2px 8px; border-radius: 4px;
                       letter-spacing: 1px; vertical-align: middle; margin-left: 8px; }}
        .lock-badge.locked {{ background: #2e7d32; color: #fff; }}
        .lock-badge.pending {{ background: #f57c00; color: #fff; }}

        .section-header {{ color: #041E42; font-size: 0.75em; text-transform: uppercase;
                           letter-spacing: 2px; font-weight: 700; margin: 28px 0 12px;
                           padding-bottom: 8px; border-bottom: 2px solid #ddd; }}

        table {{ width: 100%; border-collapse: collapse; background: #fff;
                 border-radius: 8px; overflow: visible;
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
        col.col-lineup {{ width: 50px; }}

        .team-logo {{ width: 24px; height: 24px; vertical-align: middle; margin-right: 6px; }}
        .batter-cell strong {{ color: #041E42; }}
        .batter-cell, .matchup-cell {{ overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
        .matchup-cell {{ color: #666; }}
        .pct-cell {{ color: #002D72; font-weight: 600; font-variant-numeric: tabular-nums; text-align: center; }}
        .double-plus {{ color: #D50032; font-weight: 800; font-size: 1.2em; }}
        .lineup-cell {{ text-align: center; font-size: 1.1em; overflow: visible; position: relative; vertical-align: middle; }}
        .notes-dot {{ color: #f57c00; font-size: 0.7em; cursor: help; vertical-align: middle;
                      position: relative; padding: 2px 4px; }}
        .notes-dot:hover::after {{ content: attr(data-tip); position: absolute; bottom: 100%;
                      right: 0; background: #333; color: #fff;
                      font-size: 11px; padding: 4px 8px; border-radius: 4px; white-space: nowrap;
                      z-index: 10; pointer-events: none; }}
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

        @keyframes pulse-border {{
            0%, 100% {{ border-color: #f59e0b; }}
            50% {{ border-color: rgba(245,158,11,0.3); }}
        }}

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

        {game_tags_html}

        {scorecard_html}

        <div class="section-header">Pick History</div>
        <table>
            <colgroup>
                <col class="col-result"><col class="col-date"><col class="col-batter">
                <col class="col-matchup"><col class="col-pct"><col class="col-lineup">
            </colgroup>
            <tr><th></th><th>Date</th><th>Batter</th><th>Matchup</th><th>P(Hit)</th><th>Lineup</th></tr>
            {pick_rows}
        </table>

        {shadow_html}

        <div class="section-header">Bluesky Posts</div>
        <div class="posts">
            {posts_html if posts_html else '<div class="post">No posts loaded</div>'}
        </div>

        <div class="footer">
            Updated {datetime.now().strftime('%Y-%m-%d %H:%M ET')} · LAN only · Not affiliated with MLB
        </div>
    </div>
    <script src="https://embed.bsky.app/static/embed.js" async charset="utf-8"></script>
    <script>
(function() {{
    var date = "{today}";
    var sc = document.getElementById("scorecard");
    if (!sc) return;
    var timer = setInterval(function() {{
        fetch("/api/live-html?date=" + date)
            .then(function(r) {{ return r.text(); }})
            .then(function(html) {{
                if (!html || html.length < 10) return;
                sc.outerHTML = html;
                sc = document.getElementById("scorecard");
                if (!sc) clearInterval(timer);
                if (html.indexOf("FINAL") > -1) clearInterval(timer);
            }})
            .catch(function() {{}});
    }}, 30000);
}})();
    </script>
</body>
</html>"""


def health_check(heartbeat_path: Path = None) -> tuple[int, dict]:
    """Check scheduler health via heartbeat file.

    Returns (status_code, response_dict).  Pure function so it can be
    tested without spinning up an HTTP server.
    """
    if heartbeat_path is None:
        heartbeat_path = HEARTBEAT_PATH

    hb = read_heartbeat(heartbeat_path)
    if hb is None:
        return 503, {"status": "stale", "reason": "no heartbeat file"}

    fresh = is_heartbeat_fresh(heartbeat_path, max_age_sec=180)
    if not fresh:
        return 503, {
            "status": "stale",
            "last_heartbeat": hb.get("timestamp", "unknown"),
            "scheduler_state": hb.get("state"),
        }

    return 200, {
        "status": "ok",
        "last_heartbeat": hb.get("timestamp"),
        "scheduler_state": hb.get("state"),
        "sleeping_until": hb.get("sleeping_until"),
    }


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(self.path)

        if parsed.path == "/health":
            status_code, data = health_check()
            self._json_response(data, status_code=status_code)
        elif parsed.path == "/api/live":
            self._handle_api_live(parse_qs(parsed.query))
        elif parsed.path == "/api/live-html":
            self._handle_api_live_html(parse_qs(parsed.query))
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(render_page().encode())

    def _handle_api_live(self, params):
        """Return live scorecard JSON for today's picked batters."""
        from bts.scorecard import fetch_live_scorecard, merge_scorecards
        date = params.get("date", [datetime.now().strftime("%Y-%m-%d")])[0]
        pick_path = PICKS_DIR / f"{date}.json"
        if not pick_path.exists():
            self._json_response({"game_status": None})
            return
        pick_data = json.loads(pick_path.read_text())
        pick = pick_data.get("pick", {})
        primary_game_pk = pick.get("game_pk")
        if not primary_game_pk:
            self._json_response({"game_status": None})
            return
        dd = pick_data.get("double_down")
        dd_game_pk = dd.get("game_pk") if dd else None

        if dd_game_pk and dd_game_pk != primary_game_pk:
            # Double-down is in a different game — fetch both and merge
            primary_ids = {pick.get("batter_id")}
            primary_ids.discard(None)
            dd_ids = {dd.get("batter_id")}
            dd_ids.discard(None)
            result = merge_scorecards(
                fetch_live_scorecard(primary_game_pk, primary_ids),
                fetch_live_scorecard(dd_game_pk, dd_ids),
            )
        else:
            # Same game (or no double-down) — fetch once with both batter IDs
            batter_ids = {pick.get("batter_id")}
            if dd:
                batter_ids.add(dd.get("batter_id"))
            batter_ids.discard(None)
            result = fetch_live_scorecard(primary_game_pk, batter_ids)

        if result is None:
            self._json_response({"game_status": None, "error": "feed unavailable"})
            return
        self._json_response(result)

    def _handle_api_live_html(self, params):
        """Return rendered scorecard HTML fragment for live polling."""
        from bts.scorecard import fetch_live_scorecard, merge_scorecards
        date = params.get("date", [datetime.now().strftime("%Y-%m-%d")])[0]
        pick_path = PICKS_DIR / f"{date}.json"
        if not pick_path.exists():
            self._html_response("")
            return
        pick_data = json.loads(pick_path.read_text())
        pick = pick_data.get("pick", {})
        primary_game_pk = pick.get("game_pk")
        if not primary_game_pk:
            self._html_response("")
            return
        dd = pick_data.get("double_down")
        dd_game_pk = dd.get("game_pk") if dd else None
        if dd_game_pk and dd_game_pk != primary_game_pk:
            primary_ids = {pick.get("batter_id")}
            primary_ids.discard(None)
            dd_ids = {dd.get("batter_id")}
            dd_ids.discard(None)
            scorecard_data = merge_scorecards(
                fetch_live_scorecard(primary_game_pk, primary_ids),
                fetch_live_scorecard(dd_game_pk, dd_ids),
            )
        else:
            batter_ids = {pick.get("batter_id")}
            if dd:
                batter_ids.add(dd.get("batter_id"))
            batter_ids.discard(None)
            scorecard_data = fetch_live_scorecard(primary_game_pk, batter_ids)
        html = render_scorecard_section(scorecard_data)
        self._html_response(html)

    def _html_response(self, html):
        body = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json_response(self, data, status_code=200):
        body = json.dumps(data).encode()
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


def main():
    server = HTTPServer(("0.0.0.0", PORT), Handler)
    print(f"BTS Dashboard running at http://0.0.0.0:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
