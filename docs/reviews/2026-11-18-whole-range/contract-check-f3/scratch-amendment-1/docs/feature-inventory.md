# Feature Inventory

Complete inventory of every feature considered, tested, and decided on.
All results from leak-free walk-forward evaluation on 2025 (184 test days).
Baseline P@1 = 90.2% with current 14 features.

## Currently Included (14 features)

### Batter Rolling Hit Rates
| Feature | Window | P@1 when dropped | Status |
|---------|--------|-------------------|--------|
| batter_hr_7g | 7 games | 85.9% (-4.3%) | KEEP — short-term form signal |
| batter_hr_30g | 30 games | 85.9% (-4.3%) | KEEP — medium-term ability |
| batter_hr_60g | 60 games | 86.4% (-3.8%) | KEEP — longer-term ability |
| batter_hr_120g | 120 games | 90.2% (0.0%) | KEEP — neutral, no harm |

### Batter Pitch Discipline
| Feature | Description | P@1 when dropped | Status |
|---------|-------------|-------------------|--------|
| batter_count_tendency_30g | Avg (balls - strikes) at PA end, rolling 30g | 87.0% (-3.3%) | KEEP |

### Batter Contact/Speed
| Feature | Description | P@1 when dropped | Status |
|---------|-------------|-------------------|--------|
| batter_gb_hit_rate | Expanding ground ball hit rate (speed proxy) | 89.1% (-1.1%) | KEEP |

### Batter Matchup
| Feature | Description | P@1 when dropped | Status |
|---------|-------------|-------------------|--------|
| platoon_hr | Expanding H/PA by batter x pitcher hand (R/L) | 88.0% (-2.2%) | KEEP |
| batter_vs_arch_hr | Expanding H/PA by batter x pitcher archetype (K=8) | 90.2% (0.0%) | KEEP — neutral but architecturally important |

### Pitcher
| Feature | Description | P@1 when dropped | Status |
|---------|-------------|-------------------|--------|
| pitcher_hr_30g | Rolling hit rate allowed, 30 games | 85.9% (-4.3%) | KEEP — strongest individual feature |
| pitcher_cluster | K-Means archetype (0-7) based on arsenal | 88.0% (-2.2%) | KEEP |
| pitcher_entropy_30g | Shannon entropy of pitch type distribution, rolling 30g | 87.5% (-2.7%) | KEEP |

### Context
| Feature | Description | P@1 when dropped | Status |
|---------|-------------|-------------------|--------|
| weather_temp | Game temperature from MLB feed | 88.0% (-2.2%) | KEEP |
| park_factor | Expanding venue hit rate / league avg | 88.0% (-2.2%) | KEEP |
| days_rest | Days since batter's last game | 88.0% (-2.2%) | KEEP |

## Excluded — Tested and Dropped

### Dropped from model
| Feature | Description | P@1 when ADDED | Why dropped |
|---------|-------------|----------------|-------------|
| lineup_position | Batting order 1-9 | 86.4% (-3.8%) | Double-counts with PA-level aggregation. Model learns spurious "position 1 = more hits" when the real effect (more PAs) is in the aggregation step. |
| is_home | Home/away flag | 89.1% (-1.1%) | Marginal at PA level. Home advantage might exist at game level but individual PAs aren't meaningfully different. |
| batter_whiff_30g | Rolling whiff rate, 30 games | 88.0% (-2.2%) | Collinear with other features. Individually correlated with hits (-0.055 in EDA) but redundant with pitcher entropy and count tendency. |
| batter_hr_14g | Rolling hit rate, 14 games | 84.2% (-6.0%) | Sandwiched between 7g and 30g. Adds noise without adding signal — the model can interpolate from the adjacent windows. |
| same_hand | Flag: batter and pitcher same handedness | 85.9% (-4.3%) | Redundant with platoon_hr which already captures handedness matchup with per-batter granularity. |
| batter_whiff_60g | Rolling whiff rate, 60 games | Dropped during optimization | Redundant once pitcher_entropy_30g captures arsenal complexity from pitcher side. |

### Dropped from EDA (never made it to model)
| Feature | EDA finding | Why dropped |
|---------|-------------|-------------|
| umpire_csr | HP umpire called strike rate | Corr +0.005 with game-level hits. Mechanism exists (zone→counts→hits) but effect is ~0.2-0.4% — noise in practice. |
| chase_rate | Batter swing rate on pitches outside zone | Corr +0.009 with hits — essentially zero. |
| zone_contact_rate | Contact rate on pitches in zone | NaN correlation — unreliable calculation. |
| month_of_season | April through September | No effect (21.7% to 22.3% range). Seasonal variation is noise. |
| day_of_week | Monday through Sunday | 89.7% P@1 when added (-0.5%). Noise. |
| batter_hr_std_30g | Std dev of hit rate (consistency) | 91.3% P@1 when added — slight harm. Inconsistent batters are unpredictable, but that doesn't make them worse picks. |
| batter_avg_pitches_30g | Avg pitches per PA (patience) | 91.8% P@1 when added — slight harm. Collinear with count tendency. |
| air_density | From Open-Meteo (pressure + humidity) | Not tested — temperature captures 87% of weather effect per Nathan's research. Marginal improvement expected. |
| sprint_speed | Baseball Savant seasonal leaderboard | Not tested directly — proxied by batter_gb_hit_rate instead. |
| pitcher_fatigue_in_game | Pitcher pitch count within current game | Not knowable pre-game. Irrelevant for prediction. |

## Not Yet Tested — Ideas for Future Investigation

### Batter features
- Batter career PA count (experience/rookie vs veteran)
- Batter vs specific pitcher H2H history (sparse but complementary to archetype)
- Rolling launch speed / launch angle trends (batted ball quality)
- Batter home/road split (some players have extreme home/road splits)
- Pinch hitter flag (pinch hitters have lower hit rates — 26.1% per Garnett)

### Pitcher features
- Pitcher recent workload (days rest, innings in last 7/14 days)
- Pitcher home/road ERA split
- Opponent team bullpen quality (for game-level aggregation of later PAs)
- Pitcher pitch velocity trend (declining velo = tiring arm)

### Context features
- Day/night game (available in feed as dayNight field — not yet in PA table)
- Wind direction impact (in/out/cross — already in PA table but not used)
- Altitude (static per venue, correlated with park_factor but distinct physics)
- Double-header flag
- Interleague play flag

### Architectural changes
- Different model for pinch hitters vs starters
- Separate models by batting handedness
- Time-weighted expanding features (recent games weighted more)
