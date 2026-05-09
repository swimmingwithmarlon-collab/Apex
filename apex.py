# CRIMSON APEX v5.3 Juno Console
# v5.3: importlib patch, MATCHUP bootstrap, ML cold-start fix, O/U probability
# All strings are ASCII-only to prevent smart-quote crashes.

# v5.2 CHANGES vs v5.1:
# Tracking PPP: column dump on miss broader candidate list
# FIX 2 Streak: WL column fallback when PLUS_MINUS absent from TeamGameLog
# FIX 3 Playoff mode: auto-detected from ScoreboardV3 seriesGameNumber
# Applies PLAYOFF_PACE_DISCOUNT + PLAYOFF_SCORING_DISCOUNT in build_pr
# Availability window widened 2->5 games for playoff rotations
# is_playoff + series_game_num added to ML feature vector (18 features)
# Split line None guard (LAL edge case)

import importlib.metadata
importlib.metadata.version = lambda name: "0.0"

import datetime
import math
import random
import re
import statistics
import time
import threading
import json
import os
import urllib.request
import urllib.parse
import sys
import types
import difflib
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor, as_completed

version = "5.2"

# NUMPY / SKLEARN
try:
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    ML_AVAILABLE = True
    print("[OK] numpy + sklearn loaded ML layer active")
except ImportError:
    ML_AVAILABLE = False
    print("[WARN] numpy/sklearn not available ML layer disabled")

# NBA STATS HEADERS
NBA_HEADERS = {
    "Host": "stats.nba.com",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
    "Connection": "keep-alive",
    "Referer": "https://stats.nba.com/",
    "Pragma": "no-cache",
    "Cache-Control": "no-cache",
}

def _inject_headers():
    injected = False
    try:
        from nba_api.library.http import NBAStatsHTTP
        NBAStatsHTTP.headers = NBA_HEADERS
        print("[OK] Headers injected via NBAStatsHTTP")
        injected = True
    except Exception:
        pass

    if not injected:
        try:
            import nba_api.library.http as _http
            if hasattr(_http, "HEADERS"):
                _http.HEADERS = NBA_HEADERS
                injected = True
            elif hasattr(_http, "headers"):
                _http.headers = NBA_HEADERS
                injected = True
            if injected:
                print("[OK] Headers injected via nba_api.library.http")
        except Exception:
            pass

    if not injected:
        try:
            import requests
            _orig_get = requests.get
            def patched_get(url, **kwargs):
                h = kwargs.get("headers", {})
                h.update(NBA_HEADERS)
                return _orig_get(url, headers=h, **kwargs)
            requests.get = patched_get
            print("[OK] Headers injected via requests.get monkey-patch")
            injected = True
        except Exception as e:
            print("[WARN] All header injection methods failed:", e)

    if not injected:
        print("[WARN] Running without custom headers")

_inject_headers()

# PANDAS STUB
try:
    import pandas
    print("[OK] pandas available natively")
except Exception:
    _fake_pd = types.ModuleType("pandas")

    class FakeDF:
        def __init__(self, *a, **kw): pass
        def __iter__(self): return iter([])
        def __len__(self): return 0
        def __getitem__(self, k): return FakeSeries()
        def __setitem__(self, k, v): pass
        def get(self, k, d=None): return d

    class FakeSeries(FakeDF):
        def __init__(self, a=None, **kw): pass
        def tolist(self): return []
        def values(self): return []

    def _noop(*a, **kw):
        return FakeDF()

    _fake_pd.DataFrame = FakeDF
    _fake_pd.Series = FakeSeries
    _fake_pd.MultiIndex = FakeDF
    _fake_pd.concat = _noop
    _fake_pd.read_json = _noop
    _fake_pd.isna = lambda x: False
    _fake_pd.isnull = lambda x: False
    _fake_pd.notna = lambda x: True
    _fake_pd.notnull = lambda x: True
    _fake_pd.NA = None
    _fake_pd.NaT = None
    _fake_pd.options = FakeDF()
    sys.modules["pandas"] = _fake_pd
    print("[OK] Thickened pandas stub injected")

# NBA API IMPORTS
from nba_api.stats.endpoints import (
    ScoreboardV3,
    LeagueDashTeamStats,
    LeagueDashTeamClutch,
    LeagueDashPlayerStats,
    TeamGameLog
)

try:
    from nba_api.stats.endpoints import LeagueDashPtStats
    PT_STATS_AVAILABLE = True
    print("[OK] LeagueDashPtStats loaded (tracking integration)")
except ImportError:
    PT_STATS_AVAILABLE = False
    print("[WARN] LeagueDashPtStats not available")

try:
    from nba_api.stats.endpoints import LeagueHustleStatsTeam
    HUSTLE_AVAILABLE = True
    print("[OK] LeagueHustleStatsTeam loaded")
except ImportError:
    HUSTLE_AVAILABLE = False
    print("[WARN] LeagueHustleStatsTeam not available")

print("[OK] Core nba_api endpoints loaded")
from nba_api.stats.static import teams as nba_teams_static
TRICODE_TO_ID = {t["abbreviation"]: t["id"] for t in nba_teams_static.get_teams()}
_TRICODE_FROM_ID = {v: k for k, v in TRICODE_TO_ID.items()}


# AUTO SEASON
now = datetime.datetime.now()
yr = now.year
if now.month >= 10:
    _start, _end = yr, yr + 1
else:
    _start, _end = yr - 1, yr
SEASON = "{}-{:02d}".format(_start, _end % 100)
print("[AUTO] Season set to " + SEASON)

# CONSTANTS
SIMS = 3000
IN_GAME_SIMS = 1000
EWMA_ALPHA = 0.20
LEAGUE_ORTG = 114.0
LEAGUE_TOV = 0.14
LEAGUE_OREB = 0.25
LEAGUE_EFG = 0.52
LEAGUE_FTA = 0.20
LEAGUE_PPP = 1.10
TIMEOUT = 60
REGULATION_SECS = 2880
OT_PERIOD_SECS = 300
BLEND_START_ELAPSED = 480
HOME_COURT_BOOST_FALLBACK = 1.013
OPP_BLEND = 0.40
VEGAS_WEIGHT = 0.75
RECENT_FORM_WEIGHT = 0.65
RECENT_GAMES_WINDOW = 20
TRACKING_PPP_WEIGHT = 0.40

_LOC_FF_KEYS = (
    "efg", "tov_rate", "oreb_rate", "fta_rate",
    "opp_efg", "opp_tov", "opp_oreb", "opp_fta",
)

# PLAYOFF CONSTANTS (v5.2)
PLAYOFF_PACE_DISCOUNT = 0.935
PLAYOFF_SCORING_DISCOUNT = 0.965
AVAIL_WINDOW_REGULAR = 2
AVAIL_WINDOW_PLAYOFF = 5

# THE ODDS API CONFIG
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds/"
ODDS_TEAM_MAP = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE", "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET", "Golden State Warriors": "GSW", "Houston Rockets": "HOU",
    "Indiana Pacers": "IND", "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM", "Miami Heat": "MIA", "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NOP", "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX", "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS", "Toronto Raptors": "TOR", "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}

# HUSTLE CONSTANTS
LEAGUE_AVG_DEFLECTIONS = 14.5
LEAGUE_AVG_SCREEN_ASSISTS = 28.0
HUSTLE_DEFL_CLAMP = 0.10
HUSTLE_SCREEN_EFG_CLAMP = 0.012

# AVAILABILITY CONSTANTS
AVAILABILITY_NET_IMPACT = 0.45
AVAILABILITY_MAX_SHARE = 0.35
AVAILABILITY_FLOOR = 0.80

# ML CONSTANTS
ML_TRAINING_WINDOW = 80
ML_MIN_TO_ACTIVATE = 25
ML_MAX_CORRECTION = 6.0
ML_RIDGE_ALPHA = 25.0
ML_TOP_PLAYERS = 5
ML_HISTORY_SAVE_FILE = "apex_ml_snapshots_v5.json"

# CACHES
TEAM_STATS_SEASON = {}
TEAM_STATS_RECENT = {}
HOME_STATS = {}
ROAD_STATS = {}
TRACKING_SEASON = {}
TRACKING_RECENT = {}
DRIVES_SEASON = {}
DRIVES_RECENT = {}
CLUTCH_CACHE = {}
TEAM_LOG_CACHE = {}
PLAYER_RECENT_CACHE = {}
HUSTLE_CACHE = {}
PLAYER_LAST_GAME_CACHE = {}
_AVAIL_FACTOR_CACHE = {}

_SEASON_LOADED = False
_RECENT_LOADED = False
_LOCATION_LOADED = False
_TRACKING_LOADED = False
_DRIVES_LOADED = False
_CLUTCH_LOADED = False
_PLAYERS_LOADED = False
_HUSTLE_LOADED = False
_AVAILABILITY_LOADED = False

_IS_PLAYOFF_SESSION = False

# ML STATE
ML_MODEL = None
ML_SCALER = None
ML_TRAINED = False
ML_CV_MAE = None
ML_RAW_MAE = None
ML_BIAS = None
ML_SNAPSHOTS = {}

# HELPERS
def with_retry(fn, retries=3, base_delay=2.0):
    last_exc = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if attempt < retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print("[RETRY {}/{}] waiting ({:.1f}s)".format(attempt + 1, retries, delay))
                time.sleep(delay)
    raise last_exc

def _find_col(headers, *candidates):
    for name in candidates:
        if name in headers:
            return headers.index(name)
    return None

def _safe_float(val, default=0.0):
    try:
        return float(val) if val is not None else default
    except (ValueError, TypeError):
        return default

# UNIFIED STAT LOADER
def _fetch_team_table(measure_type, last_n_games=0, location=""):
    def _call_with_loc(loc_kwarg):
        kwargs = dict(
            season=SEASON,
            measure_type_detailed_defense=measure_type,
            per_mode_detailed="PerGame",
            last_n_games=last_n_games,
            timeout=TIMEOUT,
        )
        kwargs[loc_kwarg] = location
        return LeagueDashTeamStats(**kwargs).get_dict()

    def _call_no_loc():
        return LeagueDashTeamStats(
            season=SEASON,
            measure_type_detailed_defense=measure_type,
            per_mode_detailed="PerGame",
            last_n_games=last_n_games,
            timeout=TIMEOUT,
        ).get_dict()

    if not location:
        resp = with_retry(_call_no_loc)
        return resp["resultSets"][0]["headers"], resp["resultSets"][0]["rowSet"]

    for loc_kwarg in ("location_nullable", "location"):
        try:
            resp = with_retry(lambda lk=loc_kwarg: _call_with_loc(lk))
            return resp["resultSets"][0]["headers"], resp["resultSets"][0]["rowSet"]
        except TypeError:
            continue
        except Exception:
            raise

    print("[WARN] location split param not supported - returning full-season data")
    resp = with_retry(_call_no_loc)
    return resp["resultSets"][0]["headers"], resp["resultSets"][0]["rowSet"]

def _populate_advanced(target_dict, last_n_games=0, location=""):
    headers, rows = _fetch_team_table("Advanced", last_n_games, location)
    tid_i = _find_col(headers, "TEAM_ID")
    off_i = _find_col(headers, "OFF_RATING", "E_OFF_RATING", "OFFENSIVE_RATING")
    def_i = _find_col(headers, "DEF_RATING", "E_DEF_RATING", "DEFENSIVE_RATING")
    pace_i = _find_col(headers, "PACE", "E_PACE")

    if None in (tid_i, off_i, def_i, pace_i):
        print("[WARN] Advanced: required columns missing")
        return 0

    for row in rows:
        tid = row[tid_i]
        target_dict.setdefault(tid, {}).update({
            "ortg": _safe_float(row[off_i], LEAGUE_ORTG),
            "drtg": _safe_float(row[def_i], LEAGUE_ORTG),
            "pace": _safe_float(row[pace_i], 100.0),
        })
    return len(rows)

def _populate_four_factors(target_dict, last_n_games=0, location=""):
    headers, rows = _fetch_team_table("Four Factors", last_n_games, location)
    tid_i = _find_col(headers, "TEAM_ID")
    efg_i = _find_col(headers, "EFG_PCT")
    tov_i = _find_col(headers, "TM_TOV_PCT", "TOV_PCT")
    oreb_i = _find_col(headers, "OREB_PCT")
    fta_i = _find_col(headers, "FTA_RATE")
    opp_efg_i = _find_col(headers, "OPP_EFG_PCT")
    opp_tov_i = _find_col(headers, "OPP_TOV_PCT", "OPP_TM_TOV_PCT")
    opp_oreb_i = _find_col(headers, "OPP_OREB_PCT")
    opp_fta_i = _find_col(headers, "OPP_FTA_RATE")

    if None in (tid_i, efg_i, tov_i, oreb_i, fta_i):
        print("[WARN] Four Factors: required columns missing")
        return 0

    has_opp = None not in (opp_efg_i, opp_tov_i, opp_oreb_i, opp_fta_i)
    for row in rows:
        tid = row[tid_i]
        target_dict.setdefault(tid, {}).update({
            "efg": _safe_float(row[efg_i], LEAGUE_EFG),
            "tov_rate": _safe_float(row[tov_i], LEAGUE_TOV),
            "oreb_rate": _safe_float(row[oreb_i], LEAGUE_OREB),
            "fta_rate": _safe_float(row[fta_i], LEAGUE_FTA),
            "opp_efg": _safe_float(row[opp_efg_i], LEAGUE_EFG) if has_opp else LEAGUE_EFG,
            "opp_tov": _safe_float(row[opp_tov_i], LEAGUE_TOV) if has_opp else LEAGUE_TOV,
            "opp_oreb": _safe_float(row[opp_oreb_i], LEAGUE_OREB) if has_opp else LEAGUE_OREB,
            "opp_fta": _safe_float(row[opp_fta_i], LEAGUE_FTA) if has_opp else LEAGUE_FTA,
        })
    return len(rows)

def _populate_base(target_dict, last_n_games=0, location=""):
    headers, rows = _fetch_team_table("Base", last_n_games, location)
    tid_i = _find_col(headers, "TEAM_ID")
    fgm_i = _find_col(headers, "FGM")
    fga_i = _find_col(headers, "FGA")
    fg3m_i = _find_col(headers, "FG3M")
    fg3a_i = _find_col(headers, "FG3A")
    pts_i = _find_col(headers, "PTS")
    fta_i = _find_col(headers, "FTA")
    fg3_pct_i = _find_col(headers, "FG3_PCT")
    ft_pct_i = _find_col(headers, "FT_PCT")
    stl_i = _find_col(headers, "STL")
    ast_i = _find_col(headers, "AST")
    tov_i = _find_col(headers, "TOV")

    if None in (tid_i, fgm_i, fga_i, fg3m_i, fg3a_i, pts_i, fta_i):
        print("[WARN] Base: required columns missing")
        return 0

    LEAGUE_AVG_STL = 8.5
    for row in rows:
        tid = row[tid_i]
        fgm = _safe_float(row[fgm_i])
        fga = _safe_float(row[fga_i], 1.0)
        fg3m = _safe_float(row[fg3m_i])
        fg3a = _safe_float(row[fg3a_i])
        fta = _safe_float(row[fta_i])
        pts = _safe_float(row[pts_i])
        stl = _safe_float(row[stl_i], LEAGUE_AVG_STL) if stl_i else LEAGUE_AVG_STL
        ast = _safe_float(row[ast_i], 22.0) if ast_i else 22.0
        tov = _safe_float(row[tov_i], 14.0) if tov_i else 14.0

        ts = pts / (2.0 * (fga + 0.44 * fta)) if (fga + fta) > 0 else 0.58
        fg2_att = max(fga - fg3a, 1.0)
        fg2_pct = (fgm - fg3m) / fg2_att
        fg3_pct = _safe_float(row[fg3_pct_i], 0.36) if fg3_pct_i else 0.36
        ft_pct = _safe_float(row[ft_pct_i], 0.78) if ft_pct_i else 0.78
        fb_yield = max(0.5, min(2.0, stl / LEAGUE_AVG_STL))

        target_dict.setdefault(tid, {}).update({
            "ts": ts,
            "3rate": fg3a / max(fga, 1.0),
            "fg2_pct": max(0.30, min(0.75, fg2_pct)),
            "fg3_pct": fg3_pct,
            "ft_pct": ft_pct,
            "fastbreak_yield": fb_yield,
            "opp_fastbreak_yield": 1.0,
            "ast_to_ratio": ast / max(tov, 1.0),
            "pts_per_game": pts,
        })
    return len(rows)

def _populate_scoring(target_dict, last_n_games=0, location=""):
    try:
        headers, rows = _fetch_team_table("Scoring", last_n_games, location)
        tid_i = _find_col(headers, "TEAM_ID")
        paint_i = _find_col(headers, "PCT_PTS_PAINT")
        fb_i = _find_col(headers, "PCT_PTS_FB")
        sc2_i = _find_col(headers, "PCT_PTS_2ND_CHANCE")

        if tid_i is None:
            return 0

        for row in rows:
            tid = row[tid_i]
            target_dict.setdefault(tid, {}).update({
                "pct_pts_paint": _safe_float(row[paint_i], 0.28) if paint_i else 0.28,
                "pct_pts_fastbreak": _safe_float(row[fb_i], 0.13) if fb_i else 0.13,
                "pct_pts_2nd_chance": _safe_float(row[sc2_i], 0.07) if sc2_i else 0.07,
            })
        return len(rows)
    except Exception as e:
        print("[WARN] Scoring measure failed:", e)
        return 0

# =====================================================
# SEASON + RECENT LOADERS
# =====================================================
def _load_season_stats():
    global _SEASON_LOADED
    if _SEASON_LOADED: return
    try:
        n_adv = _populate_advanced(TEAM_STATS_SEASON)
        n_ff = _populate_four_factors(TEAM_STATS_SEASON)
        n_b = _populate_base(TEAM_STATS_SEASON)
        n_sc = _populate_scoring(TEAM_STATS_SEASON)
        _SEASON_LOADED = True
        print("[OK] Season stats loaded (Adv={} FF={} Base={} Scoring={})".format(n_adv, n_ff, n_b, n_sc))
    except Exception as e:
        print("[WARN] Season stats load failed:", e)

def _load_recent_stats():
    global _RECENT_LOADED
    if _RECENT_LOADED: return
    try:
        n_adv = _populate_advanced(TEAM_STATS_RECENT, last_n_games=RECENT_GAMES_WINDOW)
        n_ff = _populate_four_factors(TEAM_STATS_RECENT, last_n_games=RECENT_GAMES_WINDOW)
        n_b = _populate_base(TEAM_STATS_RECENT, last_n_games=RECENT_GAMES_WINDOW)
        n_sc = _populate_scoring(TEAM_STATS_RECENT, last_n_games=RECENT_GAMES_WINDOW)
        _RECENT_LOADED = True
        print("[OK] Recent-{} stats loaded (Adv={} FF={} Base={} Scoring={})".format(
            RECENT_GAMES_WINDOW, n_adv, n_ff, n_b, n_sc))
    except Exception as e:
        print("[WARN] Recent stats load failed:", e)

# =====================================================
# LOCATION SPLITS (Home + Road parallel)
# =====================================================
def _load_location_splits():
    global _LOCATION_LOADED
    if _LOCATION_LOADED: return
    probe_ok = False
    for loc_kwarg in ("location_nullable", "location"):
        try:
            LeagueDashTeamStats(
                season=SEASON,
                measure_type_detailed_defense="Advanced",
                per_mode_detailed="PerGame",
                last_n_games=0,
                timeout=TIMEOUT,
                **{loc_kwarg: "Home"}
            ).get_dict()
            probe_ok = True
            print("[OK] Location param '{}' supported".format(loc_kwarg))
            break
        except TypeError:
            continue
        except Exception as e:
            probe_ok = True
            print("[WARN] Location probe network error:", e)
            break

    if not probe_ok:
        print("[WARN] Location splits: param not supported - using fallback")
        return

    results = {}
    def _load_one(loc, target, label):
        try:
            n_adv = _populate_advanced(target, last_n_games=0, location=loc)
            n_ff = _populate_four_factors(target, last_n_games=0, location=loc)
            print("[OK] {} splits: Adv={} FF={}".format(label, n_adv, n_ff))
            results[label] = True
        except Exception as e:
            print("[WARN] {} splits failed: {}".format(label, e))
            results[label] = False

    home_t = threading.Thread(target=_load_one, args=("Home", HOME_STATS, "Home"), daemon=True)
    road_t = threading.Thread(target=_load_one, args=("Road", ROAD_STATS, "Road"), daemon=True)
    home_t.start()
    road_t.start()
    home_t.join(timeout=55)
    road_t.join(timeout=55)
    if results.get("Home") and results.get("Road"):
        _LOCATION_LOADED = True

def _location_split_available():
    return bool(HOME_STATS and ROAD_STATS)

# =====================================================
# TRACKING PPP (v5.2 - broader column search + dump on miss)
# =====================================================
def _load_tracking_ppp(target_dict, last_n_games=0):
    if not PT_STATS_AVAILABLE:
        return 0
    try:
        def _call():
            return LeagueDashPtStats(
                season=SEASON,
                pt_measure_type="Possessions",
                player_or_team="Team",
                per_mode_simple="PerGame",
                last_n_games=last_n_games,
                timeout=TIMEOUT,
            ).get_dict()
        resp = with_retry(_call)
        headers = resp["resultSets"][0]["headers"]
        rows = resp["resultSets"][0]["rowSet"]
        tid_i = _find_col(headers, "TEAM_ID")

        # v5.2: broader PPP column search
        ppp_i = _find_col(headers,
            "PPP", "POINTS_PER_POSSESSION", "PTS_PER_POSS",
            "TEAM_PPP", "OFF_PPP", "OFF_POINTS_PER_POSS")
        poss_i = _find_col(headers, "POSS", "POSS_G", "POSSESSIONS", "NUM_POSS")
        touch_i = _find_col(headers, "TOUCHES", "TEAM_TOUCHES", "NUM_TOUCHES")

        if tid_i is None or ppp_i is None:
            label = "season" if last_n_games == 0 else "recent-{}".format(last_n_games)
            print("[WARN] Tracking PPP ({}): PPP column not found".format(label))
            print("[DEBUG] Available cols: {}".format(", ".join(headers)))
            return 0

        for row in rows:
            tid = row[tid_i]
            target_dict.setdefault(tid, {}).update({
                "tracking_ppp": _safe_float(row[ppp_i], LEAGUE_PPP),
                "tracking_poss": _safe_float(row[poss_i], 100.0) if poss_i else 100.0,
                "touches_pg": _safe_float(row[touch_i], 95.0) if touch_i else 95.0,
            })
        return len(rows)
    except Exception as e:
        print("[WARN] Tracking PPP load failed:", e)
        return 0

def _load_tracking():
    global _TRACKING_LOADED
    if _TRACKING_LOADED: return
    n_s = _load_tracking_ppp(TRACKING_SEASON, last_n_games=0)
    n_r = _load_tracking_ppp(TRACKING_RECENT, last_n_games=RECENT_GAMES_WINDOW)
    if n_s > 0 or n_r > 0:
        _TRACKING_LOADED = True
        print("[OK] Tracking PPP loaded (Season={} Recent={})".format(n_s, n_r))

# =====================================================
# DRIVES LOADER
# =====================================================
def _load_drives_data(target_dict, last_n_games=0):
    if not PT_STATS_AVAILABLE:
        return 0
    try:
        def _call():
            return LeagueDashPtStats(
                season=SEASON,
                pt_measure_type="Drives",
                player_or_team="Team",
                per_mode_simple="PerGame",
                last_n_games=last_n_games,
                timeout=TIMEOUT,
            ).get_dict()
        resp = with_retry(_call)
        headers = resp["resultSets"][0]["headers"]
        rows = resp["resultSets"][0]["rowSet"]
        tid_i = _find_col(headers, "TEAM_ID")
        drv_i = _find_col(headers, "DRIVES")
        drv_pts_i = _find_col(headers, "DRIVE_PTS", "DRIVE_POINTS")
        drv_fta_i = _find_col(headers, "DRIVE_FTA")

        if tid_i is None or drv_i is None:
            return 0

        LEAGUE_AVG_DRIVES = 14.0
        for row in rows:
            tid = row[tid_i]
            drives = _safe_float(row[drv_i], LEAGUE_AVG_DRIVES)
            drv_pts = _safe_float(row[drv_pts_i], 18.0) if drv_pts_i else 18.0
            drv_fta = _safe_float(row[drv_fta_i], 4.0) if drv_fta_i else 4.0
            drive_fta_factor = max(0.80, min(1.25, (drv_fta / max(drv_pts, 1.0)) / 0.22))

            target_dict.setdefault(tid, {}).update({
                "drives_pg": drives,
                "drive_pts_pg": drv_pts,
                "drive_fta_factor": drive_fta_factor,
            })
        return len(rows)
    except Exception as e:
        print("[WARN] Drives load failed:", e)
        return 0

def _load_drives():
    global _DRIVES_LOADED
    if _DRIVES_LOADED: return
    n_s = _load_drives_data(DRIVES_SEASON, last_n_games=0)
    n_r = _load_drives_data(DRIVES_RECENT, last_n_games=RECENT_GAMES_WINDOW)
    if n_s > 0 or n_r > 0:
        _DRIVES_LOADED = True
        print("[OK] Drives data loaded (Season={} Recent={})".format(n_s, n_r))

# =====================================================
# HUSTLE STATS
# =====================================================
def _load_hustle():
    global _HUSTLE_LOADED
    if not HUSTLE_AVAILABLE or _HUSTLE_LOADED: return
    try:
        def _call():
            try:
                return LeagueHustleStatsTeam(season=SEASON, per_mode_time="PerGame", timeout=TIMEOUT).get_dict()
            except TypeError:
                return LeagueHustleStatsTeam(season=SEASON, timeout=TIMEOUT).get_dict()
        resp = with_retry(_call)
        headers = resp["resultSets"][0]["headers"]
        rows = resp["resultSets"][0]["rowSet"]
        tid_i = _find_col(headers, "TEAM_ID")
        defl_i = _find_col(headers, "DEFLECTIONS")
        screen_i= _find_col(headers, "SCREEN_ASSISTS")

        if tid_i is None:
            print("[WARN] Hustle: TEAM_ID column missing")
            return

        for row in rows:
            tid = row[tid_i]
            deflections = _safe_float(row[defl_i], LEAGUE_AVG_DEFLECTIONS) if defl_i else LEAGUE_AVG_DEFLECTIONS
            screens = _safe_float(row[screen_i], LEAGUE_AVG_SCREEN_ASSISTS) if screen_i else LEAGUE_AVG_SCREEN_ASSISTS
            raw_defl = deflections / LEAGUE_AVG_DEFLECTIONS
            defl_factor = max(1.0 - HUSTLE_DEFL_CLAMP, min(1.0 + HUSTLE_DEFL_CLAMP, raw_defl))
            raw_screen_boost = (screens / LEAGUE_AVG_SCREEN_ASSISTS - 1.0) * 0.015
            screen_efg_boost = max(-HUSTLE_SCREEN_EFG_CLAMP, min(HUSTLE_SCREEN_EFG_CLAMP, raw_screen_boost))

            HUSTLE_CACHE[tid] = {
                "deflections": deflections, "screen_assists": screens,
                "defl_factor": defl_factor, "screen_efg_boost": screen_efg_boost,
            }
        _HUSTLE_LOADED = True
        print("[OK] Hustle stats loaded ({} teams, defl + screen_efg)".format(len(HUSTLE_CACHE)))
    except Exception as e:
        print("[WARN] Hustle stats failed:", e)

# =====================================================
# PLAYER AVAILABILITY (v5.2 - window varies by context)
# =====================================================
def _load_player_availability():
    global _AVAILABILITY_LOADED
    if _AVAILABILITY_LOADED: return
    # Use wider window during playoffs so stars aren't flagged for regular season rest
    window = AVAIL_WINDOW_PLAYOFF if _IS_PLAYOFF_SESSION else AVAIL_WINDOW_REGULAR
    try:
        def _call():
            return LeagueDashPlayerStats(
                season=SEASON,
                measure_type_detailed_defense="Base",
                per_mode_detailed="PerGame",
                last_n_games=window,
                timeout=TIMEOUT,
            ).get_dict()
        resp = with_retry(_call)
        headers = resp["resultSets"][0]["headers"]
        rows = resp["resultSets"][0]["rowSet"]
        tid_i = _find_col(headers, "TEAM_ID")
        pid_i = _find_col(headers, "PLAYER_ID")
        min_i = _find_col(headers, "MIN")
        gp_i = _find_col(headers, "GP")

        if None in (tid_i, pid_i, min_i):
            print("[WARN] Availability: required columns missing")
            return

        by_team = {}
        for row in rows:
            tid = row[tid_i]
            pid = row[pid_i]
            mins = _safe_float(row[min_i])
            gp = _safe_float(row[gp_i]) if gp_i else 1.0
            if mins >= 5.0 and gp >= 1.0:
                by_team.setdefault(tid, {})[pid] = mins
        PLAYER_LAST_GAME_CACHE.update(by_team)
        _AVAILABILITY_LOADED = True
        print("[OK] Player availability (last-{} games) loaded ({} teams)".format(window, len(by_team)))
    except Exception as e:
        print("[WARN] Player availability load failed:", e)

def get_availability_factor(team_id):
    # Cached - prints [AVAIL] exactly once per team per run
    if team_id in _AVAIL_FACTOR_CACHE:
        return _AVAIL_FACTOR_CACHE[team_id]

    if not _AVAILABILITY_LOADED or not _PLAYERS_LOADED:
        _AVAIL_FACTOR_CACHE[team_id] = 1.0
        return 1.0

    top_players = PLAYER_RECENT_CACHE.get(team_id, [])
    if not top_players:
        _AVAIL_FACTOR_CACHE[team_id] = 1.0
        return 1.0

    recent_active = PLAYER_LAST_GAME_CACHE.get(team_id, {})
    team_ppg = TEAM_STATS_RECENT.get(team_id, {}).get("pts_per_game", 113.0)
    if team_ppg <= 0:
        team_ppg = 113.0

    missing_pts = 0.0
    flagged = []
    for player in top_players:
        pid = player["id"]
        last_mins = recent_active.get(pid, 0.0)
        # v5.2: in playoffs require missing from more games before flagging
        min_threshold = 10.0 if _IS_PLAYOFF_SESSION else 5.0
        if last_mins < min_threshold and player["min"] >= 15.0:
            missing_pts += player["pts"]
            flagged.append(player.get("name", str(pid)))

    if flagged:
        print("[AVAIL] {} likely missing: {}".format(team_id, ", ".join(flagged)))

    missing_share = min(missing_pts / max(team_ppg, 1.0), AVAILABILITY_MAX_SHARE)
    factor = 1.0 - (missing_share * AVAILABILITY_NET_IMPACT)
    result = max(AVAILABILITY_FLOOR, factor)
    _AVAIL_FACTOR_CACHE[team_id] = result
    return result

# =====================================================
# CLUTCH LOADER
# =====================================================
def _load_clutch():
    global _CLUTCH_LOADED
    if _CLUTCH_LOADED: return
    try:
        def _call():
            return LeagueDashTeamClutch(
                season=SEASON,
                measure_type_detailed_defense="Advanced",
                per_mode_detailed="PerGame",
                timeout=TIMEOUT,
            ).get_dict()
        resp = with_retry(_call)
        headers = resp["resultSets"][0]["headers"]
        rows = resp["resultSets"][0]["rowSet"]
        tid_i = _find_col(headers, "TEAM_ID")
        off_i = _find_col(headers, "OFF_RATING", "E_OFF_RATING", "OFFENSIVE_RATING")
        def_i = _find_col(headers, "DEF_RATING", "E_DEF_RATING", "DEFENSIVE_RATING")
        pace_i = _find_col(headers, "PACE", "E_PACE")

        if None in (tid_i, off_i, def_i, pace_i):
            print("[WARN] Clutch: expected columns missing")
            return

        for row in rows:
            tid = row[tid_i]
            CLUTCH_CACHE[tid] = {
                "clutch_ortg": _safe_float(row[off_i], LEAGUE_ORTG),
                "clutch_drtg": _safe_float(row[def_i], LEAGUE_ORTG),
                "clutch_pace": _safe_float(row[pace_i], 100.0),
            }
        _CLUTCH_LOADED = True
        print("[OK] Clutch stats loaded ({} teams)".format(len(rows)))
    except Exception as e:
        print("[WARN] Clutch stats failed:", e)

# =====================================================
# PLAYER RECENT LOADER
# =====================================================
def _load_player_recent():
    global _PLAYERS_LOADED
    if _PLAYERS_LOADED: return
    try:
        def _call():
            return LeagueDashPlayerStats(
                season=SEASON,
                measure_type_detailed_defense="Base",
                per_mode_detailed="PerGame",
                last_n_games=RECENT_GAMES_WINDOW,
                timeout=TIMEOUT,
            ).get_dict()
        resp = with_retry(_call)
        headers = resp["resultSets"][0]["headers"]
        rows = resp["resultSets"][0]["rowSet"]
        pid_i = _find_col(headers, "PLAYER_ID")
        pname_i = _find_col(headers, "PLAYER_NAME")
        tid_i = _find_col(headers, "TEAM_ID")
        min_i = _find_col(headers, "MIN")
        pts_i = _find_col(headers, "PTS")
        ast_i = _find_col(headers, "AST")
        gp_i = _find_col(headers, "GP")

        if None in (pid_i, tid_i, min_i, pts_i):
            print("[WARN] Player stats: required columns missing")
            return

        by_team = {}
        for row in rows:
            tid = row[tid_i]
            mins = _safe_float(row[min_i])
            gp = _safe_float(row[gp_i]) if gp_i else 0
            if mins < 8 or gp < 3:
                continue
            by_team.setdefault(tid, []).append({
                "id": row[pid_i],
                "name": row[pname_i] if pname_i else "",
                "min": mins,
                "pts": _safe_float(row[pts_i]),
                "ast": _safe_float(row[ast_i]) if ast_i else 0.0,
                "gp": gp,
            })

        for tid, players in by_team.items():
            players.sort(key=lambda p: p["min"], reverse=True)
            PLAYER_RECENT_CACHE[tid] = players[:ML_TOP_PLAYERS]
        _PLAYERS_LOADED = True
        print("[OK] Recent-{} player stats loaded ({} teams)".format(RECENT_GAMES_WINDOW, len(by_team)))
    except Exception as e:
        print("[WARN] Player stats failed:", e)

# =====================================================
# UNIFIED ACCESSORS
# =====================================================
def _blended_team_view(team_id):
    season = TEAM_STATS_SEASON.get(team_id, {})
    recent = TEAM_STATS_RECENT.get(team_id, {})
    out = {}
    w = RECENT_FORM_WEIGHT
    keys = set(season.keys()) | set(recent.keys())

    for k in keys:
        s_val = season.get(k)
        r_val = recent.get(k)
        if r_val is not None and s_val is not None:
            out[k] = w * r_val + (1 - w) * s_val
        elif r_val is not None:
            out[k] = r_val
        else:
            out[k] = s_val

    drv_s = DRIVES_SEASON.get(team_id, {})
    drv_r = DRIVES_RECENT.get(team_id, {})
    for k in set(drv_s.keys()) | set(drv_r.keys()):
        sv = drv_s.get(k)
        rv = drv_r.get(k)
        if rv is not None and sv is not None:
            out[k] = w * rv + (1 - w) * sv
        elif rv is not None:
            out[k] = rv
        else:
            out[k] = sv

    return out

def _blended_tracking(team_id):
    s = TRACKING_SEASON.get(team_id, {})
    r = TRACKING_RECENT.get(team_id, {})
    w = RECENT_FORM_WEIGHT
    out = {}
    for k in set(s.keys()) | set(r.keys()):
        sv = s.get(k)
        rv = r.get(k)
        if rv is not None and sv is not None:
            out[k] = w * rv + (1 - w) * sv
        elif rv is not None:
            out[k] = rv
        else:
            out[k] = sv
    return out

def get_top_players_pts(team_id):
    players = PLAYER_RECENT_CACHE.get(team_id, [])
    return sum(p["pts"] for p in players) if players else 50.0

def get_team_stats(team_id):
    blended = _blended_team_view(team_id)
    clutch = CLUTCH_CACHE.get(team_id, {})
    return {
        "ortg": blended.get("ortg", LEAGUE_ORTG),
        "drtg": blended.get("drtg", LEAGUE_ORTG),
        "pace": blended.get("pace", 100.0),
        "ts": blended.get("ts", 0.58),
        "3rate": blended.get("3rate", 0.38),
        "efg": blended.get("efg", LEAGUE_EFG),
        "fta_rate": blended.get("fta_rate", LEAGUE_FTA),
        "oreb_rate": blended.get("oreb_rate", LEAGUE_OREB),
        "tov_rate": blended.get("tov_rate", LEAGUE_TOV),
        "opp_efg": blended.get("opp_efg", LEAGUE_EFG),
        "opp_tov": blended.get("opp_tov", LEAGUE_TOV),
        "opp_oreb": blended.get("opp_oreb", LEAGUE_OREB),
        "opp_fta": blended.get("opp_fta", LEAGUE_FTA),
        "ast_to_ratio": blended.get("ast_to_ratio", 1.55),
        "pts_per_game": blended.get("pts_per_game", 113.0),
        "clutch_ortg": clutch.get("clutch_ortg", blended.get("ortg", LEAGUE_ORTG)),
        "clutch_drtg": clutch.get("clutch_drtg", blended.get("drtg", LEAGUE_ORTG)),
        "clutch_pace": clutch.get("clutch_pace", blended.get("pace", 100.0)),
        "fg2_pct": blended.get("fg2_pct", 0.52),
        "fg3_pct": blended.get("fg3_pct", 0.36),
        "ft_pct": blended.get("ft_pct", 0.78),
        "fastbreak_yield": blended.get("fastbreak_yield", 1.0),
        "opp_fastbreak_yield": blended.get("opp_fastbreak_yield", 1.0),
        "pct_pts_paint": blended.get("pct_pts_paint", 0.28),
        "pct_pts_fastbreak": blended.get("pct_pts_fastbreak", 0.13),
        "pct_pts_2nd_chance": blended.get("pct_pts_2nd_chance", 0.07),
        "drive_fta_factor": blended.get("drive_fta_factor", 1.0),
        "tracking_ppp": _blended_tracking(team_id).get("tracking_ppp", None),
    }

def get_team_recent_only(team_id):
    rec = TEAM_STATS_RECENT.get(team_id, {})
    trk = TRACKING_RECENT.get(team_id, {})
    drv = DRIVES_RECENT.get(team_id, {})
    return {
        "ortg": rec.get("ortg", LEAGUE_ORTG),
        "drtg": rec.get("drtg", LEAGUE_ORTG),
        "pace": rec.get("pace", 100.0),
        "efg": rec.get("efg", LEAGUE_EFG),
        "tov_rate": rec.get("tov_rate", LEAGUE_TOV),
        "opp_efg": rec.get("opp_efg", LEAGUE_EFG),
        "opp_tov": rec.get("opp_tov", LEAGUE_TOV),
        "pts_per_game": rec.get("pts_per_game", 113.0),
        "tracking_ppp": trk.get("tracking_ppp", None),
        "drives_pg": drv.get("drives_pg", 14.0),
        "drive_fta_factor": drv.get("drive_fta_factor", 1.0),
    }

# =====================================================
# GAME LOG / FATIGUE (v5.2 - WL streak fallback)
# =====================================================
def get_team_game_log(team_id):
    if team_id in TEAM_LOG_CACHE:
        return TEAM_LOG_CACHE[team_id]
    try:
        def _call():
            return TeamGameLog(
                team_id=team_id,
                season=SEASON,
                season_type_all_star="Regular Season",
                timeout=TIMEOUT,
            ).get_dict()
        resp = with_retry(_call)
        headers = resp["resultSets"][0]["headers"]
        rows = resp["resultSets"][0]["rowSet"]
        date_i = _find_col(headers, "GAME_DATE")
        pts_i = _find_col(headers, "PTS")
        pm_i = _find_col(headers, "PLUS_MINUS")
        wl_i = _find_col(headers, "WL") # v5.2: streak fallback
        matchup_i = _find_col(headers, "MATCHUP") # bootstrap: opponent tricode

        if None in (date_i, pts_i):
            return []

        log = []
        for row in rows:
            pts = _safe_float(row[pts_i])
            entry = {"date": row[date_i], "pts": pts}
            if pm_i is not None:
                pm = _safe_float(row[pm_i])
                entry["opp_pts"] = pts - pm
            if wl_i is not None: # v5.2
                entry["wl"] = row[wl_i]
            if matchup_i is not None:
                entry["matchup"] = row[matchup_i] or ""
            log.append(entry)
        TEAM_LOG_CACHE[team_id] = log
        return log
    except Exception:
        return []

def _parse_log_date(d):
    if isinstance(d, str):
        for fmt in ("%b %d, %Y", "%Y-%m-%d", "%m/%d/%Y"):
            try:
                return datetime.datetime.strptime(d, fmt).date()
            except Exception:
                continue
    return None

def get_rest_days(team_id, game_date_str):
    try:
        log = get_team_game_log(team_id)
        if not log: return 2
        today = datetime.datetime.strptime(game_date_str, "%Y-%m-%d").date()
        past = sorted(
            [_parse_log_date(g["date"]) for g in log
             if _parse_log_date(g["date"]) and _parse_log_date(g["date"]) < today],
            reverse=True)
        if not past: return 3
        return min((today - past[0]).days, 7)
    except Exception:
        return 2

def get_fatigue_factor(team_id, game_date_str):
    rest = get_rest_days(team_id, game_date_str)
    if rest == 0: return 0.970
    elif rest == 1: return 0.985
    elif rest == 2: return 1.000
    elif rest == 3: return 1.010
    else: return 1.015

def get_win_streak(team_id):
    """
    v5.2: tries opp_pts first (from PLUS_MINUS), falls back to WL column.
    Previously returned 0 for all teams when PLUS_MINUS was absent.
    """
    try:
        log = get_team_game_log(team_id)
        if not log: return 0

        # Path A: PLUS_MINUS available
        usable = [g for g in log if "opp_pts" in g]
        if usable:
            won_last = usable[0]["pts"] > usable[0]["opp_pts"]
            streak = 0
            for g in usable:
                won = g["pts"] > g["opp_pts"]
                if won == won_last:
                    streak += (1 if won else -1)
                else:
                    break
            return max(-10, min(10, streak))

        # Path B: WL column fallback
        usable = [g for g in log if "wl" in g]
        if not usable: return 0
        won_last = usable[0]["wl"] == "W"
        streak = 0
        for g in usable:
            won = g["wl"] == "W"
            if won == won_last:
                streak += (1 if won else -1)
            else:
                break
        return max(-10, min(10, streak))
    except Exception:
        return 0

def is_back_to_back(team_id, game_date_str):
    return 1 if get_rest_days(team_id, game_date_str) == 0 else 0

def get_scoring_variance(team_id, n=20):
    try:
        log = get_team_game_log(team_id)
        if not log or len(log) < 5:
            return 0.08
        pts = [g["pts"] for g in log[:n]]
        if len(pts) < 2:
            return 0.08
        cv = statistics.stdev(pts) / max(statistics.mean(pts), 1.0)
        return max(0.04, min(0.10, cv))
    except Exception:
        return 0.08

# =====================================================
# CLOCK HELPERS
# =====================================================
def parse_clock(clock_str):
    if not clock_str:
        return None
    try:
        m = re.match(r"PT(\d+)M([\d.]+)S", clock_str)
        if m:
            return int(m.group(1)), float(m.group(2))
    except Exception:
        pass
    return None

def get_total_seconds(period):
    if period <= 4:
        return REGULATION_SECS
    return REGULATION_SECS + (period - 4) * OT_PERIOD_SECS

def get_elapsed(status_id, clock, period):
    period = int(period or 1)
    if status_id == 3: return get_total_seconds(period)
    if status_id == 1: return 0
    if period <= 4:
        period_start = (period - 1) * 720
        period_length = 720
    else:
        period_start = REGULATION_SECS + (period - 5) * OT_PERIOD_SECS
        period_length = OT_PERIOD_SECS
    parsed = parse_clock(clock)
    if parsed is None:
        return period_start
    minutes, seconds = parsed
    remaining_in_period = minutes * 60 + int(seconds)
    return period_start + (period_length - remaining_in_period)

# =====================================================
# VEGAS TOTALS
# =====================================================
def fetch_vegas_totals():
    if not ODDS_API_KEY:
        print("[ODDS] ODDS_API_KEY not set - Vegas blend disabled. Set env var ODDS_API_KEY.")
        return {}
    try:
        params = urllib.parse.urlencode({
            "apiKey": ODDS_API_KEY,
            "regions": "us",
            "markets": "totals",
            "oddsFormat": "american",
            "dateFormat": "iso",
        })
        url = ODDS_API_URL + "?" + params
        req = urllib.request.Request(url, headers={"User-Agent": "CrimsonApex/5.2"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            remaining = resp.headers.get("x-requests-remaining", "?")
            data = json.loads(resp.read().decode("utf-8"))

        tricode_map = {}
        for game in data:
            home_full = game.get("home_team", "")
            away_full = game.get("away_team", "")
            home_tri = ODDS_TEAM_MAP.get(home_full)
            away_tri = ODDS_TEAM_MAP.get(away_full)

            if not home_tri and home_full:
                matches = difflib.get_close_matches(home_full, ODDS_TEAM_MAP.keys(), n=1, cutoff=0.6)
                home_tri = ODDS_TEAM_MAP[matches[0]] if matches else None
            if not away_tri and away_full:
                matches = difflib.get_close_matches(away_full, ODDS_TEAM_MAP.keys(), n=1, cutoff=0.6)
                away_tri = ODDS_TEAM_MAP[matches[0]] if matches else None

            if not home_tri or not away_tri:
                continue

            best_line = None
            for bookmaker in game.get("bookmakers", []):
                for market in bookmaker.get("markets", []):
                    if market.get("key") == "totals":
                        for outcome in market.get("outcomes", []):
                            if outcome.get("name") == "Over":
                                best_line = float(outcome["point"])
                                break
                    if best_line is not None:
                        break
                if best_line is not None:
                    break

            if best_line is not None:
                tricode_map[(away_tri, home_tri)] = best_line

        print("[ODDS] Fetched {} game totals (requests remaining: {})".format(len(tricode_map), remaining))
        return tricode_map
    except Exception as e:
        print("[ODDS] Fetch failed:", e)
        return {}

# =====================================================
# FETCH SCOREBOARD (v5.2 - playoff detection)
# =====================================================
def fetch_nba_games():
    global _IS_PLAYOFF_SESSION
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    def _call():
        return ScoreboardV3(game_date=today, timeout=TIMEOUT).get_dict()
    data = with_retry(_call)
    games = []
    playoff_detected = False

    for g in data["scoreboard"]["games"]:
        home = g["homeTeam"]
        away = g["awayTeam"]
        status_id = int(g["gameStatus"])
        gid = g.get("gameId", "{}-{}-{}".format(home["teamId"], away["teamId"], today))

        # v5.2: detect playoff from series fields in response
        series_game_num = 0
        is_playoff = False
        for field in ("seriesGameNumber", "seriesText", "gameLabel", "gameSubLabel"):
            val = g.get(field, "")
            if val:
                text = str(val).lower()
                if any(k in text for k in ("game ", "series", "playoff", "round")):
                    is_playoff = True
                    playoff_detected = True
                if "game " in text:
                    try:
                        series_game_num = int(re.search(r"game\s+(\d)", text).group(1))
                    except Exception:
                        pass
                break

        game_dict = {
            "id": gid,
            "home": home["teamTricode"],
            "away": away["teamTricode"],
            "home_score": int(home.get("score", 0) or 0),
            "away_score": int(away.get("score", 0) or 0),
            "status_id": status_id,
            "period": int(g.get("period", 1)),
            "clock": g.get("gameClock", ""),
            "date": today,
            "home_id": home["teamId"],
            "away_id": away["teamId"],
            "minutes_left": 0,
            "vegas_line": None,
            "is_playoff": is_playoff, # v5.2
            "series_game_num": series_game_num, # v5.2
        }

        elapsed = get_elapsed(status_id, game_dict["clock"], game_dict["period"])
        total_s = get_total_seconds(game_dict["period"])
        rem = max(0, total_s - elapsed)

        if status_id == 2:
            game_dict["minutes_left"] = round(rem / 60.0, 1)
        elif status_id == 1:
            game_dict["minutes_left"] = 999
        else:
            game_dict["minutes_left"] = 0

        games.append(game_dict)

    # v5.2: set global playoff flag for availability loader + build_projection
    _IS_PLAYOFF_SESSION = playoff_detected
    if playoff_detected:
        print("[PLAYOFF] Playoff games detected - applying pace/scoring discounts")

    if games:
        vegas_totals = fetch_vegas_totals()
        matched = 0
        for game_dict in games:
            key = (game_dict["away"], game_dict["home"])
            if key in vegas_totals:
                game_dict["vegas_line"] = vegas_totals[key]
                matched += 1
        if vegas_totals:
            print("[ODDS] Matched {}/{} games to Vegas totals".format(matched, len(games)))

    return games

def preload_game_logs(games):
    team_ids = {g["home_id"] for g in games} | {g["away_id"] for g in games}
    uncached = [tid for tid in team_ids if tid not in TEAM_LOG_CACHE]
    if not uncached: return
    threads = [threading.Thread(target=get_team_game_log, args=(tid,), daemon=True) for tid in uncached]
    for t in threads: t.start()
    for t in threads: t.join(timeout=25)

# =====================================================
# BUILD PROJECTION (v5.2 - playoff discounts)
# =====================================================
def build_projection(game):
    period = game["period"]
    elapsed = get_elapsed(game["status_id"], game["clock"], period)
    total_s = get_total_seconds(period)
    remaining = max(0, total_s - elapsed)
    total = game["home_score"] + game["away_score"]
    is_playoff = game.get("is_playoff", False)

    away = get_team_stats(game["away_id"])
    home = get_team_stats(game["home_id"])

    if _location_split_available():
        home_loc = HOME_STATS.get(game["home_id"], {})
        road_loc = ROAD_STATS.get(game["away_id"], {})
        away_ortg = road_loc.get("ortg", away["ortg"])
        away_drtg = road_loc.get("drtg", away["drtg"])
        home_ortg = home_loc.get("ortg", home["ortg"])
        home_drtg = home_loc.get("drtg", home["drtg"])
        away_pace = road_loc.get("pace", away["pace"])
        home_pace = home_loc.get("pace", home["pace"])

        for _k in _LOC_FF_KEYS:
            if _k in road_loc: away[_k] = road_loc[_k]
            if _k in home_loc: home[_k] = home_loc[_k]
    else:
        away_ortg = away["ortg"]
        away_drtg = away["drtg"]
        home_ortg = home["ortg"] * HOME_COURT_BOOST_FALLBACK
        home_drtg = home["drtg"]
        away_pace = away["pace"]
        home_pace = home["pace"]

    if _HUSTLE_LOADED:
        away_hustle = HUSTLE_CACHE.get(game["away_id"], {})
        home_hustle = HUSTLE_CACHE.get(game["home_id"], {})
        home_defl_factor = home_hustle.get("defl_factor", 1.0)
        away_defl_factor = away_hustle.get("defl_factor", 1.0)
        away_screen_efg = away_hustle.get("screen_efg_boost", 0.0)
        home_screen_efg = home_hustle.get("screen_efg_boost", 0.0)

        away["tov_rate"] = min(0.22, away["tov_rate"] * home_defl_factor)
        home["tov_rate"] = min(0.22, home["tov_rate"] * away_defl_factor)
        away["efg"] = max(0.44, min(0.62, away["efg"] + away_screen_efg))
        home["efg"] = max(0.44, min(0.62, home["efg"] + home_screen_efg))

    pace = math.sqrt(away_pace * home_pace)

    eff_away_tov = away["tov_rate"] * (1 - OPP_BLEND) + home["opp_tov"] * OPP_BLEND
    eff_home_tov = home["tov_rate"] * (1 - OPP_BLEND) + away["opp_tov"] * OPP_BLEND
    avg_tov = (eff_away_tov + eff_home_tov) / 2.0
    pace *= (1 - (avg_tov - LEAGUE_TOV) * 0.35)

    avg_oreb = (away["oreb_rate"] + home["oreb_rate"]) / 2.0
    pace *= (1 + (avg_oreb - LEAGUE_OREB) * 0.15)

    away_adj = away_ortg * (home_drtg / LEAGUE_ORTG)
    home_adj = home_ortg * (away_drtg / LEAGUE_ORTG)

    if remaining < 300:
        cw = 1 - (remaining / 300.0)
        away_adj_c = away["clutch_ortg"] * (home["clutch_drtg"] / LEAGUE_ORTG)
        home_adj_c = home["clutch_ortg"] * (away["clutch_drtg"] / LEAGUE_ORTG)
        away_adj = away_adj * (1 - cw) + away_adj_c * cw
        home_adj = home_adj * (1 - cw) + home_adj_c * cw
        clutch_pace = math.sqrt(away["clutch_pace"] * home["clutch_pace"])
        pace = pace * (1 - cw) + clutch_pace * cw

    if game.get("status_id") == 1:
        away_avail = get_availability_factor(game["away_id"])
        home_avail = get_availability_factor(game["home_id"])
        away_adj *= away_avail
        home_adj *= home_avail

    # v5.2: apply playoff discounts before SPI calculation
    if is_playoff:
        pace *= PLAYOFF_PACE_DISCOUNT
        away_adj *= PLAYOFF_SCORING_DISCOUNT
        home_adj *= PLAYOFF_SCORING_DISCOUNT

    scoring_power_index = ((away_adj + home_adj) / 2.0) / 100.0

    away_tppp = away["tracking_ppp"]
    home_tppp = home["tracking_ppp"]
    if away_tppp is not None and home_tppp is not None:
        tracking_ppp = (away_tppp + home_tppp) / 2.0
        scoring_power_index = (scoring_power_index * (1 - TRACKING_PPP_WEIGHT) + tracking_ppp * TRACKING_PPP_WEIGHT)

    away_fta_adj = away["fta_rate"] * away.get("drive_fta_factor", 1.0)
    home_fta_adj = home["fta_rate"] * home.get("drive_fta_factor", 1.0)
    avg_drive_fta = (away_fta_adj + home_fta_adj) / 2.0

    fatigue = (get_fatigue_factor(game["away_id"], game["date"]) +
               get_fatigue_factor(game["home_id"], game["date"])) / 2.0
    pace *= fatigue
    scoring_power_index *= fatigue

    if elapsed > BLEND_START_ELAPSED and total > 0:
        poss_elapsed = (elapsed / REGULATION_SECS) * pace * 2.0
        live_spi = total / max(poss_elapsed, 1.0)
        live_spi = max(scoring_power_index * 0.80, min(scoring_power_index * 1.20, live_spi))
        w = (elapsed - BLEND_START_ELAPSED) / (total_s - BLEND_START_ELAPSED)
        scoring_power_index = ((1 - min(w, 1.0)) * scoring_power_index + min(w, 1.0) * live_spi)

    return total, pace, scoring_power_index, remaining, away, home, away_adj, home_adj, avg_drive_fta

# =====================================================
# POSSESSION ENGINE
# =====================================================
PossResult = namedtuple("PossResult", ["off_pts", "def_pts", "sec", "next_offense", "next_defense"])

def simulate_one_possession_v2(blended_off, blended_def, offense, defense, shot_clock=24.0, is_putback=False):
    tov_prob = blended_off["tov_rate"] * (0.92 if is_putback else 1.0)
    if random.random() < tov_prob:
        is_steal = random.random() < 0.65
        if is_steal:
            fb_yield = blended_def["fastbreak_yield"]
            sec = max(4.0, random.normalvariate(6.5, 2.0))
            layup_w = 0.45 * min(fb_yield, 1.5)
            three_w = 0.15 * min(fb_yield, 1.5)
            and1_w = 0.10 * min(fb_yield, 1.5)
            roll = random.random()
            if roll < layup_w:
                def_pts = 2.0
            elif roll < layup_w + and1_w:
                def_pts = 2.0 + (1.0 if random.random() < blended_def["ft_pct"] else 0.0)
            elif roll < layup_w + and1_w + three_w:
                def_pts = 3.0 if random.random() < blended_def["fg3_pct"] else 0.0
            else:
                def_pts = 0.0
            return PossResult(0.0, def_pts, sec, defense, offense)
        else:
            sec = max(5.0, random.normalvariate(11.0, 2.5))
            return PossResult(0.0, 0.0, sec, defense, offense)

    fta_prob = blended_off["fta_rate"]
    if is_putback: fta_prob = min(0.48, fta_prob + 0.15)
    if random.random() < fta_prob:
        num_ft = 3 if random.random() < 0.18 else 2
        ft_made = sum(1 for _ in range(num_ft) if random.random() < blended_off["ft_pct"])
        pts = float(ft_made)
        if ft_made < num_ft:
            oreb_prob = blended_off["oreb_rate"] * 1.15
            if random.random() < oreb_prob:
                sec = max(8.0, random.normalvariate(22.0, 3.5))
                inner = simulate_one_possession_v2(blended_off, blended_def, offense, defense, 14.0, True)
                return PossResult(pts + inner.off_pts, inner.def_pts, sec + inner.sec, inner.next_offense, inner.next_defense)
        sec = max(8.0, random.normalvariate(20.5, 3.5))
        return PossResult(pts, 0.0, sec, defense, offense)

    is_three = random.random() < blended_off["three"]
    make_prob = blended_off["fg3_pct"] if is_three else blended_off["fg2_pct"]
    shot_val = 3 if is_three else 2
    if is_putback: make_prob = min(0.90, make_prob + 0.09)
    raw_sec = random.normalvariate(14.0 if shot_val == 2 else 13.0, 2.8)
    sec = max(5.0, min(raw_sec, shot_clock))

    if random.random() < make_prob:
        pts = float(shot_val)
        and1_r = 0.085 if shot_val == 2 else 0.035
        if random.random() < and1_r and random.random() < blended_off["ft_pct"]:
            pts += 1.0
        return PossResult(pts, 0.0, sec, defense, offense)

    oreb_prob = blended_off["oreb_rate"]
    if is_putback: oreb_prob = min(0.60, oreb_prob + 0.18)
    if random.random() < oreb_prob:
        inner = simulate_one_possession_v2(blended_off, blended_def, offense, defense, 14.0, True)
        return PossResult(inner.off_pts, inner.def_pts, sec + inner.sec, inner.next_offense, inner.next_defense)

    return PossResult(0.0, 0.0, sec, defense, offense)

def simulate_game_from_current_state(game, blended_home, blended_away, away, home, home_score, away_score, time_left, status_id):
    offense = "away" if random.random() < 0.50 else "home"
    defense = "home" if offense == "away" else "away"
    sim_home = float(home_score)
    sim_away = float(away_score)
    ot_played = False

    while time_left > 0:
        blended_off = blended_home if offense == "home" else blended_away
        blended_def = blended_home if defense == "home" else blended_away
        margin = sim_home - sim_away
        abs_margin = abs(margin)
        leading = "home" if margin > 0 else "away"

        if abs_margin <= 8 and time_left < 120 and status_id == 2:
            if offense == leading:
                sec = random.uniform(5, 12)
                ft_bl = blended_home if offense == "home" else blended_away
                ft_made = sum(1 for _ in range(2) if random.random() < ft_bl["ft_pct"])
                if offense == "home": sim_home += ft_made
                else: sim_away += ft_made
                time_left -= sec
                offense, defense = defense, offense
                continue

        result = simulate_one_possession_v2(blended_off, blended_def, offense, defense)
        if offense == "home":
            sim_home += result.off_pts
            sim_away += result.def_pts
        else:
            sim_away += result.off_pts
            sim_home += result.def_pts

        time_left -= max(3.0, result.sec)
        offense = result.next_offense
        defense = result.next_defense

        if time_left <= 0 and not ot_played:
            if (sim_home - sim_away) == 0:
                time_left += OT_PERIOD_SECS
                ot_played = True

    return round(sim_home, 1), round(sim_away, 1), round(sim_home + sim_away, 1)

def monte_carlo_possession(game):
    result = build_projection(game)
    total, pace, spi, rem, away, home, away_adj, home_adj, avg_drive_fta = result
    status_id = game["status_id"]

    if rem <= 0:
        away_share = away_adj / max(away_adj + home_adj, 1.0)
        home_share = 1 - away_share
        return (total, total, 0, total, total,
                round(total * away_share, 1), round(total * home_share, 1), [])

    def _blend(off, opp):
        base_fta = off["fta_rate"] * off.get("drive_fta_factor", 1.0)
        return {
            "tov_rate": off["tov_rate"] * (1 - OPP_BLEND) + opp["opp_tov"] * OPP_BLEND,
            "fta_rate": min(0.45, base_fta * (1 - OPP_BLEND) + opp["opp_fta"] * OPP_BLEND),
            "three": off["3rate"],
            "fg2_pct": off["fg2_pct"],
            "fg3_pct": off["fg3_pct"],
            "ft_pct": off["ft_pct"],
            "oreb_rate": off["oreb_rate"],
            "fastbreak_yield": (off["fastbreak_yield"] + opp["opp_fastbreak_yield"]) / 2.0,
        }

    blended_away = _blend(away, home)
    blended_home = _blend(home, away)
    results_home, results_away, results_total = [], [], []

    for _ in range(IN_GAME_SIMS):
        fh, fa, ft = simulate_game_from_current_state(
            game, blended_home, blended_away, away, home,
            game["home_score"], game["away_score"], float(rem), status_id)
        results_home.append(fh)
        results_away.append(fa)
        results_total.append(ft)

    median_total = statistics.median(results_total)
    mean_total = statistics.mean(results_total)
    stdev_total = statistics.stdev(results_total) if len(results_total) > 1 else 0
    q = statistics.quantiles(results_total, n=4)
    median_away = round(statistics.median(results_away), 1)
    median_home = round(statistics.median(results_home), 1)

    return (round(median_total, 1), round(mean_total, 1), round(stdev_total, 1),
            round(q[0], 1), round(q[2], 1), median_away, median_home, results_total)

def monte_carlo(game):
    if game.get("status_id") == 2 and game.get("period", 0) >= 4:
        return monte_carlo_possession(game)

    result = build_projection(game)
    total, pace, spi, rem, away, home, away_adj, home_adj, avg_drive_fta = result
    margin = game["home_score"] - game["away_score"]
    abs_margin = abs(margin)
    status_id = game["status_id"]
    away_share = away_adj / max(away_adj + home_adj, 1.0)
    home_share = 1 - away_share

    if rem <= 0:
        return (total, total, 0, total, total,
                round(total * away_share, 1), round(total * home_share, 1), [])

    eff_away_tov = away["tov_rate"] * (1 - OPP_BLEND) + home["opp_tov"] * OPP_BLEND
    eff_home_tov = home["tov_rate"] * (1 - OPP_BLEND) + away["opp_tov"] * OPP_BLEND
    tov_rate = (eff_away_tov + eff_home_tov) / 2.0

    eff_away_efg = away["efg"] * (1 - OPP_BLEND) + home["opp_efg"] * OPP_BLEND
    eff_home_efg = home["efg"] * (1 - OPP_BLEND) + away["opp_efg"] * OPP_BLEND
    avg_efg = (eff_away_efg + eff_home_efg) / 2.0

    ft_rate = avg_drive_fta
    three = (away["3rate"] + home["3rate"]) / 2.0
    base_sec = (REGULATION_SECS / max(pace, 1.0)) / 2.0
    away_var = get_scoring_variance(game["away_id"])
    home_var = get_scoring_variance(game["home_id"])
    game_var = (away_var + home_var) / 2.0

    efg_3pt_shift = max(0.0, (avg_efg - LEAGUE_EFG) * 2.0)
    paint_weight = (away.get("pct_pts_paint", 0.28) + home.get("pct_pts_paint", 0.28)) / 2.0
    fb_weight = (away.get("pct_pts_fastbreak", 0.13) + home.get("pct_pts_fastbreak", 0.13)) / 2.0

    w2 = max(0.0, 0.75 - three * 0.30 - efg_3pt_shift * 0.15 + paint_weight * 0.10)
    w3 = max(0.0, 0.20 + three * 0.30 + efg_3pt_shift * 0.15)
    w1 = max(0.0, 0.05 + fb_weight * 0.10)
    wsum = w1 + w2 + w3

    if wsum > 0:
        w1 /= wsum; w2 /= wsum; w3 /= wsum
    else:
        w1, w2, w3 = 0.05, 0.75, 0.20

    base_power = (1.0 * w1) + (2.0 * w2) + (3.0 * w3)
    power_floor = max(0.85, avg_efg * 1.5)
    scoring_power = max(base_power, power_floor)
    results = []

    for _ in range(SIMS):
        sim_spi_mult = random.normalvariate(1.0, game_var)
        sim_spi = spi * max(0.85, min(1.15, sim_spi_mult))
        scoring_prob = min(max(sim_spi / scoring_power, 0.1), 0.9)
        prob_1 = scoring_prob * w1
        prob_2 = scoring_prob * w2
        prob_3 = scoring_prob * w3
        prob_2 *= (1 - (tov_rate - LEAGUE_TOV) * 0.4)
        prob_0 = 1.0 - (prob_1 + prob_2 + prob_3)
        weights = [max(prob_0, 0), max(prob_1, 0), max(prob_2, 0), max(prob_3, 0)]

        if sum(weights) == 0:
            weights = [0.10, 0.05, 0.70, 0.15]

        if status_id == 2:
            if abs_margin <= 5: pace_mult = random.uniform(1.10, 1.25)
            elif abs_margin >= 15: pace_mult = random.uniform(0.85, 1.05)
            else: pace_mult = random.uniform(0.95, 1.05)
        else:
            pace_mult = 1.0

        adjusted_sec = base_sec * pace_mult
        sim_score = total
        time_left = float(rem)
        ot_played = False

        while time_left > 0:
            sec = random.normalvariate(adjusted_sec, adjusted_sec * 0.08)
            if abs_margin <= 8 and time_left < 120 and status_id == 2:
                sec = random.uniform(5, 12)
                ft_made = sum(1 for _ in range(2) if random.random() < ft_rate)
                sim_score += ft_made
                time_left -= sec
                continue

            time_left -= sec
            if time_left >= 0:
                sim_score += random.choices([0, 1, 2, 3], weights=weights)[0]

            if time_left <= 0 and not ot_played:
                if abs_margin < 1 and random.random() < 0.055:
                    time_left += OT_PERIOD_SECS
                    ot_played = True

        results.append(sim_score)

    median = statistics.median(results)
    mean = statistics.mean(results)
    stdev = statistics.stdev(results) if len(results) > 1 else 0
    q = statistics.quantiles(results, n=4)

    if game.get("status_id") == 1 and game.get("vegas_line") is not None:
        median = (median * (1 - VEGAS_WEIGHT)) + (game["vegas_line"] * VEGAS_WEIGHT)

    return (round(median, 1), round(mean, 1), round(stdev, 1),
            round(q[0], 1), round(q[2], 1),
            round(median * away_share, 1), round(median * home_share, 1), results)

# =====================================================
# PERSISTENCE
# =====================================================
REFRESH_INTERVAL = 60
SAVE_FILE = "apex_tracker_data_v5.json"
data_lock = threading.RLock()
tracked_games = set()
user_lines = {}
game_projections = {}
game_history = {}

def safe_save_data():
    with data_lock:
        try:
            data = {
                "timestamp": time.time(),
                "tracked_games": list(tracked_games),
                "user_lines": user_lines,
                "game_projections": game_projections,
                "game_history": game_history,
            }
            with open(SAVE_FILE, "w") as f:
                json.dump(data, f)
        except Exception as e:
            print("Save error:", e)

def load_data():
    global tracked_games, user_lines, game_projections, game_history
    if os.path.exists(SAVE_FILE):
        try:
            with open(SAVE_FILE, "r") as f:
                data = json.load(f)
            tracked_games = set(str(x) for x in data.get("tracked_games", []))
            user_lines = data.get("user_lines", {})
            game_projections = data.get("game_projections", {})
            game_history = data.get("game_history", {})
        except Exception as e:
            print("Load error:", e)

def update_game_history(games):
    global game_history
    changed = False
    with data_lock:
        for g in games:
            gid = str(g["id"])
            if gid not in game_history:
                game_history[gid] = {
                    "date": g["date"], "home": g["home"], "away": g["away"],
                    "proj": None, "proj_away": None, "proj_home": None,
                    "proj_low": None, "proj_high": None, "actual": None,
                    "is_playoff": g.get("is_playoff", False), # v5.2
                    "series_game_num": g.get("series_game_num", 0), # v5.2
                }
                changed = True

            if g.get("status_id") == 1 and g.get("projection") is not None:
                if game_history[gid]["proj"] is None:
                    game_history[gid].update({
                        "proj": g["projection"],
                        "proj_away": g.get("proj_away"),
                        "proj_home": g.get("proj_home"),
                        "proj_low": g.get("proj_low"),
                        "proj_high": g.get("proj_high"),
                    })
                changed = True

            if g.get("status_id") == 3:
                actual = g["home_score"] + g["away_score"]
                if game_history[gid]["actual"] != actual:
                    game_history[gid]["actual"] = actual
                    changed = True

        if changed:
            safe_save_data()

# =====================================================
# ML SNAPSHOT (v5.2 - 18 features, same filename)
# =====================================================
def _save_ml_snapshots():
    try:
        with open(ML_HISTORY_SAVE_FILE, "w") as f:
            json.dump(ML_SNAPSHOTS, f)
    except Exception as e:
        print("[ML] Snapshot save error:", e)

def _load_ml_snapshots():
    global ML_SNAPSHOTS
    if os.path.exists(ML_HISTORY_SAVE_FILE):
        try:
            with open(ML_HISTORY_SAVE_FILE, "r") as f:
                ML_SNAPSHOTS = json.load(f)
            print("[ML] Loaded {} feature snapshots".format(len(ML_SNAPSHOTS)))
        except Exception as e:
            print("[ML] Snapshot load error:", e)

# v5.2: added is_playoff + series_game_num (18 total)
FEATURE_NAMES = [
    "proj", "spread", "pace", "away_off_edge", "home_off_edge",
    "avg_efg", "avg_tov", "away_top5_pts", "home_top5_pts",
    "recent_total_avg", "away_rest", "home_rest",
    "away_streak", "home_streak", "away_var", "home_var",
    "is_playoff", "series_game_num",
]

def _build_feature_vector(g, proj, proj_low, proj_high):
    away_r = get_team_recent_only(g["away_id"])
    home_r = get_team_recent_only(g["home_id"])
    pace_blend = math.sqrt(away_r["pace"] * home_r["pace"])
    spread = proj_high - proj_low

    away_off_edge = away_r["ortg"] - home_r["drtg"]
    home_off_edge = home_r["ortg"] - away_r["drtg"]

    avg_efg = (away_r["efg"] * (1 - OPP_BLEND) + home_r["opp_efg"] * OPP_BLEND
               + home_r["efg"] * (1 - OPP_BLEND) + away_r["opp_efg"] * OPP_BLEND) / 2.0
    avg_tov = (away_r["tov_rate"] * (1 - OPP_BLEND) + home_r["opp_tov"] * OPP_BLEND
               + home_r["tov_rate"] * (1 - OPP_BLEND) + away_r["opp_tov"] * OPP_BLEND) / 2.0

    away_top5 = get_top_players_pts(g["away_id"])
    home_top5 = get_top_players_pts(g["home_id"])
    recent_total_avg = away_r["pts_per_game"] + home_r["pts_per_game"]
    away_rest = get_rest_days(g["away_id"], g["date"])
    home_rest = get_rest_days(g["home_id"], g["date"])
    away_streak = get_win_streak(g["away_id"])
    home_streak = get_win_streak(g["home_id"])
    away_var = get_scoring_variance(g["away_id"])
    home_var = get_scoring_variance(g["home_id"])
    is_playoff = 1.0 if g.get("is_playoff") else 0.0 # v5.2
    series_game_num = float(g.get("series_game_num", 0)) # v5.2

    return [
        proj, spread, pace_blend, away_off_edge, home_off_edge,
        avg_efg, avg_tov, away_top5, home_top5, recent_total_avg,
        float(away_rest), float(home_rest), float(away_streak), float(home_streak),
        away_var, home_var, is_playoff, series_game_num,
    ]

def _snapshot_game_features(g, median, mean, stdev, low, high):
    if not ML_AVAILABLE: return
    gid = str(g["id"])
    if gid in ML_SNAPSHOTS: return
    try:
        features = _build_feature_vector(g, median, low, high)
        ML_SNAPSHOTS[gid] = {
            "features": features,
            "names": FEATURE_NAMES,
            "proj": median,
            "date": g["date"],
            "home": g["home"],
            "away": g["away"],
        }
        _save_ml_snapshots()
    except Exception as e:
        print("[ML] Snapshot failed for {}: {}".format(gid, e))


def set_line(game_key, line):
    """Set O/U line for a game. Use tricode format 'AWAY@HOME' or game id."""
    user_lines[str(game_key)] = float(line)
    safe_save_data()
    print("Line set: {} = {}".format(game_key, line))

def ou_probability(sim_results, line):
    """Return (p_over, p_under) from MC simulation results."""
    if not sim_results or line is None:
        return None, None
    n = len(sim_results)
    p_over = sum(1 for s in sim_results if s > line) / n
    return round(p_over, 3), round(1.0 - p_over, 3)

def _parse_opp_tricode(matchup, my_tricode):
    """Parse opponent tricode from NBA matchup string like 'LAL @ BOS' or 'LAL vs. BOS'."""
    if " @ " in matchup:
        parts = matchup.split(" @ ")
    elif " vs. " in matchup:
        parts = matchup.split(" vs. ")
    else:
        return None
    if len(parts) != 2:
        return None
    left, right = parts[0].strip(), parts[1].strip()
    if left == my_tricode:
        return right
    if right == my_tricode:
        return left
    return None

def _bootstrap_ml_from_logs():
    """Mine TEAM_LOG_CACHE using MATCHUP field to build ML training samples on cold start."""
    if not ML_AVAILABLE:
        return
    seen = set()
    bootstrap_snapshots = {}
    bootstrap_history = {}

    for team_id, log in TEAM_LOG_CACHE.items():
        my_tri = _TRICODE_FROM_ID.get(team_id, "")
        for entry in log:
            if "opp_pts" not in entry or "matchup" not in entry:
                continue
            actual_total = entry["pts"] + entry["opp_pts"]
            if not (170 <= actual_total <= 290):
                continue
            date_str = entry.get("date", "")
            matchup = entry["matchup"]
            opp_tri = _parse_opp_tricode(matchup, my_tri)
            if not opp_tri:
                continue
            opp_id = TRICODE_TO_ID.get(opp_tri)
            if not opp_id:
                continue
            # Determine home/away: "MY @ OPP" means MY is away; "MY vs. OPP" means MY is home
            if " @ " in matchup and matchup.split(" @ ")[0].strip() == my_tri:
                away_id, home_id = team_id, opp_id
            elif " vs. " in matchup and matchup.split(" vs. ")[0].strip() == my_tri:
                home_id, away_id = team_id, opp_id
            else:
                away_id, home_id = opp_id, team_id
            # Dedup: same game appears in both teams' logs
            dedup_key = tuple(sorted([team_id, opp_id])) + (date_str,)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            # Build synthetic game dict
            gid = "boot_{}_{}".format(date_str.replace(" ", ""), team_id)
            fake_g = {
                "id": gid,
                "away_id": away_id,
                "home_id": home_id,
                "date": date_str,
                "is_playoff": False,
                "series_game_num": 0,
            }
            try:
                away_r = get_team_recent_only(away_id)
                home_r = get_team_recent_only(home_id)
                est_proj = away_r["pts_per_game"] + home_r["pts_per_game"]
                est_low = est_proj * 0.93
                est_high = est_proj * 1.07
                features = _build_feature_vector(fake_g, est_proj, est_low, est_high)
            except Exception:
                continue
            away_tri = _TRICODE_FROM_ID.get(away_id, str(away_id))
            home_tri = _TRICODE_FROM_ID.get(home_id, str(home_id))
            bootstrap_snapshots[gid] = {
                "features": features,
                "names": FEATURE_NAMES,
                "proj": est_proj,
                "date": date_str,
                "home": home_tri,
                "away": away_tri,
            }
            bootstrap_history[gid] = {
                "date": date_str,
                "home": home_tri,
                "away": away_tri,
                "proj": est_proj,
                "actual": actual_total,
                "proj_away": None,
                "proj_home": None,
                "proj_low": est_low,
                "proj_high": est_high,
                "is_playoff": False,
                "series_game_num": 0,
            }

    n = len(bootstrap_snapshots)
    if n == 0:
        print("[ML BOOTSTRAP] No samples found (logs missing MATCHUP or opp_pts)")
        return
    print("[ML BOOTSTRAP] Built {} training samples from game logs".format(n))
    ML_SNAPSHOTS.update(bootstrap_snapshots)
    game_history.update(bootstrap_history)
    train_ml_model()

# =====================================================
# TRAIN ML
# =====================================================
def train_ml_model():
    global ML_MODEL, ML_SCALER, ML_TRAINED, ML_CV_MAE, ML_RAW_MAE, ML_BIAS
    if not ML_AVAILABLE: return

    rows = []
    for gid, snap in ML_SNAPSHOTS.items():
        rec = game_history.get(gid)
        if not rec or rec.get("actual") is None: continue
        # v5.2: accept both 16-feature (v5) and 18-feature (v5.2) snapshots
        if "features" not in snap: continue
        n_feat = len(snap["features"])
        if n_feat not in (16, len(FEATURE_NAMES)): continue
        residual = rec["actual"] - snap["proj"]
        rows.append((snap["date"], snap["features"], residual, n_feat))

    rows.sort(key=lambda x: x[0] or "", reverse=True)
    rows = rows[:ML_TRAINING_WINDOW]
    n = len(rows)
    if n < ML_MIN_TO_ACTIVATE:
        print("[ML] Not enough data: {}/{} (need {} more)".format(
            n, ML_MIN_TO_ACTIVATE, ML_MIN_TO_ACTIVATE - n))
        return

    # Pad old 16-feature vectors with zeros for the 2 new fields
    target_len = len(FEATURE_NAMES)
    X = np.array(
        [r[1] + [0.0] * (target_len - r[3]) for r in rows],
        dtype=float
    )
    y = np.array([r[2] for r in rows], dtype=float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = Ridge(alpha=ML_RIDGE_ALPHA)
    model.fit(X_scaled, y)

    ML_MODEL = model
    ML_SCALER = scaler
    ML_TRAINED = True

    in_sample_pred = model.predict(X_scaled)
    in_sample_mae = float(np.mean(np.abs(y - in_sample_pred)))
    raw_mae = float(np.mean(np.abs(y)))
    bias = float(np.mean(y))
    cv_mae = None

    if n >= 15:
        try:
            k = min(5, max(2, n // 5))
            cv_scores = cross_val_score(Ridge(alpha=ML_RIDGE_ALPHA), X_scaled, y,
                                        cv=k, scoring="neg_mean_absolute_error")
            cv_mae = float(-cv_scores.mean())
        except Exception as e:
            print("[ML] CV failed:", e)

    ML_RAW_MAE = raw_mae
    ML_CV_MAE = cv_mae
    ML_BIAS = bias

    print("[ML] Trained on {} games (window={} alpha={})".format(n, ML_TRAINING_WINDOW, ML_RIDGE_ALPHA))
    print("[ML] Raw MC bias : {:+.2f} pts".format(bias))
    print("[ML] Raw MC MAE : {:.2f} pts".format(raw_mae))
    print("[ML] After ML (in-samp): {:.2f} pts".format(in_sample_mae))

    if cv_mae is not None:
        delta = raw_mae - cv_mae
        verdict = "HELPING" if delta > 0.2 else ("NEUTRAL" if delta > -0.2 else "HURTING")
        print("[ML] After ML (5-fold CV): {:.2f} pts [{} {:+.2f}]".format(cv_mae, verdict, delta))

# =====================================================
# ML CORRECTION
# =====================================================
def ml_correct(g, median, low, high, mean, stdev):
    if not ML_AVAILABLE or not ML_TRAINED: return median, 0.0
    try:
        gid = str(g["id"])
        snap = ML_SNAPSHOTS.get(gid)
        if snap and "features" in snap and len(snap["features"]) == len(FEATURE_NAMES):
            features = snap["features"]
        else:
            features = _build_feature_vector(g, median, low, high)

        # Pad if loaded from old 16-feature snapshot
        target_len = len(FEATURE_NAMES)
        if len(features) < target_len:
            features = features + [0.0] * (target_len - len(features))
        if len(features) != target_len:
            return median, 0.0

        X = np.array([features], dtype=float)
        X_scaled = ML_SCALER.transform(X)
        raw_delta = float(ML_MODEL.predict(X_scaled)[0])
        delta = max(-ML_MAX_CORRECTION, min(ML_MAX_CORRECTION, raw_delta))
        return round(median + delta, 1), delta
    except Exception as e:
        print("[ML] Correction error:", e)
        return median, 0.0

# =====================================================
# CALCULATE PROJECTION
# =====================================================
def calculate_projection(g):
    try:
        median, mean, stdev, low, high, away_t, home_t, sim_results = monte_carlo(g)
        ml_delta = 0.0
        final_median = median

        if g.get("status_id") == 1 and ML_AVAILABLE:
            _snapshot_game_features(g, median, mean, stdev, low, high)
            final_median, ml_delta = ml_correct(g, median, low, high, mean, stdev)

        g["projection"] = final_median
        g["proj_raw"] = median
        g["proj_ml_delta"] = ml_delta
        g["proj_away"] = away_t
        g["proj_home"] = home_t
        g["proj_low"] = low
        g["proj_high"] = high
        g["mc_results"] = sim_results

        # O/U probability from MC simulation
        line = None
        game_key_str = "{}@{}".format(g.get("away", ""), g.get("home", ""))
        if str(g["id"]) in user_lines:
            line = user_lines[str(g["id"])]
        elif game_key_str in user_lines:
            line = user_lines[game_key_str]
        elif g.get("vegas_line"):
            line = g["vegas_line"]
        if line is not None and sim_results:
            p_over, p_under = ou_probability(sim_results, line)
            g["ou_line"] = line
            g["p_over"] = p_over
            g["p_under"] = p_under

        if g.get("status_id") == 1:
            away_s = get_team_stats(g["away_id"])
            home_s = get_team_stats(g["home_id"])
            loc_note = " [loc-split FF]" if _location_split_available() else " [HCA fallback]"
            po_note = " [PLAYOFF]" if g.get("is_playoff") else ""
            vegas_note = " Vegas: {}".format(g["vegas_line"]) if g.get("vegas_line") else " [no Vegas line]"

            text = "APEX TOTAL : {}\n".format(final_median)
            text += loc_note + po_note + "\n"
            text += vegas_note + "\n"

            if ML_TRAINED and ml_delta != 0.0:
                text += " (MC raw: {} ML adj: {:+.1f})\n".format(median, ml_delta)
            elif ML_AVAILABLE:
                text += " (ML inactive - need {} snapshots)\n".format(ML_MIN_TO_ACTIVATE)
            else:
                text += " (ML unavailable)\n"

            text += " (mean {} stdev {})\n".format(mean, stdev)
            text += "Split : {} {} | {} {}\n".format(g["away"], away_t, g["home"], home_t)
            text += "Range : {} - {}\n\n".format(low, high)
            text += "Recent-{} Ratings:\n".format(RECENT_GAMES_WINDOW)
            text += " {} ortg={:.1f} drtg={:.1f} rest={}\n".format(
                g["away"], away_s["ortg"], away_s["drtg"], get_rest_days(g["away_id"], g["date"]))
            text += " {} ortg={:.1f} drtg={:.1f} rest={}".format(
                g["home"], home_s["ortg"], home_s["drtg"], get_rest_days(g["home_id"], g["date"]))

            away_avail = get_availability_factor(g["away_id"])
            home_avail = get_availability_factor(g["home_id"])
            if away_avail < 1.0 or home_avail < 1.0:
                text += "\nAvailability: {} {:.0f}% | {} {:.0f}%".format(
                    g["away"], away_avail * 100, g["home"], home_avail * 100)
            g["proj_text"] = text

        gid = str(g["id"])
        ml = g.get("minutes_left", 0)
        with data_lock:
            if gid not in game_projections:
                game_projections[gid] = {}
            if 8 < ml <= 10 and "10" not in game_projections[gid]: game_projections[gid]["10"] = final_median
            elif 6 < ml <= 8 and "8" not in game_projections[gid]: game_projections[gid]["8"] = final_median
            elif 4 < ml <= 6 and "6" not in game_projections[gid]: game_projections[gid]["6"] = final_median
            elif 0 < ml <= 4 and "4" not in game_projections[gid]: game_projections[gid]["4"] = final_median
            safe_save_data()
        return final_median
    except Exception as e:
        print("[WARN] Projection failed:", e)
        import traceback
        traceback.print_exc()
        g["projection"] = None
        return None

def get_period_label(game):
    p = game.get("period", 0)
    sid = game.get("status_id", 0)
    if sid == 3: return "FINAL"
    if sid == 1: return "PREGAME"
    if sid != 2 or p == 0: return ""

    parsed = parse_clock(game.get("clock", ""))
    clock_str = "{}:{:02d}".format(int(parsed[0]), int(parsed[1])) if parsed else ""
    if p <= 4:
        label = "Q{}".format(p)
    else:
        ot = p - 4
        label = "{}OT".format(ot) if ot > 1 else "OT"
    return (label + " " + clock_str).strip()

# =====================================================
# DIAGNOSTICS
# =====================================================
def ml_diagnostics():
    print("\n========== ML DIAGNOSTICS (v5.2) ==========")
    print(" ML_AVAILABLE :", ML_AVAILABLE)
    print(" ML_TRAINED :", ML_TRAINED)
    print(" ML_WINDOW :", ML_TRAINING_WINDOW)
    print(" ML_MIN_ACTIVATE :", ML_MIN_TO_ACTIVATE)
    print(" Ridge alpha :", ML_RIDGE_ALPHA)
    print(" Max correction : +/- {:.1f} pts".format(ML_MAX_CORRECTION))
    print(" Feature count :", len(FEATURE_NAMES))
    print(" Playoff session :", _IS_PLAYOFF_SESSION)
    print(" Pace discount :", PLAYOFF_PACE_DISCOUNT if _IS_PLAYOFF_SESSION else "N/A")
    print(" Score discount :", PLAYOFF_SCORING_DISCOUNT if _IS_PLAYOFF_SESSION else "N/A")
    print("")
    print(" Vegas API key :", "SET" if ODDS_API_KEY else "NOT SET")
    print(" Hustle loaded :", _HUSTLE_LOADED)
    print(" Avail loaded :", _AVAILABILITY_LOADED)
    print(" Location FF :", _location_split_available())
    print(" Tracking PPP :", _TRACKING_LOADED)
    print(" Drives data :", _DRIVES_LOADED)

    if _HUSTLE_LOADED and HUSTLE_CACHE:
        defl_vals = [v["deflections"] for v in HUSTLE_CACHE.values()]
        screen_vals = [v["screen_assists"] for v in HUSTLE_CACHE.values()]
        print(" Hustle teams :", len(HUSTLE_CACHE))
        print(" Deflections range: {:.1f} - {:.1f} (avg {:.1f})".format(
            min(defl_vals), max(defl_vals), sum(defl_vals)/len(defl_vals)))
        print(" Screen ast range : {:.1f} - {:.1f} (avg {:.1f})".format(
            min(screen_vals), max(screen_vals), sum(screen_vals)/len(screen_vals)))

    if _AVAILABILITY_LOADED:
        print(" Avail teams :", len(PLAYER_LAST_GAME_CACHE))

    completed = [v for v in game_history.values()
                 if v.get("proj") is not None and v.get("actual") is not None]
    po_comp = [v for v in completed if v.get("is_playoff")]
    print("\n Total completed :", len(completed))
    print(" Playoff games :", len(po_comp))
    print(" ML snapshots :", len(ML_SNAPSHOTS))

    if not completed:
        print(" No completed games yet.")
        print("==========================================\n")
        return

    errors = [v["actual"] - v["proj"] for v in completed]
    abs_errors = [abs(e) for e in errors]
    bias = sum(errors) / len(errors)
    mae = sum(abs_errors) / len(abs_errors)
    over = sum(1 for e in errors if e > 0)
    under = sum(1 for e in errors if e < 0)

    print("\n --- Raw MC Performance (all) ---")
    print(" Bias : {:+.2f} pts".format(bias))
    print(" MAE : {:.2f} pts".format(mae))
    print(" Over : {} ({:.0f}%)".format(over, 100*over / max(len(errors),1)))
    print(" Under : {} ({:.0f}%)".format(under, 100*under / max(len(errors),1)))

    if po_comp:
        po_errors = [v["actual"] - v["proj"] for v in po_comp]
        po_bias = sum(po_errors) / len(po_errors)
        po_mae = sum(abs(e) for e in po_errors) / len(po_errors)
        print("\n --- Playoff Only ---")
        print(" Bias : {:+.2f} pts".format(po_bias))
        print(" MAE : {:.2f} pts".format(po_mae))

    buckets = {"0-3": 0, "3-6": 0, "6-10": 0, "10+": 0}
    for ae in abs_errors:
        if ae <= 3: buckets["0-3"] += 1
        elif ae <= 6: buckets["3-6"] += 1
        elif ae <= 10: buckets["6-10"] += 1
        else: buckets["10+"] += 1

    print("\n --- Error Distribution ---")
    for b, c in buckets.items():
        pct = 100 * c / max(len(abs_errors), 1)
        bar = "#" * int(pct / 5)
        print(" {:>5} pts: {:3d} ({:4.0f}%) {}".format(b, c, pct, bar))

    if ML_TRAINED and ML_MODEL is not None:
        print("\n --- ML Model Active ---")
        if ML_RAW_MAE: print(" Raw MAE : {:.2f} pts".format(ML_RAW_MAE))
        if ML_CV_MAE:
            delta = (ML_RAW_MAE or 0) - ML_CV_MAE
            print(" CV MAE : {:.2f} pts (delta {:+.2f})".format(ML_CV_MAE, delta))
        ranked = sorted(zip(FEATURE_NAMES, ML_MODEL.coef_), key=lambda x: abs(x[1]), reverse=True)
        print("\n Top features by influence:")
        for name, coef in ranked[:8]:
            print(" {:<22s} {:+.3f}".format(name, coef))
    else:
        needed = ML_MIN_TO_ACTIVATE - len(ML_SNAPSHOTS)
        print("\n --- ML Model Inactive ---")
        print(" Need {} more pregame snapshots.".format(max(0, needed)))
    print("==========================================\n")

# =====================================================
# PARALLEL PRELOAD (v5.1+)
# =====================================================
def _preload_stats():
    tasks = [
        ("season stats", _load_season_stats),
        ("recent stats", _load_recent_stats),
        ("location splits", _load_location_splits),
        ("clutch", _load_clutch),
        ("tracking PPP", _load_tracking),
        ("drives", _load_drives),
        ("player recent", _load_player_recent),
        ("hustle", _load_hustle),
        ("player availability", _load_player_availability),
    ]

    with ThreadPoolExecutor(max_workers=9) as executor:
        futures = {executor.submit(fn): name for name, fn in tasks}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print("[WARN] {} loader raised: {}".format(futures[future], e))

    print("[STARTUP] All stats loaded.")

# =====================================================
# CONSOLE STARTUP
# =====================================================
print("\n=== CRIMSON APEX v5.2 - Juno Console ===\n")
load_data()
_load_ml_snapshots()

# Fetch games first so playoff flag is set before availability loader runs
print("[STARTUP] Fetching today's games (playoff detection)...")
try:
    games = fetch_nba_games()
    print("[STARTUP] {} games found. Playoff={}.".format(len(games), _IS_PLAYOFF_SESSION))
except Exception as e:
    print("[WARN] Scoreboard fetch failed:", e)
    games = []

print("[STARTUP] Preloading stats (parallel, timeout 60s)...")
preload_thread = threading.Thread(target=_preload_stats, daemon=True)
preload_thread.start()
preload_thread.join(timeout=60)

if preload_thread.is_alive():
    print("[WARN] Preload still running after 60s - projecting with partial data")

try:
    if not games:
        games = fetch_nba_games()
    preload_game_logs(games)
    print("\nFound {} games today.\n".format(len(games)))

    if ML_AVAILABLE and not ML_TRAINED:
        print("[STARTUP] Bootstrapping ML from game logs...")
        _bootstrap_ml_from_logs()

    for g in games:
        calculate_projection(g)
        update_game_history([g])
        status = get_period_label(g)
        total = g["away_score"] + g["home_score"]
        proj = g.get("projection", "N/A")
        proj_raw = g.get("proj_raw")
        ml_delta = g.get("proj_ml_delta", 0.0)

        po_tag = " [G{}]".format(g["series_game_num"]) if g.get("is_playoff") and g.get("series_game_num") else ""
        print("{} @ {}{}".format(g["away"], g["home"], po_tag))
        print(" Status : {}".format(status))
        print(" Score : {}-{} (Total: {})".format(g["away_score"], g["home_score"], total))

        line = " Apex : {}".format(proj)
        if ML_TRAINED and ml_delta != 0.0 and g.get("status_id") == 1:
            line += " (MC: {} ML: {:+.1f})".format(proj_raw, ml_delta)
        print(line)

        if g.get("vegas_line") and g.get("status_id") == 1:
            print(" Vegas : {}".format(g["vegas_line"]))

        if g.get("p_over") is not None:
            conf_pct = max(g["p_over"], g["p_under"]) * 100
            side = "OVER" if g["p_over"] >= 0.5 else "UNDER"
            print(" O/U   : Line {:.1f} | P(over)={:.1%} P(under)={:.1%} [{} {:.0f}% conf]".format(
                g["ou_line"], g["p_over"], g["p_under"], side, conf_pct))

        # v5.2: guard against None split (LAL edge case)
        if g.get("proj_away") is not None and g.get("proj_home") is not None:
            print(" Split : {} {} | {} {}".format(
                g["away"], g["proj_away"], g["home"], g["proj_home"]))

        if g.get("status_id") == 1:
            loc_tag = "[loc-split FF]" if _location_split_available() else "[HCA fallback]"
            hustle_tag = "[hustle]" if _HUSTLE_LOADED else ""
            playoff_tag = "[PLAYOFF]" if g.get("is_playoff") else ""
            print(" Signals: {} {} {}".format(loc_tag, hustle_tag, playoff_tag).rstrip())

        away_avail = get_availability_factor(g["away_id"])
        home_avail = get_availability_factor(g["home_id"])
        if away_avail < 1.0 or home_avail < 1.0:
            print(" Avail : {} {:.0f}% | {} {:.0f}%".format(
                g["away"], away_avail * 100, g["home"], home_avail * 100))

        print(" Rest : {} {}d | {} {}d".format(
            g["away"], get_rest_days(g["away_id"], g["date"]),
            g["home"], get_rest_days(g["home_id"], g["date"])))
        print(" Streak : {} {:+d} | {} {:+d}".format(
            g["away"], get_win_streak(g["away_id"]),
            g["home"], get_win_streak(g["home_id"])))
        print("-" * 70)

except Exception as e:
    print("Error:", e)
    import traceback
    traceback.print_exc()

print("\nScript finished.")
print("Tip: run ml_diagnostics() to inspect model health.")
print("Tip: set ODDS_API_KEY env var to enable automated Vegas totals.")