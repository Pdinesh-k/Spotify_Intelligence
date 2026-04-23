import json
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy

FEATURE_NAMES = [
    "skip_rate_trend",
    "session_freq_delta",
    "listen_depth",
    "genre_entropy_drop",
    "time_of_day_shift",
    "days_new_artist",
    "repeat_play_ratio",
]

FEATURE_LABELS = {
    "skip_rate_trend": "Skip Rate Trend",
    "session_freq_delta": "Session Frequency Δ",
    "listen_depth": "Listen Depth",
    "genre_entropy_drop": "Genre Entropy Drop",
    "time_of_day_shift": "Time-of-Day Shift",
    "days_new_artist": "Days Since New Artist",
    "repeat_play_ratio": "Repeat Play Ratio",
}

FEATURE_DESCRIPTIONS = {
    "skip_rate_trend": "% more songs skipped vs prior week",
    "session_freq_delta": "Change in daily session count",
    "listen_depth": "Avg fraction of song actually heard",
    "genre_entropy_drop": "Narrowing of genre diversity",
    "time_of_day_shift": "Hours shifted in listening schedule",
    "days_new_artist": "Days since discovering a new artist",
    "repeat_play_ratio": "Fraction of songs played repeatedly",
}

AVG_TRACK_MS = 210_000  # ~3.5 min average track length


def load_history_files(file_objects) -> pd.DataFrame:
    """Parse uploaded Spotify Extended Streaming History JSON files into a DataFrame."""
    records = []
    for f in file_objects:
        try:
            data = json.load(f)
            if isinstance(data, list):
                records.extend(data)
        except Exception:
            pass

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    if "ts" not in df.columns:
        return pd.DataFrame()

    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

    if "ms_played" not in df.columns:
        df["ms_played"] = 0
    df["ms_played"] = pd.to_numeric(df["ms_played"], errors="coerce").fillna(0)

    if "skipped" not in df.columns:
        df["skipped"] = False
    df["skipped"] = df["skipped"].fillna(False).astype(bool)

    return df


def _skip_rate(window: pd.DataFrame) -> float:
    if len(window) == 0:
        return 0.0
    explicitly_skipped = window["skipped"].sum() if "skipped" in window.columns else 0
    short_plays = (window["ms_played"] < 30_000).sum()
    return float((explicitly_skipped + short_plays) / (2 * max(len(window), 1)))


def _count_sessions(window: pd.DataFrame) -> int:
    if len(window) == 0:
        return 0
    times = window["ts"].sort_values()
    gaps = times.diff().dt.total_seconds().fillna(9999)
    return int((gaps > 1800).sum()) + 1


def extract_features_from_history(df: pd.DataFrame) -> dict:
    """Derive the 7 features from a full Extended Streaming History DataFrame."""
    if df.empty:
        return {}

    now = df["ts"].max()
    week_ago = now - pd.Timedelta(days=7)
    two_weeks_ago = now - pd.Timedelta(days=14)

    last_week = df[df["ts"] >= week_ago]
    prev_week = df[(df["ts"] >= two_weeks_ago) & (df["ts"] < week_ago)]
    all_before_last = df[df["ts"] < week_ago]

    artist_col = "master_metadata_album_artist_name"
    track_col = "master_metadata_track_name"

    features: dict = {}

    # 1. Skip rate trend
    features["skip_rate_trend"] = float(_skip_rate(last_week) - _skip_rate(prev_week))

    # 2. Session frequency delta (sessions per day)
    sess_now = _count_sessions(last_week) / 7.0
    sess_prev = _count_sessions(prev_week) / 7.0
    features["session_freq_delta"] = float(sess_now - sess_prev)

    # 3. Listen depth
    if len(last_week) > 0:
        depth = (last_week["ms_played"] / AVG_TRACK_MS).clip(upper=1.0)
        features["listen_depth"] = float(depth.mean())
    else:
        features["listen_depth"] = 0.5

    # 4. Genre entropy drop (using artist diversity as proxy)
    def _artist_entropy(window: pd.DataFrame) -> float:
        if artist_col not in window.columns or len(window) == 0:
            return 0.0
        counts = window[artist_col].dropna().value_counts(normalize=True).values
        return float(scipy_entropy(counts)) if len(counts) > 0 else 0.0

    entropy_now = _artist_entropy(last_week)
    entropy_prev = _artist_entropy(prev_week)
    features["genre_entropy_drop"] = float(max(0.0, entropy_prev - entropy_now))

    # 5. Time-of-day shift
    if len(last_week) > 0:
        hour_now = last_week["ts"].dt.hour.mean()
        hour_prev = prev_week["ts"].dt.hour.mean() if len(prev_week) > 0 else hour_now
        features["time_of_day_shift"] = float(abs(hour_now - hour_prev))
    else:
        features["time_of_day_shift"] = 0.0

    # 6. Days since new artist
    if artist_col in df.columns:
        past_artists = set(all_before_last[artist_col].dropna())
        new_in_last_week = set(last_week[artist_col].dropna()) - past_artists
        features["days_new_artist"] = 0.0 if new_in_last_week else 7.0
    else:
        features["days_new_artist"] = 7.0

    # 7. Repeat play ratio
    if track_col in df.columns and len(last_week) > 0:
        counts = last_week[track_col].dropna().value_counts()
        repeats = int((counts > 1).sum())
        features["repeat_play_ratio"] = float(repeats / max(len(counts), 1))
    else:
        features["repeat_play_ratio"] = 0.0

    return features


def _parse_played_at(ts_str: str):
    """Parse Spotify's played_at ISO string to a timezone-aware datetime."""
    from datetime import datetime, timezone
    try:
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except Exception:
        return None


def extract_features_from_api(api_data: dict) -> dict:
    """
    Approximate the 7 features from Spotify API data only (no history file).
    Less accurate than history-based features — shown with a warning in the UI.
    """
    from datetime import datetime, timezone

    features: dict = {}

    top_recent = api_data.get("top_tracks_recent", [])
    top_alltime = api_data.get("top_tracks_alltime", [])
    recent_played = api_data.get("recently_played", [])
    genre_counts = api_data.get("genre_counts", {})

    # Parse timestamps from recently_played (Spotify returns newest-first)
    played_times: list = []
    for t in recent_played:
        dt = _parse_played_at(t.get("played_at", ""))
        if dt:
            played_times.append(dt)

    # 1. Skip rate trend — popularity drop between all-time and recent is a weak proxy
    pop_recent = np.mean([t["popularity"] for t in top_recent]) if top_recent else 50
    pop_alltime = np.mean([t["popularity"] for t in top_alltime]) if top_alltime else 50
    features["skip_rate_trend"] = float(max(0.0, (pop_alltime - pop_recent) / 200))

    # 2. Session frequency delta — sessions per day in first vs second half of recent plays
    if len(played_times) >= 6:
        sorted_times = sorted(played_times)  # oldest first
        mid = len(sorted_times) // 2
        older_half = sorted_times[:mid]
        recent_half = sorted_times[mid:]

        def _count_api_sessions(ts_list: list) -> int:
            if not ts_list:
                return 0
            sessions = 1
            for i in range(1, len(ts_list)):
                gap = (ts_list[i] - ts_list[i - 1]).total_seconds()
                if gap > 1800:
                    sessions += 1
            return sessions

        span_older = max((older_half[-1] - older_half[0]).total_seconds() / 86400, 0.5)
        span_recent = max((recent_half[-1] - recent_half[0]).total_seconds() / 86400, 0.5)
        sess_older = _count_api_sessions(older_half) / span_older
        sess_recent = _count_api_sessions(recent_half) / span_recent
        features["session_freq_delta"] = float(sess_recent - sess_older)
    else:
        # No timestamp data — negative delta (sessions falling) is a better assumption
        # than 0.0 when we have no listening history at all
        features["session_freq_delta"] = -0.5 if not recent_played else 0.0

    # 3. Listen depth — energy as proxy (engaged listeners tend toward higher energy)
    features["listen_depth"] = float(api_data.get("avg_energy", 0.6))

    # 4. Genre entropy drop — concentration of top genres
    if genre_counts:
        vals = np.array(list(genre_counts.values()), dtype=float)
        vals = vals / vals.sum()
        max_entropy = np.log(max(len(vals), 2))
        features["genre_entropy_drop"] = float(1.0 - scipy_entropy(vals) / max_entropy)
    else:
        features["genre_entropy_drop"] = 0.4

    # 5. Time-of-day shift — hour drift between first and second half of recent plays
    if len(played_times) >= 6:
        sorted_times = sorted(played_times)
        mid = len(sorted_times) // 2
        older_hours = [t.hour for t in sorted_times[:mid]]
        recent_hours = [t.hour for t in sorted_times[mid:]]
        shift = abs(sum(recent_hours) / len(recent_hours) - sum(older_hours) / len(older_hours))
        features["time_of_day_shift"] = float(shift)
    else:
        # No timestamp data — use neutral value (4.0 = midpoint of engaged/churned range)
        features["time_of_day_shift"] = 4.0

    # 6. Days since new artist — compare recently_played artists vs long-term top artists
    long_term_artist_names = {a["name"] for a in api_data.get("top_artists_long", [])}
    short_term_artist_names = {a["name"] for a in api_data.get("top_artists_short", [])}

    if long_term_artist_names and recent_played and played_times:
        # Primary: use exact timestamps from recently_played
        now = datetime.now(timezone.utc)
        days_list = []
        for t, dt in zip(recent_played, played_times):
            artist = t.get("artist", "")
            if artist and artist not in long_term_artist_names:
                days_ago = (now - dt).total_seconds() / 86400.0
                days_list.append(days_ago)
        if days_list:
            features["days_new_artist"] = float(min(days_list))
        else:
            features["days_new_artist"] = 7.0  # no new artists in recent plays
    elif long_term_artist_names and short_term_artist_names:
        # Fallback: compare short-term vs long-term top artists (no timestamps)
        new_in_short = short_term_artist_names - long_term_artist_names
        if new_in_short:
            features["days_new_artist"] = 2.0  # new artists in short-term top = recent discovery
        else:
            features["days_new_artist"] = 7.0  # same artists as always
    else:
        features["days_new_artist"] = 7.0  # not enough data — assume no new discovery

    # 7. Repeat play ratio — overlap between recent and all-time
    # When recently_played is empty we cannot compute this — use neutral 0.5
    # (0.0 falsely signals "engaged" since churned users are trained at ~0.55)
    if recent_played:
        recent_names = {t["name"] for t in recent_played[:20]}
        alltime_names = {t["name"] for t in top_alltime}
        overlap = len(recent_names & alltime_names)
        features["repeat_play_ratio"] = float(overlap / max(len(recent_names), 1))
    else:
        features["repeat_play_ratio"] = 0.5  # neutral — unknown, not falsely engaged

    return features


def features_to_array(features: dict) -> np.ndarray:
    return np.array([[features.get(f, 0.0) for f in FEATURE_NAMES]], dtype=float)
