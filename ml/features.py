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


def extract_features_from_api(api_data: dict) -> dict:
    """
    Approximate the 7 features from Spotify API data only (no history file).
    Less accurate than history-based features — shown with a warning in the UI.
    """
    features: dict = {}

    top_recent = api_data.get("top_tracks_recent", [])
    top_alltime = api_data.get("top_tracks_alltime", [])
    top_artists = api_data.get("top_artists", [])
    recent_played = api_data.get("recently_played", [])
    genre_counts = api_data.get("genre_counts", {})

    # 1. Skip rate trend — popularity drop between all-time and recent is a weak proxy
    pop_recent = np.mean([t["popularity"] for t in top_recent]) if top_recent else 50
    pop_alltime = np.mean([t["popularity"] for t in top_alltime]) if top_alltime else 50
    features["skip_rate_trend"] = float(max(0.0, (pop_alltime - pop_recent) / 200))

    # 2. Session frequency delta — valence drop as proxy
    avg_valence = api_data.get("avg_valence", 0.5)
    features["session_freq_delta"] = float((avg_valence - 0.5) * 2)

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

    # 5. Time-of-day shift — cannot compute without timestamps; neutral default
    features["time_of_day_shift"] = 2.0

    # 6. Days since new artist — neutral default
    features["days_new_artist"] = 3.0

    # 7. Repeat play ratio — overlap between recent and all-time
    recent_names = {t["name"] for t in recent_played[:20]}
    alltime_names = {t["name"] for t in top_alltime}
    overlap = len(recent_names & alltime_names)
    features["repeat_play_ratio"] = float(overlap / max(len(recent_names), 1))

    return features


def features_to_array(features: dict) -> np.ndarray:
    return np.array([[features.get(f, 0.0) for f in FEATURE_NAMES]], dtype=float)
