"""
analytics.py — Pure-API data science analytics engine.

Computes every metric possible from Spotify API data:
  - Temporal listening patterns (heatmap by hour/day)
  - Artist loyalty & taste trajectory (short vs long term drift)
  - Genre diversity & distribution
  - Listening velocity
  - Taste stability score
  - Discovery funnel metrics
  - Top rising / falling artists
"""

from datetime import datetime, timezone
from collections import Counter
import math


def _jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 0.0
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else 0.0


def _shannon_entropy(counts: dict) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = [c / total for c in counts.values() if c > 0]
    return -sum(p * math.log2(p) for p in probs)


def compute_temporal_heatmap(recently_played: list) -> dict:
    """
    Returns listening counts by hour-of-day (0-23) and day-of-week (0=Mon).
    """
    hour_counts = Counter()
    dow_counts = Counter()
    for track in recently_played:
        try:
            dt = datetime.fromisoformat(track["played_at"].replace("Z", "+00:00"))
            hour_counts[dt.hour] += 1
            dow_counts[dt.weekday()] += 1
        except Exception:
            pass

    # Convert to lists for charting (index = hour/day)
    hours = [hour_counts.get(h, 0) for h in range(24)]
    days = [dow_counts.get(d, 0) for d in range(7)]
    peak_hour = max(range(24), key=lambda h: hour_counts.get(h, 0)) if hour_counts else 12
    peak_day = max(range(7), key=lambda d: dow_counts.get(d, 0)) if dow_counts else 0
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    return {
        "hours": hours,
        "days": days,
        "peak_hour": peak_hour,
        "peak_hour_label": f"{peak_hour:02d}:00–{peak_hour:02d}:59",
        "peak_day": peak_day,
        "peak_day_label": day_names[peak_day],
        "total_plays": len(recently_played),
    }


def compute_listening_velocity(recently_played: list) -> dict:
    """
    Measures tracks-per-hour and ms-per-session from recent plays.
    """
    if not recently_played:
        return {"tracks_per_hour": 0.0, "total_hrs": 0.0, "avg_duration_min": 3.5}

    total_ms = sum(t.get("ms_played", 210_000) or 210_000 for t in recently_played)
    total_hrs = total_ms / 3_600_000.0

    # Time span of the 50 plays
    try:
        timestamps = sorted([
            datetime.fromisoformat(t["played_at"].replace("Z", "+00:00"))
            for t in recently_played
        ])
        span_hrs = max((timestamps[-1] - timestamps[0]).total_seconds() / 3600.0, 1.0)
        tracks_per_hour = len(recently_played) / span_hrs
    except Exception:
        tracks_per_hour = 0.0

    avg_duration_min = (total_ms / max(len(recently_played), 1)) / 60_000.0

    return {
        "tracks_per_hour": round(tracks_per_hour, 2),
        "total_hrs": round(total_hrs, 2),
        "avg_duration_min": round(avg_duration_min, 2),
        "total_tracks": len(recently_played),
    }


def compute_artist_loyalty(
    top_short: list, top_medium: list, top_long: list
) -> dict:
    """
    Measures how loyal a user is to their long-term artists vs exploring new ones.
    Loyalty = Jaccard similarity between short-term and long-term artist sets.
    Trajectory = Which artists are rising (in short but not long) vs falling
    """
    short_names = {a["name"] for a in top_short}
    medium_names = {a["name"] for a in top_medium}
    long_names = {a["name"] for a in top_long}

    loyalty_score = _jaccard(short_names, long_names)       # 0 = totally new taste, 1 = identical
    medium_loyalty = _jaccard(short_names, medium_names)

    rising = list(short_names - long_names)[:5]   # In your recent but not all-time
    falling = list(long_names - short_names)[:5]  # In your all-time but dropped off recently
    stable = list(short_names & long_names)[:5]   # Consistent all along

    explorer_score = 1.0 - loyalty_score  # Higher = discovering more

    if loyalty_score > 0.6:
        loyalty_label = "Deeply Loyal"
        loyalty_desc = "Your recent listening closely mirrors your all-time preferences. You know what you love and stick to it."
    elif loyalty_score > 0.35:
        loyalty_label = "Balanced Explorer"
        loyalty_desc = "You have a stable core of artists but are consistently bringing in new ones."
    else:
        loyalty_label = "Active Explorer"
        loyalty_desc = "Your recent listening barely resembles your all-time chart — you're actively discovering new music."

    return {
        "loyalty_score": round(loyalty_score, 3),
        "medium_loyalty": round(medium_loyalty, 3),
        "explorer_score": round(explorer_score, 3),
        "loyalty_label": loyalty_label,
        "loyalty_desc": loyalty_desc,
        "rising_artists": rising,
        "falling_artists": falling,
        "stable_artists": stable,
    }


def compute_genre_profile(genre_counts: dict, top_artists: list) -> dict:
    """
    Computes genre diversity metrics and distribution for visualization.
    """
    if not genre_counts:
        return {
            "diversity_score": 0.0,
            "dominant_genre": "unknown",
            "genre_labels": [],
            "genre_values": [],
            "total_unique": 0,
            "concentration": 1.0,
        }

    total = sum(genre_counts.values())
    diversity_score = round(_shannon_entropy(genre_counts), 3)
    dominant_genre = max(genre_counts, key=genre_counts.get)
    concentration = genre_counts[dominant_genre] / max(total, 1)

    # Top 8 genres for chart
    sorted_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:8]
    genre_labels = [g for g, _ in sorted_genres]
    genre_values = [c for _, c in sorted_genres]

    return {
        "diversity_score": diversity_score,
        "dominant_genre": dominant_genre,
        "concentration": round(concentration, 3),
        "genre_labels": genre_labels,
        "genre_values": genre_values,
        "total_unique": len(genre_counts),
        "top_genres_list": genre_labels[:5],
    }


def compute_taste_trajectory(
    top_short: list, top_medium: list, top_long: list,
    top_artists_short: list, top_artists_long: list,
) -> dict:
    """
    Determines if the user's taste is stable, shifting, or transforming.
    Looks at both artists AND tracks across time ranges.
    """
    short_track_ids = {t["id"] for t in top_short}
    medium_track_ids = {t["id"] for t in top_medium}
    long_track_ids = {t["id"] for t in top_long}

    # Similarity between consecutive windows
    short_to_medium = _jaccard(short_track_ids, medium_track_ids)
    medium_to_long = _jaccard(medium_track_ids, long_track_ids)
    short_to_long = _jaccard(short_track_ids, long_track_ids)

    # Popularity delta: are you listening to more mainstream or niche?
    short_pop = sum(t.get("popularity", 50) for t in top_short) / max(len(top_short), 1)
    long_pop = sum(t.get("popularity", 50) for t in top_long) / max(len(top_long), 1)
    pop_delta = short_pop - long_pop  # positive = going more mainstream

    # Classify trajectory
    if short_to_long > 0.5:
        trajectory = "Stable"
        trajectory_desc = "Your taste has remained very consistent over time. You have a clear, defined musical identity."
        trajectory_color = "#1db954"
    elif short_to_medium < 0.2:
        trajectory = "Rapidly Shifting"
        trajectory_desc = "Something changed recently — your short-term listening is very different from just a few months ago."
        trajectory_color = "#ff4b4b"
    else:
        trajectory = "Gradually Evolving"
        trajectory_desc = "Your taste is slowly but consistently exploring new territory. Healthy musical growth."
        trajectory_color = "#ffa500"

    # Mood shift (if we can compare popularities)
    if pop_delta > 10:
        pop_trend = "Moving Mainstream"
    elif pop_delta < -10:
        pop_trend = "Going Indie/Niche"
    else:
        pop_trend = "Consistent Taste"

    return {
        "short_to_medium_sim": round(short_to_medium, 3),
        "medium_to_long_sim": round(medium_to_long, 3),
        "short_to_long_sim": round(short_to_long, 3),
        "trajectory": trajectory,
        "trajectory_desc": trajectory_desc,
        "trajectory_color": trajectory_color,
        "avg_popularity_short": round(short_pop, 1),
        "avg_popularity_long": round(long_pop, 1),
        "popularity_trend": pop_trend,
        "pop_delta": round(pop_delta, 1),
    }


def compute_discovery_metrics(
    top_artists_short: list,
    top_artists_long: list,
    related_artists: list,
    recently_played: list,
) -> dict:
    """
    Builds a discovery funnel: how much of your short-term is genuinely new?
    Which related artists haven't you explored yet?
    """
    long_artist_names = {a["name"].lower() for a in top_artists_long}
    short_artist_names = {a["name"].lower() for a in top_artists_short}
    recent_artist_names = {t["artist"].lower() for t in recently_played}

    # New-to-you related artists (not in long-term history)
    undiscovered = [
        ra for ra in related_artists
        if ra["name"].lower() not in long_artist_names
        and ra["name"].lower() not in short_artist_names
    ]
    # Deduplicate by name
    seen = set()
    unique_undiscovered = []
    for ra in undiscovered:
        if ra["name"].lower() not in seen:
            seen.add(ra["name"].lower())
            unique_undiscovered.append(ra)

    # Discovery rate = fraction of recent plays NOT in long-term artists
    new_in_recent = len(recent_artist_names - long_artist_names)
    discovery_rate = new_in_recent / max(len(recent_artist_names), 1)

    return {
        "undiscovered_artists": unique_undiscovered[:8],
        "discovery_rate": round(discovery_rate, 3),
        "new_recent_artists": new_in_recent,
        "total_recent_artists": len(recent_artist_names),
    }


def compute_all(user_profile: dict) -> dict:
    """
    Master function — compute every analytics metric and return as one payload.
    """
    recently_played = user_profile.get("recently_played", [])
    top_short = user_profile.get("top_tracks_short", [])
    top_medium = user_profile.get("top_tracks_medium", [])
    top_long = user_profile.get("top_tracks_long", [])
    top_artists_short = user_profile.get("top_artists_short", [])
    top_artists_medium = user_profile.get("top_artists_medium", [])
    top_artists_long = user_profile.get("top_artists_long", [])
    genre_counts = user_profile.get("genre_counts", {})
    related_artists = user_profile.get("related_artists", [])

    return {
        "temporal": compute_temporal_heatmap(recently_played),
        "velocity": compute_listening_velocity(recently_played),
        "loyalty": compute_artist_loyalty(top_artists_short, top_artists_medium, top_artists_long),
        "genre_profile": compute_genre_profile(genre_counts, top_artists_medium),
        "trajectory": compute_taste_trajectory(top_short, top_medium, top_long, top_artists_short, top_artists_long),
        "discovery": compute_discovery_metrics(top_artists_short, top_artists_long, related_artists, recently_played),
    }
