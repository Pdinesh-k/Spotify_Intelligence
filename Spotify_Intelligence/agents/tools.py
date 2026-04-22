import json


# ── Tool 1 ────────────────────────────────────────────────────────────────────

def analyze_genre_entropy(features: dict, user_profile: dict, focus_period: str = "recent_week") -> dict:
    """
    Detect musical tunnel vision: has the user's genre diversity collapsed?
    """
    entropy_drop = features.get("genre_entropy_drop", 0.0)
    top_genres = user_profile.get("top_genres", [])[:8]
    genre_counts = user_profile.get("genre_counts", {})

    total_plays = sum(genre_counts.values()) if genre_counts else 0
    if genre_counts and total_plays > 0:
        top_genre = max(genre_counts, key=genre_counts.get)
        top_concentration = genre_counts[top_genre] / total_plays
        unique_active = len([g for g, c in genre_counts.items() if c >= 2])
    else:
        top_genre = top_genres[0] if top_genres else "unknown"
        top_concentration = 0.5
        unique_active = len(top_genres)

    if entropy_drop > 0.40:
        severity = "severe"
        interpretation = (
            f"Genre diversity has collapsed. {top_concentration:.0%} of plays are "
            f"concentrated in '{top_genre}' — the user is locked in a listening rut."
        )
        hint = "reintroduce adjacent genres very gradually — one session at a time"
    elif entropy_drop > 0.20:
        severity = "moderate"
        interpretation = (
            f"Noticeable genre narrowing. User is retreating toward '{top_genre}'. "
            f"Only {unique_active} genres still active."
        )
        hint = "introduce one new genre per session, anchored to a familiar artist"
    else:
        severity = "mild"
        interpretation = (
            f"Genre diversity is relatively healthy with {unique_active} distinct genres active."
        )
        hint = "user is open to new genres — can be bolder with recommendations"

    return {
        "focus_period": focus_period,
        "entropy_drop": round(entropy_drop, 3),
        "severity": severity,
        "dominant_genre": top_genre,
        "dominant_concentration": round(top_concentration, 3) if genre_counts else None,
        "unique_genres_active": unique_active,
        "top_genres": top_genres[:5],
        "interpretation": interpretation,
        "recommendation_hint": hint,
    }


# ── Tool 2 ────────────────────────────────────────────────────────────────────

def analyze_mood_trajectory(user_profile: dict, features: dict, depth: str = "full") -> dict:
    """
    Map the user's current emotional state from audio feature trends.
    """
    valence = user_profile.get("avg_valence")
    energy = user_profile.get("avg_energy")
    danceability = user_profile.get("avg_danceability")
    listen_depth = features.get("listen_depth", 0.5)
    time_shift = features.get("time_of_day_shift", 0.0)

    if valence is not None and energy is not None:
        if valence >= 0.5 and energy >= 0.5:
            quadrant = "energetic_positive"
            label = "Energetic & Positive"
            mood_desc = "High energy and positive valence — user is engaged and in a good state."
            rec_hint = "user is primed for discovery; bold recommendations will land"
        elif valence >= 0.5 and energy < 0.5:
            quadrant = "calm_content"
            label = "Calm & Content"
            mood_desc = "Low energy but positive — user is unwinding. Music as comfort, not discovery."
            rec_hint = "gentle, familiar-adjacent tracks; avoid anything jarring"
        elif valence < 0.5 and energy >= 0.5:
            quadrant = "tense_stressed"
            label = "Tense or Stressed"
            mood_desc = "High energy + low valence = stress or frustration. Music as an emotional outlet."
            rec_hint = "match energy first, then gradually shift valence upward across sessions"
        else:
            quadrant = "withdrawn_low"
            label = "Low Energy & Withdrawn"
            mood_desc = (
                "Low valence + low energy is the strongest disengagement signal in the audio profile. "
                "User may be going through a difficult personal period."
            )
            rec_hint = "comfort music first — familiar artists, low tempo, high acoustic"
    else:
        quadrant = "unknown"
        label = "Unknown (audio features unavailable for this Spotify app)"
        mood_desc = "Audio features API not accessible — using behavioural signals only."
        rec_hint = "rely on skip rate and genre entropy signals for strategy"

    depth_label = "shallow" if listen_depth < 0.45 else "deep" if listen_depth > 0.65 else "moderate"
    schedule_shifted = time_shift > 3.0

    return {
        "analysis_depth": depth,
        "mood_quadrant": quadrant,
        "mood_label": label,
        "avg_valence": round(valence, 3) if valence is not None else None,
        "avg_energy": round(energy, 3) if energy is not None else None,
        "avg_danceability": round(danceability, 3) if danceability is not None else None,
        "listen_depth_value": round(listen_depth, 3),
        "listen_depth_label": depth_label,
        "listening_schedule_shifted": schedule_shifted,
        "time_shift_hours": round(time_shift, 1),
        "interpretation": mood_desc,
        "engagement_note": (
            f"Listen depth is {depth_label} ({listen_depth:.0%}) — "
            + (
                "songs are being abandoned early; strong disengagement signal."
                if depth_label == "shallow"
                else "songs are being heard through; healthy engagement."
                if depth_label == "deep"
                else "average completion — monitor for further decline."
            )
        ),
        "schedule_note": (
            f"Listening schedule has shifted by {time_shift:.1f}h — "
            "possible lifestyle change (new job, travel, relationship)."
            if schedule_shifted
            else "Listening schedule is consistent."
        ),
        "recommendation_hint": rec_hint,
    }


# ── Tool 3 ────────────────────────────────────────────────────────────────────

def evaluate_discovery_health(features: dict, user_profile: dict) -> dict:
    """
    Evaluate new-artist discovery rate and repeat-play behaviour.
    """
    days_new = features.get("days_new_artist", 7.0)
    repeat_ratio = features.get("repeat_play_ratio", 0.5)
    skip_trend = features.get("skip_rate_trend", 0.0)
    session_delta = features.get("session_freq_delta", 0.0)
    top_artists = [a["name"] for a in user_profile.get("top_artists", [])[:5]]

    # Composite discovery health score [0..1]
    score = 1.0
    score -= min(days_new / 7.0, 1.0) * 0.40
    score -= min(repeat_ratio, 1.0) * 0.30
    score -= max(skip_trend, 0.0) * 0.30
    score = round(max(0.0, min(1.0, score)), 3)

    health = (
        "critical" if score < 0.30
        else "poor" if score < 0.50
        else "fair" if score < 0.70
        else "good"
    )

    if skip_trend > 0.15:
        skip_note = f"Skip rate up sharply (+{skip_trend:.0%}) — the catalogue is actively boring the user."
    elif skip_trend > 0.05:
        skip_note = f"Mild skip rate increase (+{skip_trend:.0%}) — early warning."
    elif skip_trend < -0.05:
        skip_note = f"Skip rate improving ({skip_trend:.0%}) — content is resonating."
    else:
        skip_note = "Skip rate stable."

    sessions_trend = (
        "declining" if session_delta < -0.5
        else "stable" if abs(session_delta) <= 0.5
        else "growing"
    )

    return {
        "discovery_score": score,
        "health_label": health,
        "days_since_new_artist": days_new,
        "repeat_play_ratio": round(repeat_ratio, 3),
        "session_frequency_trend": sessions_trend,
        "session_delta_per_day": round(session_delta, 3),
        "skip_trend": round(skip_trend, 3),
        "anchor_artists": top_artists,
        "interpretation": (
            f"Discovery health is {health}. "
            + (
                "No new artists discovered in the past week — user is stuck in a listening rut."
                if days_new >= 7
                else f"Last new artist discovered {days_new:.0f} day(s) ago."
            )
            + f" Sessions are {sessions_trend}. {skip_note}"
        ),
        "recommendation_hint": (
            "reintroduce discovery via adjacent artists — stay close to existing favourites"
            if health in ("critical", "poor")
            else "one new artist alongside a familiar anchor will land well"
            if health == "fair"
            else "user is open to genuine discovery — be ambitious"
        ),
    }


# ── Dispatch ──────────────────────────────────────────────────────────────────

TOOL_REGISTRY = {
    "analyze_genre_entropy": analyze_genre_entropy,
    "analyze_mood_trajectory": analyze_mood_trajectory,
    "evaluate_discovery_health": evaluate_discovery_health,
}


def execute_tool(name: str, args: dict, features: dict, user_profile: dict) -> dict:
    """Dispatch a tool call by name. LLM args are passed through but data comes from Python."""
    if name == "analyze_genre_entropy":
        return analyze_genre_entropy(features, user_profile, args.get("focus_period", "recent_week"))
    if name == "analyze_mood_trajectory":
        return analyze_mood_trajectory(user_profile, features, args.get("depth", "full"))
    if name == "evaluate_discovery_health":
        return evaluate_discovery_health(features, user_profile)
    return {"error": f"Unknown tool: {name}"}
