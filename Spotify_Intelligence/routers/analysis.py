from typing import Annotated

from fastapi import APIRouter, Form, HTTPException

from agents.diagnosis import generate_diagnosis
from agents.recommender import get_recommendations
from ml.feedback import FeedbackStore
from ml.features import extract_features_from_api
from ml.model import predict
from ml import analytics
from spotify.collector import collect_user_data

router = APIRouter()

AVG_TRACK_MS = 210_000





@router.post("/analyze")
async def analyze(token: Annotated[str, Form()]):
    try:
        user_profile = collect_user_data(token)
    except Exception as e:
        raise HTTPException(401, f"Spotify token error: {e}")

    features = extract_features_from_api(user_profile)

    # Calculate explicit API-based listening stats
    recent_played = user_profile.get("recently_played", [])
    recent_ms = sum(t.get("ms_played", 0) for t in recent_played)
    recent_hours = recent_ms / 3600000.0

    top_tracks_recent = user_profile.get("top_tracks_recent", [])
    top_tracks_alltime = user_profile.get("top_tracks_alltime", [])
    overlap = len({t["name"] for t in top_tracks_recent} & {t["name"] for t in top_tracks_alltime})
    obsession_rate = overlap / max(len(top_tracks_recent), 1)

    listening_stats = {
        "recent_hours": round(recent_hours, 2),
        "obsession_rate": round(obsession_rate, 2),
        "total_recent_tracks": len(recent_played)
    }
    
    # Inject into user_profile so agents can access it
    user_profile["listening_stats"] = listening_stats

    # Full data science analytics
    analytics_data = analytics.compute_all(user_profile)

    try:
        model_result = predict(features)
    except Exception as e:
        raise HTTPException(500, f"Model inference failed: {e}")

    try:
        diagnosis, agent_chain = generate_diagnosis(model_result, user_profile)
    except Exception as e:
        from agents.diagnosis import _fallback
        diagnosis = _fallback(model_result, user_profile, e)
        agent_chain = []

    try:
        recommendations = get_recommendations(diagnosis, user_profile, token)
    except Exception:
        recommendations = []

    # Store recommendations as pending for the auto-feedback loop
    store = FeedbackStore()
    prob = model_result["churn_probability"]
    for rec in recommendations:
        store.store_pending(rec["id"], rec["name"], rec["artist"], prob)

    return {
        "user_profile": {
            "display_name": user_profile.get("display_name", ""),
            "followers": user_profile.get("followers", 0),
            "country": user_profile.get("country", ""),
            "top_genres": user_profile.get("top_genres", []),
            "top_artists": user_profile.get("top_artists", []),
            "avg_energy": user_profile.get("avg_energy"),
            "avg_valence": user_profile.get("avg_valence"),
            "avg_danceability": user_profile.get("avg_danceability"),
        },
        "features": features,
        "model_result": model_result,
        "diagnosis": diagnosis,
        "agent_chain": agent_chain,
        "recommendations": recommendations,
        "listening_stats": listening_stats,
        "analytics": analytics_data,
    }
