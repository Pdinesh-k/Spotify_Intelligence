from fastapi import APIRouter, Query, HTTPException

from ml.features import extract_features_from_api
from ml.model import predict
from spotify.collector import collect_user_data

router = APIRouter()


@router.get("/raw")
async def debug_raw(token: str = Query(...)):
    """
    Returns the raw Spotify API payload + computed features + model result.
    Use this to diagnose why churn probabilities are wrong for a user.
    """
    try:
        data = collect_user_data(token)
    except Exception as e:
        raise HTTPException(401, f"Token error: {e}")

    features = extract_features_from_api(data)
    model_result = predict(features)

    recently_played = data.get("recently_played", [])
    top_artists_long = data.get("top_artists_long", [])

    return {
        "user_id": data.get("user_id"),
        "display_name": data.get("display_name"),
        "product": data.get("product"),
        "counts": {
            "recently_played": len(recently_played),
            "top_tracks_short": len(data.get("top_tracks_short", [])),
            "top_tracks_medium": len(data.get("top_tracks_medium", [])),
            "top_tracks_long": len(data.get("top_tracks_long", [])),
            "top_artists_short": len(data.get("top_artists_short", [])),
            "top_artists_medium": len(data.get("top_artists_medium", [])),
            "top_artists_long": len(top_artists_long),
            "genre_counts": len(data.get("genre_counts", {})),
            "track_audio_features": len(data.get("track_audio_features", {})),
        },
        "audio_features": {
            "available": data.get("audio_features_available", False),
            "avg_energy": data.get("avg_energy"),
            "avg_valence": data.get("avg_valence"),
            "avg_danceability": data.get("avg_danceability"),
            "avg_tempo": data.get("avg_tempo"),
            "avg_acousticness": data.get("avg_acousticness"),
        },
        "top_genres": data.get("top_genres", [])[:10],
        "genre_counts": dict(list(data.get("genre_counts", {}).items())[:20]),
        "recently_played_sample": recently_played[:5],
        "top_artists_long_sample": [a["name"] for a in top_artists_long[:10]],
        "computed_features": features,
        "model_result": model_result,
    }
