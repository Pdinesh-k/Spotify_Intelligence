import io
import json
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from agents.diagnosis import generate_diagnosis
from agents.recommender import get_recommendations
from ml.feedback import FeedbackStore
from ml.features import (
    FEATURE_NAMES,
    extract_features_from_api,
    extract_features_from_history,
)
from ml.model import predict
from spotify.collector import collect_user_data

router = APIRouter()

AVG_TRACK_MS = 210_000


def _parse_uploaded_files(raw_files: list[UploadFile]) -> pd.DataFrame:
    records = []
    for f in raw_files:
        try:
            content = f.file.read()
            data = json.loads(content.decode("utf-8"))
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
    df["ms_played"] = pd.to_numeric(df.get("ms_played", 0), errors="coerce").fillna(0)
    if "skipped" not in df.columns:
        df["skipped"] = False
    df["skipped"] = df["skipped"].fillna(False).astype(bool)
    return df


@router.post("/analyze")
async def analyze(
    token: Annotated[str, Form()],
    files: Annotated[list[UploadFile], File()] = [],
):
    try:
        user_profile = collect_user_data(token)
    except Exception as e:
        raise HTTPException(401, f"Spotify token error: {e}")

    history_mode = False
    features: dict = {}

    if files:
        df = _parse_uploaded_files(files)
        if not df.empty:
            features = extract_features_from_history(df)
            history_mode = bool(features)

    if not features:
        features = extract_features_from_api(user_profile)

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
        "history_mode": history_mode,
    }
