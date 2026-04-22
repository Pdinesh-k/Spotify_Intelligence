from datetime import datetime, timezone

import spotipy

from ml.feedback import FeedbackStore


def _fetch_recent_plays(access_token: str, after_ms: int) -> list[dict]:
    sp = spotipy.Spotify(auth=access_token)
    try:
        result = sp.current_user_recently_played(limit=50, after=after_ms)
        return [
            {
                "id": item["track"]["id"],
                "name": item["track"]["name"],
                "artist": (
                    item["track"]["artists"][0]["name"] if item["track"]["artists"] else ""
                ),
                "played_at": item["played_at"],
            }
            for item in result.get("items", [])
        ]
    except Exception:
        return []


def _get_user_id(access_token: str) -> str:
    try:
        sp = spotipy.Spotify(auth=access_token)
        return sp.current_user().get("id", "global")
    except Exception:
        return "global"


def run_auto_feedback(access_token: str) -> list[dict]:
    user_id = _get_user_id(access_token)
    store = FeedbackStore(user_id)

    expired = store.expire_old_pending()

    pending = store.get_pending()
    if not pending:
        return []

    oldest_ts = min(
        datetime.fromisoformat(
            r["recommended_at"].replace("Z", "+00:00")
            if r["recommended_at"].endswith("Z")
            else r["recommended_at"]
        )
        for r in pending
    )
    if oldest_ts.tzinfo is None:
        oldest_ts = oldest_ts.replace(tzinfo=timezone.utc)

    after_ms = int(oldest_ts.timestamp() * 1000)
    recent_plays = _fetch_recent_plays(access_token, after_ms)
    recent_ids = {p["id"] for p in recent_plays}

    auto_outcomes = []
    for rec in list(pending):
        if rec["track_id"] in recent_ids:
            store.log_interaction(
                rec["track_id"], rec["track_name"], rec["artist"],
                "listened", rec["churn_prob"], auto=True,
            )
            auto_outcomes.append({
                "track_name": rec["track_name"],
                "artist": rec["artist"],
                "outcome": "listened",
            })

    return auto_outcomes
