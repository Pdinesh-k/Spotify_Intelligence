import json
import os
import re
from datetime import datetime, timezone, timedelta

from config import FEEDBACK_PATH

AUTO_RESOLVE_HOURS = 48


def _user_path(user_id: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", user_id)[:64]
    base = os.path.dirname(FEEDBACK_PATH)
    return os.path.join(base, f"feedback_{safe}.json")


class FeedbackStore:
    def __init__(self, user_id: str = "global"):
        self._path = _user_path(user_id)
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        self._data = self._load()

    def _load(self) -> dict:
        if os.path.exists(self._path):
            try:
                with open(self._path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return {"interactions": [], "track_scores": {}, "pending_recommendations": []}

    def _save(self):
        with open(self._path, "w") as f:
            json.dump(self._data, f, indent=2)

    def log_interaction(self, track_id, track_name, artist, outcome, churn_prob, auto=False):
        self._data["interactions"].append({
            "track_id": track_id,
            "track_name": track_name,
            "artist": artist,
            "outcome": outcome,
            "churn_prob": round(churn_prob, 3),
            "auto_detected": auto,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        current = self._data["track_scores"].get(track_id, 0.0)
        delta = 1.0 if outcome == "listened" else -0.5
        self._data["track_scores"][track_id] = round(current + delta, 2)
        self._data["pending_recommendations"] = [
            r for r in self._data.get("pending_recommendations", [])
            if r["track_id"] != track_id
        ]
        self._save()

    def store_pending(self, track_id, track_name, artist, churn_prob):
        existing = {r["track_id"] for r in self._data.get("pending_recommendations", [])}
        if track_id in existing:
            return
        pending = self._data.setdefault("pending_recommendations", [])
        pending.append({
            "track_id": track_id,
            "track_name": track_name,
            "artist": artist,
            "churn_prob": round(churn_prob, 3),
            "recommended_at": datetime.now(timezone.utc).isoformat(),
        })
        self._save()

    def get_pending(self) -> list[dict]:
        return self._data.get("pending_recommendations", [])

    def expire_old_pending(self):
        cutoff = datetime.now(timezone.utc) - timedelta(hours=AUTO_RESOLVE_HOURS)
        expired, still_pending = [], []
        for rec in self._data.get("pending_recommendations", []):
            rec_time = datetime.fromisoformat(rec["recommended_at"])
            if rec_time.tzinfo is None:
                rec_time = rec_time.replace(tzinfo=timezone.utc)
            if rec_time < cutoff:
                expired.append(rec)
            else:
                still_pending.append(rec)
        for rec in expired:
            self.log_interaction(
                rec["track_id"], rec["track_name"], rec["artist"],
                "skipped", rec["churn_prob"], auto=True,
            )
        self._data["pending_recommendations"] = still_pending
        if expired:
            self._save()
        return expired

    def get_track_scores(self) -> dict[str, float]:
        return self._data.get("track_scores", {})

    def get_interactions(self) -> list[dict]:
        return self._data.get("interactions", [])

    def get_stats(self) -> dict:
        interactions = self.get_interactions()
        if not interactions:
            return {"total": 0, "listened": 0, "skipped": 0, "success_rate": 0.0,
                    "auto_detected": 0, "trend": []}
        listened = [i for i in interactions if i["outcome"] == "listened"]
        skipped  = [i for i in interactions if i["outcome"] == "skipped"]
        auto     = [i for i in interactions if i.get("auto_detected")]
        trend = []
        for i in range(len(interactions)):
            window = interactions[max(0, i - 4): i + 1]
            rate = sum(1 for x in window if x["outcome"] == "listened") / len(window)
            trend.append({"index": i + 1, "success_rate": round(rate, 3)})
        return {
            "total": len(interactions),
            "listened": len(listened),
            "skipped": len(skipped),
            "success_rate": round(len(listened) / len(interactions), 3),
            "auto_detected": len(auto),
            "trend": trend,
        }

    def clear(self):
        self._data = {"interactions": [], "track_scores": {}, "pending_recommendations": []}
        self._save()
