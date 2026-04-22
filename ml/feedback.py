import json
import os
from datetime import datetime, timezone, timedelta

from config import FEEDBACK_PATH

AUTO_RESOLVE_HOURS = 48


class FeedbackStore:
    def __init__(self):
        os.makedirs(os.path.dirname(FEEDBACK_PATH), exist_ok=True)
        self._data = self._load()

    def _load(self) -> dict:
        if os.path.exists(FEEDBACK_PATH):
            try:
                with open(FEEDBACK_PATH) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return {"interactions": [], "track_scores": {}, "pending_recommendations": []}

    def _save(self):
        with open(FEEDBACK_PATH, "w") as f:
            json.dump(self._data, f, indent=2)

    # ── Logging interactions ──────────────────────────────────────────────────

    def log_interaction(
        self,
        track_id: str,
        track_name: str,
        artist: str,
        outcome: str,
        churn_prob: float,
        auto: bool = False,
    ):
        """
        outcome: 'listened' (+1.0) or 'skipped' (−0.5).
        Asymmetric: success signal is stronger than failure (mirrors production systems).
        """
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

        # Mark any matching pending recommendation as resolved
        self._data["pending_recommendations"] = [
            r for r in self._data.get("pending_recommendations", [])
            if r["track_id"] != track_id
        ]

        self._save()

    # ── Pending recommendations ───────────────────────────────────────────────

    def store_pending(self, track_id: str, track_name: str, artist: str, churn_prob: float):
        """Store a recommendation so auto-feedback can resolve it later."""
        # Don't duplicate if already pending
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
        """Auto-log very old unresolved recommendations as 'skipped' (gave up waiting)."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=AUTO_RESOLVE_HOURS)
        expired = []
        still_pending = []
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

    # ── Stats ─────────────────────────────────────────────────────────────────

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
        skipped = [i for i in interactions if i["outcome"] == "skipped"]
        auto = [i for i in interactions if i.get("auto_detected")]

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
