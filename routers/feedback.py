from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from ml.feedback import FeedbackStore
from agents.auto_feedback import run_auto_feedback

router = APIRouter()


class FeedbackRequest(BaseModel):
    track_id: str
    track_name: str
    artist: str
    outcome: str          # 'listened' | 'skipped'
    churn_prob: float


@router.post("/log")
async def log_feedback(req: FeedbackRequest):
    if req.outcome not in ("listened", "skipped"):
        raise HTTPException(400, "outcome must be 'listened' or 'skipped'")
    store = FeedbackStore()
    store.log_interaction(
        req.track_id, req.track_name, req.artist,
        req.outcome, req.churn_prob,
    )
    return {"ok": True, "stats": store.get_stats()}


@router.get("/stats")
async def get_stats():
    return FeedbackStore().get_stats()


@router.get("/auto")
async def auto_feedback(token: str = Query(...)):
    try:
        outcomes = run_auto_feedback(token)
        return {"outcomes": outcomes}
    except Exception as e:
        return {"outcomes": [], "error": str(e)}
