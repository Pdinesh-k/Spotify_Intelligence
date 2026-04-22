from fastapi import APIRouter, Query
from fastapi.responses import RedirectResponse, HTMLResponse
from spotify.auth import get_oauth_url, exchange_code
from config import SPOTIFY_CLIENT_ID

router = APIRouter()

FRONTEND_URL = "/"


@router.get("/url")
async def get_auth_url():
    if not SPOTIFY_CLIENT_ID:
        return {"error": "SPOTIFY_CLIENT_ID not configured", "url": None}
    return {"url": get_oauth_url("intelligence")}


@router.get("/callback")
async def spotify_callback(
    code: str = Query(default=""),
    error: str = Query(default=""),
):
    if error:
        return HTMLResponse(f"<script>window.location='/?auth_error={error}'</script>")
    try:
        token_info = exchange_code(code)
        token = token_info["access_token"]
        return RedirectResponse(url=f"/?token={token}")
    except Exception as e:
        return HTMLResponse(f"<script>window.location='/?auth_error={str(e)}'</script>")
