from fastapi import FastAPI, Query
from fastapi.responses import RedirectResponse, HTMLResponse
from spotify.auth import exchange_code

app = FastAPI()

STREAMLIT_URL = "http://localhost:8501"


@app.get("/api/spotify/callback")
async def spotify_callback(
    code: str = Query(default=""),
    error: str = Query(default=""),
):
    if error:
        return HTMLResponse(f"<h3>Spotify auth error: {error}</h3>", status_code=400)

    try:
        token_info = exchange_code(code)
        token = token_info["access_token"]
        return RedirectResponse(url=f"{STREAMLIT_URL}?spotify_token={token}")
    except Exception as e:
        return HTMLResponse(f"<h3>Token exchange failed: {e}</h3>", status_code=500)
