import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from routers import auth, analysis, feedback, debug

app = FastAPI(title="Spotify Intelligence Agent", docs_url=None, redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router,      prefix="/api/spotify",  tags=["auth"])
app.include_router(analysis.router,  prefix="/api",          tags=["analysis"])
app.include_router(feedback.router,  prefix="/api/feedback", tags=["feedback"])
app.include_router(debug.router,     prefix="/api/debug",    tags=["debug"])

FRONTEND = os.path.join(os.path.dirname(__file__), "frontend")

if os.path.isdir(FRONTEND):
    app.mount("/static", StaticFiles(directory=FRONTEND), name="static")

    @app.get("/", include_in_schema=False)
    async def root():
        return FileResponse(os.path.join(FRONTEND, "index.html"))

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(404)
        return FileResponse(os.path.join(FRONTEND, "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)
