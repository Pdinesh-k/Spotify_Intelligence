import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

SPOTIFY_CLIENT_ID: str = os.getenv("SPOTIFY_CLIENT_ID", "")
SPOTIFY_CLIENT_SECRET: str = os.getenv("SPOTIFY_CLIENT_SECRET", "")
SPOTIFY_REDIRECT_URI: str = os.getenv("SPOTIFY_REDIRECT_URI", "http://localhost:8501")

GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = "gemini-2.5-flash"
GEMINI_FALLBACK_MODELS: list = ["gemini-2.5-flash", "gemini-flash-latest", "gemini-2.5-pro"]

CHURN_RISK_THRESHOLD: float = 0.55
MODEL_PATH: str = os.path.join(os.path.dirname(__file__), "data", "churn_model.pkl")
FEEDBACK_PATH: str = os.path.join(os.path.dirname(__file__), "data", "feedback_store.json")
