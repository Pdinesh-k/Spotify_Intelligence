# Spotify Intelligence Agent

Spotify Intelligence Agent is a full-stack, data-driven application designed to analyze your Spotify listening history, detect behavioral "churn" (disengagement), and generate targeted track recommendations to re-engage you.

It achieves this through a unique marriage of **Classical Machine Learning** and **Agentic LLMs**. Instead of naively sending your raw data into an LLM, this system relies on a mathematical XGBoost layer to extract clear statistical signals from your listening habits, which it then feeds to an LLM agent that acts as a behavioral analyst.

For an extensive explanation of the architecture and workflow, see [DOCUMENTATION.md](./DOCUMENTATION.md).

## Features
- **Classical ML (XGBoost + SHAP):** Derives 7 engineered features like `skip_rate_trend`, `listen_depth`, and `genre_entropy_drop` from streaming history.
- **Agentic LLM (Google Gemini):** Synthesizes machine learning signals via explicitly defined Python tools to produce a human-readable diagnosis and strategy.
- **Content-Based Action:** Ranks and recommends Spotify tracks based on cosine similarity logic and feedback loop data.
- **Feedback Loop:** Continuously self-optimizes via automatic background checks and manual user feedback logs (Listened ✓ / Skipped ✗).
- **Dual Interfaces:** Features both an exploratory Streamlit app (`app.py`) and a production-grade FastAPI REST server (`main.py`) offering the same backend integrations.

## Installation

1. **Clone the Repository** and navigate to it:
   ```bash
   git clone <repository-url>
   cd Spotify_Project
   ```

2. **Set up a Virtual Environment (Recommended):**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On macOS/Linux
   # or on Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Environment Configuration

Create a `.env` file in the root directory and populate it with your API keys:

```env
# Required for connecting to Spotify to pull profile metrics and retrieve tracks
SPOTIFY_CLIENT_ID=your_spotify_client_id_here
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret_here  # If applicable to your auth flow

# Required for the LLM Agent reasoning
GEMINI_API_KEY=your_google_gemini_api_key_here
```

## Running the Application

This project provides two separate interfaces for interaction—a developer-friendly interactive dashboard and a decoupled FastAPI production server.

### Option 1: Streamlit Dashboard
Best for testing, rapid prototyping, and easily viewing SHAP and LLM output diagnostics.
```bash
streamlit run app.py
```

### Option 2: FastAPI Server
Runs the REST API and mounts the Single Page Application (SPA) frontend found in the `frontend` folder.
```bash
python main.py
```
*(By default, this will expose the app on `http://0.0.0.0:8000`)*

## Uploading Extended Streaming History
While the application can estimate features using the public Spotify API, its ML engine performs optimally with full **Extended Streaming History**.
- To retrieve this, log into your Spotify Account online.
- Go to **Privacy Settings > Download your Data** and request your **Extended Streaming History**.
- You can upload these JSON files directly via the app's UI for comprehensive ML processing.

## License
Provided as-is. See root directory for applicable license terms.
