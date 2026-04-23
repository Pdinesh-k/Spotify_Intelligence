<div align="center">

# 🎵 Spotify Listening Intelligence Agent

**A production ML system that predicts Spotify user churn, explains it with SHAP,
and recommends personalised re-engagement — all from a single OAuth login.**

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-ML_Model-FF6600?style=flat)](https://xgboost.readthedocs.io)
[![Gemini](https://img.shields.io/badge/Gemini-2.5_Flash-4285F4?style=flat&logo=google&logoColor=white)](https://ai.google.dev)
[![Render](https://img.shields.io/badge/Deployed_on-Render-46E3B7?style=flat&logo=render&logoColor=white)](https://render.com)

**[Live Demo](https://spotify-intelligence-367v.onrender.com)**

</div>

---

## What It Does

Most music analytics tools tell you what you already listened to. This one tells you **what's about to happen** — and why.

Connect your Spotify account and in seconds you get:

- **Churn Probability** — a calibrated 0–100% score: how likely you are to disengage from Spotify
- **SHAP Signal Breakdown** — which of your 7 listening behaviours are pushing the score up or down
- **AI Diagnosis** — Gemini analyses your genre entropy, mood trajectory, and discovery health to explain the pattern in plain English
- **Personalised Recommendations** — 3 tracks selected by AI strategy and ranked by cosine similarity to your audio profile
- **Music Profile** — listening heatmap by hour and day, taste trajectory, artist loyalty score, genre diversity chart
- **Self-Learning Feedback Loop** — auto-detects if you played recommended songs next session and improves future picks

---

## Why It Is Different

| Other Music Tools | This System |
|---|---|
| Shows past listening history | Predicts future disengagement risk |
| Static charts and stats | Live ML inference on every login |
| No learning mechanism | Self-improves via per-user feedback loop |
| Requires GDPR data export | Works purely from Spotify OAuth — zero friction |
| Black-box output | Fully explainable via SHAP feature attributions |
| Same experience for all users | Personalised AI diagnosis with mood quadrant + urgency level |

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                  Browser  (Vanilla JS SPA)               │
│            /home  ──►  /analyze  ──►  /results           │
│     Plotly.js: gauge · SHAP waterfall · heatmap · pie    │
└───────────────────────────┬──────────────────────────────┘
                            │  HTTPS
┌───────────────────────────▼──────────────────────────────┐
│              FastAPI Backend  (Python)                   │
│                  Render · port 10000                     │
│                                                          │
│  /api/spotify/url        →  OAuth URL generator          │
│  /api/spotify/callback   →  Token exchange (307)         │
│  /api/analyze            →  Full ML pipeline             │
│  /api/feedback/auto      →  Auto-detect listens          │
│  /api/feedback/stats     →  Per-user feedback metrics    │
│  /api/debug/raw          →  Raw API payload inspector    │
└──────┬──────────────┬──────────────┬─────────────────────┘
       │              │              │
 ┌─────▼──────┐ ┌─────▼──────┐ ┌────▼───────────────────┐
 │  Spotify   │ │ ML Engine  │ │    Gemini Agent         │
 │  Collector │ │            │ │                         │
 │            │ │ 7 Features │ │  Multi-turn tool calls  │
 │  spotipy   │ │  XGBoost   │ │  → analyze_genre_entropy│
 │  OAuth 2.0 │ │  SHAP      │ │  → analyze_mood_traj    │
 │  MemCache  │ │  Feedback  │ │  → eval_discovery_health│
 └─────┬──────┘ └────────────┘ └────────────────────────┘
       │
 ┌─────▼──────────────────────────────────────────────────┐
 │                  Spotify Web API                       │
 │   top tracks · top artists · recently played          │
 │   audio features · genre counts · playlists           │
 └────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
Spotify_Intelligence/
│
├── main.py                   FastAPI app, CORS middleware, SPA fallback route
├── config.py                 Environment variable loader
│
├── spotify/
│   ├── auth.py               OAuth URL generation + code exchange
│   │                         MemoryCacheHandler — prevents shared token bug
│   └── collector.py          Pulls 12+ Spotify API endpoints per user session
│
├── ml/
│   ├── features.py           7-signal feature engineering from raw API data
│   ├── model.py              XGBoost + Sigmoid Calibration + SHAP TreeExplainer
│   ├── feedback.py           Per-user feedback store with auto-expiry
│   └── analytics.py         Heatmap · velocity · loyalty · trajectory · discovery
│
├── agents/
│   ├── diagnosis.py          Gemini multi-turn tool-calling agent
│   ├── tools.py              3 Python tools dispatched by Gemini
│   ├── recommender.py        Spotify search + cosine similarity + feedback blend
│   └── auto_feedback.py     Auto-detects if recommended tracks were played
│
├── routers/
│   ├── auth.py               /api/spotify/* endpoints
│   ├── analysis.py           /api/analyze — pipeline orchestrator
│   ├── feedback.py           /api/feedback/* endpoints
│   └── debug.py              /api/debug/raw — API inspector
│
├── frontend/
│   ├── index.html            Single-page app (all views in one file)
│   └── app.js                Client-side router + all Plotly chart rendering
│
└── ARCHITECTURE.md           Deep-dive: approach, workflow, design decisions
```

---

## The 7 Engineered Features

The entire churn prediction is built on 7 behavioural signals derived from the Spotify API:

| # | Feature | What It Detects | How It Is Computed |
|---|---|---|---|
| 1 | `skip_rate_trend` | Skipping more songs than usual | Popularity drop: short-term vs all-time top tracks |
| 2 | `session_freq_delta` | Sessions growing or declining | Sessions per day: recent vs older half of `recently_played` timestamps |
| 3 | `listen_depth` | How much of each song you actually hear | `avg_energy` from Spotify audio features |
| 4 | `genre_entropy_drop` | Genre diversity collapsing into one style | Shannon entropy concentration from `genre_counts` |
| 5 | `time_of_day_shift` | Listening schedule shifting (new job, travel) | Hour drift between older and newer `recently_played` timestamps |
| 6 | `days_new_artist` | Days since you discovered a new artist | Timestamp of last `recently_played` artist not in `top_artists_long` |
| 7 | `repeat_play_ratio` | Stuck replaying the same songs | Overlap between `recently_played` and `top_tracks_alltime` |

---

## ML Pipeline

```
Synthetic Training Data  (4 000 samples · overlapping class distributions)
              ↓
     XGBClassifier
     80 trees · depth 3 · heavy regularisation
     Prevents overfitting to clean synthetic boundaries
              ↓
     CalibratedClassifierCV  (method="sigmoid", cv=5)
     Platt scaling → smooth 0–100% probability output
     Prevents the 0% / 100% step-function snap
              ↓
     SHAP TreeExplainer  (on base XGBClassifier)
     Per-feature attribution → drives the waterfall chart in the UI
```

**Training data design** — class distributions intentionally overlap so that real-world API values produce genuinely intermediate predictions (6%, 48%, 76%) rather than snapping to extremes.

---

## AI Diagnosis Agent

Gemini 2.5 Flash receives the ML output and **must call all 3 tools** before writing a diagnosis:

```
Model output → Gemini receives: churn %, top SHAP drivers, genres, artists, stats
                      ↓
    ┌─────────────────────────────────────────────────────────┐
    │  Tool 1 · analyze_genre_entropy                        │
    │  Detects musical tunnel vision — genre narrowing        │
    │  Returns: severity, dominant genre, concentration %     │
    └─────────────────────────────────────────────────────────┘
                      ↓
    ┌─────────────────────────────────────────────────────────┐
    │  Tool 2 · analyze_mood_trajectory                      │
    │  Maps valence + energy to 4 emotional quadrants:       │
    │  energetic_positive / calm_content /                   │
    │  tense_stressed / withdrawn_low                        │
    └─────────────────────────────────────────────────────────┘
                      ↓
    ┌─────────────────────────────────────────────────────────┐
    │  Tool 3 · evaluate_discovery_health                    │
    │  Composite score from days_new_artist +                │
    │  repeat_ratio + skip_trend                             │
    │  Returns: health label (critical / poor / fair / good) │
    └─────────────────────────────────────────────────────────┘
                      ↓
    Gemini synthesises all 3 outputs → final JSON:
    {
      "diagnosis":       "What the listening pattern shows",
      "hypothesis":      "Likely underlying cause (life event, mood)",
      "strategy":        "Concrete re-engagement plan",
      "strategy_genre":  "jazz",
      "strategy_artist": "Norah Jones",
      "urgency":         "monitor | act_soon | act_now"
    }
                      ↓
    If Gemini is rate-limited → rule-based fallback using churn% + SHAP drivers
```

---

## Self-Improving Feedback Loop

```
Session 1  →  Analyse user
           →  Generate 3 recommendations
           →  Store as "pending" with timestamp per user

Session 2  →  Auto-feedback runs BEFORE analysis:
              Fetch recently_played since recommendation time
                      ↓
              Track appeared in plays?   → "listened"  score +1.0
              48 hours passed, no play?  → "skipped"   score −0.5
                      ↓
              Scores feed into next ranking:
              final_score = 0.7 × cosine_similarity
                          + 0.3 × tanh(feedback_score)
```

Tracks the user consistently plays after being recommended rise in future sessions. Tracks always ignored sink. The system adapts per user, per session.

---

## Running Locally

### Prerequisites
- Python 3.11
- A [Spotify Developer app](https://developer.spotify.com/dashboard)
- A [Gemini API key](https://aistudio.google.com/app/apikey)

### Steps

**1. Clone and install**
```bash
git clone https://github.com/Pdinesh-k/Spotify_Intelligence.git
cd Spotify_Intelligence
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**2. Create `.env`**
```env
SPOTIFY_CLIENT_ID=your_client_id
SPOTIFY_CLIENT_SECRET=your_client_secret
SPOTIFY_REDIRECT_URI=http://127.0.0.1:8000/api/spotify/callback
GEMINI_API_KEY=your_gemini_key
```

**3. Register redirect URI in Spotify Dashboard**

Go to [developer.spotify.com/dashboard](https://developer.spotify.com/dashboard) → your app → Settings → Redirect URIs → Add:
```
http://127.0.0.1:8000/api/spotify/callback
```

**4. Start the server**
```bash
uvicorn main:app --port 8000 --reload
```

Open **http://localhost:8000** in your browser.

---

## Deploying to Render

| Field | Value |
|---|---|
| Build command | `pip install -r requirements.txt` |
| Start command | `uvicorn main:app --host 0.0.0.0 --port $PORT` |
| Environment vars | `SPOTIFY_CLIENT_ID` · `SPOTIFY_CLIENT_SECRET` · `SPOTIFY_REDIRECT_URI` · `GEMINI_API_KEY` |
| Redirect URI | `https://your-app.onrender.com/api/spotify/callback` |

Add the Render redirect URI to your Spotify Dashboard alongside the localhost one.

---

## Spotify OAuth Scopes

| Scope | Purpose |
|---|---|
| `user-top-read` | Top tracks and artists across short, medium, and long time ranges |
| `user-read-recently-played` | Last 50 plays with timestamps — critical for feature engineering |
| `playlist-read-private` | Playlist metadata |
| `user-read-private` | Profile, country, subscription tier |

---

## Debug Endpoint

Inspect exactly what the Spotify API returns for any user:

```
GET /api/debug/raw?token=<spotify_access_token>
```

Returns: user profile · API endpoint counts · audio features · genre counts · computed features · model result — all in one response. Useful for diagnosing why a user's churn prediction looks unexpected.

---

## Known Limitations

| Limitation | Cause |
|---|---|
| Audio features return 403 | Spotify deprecated the endpoint for non-extended-quota apps |
| Related artists return 403 | Same Spotify API restriction since 2024 |
| Max 5 users in dev mode | Spotify development mode limit — need extended quota for public launch |
| Feedback resets on redeploy | Render free tier uses ephemeral disk storage |
| AI Diagnosis falls back | Gemini free tier rate limits — rule-based fallback activates automatically |

---

<div align="center">

Built with **FastAPI · XGBoost · SHAP · Gemini 2.5 Flash · Spotipy · Plotly.js**

See [ARCHITECTURE.md](./ARCHITECTURE.md) for the full technical deep-dive.

</div>
