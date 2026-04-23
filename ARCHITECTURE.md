# Spotify Listening Intelligence Agent — Architecture & Design

## What It Is

A production ML system that analyses a Spotify user's listening behaviour in real time, predicts their **churn probability** (likelihood of disengaging from Spotify), explains the prediction with SHAP-based signal breakdowns, and generates personalised re-engagement recommendations — all from a single OAuth login, no manual data upload required.

---

## Why It Is Unique

Most music analytics tools (Spotify Wrapped, Last.fm, Receiptify) are **retrospective** — they tell you what you already listened to. This system is **predictive and prescriptive**:

| Typical Music App | This System |
|---|---|
| Shows past listening history | Predicts future disengagement risk |
| Static charts and stats | Live ML inference on every login |
| No feedback loop | Learns from whether you actually played recommended songs |
| Same interface for every user | Personalised AI diagnosis with mood quadrant and urgency level |
| Requires data download (GDPR export) | Works purely from the Spotify API — zero friction |

The core insight: **behavioural signals in listening patterns (genre narrowing, session drop, discovery stagnation) predict churn weeks before a user consciously decides to stop listening.** This is the same problem Spotify's own data science team solves internally — this project replicates it as an open, explainable system.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      User's Browser                         │
│              Vanilla JS SPA (no framework)                  │
│         /home  →  /analyze  →  /results                     │
│  Plotly.js charts: gauge, SHAP waterfall, heatmap, pie      │
└────────────────────┬────────────────────────────────────────┘
                     │ HTTPS (REST + OAuth redirect)
┌────────────────────▼────────────────────────────────────────┐
│                  FastAPI Backend (Python)                    │
│                  Render free tier · port 10000              │
│                                                             │
│  /api/spotify/url          →  OAuth URL generator           │
│  /api/spotify/callback     →  Token exchange (307 redirect) │
│  /api/analyze              →  Full analysis pipeline        │
│  /api/feedback/auto        →  Auto-detect listened songs    │
│  /api/feedback/stats       →  Per-user feedback metrics     │
│  /api/debug/raw            →  Raw API payload inspector     │
└──────┬──────────────┬──────────────┬────────────────────────┘
       │              │              │
┌──────▼──────┐ ┌─────▼──────┐ ┌────▼──────────────────────┐
│  Spotify    │ │  ML Engine  │ │   Gemini Agent (LLM)       │
│  Collector  │ │             │ │                            │
│             │ │  Features   │ │  Multi-turn tool calling   │
│  spotipy    │ │  → Model    │ │  3 tools dispatched        │
│  OAuth 2.0  │ │  → SHAP     │ │  → Rule-based fallback     │
│  MemCache   │ │  → Feedback │ │                            │
└──────┬──────┘ └─────┬──────┘ └────────────────────────────┘
       │              │
┌──────▼──────────────▼──────────────────────────────────────┐
│              Spotify Web API                                │
│  user-top-read · user-read-recently-played                  │
│  playlist-read-private · user-read-private                  │
└─────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
Spotify_Intelligence/
│
├── main.py                      FastAPI app, CORS, SPA fallback route
├── config.py                    Env vars: Spotify keys, Gemini key, paths
│
├── spotify/
│   ├── auth.py                  OAuth URL generation + code exchange
│   │                            MemoryCacheHandler — no shared token files
│   └── collector.py             Pulls all Spotify API data for a user:
│                                top tracks (3 ranges), top artists (3 ranges),
│                                recently played, audio features, genre counts,
│                                related artists, saved tracks, playlists
│
├── ml/
│   ├── features.py              Feature engineering (7 signals from API data)
│   ├── model.py                 XGBoost + Sigmoid Calibration + SHAP
│   ├── feedback.py              Per-user feedback store (JSON files)
│   └── analytics.py            Pure analytics: heatmap, velocity, loyalty,
│                                genre profile, trajectory, discovery funnel
│
├── agents/
│   ├── diagnosis.py             Gemini multi-turn tool-calling agent
│   ├── tools.py                 3 Python tools: genre entropy, mood, discovery
│   ├── recommender.py           Spotify search + cosine similarity ranking
│   └── auto_feedback.py         Auto-detect if user played recommended tracks
│
├── routers/
│   ├── auth.py                  /api/spotify/* endpoints
│   ├── analysis.py              /api/analyze — full pipeline orchestration
│   ├── feedback.py              /api/feedback/* — manual + auto feedback
│   └── debug.py                 /api/debug/raw — raw API payload inspector
│
└── frontend/
    ├── index.html               Single HTML file — all pages via show/hide
    └── app.js                   Client-side router + all chart rendering
```

---

## The 7 Engineered Features

The entire churn prediction rests on **7 behavioural signals** derived from Spotify API data. These are computed in `ml/features.py`.

| # | Feature | What It Measures | How Computed |
|---|---|---|---|
| 1 | `skip_rate_trend` | Are you skipping more songs than before? | Popularity drop (short vs long term) as proxy |
| 2 | `session_freq_delta` | Are your listening sessions increasing or declining? | Sessions per day in recent vs older half of `recently_played` timestamps |
| 3 | `listen_depth` | How much of each song are you actually hearing? | `avg_energy` from Spotify audio features |
| 4 | `genre_entropy_drop` | Is your genre diversity collapsing? | Shannon entropy concentration from `genre_counts` |
| 5 | `time_of_day_shift` | Has your listening schedule shifted (new job, travel)? | Hour drift between older vs newer half of `recently_played` timestamps |
| 6 | `days_new_artist` | How long since you discovered a new artist? | Days since last `recently_played` artist not in `top_artists_long`. Falls back to `top_artists_short vs long` comparison when no play history |
| 7 | `repeat_play_ratio` | Are you stuck replaying the same songs? | Overlap between `recently_played` names and `top_tracks_alltime`. Neutral (0.5) when no play history |

**Key design decision:** Features 2, 5, and 6 all use real timestamps from `recently_played`. When that data is unavailable (e.g. account with no recent plays), fallbacks are set to neutral-to-slightly-churned values — not falsely-engaged values (the original bug that caused everyone to show 0% churn).

---

## The ML Model

**File:** `ml/model.py`

### Architecture

```
Synthetic Training Data (4000 samples)
        ↓
XGBClassifier (base model)
  - 80 estimators, depth 3
  - Heavy regularisation (gamma=0.5, reg_alpha=1.0, reg_lambda=2.0)
  - Prevents overfitting to clean synthetic boundaries
        ↓
CalibratedClassifierCV (method="sigmoid", cv=5)
  - Platt scaling — produces smooth probability curves
  - Prevents the 0%/100% step-function problem of isotonic calibration
        ↓
SHAP TreeExplainer (on base XGBClassifier)
  - Per-feature attribution: how much each signal pushed toward or away from churn
  - Drives the waterfall chart in the UI
```

### Synthetic Data Design

The model is trained on synthetic data because real labelled churn data is unavailable. The key was making the two classes **genuinely overlap** so the calibrated model outputs probabilities across the full 0–100% range:

| Feature | Engaged (label=0) | Churning (label=1) | Real API range |
|---|---|---|---|
| `skip_rate_trend` | N(0.01, 0.07) | N(0.12, 0.08) | 0.00 – 0.25 |
| `session_freq_delta` | N(0.8, 2.0) | N(-1.2, 2.0) | -3.0 – 3.0 |
| `listen_depth` | N(0.68, 0.14) | N(0.56, 0.14) | 0.45 – 0.80 |
| `genre_entropy_drop` | N(0.18, 0.18) | N(0.55, 0.22) | 0.0 – 1.0 |
| `time_of_day_shift` | N(2.0, 2.2) | N(5.5, 2.5) | 0.0 – 10.0 |
| `days_new_artist` | N(2.0, 2.0) | N(5.0, 1.8) | 0.0 – 7.0 |
| `repeat_play_ratio` | N(0.25, 0.18) | N(0.55, 0.20) | 0.0 – 0.80 |

The wide standard deviations ensure that real-world API values (which don't cleanly separate into two groups) produce genuinely intermediate probabilities like 6%, 48%, 76%.

---

## The AI Diagnosis Agent

**File:** `agents/diagnosis.py`

The diagnosis layer sits on top of the ML model and adds human-readable interpretation using **Gemini 2.5 Flash** with structured tool-calling.

### Flow

```
Model Output (churn probability, SHAP values, feature values)
        ↓
Gemini receives: churn%, risk level, top drivers, listening stats, genres, artists
        ↓
Gemini MUST call all 3 tools before writing diagnosis:

  Tool 1: analyze_genre_entropy
    → Computes severity of genre narrowing from genre_counts
    → Returns: severity, dominant genre, concentration %, interpretation

  Tool 2: analyze_mood_trajectory
    → Maps valence + energy to 4 emotional quadrants:
      energetic_positive / calm_content / tense_stressed / withdrawn_low
    → Returns: mood label, listen depth label, schedule shift note

  Tool 3: evaluate_discovery_health
    → Composite score from days_new_artist + repeat_ratio + skip_trend
    → Returns: health label (critical/poor/fair/good), anchor artists

        ↓
Gemini synthesises tool outputs into final JSON:
  {
    "diagnosis":       "What the listening pattern shows",
    "hypothesis":      "Likely underlying cause (life event, mood)",
    "strategy":        "Concrete re-engagement plan",
    "strategy_genre":  "Single genre for Spotify search",
    "strategy_artist": "Single artist name for search",
    "urgency":         "monitor | act_soon | act_now"
  }
        ↓
If Gemini fails/rate-limited → rule-based fallback using churn% + top SHAP drivers
```

---

## The Recommendation Engine

**File:** `agents/recommender.py`

Uses the AI strategy output to search Spotify and rank candidates by audio feature similarity.

```
strategy_artist + strategy_genre (from Gemini)
        ↓
Spotify Search API (with user's country as market)
  1. Search by strategy artist (limit 10)
  2. Search by strategy genre (limit 10, if < 10 candidates)
  3. Fallback: search user's top 3 artists (limit 5 each)
        ↓
Audio Features (energy, valence, danceability, acousticness)
  → Build 5-dim user vector from avg_energy, avg_valence, avg_danceability
  → Build track vector for each candidate
  → Cosine similarity score
  → If audio features 403'd: use popularity-based score
        ↓
Feedback Scoring (per-user)
  → Retrieve historical track scores from FeedbackStore(user_id)
  → final_score = 0.7 × similarity + 0.3 × tanh(feedback_score)
        ↓
Diversity Filter: one track per artist
Top 3 recommendations returned
```

---

## The Feedback Loop (Self-Improving)

**File:** `ml/feedback.py`, `agents/auto_feedback.py`

This is what makes the system learn over sessions rather than giving static recommendations.

```
Session 1: Analyse user → generate 3 recommendations
        ↓ store as "pending" in feedback_<user_id>.json
        ↓ tagged with recommendation timestamp

Session 2 (next login): Auto-feedback runs BEFORE analysis
        ↓
  auto_feedback fetches recently_played since recommendation timestamp
  ↓
  If recommended track appears in recent plays → outcome = "listened" (+1.0 score)
  If 48 hours pass with no play → outcome = "skipped" (−0.5 score)
        ↓
  Scores feed back into cosine ranking next session:
  final_score = 0.7 × similarity + 0.3 × tanh(feedback_score)
```

**Effect:** Tracks that users consistently play after being recommended get boosted in future recommendations. Tracks that are always ignored get penalised. The system adapts per user, per session.

---

## Authentication Design

**File:** `spotify/auth.py`

### Critical: MemoryCacheHandler

Spotipy's default OAuth handler caches tokens to a `.cache` file on disk. In a shared deployment environment this means **every user gets the same cached token** — a critical security bug discovered and fixed early in this project.

The fix: `MemoryCacheHandler()` — tokens live only in process memory for the duration of the exchange. Each user's token is passed explicitly in every API call, never stored server-side.

```python
SpotifyOAuth(
    ...
    cache_handler=spotipy.cache_handler.MemoryCacheHandler(),
    open_browser=False,
)
```

### Token Flow

```
1. User clicks "Connect Spotify"
2. Frontend: GET /api/spotify/url  →  receives Spotify OAuth URL
3. Browser redirects to accounts.spotify.com/authorize
4. User approves → Spotify redirects to /api/spotify/callback?code=...
5. Backend: exchange code for access_token  →  307 redirect to /?token=...
6. Frontend: stores token in localStorage["si_token"]
7. All subsequent API calls pass token explicitly in request body/params
```

---

## Data Flow: Full Analysis Pipeline

When a user logs in, this is the exact sequence:

```
1. OAuth Token obtained (localStorage)
           ↓
2. GET /api/feedback/auto?token=...
   → Checks if any previously recommended tracks were played
   → Logs outcomes, updates scores
   → Returns auto-detected outcomes for UI banner
           ↓
3. POST /api/analyze  (FormData: token)
           ↓
   3a. collect_user_data(token)
       → 12+ Spotify API calls in sequence:
         profile, top_tracks×3, top_artists×3,
         recently_played, audio_features, related_artists,
         saved_tracks, playlists, genre aggregation
           ↓
   3b. extract_features_from_api(user_profile)
       → Computes the 7 features from raw API data
       → Uses timestamps from recently_played for features 2, 5, 6
       → Returns feature dict
           ↓
   3c. predict(features)
       → XGBClassifier → CalibratedClassifierCV → churn_probability
       → SHAP TreeExplainer → per-feature attributions
       → Returns: probability, risk_level, shap_values, top_drivers
           ↓
   3d. generate_diagnosis(model_result, user_profile)
       → Gemini agent: 3 tool calls → JSON diagnosis
       → Falls back to rule-based if Gemini unavailable
           ↓
   3e. get_recommendations(diagnosis, user_profile, token)
       → Spotify search → cosine similarity → feedback blend → top 3
           ↓
   3f. FeedbackStore(user_id).store_pending(recommendations)
       → Saves recs as pending for next session's auto-feedback
           ↓
   3g. analytics.compute_all(user_profile)
       → Temporal heatmap, listening velocity, artist loyalty,
         genre profile, taste trajectory, discovery funnel
           ↓
4. Response JSON → Frontend renders:
   - Gauge: churn probability
   - SHAP waterfall: feature signal breakdown
   - Stats bar: hours, tracks, obsession rate, days since new artist
   - AI Diagnosis: diagnosis + hypothesis + strategy + urgency
   - Recommendations: 3 tracks with similarity scores
   - Music Profile tab: heatmap, loyalty, trajectory, genre pie
   - Feedback Loop tab: success rate trend, auto-detected count
```

---

## Analytics Engine

**File:** `ml/analytics.py`

Runs in parallel with the ML pipeline and computes purely descriptive metrics:

| Metric | What It Computes |
|---|---|
| **Temporal Heatmap** | Listening counts by hour (0–23) and day-of-week from `recently_played` timestamps |
| **Listening Velocity** | Tracks per hour, total hours, average track duration across 50 most recent plays |
| **Artist Loyalty** | Jaccard similarity between short-term and long-term top artists. Labels: Deeply Loyal / Balanced Explorer / Active Explorer |
| **Taste Trajectory** | Jaccard similarity of track IDs across short/medium/long ranges. Labels: Stable / Gradually Evolving / Rapidly Shifting |
| **Genre Profile** | Shannon entropy, dominant genre, concentration %, top 8 genres for pie chart |
| **Discovery Funnel** | Artists in short-term not in long-term. Related artists not yet in listening history |

---

## Deployment

| Component | Platform |
|---|---|
| Backend + Frontend | Render Free Tier (Web Service) |
| Keep-alive | UptimeRobot (pings every 5 min to prevent 15-min sleep) |
| Environment Variables | Render Dashboard (SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, SPOTIFY_REDIRECT_URI, GEMINI_API_KEY) |
| Spotify App | Developer Dashboard — Development Mode (max 5 registered users) |
| Model | Trained on first startup if `.pkl` not present (takes ~0.5s) |
| Feedback Store | Per-user JSON files on Render's ephemeral disk (resets on redeploy) |

### Scopes Requested

```
user-top-read          → Top tracks and artists across all time ranges
user-read-recently-played → Last 50 plays with timestamps (critical for feature engineering)
playlist-read-private  → Playlist metadata
user-read-private      → Profile, country, subscription tier
```

---

## Known Limitations

| Limitation | Reason | Impact |
|---|---|---|
| Audio features return 403 | Spotify deprecated the endpoint for non-extended-quota apps | `listen_depth` uses `avg_energy` proxy instead of actual completion rate |
| Related artists return 403 | Same Spotify restriction | Discovery funnel undiscovered artists list is empty |
| `recently_played` sometimes empty | New accounts or rarely-used accounts | Features 2, 5, 6 use fallback values; prediction is less personalised |
| Feedback store resets on redeploy | Render free tier has ephemeral disk | Feedback loop learning does not persist across deployments |
| Max 5 users in dev mode | Spotify development mode restriction | Need extended quota approval for public launch |
| Gemini rate limits | Free tier quota | AI Diagnosis falls back to rule-based on quota exhaustion |
