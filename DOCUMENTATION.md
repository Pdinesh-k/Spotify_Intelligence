# Spotify Intelligence Agent

## Overview
The **Spotify Intelligence Agent** is an advanced application that analyzes Spotify listening habits to detect and counteract user disengagement ("churn"). Instead of a naive LLM wrapper, it employs a rigorous **4-layer architecture**. It first uses classical Machine Learning (XGBoost + SHAP) to derive strong empirical signals from listening histories, then invokes an **agentic LLM** (Google Gemini) that reasons over these signals via specific tool calls to create a tailored re-engagement strategy. 

The application offers two interfaces:
1. **Streamlit App:** Run via `app.py` for a rapid prototyping and data visualization UI.
2. **FastAPI + SPA Backend:** Run via `main.py` serving a RESTful API (`routers/`) and a frontend web interface.

## Architecture & How It Works

The system is broken down into four explicit modules:

### Module 1: Signal Layer (Classical ML)
**Files Used:** `ml/features.py`, `ml/model.py`
To avoid LLM hallucinations, the system begins with hard data. It takes in either a user's uploaded Extended Streaming History JSON (ideal for rich historical data) or queries the Spotify API (as a proxy).

It extracts **7 engineered features**:
1. `skip_rate_trend`: Increase or decrease in skips.
2. `session_freq_delta`: Change in daily listening sessions.
3. `listen_depth`: Average fraction of a song completed.
4. `genre_entropy_drop`: Measuring narrowing diversity ("tunnel vision").
5. `time_of_day_shift`: Shifts in schedule (indicating lifestyle changes).
6. `days_new_artist`: Days since discovering new music.
7. `repeat_play_ratio`: Fraction of songs played on repeat.

These features are passed into a calibrated **XGBoost Classifier** that predicts a **Churn Probability** (risk of disengagement: Low, Medium, High). A **SHAP** TreeExplainer extracts the top drivers—i.e., *why* the model predicted this risk.

### Module 2: Reasoning Layer (LLM Agent)
**Files Used:** `agents/diagnosis.py`, `agents/tools.py`
Next, a multi-turn Google Gemini agent acts as a music behavior analyst. **Crucially, the LLM never sees raw data.** Instead, it is given the Churn Probability, top SHAP drivers, and the user's top genres/artists. 

The agent is forced to execute three specific analytical **tools** (Python functions) before formulating an answer:
1. `analyze_genre_entropy`: Checks if listening is locked into a single genre loop.
2. `analyze_mood_trajectory`: Checks valence (positivity) and energy to detect emotional state (e.g., *Stressed*, *Calm & Content*, *Low Energy & Withdrawn*).
3. `evaluate_discovery_health`: Checks if the user is failing to discover new music.

Once these tools return data, the LLM integrates them to output a structured JSON containing a **Diagnosis**, a psychological **Hypothesis**, and a concrete **Strategy** (e.g., "Re-engage via slow tempo Jazz").

### Module 3: Action Layer (Recommendations)
**Files Used:** `agents/recommender.py`
Using the strategy (target genre & anchor artist) from the LLM, the system queries the Spotify API for tracks. It then ranks these recommendations using a formula blending:
- **Cosine Similarity (70%)**: To match the track's audio vector with the user's historical profile.
- **Learned Feedback Score (30%)**: Based on past user feedback (see below).

### Module 4: Agentic Feedback Loop
**Files Used:** `ml/feedback.py`, `agents/auto_feedback.py`
The loop closes with continuous learning:
- **Manual Feedback**: Users can explicitly click "✓ Listened" (+1 score) or "✗ Skipped" (-0.5 score) on recommended tracks.
- **Auto Feedback**: On subsequent logins, the system passively checks Spotify's "Recently Played" endpoints to identify if the user listened to the previous session's recommendations naturally.
The system is self-optimizing; the feedback modifies the mathematical ranking layer for future sessions.

---

## Example Walkthrough

Let's assume a user named **Alex** is analyzed using their streaming history.

### Step 1: Feature Extraction & ML
- Alex has recently stopped listening to new albums, is skipping tracks often, and playing the same 10 songs.
- The **XGBoost Model** output: *Churn Probability: 82% (High Risk).*
- **SHAP Drivers:** High `skip_rate_trend` (+0.35 risk), high `repeat_play_ratio`, collapsing `genre_entropy`. 

### Step 2: Agent Reasoning
- **Agent Initialization:** The Gemini agent receives "82% Churn Risk" and top drivers.
- **Tool Call 1 (`analyze_mood_trajectory`):** Returns "Low Energy & Withdrawn" because Alex's average valence and energy dropped significantly.
- **Tool Call 2 (`analyze_genre_entropy`):** Returns "Severe tunnel vision: 80% concentrated in 'sad lo-fi'."
- **Tool Call 3 (`evaluate_discovery_health`):** Returns "Critical. 15 days since a new artist."
- **LLM Synthesis (Output):**
  - **Diagnosis:** Alex is highly disengaged, exhibiting a severe genre tunnel-vision coupled with a low-energy emotional state.
  - **Hypothesis:** This pattern suggests Alex is using music purely as passive comfort during a stressful or withdrawn period.
  - **Strategy:** Provide familiar but adjacent comfort music; introduce one low-tempo, familiar acoustic track to gently break the loop.

### Step 3: Action & Feedback
- The recommender searches for acoustic tracks related to Alex's top artists, fetching 10 tracks.
- It ranks them via cosine similarity to ensure they fit the "low energy comfort" profile, avoiding jarring loud tracks.
- It presents the Top 3 to Alex.
- If Alex listens to one, it logs a success. If skipped, the system penalizes that track profile, learning to avoid it in future recommendations.
