import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import SPOTIFY_CLIENT_ID, CHURN_RISK_THRESHOLD
from ml.features import (
    FEATURE_LABELS,
    FEATURE_DESCRIPTIONS,
    extract_features_from_api,
    extract_features_from_history,
    features_to_array,
    load_history_files,
)
from ml.feedback import FeedbackStore
from ml.model import load_models, predict
from spotify.auth import get_oauth_url
from spotify.collector import collect_user_data
from agents.diagnosis import generate_diagnosis
from agents.recommender import get_recommendations
from agents.auto_feedback import run_auto_feedback


# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Spotify Intelligence Agent",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  /* Card containers */
  .intel-card {
    background: #1a1a2e;
    border: 1px solid #16213e;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
  }
  .risk-high   { border-left: 4px solid #ff4b4b; }
  .risk-medium { border-left: 4px solid #ffa500; }
  .risk-low    { border-left: 4px solid #00c851; }

  /* Track recommendation cards */
  .track-card {
    background: #0f3460;
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 0.6rem;
  }

  /* Diagnosis text */
  .diagnosis-text {
    font-size: 1.05rem;
    line-height: 1.7;
    color: #e0e0e0;
  }
  .hypothesis-text {
    font-size: 0.95rem;
    color: #a0aec0;
    font-style: italic;
  }

  /* Metric pills */
  .pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-right: 6px;
  }
  .pill-green  { background: #1a4731; color: #48bb78; }
  .pill-red    { background: #4a1515; color: #fc8181; }
  .pill-orange { background: #4a3000; color: #f6ad55; }

  /* Section headers */
  .section-header {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #718096;
    margin-bottom: 0.5rem;
  }

  /* Hide Streamlit branding */
  #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── OAuth callback handling ───────────────────────────────────────────────────

params = st.query_params
if "spotify_token" in params and "spotify_token" not in st.session_state:
    st.session_state["spotify_token"] = params["spotify_token"]
    st.query_params.clear()
    st.rerun()

# ── Auto-feedback: run once per session on first load ─────────────────────────
if (
    "spotify_token" in st.session_state
    and "auto_feedback_ran" not in st.session_state
):
    try:
        auto_outcomes = run_auto_feedback(st.session_state["spotify_token"])
        st.session_state["auto_outcomes"] = auto_outcomes
    except Exception:
        st.session_state["auto_outcomes"] = []
    st.session_state["auto_feedback_ran"] = True


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🎵 Spotify Intelligence")
    st.caption("Classical ML · LLM Reasoning · Agentic Loop")
    st.divider()

    # Connection status
    if "spotify_token" not in st.session_state:
        if SPOTIFY_CLIENT_ID:
            auth_url = get_oauth_url("intelligence")
            st.link_button(
                "🔗 Connect Spotify",
                auth_url,
                use_container_width=True,
            )
            st.caption("Redirects to Spotify login, returns here automatically.")
        else:
            st.error("Add `SPOTIFY_CLIENT_ID` to `.env` first.")
    else:
        profile_name = st.session_state.get("user_profile", {}).get("display_name", "")
        if profile_name:
            st.success(f"Connected: **{profile_name}**")
        else:
            st.success("Spotify connected ✓")
        if st.button("Disconnect", use_container_width=True):
            for key in ["spotify_token", "user_profile", "features",
                        "model_result", "diagnosis", "recommendations",
                        "history_mode"]:
                st.session_state.pop(key, None)
            st.rerun()

    st.divider()
    st.markdown("**Upload Streaming History** *(optional but better)*")
    uploaded_files = st.file_uploader(
        "Spotify Extended Streaming History",
        type=["json"],
        accept_multiple_files=True,
        help=(
            "Spotify account → Privacy settings → Request data → "
            "Extended Streaming History. Unlocks all 7 features."
        ),
        label_visibility="collapsed",
    )
    if uploaded_files:
        st.caption(f"{len(uploaded_files)} file(s) ready")

    st.divider()
    analyze_btn = st.button(
        "▶ Run Analysis",
        use_container_width=True,
        type="primary",
        disabled="spotify_token" not in st.session_state,
    )

    st.divider()
    st.markdown("""
    <div style='font-size:0.78rem; color:#4a5568; line-height:1.6'>
    <b>Architecture</b><br>
    Module 1 → Feature store (XGBoost)<br>
    Module 2 → LLM diagnosis (Gemini)<br>
    Module 3 → Content retrieval<br>
    Module 4 → Feedback loop<br>
    </div>
    """, unsafe_allow_html=True)


# ── Helper: SHAP waterfall chart ──────────────────────────────────────────────

def _shap_waterfall(shap_values: dict, base_value: float, final_prob: float) -> go.Figure:
    items = sorted(shap_values.items(), key=lambda x: x[1])
    labels = [FEATURE_LABELS.get(k, k) for k, _ in items]
    values = [v for _, v in items]
    colors = ["#fc8181" if v > 0 else "#68d391" for v in values]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=colors,
        hovertemplate="%{y}: %{x:+.3f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color="#718096", line_width=1)
    fig.add_annotation(
        x=max(abs(v) for v in values) * 0.9,
        y=len(labels) - 0.3,
        text=f"Final: {final_prob:.1%}",
        showarrow=False,
        font=dict(color="#e2e8f0", size=12),
    )
    fig.update_layout(
        title="SHAP Feature Contributions",
        plot_bgcolor="#0d1117",
        paper_bgcolor="#0d1117",
        font_color="#e2e8f0",
        margin=dict(l=10, r=10, t=40, b=10),
        height=280,
        xaxis=dict(
            title="SHAP value (→ increases churn risk)",
            gridcolor="#1e2533",
            zerolinecolor="#718096",
        ),
        yaxis=dict(gridcolor="#1e2533"),
    )
    return fig


# ── Helper: churn risk gauge ──────────────────────────────────────────────────

def _risk_gauge(prob: float, risk_level: str) -> go.Figure:
    color = "#ff4b4b" if risk_level == "High" else "#ffa500" if risk_level == "Medium" else "#00c851"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 36, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#718096"},
            "bar": {"color": color},
            "bgcolor": "#1a1a2e",
            "bordercolor": "#2d3748",
            "steps": [
                {"range": [0, 38], "color": "#0a2a14"},
                {"range": [38, 65], "color": "#2a1e08"},
                {"range": [65, 100], "color": "#2a0a0a"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.75,
                "value": prob * 100,
            },
        },
        title={"text": f"Churn Risk · {risk_level}", "font": {"color": "#a0aec0", "size": 14}},
    ))
    fig.update_layout(
        paper_bgcolor="#0d1117",
        font_color="#e2e8f0",
        height=220,
        margin=dict(l=20, r=20, t=10, b=10),
    )
    return fig


# ── Helper: feature table ─────────────────────────────────────────────────────

def _feature_table(feature_values: dict, shap_values: dict):
    rows = []
    for feat, val in feature_values.items():
        sv = shap_values.get(feat, 0.0)
        label = FEATURE_LABELS.get(feat, feat)
        desc = FEATURE_DESCRIPTIONS.get(feat, "")
        impact = "🔴 Risk" if sv > 0.02 else "🟢 Safe" if sv < -0.02 else "⚪ Neutral"
        rows.append({
            "Feature": label,
            "Value": f"{val:.3f}",
            "SHAP": f"{sv:+.3f}",
            "Impact": impact,
            "What it measures": desc,
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


# ── Helper: recommendation card ──────────────────────────────────────────────

def _recommendation_cards(recommendations: list, model_result: dict):
    store = FeedbackStore()

    for i, track in enumerate(recommendations):
        col_img, col_info, col_btns = st.columns([1, 5, 2])

        with col_img:
            if track.get("album_image"):
                st.image(track["album_image"], width=70)
            else:
                st.markdown("🎵")

        with col_info:
            name = track["name"]
            artist = track["artist"]
            url = track.get("external_url", "")
            link = f"[**{name}**]({url})" if url else f"**{name}**"
            st.markdown(f"{link}  \n*{artist}*")
            sim = track.get("similarity", 0.0)
            fb = track.get("feedback_score", 0.0)
            st.markdown(
                f'<span class="pill pill-green">Similarity {sim:.0%}</span>'
                f'<span class="pill pill-orange">Feedback {fb:+.1f}</span>',
                unsafe_allow_html=True,
            )

        with col_btns:
            listened_key = f"listened_{i}_{track['id']}"
            skipped_key = f"skipped_{i}_{track['id']}"
            done_key = f"done_{track['id']}"

            if st.session_state.get(done_key):
                st.caption("✓ logged")
            else:
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("✓", key=listened_key, help="I listened"):
                        store.log_interaction(
                            track["id"], track["name"], track["artist"],
                            "listened", model_result["churn_probability"],
                        )
                        st.session_state[done_key] = True
                        st.rerun()
                with c2:
                    if st.button("✗", key=skipped_key, help="I skipped"):
                        store.log_interaction(
                            track["id"], track["name"], track["artist"],
                            "skipped", model_result["churn_probability"],
                        )
                        st.session_state[done_key] = True
                        st.rerun()

        st.divider()


# ── Helper: feedback loop tab ─────────────────────────────────────────────────

def _feedback_loop_tab():
    store = FeedbackStore()
    stats = store.get_stats()
    interactions = store.get_interactions()

    if stats["total"] == 0:
        st.info(
            "No feedback logged yet. Analyse your listening, get recommendations, "
            "then click ✓ or ✗ on each track. The system learns from your responses."
        )
        _feedback_explainer()
        return

    # Stats row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Interactions", stats["total"])
    c2.metric("Listened", stats["listened"])
    c3.metric("Skipped", stats["skipped"])
    c4.metric("Success Rate", f"{stats['success_rate']:.0%}")
    c5.metric("Auto-detected", stats.get("auto_detected", 0),
              help="Outcomes detected automatically by re-fetching recently-played")

    # Trend chart
    if len(stats["trend"]) > 1:
        trend_df = pd.DataFrame(stats["trend"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=trend_df["index"],
            y=trend_df["success_rate"],
            mode="lines+markers",
            line=dict(color="#1db954", width=2),
            marker=dict(size=6),
            name="Rolling success rate",
            hovertemplate="Interaction %{x}: %{y:.0%}<extra></extra>",
        ))
        fig.add_hline(y=0.5, line_dash="dash", line_color="#718096",
                      annotation_text="50% baseline")
        fig.update_layout(
            title="Recommendation Success Rate (rolling 5)",
            plot_bgcolor="#0d1117",
            paper_bgcolor="#0d1117",
            font_color="#e2e8f0",
            xaxis=dict(title="Interaction #", gridcolor="#1e2533"),
            yaxis=dict(title="Success Rate", range=[0, 1], gridcolor="#1e2533"),
            height=280,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Interaction log
    st.markdown("#### Interaction Log")
    log_rows = [
        {
            "Track": i["track_name"],
            "Artist": i["artist"],
            "Outcome": "✓ Listened" if i["outcome"] == "listened" else "✗ Skipped",
            "Churn Risk at Time": f"{i['churn_prob']:.0%}",
            "Timestamp": i["timestamp"][:19].replace("T", " "),
        }
        for i in reversed(interactions)
    ]
    st.dataframe(pd.DataFrame(log_rows), use_container_width=True, hide_index=True)

    if st.button("Clear feedback history", type="secondary"):
        store.clear()
        st.rerun()

    _feedback_explainer()


def _feedback_explainer():
    st.markdown("---")
    st.markdown("""
    <div class="section-header">How the agentic loop works</div>
    <div style='font-size:0.88rem; color:#a0aec0; line-height:1.9'>
    <b>1. Sensor</b> — XGBoost detects disengagement from 7 engineered listening signals<br>
    <b>2. Agent</b> — Gemini calls 3 tools (genre entropy · mood trajectory · discovery health),
    reasons across their outputs, then produces a diagnosis + strategy<br>
    <b>3. Action</b> — Spotify API retrieval + cosine similarity executes the strategy → top-3 tracks<br>
    <b>4. Loop (manual)</b> — Your ✓/✗ feedback scores each track immediately<br>
    <b>5. Loop (auto)</b> — On next session, recently-played is re-fetched and compared
    against pending recommendations — outcomes logged without any user action<br><br>
    Future rankings blend <b>cosine similarity (70%)</b> + <b>learned feedback signal (30%)</b>.<br>
    The system improves from its own actions. That is what <em>agentic</em> means, stripped of the buzzword.
    </div>
    """, unsafe_allow_html=True)


# ── Main: landing page ────────────────────────────────────────────────────────

def _landing():
    st.markdown("""
    <div style='text-align:center; padding: 3rem 0 2rem'>
      <div style='font-size:3rem'>🎵</div>
      <h1 style='font-size:2.2rem; margin-bottom:0.3rem'>Spotify Listening Intelligence Agent</h1>
      <p style='color:#a0aec0; font-size:1.1rem'>
        Classical ML + LLM Reasoning + Agentic Feedback Loop
      </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, icon, title, body in [
        (c1, "📊", "Module 1 — Signal Layer",
         "7 engineered features from your listening history. XGBoost + SHAP. Probability calibrated."),
        (c2, "🧠", "Module 2 — Reasoning Layer",
         "Top SHAP drivers packaged into a structured prompt. Gemini returns a plain-English diagnosis."),
        (c3, "🎯", "Module 3 — Action Layer",
         "LLM sets strategy. Spotify API + cosine similarity executes it. Top-3 ranked tracks."),
        (c4, "🔄", "Module 4 — Feedback Loop",
         "Log outcomes. Scores blend into future rankings. The system improves from its own actions."),
    ]:
        with col:
            st.markdown(f"""
            <div class="intel-card">
              <div style='font-size:1.8rem'>{icon}</div>
              <div style='font-weight:700; margin:0.4rem 0'>{title}</div>
              <div style='font-size:0.88rem; color:#a0aec0'>{body}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.info("Connect Spotify via the sidebar to begin. Optionally upload your Extended Streaming History for full feature accuracy.")


# ── Main: run analysis ────────────────────────────────────────────────────────

def _run_analysis(uploaded_files):
    token = st.session_state["spotify_token"]

    with st.status("Running analysis pipeline…", expanded=True) as status:
        # Step 1: Collect API data
        st.write("📡 Fetching Spotify profile and listening data…")
        user_profile = collect_user_data(token)
        st.session_state["user_profile"] = user_profile

        # Step 2: Feature engineering
        if uploaded_files:
            st.write("📂 Parsing Extended Streaming History files…")
            history_df = load_history_files(uploaded_files)
            if not history_df.empty:
                features = extract_features_from_history(history_df)
                st.session_state["history_mode"] = True
                st.write(f"   → {len(history_df):,} plays found, {len(features)} features computed")
            else:
                st.write("   ⚠️ Could not parse history files, falling back to API features")
                features = extract_features_from_api(user_profile)
                st.session_state["history_mode"] = False
        else:
            st.write("📊 Deriving features from Spotify API data…")
            features = extract_features_from_api(user_profile)
            st.session_state["history_mode"] = False

        st.session_state["features"] = features

        # Step 3: XGBoost + SHAP
        st.write("🌲 Loading XGBoost model and computing SHAP values…")
        load_models()  # warm up / train if first run
        model_result = predict(features)
        st.session_state["model_result"] = model_result
        st.write(
            f"   → Churn probability: **{model_result['churn_probability']:.1%}** "
            f"({model_result['risk_level']} risk)"
        )

        # Step 4: LLM agent — multi-step tool calling
        st.write("🧠 Running multi-step Gemini agent (3 tool calls)…")
        diagnosis, chain = generate_diagnosis(model_result, user_profile)
        st.session_state["diagnosis"] = diagnosis
        st.session_state["agent_chain"] = chain
        if diagnosis.get("_fallback"):
            st.write("   ⚠️ Gemini unavailable, using rule-based fallback")
        else:
            tools_called = [s["tool"] for s in chain]
            st.write(f"   → Tools called: {', '.join(tools_called) if tools_called else 'none (fallback)'}")
            st.write(f"   → Urgency: {diagnosis.get('urgency', 'unknown')}")

        # Step 5: Recommendations
        st.write("🎯 Retrieving and ranking recommendations…")
        recs = get_recommendations(diagnosis, user_profile, token)
        st.session_state["recommendations"] = recs
        st.write(f"   → {len(recs)} tracks ranked and returned")

        # Store recommendations as pending for auto-feedback loop
        store = FeedbackStore()
        prob = model_result["churn_probability"]
        for rec in recs:
            store.store_pending(rec["id"], rec["name"], rec["artist"], prob)

        status.update(label="Analysis complete ✓", state="complete")


# ── Main: results view ────────────────────────────────────────────────────────

def _show_results():
    model_result: dict = st.session_state["model_result"]
    diagnosis: dict = st.session_state["diagnosis"]
    recommendations: list = st.session_state["recommendations"]
    user_profile: dict = st.session_state["user_profile"]
    history_mode: bool = st.session_state.get("history_mode", False)
    agent_chain: list = st.session_state.get("agent_chain", [])

    # ── Auto-feedback notification banner ────────────────────────────────────
    auto_outcomes = st.session_state.get("auto_outcomes", [])
    if auto_outcomes:
        names = ", ".join(f"**{o['track_name']}**" for o in auto_outcomes[:3])
        st.success(
            f"🔄 Auto-feedback loop detected: {names} "
            f"{'was' if len(auto_outcomes) == 1 else 'were'} listened to since last session "
            f"— outcomes logged automatically.",
            icon="✅",
        )
        st.session_state["auto_outcomes"] = []  # dismiss after showing

    if not history_mode:
        st.warning(
            "⚡ API-only mode: features are approximated. "
            "Upload your Extended Streaming History for full accuracy.",
            icon="⚠️",
        )

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Risk Intelligence",
        "🧠 AI Diagnosis",
        "🎯 Recommendations",
        "🔄 Feedback Loop",
    ])

    # ── Tab 1: Risk Intelligence ──────────────────────────────────────────────
    with tab1:
        prob = model_result["churn_probability"]
        risk = model_result["risk_level"]
        risk_css = risk.lower()

        col_gauge, col_summary = st.columns([1, 2])
        with col_gauge:
            st.plotly_chart(_risk_gauge(prob, risk), use_container_width=True)

        with col_summary:
            st.markdown(f"""
            <div class="intel-card risk-{risk_css}">
              <div class="section-header">Engagement Status</div>
              <div style='font-size:1.6rem; font-weight:700; margin-bottom:0.4rem'>
                {risk} Risk
              </div>
              <div style='color:#a0aec0; font-size:0.92rem; line-height:1.7'>
                Calibrated churn probability: <b>{prob:.1%}</b><br>
                Top signal: <b>{model_result['top_drivers'][0][0]}</b>
                  (SHAP {model_result['top_drivers'][0][1]:+.3f})<br>
                User: <b>{user_profile.get('display_name', '')}</b> ·
                {user_profile.get('followers', 0):,} followers ·
                {len(user_profile.get('top_artists', []))} tracked artists
              </div>
            </div>
            """, unsafe_allow_html=True)

            top_genres = user_profile.get("top_genres", [])[:5]
            if top_genres:
                st.markdown(
                    " ".join(
                        f'<span class="pill pill-green">{g}</span>'
                        for g in top_genres
                    ),
                    unsafe_allow_html=True,
                )

        st.markdown("#### SHAP Feature Contributions")
        st.plotly_chart(
            _shap_waterfall(
                model_result["shap_values"],
                model_result["base_value"],
                prob,
            ),
            use_container_width=True,
        )

        with st.expander("All feature values and SHAP breakdown"):
            _feature_table(model_result["feature_values"], model_result["shap_values"])

    # ── Tab 2: AI Diagnosis ───────────────────────────────────────────────────
    with tab2:
        urgency_color = {
            "act_now": "#ff4b4b",
            "act_soon": "#ffa500",
            "monitor": "#00c851",
        }.get(diagnosis.get("urgency", "monitor"), "#718096")

        urgency_label = {
            "act_now": "🔴 Act Now",
            "act_soon": "🟡 Act Soon",
            "monitor": "🟢 Monitor",
        }.get(diagnosis.get("urgency", "monitor"), "⚪ Unknown")

        col_badge, _ = st.columns([1, 3])
        with col_badge:
            st.markdown(
                f'<div style="background:{urgency_color}22; border:1px solid {urgency_color}; '
                f'border-radius:8px; padding:8px 16px; display:inline-block; '
                f'font-weight:700; color:{urgency_color}">{urgency_label}</div>',
                unsafe_allow_html=True,
            )

        st.markdown("#### Diagnosis")
        st.markdown(
            f'<div class="intel-card">'
            f'<div class="diagnosis-text">{diagnosis.get("diagnosis", "")}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown("#### Hypothesis")
        st.markdown(
            f'<div class="intel-card">'
            f'<div class="hypothesis-text">💭 {diagnosis.get("hypothesis", "")}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown("#### Re-engagement Strategy")
        st.markdown(
            f'<div class="intel-card" style="border-left:4px solid #1db954">'
            f'<div class="diagnosis-text">🎯 {diagnosis.get("strategy", "")}</div>'
            f'<div style="margin-top:0.6rem">'
            f'<span class="pill pill-green">Genre: {diagnosis.get("strategy_genre", "—")}</span>'
            f'<span class="pill pill-orange">Artist: {diagnosis.get("strategy_artist", "—")}</span>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        if diagnosis.get("_fallback"):
            st.caption("ℹ️ Gemini unavailable — rule-based fallback used. Add GEMINI_API_KEY to .env for the full agent.")

        # ── Agent reasoning chain ─────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 🔬 Agent Reasoning Chain")

        if not agent_chain:
            st.caption("No tool calls recorded (fallback mode or first run without Gemini).")
        else:
            tool_icons = {
                "analyze_genre_entropy": "🎵",
                "analyze_mood_trajectory": "🌡️",
                "evaluate_discovery_health": "🔭",
            }
            tool_labels = {
                "analyze_genre_entropy": "Genre Entropy Analysis",
                "analyze_mood_trajectory": "Mood Trajectory Analysis",
                "evaluate_discovery_health": "Discovery Health Evaluation",
            }

            for i, step in enumerate(agent_chain):
                icon = tool_icons.get(step["tool"], "🔧")
                label = tool_labels.get(step["tool"], step["tool"])
                result = step["result"]

                with st.expander(f"Step {i + 1} — {icon} `{label}`", expanded=(i == 0)):
                    col_meta, col_result = st.columns([1, 2])

                    with col_meta:
                        st.markdown(f"**Tool:** `{step['tool']}`")
                        if step.get("args"):
                            st.markdown("**Called with:**")
                            for k, v in step["args"].items():
                                st.markdown(f"- `{k}`: `{v}`")

                        # Key metric highlight
                        if "severity" in result:
                            color = {"severe": "#ff4b4b", "moderate": "#ffa500", "mild": "#00c851"}.get(result["severity"], "#718096")
                            st.markdown(
                                f'<span class="pill" style="background:{color}22;color:{color};border:1px solid {color}">'
                                f'{result["severity"].upper()}</span>',
                                unsafe_allow_html=True,
                            )
                        if "health_label" in result:
                            color = {"critical": "#ff4b4b", "poor": "#ffa500", "fair": "#f6e05e", "good": "#00c851"}.get(result["health_label"], "#718096")
                            st.markdown(
                                f'<span class="pill" style="background:{color}22;color:{color};border:1px solid {color}">'
                                f'Discovery: {result["health_label"].upper()}</span>',
                                unsafe_allow_html=True,
                            )
                        if "mood_label" in result:
                            st.markdown(f'<span class="pill pill-orange">{result["mood_label"]}</span>', unsafe_allow_html=True)

                    with col_result:
                        st.markdown(f"*{result.get('interpretation', '')}*")
                        if result.get("recommendation_hint"):
                            st.markdown(
                                f'<div style="background:#1a2a1a;border-left:3px solid #1db954;'
                                f'padding:8px 12px;border-radius:4px;font-size:0.85rem;color:#a0aec0;margin-top:8px">'
                                f'💡 {result["recommendation_hint"]}</div>',
                                unsafe_allow_html=True,
                            )

        # ── Architecture note ─────────────────────────────────────────────────
        with st.expander("Architecture note (interview talking point)"):
            st.markdown(f"""
            **The LLM never sees raw Spotify data.** The agent reasons over tool outputs only:

            | Step | What happens |
            |------|-------------|
            | 1 | XGBoost computes churn probability + SHAP values from raw features |
            | 2 | Agent receives only: probability, top SHAP drivers, user genre/artist context |
            | 3 | Agent calls `analyze_genre_entropy` → gets entropy severity |
            | 4 | Agent calls `analyze_mood_trajectory` → gets mood quadrant + listen depth |
            | 5 | Agent calls `evaluate_discovery_health` → gets discovery score |
            | 6 | Agent synthesises across all three tool outputs → final diagnosis |

            Each tool is executed in Python with real data. The LLM interprets structured
            intermediate representations — it **cannot hallucinate** about data it never saw.
            This is the separation of concerns that makes it rigorous, not a wrapper.
            """)
            st.code(f"""# What the agent received as context
Churn Probability : {prob:.1%}
Risk Level        : {risk}
Top SHAP drivers  :
{chr(10).join(f"  {FEATURE_LABELS.get(f,f)}: {model_result['feature_values'].get(f,0):.3f} (SHAP={v:+.3f})" for f,v in model_result['top_drivers'])}
User genres       : {', '.join(user_profile.get('top_genres', [])[:5])}
User artists      : {', '.join(a['name'] for a in user_profile.get('top_artists', [])[:5])}
""", language="text")

    # ── Tab 3: Recommendations ────────────────────────────────────────────────
    with tab3:
        if not recommendations:
            st.warning("No recommendations returned. Check Spotify API credentials.")
        else:
            st.markdown(
                f"*Strategy: {diagnosis.get('strategy', '')}*",
                help="Generated by the LLM, executed by content-based retrieval",
            )
            st.markdown("---")
            _recommendation_cards(recommendations, model_result)

            with st.expander("How ranking works"):
                st.markdown("""
                **Score = 0.7 × cosine_similarity + 0.3 × tanh(feedback_score)**

                - `cosine_similarity`: audio feature vector of the track vs your all-time listen profile
                  (energy, valence, danceability, acousticness, instrumentalness)
                - `feedback_score`: cumulative score from your ✓/✗ clicks
                  (+1.0 for listened, −0.5 for skipped)
                - `tanh()` squashes the feedback score to prevent outlier tracks from dominating
                - Artist diversity enforced: no two recommendations from the same artist
                """)

    # ── Tab 4: Feedback Loop ──────────────────────────────────────────────────
    with tab4:
        _feedback_loop_tab()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    if "spotify_token" not in st.session_state:
        _landing()
        return

    if analyze_btn:
        for key in ["model_result", "diagnosis", "recommendations", "features"]:
            st.session_state.pop(key, None)
        _run_analysis(uploaded_files)

    if "model_result" in st.session_state:
        _show_results()
    elif not analyze_btn:
        user_name = st.session_state.get("user_profile", {}).get("display_name", "")
        st.markdown(f"## {'Welcome back, ' + user_name + '!' if user_name else 'Ready to analyse.'}")
        st.info("Click **▶ Run Analysis** in the sidebar to start.")
        _feedback_loop_tab()


main()
