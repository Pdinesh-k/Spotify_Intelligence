import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import SPOTIFY_CLIENT_ID, CHURN_RISK_THRESHOLD
from ml.features import (
    FEATURE_LABELS,
    FEATURE_DESCRIPTIONS,
    extract_features_from_api,
    features_to_array,
)
from ml.feedback import FeedbackStore
from ml.model import load_models, predict
from spotify.auth import get_oauth_url
from spotify.collector import collect_user_data
from agents.diagnosis import generate_diagnosis
from agents.recommender import get_recommendations
from agents.auto_feedback import run_auto_feedback
from ml import analytics as analytics_engine


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
    analyze_btn = st.button(
        "▶ Run Analysis",
        use_container_width=True,
        type="primary",
        disabled="spotify_token" not in st.session_state,
    )

    st.divider()
    st.markdown("""
    <div style='font-size:0.78rem; color:#4a5568; line-height:1.6'>
    <b>How it works:</b><br>
    Step 1 → Deep Habit Analysis<br>
    Step 2 → Deep Insights Engine<br>
    Step 3 → Curated Discovery<br>
    Step 4 → Continuous Learning<br>
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
    <div class="section-header">How our Continuous Learning loop works</div>
    <div style='font-size:0.88rem; color:#a0aec0; line-height:1.9'>
    <b>1. Analyze</b> — We detect patterns in your daily listening habits.<br>
    <b>2. Insight</b> — Our engine figures out your mood, your genre diversity, and your discovery rate.<br>
    <b>3. Curate</b> — We search high and low to find the top 3 tracks to perfectly match your profile.<br>
    <b>4. Tune (Manual)</b> — Your ✓/✗ feedback scores each track immediately to customize your profile.<br>
    <b>5. Tune (Auto)</b> — Simply listen on Spotify! We check your history and automatically log successful recommendations.<br><br>
    Recommendations are uniquely tailored just for you. As you listen, your personalized profile adapts.<br>
    The system gets smarter every time you use it.
    </div>
    """, unsafe_allow_html=True)


# ── Main: landing page ────────────────────────────────────────────────────────

def _landing():
    st.markdown("""
    <div style='text-align:center; padding: 3rem 0 2rem'>
      <div style='font-size:3rem'>
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#1db954" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 18V5l12-2v13"></path><circle cx="6" cy="18" r="3"></circle><circle cx="18" cy="16" r="3"></circle></svg>
      </div>
      <h1 style='font-size:2.2rem; margin-bottom:0.3rem'>Listening Intelligence</h1>
      <p style='color:#a0aec0; font-size:1.1rem'>
        Understand your listening habits. Find music that fits.
      </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, icon_svg, title, body in [
        (c1, '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#1db954" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>', "Habit Tracking", "See how often you skip tracks, when you listen, and how your genre tastes evolve over time."),
        (c2, '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#1db954" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><path d="M8 14s1.5 2 4 2 4-2 4-2"></path><line x1="9" y1="9" x2="9.01" y2="9"></line><line x1="15" y1="9" x2="15.01" y2="9"></line></svg>', "Mood Mapping", "Understand the underlying energy and emotional tone of your recent plays."),
        (c3, '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#1db954" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 18V5l12-2v13"></path><circle cx="6" cy="18" r="3"></circle><circle cx="18" cy="16" r="3"></circle></svg>', "Fresh Tracks", "Get recommendations that actually make sense for the phase you're going through."),
        (c4, '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#1db954" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.92-10.44l5.36 2.87"></path></svg>', "Feedback Loop", "Tell us what you like, and the rankings adjust instantly to fit your taste."),
    ]:
        with col:
            st.markdown(f"""
            <div class="intel-card">
              <div style='margin-bottom:0.5rem;'>{icon_svg}</div>
              <div style='font-weight:700; margin:0.4rem 0'>{title}</div>
              <div style='font-size:0.88rem; color:#a0aec0'>{body}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.info("Connect Spotify via the sidebar to get started. Your analysis is powered entirely by the Spotify API.")


# ── Main: run analysis ────────────────────────────────────────────────────────

def _run_analysis():
    token = st.session_state["spotify_token"]

    with st.status("Running analysis pipeline…", expanded=True) as status:
        # Step 1: Collect API data
        st.write("📡 Fetching Spotify profile and listening data…")
        user_profile = collect_user_data(token)
        st.session_state["user_profile"] = user_profile

        # Step 2: Feature engineering — pure API
        st.write("📊 Deriving features from Spotify API data…")
        features = extract_features_from_api(user_profile)

        # Compute listening stats
        recent_played = user_profile.get("recently_played", [])
        recent_ms = sum(t.get("ms_played", 0) or 0 for t in recent_played)
        recent_hours = recent_ms / 3_600_000.0
        top_tracks_recent = user_profile.get("top_tracks_recent", [])
        top_tracks_alltime = user_profile.get("top_tracks_alltime", [])
        overlap = len({t["name"] for t in top_tracks_recent} & {t["name"] for t in top_tracks_alltime})
        obsession_rate = overlap / max(len(top_tracks_recent), 1)
        listening_stats = {
            "recent_hours": round(recent_hours, 2),
            "obsession_rate": round(obsession_rate, 2),
            "total_recent_tracks": len(recent_played),
        }
        user_profile["listening_stats"] = listening_stats
        st.session_state["listening_stats"] = listening_stats
        st.session_state["user_profile"] = user_profile

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

        # Step 6: Compute full analytics
        st.write("🔬 Computing music profile analytics…")
        analytics_data = analytics_engine.compute_all(user_profile)
        st.session_state["analytics"] = analytics_data

        status.update(label="Analysis complete ✓", state="complete")


# ── Main: results view ────────────────────────────────────────────────────────

def _show_results():
    model_result: dict = st.session_state["model_result"]
    diagnosis: dict = st.session_state["diagnosis"]
    recommendations: list = st.session_state["recommendations"]
    user_profile: dict = st.session_state["user_profile"]
    listening_stats: dict = st.session_state.get("listening_stats", {})
    agent_chain: list = st.session_state.get("agent_chain", [])
    analytics_data: dict = st.session_state.get("analytics", {})

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

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Risk Intelligence",
        "📈 Music Profile",
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

        # ── Listening Analytics Block ──────────────────────────────────────
        st.markdown("#### Your Listening Analytics")
        recent_hours = listening_stats.get("recent_hours", 0.0)
        obsession_pct = listening_stats.get("obsession_rate", 0.0)
        total_tracks = listening_stats.get("total_recent_tracks", 0)
        top_artists = user_profile.get("top_artists", [])
        top_genres_list = user_profile.get("top_genres", [])
        repeat_ratio = model_result["feature_values"].get("repeat_play_ratio", 0.0)
        skip_trend = model_result["feature_values"].get("skip_rate_trend", 0.0)
        days_new = model_result["feature_values"].get("days_new_artist", 7.0)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "Recent Listening",
            f"{recent_hours:.1f} hrs",
            help="Estimated from your last 50 Spotify plays"
        )
        m2.metric(
            "Obsession Rate",
            f"{obsession_pct:.0%}",
            help="How many of your short-term favourites also appear in your all-time top tracks"
        )
        m3.metric(
            "Repeat Ratio",
            f"{repeat_ratio:.0%}",
            delta=f"{repeat_ratio - 0.3:+.0%} vs avg",
            help="Fraction of recently played songs you've heard before"
        )
        m4.metric(
            "Skip Trend",
            f"{skip_trend:+.0%}",
            delta_color="inverse",
            help="Change in skip rate vs prior period"
        )

        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Tracks Analysed", total_tracks)
        m6.metric("Days Since New Artist", f"{days_new:.0f}")
        m7.metric("Top Genres Tracked", len(top_genres_list))
        m8.metric("Artists in Profile", len(top_artists))

    # ── Tab 2: Music Profile (Data Science) ──────────────────────────────────
    with tab2:
        _music_profile_tab()

    # ── Tab 3: AI Diagnosis ──────────────────────────────────────────────────
    with tab3:
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
        with st.expander("Under the hood: How we create your insights"):
            st.markdown(f"""
            **We prioritize your privacy above all else.** The insight generator only sees aggregated trends from your listening data rather than raw sensitive information.

            | Step | What happens |
            |------|-------------|
            | 1 | We calculate your current engagement probability from 7 key listening habits. |
            | 2 | Our system receives only: probability, top factors, and the names of your top genres. |
            | 3 | We analyze genre diversity across your profile. |
            | 4 | We calculate an emotional trajectory based on track audio features (like valence and energy). |
            | 5 | We evaluate how often you've historically discovered new artists. |
            | 6 | A custom synthesis engine safely merges these trends to form a complete, accurate diagnosis just for you. |
            
            This allows us to maintain a pristine, accurate recommendation pipeline while fully protecting your raw Spotify data.
            """)
            st.code(f"""# Encrypted Payload Example
Engagement Probability : {prob:.1%}
Current State        : {risk}
Key Influencing Factors  :
{chr(10).join(f"  {FEATURE_LABELS.get(f,f)}: {model_result['feature_values'].get(f,0):.3f} (Profile Weight={v:+.3f})" for f,v in model_result['top_drivers'])}
User Core Genres       : {', '.join(user_profile.get('top_genres', [])[:5])}
User Core Artists      : {', '.join(a['name'] for a in user_profile.get('top_artists', [])[:5])}
""", language="text")

    # ── Tab 4: Recommendations ───────────────────────────────────────────────
    with tab4:
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

    # ── Tab 2: Music Profile (Data Science) ──────────────────────────────────
    with tab2:
        _music_profile_tab()

    # ── Tab 5: Feedback Loop ──────────────────────────────────────────────────
    with tab5:
        _feedback_loop_tab()



# ── Music Profile Tab ─────────────────────────────────────────────────────────

def _music_profile_tab():
    """Full data science analytics dashboard."""
    user_profile: dict = st.session_state.get("user_profile", {})
    an: dict = st.session_state.get("analytics", {})
    if not an:
        st.info("Run an analysis to see your full music profile.")
        return

    temporal   = an.get("temporal", {})
    velocity   = an.get("velocity", {})
    loyalty    = an.get("loyalty", {})
    genre_p    = an.get("genre_profile", {})
    trajectory = an.get("trajectory", {})
    discovery  = an.get("discovery", {})

    # ── Row 1: Key metrics ────────────────────────────────────────────────────
    st.markdown("### Your Music DNA")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Listening Velocity", f"{velocity.get('tracks_per_hour',0):.1f} trk/hr",
              help="Tracks consumed per hour across your last 50 plays")
    c2.metric("Taste Stability",
              f"{trajectory.get('short_to_long_sim',0):.0%}",
              help="How similar your recent taste is to your all-time taste")
    c3.metric("Artist Loyalty",
              f"{loyalty.get('loyalty_score',0):.0%}",
              help="Overlap between your recent vs all-time top artists")
    c4.metric("Genre Diversity",
              f"{genre_p.get('diversity_score',0):.2f} bits",
              help="Shannon entropy of your genre distribution. Higher = broader.")
    c5.metric("Discovery Rate",
              f"{discovery.get('discovery_rate',0):.0%}",
              help="Fraction of recent plays from artists not in your all-time chart")

    st.markdown("---")

    # ── Row 2a: Listening Heatmap ─────────────────────────────────────────────
    st.markdown("#### When You Listen")
    col_heat, col_genre = st.columns([3, 2])

    with col_heat:
        hours = temporal.get("hours", [0]*24)
        peak_h = temporal.get("peak_hour_label", "")
        max_h = max(hours) if hours and max(hours) > 0 else 1

        fig_heat = go.Figure(go.Bar(
            x=list(range(24)),
            y=hours,
            marker_color=[
                f"rgba(29,185,84,{max(0.15, v/max_h)})" for v in hours
            ],
            hovertemplate="%{x}:00 — %{y} plays<extra></extra>",
        ))
        fig_heat.update_layout(
            title=f"Plays by Hour · Peak: {peak_h}",
            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
            font_color="#e2e8f0", height=220,
            xaxis=dict(title="Hour of Day", tickvals=list(range(0, 24, 3)),
                       ticktext=[f"{h}h" for h in range(0, 24, 3)], gridcolor="#1e2533"),
            yaxis=dict(title="Plays", gridcolor="#1e2533"),
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        days = temporal.get("days", [0]*7)
        peak_d = temporal.get("peak_day", 0)
        day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        fig_dow = go.Figure(go.Bar(
            x=day_labels, y=days,
            marker_color=["#1db954" if d == peak_d else "#2d3748" for d in range(7)],
            hovertemplate="%{x}: %{y} plays<extra></extra>",
        ))
        fig_dow.update_layout(
            title=f"Plays by Day · Peak: {temporal.get('peak_day_label','')}",
            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
            font_color="#e2e8f0", height=200,
            yaxis=dict(gridcolor="#1e2533"),
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig_dow, use_container_width=True)

    with col_genre:
        st.markdown("#### Genre Distribution")
        g_labels = genre_p.get("genre_labels", [])
        g_values = genre_p.get("genre_values", [])
        if g_labels:
            fig_donut = go.Figure(go.Pie(
                labels=g_labels, values=g_values,
                hole=0.55,
                marker_colors=[
                    "#1db954","#1aa34a","#158a3e","#107032",
                    "#0b5726","#ffa500","#ff6b35","#e84393"
                ][:len(g_labels)],
                textinfo="label",
                hovertemplate="%{label}: %{value}<extra></extra>",
            ))
            fig_donut.update_layout(
                paper_bgcolor="#0d1117", font_color="#e2e8f0",
                height=430, showlegend=False,
                annotations=[dict(
                    text=f"{genre_p.get('total_unique',0)}<br>genres",
                    x=0.5, y=0.5, font_size=16, showarrow=False,
                    font_color="#e2e8f0",
                )],
                margin=dict(l=10, r=10, t=20, b=10),
            )
            st.plotly_chart(fig_donut, use_container_width=True)
        else:
            st.info("Not enough genre data from Spotify API.")

    st.markdown("---")

    # ── Row 3: Taste Trajectory + Loyalty ────────────────────────────────────
    col_traj, col_loyal = st.columns(2)

    with col_traj:
        st.markdown("#### Taste Trajectory")
        traj_color = trajectory.get("trajectory_color", "#718096")
        traj_label = trajectory.get("trajectory", "Unknown")
        traj_desc  = trajectory.get("trajectory_desc", "")
        st.markdown(
            f'<div style="background:{traj_color}22;border:1px solid {traj_color};'
            f'border-radius:10px;padding:12px 16px;margin-bottom:1rem">'
            f'<span style="font-size:1.2rem;font-weight:700;color:{traj_color}">{traj_label}</span><br>'
            f'<span style="color:#a0aec0;font-size:0.9rem">{traj_desc}</span></div>',
            unsafe_allow_html=True,
        )
        sim_labels = ["Short→Medium", "Medium→Long", "Short→Long (Overall)"]
        sim_vals = [
            trajectory.get("short_to_medium_sim", 0) * 100,
            trajectory.get("medium_to_long_sim", 0) * 100,
            trajectory.get("short_to_long_sim", 0) * 100,
        ]
        fig_traj = go.Figure(go.Bar(
            x=sim_labels, y=sim_vals,
            marker_color=["#1db954", "#ffa500", "#e84393"],
            text=[f"{v:.0f}%" for v in sim_vals], textposition="outside",
            hovertemplate="%{x}: %{y:.1f}% overlap<extra></extra>",
        ))
        fig_traj.update_layout(
            title="Taste Similarity Across Time Windows",
            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
            font_color="#e2e8f0", height=270,
            yaxis=dict(range=[0, 110], gridcolor="#1e2533", title="% Overlap"),
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig_traj, use_container_width=True)
        st.caption(
            f"Popularity trend: **{trajectory.get('popularity_trend','')}** · "
            f"Recent avg: {trajectory.get('avg_popularity_short',0)} · "
            f"All-time avg: {trajectory.get('avg_popularity_long',0)}"
        )

    with col_loyal:
        st.markdown("#### Artist Loyalty")
        l_score = loyalty.get("loyalty_score", 0)
        l_label = loyalty.get("loyalty_label", "")
        l_desc  = loyalty.get("loyalty_desc", "")
        l_color = "#1db954" if l_score > 0.5 else "#ffa500" if l_score > 0.25 else "#e84393"

        fig_loyal = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(l_score*100, 1),
            number={"suffix": "%", "font": {"size": 28, "color": l_color}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#718096"},
                "bar": {"color": l_color},
                "bgcolor": "#1a1a2e", "bordercolor": "#2d3748",
                "steps": [
                    {"range": [0, 33],  "color": "#2a1e08"},
                    {"range": [33, 66], "color": "#1a2a0a"},
                    {"range": [66, 100],"color": "#0a2a14"},
                ],
            },
            title={"text": l_label, "font": {"color": "#a0aec0", "size": 13}},
        ))
        fig_loyal.update_layout(
            paper_bgcolor="#0d1117", font_color="#e2e8f0",
            height=210, margin=dict(l=20, r=20, t=10, b=5),
        )
        st.plotly_chart(fig_loyal, use_container_width=True)
        st.caption(l_desc)

        rising  = loyalty.get("rising_artists", [])
        falling = loyalty.get("falling_artists", [])
        stable  = loyalty.get("stable_artists", [])
        if rising:
            st.markdown("🚀 **Rising:** " + " · ".join(f"`{a}`" for a in rising[:4]))
        if falling:
            st.markdown("📉 **Fading:** " + " · ".join(f"`{a}`" for a in falling[:4]))
        if stable:
            st.markdown("🔒 **Constants:** " + " · ".join(f"`{a}`" for a in stable[:4]))

    st.markdown("---")

    # ── Row 4: Artist Timeline ────────────────────────────────────────────────
    st.markdown("#### Your Artist Timeline (Last 4 weeks → 6 months → All time)")
    short_a  = [a["name"] for a in user_profile.get("top_artists_short",  [])[:10]]
    medium_a = [a["name"] for a in user_profile.get("top_artists_medium", [])[:10]]
    long_a   = [a["name"] for a in user_profile.get("top_artists_long",   [])[:10]]
    stable_set = set(loyalty.get("stable_artists", []))

    col_s, col_m, col_l = st.columns(3)
    with col_s:
        st.markdown("**🔥 Now (4 weeks)**")
        for i, a in enumerate(short_a, 1):
            badge = "🆕" if a not in long_a else ("✅" if a in stable_set else "")
            st.markdown(f"{i}. {badge} {a}")
    with col_m:
        st.markdown("**📅 6 Months**")
        for i, a in enumerate(medium_a, 1):
            st.markdown(f"{i}. {a}")
    with col_l:
        st.markdown("**🏆 All Time**")
        for i, a in enumerate(long_a, 1):
            fade = "📉" if a not in short_a else ""
            st.markdown(f"{i}. {fade} {a}")

    st.markdown("---")

    # ── Row 5: Discovery Funnel ───────────────────────────────────────────────
    st.markdown("#### Artists You Haven't Explored Yet")
    undiscovered = discovery.get("undiscovered_artists", [])
    st.caption(
        f"Based on who your top artists are related to — "
        f"{len(undiscovered)} new artists flagged for you."
    )
    if undiscovered:
        cols = st.columns(4)
        for i, artist in enumerate(undiscovered[:8]):
            with cols[i % 4]:
                genres = artist.get("genres", [])
                genre_str = ", ".join(genres[:2]) if genres else "Unknown genre"
                st.markdown(
                    f'<div class="intel-card" style="border-left:3px solid #1db954">'
                    f'<div style="font-weight:600">{artist["name"]}</div>'
                    f'<div style="color:#718096;font-size:0.8rem">{genre_str}</div>'
                    f'<div style="color:#a0aec0;font-size:0.78rem;margin-top:4px">'
                    f'Related to <b>{artist.get("anchor_artist","")}</b></div>'
                    f'<div style="color:#1db954;font-size:0.8rem">Popularity: {artist["popularity"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
    else:
        st.info("No undiscovered related artists found. You might already know everyone!")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    if "spotify_token" not in st.session_state:
        _landing()
        return

    if analyze_btn:
        for key in ["model_result", "diagnosis", "recommendations", "features", "listening_stats"]:
            st.session_state.pop(key, None)
        _run_analysis()

    if "model_result" in st.session_state:
        _show_results()
    elif not analyze_btn:
        user_name = st.session_state.get("user_profile", {}).get("display_name", "")
        st.markdown(f"## {'Welcome back, ' + user_name + '!' if user_name else 'Ready to analyse.'}")
        st.info("Click **▶ Run Analysis** in the sidebar to start.")
        _feedback_loop_tab()


main()
