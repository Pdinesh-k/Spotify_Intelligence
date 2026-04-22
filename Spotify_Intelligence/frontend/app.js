/* ── State ─────────────────────────────────────────────── */
const API = '';
let token = localStorage.getItem('si_token');
let analysisData = null;
let feedbackLogged = {};

const FEATURE_LABELS = {
  skip_rate_trend:    'Skip Rate Trend',
  session_freq_delta: 'Session Frequency Δ',
  listen_depth:       'Listen Depth',
  genre_entropy_drop: 'Genre Entropy Drop',
  time_of_day_shift:  'Time-of-Day Shift',
  days_new_artist:    'Days Since New Artist',
  repeat_play_ratio:  'Repeat Play Ratio',
};

const TOOL_LABELS = {
  analyze_genre_entropy:    { icon: '🎵', label: 'Genre Analysis' },
  analyze_mood_trajectory:  { icon: '🌡️', label: 'Mood Trajectory' },
  evaluate_discovery_health:{ icon: '🔭', label: 'Discovery Health' },
};

/* ── Init ──────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', init);

async function init() {
  const params = new URLSearchParams(window.location.search);

  if (params.get('token')) {
    token = params.get('token');
    localStorage.setItem('si_token', token);
    window.history.replaceState({}, '', '/');
  }
  if (params.get('auth_error')) {
    showToast('Spotify auth failed: ' + params.get('auth_error'), 'error');
    window.history.replaceState({}, '', '/');
  }

  if (token) {
    showApp();
  } else {
    showLanding();
  }
}

/* ── Landing ───────────────────────────────────────────── */
async function showLanding() {
  show('landing'); hide('app');
  const btn = document.getElementById('connect-btn');
  try {
    const res = await fetch(`${API}/api/spotify/url`);
    const data = await res.json();
    if (data.url) btn.href = data.url;
    else btn.onclick = () => showToast('Configure SPOTIFY_CLIENT_ID in .env', 'error');
  } catch {
    btn.onclick = () => showToast('Backend not reachable', 'error');
  }
}

/* ── App ───────────────────────────────────────────────── */
async function showApp() {
  hide('landing'); show('app');
  wireEvents();

  // Auto-feedback check (silent, non-blocking)
  fetch(`${API}/api/feedback/auto?token=${encodeURIComponent(token)}`)
    .then(r => r.json())
    .then(data => {
      if (data.outcomes && data.outcomes.length > 0) {
        const names = data.outcomes.slice(0, 2).map(o => `<b>${o.track_name}</b>`).join(', ');
        const banner = document.getElementById('auto-banner');
        banner.innerHTML = `🔄 Auto-feedback loop: ${names} detected as listened since last session — scores updated.`;
        show('auto-banner');
        setTimeout(() => hide('auto-banner'), 6000);
      }
    })
    .catch(() => {});

  // Load existing feedback stats
  refreshFeedbackStats();
}

/* ── Events ────────────────────────────────────────────── */
function wireEvents() {
  document.getElementById('analyze-btn').addEventListener('click', runAnalysis);
  document.getElementById('disconnect-btn').addEventListener('click', disconnect);
  document.getElementById('history-files').addEventListener('change', (e) => {
    const n = e.target.files.length;
    document.getElementById('upload-label-text').textContent =
      n > 0 ? `${n} file${n > 1 ? 's' : ''} selected` : 'Upload Streaming History (optional)';
  });
}

function disconnect() {
  localStorage.removeItem('si_token');
  token = null;
  analysisData = null;
  showLanding();
}

/* ── Analysis ──────────────────────────────────────────── */
async function runAnalysis() {
  const btn = document.getElementById('analyze-btn');
  const label = document.getElementById('analyze-label');
  const spinner = document.getElementById('analyze-spinner');

  btn.disabled = true;
  label.textContent = 'Analysing…';
  show('analyze-spinner');
  hide('results');
  show('empty-state');

  const formData = new FormData();
  formData.append('token', token);
  const files = document.getElementById('history-files').files;
  for (const f of files) formData.append('files', f);

  try {
    const res = await fetch(`${API}/api/analyze`, { method: 'POST', body: formData });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Analysis failed' }));
      throw new Error(err.detail || 'Analysis failed');
    }
    analysisData = await res.json();
    feedbackLogged = {};
    renderResults(analysisData);
  } catch (e) {
    showToast(e.message, 'error');
  } finally {
    btn.disabled = false;
    label.textContent = 'Run Analysis';
    hide('analyze-spinner');
  }
}

/* ── Render ────────────────────────────────────────────── */
function renderResults(data) {
  hide('empty-state');
  show('results');

  const { user_profile, model_result, diagnosis, agent_chain, recommendations, history_mode } = data;

  // Nav username
  if (user_profile.display_name) {
    document.getElementById('nav-username').textContent = user_profile.display_name;
  }

  if (!history_mode) {
    showToast('API-only mode — upload Extended Streaming History for full accuracy', 'warn');
  }

  renderRisk(model_result, user_profile);
  renderShap(model_result);
  renderAgentChain(agent_chain, diagnosis);
  renderRecommendations(recommendations, model_result.churn_probability);
  refreshFeedbackStats();
}

/* ── Risk panel ────────────────────────────────────────── */
function renderRisk(mr, profile) {
  const prob = mr.churn_probability;
  const risk = mr.risk_level;

  // Badge
  const badge = document.getElementById('risk-badge');
  badge.textContent = risk;
  badge.className = 'risk-badge risk-' + risk.toLowerCase();

  // Gauge
  const color = risk === 'High' ? '#e53e3e' : risk === 'Medium' ? '#dd6b20' : '#38a169';
  Plotly.newPlot('gauge-chart', [{
    type: 'indicator',
    mode: 'gauge+number',
    value: Math.round(prob * 1000) / 10,
    number: { suffix: '%', font: { size: 38, color } },
    gauge: {
      axis: { range: [0, 100], tickcolor: '#444', tickfont: { color: '#666', size: 10 } },
      bar: { color, thickness: 0.22 },
      bgcolor: 'transparent',
      bordercolor: 'transparent',
      steps: [
        { range: [0,  38], color: '#0a2010' },
        { range: [38, 65], color: '#1e1208' },
        { range: [65,100], color: '#200808' },
      ],
    },
    title: { text: 'Churn Probability', font: { color: '#666', size: 12 } },
  }], {
    paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
    font: { color: '#f0f0f0', family: 'Inter, sans-serif' },
    margin: { l: 20, r: 20, t: 20, b: 10 }, height: 190,
  }, { responsive: true, displayModeBar: false });

  // Meta
  const genres = (profile.top_genres || []).slice(0, 4).map(g =>
    `<span class="pill pill-green">${g}</span>`
  ).join('');
  const topDriver = mr.top_drivers[0];
  document.getElementById('risk-meta').innerHTML = `
    <div>Top signal: <b>${FEATURE_LABELS[topDriver[0]] || topDriver[0]}</b> (SHAP ${topDriver[1] > 0 ? '+' : ''}${topDriver[1].toFixed(3)})</div>
    <div>${genres}</div>
  `;
}

/* ── SHAP chart ────────────────────────────────────────── */
function renderShap(mr) {
  const sv = mr.shap_values;
  const fv = mr.feature_values;
  const entries = Object.entries(sv).sort((a, b) => a[1] - b[1]);
  const labels = entries.map(([k]) => FEATURE_LABELS[k] || k);
  const values = entries.map(([, v]) => v);
  const colors = values.map(v => v > 0 ? '#e53e3e' : '#38a169');

  Plotly.newPlot('shap-chart', [{
    type: 'bar', orientation: 'h',
    x: values, y: labels,
    marker: { color: colors },
    hovertemplate: '%{y}: %{x:+.3f}<extra></extra>',
  }], {
    paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
    font: { color: '#a0a0a0', family: 'Inter, sans-serif', size: 11 },
    margin: { l: 10, r: 20, t: 10, b: 30 }, height: 220,
    xaxis: {
      title: { text: '← reduces risk  |  increases risk →', font: { size: 10, color: '#555' } },
      gridcolor: '#1e1e1e', zerolinecolor: '#444',
    },
    yaxis: { gridcolor: 'transparent', automargin: true },
    shapes: [{ type: 'line', x0: 0, x1: 0, y0: -0.5, y1: labels.length - 0.5,
               line: { color: '#444', width: 1 } }],
  }, { responsive: true, displayModeBar: false });
}

/* ── Agent chain ───────────────────────────────────────── */
function renderAgentChain(chain, diagnosis) {
  const container = document.getElementById('agent-chain');
  container.innerHTML = '';

  const tools = [
    'analyze_genre_entropy',
    'analyze_mood_trajectory',
    'evaluate_discovery_health',
  ];

  tools.forEach((toolName, i) => {
    const step = chain.find(s => s.tool === toolName);
    const meta = TOOL_LABELS[toolName] || { icon: '🔧', label: toolName };
    const div = document.createElement('div');
    div.className = 'chain-step ' + (step ? 'active' : 'pending');

    if (!step) {
      div.innerHTML = `
        <div class="chain-step-num">${i + 1}</div>
        <div class="chain-tool-name">${meta.icon} ${toolName}</div>
        <div class="chain-interp" style="color:#555">Not called (fallback mode)</div>
      `;
    } else {
      const r = step.result;
      const sevClass = r.severity ? 'sev-' + r.severity
                     : r.health_label ? 'sev-' + r.health_label
                     : '';
      const sevText = r.severity || r.health_label || r.mood_label || '';

      div.innerHTML = `
        <div class="chain-step-num">${i + 1}</div>
        <div class="chain-tool-name">${meta.icon} ${meta.label}</div>
        ${sevText ? `<span class="chain-severity ${sevClass}">${sevText.toUpperCase()}</span>` : ''}
        <div class="chain-interp">${r.interpretation || ''}</div>
        ${r.recommendation_hint ? `<div class="chain-hint">💡 ${r.recommendation_hint}</div>` : ''}
      `;
    }

    container.appendChild(div);
  });

  // Diagnosis section
  if (!diagnosis) return;
  const diag = document.getElementById('diagnosis-section');
  document.getElementById('diag-text').textContent = diagnosis.diagnosis || '';
  document.getElementById('hypo-text').textContent = diagnosis.hypothesis || '';
  document.getElementById('strat-text').textContent = diagnosis.strategy || '';

  const pills = document.getElementById('strat-pills');
  pills.innerHTML = '';
  if (diagnosis.strategy_genre) {
    pills.innerHTML += `<span class="pill pill-green">Genre: ${diagnosis.strategy_genre}</span>`;
  }
  if (diagnosis.strategy_artist) {
    pills.innerHTML += `<span class="pill pill-orange">Artist: ${diagnosis.strategy_artist}</span>`;
  }

  // Urgency on rec header
  const urgMap = { act_now: '🔴 Act Now', act_soon: '🟡 Act Soon', monitor: '🟢 Monitor' };
  const badge = document.getElementById('risk-badge');
  if (diagnosis.urgency && urgMap[diagnosis.urgency]) {
    document.getElementById('rec-strategy-label').textContent = urgMap[diagnosis.urgency];
  }

  show('diagnosis-section');
}

/* ── Recommendations ───────────────────────────────────── */
function renderRecommendations(recs, churnProb) {
  const container = document.getElementById('recommendations');
  container.innerHTML = '';

  if (!recs || recs.length === 0) {
    container.innerHTML = '<p style="color:#555;padding:1rem">No recommendations returned.</p>';
    return;
  }

  recs.forEach(rec => {
    const card = document.createElement('div');
    card.className = 'rec-card';
    card.dataset.id = rec.id;

    const artHtml = rec.album_image
      ? `<img class="rec-art" src="${rec.album_image}" alt="${rec.name}" loading="lazy"/>`
      : `<div class="rec-art-placeholder">🎵</div>`;

    const nameHtml = rec.external_url
      ? `<a class="rec-title" href="${rec.external_url}" target="_blank" rel="noopener">${rec.name}</a>`
      : `<div class="rec-title">${rec.name}</div>`;

    const sim = typeof rec.similarity === 'number' ? rec.similarity : 0;
    const fb  = typeof rec.feedback_score === 'number' ? rec.feedback_score : 0;

    card.innerHTML = `
      ${artHtml}
      <div class="rec-body">
        ${nameHtml}
        <div class="rec-artist">${rec.artist}</div>
        <div class="rec-scores">
          <span class="pill pill-green">Match ${Math.round(sim * 100)}%</span>
          <span class="pill pill-orange">Score ${fb >= 0 ? '+' : ''}${fb.toFixed(1)}</span>
        </div>
        <div class="rec-actions" id="actions-${rec.id}">
          <button class="btn-listened" onclick="logFeedback('${rec.id}','${esc(rec.name)}','${esc(rec.artist)}','listened',${churnProb})">✓ Listened</button>
          <button class="btn-skipped"  onclick="logFeedback('${rec.id}','${esc(rec.name)}','${esc(rec.artist)}','skipped',${churnProb})">✗ Skipped</button>
        </div>
      </div>
    `;
    container.appendChild(card);
  });
}

async function logFeedback(trackId, name, artist, outcome, churnProb) {
  if (feedbackLogged[trackId]) return;
  feedbackLogged[trackId] = true;

  const actionsEl = document.getElementById(`actions-${trackId}`);
  if (actionsEl) {
    actionsEl.innerHTML = `<button class="btn-logged" disabled>${outcome === 'listened' ? '✓ Logged as Listened' : '✗ Logged as Skipped'}</button>`;
  }

  try {
    const res = await fetch(`${API}/api/feedback/log`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ track_id: trackId, track_name: name, artist, outcome, churn_prob: churnProb }),
    });
    const data = await res.json();
    if (data.stats) updateFeedbackUI(data.stats);
  } catch {
    showToast('Could not save feedback', 'error');
  }
}

/* ── Feedback stats ────────────────────────────────────── */
async function refreshFeedbackStats() {
  try {
    const res = await fetch(`${API}/api/feedback/stats`);
    const stats = await res.json();
    updateFeedbackUI(stats);
  } catch {}
}

function updateFeedbackUI(stats) {
  const el = document.getElementById('feedback-stats');
  if (!el) return;

  el.innerHTML = `
    <div class="stat-box"><div class="stat-value">${stats.total}</div><div class="stat-label">Total</div></div>
    <div class="stat-box"><div class="stat-value" style="color:#38a169">${stats.listened}</div><div class="stat-label">Listened</div></div>
    <div class="stat-box"><div class="stat-value" style="color:#e53e3e">${stats.skipped}</div><div class="stat-label">Skipped</div></div>
    <div class="stat-box"><div class="stat-value">${stats.total > 0 ? Math.round(stats.success_rate * 100) + '%' : '—'}</div><div class="stat-label">Success Rate</div></div>
    <div class="stat-box"><div class="stat-value" style="color:#1db954">${stats.auto_detected || 0}</div><div class="stat-label">Auto-detected</div></div>
  `;

  const badge = document.getElementById('loop-badge');
  if (badge && stats.total > 0) {
    badge.textContent = `${stats.total} interaction${stats.total !== 1 ? 's' : ''}`;
  }

  if (stats.trend && stats.trend.length > 1) drawTrendChart(stats.trend);
}

/* ── Trend chart ───────────────────────────────────────── */
function drawTrendChart(trend) {
  const x = trend.map(t => t.index);
  const y = trend.map(t => t.success_rate);

  Plotly.newPlot('trend-chart', [
    {
      x, y, mode: 'lines+markers',
      line: { color: '#1db954', width: 2 },
      marker: { color: '#1db954', size: 6 },
      fill: 'tozeroy', fillcolor: '#1db95415',
      hovertemplate: 'Interaction %{x}: %{y:.0%}<extra></extra>',
    },
    {
      x: [x[0], x[x.length - 1]], y: [0.5, 0.5],
      mode: 'lines', line: { color: '#444', width: 1, dash: 'dash' },
      hoverinfo: 'none', showlegend: false,
    }
  ], {
    paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
    font: { color: '#a0a0a0', family: 'Inter, sans-serif', size: 11 },
    margin: { l: 40, r: 10, t: 10, b: 40 }, height: 160,
    xaxis: { title: { text: 'Interaction #', font: { size: 10 } }, gridcolor: '#1e1e1e', tickfont: { size: 10 } },
    yaxis: { title: { text: 'Success Rate', font: { size: 10 } }, range: [0, 1], gridcolor: '#1e1e1e', tickformat: '.0%', tickfont: { size: 10 } },
  }, { responsive: true, displayModeBar: false });
}

/* ── Helpers ───────────────────────────────────────────── */
function show(id) { document.getElementById(id)?.classList.remove('hidden'); }
function hide(id) { document.getElementById(id)?.classList.add('hidden'); }
function esc(s)   { return (s || '').replace(/'/g, "\\'").replace(/"/g, '&quot;'); }

function showToast(msg, type = 'info') {
  const t = document.createElement('div');
  t.style.cssText = `
    position:fixed; bottom:1.5rem; right:1.5rem; z-index:9999;
    background:${type === 'error' ? '#2a0a0a' : type === 'warn' ? '#1e1208' : '#0a1e14'};
    border:1px solid ${type === 'error' ? '#e53e3e40' : type === 'warn' ? '#dd6b2040' : '#1db95440'};
    color:${type === 'error' ? '#e53e3e' : type === 'warn' ? '#dd6b20' : '#1db954'};
    padding:.75rem 1.2rem; border-radius:8px; font-size:.85rem;
    max-width:340px; animation:fadeUp .3s ease;
  `;
  t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 4000);
}
