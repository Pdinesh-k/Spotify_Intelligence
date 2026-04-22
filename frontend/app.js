/* ── State ─────────────────────────────────────────────── */
const API = '';
let token = localStorage.getItem('si_token');
let userId = localStorage.getItem('si_user_id') || 'global';
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

const DAY_NAMES = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'];

/* ── Router ────────────────────────────────────────────── */
const PAGES = ['page-home', 'page-analyze', 'page-results'];

function navigate(path, pushState = true) {
  if (pushState) window.history.pushState({}, '', path);
  PAGES.forEach(id => hide(id));
  const crumb = document.getElementById('nav-breadcrumb');

  if (path === '/home' || path === '/') {
    if (!token) {
      show('page-home');
      hide('app-shell');
      return;
    }
    // if logged in, /home still shows landing (disconnect flow)
    token = null;
    localStorage.removeItem('si_token');
    show('page-home');
    hide('app-shell');
    return;
  }

  if (!token) {
    navigate('/home');
    return;
  }

  show('app-shell');

  if (path === '/analyze') {
    show('page-analyze');
    if (crumb) crumb.innerHTML = breadcrumb([{ label: 'Analyze', active: true }]);
  } else if (path === '/results') {
    show('page-results');
    if (crumb) crumb.innerHTML = breadcrumb([
      { label: 'Analyze', href: '/analyze' },
      { label: 'Results', active: true },
    ]);
  }
}

function breadcrumb(steps) {
  return steps.map(s =>
    s.href
      ? `<a class="bc-link" onclick="navigate('${s.href}');return false" href="${s.href}">${s.label}</a>`
      : `<span class="bc-active">${s.label}</span>`
  ).join('<span class="bc-sep">›</span>');
}

window.addEventListener('popstate', () => navigate(location.pathname, false));

/* ── Init ──────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', init);

async function init() {
  const params = new URLSearchParams(window.location.search);
  const freshLogin = !!params.get('token');

  if (params.get('token')) {
    token = params.get('token');
    localStorage.setItem('si_token', token);
    window.history.replaceState({}, '', '/analyze');
  }
  if (params.get('auth_error')) {
    showToast('Spotify auth failed: ' + params.get('auth_error'), 'error');
    window.history.replaceState({}, '', '/home');
  }

  wireEvents();

  if (!token) {
    await setupLanding();
    navigate('/home', false);
    return;
  }

  // Always auto-run analysis on every login (fresh OAuth or returning with stored token)
  show('app-shell');
  navigate('/analyze', false);

  // Auto-feedback MUST complete before analysis starts so stats are correct
  try {
    const fbRes = await fetch(`${API}/api/feedback/auto?token=${encodeURIComponent(token)}`);
    const fbData = await fbRes.json();
    if (fbData.outcomes && fbData.outcomes.length > 0) {
      const names = fbData.outcomes.slice(0, 2).map(o => `<b>${o.track_name}</b>`).join(', ');
      const banner = document.getElementById('auto-banner');
      banner.innerHTML = `🔄 Auto-feedback loop: ${names} detected as listened — scores updated.`;
      show('auto-banner');
      setTimeout(() => hide('auto-banner'), 6000);
    }
  } catch {}

  runAnalysis();
}

/* ── Landing setup ─────────────────────────────────────── */
async function setupLanding() {
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

/* ── Events ────────────────────────────────────────────── */
function wireEvents() {
  document.getElementById('analyze-btn')?.addEventListener('click', runAnalysis);
  document.getElementById('disconnect-btn')?.addEventListener('click', disconnect);
  document.getElementById('reanalyze-btn')?.addEventListener('click', () => navigate('/analyze'));
  document.getElementById('nav-logo-link')?.addEventListener('click', () => navigate('/analyze'));

  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-panel').forEach(p => p.classList.add('hidden'));
      btn.classList.add('active');
      document.getElementById('tab-' + btn.dataset.tab).classList.remove('hidden');
      if (btn.dataset.tab === 'loop') refreshFeedbackStats();
    });
  });
}

function disconnect() {
  localStorage.removeItem('si_token');
  localStorage.removeItem('si_user_id');
  token = null;
  userId = 'global';
  analysisData = null;
  setupLanding().then(() => navigate('/home'));
}

/* ── Analysis ──────────────────────────────────────────── */
async function runAnalysis() {
  const btn   = document.getElementById('analyze-btn');
  const label = document.getElementById('analyze-label');
  if (btn) btn.disabled = true;
  if (label) label.textContent = 'Analysing…';
  hide('analyze-arrow');
  show('analyze-spinner');

  const formData = new FormData();
  formData.append('token', token);

  try {
    const res = await fetch(`${API}/api/analyze`, { method: 'POST', body: formData });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Analysis failed' }));
      throw new Error(err.detail || 'Analysis failed');
    }
    analysisData = await res.json();
    feedbackLogged = {};
    renderResults(analysisData);
    navigate('/results');
  } catch (e) {
    showToast(e.message, 'error');
  } finally {
    if (btn) btn.disabled = false;
    if (label) label.textContent = 'Run Analysis';
    hide('analyze-spinner');
    show('analyze-arrow');
  }
}

/* ── Render all results ────────────────────────────────── */
function renderResults(data) {
  const { user_profile, model_result, diagnosis, agent_chain, recommendations, listening_stats, analytics } = data;

  if (user_profile.display_name) {
    document.getElementById('nav-username').textContent = user_profile.display_name;
    const greet = document.getElementById('analyze-greeting');
    if (greet) greet.textContent = `Welcome back, ${user_profile.display_name.split(' ')[0]}`;
  }
  if (user_profile.user_id) {
    userId = user_profile.user_id;
    localStorage.setItem('si_user_id', userId);
  }

  // Reset tabs to first
  document.querySelectorAll('.tab-btn').forEach((b, i) => b.classList.toggle('active', i === 0));
  document.querySelectorAll('.tab-panel').forEach((p, i) => p.classList.toggle('hidden', i !== 0));

  renderRisk(model_result, user_profile);
  renderShap(model_result);
  renderListeningStats(listening_stats, model_result, user_profile);
  if (analytics) renderMusicProfile(analytics, user_profile);
  renderAgentChain(agent_chain, diagnosis);
  renderRecommendations(recommendations, model_result.churn_probability);
  refreshFeedbackStats();
}

/* ── Tab 1: Risk ───────────────────────────────────────── */
function renderRisk(mr, profile) {
  const prob = mr.churn_probability;
  const risk = mr.risk_level;
  const badge = document.getElementById('risk-badge');
  badge.textContent = risk;
  badge.className = 'risk-badge risk-' + risk.toLowerCase();

  const color = risk === 'High' ? '#e53e3e' : risk === 'Medium' ? '#dd6b20' : '#38a169';
  const displayVal = Math.round(prob * 1000) / 10;

  Plotly.newPlot('gauge-chart', [{
    type: 'indicator',
    mode: 'gauge+number',
    value: displayVal,
    number: {
      suffix: '%',
      font: { size: 56, color, family: 'Inter, sans-serif' },
      valueformat: '.1f',
    },
    gauge: {
      axis: {
        range: [0, 100],
        tickcolor: '#444',
        tickfont: { color: '#666', size: 11 },
        tickvals: [0, 25, 50, 75, 100],
        ticktext: ['0', '25', '50', '75', '100'],
      },
      bar: { color, thickness: 0.3 },
      bgcolor: 'transparent',
      bordercolor: 'transparent',
      steps: [
        { range: [0,  38], color: '#0d2b12' },
        { range: [38, 65], color: '#2a1c08' },
        { range: [65,100], color: '#2b0a0a' },
      ],
      threshold: {
        line: { color: '#ffffff22', width: 2 },
        thickness: 0.75,
        value: 50,
      },
    },
    title: {
      text: 'Churn Probability',
      font: { color: '#777', size: 13, family: 'Inter, sans-serif' },
    },
  }], {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: { color: '#f0f0f0', family: 'Inter, sans-serif' },
    margin: { l: 30, r: 30, t: 40, b: 10 },
    height: 270,
  }, { responsive: true, displayModeBar: false });

  const genres = (profile.top_genres || []).slice(0, 4).map(g =>
    `<span class="pill pill-green">${g}</span>`
  ).join('');
  const topDriver = mr.top_drivers[0];
  document.getElementById('risk-meta').innerHTML = `
    <div>Top signal: <b>${FEATURE_LABELS[topDriver[0]] || topDriver[0]}</b> (SHAP ${topDriver[1] > 0 ? '+' : ''}${topDriver[1].toFixed(3)})</div>
    <div style="margin-top:0.5rem">${genres}</div>
  `;
}

function renderShap(mr) {
  const sv = mr.shap_values;
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
    margin: { l: 10, r: 20, t: 10, b: 40 }, height: 260,
    xaxis: {
      title: { text: '← reduces risk  |  increases risk →', font: { size: 10, color: '#555' } },
      gridcolor: '#1e1e1e', zerolinecolor: '#444',
    },
    yaxis: { gridcolor: 'transparent', automargin: true },
    shapes: [{ type: 'line', x0: 0, x1: 0, y0: -0.5, y1: labels.length - 0.5,
               line: { color: '#444', width: 1 } }],
  }, { responsive: true, displayModeBar: false });
}

function renderListeningStats(ls, mr, profile) {
  if (!ls) return;
  const fv = mr.feature_values || {};
  const items = [
    { label: 'Recent Listening',  value: `${ls.recent_hours ?? 0} hrs`,            sub: 'from last 50 plays' },
    { label: 'Obsession Rate',    value: `${Math.round((ls.obsession_rate ?? 0)*100)}%`, sub: 'short vs all-time overlap' },
    { label: 'Tracks Analysed',   value: ls.total_recent_tracks ?? 0,               sub: 'from Spotify API' },
    { label: 'Repeat Ratio',      value: `${Math.round((fv.repeat_play_ratio ?? 0)*100)}%`, sub: 'songs replayed' },
    { label: 'Days Since New Artist', value: `${Math.round(fv.days_new_artist ?? 7)} days`, sub: 'discovery recency' },
    { label: 'Skip Trend',        value: `${fv.skip_rate_trend >= 0 ? '+' : ''}${Math.round((fv.skip_rate_trend ?? 0)*100)}%`, sub: 'vs prior period' },
    { label: 'Top Genres',        value: (profile.top_genres || []).length,          sub: 'tracked genres' },
    { label: 'Artists Tracked',   value: (profile.top_artists || []).length,         sub: 'in your profile' },
  ];
  document.getElementById('listening-stats').innerHTML = items.map(i =>
    `<div class="stat-box"><div class="stat-value">${i.value}</div><div class="stat-label">${i.label}</div><div class="stat-sub">${i.sub}</div></div>`
  ).join('');
}

/* ── Tab 2: Music Profile ──────────────────────────────── */
function renderMusicProfile(an, profile) {
  const { temporal, velocity, loyalty, genre_profile, trajectory, discovery } = an;

  const dna = [
    { label: 'Listening Velocity', value: `${(velocity?.tracks_per_hour ?? 0).toFixed(1)} trk/hr`   },
    { label: 'Taste Stability',    value: `${Math.round((trajectory?.short_to_long_sim ?? 0)*100)}%` },
    { label: 'Artist Loyalty',     value: `${Math.round((loyalty?.loyalty_score ?? 0)*100)}%`        },
    { label: 'Genre Diversity',    value: `${(genre_profile?.diversity_score ?? 0).toFixed(2)} bits` },
    { label: 'Discovery Rate',     value: `${Math.round((discovery?.discovery_rate ?? 0)*100)}%`     },
  ];
  document.getElementById('dna-stats').innerHTML = dna.map(d =>
    `<div class="stat-box"><div class="stat-value">${d.value}</div><div class="stat-label">${d.label}</div></div>`
  ).join('');

  if (temporal?.hours) {
    const hours = temporal.hours;
    const maxH = Math.max(...hours, 1);
    Plotly.newPlot('hour-chart', [{
      type: 'bar', x: Array.from({length: 24}, (_, i) => i), y: hours,
      marker: { color: hours.map(v => `rgba(29,185,84,${Math.max(0.12, v/maxH)})`) },
      hovertemplate: '%{x}:00 — %{y} plays<extra></extra>',
    }], {
      paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
      font: { color: '#a0a0a0', family: 'Inter, sans-serif', size: 11 },
      margin: { l: 10, r: 10, t: 30, b: 30 }, height: 180,
      title: { text: `Plays by Hour · Peak: ${temporal.peak_hour_label}`, font: { size: 12, color: '#666' } },
      xaxis: { tickvals: [0,3,6,9,12,15,18,21], ticktext: ['0h','3h','6h','9h','12h','15h','18h','21h'], gridcolor: '#1e1e1e' },
      yaxis: { gridcolor: '#1e1e1e' },
    }, { responsive: true, displayModeBar: false });

    Plotly.newPlot('dow-chart', [{
      type: 'bar', x: DAY_NAMES, y: temporal.days,
      marker: { color: temporal.days.map((_, i) => i === temporal.peak_day ? '#1db954' : '#2d3748') },
      hovertemplate: '%{x}: %{y} plays<extra></extra>',
    }], {
      paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
      font: { color: '#a0a0a0', family: 'Inter, sans-serif', size: 11 },
      margin: { l: 10, r: 10, t: 30, b: 30 }, height: 170,
      title: { text: `Plays by Day · Peak: ${temporal.peak_day_label}`, font: { size: 12, color: '#666' } },
      yaxis: { gridcolor: '#1e1e1e' },
    }, { responsive: true, displayModeBar: false });
  }

  if (genre_profile?.genre_labels?.length) {
    const COLORS = ['#1db954','#1aa34a','#158a3e','#107032','#0b5726','#ffa500','#ff6b35','#e84393'];
    Plotly.newPlot('genre-donut', [{
      type: 'pie',
      labels: genre_profile.genre_labels,
      values: genre_profile.genre_values,
      hole: 0.55,
      marker: { colors: COLORS.slice(0, genre_profile.genre_labels.length) },
      textinfo: 'label',
      hovertemplate: '%{label}: %{value}<extra></extra>',
    }], {
      paper_bgcolor: 'transparent', font: { color: '#a0a0a0', family: 'Inter, sans-serif' },
      height: 360, showlegend: false,
      annotations: [{text: `${genre_profile.total_unique}<br>genres`, x: 0.5, y: 0.5,
        font: { size: 15, color: '#e2e8f0' }, showarrow: false}],
      margin: { l: 10, r: 10, t: 10, b: 10 },
    }, { responsive: true, displayModeBar: false });
  }

  if (trajectory) {
    const tc = trajectory.trajectory_color || '#718096';
    document.getElementById('trajectory-badge').innerHTML = `
      <div style="background:${tc}22;border:1px solid ${tc};border-radius:8px;padding:10px 16px;margin-bottom:12px">
        <span style="font-weight:700;color:${tc};font-size:1.1rem">${trajectory.trajectory}</span><br>
        <span style="color:#a0aec0;font-size:0.88rem">${trajectory.trajectory_desc}</span>
      </div>`;

    Plotly.newPlot('traj-chart', [{
      type: 'bar',
      x: ['Short→Medium','Medium→Long','Short→Long'],
      y: [
        (trajectory.short_to_medium_sim || 0)*100,
        (trajectory.medium_to_long_sim  || 0)*100,
        (trajectory.short_to_long_sim   || 0)*100,
      ],
      marker: { color: ['#1db954','#ffa500','#e84393'] },
      text: [
        `${Math.round((trajectory.short_to_medium_sim || 0)*100)}%`,
        `${Math.round((trajectory.medium_to_long_sim  || 0)*100)}%`,
        `${Math.round((trajectory.short_to_long_sim   || 0)*100)}%`,
      ],
      textposition: 'outside',
      hovertemplate: '%{x}: %{y:.1f}%<extra></extra>',
    }], {
      paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
      font: { color: '#a0a0a0', family: 'Inter, sans-serif', size: 11 },
      margin: { l: 10, r: 10, t: 20, b: 30 }, height: 230,
      yaxis: { range: [0, 120], gridcolor: '#1e1e1e', title: { text: '% Overlap', font: { size: 10 } } },
    }, { responsive: true, displayModeBar: false });

    document.getElementById('traj-caption').textContent =
      `Popularity trend: ${trajectory.popularity_trend} · Recent avg: ${trajectory.avg_popularity_short} · All-time avg: ${trajectory.avg_popularity_long}`;
  }

  if (loyalty) {
    const ls = loyalty.loyalty_score || 0;
    const lc = ls > 0.5 ? '#1db954' : ls > 0.25 ? '#ffa500' : '#e84393';
    Plotly.newPlot('loyalty-gauge', [{
      type: 'indicator', mode: 'gauge+number',
      value: Math.round(ls * 100),
      number: { suffix: '%', font: { size: 26, color: lc } },
      gauge: {
        axis: { range: [0, 100], tickcolor: '#444' },
        bar: { color: lc },
        bgcolor: '#111', bordercolor: '#2d3748',
        steps: [
          { range: [0, 33],  color: '#2a1e08' },
          { range: [33, 66], color: '#1a2a0a' },
          { range: [66, 100],color: '#0a2a14' },
        ],
      },
      title: { text: loyalty.loyalty_label, font: { color: '#666', size: 12 } },
    }], {
      paper_bgcolor: 'transparent', font: { color: '#e2e8f0', family: 'Inter, sans-serif' },
      height: 180, margin: { l: 20, r: 20, t: 10, b: 5 },
    }, { responsive: true, displayModeBar: false });

    document.getElementById('loyalty-caption').textContent = loyalty.loyalty_desc;

    const mov = [];
    if (loyalty.rising_artists?.length)  mov.push(`🚀 <b>Rising:</b> ${loyalty.rising_artists.slice(0,4).map(a=>`<code>${a}</code>`).join(' · ')}`);
    if (loyalty.falling_artists?.length) mov.push(`📉 <b>Fading:</b> ${loyalty.falling_artists.slice(0,4).map(a=>`<code>${a}</code>`).join(' · ')}`);
    if (loyalty.stable_artists?.length)  mov.push(`🔒 <b>Constants:</b> ${loyalty.stable_artists.slice(0,4).map(a=>`<code>${a}</code>`).join(' · ')}`);
    document.getElementById('artist-movement').innerHTML = mov.map(m => `<div class="movement-row">${m}</div>`).join('');
  }

  const shortA  = (profile.top_artists_short  || []).slice(0,10).map(a => a.name);
  const mediumA = (profile.top_artists_medium || []).slice(0,10).map(a => a.name);
  const longA   = (profile.top_artists_long   || []).slice(0,10).map(a => a.name);
  const stableSet = new Set(loyalty?.stable_artists || []);
  const longSet   = new Set(longA);

  const mkList = (arr, isShort, isLong) => arr.map((a, i) => {
    let badge = '';
    if (isShort) badge = !longSet.has(a) ? '🆕 ' : stableSet.has(a) ? '✅ ' : '';
    if (isLong)  badge = !new Set(shortA).has(a) ? '📉 ' : '';
    return `<div class="tl-item">${i+1}. ${badge}${a}</div>`;
  }).join('');

  document.getElementById('artist-timeline').innerHTML = `
    <div class="tl-col"><div class="tl-head">🔥 Now (4 weeks)</div>${mkList(shortA, true, false)}</div>
    <div class="tl-col"><div class="tl-head">📅 6 Months</div>${mkList(mediumA, false, false)}</div>
    <div class="tl-col"><div class="tl-head">🏆 All Time</div>${mkList(longA, false, true)}</div>
  `;

  const undiscovered = discovery?.undiscovered_artists || [];
  document.getElementById('discovery-count').textContent = `${undiscovered.length} new artists found`;
  document.getElementById('discovery-grid').innerHTML = undiscovered.slice(0, 8).map(a => {
    const genre = (a.genres || []).slice(0, 2).join(', ') || 'Unknown genre';
    return `
      <div class="discovery-card">
        <div class="disc-name">${a.name}</div>
        <div class="disc-genre">${genre}</div>
        <div class="disc-anchor">Related to <b>${a.anchor_artist || ''}</b></div>
        <div class="disc-pop">Popularity: ${a.popularity}</div>
      </div>`;
  }).join('') || '<p style="color:#555;padding:1rem">No undiscovered related artists found.</p>';
}

/* ── Tab 3: Agent chain ─────────────────────────────────── */
function renderAgentChain(chain, diagnosis) {
  const container = document.getElementById('agent-chain');
  container.innerHTML = '';

  const tools = ['analyze_genre_entropy','analyze_mood_trajectory','evaluate_discovery_health'];
  tools.forEach((toolName, i) => {
    const step = chain.find(s => s.tool === toolName);
    const meta = TOOL_LABELS[toolName] || { icon: '🔧', label: toolName };
    const div = document.createElement('div');
    div.className = 'chain-step ' + (step ? 'active' : 'pending');

    if (!step) {
      div.innerHTML = `
        <div class="chain-step-num">${i + 1}</div>
        <div class="chain-tool-name">${meta.icon} ${meta.label}</div>
        <div class="chain-interp" style="color:#555">Not called (fallback mode)</div>
      `;
    } else {
      const r = step.result;
      const sevClass = r.severity ? 'sev-' + r.severity : r.health_label ? 'sev-' + r.health_label : '';
      const sevText  = r.severity || r.health_label || r.mood_label || '';
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

  if (!diagnosis) return;
  document.getElementById('diag-text').textContent  = diagnosis.diagnosis  || '';
  document.getElementById('hypo-text').textContent  = diagnosis.hypothesis || '';
  document.getElementById('strat-text').textContent = diagnosis.strategy   || '';

  const pills = document.getElementById('strat-pills');
  pills.innerHTML = '';
  if (diagnosis.strategy_genre)  pills.innerHTML += `<span class="pill pill-green">Genre: ${diagnosis.strategy_genre}</span>`;
  if (diagnosis.strategy_artist) pills.innerHTML += `<span class="pill pill-orange">Artist: ${diagnosis.strategy_artist}</span>`;

  const urgMap = { act_now: '🔴 Act Now', act_soon: '🟡 Act Soon', monitor: '🟢 Monitor' };
  if (diagnosis.urgency && urgMap[diagnosis.urgency]) {
    document.getElementById('rec-strategy-label').textContent = urgMap[diagnosis.urgency];
  }
  show('diagnosis-section');
}

/* ── Tab 4: Recommendations ────────────────────────────── */
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
    const artHtml  = rec.album_image
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
  if (actionsEl) actionsEl.innerHTML = `<button class="btn-logged" disabled>${outcome === 'listened' ? '✓ Logged as Listened' : '✗ Logged as Skipped'}</button>`;
  try {
    const res = await fetch(`${API}/api/feedback/log`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ track_id: trackId, track_name: name, artist, outcome, churn_prob: churnProb, user_id: userId }),
    });
    const data = await res.json();
    if (data.stats) updateFeedbackUI(data.stats);
  } catch { showToast('Could not save feedback', 'error'); }
}

/* ── Tab 5: Feedback stats ─────────────────────────────── */
async function refreshFeedbackStats() {
  try {
    const res = await fetch(`${API}/api/feedback/stats?user_id=${encodeURIComponent(userId)}`);
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
  if (badge && stats.total > 0) badge.textContent = `${stats.total} interaction${stats.total !== 1 ? 's' : ''}`;
  if (stats.trend && stats.trend.length > 1) drawTrendChart(stats.trend);
}

function drawTrendChart(trend) {
  const x = trend.map(t => t.index);
  const y = trend.map(t => t.success_rate);
  Plotly.newPlot('trend-chart', [
    { x, y, mode: 'lines+markers', line: { color: '#1db954', width: 2 },
      marker: { color: '#1db954', size: 6 }, fill: 'tozeroy', fillcolor: '#1db95415',
      hovertemplate: 'Interaction %{x}: %{y:.0%}<extra></extra>' },
    { x: [x[0], x[x.length-1]], y: [0.5, 0.5], mode: 'lines',
      line: { color: '#444', width: 1, dash: 'dash' }, hoverinfo: 'none', showlegend: false },
  ], {
    paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
    font: { color: '#a0a0a0', family: 'Inter, sans-serif', size: 11 },
    margin: { l: 40, r: 10, t: 10, b: 40 }, height: 160,
    xaxis: { title: { text: 'Interaction #', font: { size: 10 } }, gridcolor: '#1e1e1e' },
    yaxis: { title: { text: 'Success Rate', font: { size: 10 } }, range: [0, 1], gridcolor: '#1e1e1e', tickformat: '.0%' },
  }, { responsive: true, displayModeBar: false });
}

/* ── Helpers ───────────────────────────────────────────── */
function show(id) { document.getElementById(id)?.classList.remove('hidden'); }
function hide(id) { document.getElementById(id)?.classList.add('hidden'); }
function esc(s)   { return (s || '').replace(/'/g, "\\'").replace(/"/g, '&quot;'); }

function showToast(msg, type = 'info') {
  const t = document.createElement('div');
  t.style.cssText = `
    position:fixed;bottom:1.5rem;right:1.5rem;z-index:9999;
    background:${type==='error'?'#2a0a0a':type==='warn'?'#1e1208':'#0a1e14'};
    border:1px solid ${type==='error'?'#e53e3e40':type==='warn'?'#dd6b2040':'#1db95440'};
    color:${type==='error'?'#e53e3e':type==='warn'?'#dd6b20':'#1db954'};
    padding:.75rem 1.2rem;border-radius:8px;font-size:.85rem;
    max-width:340px;animation:fadeUp .3s ease;
  `;
  t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 4000);
}
