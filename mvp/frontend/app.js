/**
 * Football Analytics Dashboard — Client Application
 * ===================================================
 * Canvas-based pitch rendering, interactive player rankings,
 * and ML-powered coaching reports.
 */

const API = '';

// ── State ──
let matchesData = [];
let currentMatch = null;
let currentPlayers = [];
let currentPlayer = null;
let currentSort = 'avg_dq';
let activeTab = 'best';
let activeCompFilter = null;

// ── Role helpers ──
const ROLE_EMOJI = { GK: '🧤', DEF: '🛡️', MID: '🎯', FWD: '⚡' };
const ROLE_COLOR = { GK: '#f39c12', DEF: '#3498db', MID: '#2ecc71', FWD: '#e74c3c' };

// ── DOM Refs ──
const $views = {
  selector: document.getElementById('view-selector'),
  dashboard: document.getElementById('view-dashboard'),
  player: document.getElementById('view-player'),
};

// ═══════════════════════════════════════════════════════════════════════════════
// Boot
// ═══════════════════════════════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', async () => {
  await loadMatches();
  bindEvents();
});

async function loadMatches() {
  try {
    const res = await fetch(`${API}/api/matches`);
    const data = await res.json();
    matchesData = data.matches;
    document.getElementById('match-count').textContent = data.total;
    renderCompFilters();
    renderMatchCards();
  } catch (e) {
    console.error('Failed to load matches:', e);
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Event Binding
// ═══════════════════════════════════════════════════════════════════════════════

function bindEvents() {
  // Search
  document.getElementById('match-search').addEventListener('input', renderMatchCards);

  // Sort buttons
  document.querySelectorAll('.sort-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.sort-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentSort = btn.dataset.sort;
      renderPlayerRanking();
    });
  });

  // Back button
  document.getElementById('back-btn').addEventListener('click', () => showView('dashboard'));

  // Tab buttons
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      activeTab = btn.dataset.tab;
      renderActions();
    });
  });

  // Match badge click
  document.getElementById('match-badge').addEventListener('click', () => showView('selector'));
}

// ═══════════════════════════════════════════════════════════════════════════════
// View Management
// ═══════════════════════════════════════════════════════════════════════════════

function showView(name) {
  Object.values($views).forEach(v => v.classList.remove('active'));
  $views[name].classList.add('active');
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ═══════════════════════════════════════════════════════════════════════════════
// Match Selector
// ═══════════════════════════════════════════════════════════════════════════════

function renderCompFilters() {
  const comps = [...new Set(matchesData.map(m => m.competition))];
  const el = document.getElementById('comp-filters');
  el.innerHTML = `<button class="chip active" data-comp="all">All</button>` +
    comps.map(c => `<button class="chip" data-comp="${c}">${c}</button>`).join('');

  el.querySelectorAll('.chip').forEach(chip => {
    chip.addEventListener('click', () => {
      el.querySelectorAll('.chip').forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      activeCompFilter = chip.dataset.comp === 'all' ? null : chip.dataset.comp;
      renderMatchCards();
    });
  });
}

function renderMatchCards() {
  const query = document.getElementById('match-search').value.toLowerCase();
  const grid = document.getElementById('matches-grid');

  let filtered = matchesData;
  if (activeCompFilter) {
    filtered = filtered.filter(m => m.competition === activeCompFilter);
  }
  if (query) {
    filtered = filtered.filter(m =>
      (m.home_team + m.away_team + m.competition + m.season).toLowerCase().includes(query)
    );
  }

  grid.innerHTML = filtered.slice(0, 60).map(m => `
    <div class="match-card" data-mid="${m.match_id}" data-cid="${m.comp_id}" data-sid="${m.season_id}">
      <div class="match-card-comp">${m.competition} · ${m.season}</div>
      <div class="match-card-teams">
        <div class="match-card-team">${m.home_team}</div>
        <div class="match-card-score">${m.home_score} - ${m.away_score}</div>
        <div class="match-card-team">${m.away_team}</div>
      </div>
      <div class="match-card-date">${m.match_date}${m.stadium ? ' · ' + m.stadium : ''}</div>
    </div>
  `).join('');

  grid.querySelectorAll('.match-card').forEach(card => {
    card.addEventListener('click', () => analyzeMatch(
      +card.dataset.mid, +card.dataset.cid, +card.dataset.sid
    ));
  });
}

// ═══════════════════════════════════════════════════════════════════════════════
// Analyze Match
// ═══════════════════════════════════════════════════════════════════════════════

async function analyzeMatch(matchId, compId, seasonId) {
  const loadingEl = document.getElementById('loading-screen');
  loadingEl.classList.remove('hidden');

  try {
    const res = await fetch(`${API}/api/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ match_id: matchId, comp_id: compId, season_id: seasonId }),
    });
    const data = await res.json();
    currentMatch = data.match;
    currentPlayers = data.players;

    // Update badge
    const badge = document.getElementById('match-badge');
    badge.classList.remove('hidden');
    badge.innerHTML = `⚽ ${currentMatch.home_team} ${currentMatch.home_score} - ${currentMatch.away_score} ${currentMatch.away_team}`;

    renderTeamComparison(data.teams);
    renderPlayerRanking();
    showView('dashboard');
  } catch (e) {
    console.error('Analysis failed:', e);
    alert('Failed to analyze match. Check console for details.');
  } finally {
    loadingEl.classList.add('fade-out');
    setTimeout(() => {
      loadingEl.classList.add('hidden');
      loadingEl.classList.remove('fade-out');
    }, 500);
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Team Comparison
// ═══════════════════════════════════════════════════════════════════════════════

function renderTeamComparison(teams) {
  const el = document.getElementById('team-comparison');
  el.innerHTML = teams.map(t => `
    <div class="team-card">
      <div class="team-card-name">${t.name}</div>
      <div class="team-stats">
        <div class="team-stat">
          <div class="team-stat-value">${t.avg_dq.toFixed(2)}</div>
          <div class="team-stat-label">Avg DQ</div>
        </div>
        <div class="team-stat">
          <div class="team-stat-value">${t.total_goals}</div>
          <div class="team-stat-label">Goals</div>
        </div>
        <div class="team-stat">
          <div class="team-stat-value">${t.total_turnovers}</div>
          <div class="team-stat-label">Turnovers</div>
        </div>
        <div class="team-stat">
          <div class="team-stat-value">${t.players}</div>
          <div class="team-stat-label">Players</div>
        </div>
      </div>
    </div>
  `).join('');
}

// ═══════════════════════════════════════════════════════════════════════════════
// Player Ranking
// ═══════════════════════════════════════════════════════════════════════════════

function renderPlayerRanking() {
  const el = document.getElementById('player-ranking');
  const sorted = [...currentPlayers].sort((a, b) => {
    if (currentSort === 'turnovers') return b.turnovers - a.turnovers;
    if (currentSort === 'goals_scored') return b.goals_scored - a.goals_scored;
    if (currentSort === 'actions') return b.actions - a.actions;
    if (currentSort === 'pass_accuracy') return b.pass_accuracy - a.pass_accuracy;
    if (currentSort === 'avg_delta_xt') return b.avg_delta_xt - a.avg_delta_xt;
    return b.avg_dq - a.avg_dq;
  });

  const dqColor = dq => dq >= 0.80 ? 'metric-good' : dq >= 0.72 ? 'metric-warn' : 'metric-bad';
  const toColor = pct => pct <= 10 ? 'metric-good' : pct <= 20 ? 'metric-warn' : 'metric-bad';
  const dqGrad = dq => {
    if (dq >= 0.80) return 'var(--accent-green)';
    if (dq >= 0.72) return 'var(--accent-orange)';
    return 'var(--accent-red)';
  };

  el.innerHTML = `
    <div class="player-row-header">
      <span>#</span>
      <span>Player</span>
      <span>Position</span>
      <span>Decision Quality</span>
      <span>Goals</span>
      <span>Actions</span>
      <span>Pass %</span>
      <span>Turnovers</span>
    </div>
  ` + sorted.map((p, i) => `
    <div class="player-row" data-idx="${currentPlayers.indexOf(p)}">
      <div class="player-rank">${i + 1}</div>
      <div class="player-info">
        <div class="player-name">${p.name}</div>
        <div class="player-team">${p.team}</div>
      </div>
      <div class="player-metric">
        <span class="role-badge" style="background:${ROLE_COLOR[p.role] || '#666'}22;color:${ROLE_COLOR[p.role] || '#999'};border:1px solid ${ROLE_COLOR[p.role] || '#666'}44">
          ${ROLE_EMOJI[p.role] || ''} ${p.position || 'Unknown'}
        </span>
      </div>
      <div class="player-metric">
        <span class="${dqColor(p.avg_dq)}">${p.avg_dq.toFixed(2)}</span>
        <div class="dq-bar"><div class="dq-bar-fill" style="width:${p.avg_dq * 100}%;background:${dqGrad(p.avg_dq)}"></div></div>
      </div>
      <div class="player-metric">${p.goals_scored || '—'}</div>
      <div class="player-metric">${p.actions}</div>
      <div class="player-metric">${p.pass_accuracy}%</div>
      <div class="player-metric"><span class="${toColor(p.turnover_pct)}">${p.turnovers} (${p.turnover_pct}%)</span></div>
    </div>
  `).join('');

  el.querySelectorAll('.player-row').forEach(row => {
    row.addEventListener('click', () => {
      const idx = +row.dataset.idx;
      openPlayerDetail(currentPlayers[idx]);
    });
  });
}

// ═══════════════════════════════════════════════════════════════════════════════
// Player Detail
// ═══════════════════════════════════════════════════════════════════════════════

function openPlayerDetail(player) {
  currentPlayer = player;

  // Header
  const initials = player.short_name.slice(0, 2).toUpperCase();
  const dqColor = player.avg_dq >= 0.80 ? 'var(--accent-green)' :
    player.avg_dq >= 0.72 ? 'var(--accent-orange)' : 'var(--accent-red)';

  document.getElementById('player-header').innerHTML = `
    <div class="player-avatar" style="background:linear-gradient(135deg, ${ROLE_COLOR[player.role] || '#3498db'}, ${ROLE_COLOR[player.role] || '#3498db'}88)">${ROLE_EMOJI[player.role] || initials}</div>
    <div class="player-header-info">
      <h2>${player.name}</h2>
      <div class="player-header-sub">
        <span class="role-badge" style="background:${ROLE_COLOR[player.role] || '#666'}22;color:${ROLE_COLOR[player.role] || '#999'};border:1px solid ${ROLE_COLOR[player.role] || '#666'}44;margin-right:8px">
          ${ROLE_EMOJI[player.role] || ''} ${player.position || 'Unknown'}
        </span>
        ${player.team} · ${player.actions} actions · ${player.goals_scored} goals
      </div>
    </div>
    <div class="player-header-dq">
      <div class="player-header-dq-value" style="color:${dqColor}">${player.avg_dq.toFixed(2)}</div>
      <div class="player-header-dq-label">Decision Quality</div>
    </div>
  `;

  // Stats grid
  const stats = [
    { value: player.passes, label: 'Passes' },
    { value: `${player.pass_accuracy}%`, label: 'Pass Accuracy' },
    { value: player.carries, label: 'Carries' },
    { value: player.shots, label: 'Shots' },
    { value: player.dribbles, label: 'Dribbles' },
    { value: `${player.turnovers} (${player.turnover_pct}%)`, label: 'Turnovers' },
    { value: player.progressive_actions, label: 'Progressive' },
    { value: `${player.pressure_pct}%`, label: 'Under Pressure' },
  ];

  document.getElementById('player-stats-grid').innerHTML = stats.map(s => `
    <div class="stat-card">
      <div class="stat-card-value">${s.value}</div>
      <div class="stat-card-label">${s.label}</div>
    </div>
  `).join('');

  // Coaching report
  const reportEl = document.getElementById('coaching-report');
  reportEl.innerHTML = `
    <div class="coaching-title">🧠 AI Coaching Report</div>
    <div class="coaching-lines">
      ${player.coaching_report.map(line =>
    line.startsWith('💡')
      ? `<div class="coaching-line recommendation">${line}</div>`
      : `<div class="coaching-line">${line}</div>`
  ).join('')}
    </div>
  `;

  // Reset tab to best
  activeTab = 'best';
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelector('.tab-btn[data-tab="best"]').classList.add('active');

  renderActions();
  showView('player');
}

// ═══════════════════════════════════════════════════════════════════════════════
// Actions (Pitch rendering)
// ═══════════════════════════════════════════════════════════════════════════════

function renderActions() {
  if (!currentPlayer) return;
  const actions = activeTab === 'best' ? currentPlayer.best_actions : currentPlayer.worst_actions;
  const container = document.getElementById('actions-container');

  container.innerHTML = actions.map((a, i) => {
    const tags = [];
    if (a.is_turnover) tags.push('<span class="action-tag tag-turnover">Turnover</span>');
    if (a.goal_within_5) tags.push('<span class="action-tag tag-goal">Goal</span>');
    if (a.under_pressure) tags.push('<span class="action-tag tag-pressure">Under Pressure</span>');

    const isBest = activeTab === 'best';

    return `
      <div class="action-card">
        <div class="action-card-header">
          <div style="display:flex;align-items:center;gap:12px">
            <div class="action-rank ${isBest ? 'best' : 'worst'}">#${i + 1}</div>
            <div>
              <strong>${a.event_type}</strong>${a.outcome ? ' → ' + a.outcome : ''}
              <div style="font-size:0.75rem;color:var(--text-muted)">${a.minute}'${String(a.second).padStart(2, '0')}"</div>
            </div>
          </div>
          <div class="action-meta">
            ${tags.join('')}
          </div>
        </div>
        <div class="action-pitch-wrap">
          <canvas class="pitch-canvas" id="pitch-${i}" width="600" height="400"></canvas>
          <div class="action-detail-panel">
            <div class="action-info-row">
              <span class="action-info-label">Decision Quality</span>
              <span class="action-info-value" style="color:${a.decision_quality >= 0.8 ? 'var(--accent-green)' : a.decision_quality >= 0.5 ? 'var(--accent-orange)' : 'var(--accent-red)'}">${a.decision_quality.toFixed(3)}</span>
            </div>
            <div class="action-info-row">
              <span class="action-info-label">ΔxT</span>
              <span class="action-info-value" style="color:${a.delta_xt >= 0 ? 'var(--accent-green)' : 'var(--accent-red)'}">${a.delta_xt >= 0 ? '+' : ''}${a.delta_xt.toFixed(4)}</span>
            </div>
            <div class="action-info-row">
              <span class="action-info-label">Position</span>
              <span class="action-info-value">(${a.start_x}, ${a.start_y})</span>
            </div>
            ${a.freeze_frame.length > 0 ? `
            <div class="action-info-row">
              <span class="action-info-label">Players Visible</span>
              <span class="action-info-value">${a.freeze_frame.length}</span>
            </div>` : ''}
            ${a.recommendation ? `
            <div class="recommendation-box">
              <div class="rec-label">⭐ ML Recommendation</div>
              ${a.recommendation.text}
            </div>` : ''}
          </div>
        </div>
      </div>
    `;
  }).join('');

  // Draw pitches after DOM update
  requestAnimationFrame(() => {
    actions.forEach((a, i) => {
      const canvas = document.getElementById(`pitch-${i}`);
      if (canvas) drawPitch(canvas, a, activeTab === 'best');
    });
  });
}

// ═══════════════════════════════════════════════════════════════════════════════
// Canvas Pitch Renderer
// ═══════════════════════════════════════════════════════════════════════════════

function drawPitch(canvas, action, isBest) {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);

  const W = rect.width;
  const H = rect.height;
  const PAD = 12;

  // Scale: StatsBomb pitch is 120x80
  const scaleX = (W - PAD * 2) / 120;
  const scaleY = (H - PAD * 2) / 80;
  const tx = x => PAD + x * scaleX;
  const ty = y => PAD + y * scaleY;

  // ── Draw pitch ──
  ctx.fillStyle = '#1a1a2e';
  ctx.fillRect(0, 0, W, H);

  ctx.strokeStyle = 'rgba(255,255,255,0.12)';
  ctx.lineWidth = 1;

  // Outline
  ctx.strokeRect(tx(0), ty(0), 120 * scaleX, 80 * scaleY);

  // Center line
  ctx.beginPath();
  ctx.moveTo(tx(60), ty(0));
  ctx.lineTo(tx(60), ty(80));
  ctx.stroke();

  // Center circle
  ctx.beginPath();
  ctx.arc(tx(60), ty(40), 9.15 * scaleX, 0, Math.PI * 2);
  ctx.stroke();

  // Penalty areas
  ctx.strokeRect(tx(0), ty(18), 18 * scaleX, 44 * scaleY);
  ctx.strokeRect(tx(102), ty(18), 18 * scaleX, 44 * scaleY);

  // 6-yard boxes
  ctx.strokeRect(tx(0), ty(30), 6 * scaleX, 20 * scaleY);
  ctx.strokeRect(tx(114), ty(30), 6 * scaleX, 20 * scaleY);

  // Goal lines
  ctx.strokeStyle = 'rgba(255,255,255,0.3)';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(tx(0), ty(36)); ctx.lineTo(tx(0), ty(44));
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(tx(120), ty(36)); ctx.lineTo(tx(120), ty(44));
  ctx.stroke();

  // ── Draw freeze frame players ──
  const ff = action.freeze_frame || [];

  ff.forEach(p => {
    const px = tx(p.x);
    const py = ty(p.y);

    if (p.actor) {
      // Ball carrier — large cyan dot
      ctx.beginPath();
      ctx.arc(px, py, 8, 0, Math.PI * 2);
      ctx.fillStyle = '#00d4ff';
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 2;
      ctx.stroke();
    } else if (p.teammate) {
      if (p.keeper) {
        // Teammate GK — diamond
        ctx.fillStyle = '#f39c12';
        drawDiamond(ctx, px, py, 6);
      } else {
        // Teammate — blue circle
        ctx.beginPath();
        ctx.arc(px, py, 5, 0, Math.PI * 2);
        ctx.fillStyle = '#3498db';
        ctx.fill();
        ctx.strokeStyle = 'rgba(255,255,255,0.5)';
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    } else {
      if (p.keeper) {
        // Opponent GK — red diamond
        ctx.fillStyle = '#ff4444';
        drawDiamond(ctx, px, py, 6);
      } else {
        // Opponent — red triangle
        ctx.fillStyle = '#e74c3c';
        drawTriangle(ctx, px, py, 6);
      }
    }
  });

  // ── Draw actual action arrow ──
  const sx = tx(action.start_x);
  const sy = ty(action.start_y);

  if (action.end_x !== null && action.end_y !== null) {
    const ex = tx(action.end_x);
    const ey = ty(action.end_y);
    const color = isBest ? '#2ecc71' : (action.is_turnover ? '#e67e22' : '#e74c3c');

    drawArrow(ctx, sx, sy, ex, ey, color, 2.5, false);

    // End dot
    ctx.beginPath();
    ctx.arc(ex, ey, 4, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.7;
    ctx.fill();
    ctx.globalAlpha = 1;
  }

  // ── Draw ML recommendation arrow (worst actions only) ──
  if (!isBest && action.recommendation && action.recommendation.target_x != null) {
    const rec = action.recommendation;
    const rtx = tx(rec.target_x);
    const rty = ty(rec.target_y);
    const recColor = rec.action_type === 'shot' ? '#f1c40f' :
      rec.action_type === 'carry' ? '#00ff88' : '#f1c40f';

    drawArrow(ctx, sx, sy, rtx, rty, recColor, 2, true);

    // Recommendation marker
    if (rec.action_type === 'shot') {
      // Star
      drawStar(ctx, rtx, rty, 8, recColor);
    } else if (rec.action_type === 'carry') {
      // Diamond
      ctx.fillStyle = recColor;
      drawDiamond(ctx, rtx, rty, 7);
    } else {
      // Star for pass
      drawStar(ctx, rtx, rty, 7, recColor);
    }
  }

  // If no freeze frame, draw ball carrier dot
  if (ff.length === 0) {
    ctx.beginPath();
    ctx.arc(sx, sy, 8, 0, Math.PI * 2);
    ctx.fillStyle = '#00d4ff';
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2;
    ctx.stroke();
  }
}

// ── Drawing Helpers ──

function drawArrow(ctx, x1, y1, x2, y2, color, width, dashed) {
  ctx.beginPath();
  if (dashed) ctx.setLineDash([6, 4]);
  else ctx.setLineDash([]);
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.globalAlpha = 0.85;
  ctx.stroke();
  ctx.globalAlpha = 1;
  ctx.setLineDash([]);

  // Arrowhead
  const angle = Math.atan2(y2 - y1, x2 - x1);
  const headLen = 10;
  ctx.beginPath();
  ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - headLen * Math.cos(angle - 0.4), y2 - headLen * Math.sin(angle - 0.4));
  ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - headLen * Math.cos(angle + 0.4), y2 - headLen * Math.sin(angle + 0.4));
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.stroke();
}

function drawTriangle(ctx, x, y, size) {
  ctx.beginPath();
  ctx.moveTo(x, y - size);
  ctx.lineTo(x - size * 0.87, y + size * 0.5);
  ctx.lineTo(x + size * 0.87, y + size * 0.5);
  ctx.closePath();
  ctx.fill();
}

function drawDiamond(ctx, x, y, size) {
  ctx.beginPath();
  ctx.moveTo(x, y - size);
  ctx.lineTo(x + size, y);
  ctx.lineTo(x, y + size);
  ctx.lineTo(x - size, y);
  ctx.closePath();
  ctx.fill();
}

function drawStar(ctx, x, y, r, color) {
  ctx.fillStyle = color;
  ctx.beginPath();
  for (let i = 0; i < 5; i++) {
    const angle = (i * 4 * Math.PI) / 5 - Math.PI / 2;
    const px = x + r * Math.cos(angle);
    const py = y + r * Math.sin(angle);
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.closePath();
  ctx.fill();
}
