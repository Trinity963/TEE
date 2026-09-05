#!/usr/bin/env python3
"""
TEE — Trinity's Execution Engine
ui/server.py
Sovereign browser UI — no dependencies, no CDN, pure Python + vanilla JS.
Serves on port 8766. Talks to TEE gateway on port 8765.
MIT License — open source, sovereign, forever.
"""
import json
import logging
import os
from collections import deque
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

log = logging.getLogger("tee.ui")

# ── In-memory log buffer — last 200 lines across all TEE loggers ──────────────

_LOG_BUFFER: deque = deque(maxlen=200)

class _BufferHandler(logging.Handler):
    """Captures log records into _LOG_BUFFER for the UI log panel."""
    def emit(self, record: logging.LogRecord):
        _LOG_BUFFER.append({
            "t":     datetime.now().strftime("%H:%M:%S"),
            "level": record.levelname,
            "name":  record.name,
            "msg":   record.getMessage(),
        })

_buf_handler = _BufferHandler()
_buf_handler.setLevel(logging.DEBUG)
logging.getLogger("tee").addHandler(_buf_handler)

TEE_API   = "http://127.0.0.1:8765"
UI_HOST   = "0.0.0.0"
UI_PORT   = 8766

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>TEE — Trinity's Execution Engine</title>
<style>
  :root {
    --bg:       #0d0d0d;
    --bg2:      #141414;
    --bg3:      #1a1a1a;
    --border:   #2a2a2a;
    --accent:   #00c8ff;
    --accent2:  #7b61ff;
    --green:    #00e5a0;
    --yellow:   #f5c518;
    --red:      #ff4c4c;
    --text:     #e0e0e0;
    --muted:    #666;
    --font:     'Courier New', Courier, monospace;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--font);
    font-size: 13px;
    min-height: 100vh;
  }
  /* ── Layout ── */
  header {
    background: var(--bg2);
    border-bottom: 1px solid var(--border);
    padding: 14px 28px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  header h1 {
    font-size: 15px;
    letter-spacing: 2px;
    color: var(--accent);
    text-transform: uppercase;
  }
  header .sub {
    font-size: 11px;
    color: var(--muted);
    margin-top: 2px;
  }
  .status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--muted);
    display: inline-block;
    margin-right: 6px;
    transition: background 0.3s;
  }
  .status-dot.online { background: var(--green); }
  .status-dot.offline { background: var(--red); }
  nav {
    background: var(--bg2);
    border-bottom: 1px solid var(--border);
    display: flex;
    gap: 0;
  }
  nav button {
    background: none;
    border: none;
    border-bottom: 2px solid transparent;
    color: var(--muted);
    cursor: pointer;
    font-family: var(--font);
    font-size: 12px;
    letter-spacing: 1px;
    padding: 10px 22px;
    text-transform: uppercase;
    transition: color 0.2s, border-color 0.2s;
  }
  nav button:hover { color: var(--text); }
  nav button.active {
    color: var(--accent);
    border-bottom-color: var(--accent);
  }
  main { padding: 24px 28px; }
  .page { display: none; }
  .page.active { display: block; }
  /* ── Cards ── */
  .card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 4px;
    margin-bottom: 16px;
    padding: 18px 20px;
  }
  .card-title {
    color: var(--accent);
    font-size: 11px;
    letter-spacing: 2px;
    margin-bottom: 14px;
    text-transform: uppercase;
  }
  /* ── GPU bars ── */
  .gpu-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 16px; }
  .gpu-card { background: var(--bg3); border: 1px solid var(--border); border-radius: 4px; padding: 16px; }
  .gpu-name { color: var(--accent2); font-size: 12px; margin-bottom: 10px; }
  .vram-bar-wrap { background: #111; border-radius: 2px; height: 8px; margin: 8px 0; overflow: hidden; }
  .vram-bar { background: var(--accent); height: 100%; border-radius: 2px; transition: width 0.5s; }
  .vram-bar.warn { background: var(--yellow); }
  .vram-bar.crit { background: var(--red); }
  .vram-label { color: var(--muted); font-size: 11px; }
  .gpu-models { color: var(--green); font-size: 11px; margin-top: 8px; }
  /* ── Model table ── */
  table { border-collapse: collapse; width: 100%; }
  th {
    color: var(--muted);
    font-size: 10px;
    letter-spacing: 1px;
    padding: 6px 10px;
    text-align: left;
    text-transform: uppercase;
    border-bottom: 1px solid var(--border);
  }
  td { padding: 8px 10px; border-bottom: 1px solid var(--border); font-size: 12px; }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: var(--bg3); }
  .tag {
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 2px;
    color: var(--muted);
    display: inline-block;
    font-size: 10px;
    margin-right: 4px;
    padding: 1px 5px;
  }
  .badge {
    border-radius: 2px;
    display: inline-block;
    font-size: 10px;
    padding: 2px 6px;
  }
  .badge.registered { background: #1a2a1a; color: var(--green); border: 1px solid #2a4a2a; }
  .badge.loaded     { background: #1a1a2a; color: var(--accent); border: 1px solid #2a2a4a; }
  .badge.error      { background: #2a1a1a; color: var(--red);   border: 1px solid #4a2a2a; }
  /* ── Dirs panel ── */
  .dir-row { display: flex; align-items: center; gap: 12px; padding: 8px 0; border-bottom: 1px solid var(--border); }
  .dir-row:last-child { border-bottom: none; }
  .dir-path { color: var(--accent2); flex: 1; font-size: 12px; word-break: break-all; }
  .dir-meta { color: var(--muted); font-size: 11px; white-space: nowrap; }
  /* ── Config editor ── */
  .config-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .config-field { margin-bottom: 14px; }
  .config-field label { color: var(--muted); display: block; font-size: 10px; letter-spacing: 1px; margin-bottom: 5px; text-transform: uppercase; }
  .config-field input, .config-field select {
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 2px;
    color: var(--text);
    font-family: var(--font);
    font-size: 12px;
    outline: none;
    padding: 6px 10px;
    width: 100%;
    transition: border-color 0.2s;
  }
  .config-field input:focus, .config-field select:focus { border-color: var(--accent); }
  .btn {
    background: none;
    border: 1px solid var(--accent);
    border-radius: 2px;
    color: var(--accent);
    cursor: pointer;
    font-family: var(--font);
    font-size: 11px;
    letter-spacing: 1px;
    padding: 7px 16px;
    text-transform: uppercase;
    transition: background 0.2s, color 0.2s;
  }
  .btn:hover { background: var(--accent); color: var(--bg); }
  .btn.danger { border-color: var(--red); color: var(--red); }
  .btn.danger:hover { background: var(--red); color: var(--bg); }
  /* ── Refresh bar ── */
  .toolbar { display: flex; align-items: center; gap: 12px; margin-bottom: 18px; }
  .toolbar .ts { color: var(--muted); font-size: 11px; flex: 1; }
  /* ── Empty state ── */
  .empty { color: var(--muted); font-size: 12px; padding: 20px 0; text-align: center; }
  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: var(--bg); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
</head>
<body>

<header>
  <div>
    <h1><span class="status-dot" id="dot"></span>TEE — Trinity's Execution Engine</h1>
    <div class="sub">Sovereign LLM Router &nbsp;·&nbsp; v1.0.0 &nbsp;·&nbsp; <span id="hdr-counts">—</span></div>
  </div>
  <div style="color:var(--muted);font-size:11px;text-align:right;">
    gateway: 127.0.0.1:8765<br/>ui: 127.0.0.1:8766
  </div>
</header>

<nav>
  <button class="active" onclick="showPage('dashboard', this)">Dashboard</button>
  <button onclick="showPage('models', this)">Models</button>
  <button onclick="showPage('directories', this)">Directories</button>
  <button onclick="showPage('config', this)">Config</button>
</nav>

<main>

  <!-- ── DASHBOARD ── -->
  <div class="page active" id="page-dashboard">
    <div class="toolbar">
      <span class="ts" id="dash-ts">—</span>
      <button class="btn" onclick="fetchStatus()">↺ Refresh</button>
    </div>
    <div class="card">
      <div class="card-title">GPU Status</div>
      <div class="gpu-grid" id="gpu-grid"><div class="empty">Waiting for TEE…</div></div>
    </div>
    <div class="card">
      <div class="card-title">Loaded Models</div>
      <div id="loaded-models"><div class="empty">No models loaded.</div></div>
    </div>
    <div class="card">
      <div class="card-title">Registry Summary</div>
      <div id="reg-summary" style="color:var(--muted);font-size:12px;">—</div>
    </div>
    <div class="card">
      <div class="card-title" style="display:flex;align-items:center;gap:12px;">
        Live Logs
        <span style="color:var(--muted);font-size:11px;font-weight:normal;">last 200 lines</span>
        <button class="btn" style="margin-left:auto" onclick="_logPaused=!_logPaused;this.textContent=_logPaused?'▶ Resume':'⏸ Pause'">⏸ Pause</button>
        <button class="btn" onclick="document.getElementById('log-panel').innerHTML=''">✕ Clear</button>
      </div>
      <div id="log-panel" style="background:#0a0a0a;border:1px solid var(--border);border-radius:3px;font-family:monospace;font-size:11px;height:220px;overflow-y:auto;padding:8px 10px;color:var(--muted);line-height:1.6;"><span style="color:var(--border)">Waiting for logs…</span></div>
    </div>
  </div>

  <!-- ── MODELS ── -->
  <div class="page" id="page-models">
    <div class="toolbar">
      <span class="ts" id="models-ts">—</span>
      <button class="btn" onclick="fetchStatus()">↺ Refresh</button>
    </div>
    <div class="card">
      <div class="card-title">Registered Models</div>
      <div id="models-table"><div class="empty">Waiting for TEE…</div></div>
    </div>
  </div>

  <!-- ── DIRECTORIES ── -->
  <div class="page" id="page-directories">
    <div class="toolbar">
      <span class="ts" id="dirs-ts">—</span>
      <button class="btn" onclick="fetchStatus()">↺ Refresh</button>
    </div>
    <div class="card">
      <div class="card-title">Watched Directories</div>
      <div id="dirs-list"><div class="empty">Waiting for TEE…</div></div>
    </div>
  </div>

  <!-- ── CONFIG ── -->
  <div class="page" id="page-config">
    <div class="toolbar">
      <span class="ts" id="config-ts">—</span>
      <button class="btn" onclick="fetchConfig()">↺ Reload</button>
    </div>
    <div class="card">
      <div class="card-title">TEE Configuration</div>
      <div id="config-view"><div class="empty">Loading…</div></div>
    </div>
  </div>

</main>

<script>
const API = '';  // proxied through ui server

let _status = null;
let _config = null;

// ── Navigation ──────────────────────────────────────────────────────────────
function showPage(name, btn) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('nav button').forEach(b => b.classList.remove('active'));
  document.getElementById('page-' + name).classList.add('active');
  btn.classList.add('active');
  if (name === 'config') fetchConfig();
}

// ── Fetch status from proxy ──────────────────────────────────────────────────
async function fetchStatus() {
  try {
    const r = await fetch('/proxy/status');
    if (!r.ok) throw new Error('TEE offline');
    _status = await r.json();
    renderDashboard(_status);
    renderModels(_status);
    renderDirs(_status);
    setOnline(true);
  } catch(e) {
    setOnline(false);
  }
}

async function fetchConfig() {
  try {
    const r = await fetch('/proxy/config');
    if (!r.ok) throw new Error();
    _config = await r.json();
    renderConfig(_config);
  } catch(e) {
    document.getElementById('config-view').innerHTML = '<div class="empty">Could not load config.</div>';
  }
}

function setOnline(ok) {
  const dot = document.getElementById('dot');
  dot.className = 'status-dot ' + (ok ? 'online' : 'offline');
}

function ts() {
  return new Date().toLocaleTimeString();
}

// ── Dashboard ────────────────────────────────────────────────────────────────
function renderDashboard(s) {
  document.getElementById('dash-ts').textContent = 'Updated: ' + ts();
  document.getElementById('hdr-counts').textContent =
    s.registered_models + ' registered · ' + s.loaded_models.length + ' loaded';

  // GPU grid
  const grid = document.getElementById('gpu-grid');
  if (!s.gpus || s.gpus.length === 0) {
    grid.innerHTML = '<div class="empty">No GPUs detected — CPU mode.</div>';
  } else {
    // Build cards only on first render — then update in-place to preserve CSS transitions
    const existingCards = grid.querySelectorAll('.gpu-card');
    if (existingCards.length !== s.gpus.length) {
      grid.innerHTML = s.gpus.map(g => {
        const used_pct = 100 - g.vram_free_pct;
        const barClass = used_pct > 90 ? 'crit' : used_pct > 70 ? 'warn' : '';
        const loaded = g.models_loaded && g.models_loaded.length
          ? g.models_loaded.join(', ') : 'none';
        return `<div class="gpu-card" data-gpu="${g.index}">
          <div class="gpu-name">GPU ${g.index} &nbsp;·&nbsp; ${g.name}</div>
          <div class="vram-bar-wrap"><div class="vram-bar ${barClass}" id="vram-bar-${g.index}" style="width:${used_pct}%"></div></div>
          <div class="vram-label" id="vram-label-${g.index}">${g.vram_free_gb} GB free / ${g.vram_total_gb} GB total</div>
          <div class="gpu-models" id="gpu-models-${g.index}">Loaded: ${loaded}</div>
        </div>`;
      }).join('');
    } else {
      // Update in-place — transitions animate smoothly
      s.gpus.forEach(g => {
        const used_pct = 100 - g.vram_free_pct;
        const barClass = used_pct > 90 ? 'crit' : used_pct > 70 ? 'warn' : '';
        const bar = document.getElementById('vram-bar-' + g.index);
        if (bar) {
          bar.style.width = used_pct + '%';
          bar.className = 'vram-bar ' + barClass;
        }
        const label = document.getElementById('vram-label-' + g.index);
        if (label) label.textContent = g.vram_free_gb + ' GB free / ' + g.vram_total_gb + ' GB total';
        const models = document.getElementById('gpu-models-' + g.index);
        const loaded = g.models_loaded && g.models_loaded.length
          ? g.models_loaded.join(', ') : 'none';
        if (models) models.textContent = 'Loaded: ' + loaded;
      });
    }
  }

  // Loaded models
  const lm = document.getElementById('loaded-models');
  if (!s.loaded_models || s.loaded_models.length === 0) {
    lm.innerHTML = '<div class="empty">No models currently loaded.</div>';
  } else {
    lm.innerHTML = '<table><thead><tr><th>Name</th><th>Backend</th><th>GPU</th><th>Endpoint</th></tr></thead><tbody>' +
      s.loaded_models.map(m =>
        `<tr><td>${m.name}</td><td>${m.backend}</td><td>${(m.gpu_ids||[]).join(',')}</td><td style="color:var(--accent2)">${m.base_url}</td></tr>`
      ).join('') + '</tbody></table>';
  }

  // Registry summary
  const reg = s.registry;
  if (reg) {
    document.getElementById('reg-summary').innerHTML =
      `<span style="color:var(--text)">${reg.model_count}</span> models registered &nbsp;·&nbsp; ` +
      `TEE v${reg.tee_version} &nbsp;·&nbsp; ` +
      `<span style="color:var(--muted)">manifest at ${reg.generated_at ? reg.generated_at.substring(0,19).replace('T',' ') : '—'}</span>`;
  }
}

// ── Models page ──────────────────────────────────────────────────────────────
function renderModels(s) {
  document.getElementById('models-ts').textContent = 'Updated: ' + ts();
  const reg = s.registry;
  if (!reg || !reg.models || Object.keys(reg.models).length === 0) {
    document.getElementById('models-table').innerHTML = '<div class="empty">No models registered.</div>';
    return;
  }
  const rows = Object.values(reg.models).map(m => {
    const tags = (m.tags||[]).map(t => `<span class="tag">${t}</span>`).join('');
    const status = m.status || 'registered';
    const isLoaded = status === 'loaded';
    const btn = m.backend === 'ollama' && !isLoaded
      ? `<button class="btn" onclick="doModelAction('load','${m.name.replace(/'/g,"\\'")}')">Load</button>`
      : isLoaded
        ? `<button class="btn" style="color:var(--red)" onclick="doModelAction('unload','${m.name.replace(/'/g,"\\'")}')">Unload</button>`
        : `<button class="btn" onclick="doModelAction('load','${m.name.replace(/'/g,"\\'")}')">Load</button>`;
    return `<tr>
      <td>${m.name}</td>
      <td style="color:var(--muted)">${m.architecture||'—'}</td>
      <td style="color:var(--muted)">${m.parameters||'—'}</td>
      <td style="color:var(--yellow)">${m.quantization||'—'}</td>
      <td style="color:var(--muted)">${m.size_gb ? m.size_gb + ' GB' : '—'}</td>
      <td style="color:var(--muted)">${m.context ? m.context.toLocaleString() : '—'}</td>
      <td>${tags}</td>
      <td><span class="badge ${status}">${status}</span></td>
      <td>${btn}</td>
    </tr>`;
  }).join('');
  document.getElementById('models-table').innerHTML =
    `<table><thead><tr>
      <th>Name</th><th>Arch</th><th>Params</th><th>Quant</th><th>Size</th><th>Context</th><th>Tags</th><th>Status</th><th>Action</th>
    </tr></thead><tbody>${rows}</tbody></table>`;
}

async function doModelAction(action, name) {
  const url = action === 'load' ? '/proxy/models/load' : '/proxy/models/unload';
  try {
    const r = await fetch(url, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({model: name}),
    });
    const data = await r.json();
    if (!r.ok) {
      alert('Error: ' + (data.error?.message || JSON.stringify(data)));
    }
  } catch(e) {
    alert('Request failed: ' + e);
  }
  await fetchStatus();
}

// ── Directories page ─────────────────────────────────────────────────────────
function renderDirs(s) {
  document.getElementById('dirs-ts').textContent = 'Updated: ' + ts();
  const reg = s.registry;
  if (!reg || !reg.models) {
    document.getElementById('dirs-list').innerHTML = '<div class="empty">No data.</div>';
    return;
  }
  // Collect unique dirs from model file paths
  const dirs = {};
  Object.values(reg.models).forEach(m => {
    if (m.file) {
      // We only have filenames, not full paths — show watched dirs from config instead
    }
  });
  // Fetch dirs from config endpoint
  fetch('/proxy/config').then(r => r.json()).then(cfg => {
    const watched = cfg.models_dirs || [];
    if (watched.length === 0) {
      document.getElementById('dirs-list').innerHTML = '<div class="empty">No directories configured.</div>';
      return;
    }
    document.getElementById('dirs-list').innerHTML = watched.map(d =>
      `<div class="dir-row">
        <div class="dir-path">${d.path || d}</div>
        <div class="dir-meta">${d.watch ? 'watched' : 'inactive'} &nbsp;·&nbsp; ${d.label || 'auto'}</div>
      </div>`
    ).join('');
    document.getElementById('dirs-ts').textContent = 'Updated: ' + ts();
  }).catch(() => {
    document.getElementById('dirs-list').innerHTML = '<div class="empty">Could not load directories.</div>';
  });
}

// ── Config page ──────────────────────────────────────────────────────────────
function renderConfig(cfg) {
  document.getElementById('config-ts').textContent = 'Updated: ' + ts();
  if (!cfg) { document.getElementById('config-view').innerHTML = '<div class="empty">No config loaded.</div>'; return; }

  const val = (v) => v !== undefined && v !== null ? v : '';

  const dirs = (cfg.models_dirs || []).map(d =>
    `<div class="dir-row"><div class="dir-path">${d.path || d}</div><div class="dir-meta">${d.watch ? 'watched' : 'inactive'} &nbsp;·&nbsp; ${d.label || 'auto'}</div></div>`
  ).join('');

  const gw = cfg.gateway || {};
  const rt = cfg.runtime || {};

  document.getElementById('config-view').innerHTML = `
    <div class="config-grid">
      <div class="config-field"><label>Gateway Host</label><input type="text" value="${val(gw.host)}" readonly/></div>
      <div class="config-field"><label>Gateway Port</label><input type="number" value="${val(gw.port)}" readonly/></div>
      <div class="config-field"><label>Idle Unload (minutes)</label><input type="number" value="${val(rt.unload_idle_after_minutes)}" readonly/></div>
      <div class="config-field"><label>Default Context</label><input type="number" value="${val(rt.context_default)}" readonly/></div>
      <div class="config-field"><label>Backend</label><input type="text" value="${val(cfg.backend)}" readonly/></div>
      <div class="config-field"><label>GPU</label><input type="text" value="${val(cfg.gpu)}" readonly/></div>
    </div>
    <div class="config-field" style="grid-column:1/-1">
      <label>Watched Model Directories</label>
      <div style="background:var(--bg3);border:1px solid var(--border);border-radius:2px;padding:10px 14px;">${dirs || '<span style="color:var(--muted)">none</span>'}</div>
    </div>`;
}

// ── Log panel ────────────────────────────────────────────────────────────────
let _logPaused = false;
let _lastLogCount = 0;
const _levelColor = { INFO: 'var(--green)', WARNING: 'var(--yellow)', ERROR: 'var(--red)', DEBUG: 'var(--muted)' };

async function fetchLogs() {
  if (_logPaused) return;
  try {
    const r = await fetch('/proxy/logs');
    const lines = await r.json();
    if (lines.length === _lastLogCount) return;
    _lastLogCount = lines.length;
    const panel = document.getElementById('log-panel');
    if (!panel) return;
    const atBottom = panel.scrollHeight - panel.scrollTop <= panel.clientHeight + 32;
    panel.innerHTML = lines.map(l => {
      const col = _levelColor[l.level] || 'var(--muted)';
      return `<div><span style="color:var(--border)">${l.t}</span> <span style="color:${col}">${l.level.padEnd(7)}</span> <span style="color:var(--accent2)">${l.name}</span> <span>${l.msg}</span></div>`;
    }).join('');
    if (atBottom) panel.scrollTop = panel.scrollHeight;
  } catch(e) {}
}

// ── Auto-refresh every 5s ────────────────────────────────────────────────────
fetchStatus();
fetchLogs();
setInterval(fetchStatus, 5000);
setInterval(fetchLogs, 2000);
</script>
</body>
</html>
"""

PROXY_ROUTES = {
    "/proxy/status": f"{TEE_API}/status",
    "/proxy/health": f"{TEE_API}/health",
    "/proxy/models": f"{TEE_API}/v1/models",
}
PROXY_POST_ROUTES = {
    "/proxy/models/load":   f"{TEE_API}/v1/models/load",
    "/proxy/models/unload": f"{TEE_API}/v1/models/unload",
}


class UIHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # suppress access log noise

    def do_GET(self):
        path = self.path.split("?")[0]

        if path == "/" or path == "/index.html":
            self._serve_html()
        elif path in PROXY_ROUTES:
            self._proxy(PROXY_ROUTES[path])
        elif path == "/proxy/config":
            self._serve_config()
        elif path == "/proxy/logs":
            self._serve_logs()
        else:
            self._404()

    def do_POST(self):
        import urllib.request, urllib.error
        path = self.path.split("?")[0]
        if path not in PROXY_POST_ROUTES:
            self._404()
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else b"{}"
            req = urllib.request.Request(
                PROXY_POST_ROUTES[path],
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as r:
                resp_body = r.read()
                status = r.status
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp_body)))
            self.end_headers()
            self.wfile.write(resp_body)
        except urllib.error.HTTPError as e:
            resp_body = e.read()
            self.send_response(e.code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp_body)))
            self.end_headers()
            self.wfile.write(resp_body)
        except Exception:
            self._503()

    def _serve_html(self):
        body = HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _proxy(self, url: str):
        import urllib.request, urllib.error
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                body = r.read()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception:
            self._503()

    def _serve_config(self):
        import urllib.request, urllib.error
        # Read tee.config from disk — config path relative to this file
        cfg_path = Path(__file__).parent.parent / "tee.config"
        try:
            raw = cfg_path.read_text()
            # Strip comment lines for clean JSON parse
            lines = [l for l in raw.splitlines() if not l.strip().startswith("//")]
            data = json.loads("\n".join(lines))
            body = json.dumps(data).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:
            self._503()

    def _serve_logs(self):
        body = json.dumps(list(_LOG_BUFFER)).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _404(self):
        self.send_response(404)
        self.end_headers()

    def _503(self):
        self.send_response(503)
        self.end_headers()


def run(host=UI_HOST, port=UI_PORT):
    server = HTTPServer((host, port), UIHandler)
    log.info(f"TEE UI listening on http://{host}:{port}/")
    server.serve_forever()


if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  TEE:ui  %(levelname)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    port = int(sys.argv[1]) if len(sys.argv) > 1 else UI_PORT
    print(f"\nTEE UI — http://0.0.0.0:{port}/\n")
    run(port=port)
