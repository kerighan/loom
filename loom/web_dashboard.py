HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>loom dashboard</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #050b14;
      --surface: #08111f;
      --surface-2: #0c1728;
      --surface-3: #111f36;
      --line: rgba(148, 163, 184, .16);
      --line-strong: rgba(148, 163, 184, .26);
      --text: #e8f1ff;
      --muted: #8ea2bf;
      --subtle: #64748b;
      --primary: #7357ff;
      --primary-2: #11c7e5;
      --success: #22c55e;
      --danger: #ef4444;
      --warning: #f59e0b;
      --shadow: 0 18px 58px rgba(0, 0, 0, .30);
      --radius: 12px;
    }

    * { box-sizing: border-box; }

    html, body {
      margin: 0;
      min-height: 100%;
      background:
        radial-gradient(circle at 12% 0%, rgba(115, 87, 255, .22), transparent 28%),
        radial-gradient(circle at 82% 8%, rgba(17, 199, 229, .16), transparent 30%),
        linear-gradient(180deg, #071225 0%, var(--bg) 42%, #030712 100%);
      color: var(--text);
      font: 14px/1.45 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    button, input, textarea, select { font: inherit; }
    button { cursor: pointer; }
    a { color: inherit; text-decoration: none; }

    .app { display: grid; grid-template-columns: 300px minmax(0, 1fr); min-height: 100vh; }

    .sidebar {
      position: sticky;
      top: 0;
      height: 100vh;
      padding: 22px;
      border-right: 1px solid var(--line);
      background: rgba(4, 10, 20, .86);
      backdrop-filter: blur(22px);
      overflow: auto;
    }

    .brand { display: flex; align-items: center; gap: 12px; margin-bottom: 22px; }
    .brand-mark {
      width: 42px;
      height: 42px;
      border-radius: 11px;
      background: linear-gradient(135deg, var(--primary), var(--primary-2));
      box-shadow: 0 18px 46px rgba(115, 87, 255, .34);
    }
    .brand h1 { margin: 0; font-size: 18px; letter-spacing: -.02em; }
    .brand p { margin: 2px 0 0; color: var(--muted); font-size: 12px; }

    .side-card, .panel, .metric, .hero {
      border: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(15, 26, 46, .78), rgba(7, 16, 30, .78));
      box-shadow: var(--shadow);
      backdrop-filter: blur(16px);
    }

    .side-card { padding: 16px; border-radius: var(--radius); margin-bottom: 16px; }
    .eyebrow { color: var(--subtle); text-transform: uppercase; letter-spacing: .16em; font-size: 10px; font-weight: 800; }
    .db-path { margin-top: 10px; color: #c9d8ee; word-break: break-all; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
    .side-row { display: flex; justify-content: space-between; gap: 12px; margin-top: 12px; color: var(--muted); font-size: 12px; }

    .pill {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
      min-height: 26px;
      padding: 5px 9px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: rgba(15, 23, 42, .74);
      color: #cfe0f7;
      font-size: 11px;
      font-weight: 700;
    }

    .input, .field input, .field textarea, .field select, .toolbar input, .toolbar select {
      width: 100%;
      border: 1px solid var(--line-strong);
      background: #0a1424;
      color: var(--text);
      border-radius: 9px;
      padding: 10px 12px;
      outline: none;
      min-height: 40px;
      box-shadow: inset 0 1px 0 rgba(255,255,255,.03);
    }

    .input::placeholder, .field input::placeholder, .field textarea::placeholder { color: #53657d; }
    .input:focus, .field input:focus, .field textarea:focus, .field select:focus, .toolbar input:focus, .toolbar select:focus {
      border-color: rgba(17, 199, 229, .68);
      box-shadow: 0 0 0 4px rgba(17, 199, 229, .08);
    }
    select option { background: #0a1424; color: var(--text); }

    .structure-list { display: flex; flex-direction: column; gap: 10px; }
    .structure {
      width: 100%;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      border: 1px solid transparent;
      border-radius: 10px;
      padding: 12px;
      background: rgba(9, 18, 32, .72);
      color: var(--text);
      text-align: left;
      transition: border-color .16s, background .16s, transform .16s;
    }
    .structure:hover, .structure.active { border-color: rgba(17, 199, 229, .32); background: rgba(17, 31, 54, .94); transform: translateY(-1px); }
    .structure strong { display: block; margin-bottom: 2px; font-size: 13px; }
    .structure small { color: var(--muted); font-size: 11px; }

    .main { min-width: 0; padding: 22px 28px 40px; }
    .topbar { display: flex; justify-content: space-between; align-items: center; gap: 16px; margin-bottom: 16px; }
    .crumb { color: var(--muted); font-size: 12px; }
    .top-actions, .actions, .toolbar { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }

    .btn {
      border: 1px solid var(--line-strong);
      border-radius: 9px;
      min-height: 40px;
      padding: 9px 13px;
      background: #0d192b;
      color: var(--text);
      font-weight: 800;
      font-size: 12px;
      transition: transform .16s, border-color .16s, background .16s;
    }
    .btn:hover { transform: translateY(-1px); border-color: rgba(17, 199, 229, .34); background: #13213a; }
    .btn.primary { border-color: transparent; background: linear-gradient(135deg, var(--primary), #2563eb); box-shadow: 0 16px 34px rgba(37, 99, 235, .22); }
    .btn.danger { border-color: rgba(239, 68, 68, .28); background: rgba(239, 68, 68, .10); color: #fecaca; }
    .btn.subtle { background: transparent; }

    .hero { display: flex; justify-content: space-between; align-items: flex-start; gap: 18px; border-radius: 12px; padding: 24px; margin-bottom: 16px; }
    .hero h2 { margin: 8px 0 6px; font-size: clamp(28px, 4vw, 42px); line-height: 1.05; letter-spacing: -.05em; }
    .hero p { margin: 0; color: var(--muted); max-width: 720px; }
    .schema-row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 14px; }

    .metrics { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-bottom: 16px; }
    .metric { border-radius: var(--radius); padding: 16px; }
    .metric span { display: block; color: var(--subtle); text-transform: uppercase; letter-spacing: .14em; font-size: 10px; font-weight: 800; margin-bottom: 8px; }
    .metric strong { display: block; font-size: 24px; letter-spacing: -.04em; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

    .workspace { display: grid; grid-template-columns: minmax(0, 1.55fr) minmax(360px, .9fr); gap: 16px; align-items: start; }
    .panel { border-radius: 12px; overflow: hidden; }
    .panel-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 14px; padding: 18px 18px 14px; border-bottom: 1px solid var(--line); }
    .panel-head h3 { margin: 0; font-size: 18px; letter-spacing: -.03em; }
    .panel-head p { margin: 5px 0 0; color: var(--muted); font-size: 12px; }
    .panel-body { padding: 18px; }

    .toolbar { margin-bottom: 14px; }
    .toolbar input, .toolbar select { width: auto; min-width: 120px; }
    .table-wrap { overflow: auto; border: 1px solid var(--line); border-radius: 10px; background: rgba(3, 8, 18, .38); }
    table { width: 100%; border-collapse: collapse; min-width: 620px; }
    th, td { padding: 12px 14px; border-bottom: 1px solid rgba(148, 163, 184, .11); text-align: left; vertical-align: top; }
    th { position: sticky; top: 0; z-index: 1; background: #091324; color: var(--subtle); text-transform: uppercase; letter-spacing: .12em; font-size: 10px; font-weight: 900; }
    tr:last-child td { border-bottom: 0; }
    tr:hover td { background: rgba(17, 31, 54, .42); }

    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
    .preview { max-width: 460px; color: #cfddf2; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .empty { display: grid; place-items: center; min-height: 150px; border: 1px dashed rgba(148, 163, 184, .24); border-radius: 10px; color: var(--muted); text-align: center; padding: 22px; background: rgba(3, 8, 18, .26); }

    .detail-card { border: 1px solid var(--line); border-radius: 10px; background: rgba(3, 8, 18, .34); padding: 14px; margin-bottom: 14px; }
    .detail-title { display: flex; justify-content: space-between; gap: 12px; align-items: center; margin-bottom: 12px; color: var(--muted); font-size: 12px; font-weight: 800; text-transform: uppercase; letter-spacing: .12em; }
    .kv-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
    .kv { border: 1px solid rgba(148, 163, 184, .12); border-radius: 9px; padding: 11px; background: rgba(10, 20, 36, .58); min-width: 0; }
    .kv span { display: block; color: var(--subtle); font-size: 11px; margin-bottom: 6px; }
    .kv strong { display: block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .json { margin: 0; padding: 13px; border-radius: 9px; background: #030817; border: 1px solid var(--line); color: #cfe2ff; white-space: pre-wrap; word-break: break-word; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; max-height: 320px; overflow: auto; }

    .form-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
    .field { display: block; min-width: 0; }
    .field.wide { grid-column: 1 / -1; }
    .field-label { display: flex; justify-content: space-between; gap: 10px; margin-bottom: 7px; color: #dce8f8; font-size: 12px; font-weight: 800; }
    .field-label small { color: var(--subtle); font-weight: 800; }
    .field textarea { min-height: 112px; resize: vertical; }
    .toggle { display: flex; align-items: center; justify-content: space-between; gap: 12px; min-height: 62px; padding: 12px; border: 1px solid var(--line-strong); border-radius: 9px; background: #0a1424; }
    .toggle input { width: 20px; height: 20px; accent-color: var(--primary); }

    .result-banner { border: 1px solid var(--line); border-radius: 10px; padding: 14px; margin-top: 14px; background: rgba(17, 31, 54, .52); }
    .result-banner.good { border-color: rgba(34, 197, 94, .28); color: #bbf7d0; }
    .result-banner.bad { border-color: rgba(245, 158, 11, .30); color: #fde68a; }
    .toast-wrap { position: fixed; right: 18px; bottom: 18px; display: flex; flex-direction: column; gap: 10px; z-index: 20; }
    .toast { min-width: 280px; max-width: 420px; border: 1px solid var(--line); border-left: 4px solid var(--primary-2); border-radius: 10px; padding: 13px 14px; background: rgba(7, 16, 30, .94); box-shadow: var(--shadow); }
    .toast.success { border-left-color: var(--success); }
    .toast.error { border-left-color: var(--danger); }
    .toast.info { border-left-color: var(--primary-2); }

    @media (max-width: 1180px) { .workspace { grid-template-columns: 1fr; } }
    @media (max-width: 900px) { .app { grid-template-columns: 1fr; } .sidebar { position: relative; height: auto; border-right: 0; border-bottom: 1px solid var(--line); } .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); } .hero, .topbar { flex-direction: column; align-items: stretch; } }
    @media (max-width: 620px) { .main { padding: 16px; } .metrics, .form-grid, .kv-grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <div class="app">
    <aside class="sidebar">
      <div class="brand"><div class="brand-mark"></div><div><h1>loom dashboard</h1><p>Native admin UI for FastAPI</p></div></div>
      <div class="side-card">
        <div class="eyebrow">Database</div>
        <div id="db-file" class="db-path">Loading…</div>
        <div class="side-row"><span>Structures</span><strong id="db-count">0</strong></div>
        <div class="side-row"><span>API reference</span><a href="/docs" class="pill">/docs</a></div>
      </div>
      <input id="structure-search" class="input" placeholder="Search structures…" />
      <div style="height: 14px"></div>
      <div id="structure-list" class="structure-list"></div>
    </aside>

    <main class="main">
      <div class="topbar"><div><div class="crumb">loom / <span id="crumb-kind">dashboard</span></div></div><div class="top-actions"><a class="btn subtle" href="/docs">Open API docs</a><button id="refresh-app" class="btn primary">Refresh</button></div></div>
      <section class="hero"><div><div class="eyebrow">Integrated dashboard</div><h2 id="hero-title">Select a structure</h2><p id="hero-subtitle">Explore records, edit typed values, and operate loom structures directly from the FastAPI server.</p><div id="schema-row" class="schema-row"></div></div><span id="kind-pill" class="pill">Ready</span></section>
      <section id="metrics" class="metrics"></section>
      <section class="workspace"><div id="browser-panel" class="panel"></div><div id="editor-panel" class="panel"></div></section>
    </main>
  </div>
  <div id="toast-wrap" class="toast-wrap"></div>

  <script>
    const PATHS = {Dict:'dicts', LRUDict:'lru_dicts', List:'lists', Set:'sets', BTree:'btrees', Queue:'queues', BloomFilter:'bloomfilters', CountingBloomFilter:'counting_bloomfilters', SearchIndex:'search_indexes', Collection:'collections', PriorityQueue:'priority_queues'};
    const state = {meta:null, current:null, kind:null, info:{}, selection:null, views:{}, filter:''};

    // Visible entries = user-facing structures + collections, hiding internals
    // (anything starting with "_" or owned by a collection, i.e. "{coll}__…").
    function visibleEntries() {
      const collections = state.meta.collections || [];
      const owned = (name) => collections.some((c) => name.startsWith(c + '__'));
      const out = {};
      for (const [name, kind] of Object.entries(state.meta.structures || {})) {
        if (name.startsWith('_') || owned(name)) continue;
        out[name] = kind;
      }
      for (const c of collections) out[c] = 'Collection';
      return out;
    }
    const $ = (selector) => document.querySelector(selector);
    const view = (name) => (state.views[name] ||= {});
    const seg = (value) => encodeURIComponent(String(value));
    const path = () => `/${PATHS[state.kind]}/${seg(state.current)}`;

    function escapeHtml(value) { return String(value ?? '').replace(/[&<>"']/g, (char) => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[char])); }
    function toast(message, kind='success') { const node = document.createElement('div'); node.className = `toast ${kind}`; node.textContent = message; $('#toast-wrap').appendChild(node); setTimeout(() => node.remove(), 3200); }
    async function api(url, options={}) { const response = await fetch(url, {headers:{'Content-Type':'application/json'}, ...options}); const text = await response.text(); let body = null; try { body = text ? JSON.parse(text) : null; } catch { body = text; } if (!response.ok) { const detail = body && typeof body === 'object' && 'detail' in body ? body.detail : body; throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail || `HTTP ${response.status}`)); } return body; }
    function dtypeKind(dtype) {
      const normalized = String(dtype || '').toLowerCase().replace(/^<class '(.+)'>$/, '$1');
      if (normalized.includes('bool') || normalized === '|b1' || normalized === 'bool_' || normalized === 'boolean') return 'bool';
      if (/(^|[^a-z])(?:u?int\d*|integer)([^a-z]|$)/.test(normalized)) return 'int';
      if (/(float|double|decimal|number)/.test(normalized)) return 'float';
      if (/(date|time)/.test(normalized)) return 'datetime';
      if (/(dict|object|json|list|array|tuple|\[|\{)/.test(normalized)) return 'json';
      if (normalized === 'text' || normalized === 'blob' || normalized.includes('str_')) return 'textarea';
      return 'text';
    }
    function defaultFor(dtype) { const kind = dtypeKind(dtype); if (kind === 'bool') return false; if (kind === 'int' || kind === 'float') return 0; if (kind === 'json') return '[]'; return ''; }
    function truncate(value, max=120) { const text = typeof value === 'string' ? value : JSON.stringify(value); if (!text) return ''; return text.length > max ? `${text.slice(0, max)}…` : text; }
    function jsonPreview(value) { return escapeHtml(truncate(value, 140)); }

    function displayValue(value) {
      if (value === undefined || value === null) return '<div class="empty">No value selected yet.</div>';
      if (value && typeof value === 'object' && !Array.isArray(value)) {
        return `<div class="kv-grid">${Object.entries(value).map(([key, fieldValue]) => `<div class="kv"><span>${escapeHtml(key)}</span><strong title="${escapeHtml(JSON.stringify(fieldValue))}">${escapeHtml(truncate(fieldValue, 80))}</strong></div>`).join('')}</div>`;
      }
      return `<pre class="json">${escapeHtml(JSON.stringify(value, null, 2))}</pre>`;
    }

    function emptyState(title, subtitle) { return `<div class="empty"><div><strong>${escapeHtml(title)}</strong><div style="height: 6px"></div><span>${escapeHtml(subtitle)}</span></div></div>`; }
    function panel(title, subtitle, body, action='') { return `<div class="panel-head"><div><h3>${escapeHtml(title)}</h3><p>${escapeHtml(subtitle)}</p></div>${action}</div><div class="panel-body">${body}</div>`; }
    function renderTable(headers, rows, emptyTitle='No data', emptySubtitle='Nothing to display yet.') { if (!rows.length) return emptyState(emptyTitle, emptySubtitle); return `<div class="table-wrap"><table><thead><tr>${headers.map((header) => `<th>${escapeHtml(header)}</th>`).join('')}</tr></thead><tbody>${rows.join('')}</tbody></table></div>`; }
    function recordColumns(schema, records, leading=[], action='Action') {
      const schemaFields = Object.keys(schema || {});
      const discoveredFields = [];
      for (const record of records) {
        const value = record.value ?? record;
        if (value && typeof value === 'object' && !Array.isArray(value)) {
          for (const field of Object.keys(value)) {
            if (!schemaFields.includes(field) && !discoveredFields.includes(field)) discoveredFields.push(field);
          }
        }
      }
      return [...leading, ...schemaFields, ...discoveredFields].slice(0, 10).concat(action ? [action] : []);
    }
    function recordCell(value) {
      if (value === true) return '<span class="pill">true</span>';
      if (value === false) return '<span class="pill">false</span>';
      if (value === undefined || value === null) return '<span class="preview">—</span>';
      return `<span class="preview" title="${escapeHtml(JSON.stringify(value))}">${escapeHtml(truncate(value, 80))}</span>`;
    }

    function renderForm(schema, values={}, prefix='form') {
      const entries = Object.entries(schema || {});
      if (!entries.length) return `<label class="field wide"><span class="field-label">JSON value <small>object</small></span><textarea id="${prefix}__json">${escapeHtml(JSON.stringify(values || {}, null, 2))}</textarea></label>`;
      return `<div class="form-grid">${entries.map(([field, dtype]) => {
        const id = `${prefix}__${field}`;
        const kind = dtypeKind(dtype);
        const value = values[field] ?? defaultFor(dtype);
        if (kind === 'bool') return `<label class="toggle"><span class="field-label" style="margin:0">${escapeHtml(field)} <small>${escapeHtml(dtype)}</small></span><input id="${id}" type="checkbox" ${value ? 'checked' : ''} /></label>`;
        if (kind === 'textarea' || kind === 'json') { const text = kind === 'json' && typeof value !== 'string' ? JSON.stringify(value, null, 2) : value; return `<label class="field wide"><span class="field-label">${escapeHtml(field)} <small>${escapeHtml(dtype)}</small></span><textarea id="${id}">${escapeHtml(text)}</textarea></label>`; }
        const inputType = kind === 'int' || kind === 'float' ? 'number' : kind === 'datetime' ? 'datetime-local' : 'text';
        const step = kind === 'int' ? ' step="1"' : kind === 'float' ? ' step="any"' : '';
        return `<label class="field"><span class="field-label">${escapeHtml(field)} <small>${escapeHtml(dtype)}</small></span><input id="${id}" type="${inputType}"${step} value="${escapeHtml(value)}" /></label>`;
      }).join('')}</div>`;
    }

    function readForm(schema, prefix='form') {
      if (!schema || !Object.keys(schema).length) return JSON.parse(document.getElementById(`${prefix}__json`).value || '{}');
      const out = {};
      for (const [field, dtype] of Object.entries(schema)) {
        const element = document.getElementById(`${prefix}__${field}`);
        const kind = dtypeKind(dtype);
        if (kind === 'bool') out[field] = element.checked;
        else if (kind === 'int') out[field] = parseInt(element.value || '0', 10);
        else if (kind === 'float') out[field] = parseFloat(element.value || '0');
        else if (kind === 'json') out[field] = JSON.parse(element.value || '[]');
        else out[field] = element.value;
      }
      return out;
    }

    function renderSidebar() {
      const entries = visibleEntries();
      const names = Object.keys(entries).sort();
      const filtered = names.filter((name) => name.toLowerCase().includes(state.filter.toLowerCase()));
      $('#db-file').textContent = state.meta.filename || '';
      $('#db-count').textContent = names.length;
      $('#structure-list').innerHTML = filtered.length ? filtered.map((name) => `<button class="structure ${name === state.current ? 'active' : ''}" data-name="${escapeHtml(name)}"><span><strong>${escapeHtml(name)}</strong><small>${escapeHtml(entries[name])}</small></span><span class="pill">open</span></button>`).join('') : emptyState('No match', 'Try another search.');
      document.querySelectorAll('[data-name]').forEach((button) => button.addEventListener('click', () => selectStructure(button.dataset.name)));
    }

    function renderMetrics() {
      const info = state.info || {};
      const values = [['Type', state.kind], ['Name', info.name || state.current]];
      const labels = {
        length: 'Length',
        capacity: 'Capacity',
        expected_items: 'Expected',
        false_positive_rate: 'False positive',
        num_bits: 'Bits',
        num_buckets: 'Buckets',
        num_hashes: 'Hashes',
        max_count: 'Max count',
        scoring: 'Scoring',
        primary_field: 'Primary key',
      };
      for (const [key, label] of Object.entries(labels)) {
        if (info[key] !== undefined) values.push([label, info[key]]);
      }
      if (info.indexes) values.push(['Indexes', Object.keys(info.indexes).length]);
      if (info.schema) values.push(['Fields', Object.keys(info.schema).length]);
      $('#metrics').innerHTML = values.map(([label, value]) => `<article class="metric"><span>${escapeHtml(label)}</span><strong title="${escapeHtml(value)}">${escapeHtml(value)}</strong></article>`).join('');
    }

    function renderHeader() {
      const schema = state.info.schema || {};
      $('#crumb-kind').textContent = state.kind || 'dashboard';
      $('#hero-title').textContent = state.current || 'Select a structure';
      $('#hero-subtitle').textContent = state.current ? `${state.kind} stored in ${state.meta.filename}` : 'Explore records, edit typed values, and operate loom structures directly from the FastAPI server.';
      $('#kind-pill').textContent = state.kind || 'Ready';
      $('#schema-row').innerHTML = Object.keys(schema).length ? Object.entries(schema).map(([field, dtype]) => `<span class="pill">${escapeHtml(field)} · ${escapeHtml(dtype)}</span>`).join('') : '<span class="pill">No schema</span>';
    }

    async function loadInfo() { try { return await api(path()); } catch { return {name: state.current}; } }
    async function selectStructure(name) { state.current = name; state.kind = visibleEntries()[name]; state.selection = null; location.hash = encodeURIComponent(name); renderSidebar(); $('#browser-panel').innerHTML = panel('Loading', 'Fetching data from the API.', emptyState('Loading', 'Please wait…')); $('#editor-panel').innerHTML = panel('Editor', 'Preparing controls.', emptyState('Loading', 'Please wait…')); state.info = await loadInfo(); renderHeader(); renderMetrics(); await renderCurrent(); }
    async function renderCurrent() { renderHeader(); renderMetrics(); if (state.kind === 'Dict' || state.kind === 'LRUDict') return renderDictLike(); if (state.kind === 'List') return renderList(); if (state.kind === 'Set') return renderSet(); if (state.kind === 'BTree') return renderBTree(); if (state.kind === 'Queue') return renderQueue(); if (state.kind === 'BloomFilter' || state.kind === 'CountingBloomFilter') return renderBloom(); if (state.kind === 'SearchIndex') return renderSearch(); if (state.kind === 'Collection') return renderCollection(); if (state.kind === 'PriorityQueue') return renderPQ(); $('#browser-panel').innerHTML = panel('Unsupported', state.kind, emptyState('Unsupported structure', 'This structure does not have a dedicated dashboard view yet.')); $('#editor-panel').innerHTML = panel('Details', 'No editor available.', emptyState('No editor', 'Use the HTTP API directly for this structure.')); }
    async function openKey(key) { state.selection = {key, value: await api(`${path()}/items/${seg(key)}`)}; await renderCurrent(); }

    async function renderDictLike() {
      const local = view(state.current); local.limit ||= 50;
      const keys = state.kind === 'Dict' ? await api(`${path()}/keys?limit=${local.limit}`) : [];
      const rows = keys.map((key) => `<tr><td class="mono">${escapeHtml(key)}</td><td class="preview">Click open to inspect and edit this record.</td><td><button class="btn" data-open-key="${escapeHtml(key)}">Open</button></td></tr>`);
      const table = state.kind === 'Dict' ? renderTable(['Key', 'Preview', 'Action'], rows, 'No keys found', 'Write a first record from the editor.') : emptyState('Direct key access', 'LRUDict currently supports read/write/delete by known key only.');
      $('#browser-panel').innerHTML = panel('Records', state.kind === 'Dict' ? 'Browse keys and load records into the editor.' : 'Read or mutate a known key.', `${state.kind === 'Dict' ? `<div class="toolbar"><input id="dict-limit" type="number" min="1" value="${local.limit}" /><button id="dict-refresh" class="btn">Refresh keys</button></div>` : ''}${table}${state.selection ? `<div class="detail-card" style="margin-top: 14px"><div class="detail-title"><span>Selected record</span><span class="pill">${escapeHtml(state.selection.key)}</span></div>${displayValue(state.selection.value)}</div>` : ''}`);
      $('#editor-panel').innerHTML = panel('Editor', 'Read, write, or delete by key.', `<label class="field wide"><span class="field-label">Key <small>string</small></span><input id="dict-key" value="${escapeHtml(state.selection?.key || '')}" placeholder="alice" /></label><div style="height: 14px"></div>${renderForm(state.info.schema || {}, state.selection?.value || {}, 'dict-form')}<div class="actions" style="margin-top: 16px"><button id="dict-read" class="btn">Read</button><button id="dict-write" class="btn primary">Save record</button><button id="dict-delete" class="btn danger">Delete</button></div>`);
      if (state.kind === 'Dict') { $('#dict-refresh').addEventListener('click', async () => { local.limit = parseInt($('#dict-limit').value || '50', 10); await renderCurrent(); }); document.querySelectorAll('[data-open-key]').forEach((button) => button.addEventListener('click', async () => { try { await openKey(button.dataset.openKey); } catch (error) { toast(error.message, 'error'); } })); }
      $('#dict-read').addEventListener('click', async () => { const key = $('#dict-key').value.trim(); if (!key) return toast('Enter a key first.', 'info'); try { await openKey(key); } catch (error) { toast(error.message, 'error'); } });
      $('#dict-write').addEventListener('click', async () => { const key = $('#dict-key').value.trim(); if (!key) return toast('Enter a key first.', 'info'); try { const payload = readForm(state.info.schema || {}, 'dict-form'); await api(`${path()}/items/${seg(key)}`, {method:'PUT', body:JSON.stringify(payload)}); state.selection = {key, value: payload}; state.info = await loadInfo(); await renderCurrent(); toast('Record saved.'); } catch (error) { toast(error.message, 'error'); } });
      $('#dict-delete').addEventListener('click', async () => { const key = $('#dict-key').value.trim(); if (!key) return toast('Enter a key first.', 'info'); try { await api(`${path()}/items/${seg(key)}`, {method:'DELETE'}); state.selection = null; state.info = await loadInfo(); await renderCurrent(); toast('Record deleted.'); } catch (error) { toast(error.message, 'error'); } });
    }

    async function renderList() {
      const local = view(state.current); local.page ||= 0; local.size ||= 20;
      const start = local.page * local.size; const end = start + local.size;
      const items = await api(`${path()}/items?start=${start}&end=${end}`);
      const listHeaders = recordColumns(state.info.schema || {}, items, ['Index']);
      const listFields = listHeaders.slice(1, -1);
      const rows = items.map((item, index) => { const absoluteIndex = start + index; return `<tr><td class="mono">#${absoluteIndex}</td>${listFields.map((field) => `<td>${recordCell(item?.[field])}</td>`).join('')}<td><button class="btn" data-open-index="${absoluteIndex}">Open</button></td></tr>`; });
      $('#browser-panel').innerHTML = panel('Items', 'Paginate the list and load items into the editor.', `<div class="toolbar"><button id="list-prev" class="btn">Previous</button><input id="list-page" type="number" min="0" value="${local.page}" /><input id="list-size" type="number" min="1" value="${local.size}" /><button id="list-next" class="btn">Next</button><button id="list-go" class="btn primary">Apply</button></div>${renderTable(listHeaders, rows, 'No items on this page', 'Try another page or append a new item.')}`);
      $('#editor-panel').innerHTML = panel('Item editor', 'The index is only the list position used to select an existing item; append creates the next index automatically.', `<label class="field wide"><span class="field-label">Selected index <small>technical position, not a data field</small></span><input id="list-index" type="number" min="0" value="${state.selection?.index ?? start}" /></label><div style="height: 14px"></div>${state.selection ? `<div class="detail-card"><div class="detail-title"><span>Selected item</span><span class="pill">#${state.selection.index}</span></div>${displayValue(state.selection.value)}</div>` : ''}${renderForm(state.info.schema || {}, state.selection?.value || {}, 'list-form')}<div class="actions" style="margin-top: 16px"><button id="list-read" class="btn">Load index</button><button id="list-update" class="btn primary">Update selected</button><button id="list-append" class="btn">Append as new item</button><button id="list-delete" class="btn danger">Delete selected</button></div>`);
      const applyPagination = async () => { local.page = Math.max(0, parseInt($('#list-page').value || '0', 10)); local.size = Math.max(1, parseInt($('#list-size').value || '20', 10)); await renderCurrent(); };
      $('#list-prev').addEventListener('click', async () => { local.page = Math.max(0, local.page - 1); await renderCurrent(); });
      $('#list-next').addEventListener('click', async () => { local.page += 1; await renderCurrent(); });
      $('#list-go').addEventListener('click', applyPagination);
      document.querySelectorAll('[data-open-index]').forEach((button) => button.addEventListener('click', async () => { try { const index = parseInt(button.dataset.openIndex, 10); state.selection = {index, value: await api(`${path()}/items/${index}`)}; await renderCurrent(); } catch (error) { toast(error.message, 'error'); } }));
      $('#list-read').addEventListener('click', async () => { try { const index = parseInt($('#list-index').value || '0', 10); state.selection = {index, value: await api(`${path()}/items/${index}`)}; await renderCurrent(); } catch (error) { toast(error.message, 'error'); } });
      $('#list-update').addEventListener('click', async () => { try { const index = parseInt($('#list-index').value || '0', 10); const payload = readForm(state.info.schema || {}, 'list-form'); await api(`${path()}/items/${index}`, {method:'PUT', body:JSON.stringify(payload)}); state.selection = {index, value: payload}; state.info = await loadInfo(); await renderCurrent(); toast('Item updated.'); } catch (error) { toast(error.message, 'error'); } });
      $('#list-append').addEventListener('click', async () => { try { const payload = readForm(state.info.schema || {}, 'list-form'); const result = await api(`${path()}/items`, {method:'POST', body:JSON.stringify(payload)}); state.selection = {index: result.index, value: payload}; state.info = await loadInfo(); await renderCurrent(); toast(`Appended at #${result.index}.`); } catch (error) { toast(error.message, 'error'); } });
      $('#list-delete').addEventListener('click', async () => { try { const index = parseInt($('#list-index').value || '0', 10); await api(`${path()}/items/${index}`, {method:'DELETE'}); state.selection = null; state.info = await loadInfo(); await renderCurrent(); toast('Item deleted.'); } catch (error) { toast(error.message, 'error'); } });
    }

    async function renderSet() {
      const local = view(state.current); local.limit ||= 100;
      const members = await api(`${path()}/members?limit=${local.limit}`);
      const rows = members.map((member) => `<tr><td class="mono">${escapeHtml(member)}</td><td><button class="btn" data-member="${escapeHtml(member)}">Use</button></td></tr>`);
      $('#browser-panel').innerHTML = panel('Members', 'Browse members currently stored in the set.', `<div class="toolbar"><input id="set-limit" type="number" min="1" value="${local.limit}" /><button id="set-refresh" class="btn primary">Refresh</button></div>${renderTable(['Member', 'Action'], rows, 'No members found', 'Add a first member from the action panel.')}`);
      $('#editor-panel').innerHTML = panel('Actions', 'Check membership or mutate the set.', `<label class="field wide"><span class="field-label">Item <small>string</small></span><input id="set-item" value="${escapeHtml(state.selection?.item || '')}" /></label><div class="actions" style="margin-top: 16px"><button id="set-contains" class="btn">Contains</button><button id="set-add" class="btn primary">Add</button><button id="set-remove" class="btn danger">Remove</button></div>${state.selection?.result !== undefined ? `<div class="result-banner ${state.selection.result ? 'good' : 'bad'}">${state.selection.result ? 'Item is in the set.' : 'Item is not in the set.'}</div>` : ''}`);
      $('#set-refresh').addEventListener('click', async () => { local.limit = parseInt($('#set-limit').value || '100', 10); await renderCurrent(); });
      document.querySelectorAll('[data-member]').forEach((button) => button.addEventListener('click', () => { state.selection = {item: button.dataset.member}; renderCurrent(); }));
      $('#set-contains').addEventListener('click', async () => { const item = $('#set-item').value.trim(); if (!item) return toast('Enter an item first.', 'info'); try { const result = await api(`${path()}/contains/${seg(item)}`); state.selection = {item, result: result.contains}; await renderCurrent(); } catch (error) { toast(error.message, 'error'); } });
      $('#set-add').addEventListener('click', async () => { const item = $('#set-item').value.trim(); if (!item) return toast('Enter an item first.', 'info'); try { await api(`${path()}/members`, {method:'POST', body:JSON.stringify({item})}); state.selection = {item, result: true}; state.info = await loadInfo(); await renderCurrent(); toast('Member added.'); } catch (error) { toast(error.message, 'error'); } });
      $('#set-remove').addEventListener('click', async () => { const item = $('#set-item').value.trim(); if (!item) return toast('Enter an item first.', 'info'); try { await api(`${path()}/members/${seg(item)}`, {method:'DELETE'}); state.selection = {item, result: false}; state.info = await loadInfo(); await renderCurrent(); toast('Member removed.'); } catch (error) { toast(error.message, 'error'); } });
    }

    async function renderBTree() {
      const local = view(state.current); local.mode ||= 'range'; local.limit ||= 50;
      const url = local.mode === 'prefix' && local.prefix ? `${path()}/prefix/${seg(local.prefix)}?limit=${local.limit}` : `${path()}/range?limit=${local.limit}${local.start ? `&start=${encodeURIComponent(local.start)}` : ''}${local.end ? `&end=${encodeURIComponent(local.end)}` : ''}${local.reverse ? '&reverse=true' : ''}`;
      const results = await api(url);
      const btreeHeaders = recordColumns(state.info.schema || {}, results, ['Key']);
      const btreeFields = btreeHeaders.slice(1, -1);
      const rows = results.map((row) => `<tr><td class="mono">${escapeHtml(row.key)}</td>${btreeFields.map((field) => `<td>${recordCell(row.value?.[field])}</td>`).join('')}<td><button class="btn" data-btree-key="${escapeHtml(row.key)}">Open</button></td></tr>`);
      $('#browser-panel').innerHTML = panel('Browse', 'Run range or prefix queries and load results into the editor.', `<div class="toolbar"><select id="btree-mode"><option value="range" ${local.mode !== 'prefix' ? 'selected' : ''}>Range</option><option value="prefix" ${local.mode === 'prefix' ? 'selected' : ''}>Prefix</option></select><input id="btree-start" placeholder="start" value="${escapeHtml(local.start || '')}" /><input id="btree-end" placeholder="end" value="${escapeHtml(local.end || '')}" /><input id="btree-prefix" placeholder="prefix" value="${escapeHtml(local.prefix || '')}" /><input id="btree-limit" type="number" min="1" value="${local.limit}" /><label class="toggle" style="margin:0"><input id="btree-reverse" type="checkbox" ${local.reverse ? 'checked' : ''} /> reverse</label><button id="btree-run" class="btn primary">Run query</button></div>${renderTable(btreeHeaders, rows, 'No keys matched', 'Try another range or prefix.')}`);
      $('#editor-panel').innerHTML = panel('Editor', 'Read, write, or delete a single key.', `<label class="field wide"><span class="field-label">Key <small>string</small></span><input id="btree-key" value="${escapeHtml(state.selection?.key || '')}" /></label><div style="height: 14px"></div>${state.selection ? `<div class="detail-card"><div class="detail-title"><span>Selected key</span><span class="pill">${escapeHtml(state.selection.key)}</span></div>${displayValue(state.selection.value)}</div>` : ''}${renderForm(state.info.schema || {}, state.selection?.value || {}, 'btree-form')}<div class="actions" style="margin-top: 16px"><button id="btree-read" class="btn">Read</button><button id="btree-write" class="btn primary">Save key</button><button id="btree-delete" class="btn danger">Delete</button></div>`);
      $('#btree-run').addEventListener('click', async () => { local.mode = $('#btree-mode').value; local.start = $('#btree-start').value.trim(); local.end = $('#btree-end').value.trim(); local.prefix = $('#btree-prefix').value.trim(); local.limit = parseInt($('#btree-limit').value || '50', 10); local.reverse = $('#btree-reverse').checked; await renderCurrent(); });
      document.querySelectorAll('[data-btree-key]').forEach((button) => button.addEventListener('click', async () => { try { const key = button.dataset.btreeKey; state.selection = {key, value: await api(`${path()}/items/${seg(key)}`)}; await renderCurrent(); } catch (error) { toast(error.message, 'error'); } }));
      $('#btree-read').addEventListener('click', async () => { const key = $('#btree-key').value.trim(); if (!key) return toast('Enter a key first.', 'info'); try { state.selection = {key, value: await api(`${path()}/items/${seg(key)}`)}; await renderCurrent(); } catch (error) { toast(error.message, 'error'); } });
      $('#btree-write').addEventListener('click', async () => { const key = $('#btree-key').value.trim(); if (!key) return toast('Enter a key first.', 'info'); try { const payload = readForm(state.info.schema || {}, 'btree-form'); await api(`${path()}/items/${seg(key)}`, {method:'PUT', body:JSON.stringify(payload)}); state.selection = {key, value: payload}; state.info = await loadInfo(); await renderCurrent(); toast('Key saved.'); } catch (error) { toast(error.message, 'error'); } });
      $('#btree-delete').addEventListener('click', async () => { const key = $('#btree-key').value.trim(); if (!key) return toast('Enter a key first.', 'info'); try { await api(`${path()}/items/${seg(key)}`, {method:'DELETE'}); state.selection = null; state.info = await loadInfo(); await renderCurrent(); toast('Key deleted.'); } catch (error) { toast(error.message, 'error'); } });
    }

    async function renderQueue() {
      $('#browser-panel').innerHTML = panel('Queue head', 'Peek without removing, or pop the next item.', `<div class="actions"><button id="queue-peek" class="btn">Peek</button><button id="queue-pop" class="btn danger">Pop</button></div><div style="height: 14px"></div>${state.selection ? displayValue(state.selection.value) : emptyState('No queue item loaded', 'Peek or pop to inspect the queue head.')}`);
      $('#editor-panel').innerHTML = panel('Push item', 'Create a typed record and enqueue it.', `${renderForm(state.info.schema || {}, {}, 'queue-form')}<div class="actions" style="margin-top: 16px"><button id="queue-push" class="btn primary">Push item</button></div>`);
      $('#queue-peek').addEventListener('click', async () => { try { state.selection = {value: await api(`${path()}/peek`)}; await renderCurrent(); } catch (error) { toast(error.message, 'error'); } });
      $('#queue-pop').addEventListener('click', async () => { try { state.selection = {value: await api(`${path()}/pop`, {method:'POST'})}; state.info = await loadInfo(); await renderCurrent(); toast('Item popped.'); } catch (error) { toast(error.message, 'error'); } });
      $('#queue-push').addEventListener('click', async () => { try { const payload = readForm(state.info.schema || {}, 'queue-form'); await api(`${path()}/push`, {method:'POST', body:JSON.stringify(payload)}); state.info = await loadInfo(); await renderCurrent(); toast('Item pushed.'); } catch (error) { toast(error.message, 'error'); } });
    }

    async function renderBloom() {
      $('#browser-panel').innerHTML = panel('Membership', 'Run probabilistic membership checks.', `${state.selection?.result !== undefined ? `<div class="result-banner ${state.selection.result ? 'good' : 'bad'}">${state.selection.result ? 'Probably present.' : 'Definitely absent.'}</div>` : emptyState('No check yet', 'Enter an item and run contains.')}`);
      $('#editor-panel').innerHTML = panel('Actions', 'Check or update the filter.', `<label class="field wide"><span class="field-label">Item <small>value</small></span><input id="bloom-item" value="${escapeHtml(state.selection?.item || '')}" /></label><div class="actions" style="margin-top: 16px"><button id="bloom-contains" class="btn">Contains</button><button id="bloom-add" class="btn primary">Add</button>${state.kind === 'CountingBloomFilter' ? '<button id="bloom-remove" class="btn danger">Remove</button>' : ''}</div>`);
      $('#bloom-contains').addEventListener('click', async () => { const item = $('#bloom-item').value.trim(); if (!item) return toast('Enter an item first.', 'info'); try { const result = await api(`${path()}/contains/${seg(item)}`); state.selection = {item, result: result.contains}; await renderCurrent(); } catch (error) { toast(error.message, 'error'); } });
      $('#bloom-add').addEventListener('click', async () => { const item = $('#bloom-item').value.trim(); if (!item) return toast('Enter an item first.', 'info'); try { await api(`${path()}/items`, {method:'POST', body:JSON.stringify({item})}); state.selection = {item, result: true}; await renderCurrent(); toast('Item added.'); } catch (error) { toast(error.message, 'error'); } });
      if (state.kind === 'CountingBloomFilter') $('#bloom-remove').addEventListener('click', async () => { const item = $('#bloom-item').value.trim(); if (!item) return toast('Enter an item first.', 'info'); try { await api(`${path()}/items/${seg(item)}`, {method:'DELETE'}); state.selection = {item, result: false}; await renderCurrent(); toast('Item removed.'); } catch (error) { toast(error.message, 'error'); } });
    }

    function resultRows(schema, records, withScore) {
      const norm = records.map((r) => (r && typeof r === 'object' && 'score' in r && ('document' in r || 'doc_id' in r)) ? {record: r.document ?? {doc_id: r.doc_id}, score: r.score} : {record: r, score: null});
      const fields = recordColumns(schema, norm.map((n) => n.record), [], '');
      const headers = [...fields, ...(withScore ? ['Score'] : [])];
      const rows = norm.map((n) => `<tr>${fields.map((f) => `<td>${recordCell(n.record?.[f])}</td>`).join('')}${withScore ? `<td>${n.score == null ? '—' : escapeHtml(typeof n.score === 'number' ? n.score.toFixed(4) : n.score)}</td>` : ''}</tr>`);
      return renderTable(headers, rows, 'No matches', 'Try another query.');
    }

    async function renderSearch() {
      const local = view('search:' + state.current);
      if (local.limit === undefined) local.limit = 20;
      const schema = state.info.schema || {};
      const scored = state.info.scoring && state.info.scoring !== 'boolean';
      const table = resultRows(schema, local.results || [], !!local.withScores);
      $('#browser-panel').innerHTML = panel('Search', 'Boolean (AND / OR / AND NOT, *) or ranked full-text queries.', `<div class="toolbar"><input id="si-q" placeholder="query" value="${escapeHtml(local.q || '')}" style="flex:1" /><select id="si-mode"><option value="">default</option><option value="boolean" ${local.mode === 'boolean' ? 'selected' : ''}>boolean</option>${scored ? `<option value="bm25" ${local.mode === 'bm25' ? 'selected' : ''}>bm25</option>` : ''}</select><input id="si-limit" type="number" min="1" value="${local.limit}" /><label class="toggle" style="margin:0"><input id="si-scores" type="checkbox" ${local.withScores ? 'checked' : ''} /> scores</label><button id="si-run" class="btn primary">Search</button></div>${table}`);
      $('#editor-panel').innerHTML = panel('Documents', 'Index a new document, or read / delete one by doc-id.', `${renderForm(schema, {}, 'si-form')}<div class="actions" style="margin-top: 16px"><button id="si-add" class="btn primary">Index document</button></div><div style="height: 18px"></div><label class="field"><span class="field-label">Doc-id <small>int</small></span><input id="si-docid" type="number" min="0" value="${escapeHtml(state.selection?.id ?? '')}" /></label><div class="actions" style="margin-top: 14px"><button id="si-read" class="btn">Read</button><button id="si-del" class="btn danger">Delete</button></div>${state.selection?.value !== undefined ? `<div class="detail-card" style="margin-top: 14px"><div class="detail-title"><span>Document</span><span class="pill">#${escapeHtml(state.selection.id)}</span></div>${displayValue(state.selection.value)}</div>` : ''}`);
      $('#si-run').addEventListener('click', async () => { local.q = $('#si-q').value.trim(); local.mode = $('#si-mode').value; local.limit = parseInt($('#si-limit').value || '20', 10); local.withScores = $('#si-scores').checked; if (!local.q) return toast('Enter a query first.', 'info'); try { const qs = new URLSearchParams({q: local.q, limit: local.limit}); if (local.mode) qs.set('mode', local.mode); if (local.withScores) qs.set('with_scores', 'true'); local.results = await api(`${path()}/search?${qs}`); await renderCurrent(); } catch (error) { toast(error.message, 'error'); } });
      $('#si-add').addEventListener('click', async () => { try { const payload = readForm(schema, 'si-form'); const r = await api(`${path()}/documents`, {method:'POST', body:JSON.stringify(payload)}); state.info = await loadInfo(); await renderCurrent(); toast(`Indexed as doc #${r.doc_id}.`); } catch (error) { toast(error.message, 'error'); } });
      $('#si-read').addEventListener('click', async () => { const id = $('#si-docid').value.trim(); if (id === '') return toast('Enter a doc-id first.', 'info'); try { state.selection = {id, value: await api(`${path()}/documents/${seg(id)}`)}; await renderCurrent(); } catch (error) { toast(error.message, 'error'); } });
      $('#si-del').addEventListener('click', async () => { const id = $('#si-docid').value.trim(); if (id === '') return toast('Enter a doc-id first.', 'info'); try { await api(`${path()}/documents/${seg(id)}`, {method:'DELETE'}); state.selection = null; state.info = await loadInfo(); await renderCurrent(); toast('Document deleted.'); } catch (error) { toast(error.message, 'error'); } });
    }

    async function renderPQ() {
      const schema = state.info.schema || {};
      $('#browser-panel').innerHTML = panel('Top of queue', `${state.info.max_first === false ? 'Lowest' : 'Highest'} priority pops first; ties are FIFO.`, `<div class="actions"><button id="pq-peek" class="btn">Peek</button><button id="pq-pop" class="btn danger">Pop</button></div><div style="height: 14px"></div>${state.selection ? displayValue(state.selection.value) : emptyState('No item loaded', 'Peek or pop to inspect the top item.')}`);
      $('#editor-panel').innerHTML = panel('Push item', 'Create a typed item and enqueue it with a priority.', `${renderForm(schema, {}, 'pq-form')}<div style="height: 12px"></div><label class="field"><span class="field-label">Priority <small>number</small></span><input id="pq-priority" type="number" step="any" value="0" /></label><div class="actions" style="margin-top: 16px"><button id="pq-push" class="btn primary">Push item</button></div>`);
      $('#pq-peek').addEventListener('click', async () => { try { state.selection = {value: await api(`${path()}/peek`)}; await renderCurrent(); } catch (error) { toast(error.message, 'error'); } });
      $('#pq-pop').addEventListener('click', async () => { try { state.selection = {value: await api(`${path()}/pop`, {method:'POST'})}; state.info = await loadInfo(); await renderCurrent(); toast('Item popped.'); } catch (error) { toast(error.message, 'error'); } });
      $('#pq-push').addEventListener('click', async () => { try { const item = readForm(schema, 'pq-form'); const priority = parseFloat($('#pq-priority').value || '0'); await api(`${path()}/push`, {method:'POST', body:JSON.stringify({item, priority})}); state.info = await loadInfo(); await renderCurrent(); toast('Item pushed.'); } catch (error) { toast(error.message, 'error'); } });
    }

    function collQueryInputs(kind, local) {
      if (kind === 'range') return `<input id="cq-low" placeholder="low" value="${escapeHtml(local.low || '')}" /><input id="cq-high" placeholder="high" value="${escapeHtml(local.high || '')}" /><label class="toggle" style="margin:0"><input id="cq-desc" type="checkbox" ${local.desc ? 'checked' : ''} /> desc</label>`;
      if (kind === 'search') return `<input id="cq-q" placeholder="query" value="${escapeHtml(local.q || '')}" style="flex:1" /><select id="cq-mode"><option value="">default</option><option value="boolean" ${local.mode === 'boolean' ? 'selected' : ''}>boolean</option><option value="bm25" ${local.mode === 'bm25' ? 'selected' : ''}>bm25</option></select>`;
      if (kind === 'many' || kind === 'unique') return `<input id="cq-value" placeholder="value" value="${escapeHtml(local.value || '')}" style="flex:1" />`;
      return '';
    }

    async function renderCollection() {
      const local = view('coll:' + state.current);
      if (local.limit === undefined) local.limit = 50;
      const schema = state.info.schema || {};
      const indexes = state.info.indexes || {};
      const pkField = state.info.primary_field;
      if (local.qIndex === undefined) local.qIndex = '__all__';
      const kind = local.qIndex === '__all__' ? 'primary' : indexes[local.qIndex];
      const options = `<option value="__all__" ${local.qIndex === '__all__' ? 'selected' : ''}>(all records)</option>` + Object.entries(indexes).map(([f, k]) => `<option value="${escapeHtml(f)}" ${local.qIndex === f ? 'selected' : ''}>${escapeHtml(f)} · ${escapeHtml(k)}</option>`).join('');
      const records = local.rows || [];
      const norm = records.map((r) => (r && typeof r === 'object' && 'pk' in r && 'record' in r) ? r : {pk: r?.[pkField], record: r});
      const fields = recordColumns(schema, norm.map((n) => n.record), ['Key'], 'Action');
      const dataFields = fields.slice(1, -1);
      const rows = norm.map((n) => `<tr><td class="mono">${escapeHtml(n.pk)}</td>${dataFields.map((f) => `<td>${recordCell(n.record?.[f])}</td>`).join('')}<td><button class="btn" data-coll-key="${escapeHtml(n.pk)}">Open</button></td></tr>`);
      $('#browser-panel').innerHTML = panel('Records', 'Browse all records or query an attached index.', `<div class="toolbar"><select id="cq-index">${options}</select>${collQueryInputs(kind, local)}<input id="cq-limit" type="number" min="1" value="${local.limit}" /><button id="cq-run" class="btn primary">Run</button></div>${renderTable(fields, rows, 'No records', 'Insert a record from the editor.')}`);
      $('#editor-panel').innerHTML = panel('Record', 'Insert, read, update, delete or increment a record.', `<label class="field wide"><span class="field-label">Primary key <small>${escapeHtml(pkField || 'pk')}</small></span><input id="coll-key" value="${escapeHtml(state.selection?.key || '')}" /></label><div style="height: 12px"></div>${renderForm(schema, state.selection?.value || {}, 'coll-form')}<div class="actions" style="margin-top: 16px"><button id="coll-read" class="btn">Read</button><button id="coll-insert" class="btn primary">Insert</button><button id="coll-update" class="btn">Update</button><button id="coll-delete" class="btn danger">Delete</button></div><div style="height: 16px"></div><div class="toolbar"><input id="coll-incfield" placeholder="field" /><input id="coll-incamt" type="number" value="1" /><button id="coll-inc" class="btn">Increment</button></div>${state.selection?.value ? `<div class="detail-card" style="margin-top: 14px"><div class="detail-title"><span>Selected record</span><span class="pill">${escapeHtml(state.selection.key)}</span></div>${displayValue(state.selection.value)}</div>` : ''}`);
      $('#cq-index').addEventListener('change', async () => { local.qIndex = $('#cq-index').value; local.rows = null; await renderCurrent(); });
      $('#cq-run').addEventListener('click', async () => { local.limit = parseInt($('#cq-limit').value || '50', 10); try { let rows; if (kind === 'range') { local.low = ($('#cq-low').value || '').trim(); local.high = ($('#cq-high').value || '').trim(); local.desc = $('#cq-desc').checked; const qs = new URLSearchParams(); if (local.low !== '') qs.set('low', local.low); if (local.high !== '') qs.set('high', local.high); qs.set('limit', local.limit); if (local.desc) qs.set('desc', 'true'); rows = await api(`${path()}/range/${seg(local.qIndex)}?${qs}`); } else if (kind === 'search') { local.q = ($('#cq-q').value || '').trim(); local.mode = $('#cq-mode').value; const qs = new URLSearchParams({q: local.q, limit: local.limit}); if (local.mode) qs.set('mode', local.mode); rows = await api(`${path()}/search/${seg(local.qIndex)}?${qs}`); } else if (kind === 'many' || kind === 'unique') { local.value = ($('#cq-value').value || '').trim(); rows = await api(`${path()}/find/${seg(local.qIndex)}?${new URLSearchParams({value: local.value, limit: local.limit})}`); } else { rows = await api(`${path()}/records?limit=${local.limit}`); } local.rows = rows; await renderCurrent(); } catch (error) { toast(error.message, 'error'); } });
      document.querySelectorAll('[data-coll-key]').forEach((button) => button.addEventListener('click', async () => { try { const key = button.dataset.collKey; state.selection = {key, value: await api(`${path()}/records/${seg(key)}`)}; await renderCurrent(); } catch (error) { toast(error.message, 'error'); } }));
      $('#coll-read').addEventListener('click', async () => { const key = $('#coll-key').value.trim(); if (!key) return toast('Enter a key first.', 'info'); try { state.selection = {key, value: await api(`${path()}/records/${seg(key)}`)}; await renderCurrent(); } catch (error) { toast(error.message, 'error'); } });
      $('#coll-insert').addEventListener('click', async () => { try { const payload = readForm(schema, 'coll-form'); const r = await api(`${path()}/records`, {method:'POST', body:JSON.stringify(payload)}); state.selection = {key: r.pk, value: payload}; state.info = await loadInfo(); local.rows = null; await renderCurrent(); toast(`Inserted ${r.pk}.`); } catch (error) { toast(error.message, 'error'); } });
      $('#coll-update').addEventListener('click', async () => { const key = $('#coll-key').value.trim(); if (!key) return toast('Enter a key first.', 'info'); try { const payload = readForm(schema, 'coll-form'); delete payload[pkField]; state.selection = {key, value: await api(`${path()}/records/${seg(key)}`, {method:'PUT', body:JSON.stringify(payload)})}; state.info = await loadInfo(); local.rows = null; await renderCurrent(); toast('Record updated.'); } catch (error) { toast(error.message, 'error'); } });
      $('#coll-delete').addEventListener('click', async () => { const key = $('#coll-key').value.trim(); if (!key) return toast('Enter a key first.', 'info'); try { await api(`${path()}/records/${seg(key)}`, {method:'DELETE'}); state.selection = null; state.info = await loadInfo(); local.rows = null; await renderCurrent(); toast('Record deleted.'); } catch (error) { toast(error.message, 'error'); } });
      $('#coll-inc').addEventListener('click', async () => { const key = $('#coll-key').value.trim(); const field = $('#coll-incfield').value.trim(); const amount = parseFloat($('#coll-incamt').value || '1'); if (!key || !field) return toast('Enter a key and field.', 'info'); try { const r = await api(`${path()}/records/${seg(key)}/increment`, {method:'POST', body:JSON.stringify({field, amount})}); state.selection = {key, value: await api(`${path()}/records/${seg(key)}`)}; await renderCurrent(); toast(`${field} = ${r.value}.`); } catch (error) { toast(error.message, 'error'); } });
    }

    async function boot() {
      state.meta = await api('/');
      $('#structure-search').addEventListener('input', (event) => { state.filter = event.target.value; renderSidebar(); });
      $('#refresh-app').addEventListener('click', async () => { state.meta = await api('/'); renderSidebar(); const entries = visibleEntries(); const next = state.current && entries[state.current] ? state.current : Object.keys(entries)[0]; if (next) await selectStructure(next); toast('Dashboard refreshed.', 'info'); });
      renderSidebar();
      const entries = visibleEntries();
      const hashName = decodeURIComponent(location.hash.replace(/^#/, ''));
      const initial = entries[hashName] ? hashName : Object.keys(entries)[0];
      if (initial) await selectStructure(initial);
      else { renderHeader(); $('#metrics').innerHTML = ''; $('#browser-panel').innerHTML = panel('No structures', 'The API is available, but no structures are exposed.', emptyState('Nothing here yet', 'Create a loom structure and refresh this page.')); $('#editor-panel').innerHTML = panel('Editor', 'No structure selected.', emptyState('Ready', 'Select a structure to start editing.')); }
    }

    boot().catch((error) => { $('#browser-panel').innerHTML = panel('Error', 'The dashboard could not initialize.', `<pre class="json">${escapeHtml(error.message)}</pre>`); $('#editor-panel').innerHTML = panel('Help', 'Check that the API is running.', emptyState('Unable to load', 'Open /docs to inspect the API state.')); });
  </script>
</body>
</html>
"""


def render_dashboard_html():
    return HTML
