/* FabricAI — slim slide-out drawer widget.
 *
 * Drop-in: <script src="/widgets/fabricai_drawer.js" defer></script>
 *
 * Renders:
 *   - a sienna pill button bottom-right ("Ask FabricAI")
 *   - a 420px right-edge drawer with chat thread, role/topic/language picker,
 *     starter-questions strip, multiline composer, image attach.
 *
 * Backend: same-origin /fabricai/api/* (mounted sub-app).
 * Context: pulls window._ffPassportV2 + window.lastFabricResult and prepends a
 * single condensed "CURRENT PAGE CONTEXT" block to each question, so the model
 * knows what fabric / passport the user is looking at without needing the
 * swatchon results endpoint.
 *
 * Style: matches editorial design tokens (Libre Caslon, Schibsted Grotesk,
 * vellum + sienna). Tokens are inlined so the widget works on pages that
 * haven't defined them.
 */
(function () {
  'use strict';

  if (window.__faiDrawerLoaded) return;
  window.__faiDrawerLoaded = true;

  // Don't render inside the embed iframe (passport_v2?embed=1) or any other iframe.
  try {
    if (window.top !== window.self) return;
    var _qs = new URLSearchParams(location.search);
    if (_qs.get('embed') === '1') return;
  } catch (e) { /* cross-origin iframe — assume inside, abort */ return; }

  // ── helpers ──────────────────────────────────────────────────────────────
  const $ = (sel, root) => (root || document).querySelector(sel);
  const make = (tag, attrs, ...kids) => {
    const el = document.createElement(tag);
    if (attrs) for (const k in attrs) {
      if (k === 'class') el.className = attrs[k];
      else if (k === 'html') el.innerHTML = attrs[k];
      else if (k.startsWith('on') && typeof attrs[k] === 'function') el.addEventListener(k.slice(2), attrs[k]);
      else el.setAttribute(k, attrs[k]);
    }
    kids.flat().forEach(k => el.appendChild(typeof k === 'string' ? document.createTextNode(k) : k));
    return el;
  };
  const esc = s => String(s == null ? '' : s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  // ── styles ──────────────────────────────────────────────────────────────
  const css = `
  .fai-fab {
    /* Stack above the host page's Research Survey float (~52px tall + 24px bottom inset). */
    position: fixed; right: 24px; bottom: 84px; z-index: 9990;
    /* Hidden by default — the host page opts in by adding .fai-allow on body. */
    visibility: hidden; opacity: 0; transform: translateY(8px);
    display: inline-flex; align-items: center; gap: 8px;
    background: oklch(0.48 0.14 38); color: #fff;
    border: 1px solid oklch(0.36 0.12 32);
    font-family: 'Schibsted Grotesk', -apple-system, sans-serif;
    font-size: 12px; font-weight: 600; letter-spacing: 1.4px; text-transform: uppercase;
    padding: 12px 18px; border-radius: 999px; cursor: pointer;
    box-shadow: 0 8px 22px -10px rgba(120, 60, 40, 0.55);
    transition: opacity .25s ease, transform .25s ease, visibility .25s linear, background .18s ease, box-shadow .2s ease;
  }
  body.fai-allow .fai-fab { visibility: visible; opacity: 1; transform: translateY(0); }
  body.fai-allow .fai-fab:hover { background: oklch(0.36 0.12 32); transform: translateY(-1px); box-shadow: 0 12px 28px -12px rgba(120, 60, 40, 0.7); }
  .fai-fab .fai-dot { width: 8px; height: 8px; border-radius: 50%; background: oklch(0.78 0.13 38); display: inline-block; box-shadow: 0 0 0 3px rgba(255,255,255,.18); }
  .fai-fab.fai-open { display: none; }

  .fai-scrim {
    position: fixed; inset: 0; background: rgba(20, 24, 32, 0.28); z-index: 9991;
    opacity: 0; pointer-events: none; transition: opacity .25s ease;
  }
  .fai-scrim.show { opacity: 1; pointer-events: auto; }

  .fai-drawer {
    position: fixed; top: 0; right: 0; bottom: 0; width: min(440px, 96vw); z-index: 9992;
    background: oklch(0.980 0.004 85);
    border-left: 1px solid oklch(0.86 0.012 65);
    display: flex; flex-direction: column;
    transform: translateX(105%); transition: transform .28s cubic-bezier(.2,.7,.2,1);
    font-family: 'Schibsted Grotesk', -apple-system, BlinkMacSystemFont, sans-serif;
    color: oklch(0.18 0.020 250);
    box-shadow: -18px 0 40px -24px rgba(20, 24, 32, 0.18);
  }
  .fai-drawer.show { transform: translateX(0); }

  .fai-head {
    padding: 18px 20px 16px; background: oklch(0.36 0.045 55); color: oklch(0.96 0.012 65);
    position: relative; overflow: hidden;
  }
  .fai-head::before {
    content: ''; position: absolute; inset: 0;
    background: repeating-linear-gradient(135deg, transparent, transparent 48px, rgba(255,255,255,.05) 48px, rgba(255,255,255,.05) 49px);
    pointer-events: none;
  }
  .fai-eyebrow { position: relative; font-size: 10px; letter-spacing: 2px; text-transform: uppercase; color: oklch(0.74 0.020 60); margin-bottom: 6px; font-weight: 500; }
  .fai-title { position: relative; font-family: 'Libre Caslon Text', Georgia, serif; font-size: 22px; line-height: 1.15; font-weight: 400; letter-spacing: -0.3px; }
  .fai-sub { position: relative; font-family: 'Libre Caslon Text', Georgia, serif; font-style: italic; font-size: 12.5px; color: oklch(0.86 0.018 65); margin-top: 4px; }
  .fai-close {
    position: absolute; top: 14px; right: 14px; z-index: 1;
    background: transparent; border: 1px solid rgba(255,255,255,.28); border-radius: 4px;
    color: oklch(0.86 0.018 65); width: 32px; height: 32px; line-height: 1;
    cursor: pointer; font-size: 16px; transition: background .15s ease, color .15s ease;
  }
  .fai-close:hover { background: rgba(255,255,255,.08); color: #fff; }

  .fai-controls { padding: 12px 20px 10px; border-bottom: 1px solid oklch(0.90 0.006 250); background: oklch(0.995 0.002 85); display: grid; gap: 8px; }
  .fai-row { display: flex; gap: 8px; align-items: center; }
  .fai-row label { font-size: 10px; letter-spacing: 1.2px; text-transform: uppercase; color: oklch(0.58 0.008 250); font-weight: 500; min-width: 60px; }
  .fai-row select {
    flex: 1; padding: 6px 8px; font-family: inherit; font-size: 12.5px; color: oklch(0.18 0.020 250);
    background: oklch(0.995 0.002 85); border: 1px solid oklch(0.86 0.012 65); border-radius: 4px;
  }
  .fai-row select:focus { outline: none; border-color: oklch(0.48 0.14 38); }

  .fai-starter {
    padding: 10px 20px 12px; border-bottom: 1px solid oklch(0.90 0.006 250);
    background: oklch(0.965 0.018 38); display: flex; flex-wrap: wrap; gap: 6px;
  }
  .fai-starter-chip {
    font-size: 11px; padding: 5px 10px; background: oklch(0.995 0.002 85);
    border: 1px solid oklch(0.86 0.06 38); border-radius: 999px; cursor: pointer;
    color: oklch(0.36 0.12 32); transition: background .15s ease;
  }
  .fai-starter-chip:hover { background: oklch(0.92 0.04 38); }

  .fai-thread { flex: 1; overflow-y: auto; padding: 16px 20px 18px; display: flex; flex-direction: column; gap: 12px; background: oklch(0.980 0.004 85); }
  .fai-msg { max-width: 88%; padding: 10px 14px; border-radius: 10px; font-size: 13px; line-height: 1.55; white-space: pre-wrap; word-wrap: break-word; }
  .fai-msg.fai-user { align-self: flex-end; background: oklch(0.36 0.045 55); color: oklch(0.96 0.012 65); border-bottom-right-radius: 3px; }
  .fai-msg.fai-bot  { align-self: flex-start; background: oklch(0.995 0.002 85); border: 1px solid oklch(0.90 0.006 250); border-bottom-left-radius: 3px; }
  .fai-msg.fai-bot strong { color: oklch(0.36 0.12 32); }
  .fai-msg.fai-system { align-self: center; background: transparent; color: oklch(0.58 0.008 250); font-style: italic; font-size: 12px; text-align: center; font-family: 'Libre Caslon Text', Georgia, serif; }
  .fai-msg.fai-typing { color: oklch(0.58 0.008 250); font-style: italic; }

  .fai-compose { border-top: 1px solid oklch(0.90 0.006 250); padding: 12px 14px 14px; background: oklch(0.995 0.002 85); }
  .fai-compose-row { display: flex; align-items: flex-end; gap: 8px; }
  .fai-textarea {
    flex: 1; resize: none; min-height: 38px; max-height: 140px;
    padding: 9px 12px; font-family: inherit; font-size: 13px; line-height: 1.5;
    border: 1px solid oklch(0.86 0.012 65); border-radius: 6px;
    background: oklch(0.980 0.004 85); color: oklch(0.18 0.020 250);
  }
  .fai-textarea:focus { outline: none; border-color: oklch(0.48 0.14 38); }
  .fai-send {
    background: oklch(0.48 0.14 38); color: #fff; border: none; border-radius: 6px;
    width: 38px; height: 38px; cursor: pointer; flex-shrink: 0;
    font-size: 14px; line-height: 1; transition: background .15s ease;
  }
  .fai-send:hover { background: oklch(0.36 0.12 32); }
  .fai-send[disabled] { opacity: .5; cursor: not-allowed; }
  .fai-hint { font-size: 10.5px; color: oklch(0.58 0.008 250); margin-top: 6px; letter-spacing: 0.2px; }
  .fai-err { color: oklch(0.48 0.14 25); }

  @media (max-width: 520px) {
    .fai-drawer { width: 100vw; border-left: none; }
    .fai-fab { right: 14px; bottom: 70px; padding: 10px 14px; }
  }
  `;
  document.head.appendChild(make('style', { html: css }));

  // ── state ──────────────────────────────────────────────────────────────
  let sessionId = null;
  let cfg = null;
  let busy = false;

  // ── DOM construction ───────────────────────────────────────────────────
  const fab = make('button', { class: 'fai-fab', 'aria-label': 'Open FabricAI', html: '<span class="fai-dot"></span>ASK FABRICAI' });
  const scrim = make('div', { class: 'fai-scrim' });
  const drawer = make('div', { class: 'fai-drawer', role: 'dialog', 'aria-label': 'FabricAI assistant' });

  const head = make('div', { class: 'fai-head', html: `
    <button class="fai-close" aria-label="Close">×</button>
    <div class="fai-eyebrow">FabricAI · Assistant</div>
    <div class="fai-title">Ask about this fabric</div>
    <div class="fai-sub">Grounded in your current passport &amp; recognition result</div>
  `});
  const controls = make('div', { class: 'fai-controls' });
  const starter = make('div', { class: 'fai-starter' });
  const thread = make('div', { class: 'fai-thread' });
  const compose = make('div', { class: 'fai-compose' });

  drawer.appendChild(head);
  drawer.appendChild(controls);
  drawer.appendChild(starter);
  drawer.appendChild(thread);
  drawer.appendChild(compose);

  // Controls (populated after /api/config)
  controls.innerHTML = `
    <div class="fai-row">
      <label>Role</label>
      <select id="fai-role"></select>
    </div>
    <div class="fai-row">
      <label>Topic</label>
      <select id="fai-topic"></select>
    </div>
    <div class="fai-row">
      <label>Language</label>
      <select id="fai-lang"></select>
    </div>
  `;

  // Composer
  compose.innerHTML = `
    <div class="fai-compose-row">
      <textarea class="fai-textarea" id="fai-input" rows="1" placeholder="Ask about composition, sourcing, certifications…"></textarea>
      <button class="fai-send" id="fai-send" title="Send (⏎)">↑</button>
    </div>
    <div class="fai-hint" id="fai-hint">Enter to send · Shift+Enter for new line</div>
  `;

  document.body.appendChild(fab);
  document.body.appendChild(scrim);
  document.body.appendChild(drawer);

  // ── visibility gate ─────────────────────────────────────────────────────
  // The FAB stays hidden until the host page calls window.faiShow(). Pages
  // that should always show it (the standalone passport viewers) auto-opt-in.
  window.faiShow = function () { document.body.classList.add('fai-allow'); };
  window.faiHide = function () { document.body.classList.remove('fai-allow'); };
  try {
    var path = location.pathname.replace(/\/+$/, '');
    if (path === '/passport_v2' || path === '/passport' || /\/passport(_view)?$/.test(path)) {
      window.faiShow();
    }
  } catch (e) {}

  const $closeBtn  = $('.fai-close', head);
  const $roleSel   = $('#fai-role', controls);
  const $topicSel  = $('#fai-topic', controls);
  const $langSel   = $('#fai-lang', controls);
  const $input     = $('#fai-input', compose);
  const $send      = $('#fai-send', compose);
  const $hint      = $('#fai-hint', compose);

  // ── open / close ────────────────────────────────────────────────────────
  function open() {
    fab.classList.add('fai-open');
    scrim.classList.add('show');
    drawer.classList.add('show');
    if (!cfg) bootstrap();
    setTimeout(() => $input.focus(), 200);
  }
  function close() {
    scrim.classList.remove('show');
    drawer.classList.remove('show');
    fab.classList.remove('fai-open');
  }
  fab.addEventListener('click', open);
  $closeBtn.addEventListener('click', close);
  scrim.addEventListener('click', close);
  document.addEventListener('keydown', e => { if (e.key === 'Escape' && drawer.classList.contains('show')) close(); });

  // ── bootstrap: /api/config + session ───────────────────────────────────
  async function bootstrap() {
    try {
      const r = await fetch('/fabricai/api/config');
      if (!r.ok) throw new Error('config ' + r.status);
      cfg = await r.json();
    } catch (e) {
      systemMsg('Could not reach FabricAI backend — set OPENAI_API_KEY and restart server.', true);
      return;
    }
    fillSelect($roleSel, cfg.roles, (cfg.roles[0] || {}).id);
    fillSelect($topicSel, cfg.topics, cfg.default_topic);
    fillSelect($langSel, cfg.languages, cfg.default_language);
    renderStarter();
    $topicSel.addEventListener('change', renderStarter);

    try {
      const s = await fetch('/fabricai/api/session', { method: 'POST' });
      const j = await s.json();
      sessionId = j.session_id;
      systemMsg('Welcome to FabricAI. I can read what you have on screen — ask anything about composition, sourcing, certifications, or impact.');
    } catch (e) {
      systemMsg('Could not start a session: ' + e.message, true);
    }
  }
  function fillSelect(sel, items, defId) {
    sel.innerHTML = '';
    (items || []).forEach(it => {
      const o = make('option', { value: it.id }, it.label);
      sel.appendChild(o);
    });
    if (defId) sel.value = defId;
  }
  function renderStarter() {
    starter.innerHTML = '';
    if (!cfg || !cfg.starter_questions) return;
    const t = $topicSel.value;
    (cfg.starter_questions[t] || []).slice(0, 3).forEach(q => {
      const c = make('button', { class: 'fai-starter-chip', html: esc(q) });
      c.addEventListener('click', () => { $input.value = q; $input.focus(); });
      starter.appendChild(c);
    });
  }

  // ── messages ────────────────────────────────────────────────────────────
  function botMsg(text)   { return appendMsg(text, 'fai-bot'); }
  function userMsg(text)  { return appendMsg(text, 'fai-user'); }
  function systemMsg(text, isErr) {
    const el = appendMsg(text, 'fai-system');
    if (isErr) el.classList.add('fai-err');
    return el;
  }
  function appendMsg(text, klass) {
    const el = make('div', { class: 'fai-msg ' + klass });
    el.textContent = text;
    thread.appendChild(el);
    thread.scrollTop = thread.scrollHeight;
    return el;
  }

  // ── page-context provider ──────────────────────────────────────────────
  function gatherPageContext() {
    const bits = [];
    try {
      const r = window.lastFabricResult || window._lastFabricResult || null;
      if (r) bits.push('Latest classification: ' + (r.fullName || r.name || r.l1 || JSON.stringify(r)).slice(0, 240));
    } catch (e) {}
    try {
      const p = window._ffPassportV2 || null;
      if (p) {
        const k = ['fabric_name','passport_id','origin','supplier_short_name','supplier_grade','fabric_score','otd_rate'];
        const flat = k.map(x => p[x] != null ? `${x}=${p[x]}` : null).filter(Boolean).join(' · ');
        if (flat) bits.push('Compiled passport: ' + flat);
        if (Array.isArray(p.fibre_legend)) bits.push('Fibres: ' + p.fibre_legend.map(f => `${f.name} ${f.pct}%`).join(', '));
      }
    } catch (e) {}
    try {
      const h = (document.querySelector('h1, .hero-title, .pp-hero-name') || {}).textContent;
      if (h) bits.push('Page title: ' + h.trim().slice(0, 120));
    } catch (e) {}
    if (!bits.length) return '';
    return 'CURRENT PAGE CONTEXT (verbatim, treat as ground truth for this turn):\n- ' + bits.join('\n- ');
  }

  // ── send ────────────────────────────────────────────────────────────────
  async function send() {
    if (busy || !sessionId) return;
    const q = ($input.value || '').trim();
    if (!q) return;
    $input.value = '';
    autoResize();
    userMsg(q);

    const ctx = gatherPageContext();
    const finalQ = ctx ? (ctx + '\n\nUSER QUESTION:\n' + q) : q;

    busy = true;
    $send.disabled = true;
    const typing = appendMsg('Thinking…', 'fai-msg fai-bot fai-typing');
    try {
      const r = await fetch('/fabricai/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          question: finalQ,
          role: $roleSel.value || null,
          topic: $topicSel.value || null,
          language: $langSel.value || null,
        }),
      });
      const j = await r.json();
      typing.remove();
      if (!r.ok) {
        systemMsg('Error: ' + (j.detail || r.status), true);
      } else {
        botMsg(j.answer || '(empty response)');
      }
    } catch (e) {
      typing.remove();
      systemMsg('Network error: ' + e.message, true);
    } finally {
      busy = false;
      $send.disabled = false;
      $input.focus();
    }
  }
  $send.addEventListener('click', send);
  $input.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  });
  function autoResize() {
    $input.style.height = 'auto';
    $input.style.height = Math.min($input.scrollHeight, 140) + 'px';
  }
  $input.addEventListener('input', autoResize);
})();
