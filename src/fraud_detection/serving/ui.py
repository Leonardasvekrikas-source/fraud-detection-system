"""Minimal single-page demo UI (no build step, no JS framework)."""

DEMO_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Explainable Fraud Detection — demo</title>
<style>
  :root { --bg:#0f1419; --card:#1a2029; --fg:#e6edf3; --muted:#8b949e;
          --fraud:#f85149; --normal:#3fb950; --expert:#d29922; --accent:#58a6ff; }
  * { box-sizing:border-box; }
  body { margin:0; font:15px/1.5 system-ui,sans-serif; background:var(--bg); color:var(--fg); }
  .wrap { max-width:920px; margin:0 auto; padding:28px 20px 60px; }
  h1 { font-size:22px; margin:0 0 4px; }
  p.sub { color:var(--muted); margin:0 0 22px; }
  .card { background:var(--card); border:1px solid #2d3340; border-radius:10px; padding:18px; margin-bottom:18px; }
  textarea { width:100%; min-height:96px; background:#0d1117; color:var(--fg);
             border:1px solid #2d3340; border-radius:8px; padding:12px; font-family:ui-monospace,monospace; font-size:12px; }
  .row { display:flex; gap:10px; align-items:center; margin-top:12px; flex-wrap:wrap; }
  button { border:0; border-radius:8px; padding:10px 16px; font-weight:600; cursor:pointer; color:#0d1117; background:var(--accent); }
  button:disabled { opacity:.5; cursor:default; }
  .try { font-size:13px; }
  .try .lbl { color:var(--muted); margin-right:2px; }
  .b-normal { background:var(--normal); } .b-fraud { background:var(--fraud); } .b-expert { background:var(--expert); }
  .b-ghost { background:transparent; color:var(--muted); border:1px solid #2d3340; }
  .verdict { font-size:20px; font-weight:700; padding:10px 14px; border-radius:8px; display:inline-block; }
  .Fraud { background:rgba(248,81,73,.15); color:var(--fraud); }
  .Normal { background:rgba(63,185,80,.15); color:var(--normal); }
  .Expert-Checking { background:rgba(210,153,34,.15); color:var(--expert); }
  .exnote { color:var(--muted); font-size:13px; margin-top:8px; font-style:italic; }
  .probs { color:var(--muted); margin-top:6px; font-family:ui-monospace,monospace; font-size:13px; }
  .panels { display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-top:16px; }
  @media (max-width:680px){ .panels { grid-template-columns:1fr; } }
  .panel h3 { margin:0 0 2px; font-size:14px; }
  .panel .cap { color:var(--muted); font-size:12px; margin:0 0 6px; }
  .bar { height:14px; border-radius:4px; background:#30363d; position:relative; overflow:hidden; }
  .bar > span { position:absolute; top:0; bottom:0; }
  .pos { background:var(--fraud); left:50%; }
  .neg { background:var(--accent); right:50%; }
  table { width:100%; border-collapse:collapse; font-size:12px; }
  td { padding:5px 6px; border-bottom:1px solid #21262d; vertical-align:middle; }
  td.f { max-width:150px; }
  .fcode { font-family:ui-monospace,monospace; color:var(--fg); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .alias { color:var(--accent); font-style:italic; }
  .fsub { color:var(--muted); font-size:10.5px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  td.b { width:42%; }
  .muted { color:var(--muted); }
  .legend { margin-top:14px; font-size:11.5px; color:var(--muted); line-height:1.5; border-top:1px solid #21262d; padding-top:10px; }
  .legend b { color:var(--fg); }
  .err { color:var(--fraud); white-space:pre-wrap; font-family:ui-monospace,monospace; font-size:13px; }
  a { color:var(--accent); }
</style>
</head>
<body>
<div class="wrap">
  <h1>Explainable Fraud Detection</h1>
  <p class="sub">Decision-level fusion of LightGBM + LSTM → a Normal / Fraud / <b>Expert-Checking</b>
     verdict, explained by <b>SHAP</b> (LightGBM) and <b>LIME</b> (LSTM).</p>

  <div class="card">
    <div class="row try" id="trybar" style="margin:0 0 10px;">
      <span class="lbl">Try a real transaction:</span>
      <button type="button" class="b-normal" data-type="Normal">Random Normal</button>
      <button type="button" class="b-expert" data-type="Expert-Checking">Random Expert-Checking</button>
      <button type="button" class="b-fraud" data-type="Fraud">Random Fraud</button>
      <span class="muted" id="exhint"></span>
    </div>
    <textarea id="input" spellcheck="false" placeholder='Click a button above, or paste {"features": {...}} / {"features": [...]}'></textarea>
    <div class="row"><button id="score" type="button">Score transaction</button>
      <button id="clear" type="button" class="b-ghost">Clear</button>
      <span class="muted" id="mode"></span></div>
  </div>

  <div class="card" id="result" style="display:none;">
    <span id="verdict" class="verdict"></span>
    <div class="exnote" id="exnote" style="display:none;"></div>
    <div class="probs" id="probs"></div>
    <div class="panels">
      <div class="panel">
        <h3>LightGBM · SHAP <span class="muted">(exact)</span></h3>
        <p class="cap" id="shapcap">contributions to the fraud margin (→ fraud, ← normal)</p>
        <table id="shap"></table>
      </div>
      <div class="panel" id="limepanel">
        <h3>LSTM · LIME <span class="muted">(directional)</span></h3>
        <p class="cap" id="limecap"></p>
        <table id="lime"></table>
      </div>
    </div>
    <div class="legend">
      <b>Reading the features.</b> <b>Time</b> and <b>Amount</b> are the real, un-anonymized fields.
      <b>V1–V28</b> are <b>PCA-anonymized</b> components — the dataset's authors masked the original
      fields (merchant, location, …) for privacy, so their true meaning is unknown even to us. Each
      label instead shows the feature's <i>measured</i> role: how much this <b>model</b> relies on it
      (primary / secondary / minor driver, by SHAP importance) and which direction pushes toward fraud.
      Names in <span class="alias">"quotes"</span> are <b>illustrative</b> mnemonics for the top drivers
      — readable stand-ins, <i>not</i> the real (unknown) fields.
    </div>
  </div>

  <div class="card" id="errcard" style="display:none;"><div class="err" id="err"></div></div>
</div>

<script>
let RAW = [], FMETA = {}, EXAMPLES = {}, LAST_NOTE = "";
async function loadSchema() {
  try {
    const j = await (await fetch("schema")).json();
    RAW = j.raw_feature_columns || [];
    FMETA = j.feature_meta || {};
  } catch (e) {}
}
async function loadExamples() {
  try {
    EXAMPLES = await (await fetch("examples")).json();
    const have = Object.values(EXAMPLES).some(a => a && a.length);
    document.getElementById("trybar").style.display = have ? "" : "none";
  } catch (e) { document.getElementById("trybar").style.display = "none"; }
}
// look up feature metadata for a SHAP code ("V14") or a LIME condition ("V14 <= -0.4")
function metaFor(featStr) {
  if (FMETA[featStr]) return { code: featStr, m: FMETA[featStr] };
  const tok = featStr.split(/[\\s<>=]/)[0];
  return { code: featStr, m: FMETA[tok] };
}
function subLine(m) {
  if (!m) return "";
  if (!m.selected) return "not used by the model";
  const alias = m.alias ? `<span class="alias">"${m.alias}"</span> · ` : "";
  const dir = (m.direction || "").replace("->", "→");
  return `${alias}${m.tier || ""}${dir ? " · " + dir : ""}`;
}
function tip(code, m) {
  if (!m) return code;
  let t = m.name || code;
  if (m.unit) t += ` — ${m.unit}`;
  if (m.selected) {
    t += `\\nmodel importance: #${m.rank} (${m.tier})`;
    if (m.direction) t += `\\ndirection: ${m.direction}  (corr ${m.corr})`;
    if (m.alias) t += `\\nillustrative alias: "${m.alias}" (not the real field)`;
  } else { t += "\\nexcluded by feature selection — not used by the model"; }
  return t;
}
function bars(tableId, items, valueKey) {
  const t = document.getElementById(tableId); t.innerHTML = "";
  if (!items || !items.length) { t.innerHTML = '<tr><td class="muted">n/a</td></tr>'; return; }
  const max = Math.max(...items.map(f => Math.abs(f[valueKey])), 1e-9);
  for (const f of items) {
    const v = f[valueKey], w = (Math.abs(v) / max * 50).toFixed(1);
    const { code, m } = metaFor(f.feature);
    const bar = v >= 0 ? `<span class="pos" style="width:${w}%"></span>`
                       : `<span class="neg" style="width:${w}%"></span>`;
    const sub = subLine(m);
    t.innerHTML += `<tr><td class="f" title="${tip(code, m).replace(/"/g, '&quot;')}">`
      + `<div class="fcode">${f.feature}</div>`
      + (sub ? `<div class="fsub">${sub}</div>` : "")
      + `</td><td class="b"><div class="bar">${bar}</div></td>`
      + `<td class="muted">${v.toFixed(3)}</td></tr>`;
  }
}
function fill(obj) { document.getElementById("input").value = JSON.stringify(obj); }
document.querySelectorAll("#trybar button").forEach(btn => {
  btn.onclick = () => {
    const pool = EXAMPLES[btn.dataset.type] || [];
    if (!pool.length) return;
    const ex = pool[Math.floor(Math.random() * pool.length)];
    LAST_NOTE = ex.note || "";
    fill({ features: ex.features, top_k: 8 });
    doScore();
  };
});
document.getElementById("clear").onclick = () => {
  document.getElementById("input").value = "";
  document.getElementById("result").style.display = "none";
  LAST_NOTE = "";
};
document.getElementById("score").onclick = () => { LAST_NOTE = ""; doScore(); };
async function doScore() {
  const btn = document.getElementById("score"); btn.disabled = true;
  document.getElementById("errcard").style.display = "none";
  try {
    const body = JSON.parse(document.getElementById("input").value);
    const r = await fetch("score", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const j = await r.json();
    if (!r.ok) throw new Error(JSON.stringify(j.detail || j, null, 2));
    render(j);
  } catch (e) {
    document.getElementById("errcard").style.display = "block";
    document.getElementById("err").textContent = e.message;
  } finally { btn.disabled = false; }
}
function render(j) {
  document.getElementById("result").style.display = "block";
  const v = document.getElementById("verdict");
  v.textContent = j.decision; v.className = "verdict " + j.decision;
  const en = document.getElementById("exnote");
  if (LAST_NOTE) { en.style.display = "block"; en.textContent = "Example: " + LAST_NOTE; }
  else { en.style.display = "none"; }
  const p2 = (j.p2 === null || j.p2 === undefined) ? "n/a" : j.p2.toFixed(4);
  document.getElementById("probs").textContent =
    `P1 (LightGBM)=${j.p1.toFixed(4)}   P2 (LSTM)=${p2}   P_sum=${(j.p_sum ?? j.p1).toFixed(4)}   θ=${j.theta}   [${j.mode}]`;
  document.getElementById("shapcap").textContent =
    "contributions to the fraud margin (base " + j.explanation_lightgbm.base_value + ")";
  bars("shap", j.explanation_lightgbm.top_features, "shap_value");

  const lp = document.getElementById("limepanel");
  if (j.explanation_lstm) {
    lp.style.display = "block";
    document.getElementById("limecap").textContent =
      "local surrogate R²=" + j.explanation_lstm.local_r2 + " · directional";
    bars("lime", j.explanation_lstm.top_features, "weight");
  } else { lp.style.display = "none"; }
}
loadSchema();
loadExamples();
</script>
</body>
</html>"""
