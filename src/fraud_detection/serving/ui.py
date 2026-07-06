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
  .wrap { max-width:900px; margin:0 auto; padding:28px 20px 60px; }
  h1 { font-size:22px; margin:0 0 4px; }
  p.sub { color:var(--muted); margin:0 0 22px; }
  .card { background:var(--card); border:1px solid #2d3340; border-radius:10px; padding:18px; margin-bottom:18px; }
  textarea { width:100%; min-height:120px; background:#0d1117; color:var(--fg);
             border:1px solid #2d3340; border-radius:8px; padding:12px; font-family:ui-monospace,monospace; font-size:13px; }
  .row { display:flex; gap:10px; align-items:center; margin-top:12px; flex-wrap:wrap; }
  button { background:var(--accent); color:#0d1117; border:0; border-radius:8px; padding:10px 18px; font-weight:600; cursor:pointer; }
  button:disabled { opacity:.5; cursor:default; }
  .verdict { font-size:20px; font-weight:700; padding:10px 14px; border-radius:8px; display:inline-block; }
  .Fraud { background:rgba(248,81,73,.15); color:var(--fraud); }
  .Normal { background:rgba(63,185,80,.15); color:var(--normal); }
  .Expert-Checking { background:rgba(210,153,34,.15); color:var(--expert); }
  .probs { color:var(--muted); margin-top:8px; font-family:ui-monospace,monospace; font-size:13px; }
  .panels { display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-top:16px; }
  @media (max-width:680px){ .panels { grid-template-columns:1fr; } }
  .panel h3 { margin:0 0 2px; font-size:14px; }
  .panel .cap { color:var(--muted); font-size:12px; margin:0 0 6px; }
  .bar { height:16px; border-radius:4px; background:#30363d; position:relative; overflow:hidden; }
  .bar > span { position:absolute; top:0; bottom:0; }
  .pos { background:var(--fraud); left:50%; }
  .neg { background:var(--accent); right:50%; }
  table { width:100%; border-collapse:collapse; font-size:12px; }
  td { padding:4px 6px; border-bottom:1px solid #21262d; }
  td.f { font-family:ui-monospace,monospace; color:var(--fg); max-width:130px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  td.b { width:50%; }
  .muted { color:var(--muted); }
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
    <div class="row" style="margin:0 0 8px;">
      <button id="fill" type="button">Load example</button>
      <span class="muted" id="feathint"></span>
    </div>
    <textarea id="input" spellcheck="false" placeholder='{"features": {"Time": 0, "V1": -1.36, ...}}  or  {"features": [0, -1.36, ...]}'></textarea>
    <div class="row"><button id="score" type="button">Score transaction</button>
      <span class="muted" id="mode"></span></div>
  </div>

  <div class="card" id="result" style="display:none;">
    <span id="verdict" class="verdict"></span>
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
  </div>

  <div class="card" id="errcard" style="display:none;"><div class="err" id="err"></div></div>
</div>

<script>
let RAW = [];
async function loadSchema() {
  try {
    const r = await fetch("schema"); const j = await r.json();
    RAW = j.raw_feature_columns || [];
    document.getElementById("feathint").textContent =
      RAW.length ? (RAW.length + " raw features expected") : "";
  } catch (e) {}
}
document.getElementById("fill").onclick = () => {
  const obj = {}; RAW.forEach((c) => obj[c] = 0.0);
  document.getElementById("input").value = JSON.stringify({features: obj}, null, 0);
};
document.getElementById("score").onclick = async () => {
  const btn = document.getElementById("score"); btn.disabled = true;
  document.getElementById("errcard").style.display = "none";
  try {
    const body = JSON.parse(document.getElementById("input").value);
    const r = await fetch("score", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(body)});
    const j = await r.json();
    if (!r.ok) throw new Error(JSON.stringify(j.detail || j, null, 2));
    render(j);
  } catch (e) {
    document.getElementById("errcard").style.display = "block";
    document.getElementById("err").textContent = e.message;
  } finally { btn.disabled = false; }
};
function bars(tableId, items, valueKey, labelKey) {
  const t = document.getElementById(tableId); t.innerHTML = "";
  if (!items || !items.length) { t.innerHTML = '<tr><td class="muted">n/a</td></tr>'; return; }
  const max = Math.max(...items.map(f => Math.abs(f[valueKey])), 1e-9);
  for (const f of items) {
    const v = f[valueKey], w = (Math.abs(v)/max*50).toFixed(1);
    const bar = v >= 0 ? `<span class="pos" style="width:${w}%"></span>`
                       : `<span class="neg" style="width:${w}%"></span>`;
    t.innerHTML += `<tr><td class="f" title="${f[labelKey]}">${f[labelKey]}</td>`
      + `<td class="b"><div class="bar">${bar}</div></td>`
      + `<td class="muted">${v.toFixed(3)}</td></tr>`;
  }
}
function render(j) {
  document.getElementById("result").style.display = "block";
  const v = document.getElementById("verdict");
  v.textContent = j.decision; v.className = "verdict " + j.decision;
  const p2 = (j.p2 === null || j.p2 === undefined) ? "n/a" : j.p2.toFixed(4);
  document.getElementById("probs").textContent =
    `P1 (LightGBM)=${j.p1.toFixed(4)}   P2 (LSTM)=${p2}   P_sum=${(j.p_sum ?? j.p1).toFixed(4)}   θ=${j.theta}   [${j.mode}]`;
  document.getElementById("shapcap").textContent =
    "contributions to the fraud margin (base " + j.explanation_lightgbm.base_value + ")";
  bars("shap", j.explanation_lightgbm.top_features, "shap_value", "feature");

  const lp = document.getElementById("limepanel");
  if (j.explanation_lstm) {
    lp.style.display = "block";
    document.getElementById("limecap").textContent =
      "local surrogate R²=" + j.explanation_lstm.local_r2 + " · directional";
    bars("lime", j.explanation_lstm.top_features, "weight", "feature");
  } else {
    lp.style.display = "none";  // lightgbm-only build
  }
}
loadSchema();
</script>
</body>
</html>"""
