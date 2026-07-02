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
  .wrap { max-width:860px; margin:0 auto; padding:28px 20px 60px; }
  h1 { font-size:22px; margin:0 0 4px; }
  p.sub { color:var(--muted); margin:0 0 22px; }
  .card { background:var(--card); border:1px solid #2d3340; border-radius:10px; padding:18px; margin-bottom:18px; }
  textarea { width:100%; min-height:140px; background:#0d1117; color:var(--fg);
             border:1px solid #2d3340; border-radius:8px; padding:12px; font-family:ui-monospace,monospace; font-size:13px; }
  .row { display:flex; gap:10px; align-items:center; margin-top:12px; flex-wrap:wrap; }
  button { background:var(--accent); color:#0d1117; border:0; border-radius:8px; padding:10px 18px; font-weight:600; cursor:pointer; }
  button:disabled { opacity:.5; cursor:default; }
  .verdict { font-size:20px; font-weight:700; padding:10px 14px; border-radius:8px; display:inline-block; }
  .Fraud { background:rgba(248,81,73,.15); color:var(--fraud); }
  .Normal { background:rgba(63,185,80,.15); color:var(--normal); }
  .Expert-Checking { background:rgba(210,153,34,.15); color:var(--expert); }
  .probs { color:var(--muted); margin-top:8px; font-family:ui-monospace,monospace; font-size:13px; }
  .bar { height:20px; border-radius:4px; background:#30363d; position:relative; overflow:hidden; }
  .bar > span { position:absolute; top:0; bottom:0; }
  .pos { background:var(--fraud); left:50%; }
  .neg { background:var(--accent); right:50%; }
  table { width:100%; border-collapse:collapse; margin-top:10px; font-size:13px; }
  td { padding:5px 6px; border-bottom:1px solid #21262d; }
  td.f { font-family:ui-monospace,monospace; color:var(--fg); width:130px; }
  td.b { width:52%; }
  .muted { color:var(--muted); }
  .err { color:var(--fraud); white-space:pre-wrap; font-family:ui-monospace,monospace; font-size:13px; }
  a { color:var(--accent); }
</style>
</head>
<body>
<div class="wrap">
  <h1>Explainable Fraud Detection</h1>
  <p class="sub">Paste a transaction → fraud probability, decision-level fusion verdict, and a SHAP explanation.
     Decision-level fusion keeps LightGBM and LSTM separable, so the score is attributable and SHAP is exact & fast.</p>

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
    <h3 style="margin:18px 0 2px;font-size:15px;">Top SHAP contributions <span class="muted">(→ fraud, ← normal)</span></h3>
    <div class="muted" id="basev" style="font-size:12px;"></div>
    <table id="shap"></table>
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
  const obj = {}; RAW.forEach((c,i) => obj[c] = 0.0);
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
function render(j) {
  document.getElementById("result").style.display = "block";
  const v = document.getElementById("verdict");
  v.textContent = j.decision; v.className = "verdict " + j.decision;
  const p2 = (j.p2 === null || j.p2 === undefined) ? "n/a (LightGBM-only build)" : j.p2.toFixed(4);
  document.getElementById("probs").textContent =
    `P1 (LightGBM)=${j.p1.toFixed(4)}   P2 (LSTM)=${p2}   P_sum=${(j.p_sum ?? j.p1).toFixed(4)}   θ=${j.theta}   [${j.mode}]`;
  document.getElementById("mode").textContent = "";
  document.getElementById("basev").textContent = "base margin = " + j.explanation.base_value;
  const t = document.getElementById("shap"); t.innerHTML = "";
  const max = Math.max(...j.explanation.top_features.map(f => Math.abs(f.shap_value)), 1e-9);
  for (const f of j.explanation.top_features) {
    const w = (Math.abs(f.shap_value)/max*50).toFixed(1);
    const bar = f.shap_value >= 0
      ? `<span class="pos" style="width:${w}%"></span>`
      : `<span class="neg" style="width:${w}%"></span>`;
    t.innerHTML += `<tr><td class="f">${f.feature}</td>`
      + `<td class="b"><div class="bar">${bar}</div></td>`
      + `<td class="muted">${f.shap_value.toFixed(4)}</td></tr>`;
  }
}
loadSchema();
</script>
</body>
</html>"""
