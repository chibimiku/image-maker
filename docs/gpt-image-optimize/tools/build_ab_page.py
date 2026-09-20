# -*- coding: utf-8 -*-
"""生成 A/B 实验对比页（多组，每组可含多张图，同步缩放/平移）。

用法：
  python build_ab_page.py --data payload.json --out index.html [--title ...]

payload.json 结构：
{
  "title": "...",
  "subtitle": "...",
  "note": "...",
  "metrics": [ {...}, ... ],
  "groups": [
     {"name":"A · 重绘 prompt 对比","desc":"...","panes":[{"img":"img/x.png","name":"...","tag":"...","cls":"best|","meta":"..."}]}
  ]
}
"""
import argparse
import json
import os

TPL = r"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>__TITLE__</title>
<style>
:root{--bg:#10131a;--panel:#171b25;--panel2:#1e2430;--line:#2b3344;--txt:#e8ecf4;--dim:#96a1b5;
--accent:#ff9ec4;--accent2:#7ec8ff;--good:#7ee2a8;--warn:#ffd479}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--txt);font-family:"Segoe UI","Microsoft YaHei",system-ui,sans-serif;font-size:14px;line-height:1.6}
header{padding:20px 26px 16px;border-bottom:1px solid var(--line);background:linear-gradient(180deg,#1b2130,#10131a)}
h1{margin:0 0 6px;font-size:20px;font-weight:700}
.sub{color:var(--dim);font-size:13px}
.wrap{padding:16px 26px 60px;max-width:1780px;margin:0 auto}
.controls{display:flex;flex-wrap:wrap;align-items:center;gap:12px 18px;background:var(--panel);
border:1px solid var(--line);border-radius:12px;padding:11px 18px;margin-bottom:14px;position:sticky;top:0;z-index:30}
.seg{display:flex;border:1px solid var(--line);border-radius:8px;overflow:hidden;flex-wrap:wrap}
.seg button{background:transparent;border:0;color:var(--dim);padding:7px 13px;font-size:13px;cursor:pointer;font-family:inherit}
.seg button.on{background:var(--accent);color:#191019;font-weight:700}
.zoom{display:flex;align-items:center;gap:9px}
input[type=range]{width:190px;accent-color:var(--accent)}
.btn{background:var(--panel2);border:1px solid var(--line);color:var(--txt);border-radius:8px;padding:7px 12px;cursor:pointer;font-size:13px;font-family:inherit}
.btn:hover{border-color:var(--accent);color:var(--accent)}
.btn.on{background:var(--accent2);color:#0b1620;border-color:var(--accent2);font-weight:700}
.hint{color:var(--dim);font-size:12.5px;margin-left:auto}
.group{margin-bottom:22px}
.ghead{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap;margin:0 0 10px}
.ghead h2{margin:0;font-size:15px;color:var(--accent2);font-weight:600}
.ghead .desc{color:var(--dim);font-size:12.5px}
.grid{display:grid;gap:14px}
.col{background:var(--panel);border:1px solid var(--line);border-radius:12px;overflow:hidden;display:flex;flex-direction:column}
.col.best{border-color:var(--good);box-shadow:0 0 0 1px rgba(126,226,168,.35)}
.col.src{border-color:var(--accent2)}
.head{padding:9px 12px;border-bottom:1px solid var(--line);display:flex;align-items:baseline;gap:9px;flex-wrap:wrap}
.name{font-weight:700;font-size:13.5px}
.tag{font-size:11px;padding:2px 7px;border-radius:20px;border:1px solid var(--line);color:var(--dim)}
.tag.g{color:var(--good);border-color:rgba(126,226,168,.5)}
.tag.b{color:var(--accent2);border-color:rgba(126,200,255,.5)}
.stage{position:relative;background:#0b0e14;overflow:auto;height:min(70vh,860px);cursor:grab}
.stage::-webkit-scrollbar{width:9px;height:9px}
.stage::-webkit-scrollbar-thumb{background:#39435a;border-radius:9px}
.stage img{display:block;transform-origin:0 0;user-select:none;-webkit-user-drag:none}
.stage.drag{cursor:grabbing}
.meta{padding:8px 12px;border-top:1px solid var(--line);color:var(--dim);font-size:11.5px}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{padding:7px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
thead th{color:var(--accent2);font-weight:600;font-size:12.5px}
tbody tr:hover{background:#1b2130}
.best-cell{color:var(--good);font-weight:700}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 16px;margin:16px 0}
.card h2{margin:0 0 10px;font-size:15px;color:var(--accent2);font-weight:600}
.note{color:var(--dim);font-size:12.5px}
code{background:#0d1017;border:1px solid var(--line);border-radius:4px;padding:1px 5px;font-family:Consolas,monospace;font-size:12px;color:#cfe6ff}
</style></head>
<body>
<header><h1>__TITLE__</h1><div class="sub">__SUBTITLE__</div></header>
<div class="wrap">
  <div class="controls">
    <div class="seg" id="seg"></div>
    <div class="zoom"><span style="color:var(--dim)">缩放</span>
      <input type="range" id="zoom" min="100" max="800" step="25" value="250">
      <b id="zoomVal">250%</b>
      <button class="btn" id="fitBtn">适应</button></div>
    <button class="btn" id="syncBtn">联动：开</button>
    <span class="hint">拖任意图 = 同步平移 · 悬停滚轮 = 同步缩放</span>
  </div>
  <div id="groups"></div>
  <div class="card" id="metricsCard"><h2>量化指标（同 prompt 同参数）</h2><div id="metricsBox"></div>
    <p class="note" id="metricNote">__METRIC_NOTE__</p></div>
  <div class="card"><h2>说明</h2><p class="note">__NOTE__</p></div>
</div>
<script>
const DATA = __PAYLOAD__;
const groupsBox = document.getElementById('groups');
const segBox = document.getElementById('seg');
let zoom = 250, sync = true, stages = [];

function render(groupIndex){
  stages = [];
  groupsBox.innerHTML = '';
  const g = DATA.groups[groupIndex];
  const n = g.panes.length;
  const gh = document.createElement('div'); gh.className = 'group';
  gh.innerHTML = `<div class="ghead"><h2>${g.name}</h2><span class="desc">${g.desc||''}</span></div>
    <div class="grid" style="grid-template-columns:repeat(${n},1fr)"></div>`;
  const grid = gh.querySelector('.grid');
  g.panes.forEach(p=>{
    const col = document.createElement('div');
    col.className = 'col ' + (p.cls || '');
    col.innerHTML = `<div class="head"><span class="name">${p.name}</span>
      ${p.tag?`<span class="tag ${p.cls==='best'?'g':(p.cls==='src'?'b':'')}">${p.tag}</span>`:''}</div>
      <div class="stage"><img src="${p.img}" alt="${p.name}"></div>
      <div class="meta">${p.meta||''}</div>`;
    grid.appendChild(col);
    const stage = col.querySelector('.stage'), img = col.querySelector('img');
    stages.push({stage, img});
    bind(stage);
  });
  groupsBox.appendChild(gh);
  applyZoom(zoom, true);
}

function applyZoom(z, keep){
  zoom = z; document.getElementById('zoom').value = z;
  document.getElementById('zoomVal').textContent = z + '%';
  stages.forEach(({stage, img})=>{
    const cx = stage.scrollLeft + stage.clientWidth/2, cy = stage.scrollTop + stage.clientHeight/2;
    img.style.width = z + '%';
    if(keep){ stage.scrollLeft = Math.max(0, cx - stage.clientWidth/2); stage.scrollTop = Math.max(0, cy - stage.clientHeight/2); }
  });
}

function bind(stage){
  stage.addEventListener('scroll', ()=>{
    if(!sync) return;
    stages.forEach(o=>{ if(o.stage !== stage){ o.stage.scrollLeft = stage.scrollLeft; o.stage.scrollTop = stage.scrollTop; } });
  });
  let drag=false, sx=0, sy=0, ox=0, oy=0;
  stage.addEventListener('mousedown', e=>{ if(e.button) return; drag=true; stage.classList.add('drag');
    sx=e.clientX; sy=e.clientY; ox=stage.scrollLeft; oy=stage.scrollTop; e.preventDefault(); });
  window.addEventListener('mousemove', e=>{ if(!drag) return;
    stage.scrollLeft = ox-(e.clientX-sx); stage.scrollTop = oy-(e.clientY-sy); });
  window.addEventListener('mouseup', ()=>{ drag=false; stage.classList.remove('drag'); });
  stage.addEventListener('wheel', e=>{ if(!sync) return; e.preventDefault();
    const next = Math.min(800, Math.max(100, zoom + (e.deltaY<0?25:-25)));
    if(next!==zoom) applyZoom(next, true); }, {passive:false});
}

function metrics(){
  const rows = DATA.metrics || [];
  if(!rows.length){ document.getElementById('metricsCard').style.display='none'; return; }
  const cols = Object.keys(rows[0]).filter(k=>k!=='name'&&k!=='group');
  const lower = {flat:1, frag:1, iso:1, grad:1};
  let html = '<table><thead><tr><th>样本</th>' + cols.map(c=>`<th>${c}</th>`).join('') + '</tr></thead><tbody>';
  const best = {};
  cols.forEach(c=>{
    const vals = rows.map(r=>r[c]).filter(v=>typeof v==='number');
    best[c] = lower[c] ? Math.min(...vals) : Math.max(...vals);
  });
  rows.forEach(r=>{
    html += `<tr><td>${r.name}</td>` + cols.map(c=>{
      const v = r[c];
      if(typeof v !== 'number') return `<td>${v}</td>`;
      const isBest = Math.abs(v - best[c]) < 1e-9;
      return `<td class="${isBest?'best-cell':''}">${v.toFixed(v>=100?1:4)}</td>`;
    }).join('') + '</tr>';
  });
  document.getElementById('metricsBox').innerHTML = html + '</tbody></table>';
}

document.getElementById('zoom').addEventListener('input', e=> applyZoom(+e.target.value, true));
document.getElementById('fitBtn').addEventListener('click', ()=>{
  stages.forEach(({stage, img})=>{
    const ratio = stage.clientWidth / (img.naturalWidth || 1024);
    img.style.width = (ratio*100)+'%'; stage.scrollLeft=0; stage.scrollTop=0;
  });
  document.getElementById('zoomVal').textContent='适应';
});
document.getElementById('syncBtn').addEventListener('click', e=>{
  sync=!sync; e.target.textContent='联动：'+(sync?'开':'关'); e.target.classList.toggle('on', sync);
});

DATA.groups.forEach((g,i)=>{
  const b = document.createElement('button'); b.textContent = g.name; b.dataset.i = i;
  if(i===0) b.classList.add('on');
  b.addEventListener('click', ()=>{ segBox.querySelectorAll('button').forEach(x=>x.classList.remove('on'));
    b.classList.add('on'); render(i); });
  segBox.appendChild(b);
});
render(0); metrics();
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    payload = json.load(open(a.data, encoding="utf-8"))
    html = (TPL.replace("__TITLE__", payload["title"])
               .replace("__SUBTITLE__", payload.get("subtitle", ""))
               .replace("__METRIC_NOTE__", payload.get("metric_note", ""))
               .replace("__NOTE__", payload.get("note", ""))
               .replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False)))
    open(a.out, "w", encoding="utf-8").write(html)
    print("written", a.out, len(html), "bytes")


if __name__ == "__main__":
    main()
