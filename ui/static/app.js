const $ = (q) => document.querySelector(q);
const $$ = (q) => [...document.querySelectorAll(q)];
const state = { boot: null, hand: "all", result: null, poll: null };
const palette = ["#087e78", "#65a743", "#f26b3a", "#735aa5", "#bf8438", "#63767b", "#a3aaa5"];

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
}
function pct(value, digits=1) { return value == null || Number.isNaN(+value) ? "—" : `${(+value * 100).toFixed(digits)}%`; }
function toast(message) { const el=$("#toast"); el.textContent=message; el.classList.add("show"); setTimeout(()=>el.classList.remove("show"),2600); }
async function api(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}

async function init() {
  try {
    state.boot = await api("/api/bootstrap");
    $("#seasonPill").textContent = `${state.boot.season} STATCAST`;
    fillSelect($("#runSelect"), state.boot.runs, x=>x.id, x=>x.label);
    fillSelect($("#pitcherSelect"), state.boot.pitchers, x=>x.id, x=>`${x.name} · ${x.throws}投 · ${x.pitches.toLocaleString()} 球`);
    const yamamoto = state.boot.pitchers.find(x=>x.id===808967);
    if (yamamoto) $("#pitcherSelect").value=yamamoto.id;
    renderBatters();
    const judge = state.boot.batters.find(x=>x.id===592450);
    if (judge) $("#batterSelect").value=judge.id;
    renderHistory();
  } catch (error) { toast(`載入失敗：${error.message}`); }
}
function fillSelect(element, items, value, label) {
  element.innerHTML = items.map(x=>`<option value="${esc(value(x))}">${esc(label(x))}</option>`).join("");
}
function renderBatters() {
  const prior = +$("#batterSelect").value;
  const rows = state.boot.batters.filter(x=>state.hand==="all" || x.stands.includes(state.hand));
  fillSelect($("#batterSelect"), rows, x=>x.id, x=>`${x.name} · ${x.stands.join("/")}打`);
  if (rows.some(x=>x.id===prior)) $("#batterSelect").value=prior;
}
function renderHistory() {
  const rows=state.boot.results;
  if (!rows.length) return;
  $("#historyWrap").hidden=false;
  $("#historyList").innerHTML=rows.map(x=>`<button class="history-card" data-result="${x.id}"><b>${esc(x.pitcher)} → ${esc(x.batter)}</b><small>${x.source==="legacy"?"既有報告":(x.generated_at||"").slice(0,10)}</small></button>`).join("");
  $$(".history-card").forEach(button=>button.onclick=()=>loadResult(button.dataset.result));
}

$$('.hand-filter button').forEach(button=>button.onclick=()=>{
  $$('.hand-filter button').forEach(x=>x.classList.remove('active')); button.classList.add('active');
  state.hand=button.dataset.hand; renderBatters();
});
$("#aboutButton").onclick=()=>$("#aboutDialog").showModal();
$(".dialog-close").onclick=()=>$("#aboutDialog").close();
$("#analyzeButton").onclick=startAnalysis;
$("#treeDepth").onchange=()=>renderTree();
$("#downloadTree").onclick=downloadTree;

async function startAnalysis() {
  const button=$("#analyzeButton"); button.disabled=true;
  try {
    const job=await api("/api/analyze", {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({
      run:$("#runSelect").value,pitcher_id:+$("#pitcherSelect").value,batter_id:+$("#batterSelect").value
    })});
    if (job.status==="cached") { toast("已載入相同對戰的快取結果"); button.disabled=false; return loadResult(job.result_id); }
    $("#jobPanel").hidden=false; $("#results").hidden=true;
    $("#jobTitle").textContent=`${job.pitcher} vs ${job.batter}`;
    $("#jobLog").textContent="排入分析工作…";
    $("#jobPanel").scrollIntoView({behavior:"smooth",block:"start"});
    pollJob(job.id);
  } catch(error) { button.disabled=false; toast(error.message); }
}
async function pollJob(id) {
  clearTimeout(state.poll);
  try {
    const job=await api(`/api/jobs/${id}`);
    $("#jobLog").textContent=(job.progress||[]).slice(-30).join("\n") || "準備資料…";
    $("#jobLog").scrollTop=$("#jobLog").scrollHeight;
    if (job.status==="complete") { $("#jobPanel").hidden=true; $("#analyzeButton").disabled=false; toast("分析完成"); return loadResult(job.result_id); }
    if (job.status==="failed") throw new Error(job.error || "分析失敗");
    state.poll=setTimeout(()=>pollJob(id),1800);
  } catch(error) { $("#jobPanel").hidden=true; $("#analyzeButton").disabled=false; toast(error.message); }
}
async function loadResult(id) {
  try {
    state.result=await api(`/api/results/${id}`);
    renderResult();
    $("#results").hidden=false;
    $("#results").scrollIntoView({behavior:"smooth",block:"start"});
  } catch(error) { toast(error.message); }
}

function strategy(nameStarts) { return state.result.strategies.find(x=>x.name.startsWith(nameStarts)); }
function renderResult() {
  const r=state.result, adaptive=strategy("最佳應變"), tendency=strategy("投手實際傾向");
  $("#pitcherName").textContent=r.pitcher.name; $("#batterName").textContent=r.batter.name;
  $("#contextText").textContent=r.context; $("#handTag").textContent=r.batter.stand==="R"?"右打者":"左打者";
  $("#modelTag").textContent=r.legacy?"既有分析":"模型模擬";
  const best=adaptive?.curve?.[4], base=tendency?.curve?.[4], hard=adaptive?.outcomes?.hard;
  const reach=adaptive?.curve?.findIndex(x=>x!=null && x>=r.threshold);
  $("#metricGrid").innerHTML=[
    ["5 球內最佳三振率",pct(best),"最佳應變策略","primary"],
    ["相較實際傾向",best!=null&&base!=null?`${best-base>=0?"+":""}${((best-base)*100).toFixed(1)}<small> pt</small>`:"—",`原有配球 ${pct(base)}`,""],
    ["5 球內強擊風險",pct(hard),"最佳應變策略",""],
    ["達到門檻",reach>=0?`第 ${reach+1} 球`:"未達",`目標 ${pct(r.threshold)}`,""]
  ].map(x=>`<article class="metric panel ${x[3]}"><label>${x[0]}</label><strong>${x[1]}</strong><small>${x[2]}</small></article>`).join("");
  renderChart(); renderFixed(); renderStrategyTable(); renderPaths(); renderTree();
  $("#thresholdLabel").textContent=`門檻 ${pct(r.threshold)}`;
  $("#fullReport").textContent=r.report_markdown || "此分析沒有文字報告。";
}

function renderChart() {
  const series=state.result.strategies.filter(x=>x.curve?.some(v=>v!=null));
  const width=720,height=285,m={l:42,r:14,t:16,b:34}, iw=width-m.l-m.r,ih=height-m.t-m.b;
  const sx=i=>m.l+i*iw/4, sy=v=>m.t+(1-v)*ih;
  let svg=`<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="球數內累積三振機率">`;
  for(let i=0;i<=4;i++){const v=i*.25,y=sy(v);svg+=`<line x1="${m.l}" y1="${y}" x2="${width-m.r}" y2="${y}" stroke="#d9d7cd"/><text x="${m.l-8}" y="${y+3}" text-anchor="end">${Math.round(v*100)}%</text>`}
  for(let i=0;i<5;i++)svg+=`<text x="${sx(i)}" y="${height-9}" text-anchor="middle">${i+1} 球</text>`;
  series.forEach((s,index)=>{const points=s.curve.map((v,i)=>v==null?null:[sx(i),sy(v)]).filter(Boolean);if(!points.length)return;const color=palette[index%palette.length],dash=s.is_actual?'6 5':'none';svg+=`<polyline points="${points.map(p=>p.join(',')).join(' ')}" fill="none" stroke="${color}" stroke-width="${index===0?3:1.8}" stroke-dasharray="${dash}"/>`;points.forEach(p=>svg+=`<circle cx="${p[0]}" cy="${p[1]}" r="${index===0?4:2.5}" fill="${color}"/>`)});
  const ty=sy(state.result.threshold);svg+=`<line x1="${m.l}" y1="${ty}" x2="${width-m.r}" y2="${ty}" stroke="#b84232" stroke-dasharray="3 4"/><text x="${width-m.r}" y="${ty-5}" text-anchor="end" fill="#b84232">門檻 ${pct(state.result.threshold,0)}</text></svg>`;
  $("#curveChart").innerHTML=svg;
  $("#chartLegend").innerHTML=series.map((s,i)=>`<span style="--legend:${palette[i%palette.length]}">${esc(s.name.replace(/ \(.+/,""))}</span>`).join("");
}
function renderFixed() {
  const rows=state.result.fixed_sequences||[], top=rows[0];
  $("#fixedK").textContent=top?pct(top.k_probability):"—";
  $("#pitchSequence").innerHTML=top?top.pitches.map((p,i)=>`<span class="pitch"><b>${esc(p)}</b>${i<top.pitches.length-1?'<i>→</i>':''}</span>`).join(""):"沒有固定序列資料";
  $("#rankedList").innerHTML=rows.slice(1,5).map(x=>`<div class="ranked-row"><span>#${x.rank}</span><div>${x.pitches.map(esc).join(" → ")}</div><b>${pct(x.k_probability)}</b></div>`).join("");
}
function renderStrategyTable() {
  const rows=state.result.strategies,best=Math.max(...rows.filter(x=>!x.is_actual).map(x=>x.curve?.[4]??-1));
  $("#strategyRows").innerHTML=rows.map(s=>`<tr class="${s.is_actual?'actual-row ':''}${s.curve?.[4]===best?'best-row':''}"><td>${esc(s.name)}</td><td>${pct(s.curve?.[2])}</td><td>${pct(s.curve?.[3])}</td><td>${pct(s.curve?.[4])}</td><td>${pct(s.outcomes?.BB)}</td><td>${pct(s.outcomes?.hard)}</td><td>${pct(s.outcomes?.soft)}</td><td>${pct(s.outcomes?.unfinished)}</td></tr>`).join("");
}
function renderPaths() {
  const choices=state.result.strategies.filter(s=>s.top_paths?.K?.length);
  const panel=$("#pathsPanel"), select=$("#pathStrategy");
  if(!choices.length){panel.hidden=true;return} panel.hidden=false;
  select.innerHTML=choices.map((s,i)=>`<option value="${i}">${esc(s.name)}</option>`).join("");
  const draw=()=>{
    const paths=choices[+select.value].top_paths.K;
    $("#pathList").innerHTML=paths.map((p,i)=>`<div class="path-row"><strong>${pct(p.probability,2)}</strong><p><small>#${i+1}</small> ${esc(p.sequence).replaceAll("；","<br>").replaceAll("**三振**","<strong>三振</strong>")}</p></div>`).join("");
  };
  select.onchange=draw; draw();
}

function renderTree() {
  if (!state.result) return;
  const tree=state.result.policy_tree, empty=$("#treeEmpty"), host=$("#decisionTree");
  if(!tree){host.innerHTML="";empty.hidden=false;return} empty.hidden=true;
  const max=+$("#treeDepth").value;
  host.innerHTML=treeHtml(tree,1,max,null,true);
}
function treeHtml(node,depth,max,edge,root=false) {
  const branch=edge?`<span class="branch-label"><b>${esc(edge.reaction)}</b> ${pct(edge.probability,0)}${edge.next_count?` · ${edge.next_count}`:""}</span>`:"";
  if(node.type==="terminal"){
    const cls=node.result==="三振"?"k":(["強擊","保送"].includes(node.result)?"risk":"");
    return `<div class="tree-node ${root?'root':''}">${branch}<span class="terminal ${cls}">${esc(node.result)}</span></div>`;
  }
  if(depth>max)return `<div class="tree-node">${branch}<span class="cutoff">還有後續策略 · 調高顯示深度</span></div>`;
  return `<div class="tree-node ${root?'root':''}"><details ${depth<=2?'open':''}><summary>${branch}<span class="decision-pill"><small>${esc(node.count)}</small>${esc(node.pitch)}</span></summary><div>${(node.children||[]).map(child=>treeHtml(child.target,depth+1,max,child)).join("")}</div></details></div>`;
}

function makeTreeSvg(tree,maxDepth) {
  let leaves=0,nodes=[],edges=[];
  function walk(node,depth,parent=null,edge=null){
    const item={node,depth,parent,edge,x:0};nodes.push(item);if(parent)edges.push([parent,item]);
    if(node.type!=="decision"||depth>=maxDepth||!node.children?.length){item.x=leaves++*190+95;return item.x}
    const xs=node.children.map(c=>walk(c.target,depth+1,item,c).x);item.x=(Math.min(...xs)+Math.max(...xs))/2;return item.x;
  }
  walk(tree,1);const width=Math.max(600,leaves*190),height=maxDepth*145+70;
  const xe=s=>String(s).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&apos;"}[c]));
  let svg=`<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}"><rect width="100%" height="100%" fill="#f4f1e8"/><style>text{font-family:sans-serif}.e{font-size:10px;fill:#63767b}.n{font-size:11px;font-weight:bold;fill:white}.c{font-size:9px;fill:#c9ed62}.t{font-size:11px;font-weight:bold;fill:#09232b}</style>`;
  edges.forEach(([a,b])=>{const y1=a.depth*135-75,y2=b.depth*135-98;svg+=`<path d="M${a.x} ${y1+28} C${a.x} ${y1+70},${b.x} ${y2-30},${b.x} ${y2}" fill="none" stroke="#aab4ae"/><text class="e" x="${(a.x+b.x)/2}" y="${(y1+y2)/2+20}" text-anchor="middle">${xe(b.edge.reaction)} ${pct(b.edge.probability,0)}</text>`});
  nodes.forEach(x=>{const y=x.depth*135-100;if(x.node.type==="decision"){svg+=`<rect x="${x.x-62}" y="${y}" width="124" height="48" rx="10" fill="#09232b"/><text class="c" x="${x.x}" y="${y+15}" text-anchor="middle">${xe(x.node.count)}</text><text class="n" x="${x.x}" y="${y+34}" text-anchor="middle">${xe(x.node.pitch)}</text>`}else{const fill=x.node.result==="三振"?'#9ed9c9':(["強擊","保送"].includes(x.node.result)?'#f5b093':'#dfe4d3');svg+=`<rect x="${x.x-55}" y="${y}" width="110" height="34" rx="9" fill="${fill}"/><text class="t" x="${x.x}" y="${y+22}" text-anchor="middle">${xe(x.node.result)}</text>`}});return svg+`</svg>`;
}
function downloadTree() {
  if(!state.result?.policy_tree)return toast("此結果沒有決策樹資料");
  const svg=makeTreeSvg(state.result.policy_tree,+$("#treeDepth").value),blob=new Blob([svg],{type:"image/svg+xml"}),url=URL.createObjectURL(blob),a=document.createElement("a");
  a.href=url;a.download=`${state.result.pitcher.name}_vs_${state.result.batter.name}_策略樹.svg`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}

init();
