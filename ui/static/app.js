const $ = (q) => document.querySelector(q);
const $$ = (q) => [...document.querySelectorAll(q)];
const state = { boot: null, hand: "all", result: null, resultId: null, poll: null, jobsPoll: null, activeJobIds: new Set() };
const palette = ["#087e78", "#65a743", "#f26b3a", "#735aa5", "#bf8438", "#63767b", "#a3aaa5"];
const pitchColors = {"四縫線":"#2474c8","伸卡":"#e88931","卡特":"#23966f","滑球":"#7654b3","曲球":"#d14b4b","變速":"#1b9aaa","其他":"#718087"};

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
  renderZoneDefinition();
  try {
    state.boot = await api("/api/bootstrap");
    $("#seasonPill").textContent = `${state.boot.season} STATCAST`;
    $("#modelName").textContent=state.boot.model.label;
    fillSelect($("#pitcherSelect"), state.boot.pitchers, x=>x.id, x=>`${x.name} · ${x.throws}投 · ${x.pitches.toLocaleString()} 球`);
    const yamamoto = state.boot.pitchers.find(x=>x.id===808967);
    if (yamamoto) $("#pitcherSelect").value=yamamoto.id;
    renderBatters();
    const judge = state.boot.batters.find(x=>x.id===592450);
    if (judge) $("#batterSelect").value=judge.id;
    renderHistory();
    await refreshActiveJobs();
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
  if (!rows.length) {$("#historyWrap").hidden=true;return}
  $("#historyWrap").hidden=false;
  $("#historyList").innerHTML=rows.map(x=>`<button class="history-card ${x.outdated?'outdated':''}" data-result="${x.id}"><b>${esc(x.pitcher)} → ${esc(x.batter)}</b><small>${x.outdated?'舊版結果':`${(x.generated_at||'').slice(0,10)} · 已保存`}</small></button>`).join("");
  $$(".history-card").forEach(button=>button.onclick=()=>loadResult(button.dataset.result));
}
async function refreshHistory() {
  const fresh=await api("/api/bootstrap");
  state.boot.results=fresh.results;
  renderHistory();
}

function elapsedTime(startedAt) {
  const seconds=Math.max(0,Math.floor((Date.now()-new Date(startedAt).getTime())/1000));
  if(!Number.isFinite(seconds))return "剛剛開始";
  const minutes=Math.floor(seconds/60),remainder=seconds%60;
  return minutes?`${minutes} 分 ${remainder} 秒`:`${remainder} 秒`;
}
function renderActiveJobs(jobs) {
  const panel=$("#jobPanel"),button=$("#analyzeButton");
  panel.hidden=!jobs.length;
  button.disabled=jobs.length>0;
  if(!jobs.length){$("#jobList").innerHTML="";return}
  $("#activeJobCount").textContent=`${jobs.length} 項執行中`;
  $("#jobList").innerHTML=jobs.map(job=>{
    const lines=(job.progress||[]).slice(-12).join("\n") || "準備資料與載入模型…";
    return `<article class="job-item" data-job-id="${esc(job.id)}"><div class="job-item-head"><div><span class="job-status"><i></i>${job.status==="queued"?"等待執行":"分析中"}</span><h3>${esc(job.pitcher)} <small>VS</small> ${esc(job.batter)}</h3><p>已執行 ${elapsedTime(job.started_at)} · 開始於 ${esc((job.started_at||"").replace("T"," "))}</p></div><div class="spinner" aria-hidden="true"></div></div><div class="progress-track"><i></i></div><details open><summary>最新執行紀錄</summary><pre>${esc(lines)}</pre></details></article>`;
  }).join("");
}
async function refreshActiveJobs() {
  clearTimeout(state.jobsPoll);
  try {
    const payload=await api("/api/jobs"),jobs=payload.jobs||[];
    const previous=state.activeJobIds;
    state.activeJobIds=new Set(jobs.map(job=>job.id));
    renderActiveJobs(jobs);
    if(previous.size && !jobs.length)await refreshHistory();
  } catch(error) {
    if(state.boot)toast(`無法更新分析狀態：${error.message}`);
  } finally {
    state.jobsPoll=setTimeout(refreshActiveJobs,1800);
  }
}

$$('.hand-filter button').forEach(button=>button.onclick=()=>{
  $$('.hand-filter button').forEach(x=>x.classList.remove('active')); button.classList.add('active');
  state.hand=button.dataset.hand; renderBatters();
});
$("#aboutButton").onclick=()=>$("#aboutDialog").showModal();
$(".dialog-close").onclick=()=>$("#aboutDialog").close();
$("#analyzeButton").onclick=startAnalysis;
$("#downloadBundle").onclick=()=>downloadBundle(false);
$("#treeDepth").onchange=()=>renderTree();
$("#downloadTree").onclick=downloadTree;

async function startAnalysis() {
  const button=$("#analyzeButton"); button.disabled=true;
  try {
    const job=await api("/api/analyze", {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({
      pitcher_id:+$("#pitcherSelect").value,batter_id:+$("#batterSelect").value
    })});
    if (job.status==="cached") {
      toast("已載入相同對戰的快取結果"); button.disabled=false;
      if(await loadResult(job.result_id)){await refreshHistory();downloadBundle(true)} return;
    }
    state.activeJobIds.add(job.id); renderActiveJobs([job]); $("#results").hidden=true;
    $("#jobPanel").scrollIntoView({behavior:"smooth",block:"start"});
    pollJob(job.id);
  } catch(error) { button.disabled=false; toast(error.message); }
}
async function pollJob(id) {
  clearTimeout(state.poll);
  try {
    const job=await api(`/api/jobs/${id}`);
    if (job.status==="complete") {
      await refreshActiveJobs(); $("#analyzeButton").disabled=false;
      if(await loadResult(job.result_id)){await refreshHistory();downloadBundle(true)} return;
    }
    if (job.status==="failed") throw new Error(job.error || "分析失敗");
    renderActiveJobs([job]);
    state.poll=setTimeout(()=>pollJob(id),1800);
  } catch(error) { await refreshActiveJobs(); $("#analyzeButton").disabled=false; toast(error.message); }
}
async function loadResult(id) {
  try {
    state.result=await api(`/api/results/${id}`);
    state.resultId=id;
    renderResult();
    $("#results").hidden=false;
    $("#results").scrollIntoView({behavior:"smooth",block:"start"});
    return true;
  } catch(error) { state.resultId=null; toast(error.message); return false; }
}
function downloadBundle(automatic=false) {
  if(!state.resultId) return toast("請先開啟一份分析結果");
  const link=document.createElement("a");
  link.href=`/api/results/${state.resultId}/download`;
  link.download=""; document.body.appendChild(link); link.click(); link.remove();
  if(automatic) toast("分析完成，正在下載完整結果 ZIP");
}

function strategy(nameStarts) { return state.result.strategies.find(x=>x.name.startsWith(nameStarts)); }
function renderResult() {
  const r=state.result, adaptive=strategy("最佳應變"), tendency=strategy("投手實際傾向");
  $("#pitcherName").textContent=r.pitcher.name; $("#batterName").textContent=r.batter.name;
  $("#contextText").textContent=r.context; $("#handTag").textContent=r.batter.stand==="R"?"右打者":"左打者";
  $("#modelTag").textContent=(r.schema_version||0)<2?"舊版分析 · 已保存":"最新模型 · 已保存";
  const best=adaptive?.curve?.[4], base=tendency?.curve?.[4], hard=adaptive?.outcomes?.hard;
  const reach=adaptive?.curve?.findIndex(x=>x!=null && x>=r.threshold);
  $("#metricGrid").innerHTML=[
    ["5 球內最佳三振率",pct(best),"最佳應變策略","primary"],
    ["相較實際傾向",best!=null&&base!=null?`${best-base>=0?"+":""}${((best-base)*100).toFixed(1)}<small> pt</small>`:"—",`原有配球 ${pct(base)}`,""],
    ["5 球內強擊風險",pct(hard),"最佳應變策略",""],
    ["達到門檻",reach>=0?`第 ${reach+1} 球`:"未達",`目標 ${pct(r.threshold)}`,""]
  ].map(x=>`<article class="metric panel ${x[3]}"><label>${x[0]}</label><strong>${x[1]}</strong><small>${x[2]}</small></article>`).join("");
  renderChart(); renderFixed(); renderStrategyTable(); renderTree();
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
  $("#strategyRows").innerHTML=rows.map((s,index)=>{
    const sequences=strategySequences(s), expandable=sequences.length>0;
    const name=expandable?`<button class="strategy-toggle" data-strategy="${index}" aria-expanded="false"><span>${esc(s.name)}</span><i>⌄</i></button>`:esc(s.name);
    const main=`<tr class="strategy-row ${s.is_actual?'actual-row ':''}${s.curve?.[4]===best?'best-row':''}"><td>${name}</td><td>${pct(s.curve?.[2])}</td><td>${pct(s.curve?.[3])}</td><td>${pct(s.curve?.[4])}</td><td>${pct(s.outcomes?.BB)}</td><td>${pct(s.outcomes?.hard)}</td><td>${pct(s.outcomes?.soft)}</td><td>${pct(s.outcomes?.unfinished)}</td></tr>`;
    const detail=expandable?`<tr class="strategy-detail-row" id="strategyDetail${index}" hidden><td colspan="8"><div class="strategy-visual" id="strategyVisual${index}"></div></td></tr>`:"";
    return main+detail;
  }).join("");
  $$(".strategy-toggle").forEach(button=>button.onclick=()=>{
    const index=+button.dataset.strategy, detail=$(`#strategyDetail${index}`), opening=detail.hidden;
    $$(".strategy-detail-row").forEach(row=>row.hidden=true);
    $$(".strategy-toggle").forEach(item=>item.setAttribute("aria-expanded","false"));
    if(opening){detail.hidden=false;button.setAttribute("aria-expanded","true");renderStrategyVisual(index,0)}
  });
}
function candidateDetails(variants) {
  const candidates=state.result.candidates||[];
  return variants.map((variant,pitchIndex)=>{
    const row=candidates.find(c=>c.variant===variant);
    return row?{...row,variant,family_zh:variant.split("_",1)[0],region:variant.split("_").slice(1).join("_"),pitch_number:pitchIndex+1}:null;
  }).filter(Boolean);
}
function detailsFromPath(path) {
  const variants=[...String(path||"").matchAll(/第\s*\d+\s*球\s*\([^)]+\)\s*([^→；]+?)\s*→/g)].map(match=>match[1].trim());
  return candidateDetails(variants);
}
function strategySequences(strategyRow) {
  if(strategyRow.name.startsWith("最佳固定")){
    return (state.result.fixed_sequences||[]).slice(0,5).map(row=>({
      probability:row.k_probability,label:`固定序列 #${row.rank}`,
      pitches:row.pitch_details?.length?row.pitch_details:candidateDetails(row.pitches||[]),sequence:(row.pitches||[]).join(" → ")
    })).filter(row=>row.pitches.length);
  }
  return (strategyRow.top_paths?.K||[]).slice(0,5).map((row,index)=>({
    probability:row.probability,label:`三振路徑 #${index+1}`,
    pitches:row.pitches?.length?row.pitches:detailsFromPath(row.sequence),sequence:row.sequence
  })).filter(row=>row.pitches.length);
}
function renderStrategyVisual(strategyIndex,sequenceIndex) {
  const strategyRow=state.result.strategies[strategyIndex],sequences=strategySequences(strategyRow),selected=sequences[sequenceIndex];
  if(!selected)return;
  const host=$(`#strategyVisual${strategyIndex}`);
  host.innerHTML=`<div class="visual-controls"><div><p class="eyebrow">PITCH LOCATION MAP</p><h4>${esc(strategyRow.name)}</h4></div><label>選擇序列<select class="sequence-choice">${sequences.map((row,i)=>`<option value="${i}" ${i===sequenceIndex?'selected':''}>${esc(row.label)} · ${pct(row.probability,2)}</option>`).join("")}</select></label></div><div class="location-layout"><div class="pitch-map">${pitchMapSvg(selected.pitches,state.result.batter.stand)}</div><aside><div class="pitch-map-legend">${pitchLegend(selected.pitches)}</div><div class="mapped-sequence">${selected.pitches.map((pitch,i)=>`<div><b style="--pitch:${pitchColor(pitch)}">${i+1}</b><span><strong>${esc(pitch.variant)}</strong><small>${pitch.count?`球數 ${esc(pitch.count)} · `:""}${formatCoordinate(pitch)}</small></span></div>`).join("")}</div></aside></div>`;
  host.querySelector(".sequence-choice").onchange=event=>renderStrategyVisual(strategyIndex,+event.target.value);
}
function pitchColor(pitch) {
  const family=pitch.family_zh||String(pitch.variant||"").split("_")[0]||"其他";
  return pitchColors[family]||pitchColors.其他;
}
function formatCoordinate(pitch) {
  const x=Number(pitch.plate_x_bv),z=Number(pitch.plate_z_norm);
  return `${esc(pitch.region||"")} · x ${Number.isFinite(x)?x.toFixed(2):"—"} / z ${Number.isFinite(z)?z.toFixed(2):"—"}`;
}
function pitchLegend(pitches) {
  const families=[...new Map(pitches.map(p=>[p.family_zh||String(p.variant).split("_")[0],pitchColor(p)])).entries()];
  return families.map(([name,color])=>`<span><i style="--pitch:${color}"></i>${esc(name)}</span>`).join("");
}
function batterSilhouette(stand) {
  const transform=stand==="L"?'translate(620 0) scale(-1 1)':'';
  return `<g transform="${transform}" opacity=".14" fill="#09232b" stroke="#09232b" stroke-linecap="round"><circle cx="48" cy="166" r="15"/><path d="M44 183 Q55 205 48 244 L29 300 M48 244 L69 300" fill="none" stroke-width="13"/><path d="M48 198 L78 220" fill="none" stroke-width="11"/><path d="M51 191 L80 169" fill="none" stroke-width="10"/><line x1="72" y1="177" x2="127" y2="92" stroke-width="8"/></g>`;
}
function pitchMapSvg(pitches,stand) {
  const width=620,height=410,left=110,right=510,top=28,bottom=354;
  const xmin=-2.1,xmax=2.1,ymin=-.55,ymax=1.55;
  const sx=x=>left+(x-xmin)/(xmax-xmin)*(right-left),sy=y=>bottom-(y-ymin)/(ymax-ymin)*(bottom-top);
  const zx0=sx(-1),zx1=sx(1),zy0=sy(0),zy1=sy(1),zw=zx1-zx0,zh=zy0-zy1;
  const located=pitches.map((pitch,index)=>{
    const relative=Number(pitch.plate_x_bv),z=Number(pitch.plate_z_norm);
    if(!Number.isFinite(relative)||!Number.isFinite(z))return null;
    return {...pitch,index,x:stand==="L"?-relative:relative,z};
  }).filter(Boolean);
  const insideLeft=stand==="L"?"外角":"內角",insideRight=stand==="L"?"內角":"外角";
  let svg=`<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${stand==='L'?'左打':'右打'}配球落點圖"><rect x="${left}" y="${top}" width="${right-left}" height="${bottom-top}" rx="13" fill="#f2f0e8" stroke="#d9d7cd"/>${batterSilhouette(stand)}<text x="${stand==='L'?570:50}" y="326" text-anchor="middle" class="batter-label">${stand==='L'?'左打':'右打'}</text>`;
  svg+=`<rect x="${zx0}" y="${zy1}" width="${zw}" height="${zh}" fill="#fffdf7" stroke="#09232b" stroke-width="2"/>`;
  for(let i=1;i<3;i++){svg+=`<line x1="${zx0+zw*i/3}" y1="${zy1}" x2="${zx0+zw*i/3}" y2="${zy0}" class="zone-line"/><line x1="${zx0}" y1="${zy1+zh*i/3}" x2="${zx1}" y2="${zy1+zh*i/3}" class="zone-line"/>`}
  svg+=`<text x="${zx0+6}" y="${zy1-8}" class="corner-label">${insideLeft}</text><text x="${zx1-6}" y="${zy1-8}" text-anchor="end" class="corner-label">${insideRight}</text><path d="M${sx(-.34)} ${bottom+5} L${sx(.34)} ${bottom+5} L${sx(.25)} ${bottom+20} L${sx(0)} ${bottom+29} L${sx(-.25)} ${bottom+20} Z" fill="none" stroke="#9aa7a5"/>`;
  if(located.length>1)svg+=`<polyline points="${located.map(p=>`${sx(p.x)},${sy(p.z)}`).join(" ")}" fill="none" stroke="#899794" stroke-width="2" stroke-dasharray="5 6"/>`;
  located.forEach(p=>{const cx=sx(Math.max(xmin,Math.min(xmax,p.x))),cy=sy(Math.max(ymin,Math.min(ymax,p.z))),color=pitchColor(p);svg+=`<g><circle cx="${cx}" cy="${cy}" r="14" fill="${color}" stroke="#fffdf7" stroke-width="3"><title>第 ${p.index+1} 球 ${esc(p.variant)}｜${formatCoordinate(p)}</title></circle><text x="${cx}" y="${cy+4}" text-anchor="middle" class="pitch-number">${p.index+1}</text></g>`});
  svg+=`<text x="${(left+right)/2}" y="394" text-anchor="middle" class="axis-note">投捕視角 · 虛線連接投球順序 · 方框為打者個人化好球帶</text></svg>`;
  return svg;
}
function renderZoneDefinition() {
  const host=$("#zoneDefinition"); if(!host)return;
  const cells=[["內角高","中間高","外角高"],["內角中","正中","外角中"],["內角低","中間低","外角低"]];
  let svg=`<svg viewBox="0 0 380 310" role="img" aria-label="好球帶九宮格和壞球區域定義"><rect x="18" y="12" width="344" height="46" rx="9" class="outside-cell"/><text x="190" y="40" text-anchor="middle">帶外高</text><rect x="18" y="58" width="60" height="180" rx="9" class="outside-cell"/><text x="48" y="138" text-anchor="middle"><tspan x="48">帶外</tspan><tspan x="48" dy="18">內角</tspan></text><rect x="302" y="58" width="60" height="180" rx="9" class="outside-cell"/><text x="332" y="138" text-anchor="middle"><tspan x="332">帶外</tspan><tspan x="332" dy="18">外角</tspan></text><rect x="18" y="238" width="344" height="46" rx="9" class="outside-cell"/><text x="190" y="267" text-anchor="middle">帶外低</text>`;
  cells.forEach((row,r)=>row.forEach((name,c)=>{const x=78+c*224/3,y=58+r*60;svg+=`<rect x="${x}" y="${y}" width="${224/3}" height="60" class="zone-cell ${name==='正中'?'heart':''}"/><text x="${x+224/6}" y="${y+34}" text-anchor="middle">${name}</text>`}));
  host.innerHTML=svg+`<text x="78" y="304" class="definition-axis">← 內角</text><text x="302" y="304" text-anchor="end" class="definition-axis">外角 →</text></svg>`;
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
