#!/usr/bin/env python3
"""Single-file interactive trajectory dashboard, one HTML per activation bundle.

Pick a map, then a trajectory on that map, then inspect timestep-level quantities.

  * map canvas: terrain image + the selected trajectory path, start/end/current
    markers, points colourable by time / reward / V(h_t) / action.
  * plots (Plotly): V(h_t) vs return-to-go G_t · reward · action-prob stack ·
    policy entropy · hidden-state norm ||h_t|| and step-change ||h_t - h_{t-1}||.
  * hovering the map path OR any plot highlights the same timestep everywhere.

Quantities are taken from the bundle where stored (value=critic V, action_probs,
hidden state gru_h[PPO]/rssm_deter[Dreamer]); reward is reconstructed from the env
formula (slack -0.005 + 0.01·Δctg PBRS + 1.0 terminal on reach); G_t is the
discounted return-to-go with the agent's training gamma.

Output: outputs/report/trajectory_dashboard_<dataset>.html  (per dataset)
"""
from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mechinterp.analysis.bundle import ActivationBundle

GAMMA = {"bt_ppo": 0.997, "btc_ppo": 0.99, "bt_dreamer": 0.997, "btc_dreamer": 0.997}
SLACK = -0.005
PBRS = 0.01
GOAL_R = 1.0
ACTION_COLORS = ["#1f77b4", "#2ca02c", "#9467bd", "#17becf", "#ffd000", "#d62728"]


def hidden_source(b: ActivationBundle) -> str:
    return "gru_h" if "gru_h" in b.sources else "rssm_deter"


def map_png(b: ActivationBundle, map_id: int, cell: int = 10) -> str:
    rgb = b.render_map(map_id)
    big = np.kron(rgb, np.ones((cell, cell, 1), np.uint8))
    buf = io.BytesIO(); Image.fromarray(big).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def reward_and_return(ctg, reached, gamma):
    """Reconstruct per-step reward + discounted return-to-go for one trajectory."""
    n = len(ctg)
    r = np.full(n, SLACK, np.float64)
    dctg = np.zeros(n); dctg[1:] = ctg[:-1] - ctg[1:]
    r += PBRS * dctg
    if reached[-1]:
        r[-1] += GOAL_R
    G = np.zeros(n)
    acc = 0.0
    for i in range(n - 1, -1, -1):
        acc = r[i] + gamma * acc
        G[i] = acc
    return r, G


def build_traj(b, hsrc, g, sub):
    """sub: per-trajectory label DataFrame sorted by t. Returns the JS record."""
    ids = sub["row_id"].to_numpy()
    probs = b.load_extra("action_probs", ids).astype(np.float64)
    h = b.load_activations(hsrc, ids).astype(np.float64)
    ent = -(probs * np.log(np.clip(probs, 1e-9, 1))).sum(1)
    hn = np.linalg.norm(h, axis=1)
    hd = np.zeros(len(h)); hd[1:] = np.linalg.norm(np.diff(h, axis=0), axis=1)
    ctg = sub["ctg_to_goal"].to_numpy().astype(np.float64)
    reached = sub["reached"].to_numpy().astype(bool)
    rew, G = reward_and_return(ctg, reached, g)
    rnd = lambda a, k: [round(float(x), k) for x in a]
    rec = {
        "t": [int(x) for x in sub["t"].to_numpy()],
        "pr": [int(x) for x in sub["pos_r"].to_numpy()],
        "pc": [int(x) for x in sub["pos_c"].to_numpy()],
        "act": [int(x) for x in sub["action"].to_numpy()],
        "value": rnd(sub["value"].to_numpy(), 3),
        "reward": rnd(rew, 4), "G": rnd(G, 3), "ent": rnd(ent, 3),
        "hn": rnd(hn, 2), "hd": rnd(hd, 2),
        "probs": [rnd(probs[:, k], 3) for k in range(probs.shape[1])],
        "reached": bool(reached[-1]),
    }
    if "commit_state" in sub.columns:        # BTC: per-step commit phase (0/1/2)
        cmap = {"none": 0, "mine": 1, "build": 2}
        rec["phase"] = [cmap.get(str(x), 0) for x in sub["commit_state"].to_numpy()]
    SEG = {"free": 0, "approach": 1, "avoid": 2, "bridge": 3, "tunnel": 4}
    if "segment" in sub.columns:
        rec["seg"] = [SEG.get(str(x), 0) for x in sub["segment"].to_numpy()]
    return rec


def build_dataset(name, n_maps, n_traj):
    b = ActivationBundle(f"activation_datasets/{name}")
    hsrc = hidden_source(b)
    g = GAMMA.get(name, 0.997)
    lab = b.labels
    H, W = b.maps["terrain"].shape[1], b.maps["terrain"].shape[2]
    rng = np.random.default_rng(0)
    map_ids = sorted(lab["map_id"].unique())
    if n_maps:
        map_ids = list(rng.choice(map_ids, min(n_maps, len(map_ids)), replace=False))
        map_ids = sorted(int(m) for m in map_ids)
    DATA = {}
    anames = dict(lab[["action", "action_name"]].drop_duplicates().values)
    for mid in map_ids:
        ml = lab[lab["map_id"] == mid]
        tids = sorted(ml["traj_id"].unique())
        if n_traj and len(tids) > n_traj:
            tids = sorted(int(t) for t in rng.choice(tids, n_traj, replace=False))
        trajs = {}
        for tid in tids:
            sub = ml[ml["traj_id"] == tid].sort_values("t")
            if len(sub) < 3:
                continue
            trajs[str(int(tid))] = build_traj(b, hsrc, g, sub)
        if not trajs:
            continue
        sp = b.maps["spawn"][mid]; tg = b.maps["target"][mid]
        extra = {}
        if b.has_belief:
            extra["category"] = str(ml["category"].iloc[0])
        DATA[str(int(mid))] = {
            "img": map_png(b, mid), "H": H, "W": W,
            "spawn": [int(sp[0]), int(sp[1])], "target": [int(tg[0]), int(tg[1])],
            "trajs": trajs, **extra}
        print(f"  map {mid}: {len(trajs)} trajs", flush=True)
    meta = {"name": name, "hsrc": hsrc, "gamma": g,
            "action_names": [anames[k] for k in sorted(anames)],
            "is_commit": b.is_commit}
    return DATA, meta


def render_html(DATA, meta):
    return _TMPL.replace("__TITLE__", meta["name"]) \
                .replace("/*DATA*/null", json.dumps(DATA)) \
                .replace("/*META*/null", json.dumps(meta)) \
                .replace("/*ACOL*/null", json.dumps(ACTION_COLORS))


_TMPL = r"""<!doctype html><html><head><meta charset="utf-8">
<title>__TITLE__</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
 body{font-family:sans-serif;color:#223;margin:0;padding:16px;background:#fff}
 h1{color:#1b4f72;font-size:19px;margin:0 0 8px}
 .row{display:flex;gap:18px;flex-wrap:wrap;align-items:flex-start}
 .ctl{margin:6px 0}select{font-size:13px;padding:2px}
 label{font-size:13px;font-weight:600;margin-right:6px}
 #mapwrap{position:relative} canvas{border:1px solid #c6d2de;border-radius:6px;background:#eef3f8}
 .plots{flex:1;min-width:520px} .pl{width:100%;height:185px}
 .meta{font-size:12px;color:#456;margin:4px 0}
 #curinfo{font-size:12px;background:#eef3f8;border-radius:5px;padding:6px;margin-top:8px;min-height:34px}
</style></head><body>
<h1 id="dsname"></h1>
<div class="meta" id="dsmeta"></div>
<div class="row">
 <div>
  <div class="ctl"><label>map</label><select id="selMap"></select>
       <span id="mapinfo" class="meta"></span></div>
  <div class="ctl"><label>trajectory</label><select id="selTraj"></select>
       <span id="trajinfo" class="meta"></span></div>
  <div class="ctl"><label>colour by</label>
   <select id="selColor">
    <option value="action">action (move/place/mine)</option>
    <option value="time">time</option><option value="reward">reward (symlog)</option>
    <option value="value">V(h_t)</option>
    <option value="occupancy">occupancy time (cumulative)</option>
   </select>
   <span id="legend" class="meta" style="margin-left:8px"></span></div>
  <div id="mapwrap"><canvas id="map" width="640" height="320"></canvas></div>
  <div id="curinfo">hover the path or a plot &rarr;</div>
 </div>
 <div class="plots">
  <div id="pV" class="pl"></div><div id="pR" class="pl"></div>
  <div id="pA" class="pl"></div><div id="pE" class="pl"></div>
  <div id="pH" class="pl"></div>
 </div>
</div>
<script>
const DATA=/*DATA*/null, META=/*META*/null, ACOL=/*ACOL*/null;
const AN=META.action_names;
const $=id=>document.getElementById(id);
$("dsname").textContent=META.name;
$("dsmeta").innerHTML="hidden state ||h_t|| from <b>"+META.hsrc+"</b> &middot; "+
  "&gamma;="+META.gamma+" &middot; reward = slack(-0.005) + 0.01&middot;&Delta;ctg + terminal(+1 on reach)";

// ---- colormaps (compact LUTs via piecewise-linear control points) ----
function lerp(a,b,t){return a+(b-a)*t;}
function ramp(stops,t){t=Math.max(0,Math.min(1,t));const n=stops.length-1;
 let i=Math.min(n-1,Math.floor(t*n));const f=t*n-i,a=stops[i],b=stops[i+1];
 return"rgb("+Math.round(lerp(a[0],b[0],f))+","+Math.round(lerp(a[1],b[1],f))+","+Math.round(lerp(a[2],b[2],f))+")";}
const VIRIDIS=[[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]];
const RDBU=[[178,24,43],[239,138,98],[247,247,247],[103,169,207],[33,102,172]];
const HEAT=[[20,12,60],[110,30,110],[212,55,80],[248,150,40],[252,252,180]];
const viridis=t=>ramp(VIRIDIS,t);
const rdbu=t=>ramp(RDBU,1-t);  // high reward -> red end
const heat=t=>ramp(HEAT,t);
function rampA(stops,t,a){return ramp(stops,t).replace("rgb(","rgba(").replace(")",","+a+")");}
function norm(arr){const mn=Math.min(...arr),mx=Math.max(...arr);
 return arr.map(v=>mx>mn?(v-mn)/(mx-mn):0.5);}
const LIN=0.01;                       // symlog linear threshold for reward
function symlog(r){return Math.sign(r)*Math.log10(1+Math.abs(r)/LIN);}
// action rule: move(0-3)=blue, place(4)=yellow, mine(5)=red
const MPMA=a=>a===4?"rgba(235,190,0,.62)":a===5?"rgba(230,45,35,.62)":"rgba(40,90,230,.40)";
// cumulative cell occupancy across ALL trajectories of a map (log-scaled to [0,1])
function cumOcc(m){if(m._occ)return m._occ;const cnt={};
 for(const tr of Object.values(m.trajs))for(let i=0;i<tr.t.length;i++){
  const k=tr.pr[i]+","+tr.pc[i];cnt[k]=(cnt[k]||0)+1;}
 let mx=1;for(const k in cnt)if(cnt[k]>mx)mx=cnt[k];
 m._occ={cnt,lmx:Math.log(1+mx)};return m._occ;}
function occVal(m,r,c){const o=cumOcc(m);return Math.log(1+(o.cnt[r+","+c]||0))/o.lmx;}
// cached per-vertex jitter so overlapping paths separate -> density becomes visible
function jit(tr){if(tr._jx)return;tr._jx=[];tr._jy=[];const A=0.34;
 for(let i=0;i<tr.t.length;i++){tr._jx.push((Math.random()-0.5)*A);tr._jy.push((Math.random()-0.5)*A);}}

// ---- state ----
let curMap=null,curTraj=null,curT=0,overlayMode=false;

function fillMaps(){const s=$("selMap");s.innerHTML="";
 Object.keys(DATA).sort((a,b)=>a-b).forEach(m=>{const o=document.createElement("option");
  o.value=m;o.textContent="map "+m+(DATA[m].category?" ("+DATA[m].category+")":"");s.appendChild(o);});}
function fillTrajs(m){const s=$("selTraj");s.innerHTML="";
 const oa=document.createElement("option");oa.value="__ALL__";
 oa.textContent="▦ all trajectories (overlay grid)";s.appendChild(oa);
 Object.keys(DATA[m].trajs).sort((a,b)=>a-b).forEach(t=>{const tr=DATA[m].trajs[t];
  const o=document.createElement("option");o.value=t;
  o.textContent="traj "+t+" ("+tr.t.length+" steps, "+(tr.reached?"reached":"timeout")+")";
  s.appendChild(o);});}

function colorFor(tr,mode){
 if(mode==="time")return tr.t.map((_,i)=>viridis(i/Math.max(1,tr.t.length-1)));
 if(mode==="reward")return norm(tr.reward.map(symlog)).map(rdbu);
 if(mode==="value")return norm(tr.value).map(viridis);
 if(mode==="action")return tr.act.map(a=>ACOL[a]||"#888");
 if(mode==="occupancy"){const m=DATA[curMap];
  return tr.t.map((_,i)=>heat(occVal(m,tr.pr[i],tr.pc[i])));}
 return tr.t.map(()=>"#4c78a8");}

// ---- legend for the active colour rule ----
function sw(c){return "<span style='display:inline-block;width:11px;height:11px;background:"+
 c+";border-radius:2px;vertical-align:middle'></span>";}
function gbar(stops,rev){const cs=stops.map(s=>"rgb("+s[0]+","+s[1]+","+s[2]+")");
 return "<span style='display:inline-block;width:80px;height:11px;border-radius:3px;"+
  "vertical-align:middle;background:linear-gradient(90deg,"+(rev?cs.reverse():cs).join(",")+")'></span>";}
function updateLegend(){const mode=$("selColor").value;let h;
 if(mode==="action")h=overlayMode
   ?sw("#284fe6")+" move &nbsp;"+sw("#ffcd00")+" bridge (place) &nbsp;"+sw("#eb2319")+" tunnel (mine)"
   :AN.map((n,k)=>sw(ACOL[k])+" "+n).join(" &nbsp;");
 else if(mode==="time")h=gbar(VIRIDIS)+" <span style='font-size:10px'>early&rarr;late</span>";
 else if(mode==="reward")h=gbar(RDBU,true)+" <span style='font-size:10px'>low&rarr;high reward</span>";
 else if(mode==="value")h=gbar(VIRIDIS)+" <span style='font-size:10px'>low&rarr;high V</span>";
 else if(mode==="occupancy")h=gbar(HEAT)+" <span style='font-size:10px'>rare&rarr;frequent (all traj)</span>";
 $("legend").innerHTML="<b style='font-size:11px'>colour:</b> "+h;}

// ---- map canvas ----
const cv=$("map"),ctx=cv.getContext("2d");let mapImg=new Image(),cell=10;
let PX=[],PY=[];  // pixel coords of each timestep
function setCanvas(m){cell=Math.max(6,Math.floor(Math.min(900/m.W,360/m.H)));
 cv.width=m.W*cell;cv.height=m.H*cell;}
function withImg(cb){const m=DATA[curMap];
 if(mapImg.dataset.m!==curMap){mapImg=new Image();mapImg.dataset.m=curMap;
  mapImg.onload=cb;mapImg.src="data:image/png;base64,"+m.img;}else cb();}
function drawMap(){const m=DATA[curMap],tr=m.trajs[curTraj];setCanvas(m);
 withImg(()=>{ctx.clearRect(0,0,cv.width,cv.height);
  ctx.drawImage(mapImg,0,0,cv.width,cv.height);overlay(tr);});}
function drawOverlay(){const m=DATA[curMap];setCanvas(m);const mode=$("selColor").value;
 withImg(()=>{ctx.clearRect(0,0,cv.width,cv.height);
  ctx.drawImage(mapImg,0,0,cv.width,cv.height);
  ctx.fillStyle="rgba(10,12,30,0.22)";ctx.fillRect(0,0,cv.width,cv.height);
  ctx.lineCap="round";ctx.lineJoin="round";
  const T=Object.values(m.trajs);T.forEach(jit);
  if(mode==="action"){
   // pass 1: movement (blue, thin); pass 2: bridge(yellow)/tunnel(red) bright + thick, on top
   ctx.lineWidth=0.9;ctx.strokeStyle="rgba(40,90,230,0.30)";
   for(const tr of T)for(let i=0;i+1<tr.t.length;i++){
    const sg=tr.seg?tr.seg[i]:0;if(sg===3||sg===4)continue;
    const p=jcc(tr,i),q=jcc(tr,i+1);ctx.beginPath();ctx.moveTo(p[0],p[1]);ctx.lineTo(q[0],q[1]);ctx.stroke();}
   ctx.lineWidth=1.9;
   for(const tr of T)for(let i=0;i+1<tr.t.length;i++){
    const sg=tr.seg?tr.seg[i]:0;if(sg!==3&&sg!==4)continue;
    const p=jcc(tr,i),q=jcc(tr,i+1);
    ctx.strokeStyle=sg===3?"rgba(255,205,0,0.92)":"rgba(235,35,25,0.92)";
    ctx.beginPath();ctx.moveTo(p[0],p[1]);ctx.lineTo(q[0],q[1]);ctx.stroke();}
  } else {
   ctx.lineWidth=1.0;
   for(const tr of T){
    let seg=null;
    if(mode==="time")seg=tr.t.map((_,i)=>rampA(VIRIDIS,i/Math.max(1,tr.t.length-1),.5));
    else if(mode==="reward"){const s=norm(tr.reward.map(symlog));seg=s.map(v=>rampA(RDBU,1-v,.5));}
    else if(mode==="value"){const s=norm(tr.value);seg=s.map(v=>rampA(VIRIDIS,v,.5));}
    else if(mode==="occupancy")seg=tr.t.map((_,i)=>rampA(HEAT,occVal(m,tr.pr[i],tr.pc[i]),.6));
    for(let i=0;i+1<tr.t.length;i++){const p=jcc(tr,i),q=jcc(tr,i+1);
     ctx.strokeStyle=seg?seg[i]:"rgba(70,110,200,.4)";
     ctx.beginPath();ctx.moveTo(p[0],p[1]);ctx.lineTo(q[0],q[1]);ctx.stroke();}}
  }
  const tp=cc(m.target[0],m.target[1]);ctx.strokeStyle="#fff";ctx.lineWidth=2;
  ctx.strokeRect(tp[0]-cell/2,tp[1]-cell/2,cell,cell);});}
function cc(r,c){return [(c+0.5)*cell,(r+0.5)*cell];}
function jcc(tr,i){return [(tr.pc[i]+0.5+tr._jx[i])*cell,(tr.pr[i]+0.5+tr._jy[i])*cell];}
function overlay(tr){const cols=colorFor(tr,$("selColor").value);PX=[];PY=[];
 ctx.lineWidth=2;ctx.strokeStyle="rgba(20,20,20,.55)";ctx.beginPath();
 for(let i=0;i<tr.t.length;i++){const [x,y]=cc(tr.pr[i],tr.pc[i]);PX.push(x);PY.push(y);
  i?ctx.lineTo(x,y):ctx.moveTo(x,y);}ctx.stroke();
 for(let i=0;i<tr.t.length;i++){ctx.beginPath();ctx.arc(PX[i],PY[i],3,0,7);
  ctx.fillStyle=cols[i];ctx.fill();}
 mark(PX[0],PY[0],"#10b010");mark(PX[PX.length-1],PY[PY.length-1],"#d01010");
 const m=DATA[curMap];const [tx,ty]=cc(m.target[0],m.target[1]);
 ctx.strokeStyle="#000";ctx.lineWidth=2;ctx.strokeRect(tx-cell/2,ty-cell/2,cell,cell);
 drawCur();}
function mark(x,y,c){ctx.beginPath();ctx.arc(x,y,6,0,7);ctx.lineWidth=2.5;
 ctx.strokeStyle=c;ctx.stroke();}
let curDot=null;
function drawCur(){if(curDot)return;}
function redrawCur(){drawMap();const x=PX[curT],y=PY[curT];if(x==null)return;
 ctx.beginPath();ctx.arc(x,y,7,0,7);ctx.fillStyle="#fff";ctx.strokeStyle="#000";
 ctx.lineWidth=2;ctx.fill();ctx.stroke();}

// ---- plots ----
function lay(title){return{margin:{l:46,r:8,t:22,b:22},title:{text:title,font:{size:12}},
 height:185,paper_bgcolor:"#fff",plot_bgcolor:"#eef3f8",showlegend:false,
 xaxis:{title:"",gridcolor:"#fff"},yaxis:{gridcolor:"#fff"},
 shapes:[{type:"line",x0:0,x1:0,y0:0,y1:1,yref:"paper",line:{color:"#e6005c",width:1.5}}]};}
function vline(t){return[{type:"line",x0:t,x1:t,y0:0,y1:1,yref:"paper",line:{color:"#e6005c",width:1.5}}];}
function drawPlots(){const tr=DATA[curMap].trajs[curTraj],T=tr.t;
 const hov={hovermode:"x unified"};
 Plotly.react("pV",[
   {x:T,y:tr.value,name:"V(h_t)",line:{color:"#1b4f72"}},
   {x:T,y:tr.G,name:"G_t",line:{color:"#e6a800",dash:"dot"}}],
   Object.assign(lay("V(h_t) vs return-to-go G_t"),{showlegend:true,
     legend:{orientation:"h",y:1.25,font:{size:10}}}),{displayModeBar:false});
 const refs=[-0.05,-0.01,0,0.01,0.1,0.5,1];
 Plotly.react("pR",[{x:T,y:tr.reward.map(symlog),customdata:tr.reward,name:"reward",
   line:{color:"#2ca02c"},fill:"tozeroy",
   hovertemplate:"t=%{x}<br>reward=%{customdata:.4f}<extra></extra>"}],
   Object.assign(lay("reward (symlog)"),{yaxis:{tickvals:refs.map(symlog),
     ticktext:refs.map(String),gridcolor:"#fff",zeroline:true,zerolinecolor:"#bbb"}}),
   {displayModeBar:false});
 const ap=tr.probs.map((p,k)=>({x:T,y:p,name:AN[k],stackgroup:"a",
   line:{width:0},fillcolor:ACOL[k]}));
 Plotly.react("pA",ap,Object.assign(lay("action probabilities (stacked)"),
   {showlegend:true,legend:{orientation:"h",y:1.3,font:{size:9}},yaxis:{range:[0,1],gridcolor:"#fff"}}),
   {displayModeBar:false});
 Plotly.react("pE",[{x:T,y:tr.ent,name:"entropy",line:{color:"#9467bd"}}],
   lay("policy entropy"),{displayModeBar:false});
 Plotly.react("pH",[
   {x:T,y:tr.hn,name:"||h_t||",line:{color:"#1f77b4"}},
   {x:T,y:tr.hd,name:"||h_t-h_{t-1}||",line:{color:"#d62728"},yaxis:"y2"}],
   Object.assign(lay("hidden-state norm & change ("+META.hsrc+")"),{showlegend:true,
     legend:{orientation:"h",y:1.25,font:{size:10}},
     yaxis2:{overlaying:"y",side:"right",gridcolor:"transparent"}}),{displayModeBar:false});
 ["pV","pR","pA","pE","pH"].forEach(id=>{$(id).on("plotly_hover",e=>{
   const xt=e.points[0].x;const i=T.indexOf(xt);if(i>=0)setT(i);});});
}
function setShapes(t){["pV","pR","pA","pE","pH"].forEach(id=>
  Plotly.relayout(id,{shapes:vline(t)}));}

function setT(i){curT=i;redrawCur();setShapes(DATA[curMap].trajs[curTraj].t[i]);
 const tr=DATA[curMap].trajs[curTraj];
 $("curinfo").innerHTML="<b>t="+tr.t[i]+"</b> &middot; pos=("+tr.pr[i]+","+tr.pc[i]+
  ") &middot; action=<b>"+AN[tr.act[i]]+"</b> &middot; V="+tr.value[i].toFixed(3)+
  " &middot; G="+tr.G[i].toFixed(3)+" &middot; reward="+tr.reward[i].toFixed(4)+
  " &middot; entropy="+tr.ent[i].toFixed(3)+" &middot; ||h||="+tr.hn[i].toFixed(2)+
  " &middot; ||&Delta;h||="+tr.hd[i].toFixed(2);}

function clearPlots(){["pV","pR","pA","pE","pH"].forEach(id=>Plotly.react(id,[],
 Object.assign(lay(""),{annotations:[{text:"overlay mode — pick a single trajectory",
  showarrow:false,xref:"paper",yref:"paper",x:0.5,y:0.5,font:{color:"#789",size:12}}],
  shapes:[]})));}

// map hover -> nearest timestep
cv.addEventListener("mousemove",e=>{if(overlayMode)return;
 const r=cv.getBoundingClientRect();
 const mx=e.clientX-r.left,my=e.clientY-r.top;let best=0,bd=1e9;
 for(let i=0;i<PX.length;i++){const dx=PX[i]-mx,dy=PY[i]-my,d=dx*dx+dy*dy;
  if(d<bd){bd=d;best=i;}}if(bd<400)setT(best);});

function selectMap(m){curMap=m;fillTrajs(m);curTraj=$("selTraj").value;
 const dm=DATA[m];$("mapinfo").textContent="spawn("+dm.spawn+") target("+dm.target+")"+
  (dm.category?" "+dm.category:"");selectTraj(curTraj);}
function selectTraj(t){curTraj=t;
 if(t==="__ALL__"){overlayMode=true;
  if($("selColor").value==="time")$("selColor").value="action";  // default rule
  $("trajinfo").textContent=Object.keys(DATA[curMap].trajs).length+" trajectories overlaid";
  drawOverlay();updateLegend();clearPlots();
  $("curinfo").innerHTML="<b>overlay grid</b> &middot; "+
   Object.keys(DATA[curMap].trajs).length+" trajectories &middot; lines jittered to expose "+
   "density &middot; colour rule from the dropdown (see legend)";return;}
 overlayMode=false;const tr=DATA[curMap].trajs[t];
 $("trajinfo").textContent=tr.t.length+" steps, "+(tr.reached?"reached":"timeout");
 drawMap();updateLegend();drawPlots();setT(0);}

$("selMap").addEventListener("change",e=>selectMap(e.target.value));
$("selTraj").addEventListener("change",e=>selectTraj(e.target.value));
$("selColor").addEventListener("change",()=>{updateLegend();
 if(overlayMode){drawOverlay();return;}drawMap();setT(curT);});
fillMaps();selectMap($("selMap").value);
</script></body></html>"""


def main():
    args = sys.argv[1:]
    names = [a for a in args if not a.isdigit()]
    nums = [int(a) for a in args if a.isdigit()]
    n_maps = nums[0] if len(nums) > 0 else 0   # 0 = all maps
    n_traj = nums[1] if len(nums) > 1 else 0   # 0 = all trajectories
    if not names:
        names = ["bt_ppo", "btc_ppo", "bt_dreamer", "btc_dreamer"]
    out_dir = Path("outputs/report"); out_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        print(f"[{name}] building dashboard (maps={n_maps}, traj/map={n_traj}) ...",
              flush=True)
        DATA, meta = build_dataset(name, n_maps, n_traj)
        html = render_html(DATA, meta)
        out = out_dir / f"trajectory_dashboard_{name}.html"
        out.write_text(html)
        print(f"  wrote {out} ({out.stat().st_size/1e6:.1f} MB, {len(DATA)} maps)")


if __name__ == "__main__":
    main()
