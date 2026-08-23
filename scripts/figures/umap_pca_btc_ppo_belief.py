#!/usr/bin/env python3
"""UMAP + Global PCA + per-map PCA of btc_ppo_belief gru_h activations.

Produces outputs/report/umap_pca_btc_ppo_belief/report.html:

  §1  Global UMAP 3-D (40k frames)
  §2  Global PCA 3-D (same 40k frames, PCA fit on gru_h)
  §3  Per-map views — map selector (10 maps) drives two live figures:
        · Per-map UMAP  : selected map's frames in the global UMAP space
        · Per-map PCA   : fresh PCA of OOB-free frames for that map

  Controls (global, apply to all 4 plots):
    · Colour-by selector: category / skill committed / segment / value / map belief
    · Episode % range slider: dual-handle filter by position in episode (0–100%)

  Hover on any point → egocentric obs + map + trajectory up to that step.
  All images are rendered client-side from gzip-compressed tile blobs.

Selected maps (10, all 100% success, balanced across categories by n_clean OOB-free frames):
  balanced: 8,11,16,24  lakes: 32,38,40  rocky: 68,76,79

Belief colour mode: 3-way probability mix of the model's auxiliary belief head output.
  white = 100% balanced  blue = 100% lakes  red = 100% rocky
  mixed colours reflect probability distributions (class order: 0=balanced,1=lakes,2=rocky).
"""
from __future__ import annotations
import base64
import gzip
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
import umap

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.mechinterp.analysis.bundle import ActivationBundle

DATASET      = "activation_datasets/btc_ppo_belief"
BELIEF_CKPT  = "released_models/bridge_tunnel_commit/ppo_gru_commit_aux_belief.pt"
SRC          = "gru_h"
UMAP_N       = 40_000
UMAP_SEED    = 42
OOB_TILE     = 5
VIEW         = 21
MAP_H, MAP_W = 32, 64
OUT          = Path("outputs/report/umap_pca_btc_ppo_belief")

# 10 maps: 100% success, balanced across categories, chosen by n_clean OOB-free frames
# balanced: 8,11,16,24  lakes: 32,38,40  rocky: 68,76,79  (all ≥3.7k clean frames)
SELECTED_MAPS = [8, 11, 16, 24, 32, 38, 40, 68, 76, 79]

CAT_COLORS   = {"balanced": "#2ecc71", "lakes": "#3498db", "rocky": "#e74c3c"}
SKILL_COLORS = {"none": "#95a5a6", "build": "#f39c12", "mine": "#9b59b6"}
SEG_COLORS   = {
    "avoid":   "#1f5fd0", "bridge": "#e6a800", "tunnel": "#a800e6",
    "none":    "#95a5a6", "approach": "#27ae60", "free":  "#888888",
}

HOVER_TMPL = "<extra></extra>"


# ------------------------------------------------------------------ data helpers
def gz_b64(arr: np.ndarray) -> str:
    return base64.b64encode(gzip.compress(arr.astype(np.uint8).flatten().tobytes(), 6)).decode()


def gz_b64_text(text: str) -> str:
    return base64.b64encode(gzip.compress(text.encode("utf-8"), 6)).decode()


# ------------------------------------------------------------------ belief head
def load_belief_head(ckpt_path: str):
    """Return (W, bias) numpy arrays for the belief linear head."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["policy"]
    return sd["belief.weight"].numpy(), sd["belief.bias"].numpy()


def compute_belief_probs(gru_h: np.ndarray, W: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """(N,128) gru_h → (N,3) float32 softmax probs [P(balanced),P(lakes),P(rocky)]."""
    logits = gru_h @ W.T + bias               # (N, 3)
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)


def belief_to_hex(probs: np.ndarray) -> list[str]:
    """(N,3) → list of '#rrggbb'. white=balanced, blue=lakes, red=rocky.
    Colour = P(bal)·white + P(lakes)·blue + P(rocky)·red."""
    pb, pl, pr = probs[:, 0], probs[:, 1], probs[:, 2]
    R = np.clip((pb + pr) * 255, 0, 255).astype(np.uint8)
    G = np.clip(pb * 255,        0, 255).astype(np.uint8)
    B = np.clip((pb + pl) * 255, 0, 255).astype(np.uint8)
    return [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in zip(R, G, B)]


def load_minimaps(b: ActivationBundle, row_ids: np.ndarray) -> np.ndarray:
    with h5py.File(b.path / "activations.h5", "r") as f:
        return f["minimap"][row_ids].astype(np.uint8)


def compute_ep_pct(lab: pd.DataFrame, max_t_of: dict) -> np.ndarray:
    """Episode completion fraction [0,1]: t / max_t_in_that_trajectory."""
    keys = list(zip(lab["map_id"].astype(int), lab["traj_id"].astype(int)))
    mt = np.array([max_t_of.get(k, 1) for k in keys], dtype=float)
    return (lab["t"].to_numpy(float) / np.maximum(mt, 1.0)).clip(0.0, 1.0)


def pack_trajectories(b: ActivationBundle) -> str:
    lab = b.labels.sort_values(["map_id", "traj_id", "t"]).reset_index(drop=True)
    dt = np.dtype([("map_id", "u1"), ("traj_id", "u1"),
                   ("t", "<u2"), ("pos_r", "u1"), ("pos_c", "u1")])
    arr = np.empty(len(lab), dtype=dt)
    arr["map_id"]  = lab["map_id"].to_numpy(np.uint8)
    arr["traj_id"] = lab["traj_id"].to_numpy(np.uint8)
    arr["t"]       = lab["t"].to_numpy(np.uint16)
    arr["pos_r"]   = lab["pos_r"].to_numpy(np.uint8)
    arr["pos_c"]   = lab["pos_c"].to_numpy(np.uint8)
    return base64.b64encode(gzip.compress(arr.tobytes(), 6)).decode()


# ------------------------------------------------------------------ UMAP
def load_umap_sample(b: ActivationBundle, n: int, seed: int):
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(b.labels), min(n, len(b.labels)), replace=False))
    lab = b.labels.iloc[idx].reset_index(drop=True)
    row_ids = lab["row_id"].to_numpy()
    X  = b.load_activations(SRC, row_ids)
    mm = load_minimaps(b, row_ids)
    return X, mm, lab


def fit_umap(X: np.ndarray) -> np.ndarray:
    sc = StandardScaler()
    reducer = umap.UMAP(n_components=3, n_neighbors=30, min_dist=0.1,
                        random_state=UMAP_SEED, verbose=False)
    return reducer.fit_transform(sc.fit_transform(X))


def fit_global_pca(X: np.ndarray):
    """PCA on X → (n,3) embedding + explained variance ratios (%)."""
    sc  = StandardScaler()
    pca = PCA(n_components=3, random_state=0)
    emb = pca.fit_transform(sc.fit_transform(X))
    return emb, pca.explained_variance_ratio_ * 100


# ------------------------------------------------------------------ PCA per map
def build_pca(b: ActivationBundle, map_id: int):
    lab_map = b.labels[b.labels["map_id"] == map_id].reset_index(drop=True)
    row_ids = lab_map["row_id"].to_numpy()
    with h5py.File(b.path / "activations.h5", "r") as f:
        mm_all = f["minimap"][row_ids].astype(np.uint8)
    mask   = ~(mm_all == OOB_TILE).any(axis=(1, 2))
    lab_c  = lab_map[mask].reset_index(drop=True)
    mm_c   = mm_all[mask]
    X_raw  = b.load_activations(SRC, lab_c["row_id"].to_numpy())
    sc     = StandardScaler()
    pca    = PCA(n_components=3, random_state=0)
    emb    = pca.fit_transform(sc.fit_transform(X_raw))
    evr    = pca.explained_variance_ratio_ * 100
    stats  = dict(n_total=len(lab_map), n_kept=int(mask.sum()),
                  n_oob=int((~mask).sum()), evr=evr.tolist())
    return emb, lab_c, mm_c, stats


# ------------------------------------------------------------------ customdata
def make_cd(lab: pd.DataFrame, mm_idx: list | np.ndarray,
            belief_probs: np.ndarray | None = None) -> list:
    """Per-point customdata: [mm_idx,map_id,traj_id,t,pos_r,pos_c,
                               cat,fc,seg,val, b0,b1,b2]
    b0=P(balanced), b1=P(lakes), b2=P(rocky); None stored when unavailable."""
    cat = (lab["category"].to_numpy(str)
           if "category" in lab.columns else np.full(len(lab), ""))
    fc  = (lab["final_commit"].to_numpy(str)
           if "final_commit" in lab.columns else np.full(len(lab), ""))
    seg = (lab["segment"].to_numpy(str)
           if "segment" in lab.columns else np.full(len(lab), ""))
    val = lab["value"].round(3).astype(str).to_numpy()
    rows = []
    for i, (g, m, tr, t, r, c, ca, f, s, v) in enumerate(zip(
            mm_idx,
            lab["map_id"], lab["traj_id"], lab["t"],
            lab["pos_r"],  lab["pos_c"],
            cat, fc, seg, val)):
        bp = [round(float(belief_probs[i, k]), 4) for k in range(3)] \
             if belief_probs is not None else [None, None, None]
        rows.append([int(g), int(m), int(tr), int(t), int(r), int(c),
                     str(ca), str(f), str(s), str(v), bp[0], bp[1], bp[2]])
    return rows




# ------------------------------------------------------------------ JS / HTML
_JS_VARS = """\
<script>
const PALETTE = {palette};
const MMDATA_UMAP_B64 = "{mm_umap}";
const MMDATA_PCA_B64  = "{mm_pca}";
const TRAJ_B64        = "{traj}";
const TERRAIN_B64     = "{terrain}";
const ALL_DATA_B64    = "{all_data}";
const VIEW={view}, MAP_H={map_h}, MAP_W={map_w};
</script>
"""

_JS_LOGIC = r"""
<script>
// ========== global state ==========
let G_MM_UMAP=null, G_MM_PCA=null, G_TERRAIN=null;
let G_TRAJ_DATA=null, G_TRAJ_IDX=null;
let _hoverReady=false, _hoverIniting=false;
let ALL_DATA=null;
let currentMap=null, currentColorMode='category';
let G_EP_MIN=0, G_EP_MAX=100;
let _epTimer=null;
let _mouseX=0, _mouseY=0;
document.addEventListener('mousemove', e => { _mouseX=e.clientX; _mouseY=e.clientY; });

// ========== colour maps ==========
const CAT_C  = {balanced:'#2ecc71', lakes:'#3498db', rocky:'#e74c3c'};
const FC_C   = {none:'#95a5a6', build:'#f39c12', mine:'#9b59b6'};
const SEG_C  = {avoid:'#1f5fd0', bridge:'#e6a800', tunnel:'#a800e6',
                none:'#95a5a6', approach:'#27ae60', free:'#888888'};

// ========== decompress ==========
async function decompressB64Gzip(b64) {
    const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
    const ds = new DecompressionStream('gzip');
    const writer = ds.writable.getWriter();
    writer.write(bytes); writer.close();
    const reader = ds.readable.getReader();
    const chunks = [];
    while (true) {
        const {done, value} = await reader.read();
        if (done) break;
        chunks.push(value);
    }
    const total = chunks.reduce((s,c) => s+c.length, 0);
    const out = new Uint8Array(total);
    let off = 0;
    for (const c of chunks) { out.set(c, off); off += c.length; }
    return out;
}
async function decompressB64GzipText(b64) {
    return new TextDecoder().decode(await decompressB64Gzip(b64));
}

// ========== init ==========
async function init() {
    ALL_DATA = JSON.parse(await decompressB64GzipText(ALL_DATA_B64));
    buildGlobalPlots();
    updatePerMap(ALL_DATA.maps[0]);
    loadHoverData();
}

async function loadHoverData() {
    if (_hoverIniting || _hoverReady) return;
    _hoverIniting = true;
    [G_MM_UMAP, G_MM_PCA, G_TRAJ_DATA] = await Promise.all([
        decompressB64Gzip(MMDATA_UMAP_B64),
        decompressB64Gzip(MMDATA_PCA_B64),
        decompressB64Gzip(TRAJ_B64),
    ]);
    G_TERRAIN = Uint8Array.from(atob(TERRAIN_B64), c => c.charCodeAt(0));
    G_TRAJ_IDX = {};
    const n = G_TRAJ_DATA.length / 6;
    let i = 0;
    while (i < n) {
        const mapId = G_TRAJ_DATA[i*6], trajId = G_TRAJ_DATA[i*6+1];
        const key = `${mapId}_${trajId}`;
        let j = i;
        while (j < n && G_TRAJ_DATA[j*6]===mapId && G_TRAJ_DATA[j*6+1]===trajId) j++;
        G_TRAJ_IDX[key] = {start:i, count:j-i};
        i = j;
    }
    _hoverReady = true; _hoverIniting = false;
    document.getElementById('hover-panel').innerHTML = `
      <div style="display:flex;gap:10px;align-items:flex-start">
        <div><div class="hp-label">Observation (egocentric)</div><canvas id="mm-canvas"></canvas></div>
        <div><div class="hp-label">Map + trajectory</div><canvas id="map-canvas"></canvas></div>
      </div>
      <div id="hover-info" class="hp-info"></div>`;
    document.getElementById('hover-panel').style.display = 'none';
}

// ========== colour helpers ==========
function beliefHex(pb, pl, pr) {
    return `rgb(${Math.round((pb+pr)*255)},${Math.round(pb*255)},${Math.round((pb+pl)*255)})`;
}

// ========== episode % filter ==========
function onEpSlider(which) {
    let lo = parseInt(document.getElementById('ep-min').value);
    let hi = parseInt(document.getElementById('ep-max').value);
    if (lo > hi) {
        if (which==='min') { lo=hi; document.getElementById('ep-min').value=lo; }
        else               { hi=lo; document.getElementById('ep-max').value=hi; }
    }
    G_EP_MIN=lo; G_EP_MAX=hi;
    document.getElementById('ep-range-label').textContent = `${lo}% → ${hi}%`;
    const fill = document.getElementById('ep-fill');
    if (fill) { fill.style.left=lo+'%'; fill.style.width=(hi-lo)+'%'; }
    clearTimeout(_epTimer);
    _epTimer = setTimeout(updateAllPlots, 80);
}

// ========== buildTraces (ep_pct-filtered, shared by all 4 plots) ==========
function buildTraces(data, mode, markerSize) {
    const sz = markerSize || 3;
    const n = data.x.length;
    const epLo = G_EP_MIN/100, epHi = G_EP_MAX/100;

    // collect kept indices after ep_pct filter
    const kept = [];
    for (let i=0; i<n; i++) {
        const ep = data.ep_pct ? data.ep_pct[i] : 0.5;
        if (ep >= epLo && ep <= epHi) kept.push(i);
    }
    if (kept.length === 0) return [];

    // build filtered arrays
    const xs=[], ys=[], zs=[], allCd=[];
    for (const i of kept) {
        xs.push(data.x[i]); ys.push(data.y[i]); zs.push(data.z[i]);
        allCd.push([
            data.mm_idx[i], data.map_id[i], data.traj_id[i], data.t[i],
            data.pos_r[i],  data.pos_c[i],
            data.cat[i],    data.fc[i],     data.seg[i],     String(data.val[i]),
            data.b0 ? Number(data.b0[i]) : null,
            data.b1 ? Number(data.b1[i]) : null,
            data.b2 ? Number(data.b2[i]) : null,
        ]);
    }

    // belief mode
    if (mode === 'belief') {
        if (!data.b0) return [{type:'scatter3d',mode:'markers',x:[],y:[],z:[],
                               marker:{size:sz},showlegend:false,name:'belief (unavailable)'}];
        return [{
            type:'scatter3d', mode:'markers', x:xs, y:ys, z:zs,
            marker:{size:sz, color:kept.map(i=>beliefHex(data.b0[i],data.b1[i],data.b2[i])), opacity:0.8},
            customdata:allCd, hovertemplate:'<extra></extra>', showlegend:false, name:'belief',
        }];
    }

    // value mode
    if (mode === 'value') {
        const vals = kept.map(i => Number(data.val[i]));
        const sorted = vals.slice().sort((a,b)=>a-b);
        const p2=sorted[Math.floor(vals.length*0.02)], p98=sorted[Math.floor(vals.length*0.98)];
        return [{
            type:'scatter3d', mode:'markers', x:xs, y:ys, z:zs,
            marker:{size:sz, color:vals, colorscale:'RdYlGn', cmin:p2, cmax:p98, opacity:0.65,
                    colorbar:{title:'value', thickness:12, len:0.5}},
            customdata:allCd, hovertemplate:'<extra></extra>', showlegend:false,
        }];
    }

    // categorical modes
    const cmap = {category:CAT_C, final_commit:FC_C, segment:SEG_C}[mode] || CAT_C;
    const labelArr = mode==='category'?data.cat : mode==='final_commit'?data.fc : data.seg;
    const groups = {};
    for (let ki=0; ki<kept.length; ki++) {
        const i=kept[ki], lv=labelArr[i];
        if (!groups[lv]) groups[lv]={x:[],y:[],z:[],cd:[]};
        groups[lv].x.push(xs[ki]); groups[lv].y.push(ys[ki]); groups[lv].z.push(zs[ki]);
        groups[lv].cd.push(allCd[ki]);
    }
    return Object.entries(groups).sort(([a],[b])=>a.localeCompare(b)).map(([lv,g])=>({
        type:'scatter3d', mode:'markers', x:g.x, y:g.y, z:g.z,
        marker:{size:sz, color:cmap[lv]||'#ccc', opacity:0.65},
        name:lv, legendgroup:lv, showlegend:true,
        customdata:g.cd, hovertemplate:'<extra></extra>',
    }));
}

// ========== layouts ==========
function globalUmapLayout() {
    return {
        title:'Global UMAP 3D — gru_h',
        scene:{xaxis:{title:'UMAP-1'}, yaxis:{title:'UMAP-2'}, zaxis:{title:'UMAP-3'}},
        legend:{itemsizing:'constant', font:{size:11}},
        margin:{l:0,r:0,b:40,t:60}, height:700,
    };
}
function globalPcaLayout() {
    const evr = ALL_DATA.global_pca.evr;
    return {
        title:`Global PCA 3D — gru_h  ·  PC1 ${evr[0].toFixed(1)}%  PC2 ${evr[1].toFixed(1)}%  PC3 ${evr[2].toFixed(1)}%`,
        scene:{
            xaxis:{title:`PC1 (${evr[0].toFixed(1)}%)`},
            yaxis:{title:`PC2 (${evr[1].toFixed(1)}%)`},
            zaxis:{title:`PC3 (${evr[2].toFixed(1)}%)`},
        },
        legend:{itemsizing:'constant', font:{size:11}},
        margin:{l:0,r:0,b:40,t:60}, height:700,
    };
}
function pcaLayout(mapId, info) {
    const evr = info.evr;
    return {
        title:`PCA — map ${mapId} (${info.category}) — ${info.n_kept.toLocaleString()} OOB-free frames<br>` +
              `PC1 ${evr[0].toFixed(1)}%  PC2 ${evr[1].toFixed(1)}%  PC3 ${evr[2].toFixed(1)}%`,
        scene:{
            xaxis:{title:`PC1 (${evr[0].toFixed(1)}%)`},
            yaxis:{title:`PC2 (${evr[1].toFixed(1)}%)`},
            zaxis:{title:`PC3 (${evr[2].toFixed(1)}%)`},
        },
        legend:{itemsizing:'constant', font:{size:11}},
        margin:{l:0,r:0,b:40,t:80}, height:620,
    };
}
function umapMapLayout(mapId, info) {
    const rng = ALL_DATA.umap_range;
    return {
        title:`Per-map UMAP — map ${mapId} (${info.category}) — ${info.n_umap.toLocaleString()} frames in global space`,
        scene:{
            xaxis:{title:'UMAP-1', range:rng.x},
            yaxis:{title:'UMAP-2', range:rng.y},
            zaxis:{title:'UMAP-3', range:rng.z},
        },
        legend:{itemsizing:'constant', font:{size:11}},
        margin:{l:0,r:0,b:40,t:60}, height:620,
    };
}

// ========== plot builders ==========
function buildGlobalPlots() {
    if (!ALL_DATA) return;
    Plotly.react('plot-umap',       buildTraces(ALL_DATA.global_umap, currentColorMode, 2.5), globalUmapLayout());
    Plotly.react('plot-global-pca', buildTraces(ALL_DATA.global_pca,  currentColorMode, 2.5), globalPcaLayout());
}

function updatePerMap(mapId) {
    if (!ALL_DATA) return;
    currentMap = mapId;
    const info  = ALL_DATA.info[mapId];
    Plotly.react('plot-pca',      buildTraces(ALL_DATA.pca[mapId],  currentColorMode, 3), pcaLayout(mapId, info));
    Plotly.react('plot-umap-map', buildTraces(ALL_DATA.umap[mapId], currentColorMode, 4), umapMapLayout(mapId, info));
    document.querySelectorAll('.map-btn').forEach(b => b.classList.remove('active'));
    const btn = document.getElementById(`btn-${mapId}`);
    if (btn) btn.classList.add('active');
    attachPerMapHover();
}

function updateAllPlots() {
    buildGlobalPlots();
    if (currentMap !== null) updatePerMap(currentMap);
}

function setColorMode(mode) {
    currentColorMode = mode;
    document.querySelectorAll('.color-btn').forEach(b => b.classList.remove('active'));
    const btn = document.getElementById(`cbtn-${mode}`);
    if (btn) btn.classList.add('active');
    const leg = document.getElementById('belief-legend');
    if (leg) leg.style.display = (mode==='belief') ? 'block' : 'none';
    updateAllPlots();
}

// ========== canvas rendering ==========
function renderMinimap(mmIdx, mmArr, canvas) {
    const S=7;
    canvas.width=VIEW*S; canvas.height=VIEW*S;
    const ctx=canvas.getContext('2d'), img=ctx.createImageData(VIEW*S,VIEW*S);
    const off=mmIdx*VIEW*VIEW;
    for (let r=0;r<VIEW;r++) for (let c=0;c<VIEW;c++) {
        const tile=mmArr[off+r*VIEW+c]&0xFF, [R,G,B]=tile<PALETTE.length?PALETTE[tile]:[30,30,30];
        for (let dr=0;dr<S;dr++) for (let dc=0;dc<S;dc++) {
            const px=((r*S+dr)*VIEW*S+c*S+dc)*4;
            img.data[px]=R; img.data[px+1]=G; img.data[px+2]=B; img.data[px+3]=255;
        }
    }
    ctx.putImageData(img,0,0);
    const cr=Math.floor(VIEW/2), cc=Math.floor(VIEW/2);
    ctx.strokeStyle='rgba(255,255,255,0.85)'; ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(cc*S,cr*S+S/2); ctx.lineTo((cc+1)*S,cr*S+S/2); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(cc*S+S/2,cr*S); ctx.lineTo(cc*S+S/2,(cr+1)*S); ctx.stroke();
}

function renderMap(mapId, trajId, curT, posR, posC, canvas) {
    const S=5;
    canvas.width=MAP_W*S; canvas.height=MAP_H*S;
    const ctx=canvas.getContext('2d'), img=ctx.createImageData(MAP_W*S,MAP_H*S);
    const off=mapId*MAP_H*MAP_W;
    for (let r=0;r<MAP_H;r++) for (let c=0;c<MAP_W;c++) {
        const tile=G_TERRAIN[off+r*MAP_W+c]&0xFF, [R,G,B]=tile<PALETTE.length?PALETTE[tile]:[30,30,30];
        for (let dr=0;dr<S;dr++) for (let dc=0;dc<S;dc++) {
            const px=((r*S+dr)*MAP_W*S+c*S+dc)*4;
            img.data[px]=R; img.data[px+1]=G; img.data[px+2]=B; img.data[px+3]=255;
        }
    }
    ctx.putImageData(img,0,0);
    const entry=G_TRAJ_IDX&&G_TRAJ_IDX[`${mapId}_${trajId}`];
    if (entry) {
        const dv=new DataView(G_TRAJ_DATA.buffer,G_TRAJ_DATA.byteOffset);
        ctx.shadowColor='#000'; ctx.shadowBlur=2;
        ctx.strokeStyle='rgba(255,255,255,0.85)'; ctx.lineWidth=1.5;
        ctx.beginPath(); let started=false;
        for (let k=entry.start;k<entry.start+entry.count;k++) {
            const t=dv.getUint16(k*6+2,true);
            if (t>curT) break;
            const r=G_TRAJ_DATA[k*6+4],c=G_TRAJ_DATA[k*6+5];
            if (!started){ctx.moveTo(c*S+S/2,r*S+S/2);started=true;}else ctx.lineTo(c*S+S/2,r*S+S/2);
        }
        ctx.stroke(); ctx.shadowBlur=0;
    }
    ctx.fillStyle='#ff2222'; ctx.strokeStyle='#fff'; ctx.lineWidth=1.5;
    ctx.beginPath(); ctx.arc(posC*S+S/2,posR*S+S/2,S*0.75,0,Math.PI*2); ctx.fill(); ctx.stroke();
}

// ========== hover panel ==========
function showPanel(cd, usePca) {
    if (!_hoverReady) return;
    const panel=document.getElementById('hover-panel');
    if (!document.getElementById('mm-canvas')) return;
    const [mmIdx,mapId,trajId,t,posR,posC,cat,fc,seg,val,b0,b1,b2]=cd;
    panel.style.display='flex';
    let px=_mouseX+18, py=_mouseY+18;
    const pw=MAP_W*5+VIEW*7+60, ph=Math.max(MAP_H*5,VIEW*7)+100;
    if (px+pw>window.innerWidth)  px=_mouseX-pw-8;
    if (py+ph>window.innerHeight) py=_mouseY-ph-8;
    panel.style.left=px+'px'; panel.style.top=py+'px';
    renderMinimap(mmIdx, usePca?G_MM_PCA:G_MM_UMAP, document.getElementById('mm-canvas'));
    renderMap(mapId,trajId,t,posR,posC,document.getElementById('map-canvas'));
    let beliefHtml='';
    if (b0!==null && b0!==undefined) {
        const pb=Number(b0),pl=Number(b1),pr=Number(b2);
        const maxP=Math.max(pb,pl,pr);
        const pred=pb===maxP?'balanced':pl===maxP?'lakes':'rocky';
        const predCol=CAT_C[pred]||'#ccc';
        beliefHtml=`
          <div style="margin-top:5px;border-top:1px solid #444;padding-top:4px">
            <div style="font-size:10px;color:#aaa;margin-bottom:3px">
              Model belief → <span style="color:${predCol};font-weight:bold">${pred}</span>
              <span style="color:#888">(${(maxP*100).toFixed(0)}% conf)</span>
            </div>
            <div style="display:flex;height:9px;border-radius:4px;overflow:hidden;border:1px solid #555">
              <div style="width:${pb*100}%;background:#e0e0e0" title="balanced ${(pb*100).toFixed(1)}%"></div>
              <div style="width:${pl*100}%;background:#3498db" title="lakes ${(pl*100).toFixed(1)}%"></div>
              <div style="width:${pr*100}%;background:#e74c3c" title="rocky ${(pr*100).toFixed(1)}%"></div>
            </div>
            <div style="display:flex;justify-content:space-between;font-size:9px;margin-top:2px;color:#999">
              <span style="color:#bbb">bal ${(pb*100).toFixed(1)}%</span>
              <span style="color:#3498db">lakes ${(pl*100).toFixed(1)}%</span>
              <span style="color:#e74c3c">rocky ${(pr*100).toFixed(1)}%</span>
            </div>
          </div>`;
    }
    document.getElementById('hover-info').innerHTML =
        `map <b>${mapId}</b> · traj <b>${trajId}</b> · t=<b>${t}</b> · pos=(${posR},${posC})<br>`+
        `cat=<span style="color:${CAT_C[cat]||'#ccc'}"><b>${cat}</b></span> `+
        `skill=<span style="color:${FC_C[fc]||'#ccc'}"><b>${fc}</b></span> `+
        `seg=<b>${seg}</b> · V=<b>${parseFloat(val).toFixed(3)}</b>`+beliefHtml;
}
function hidePanel() { document.getElementById('hover-panel').style.display='none'; }

// ========== attach hover listeners ==========
function _attachHover(divId, isPca) {
    const el=document.getElementById(divId);
    if (!el||el._hoverAttached) return;
    el._hoverAttached=true;
    el.on('plotly_hover', d => { const pt=d.points[0]; if(pt&&pt.customdata) showPanel(pt.customdata,isPca); });
    el.on('plotly_unhover', hidePanel);
}
function attachPerMapHover() {
    // reset flags so listeners re-attach after Plotly.react
    ['plot-pca','plot-umap-map'].forEach(id => {
        const el=document.getElementById(id); if(el) el._hoverAttached=false;
    });
    _attachHover('plot-pca', true);
    _attachHover('plot-umap-map', false);
}

window.addEventListener('load', function() {
    init().then(() => {
        _attachHover('plot-umap', false);
        _attachHover('plot-global-pca', false);
        attachPerMapHover();
    });
});
</script>
"""

_HOVER_PANEL = """\
<div id="hover-panel">
  <div style="padding:16px;color:#888;font-size:13px">Hover a point — images load shortly…</div>
</div>"""

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>BTC PPO+aux-belief — UMAP &amp; PCA</title>
<style>
body {{ font-family: system-ui, sans-serif; background: #f4f6f8; margin: 0; padding: 16px; }}
h1   {{ font-size: 1.4em; color: #2c3e50; margin-bottom: 4px; }}
h2   {{ font-size: 1.1em; color: #34495e; margin-top: 28px; border-bottom: 1px solid #ccc; padding-bottom: 4px; }}
h3   {{ font-size: 0.95em; color: #555; margin: 14px 0 4px; }}
p    {{ color: #555; font-size: 0.9em; }}
.plot-wrap {{ background: #fff; border-radius: 6px; box-shadow: 0 1px 4px #0002; padding: 8px; margin-bottom: 18px; }}

/* global controls bar */
.controls-bar {{
    background: #fff; border-radius: 6px; box-shadow: 0 1px 4px #0002;
    padding: 12px 16px; margin-bottom: 16px;
    display: flex; gap: 28px; align-items: flex-start; flex-wrap: wrap;
}}
.ctrl-group {{ display: flex; flex-direction: column; gap: 6px; }}
.ctrl-group label {{ font-size: 0.82em; color: #666; font-weight: 600; text-transform: uppercase; letter-spacing: .04em; }}

/* color mode buttons */
.color-row {{ display: flex; gap: 5px; flex-wrap: wrap; }}
.color-btn {{
    padding: 5px 11px; border: 1px solid #ccc; border-radius: 4px;
    background: #fff; cursor: pointer; font-size: 0.82em; transition: all 0.12s;
}}
.color-btn:hover  {{ background: #ebf5fb; border-color: #3498db; }}
.color-btn.active {{ background: #2c3e50; color: #fff; border-color: #2c3e50; }}
.belief-btn        {{ border-color: #9b59b6; color: #6c3483; }}
.belief-btn:hover  {{ background: #f5eef8; border-color: #9b59b6; }}
.belief-btn.active {{ background: #9b59b6; border-color: #9b59b6; color:#fff; }}

/* dual range slider (episode % filter) */
.ep-row  {{ display: flex; align-items: center; gap: 10px; }}
.ep-row b {{ font-size: 0.9em; color: #2c3e50; min-width: 80px; }}
.dual-range {{ position: relative; width: 240px; height: 24px; }}
.dual-range input[type=range] {{
    -webkit-appearance: none; appearance: none;
    position: absolute; top: 50%; transform: translateY(-50%);
    width: 100%; height: 4px; background: transparent; pointer-events: none;
    margin: 0;
}}
.dual-range input[type=range]::-webkit-slider-thumb {{
    -webkit-appearance: none; width: 16px; height: 16px; border-radius: 50%;
    background: #3498db; cursor: pointer; pointer-events: all;
    border: 2px solid #fff; box-shadow: 0 1px 4px rgba(0,0,0,.3);
}}
.dual-range input[type=range]::-moz-range-thumb {{
    width: 16px; height: 16px; border-radius: 50%; border: 2px solid #fff;
    background: #3498db; cursor: pointer; pointer-events: all;
    box-shadow: 0 1px 4px rgba(0,0,0,.3);
}}
.range-bg   {{ position: absolute; top: 50%; transform: translateY(-50%); left:0; right:0; height:4px; background:#ddd; border-radius:2px; pointer-events:none; }}
.range-fill {{ position: absolute; top:0; height:4px; background:#3498db; border-radius:2px; left:0%; width:100%; }}

/* map selector */
.map-selector {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0; }}
.map-btn {{
    padding: 8px 12px; border: 2px solid #ccc; border-radius: 6px;
    background: #fff; cursor: pointer; font-size: 0.82em; text-align: center; line-height: 1.4;
    transition: all 0.15s;
}}
.map-btn:hover {{ border-color: #3498db; background: #ebf5fb; }}
.map-btn.active {{ border-color: #2c3e50; background: #2c3e50; color: #fff; }}
.map-btn small {{ display: block; font-size: 0.9em; opacity: 0.75; }}

/* hover panel */
#hover-panel {{
    position: fixed; z-index: 9999; display: none;
    flex-direction: column; gap: 6px;
    background: rgba(20,24,30,0.94); border: 1px solid #444;
    border-radius: 8px; padding: 10px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.55);
    pointer-events: none;
}}
.hp-label {{ font-size: 10px; color: #aaa; text-align: center; margin-bottom: 2px; }}
.hp-info  {{ font-size: 11px; color: #ccc; line-height: 1.6; padding-top: 4px; border-top: 1px solid #333; }}
</style>
{plotly_js}
{js_vars}
</head>
<body>
{hover_panel}

<h1>BTC PPO + auxiliary belief loss — gru_h activation geometry</h1>
<p>Dataset: <code>btc_ppo_belief</code> — {n_rows:,} rows · 90 maps · 128-dim gru_h.
   <b>Hover any point</b> for egocentric obs + map trajectory (hover data loads in background).</p>

<!-- ===== GLOBAL CONTROLS (apply to all 4 plots) ===== -->
<div class="controls-bar">
  <div class="ctrl-group">
    <label>Colour by</label>
    <div class="color-row">
      <button class="color-btn active" id="cbtn-category"     onclick="setColorMode('category')">Category</button>
      <button class="color-btn"        id="cbtn-final_commit" onclick="setColorMode('final_commit')">Skill</button>
      <button class="color-btn"        id="cbtn-segment"      onclick="setColorMode('segment')">Segment</button>
      <button class="color-btn"        id="cbtn-value"        onclick="setColorMode('value')">Value</button>
      <button class="color-btn belief-btn" id="cbtn-belief"   onclick="setColorMode('belief')">Map belief</button>
    </div>
  </div>
  <div class="ctrl-group">
    <label>Episode position filter</label>
    <div class="ep-row">
      <div class="dual-range">
        <div class="range-bg"><div class="range-fill" id="ep-fill"></div></div>
        <input type="range" id="ep-min" min="0" max="100" value="0"   step="1" oninput="onEpSlider('min')">
        <input type="range" id="ep-max" min="0" max="100" value="100" step="1" oninput="onEpSlider('max')">
      </div>
      <b id="ep-range-label">0% → 100%</b>
    </div>
  </div>
</div>

<div id="belief-legend" style="display:none; background:#fafafa; border:1px solid #ddd;
     border-radius:6px; padding:10px 14px; margin-bottom:14px; font-size:0.84em; color:#333">
  <b>Map belief colour key</b> — 3-way probability mix from <code>gru_h → Linear(128,3) → softmax</code>:<br>
  <div style="display:flex; gap:18px; margin-top:8px; align-items:center; flex-wrap:wrap">
    <div style="display:flex;align-items:center;gap:6px"><div style="width:20px;height:20px;background:#fff;border:1px solid #bbb;border-radius:3px"></div><span>100% balanced</span></div>
    <div style="display:flex;align-items:center;gap:6px"><div style="width:20px;height:20px;background:#0000ff;border-radius:3px"></div><span>100% lakes</span></div>
    <div style="display:flex;align-items:center;gap:6px"><div style="width:20px;height:20px;background:#ff0000;border-radius:3px"></div><span>100% rocky</span></div>
    <span style="color:#888;font-size:0.9em">— mixed (e.g. purple = lakes+rocky confusion)</span>
  </div>
  <div style="margin-top:8px;height:16px;background:linear-gradient(to right,#0000ff,#ffffff,#ff0000);border-radius:3px;border:1px solid #ccc"></div>
  <div style="display:flex;justify-content:space-between;font-size:0.78em;color:#888;margin-top:2px"><span>100% lakes</span><span>100% balanced</span><span>100% rocky</span></div>
</div>

<!-- ===== §1 GLOBAL UMAP ===== -->
<h2>§1 — Global UMAP 3-D ({umap_n:,} frames, all maps)</h2>
<div class="plot-wrap"><div id="plot-umap" style="height:700px"></div></div>

<!-- ===== §2 GLOBAL PCA ===== -->
<h2>§2 — Global PCA 3-D (same {umap_n:,} frames)</h2>
<div class="plot-wrap"><div id="plot-global-pca" style="height:700px"></div></div>

<!-- ===== §3 PER-MAP ===== -->
<h2>§3 — Per-map views (10 maps)</h2>
<p>Select a map — both plots update instantly.</p>
<div class="map-selector">{map_buttons}</div>

<h3>§3a — Per-map UMAP (selected map in global UMAP space)</h3>
<div class="plot-wrap"><div id="plot-umap-map" style="height:620px"></div></div>

<h3>§3b — Per-map PCA (OOB-border-free frames only)</h3>
<div class="plot-wrap"><div id="plot-pca" style="height:620px"></div></div>

{js_logic}
</body>
</html>
"""


def _global_data(emb: np.ndarray, lab: pd.DataFrame, belief: np.ndarray,
                 ep: np.ndarray, extra: dict | None = None) -> dict:
    """Pack a global (UMAP or PCA) data dict for JS buildTraces."""
    d = {
        "x":      emb[:, 0].round(4).tolist(),
        "y":      emb[:, 1].round(4).tolist(),
        "z":      emb[:, 2].round(4).tolist(),
        "mm_idx": list(range(len(lab))),          # position in G_MM_UMAP array
        "map_id":  lab["map_id"].tolist(),
        "traj_id": lab["traj_id"].tolist(),
        "t":       lab["t"].tolist(),
        "pos_r":   lab["pos_r"].tolist(),
        "pos_c":   lab["pos_c"].tolist(),
        "cat":     lab["category"].tolist(),
        "fc":      lab["final_commit"].tolist(),
        "seg":     lab["segment"].tolist(),
        "val":     lab["value"].round(3).tolist(),
        "b0":      belief[:, 0].round(4).tolist(),
        "b1":      belief[:, 1].round(4).tolist(),
        "b2":      belief[:, 2].round(4).tolist(),
        "ep_pct":  ep.round(3).tolist(),
    }
    if extra:
        d.update(extra)
    return d


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    import plotly

    print("Loading bundle …", flush=True)
    b = ActivationBundle(DATASET)
    print(b.summary(), flush=True)

    # ---- belief head weights + max_t lookup (for ep_pct)
    print("\nLoading belief head …", flush=True)
    belief_W, belief_bias = load_belief_head(BELIEF_CKPT)
    max_t_of = b.labels.groupby(["map_id", "traj_id"])["t"].max().to_dict()

    # ---- sample for global plots (UMAP + PCA share the same 40k rows)
    print(f"\nSampling {UMAP_N:,} rows for global plots …", flush=True)
    X_u, mm_u, lab_u = load_umap_sample(b, UMAP_N, UMAP_SEED)
    belief_u = compute_belief_probs(X_u, belief_W, belief_bias)
    ep_u     = compute_ep_pct(lab_u, max_t_of)

    print("Fitting UMAP 3D …", flush=True)
    emb_u = fit_umap(X_u)
    print("Fitting Global PCA 3D …", flush=True)
    emb_gp, evr_gp = fit_global_pca(X_u)

    umap_range = {
        "x": [float(emb_u[:,0].min())-0.5, float(emb_u[:,0].max())+0.5],
        "y": [float(emb_u[:,1].min())-0.5, float(emb_u[:,1].max())+0.5],
        "z": [float(emb_u[:,2].min())-0.5, float(emb_u[:,2].max())+0.5],
    }

    # ---- per-map data
    cat_of    = b.labels.groupby("map_id")["category"].first().to_dict()
    n_rows_of = b.labels.groupby("map_id").size().to_dict()

    all_pca_info, all_pca_data, all_umap_data = {}, {}, {}
    pca_mm_arrays = []
    pca_offset = 0

    for mid in SELECTED_MAPS:
        print(f"\nPCA map {mid} ({cat_of[mid]}) …", flush=True)
        emb_p, lab_p, mm_p, stats = build_pca(b, mid)
        n_clean = len(lab_p)

        mask_u = (lab_u["map_id"] == mid).to_numpy()
        n_umap = int(mask_u.sum())

        all_pca_info[str(mid)] = {
            "category": cat_of[mid],
            "n_rows": int(n_rows_of[mid]),
            "n_total": stats["n_total"],
            "n_kept":  stats["n_kept"],
            "n_oob":   stats["n_oob"],
            "evr":     stats["evr"],
            "n_umap":  n_umap,
        }

        # belief + ep_pct for PCA frames
        X_pca  = b.load_activations(SRC, lab_p["row_id"].to_numpy())
        bp_pca = compute_belief_probs(X_pca, belief_W, belief_bias)
        ep_pca = compute_ep_pct(lab_p, max_t_of)

        mm_idx_pca = (np.arange(n_clean) + pca_offset).tolist()
        all_pca_data[str(mid)] = {
            "x":      emb_p[:,0].round(4).tolist(),
            "y":      emb_p[:,1].round(4).tolist(),
            "z":      emb_p[:,2].round(4).tolist(),
            "mm_idx":  mm_idx_pca,
            "map_id":  lab_p["map_id"].tolist(),
            "traj_id": lab_p["traj_id"].tolist(),
            "t":       lab_p["t"].tolist(),
            "pos_r":   lab_p["pos_r"].tolist(),
            "pos_c":   lab_p["pos_c"].tolist(),
            "cat":     lab_p["category"].tolist(),
            "fc":      lab_p["final_commit"].tolist(),
            "seg":     lab_p["segment"].tolist(),
            "val":     lab_p["value"].round(3).tolist(),
            "b0":      bp_pca[:, 0].round(4).tolist(),
            "b1":      bp_pca[:, 1].round(4).tolist(),
            "b2":      bp_pca[:, 2].round(4).tolist(),
            "ep_pct":  ep_pca.round(3).tolist(),
        }
        pca_mm_arrays.append(mm_p)
        pca_offset += n_clean

        # UMAP per-map: mm_idx references position in G_MM_UMAP
        gu_idx    = np.where(mask_u)[0]
        lab_u_m   = lab_u[mask_u].reset_index(drop=True)
        bp_umap_m = belief_u[mask_u]
        ep_umap_m = ep_u[mask_u]
        all_umap_data[str(mid)] = {
            "x":      emb_u[mask_u, 0].round(4).tolist(),
            "y":      emb_u[mask_u, 1].round(4).tolist(),
            "z":      emb_u[mask_u, 2].round(4).tolist(),
            "mm_idx":  gu_idx.tolist(),
            "map_id":  lab_u_m["map_id"].tolist(),
            "traj_id": lab_u_m["traj_id"].tolist(),
            "t":       lab_u_m["t"].tolist(),
            "pos_r":   lab_u_m["pos_r"].tolist(),
            "pos_c":   lab_u_m["pos_c"].tolist(),
            "cat":     lab_u_m["category"].tolist(),
            "fc":      lab_u_m["final_commit"].tolist(),
            "seg":     lab_u_m["segment"].tolist(),
            "val":     lab_u_m["value"].round(3).tolist(),
            "b0":      bp_umap_m[:, 0].round(4).tolist(),
            "b1":      bp_umap_m[:, 1].round(4).tolist(),
            "b2":      bp_umap_m[:, 2].round(4).tolist(),
            "ep_pct":  ep_umap_m.round(3).tolist(),
        }
        print(f"  PCA: {stats['n_kept']:,}/{stats['n_total']:,} clean  "
              f"PC1={stats['evr'][0]:.1f}%  UMAP subset: {n_umap}", flush=True)

    # ---- pack all data (global + per-map)
    all_data_json = json.dumps({
        "maps":       SELECTED_MAPS,
        "info":       all_pca_info,
        "pca":        all_pca_data,
        "umap":       all_umap_data,
        "umap_range": umap_range,
        "global_umap": _global_data(emb_u, lab_u, belief_u, ep_u),
        "global_pca":  _global_data(emb_gp, lab_u, belief_u, ep_u,
                                    extra={"evr": evr_gp.tolist()}),
    })

    # ---- pack binary blobs
    print("\nCompressing tile data …", flush=True)
    mm_umap_b64  = gz_b64(mm_u)
    mm_pca_all   = np.concatenate(pca_mm_arrays, axis=0)
    mm_pca_b64   = gz_b64(mm_pca_all)
    traj_b64     = pack_trajectories(b)
    terrain_raw  = b.maps["terrain"].astype(np.uint8).tobytes()
    terrain_b64  = base64.b64encode(terrain_raw).decode()
    all_data_b64 = gz_b64_text(all_data_json)

    print(f"  minimap UMAP  {len(mm_umap_b64)/1e6:.2f} MB")
    print(f"  minimap PCA   {len(mm_pca_b64)/1e6:.2f} MB")
    print(f"  trajectories  {len(traj_b64)/1e6:.2f} MB")
    print(f"  terrain       {len(terrain_b64)/1e6:.3f} MB")
    print(f"  all_data JSON {len(all_data_b64)/1e6:.2f} MB")

    # ---- map selector buttons
    map_buttons = ""
    for mid in SELECTED_MAPS:
        info = all_pca_info[str(mid)]
        cat  = info["category"]
        map_buttons += (
            f'<button class="map-btn" id="btn-{mid}" onclick="updatePerMap({mid})">'
            f'Map {mid}<small>{cat} · {info["n_rows"]:,} rows</small></button>'
        )

    # ---- assemble HTML
    print("\nWriting HTML …", flush=True)
    plotly_js = "<script>" + open(
        Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
    ).read() + "</script>"

    js_vars = _JS_VARS.format(
        palette=json.dumps(b.manifest["tile_colors"]),
        mm_umap=mm_umap_b64, mm_pca=mm_pca_b64,
        traj=traj_b64, terrain=terrain_b64,
        all_data=all_data_b64,
        view=VIEW, map_h=MAP_H, map_w=MAP_W,
    )

    html = HTML_TEMPLATE.format(
        plotly_js=plotly_js, js_vars=js_vars, js_logic=_JS_LOGIC,
        hover_panel=_HOVER_PANEL,
        n_rows=len(b.labels), umap_n=UMAP_N,
        map_buttons=map_buttons,
    )

    out = OUT / "report.html"
    out.write_text(html)
    print(f"\nDone → {out}  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
