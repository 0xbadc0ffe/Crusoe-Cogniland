#!/usr/bin/env python
"""Render paper-quality static figures for cogniland_overview.tex from the
belief-analysis result JSONs. Output -> paper/figures/cogniland/.

Figures:
  belief_pca.png        Experiment 3: PCA of the belief manifold + steering paths
  probes_success.png    Experiment 1: held-out success + probe accuracy + belief->door + steering
  confusion_door.png    Experiment 1: confusion matrix + category->door behaviour
  dream_composition.png Experiment 2: imagined terrain fraction by believed category
  offmanifold.png       Experiment 3: off-manifold distance, linear vs manifold path
  dream_rollout.png     Experiment 2 (qualitative): decoded imagined-future strips per category
"""
from __future__ import annotations
import json, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT = REPO / "paper/figures/cogniland"; OUT.mkdir(parents=True, exist_ok=True)
DATA = json.loads((REPO/"outputs/bridge_tunnel_forkwall/belief_report_data.json").read_text())
MAN  = json.loads((REPO/"outputs/bridge_tunnel_forkwall/manifold_steer_data.json").read_text())

# consistent palette (matches the paper + report)
C = {"balanced":"#199e70","lakes":"#2a78d6","rocky":"#eb6834"}
ACC, GOOD, INK, MUT, GRID = "#4a3aa7", "#1a7a2e", "#16211f", "#7c8b86", "#e3e8e3"
CATS = ["balanced","lakes","rocky"]
TILE_COLORS = np.array([(110,173,86),(61,113,184),(110,110,110),(140,90,50),
    (250,220,60),(0,0,0),(24,70,32),(224,205,140),(134,104,74)],dtype=np.uint8)

plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False,
    "axes.edgecolor":MUT,"axes.linewidth":0.8,"xtick.color":INK,"ytick.color":INK,
    "font.family":"DejaVu Sans","figure.dpi":150})

def save(fig, name):
    p = OUT/name; fig.savefig(p, bbox_inches="tight", dpi=150); plt.close(fig); print("wrote", p)

# ---------------------------------------------------------------- belief PCA
def fig_belief_pca():
    p = MAN["pca"]
    cloud = np.array(p["cloud_xy"]); z = np.array(p["cloud_z"])
    curve = np.array(p["curve_xy"]); lin = np.array(p["linear_xy"]); man = np.array(p["manifold_xy"])
    fig, ax = plt.subplots(figsize=(6.4,4.4))
    # diverging color by evidence coordinate z: rocky(-1)..neutral(0)..lakes(+1)
    def zcol(zz):
        zz=np.clip(zz,-1,1); out=np.zeros((len(zz),3))
        for i,v in enumerate(zz):
            a=np.array([124,139,134])/255.
            b=(np.array([235,104,52]) if v<0 else np.array([42,120,214]))/255.
            out[i]=a+(b-a)*abs(v)
        return out
    ax.scatter(cloud[:,0],cloud[:,1],c=zcol(z),s=10,alpha=.55,linewidths=0)
    ax.plot(curve[:,0],curve[:,1],color=INK,lw=2.4,label="fitted belief manifold",zorder=4)
    ax.plot(lin[:,0],lin[:,1],color=ACC,lw=2.2,ls="--",label="linear steering (chord)",zorder=5)
    ax.plot(man[:,0],man[:,1],color=GOOD,lw=2.6,label="manifold steering (geodesic)",zorder=5)
    ax.scatter([man[0,0]],[man[0,1]],c=[C["rocky"]],s=70,ec="white",lw=1.5,zorder=6)
    ax.scatter([man[-1,0]],[man[-1,1]],c=[C["lakes"]],s=70,ec="white",lw=1.5,zorder=6)
    ax.annotate("rocky end\n(top door)",man[0],textcoords="offset points",xytext=(6,-18),fontsize=7.5,color=C["rocky"])
    ax.annotate("lakes end\n(bottom door)",man[-1],textcoords="offset points",xytext=(-10,10),fontsize=7.5,color=C["lakes"])
    ax.set_xlabel("belief-manifold PC1  (tracks evidence coordinate, r=0.73)")
    ax.set_ylabel("PC2")
    ax.set_title("PCA of the belief: a curved 1-D manifold",loc="left",fontsize=11,fontweight="bold")
    ax.legend(loc="upper left",fontsize=7.5,frameon=False)
    fig.text(0.12,0.005,"dot colour = evidence coordinate z:  rocky (-1)  ->  neutral (0)  ->  lakes (+1)",
             fontsize=7.5,color=MUT)
    save(fig,"belief_pca.png")

# ---------------------------------------------------- quantitative: probes/success
def fig_probes_success():
    e1=DATA["exp1"]
    fig,axes=plt.subplots(1,3,figsize=(9.6,3.0))
    # (a) held-out success per category
    ax=axes[0]; vals=[DATA["per_cat_success"][c] for c in CATS]
    ax.bar(range(3),vals,color=[C[c] for c in CATS],width=.62)
    ax.axhline(1.0,color=MUT,lw=.7,ls=":")
    for i,v in enumerate(vals): ax.text(i,v+.02,f"{v*100:.0f}%",ha="center",fontsize=8)
    ax.set_xticks(range(3)); ax.set_xticklabels(CATS,fontsize=8); ax.set_ylim(0,1.12)
    ax.set_ylabel("success"); ax.set_title("(a) held-out success",loc="left",fontsize=9.5,fontweight="bold")
    # (b) probe accuracy nearest-mean vs logreg (pre-decision)
    ax=axes[1]
    accs=[e1["probe_acc"],e1["probe_acc_logreg"]]
    ax.bar([0,1],accs,color=[MUT,ACC],width=.55)
    ax.axhline(1/3,color=MUT,lw=.8,ls="--"); ax.text(1.4,1/3+.01,"chance",fontsize=7,color=MUT,ha="right")
    for i,v in enumerate(accs): ax.text(i,v+.02,f"{v*100:.0f}%",ha="center",fontsize=8)
    ax.set_xticks([0,1]); ax.set_xticklabels(["nearest\nmean","logistic\nregr."],fontsize=8)
    ax.set_ylim(0,1.0); ax.set_ylabel("category accuracy")
    ax.set_title("(b) belief is decodable",loc="left",fontsize=9.5,fontweight="bold")
    # (c) belief->door consistency + steering flips
    ax=axes[2]
    labels=["door vs\ndecoded","door vs\ntruth","flip:\nmean-diff","flip:\nlogreg-dir"]
    vals=[e1["pred_matches_door"],e1["true_matches_door"],e1["swap_flip_rate"],e1["swap_flip_rate_logreg"]]
    cols=[GOOD,MUT,MUT,ACC]
    ax.bar(range(4),vals,color=cols,width=.66)
    for i,v in enumerate(vals): ax.text(i,v+.02,f"{v*100:.0f}%",ha="center",fontsize=7.5)
    ax.set_xticks(range(4)); ax.set_xticklabels(labels,fontsize=7.2); ax.set_ylim(0,1.12)
    ax.set_ylabel("rate")
    ax.set_title("(c) causal: behaviour follows belief",loc="left",fontsize=9.5,fontweight="bold")
    fig.tight_layout()
    save(fig,"probes_success.png")

# ---------------------------------------------------- confusion + door matrix
def fig_confusion_door():
    e1=DATA["exp1"]
    fig,axes=plt.subplots(1,2,figsize=(8.0,3.4))
    # confusion (logreg)
    ax=axes[0]; M=np.array(e1["confusion_logreg"],float); Mn=M/M.sum(1,keepdims=True)
    im=ax.imshow(Mn,cmap="Purples",vmin=0,vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j,i,f"{int(M[i,j])}",ha="center",va="center",
                    color="white" if Mn[i,j]>.5 else INK,fontsize=9)
    ax.set_xticks(range(3)); ax.set_xticklabels(CATS,fontsize=8)
    ax.set_yticks(range(3)); ax.set_yticklabels(CATS,fontsize=8)
    ax.set_xlabel("probe prediction"); ax.set_ylabel("true category")
    ax.set_title("(a) probe confusion",loc="left",fontsize=9.5,fontweight="bold")
    # door matrix
    ax=axes[1]; D=np.array(e1["door_matrix"],float); dl=e1["door_labels"]
    x=np.arange(3); w=.26; dcol={"top":C["rocky"],"bottom":C["lakes"],"timeout":MUT}
    for k,dn in enumerate(dl):
        ax.bar(x+(k-1)*w,D[:,k],w,color=dcol.get(dn,MUT),label=dn)
    ax.set_xticks(x); ax.set_xticklabels(CATS,fontsize=8); ax.set_ylim(0,1.05)
    ax.set_ylabel("fraction of episodes"); ax.legend(fontsize=7.5,frameon=False,ncol=3,loc="upper center")
    ax.set_title("(b) category → door reached",loc="left",fontsize=9.5,fontweight="bold")
    fig.tight_layout(); save(fig,"confusion_door.png")

# ---------------------------------------------------- dream composition (exp2)
def fig_dream_composition():
    ds=DATA["exp2"]["dream_stats"]
    fig,ax=plt.subplots(figsize=(5.6,3.4))
    x=np.arange(3); w=.36
    water=[np.mean(ds[c]["water"]) for c in CATS]; rock=[np.mean(ds[c]["rock"]) for c in CATS]
    ws=[np.std(ds[c]["water"]) for c in CATS]; rs=[np.std(ds[c]["rock"]) for c in CATS]
    ax.bar(x-w/2,water,w,yerr=ws,capsize=3,color="#3d71b8",label="water")
    ax.bar(x+w/2,rock,w,yerr=rs,capsize=3,color="#6e6e6e",label="rock")
    ax.set_xticks(x); ax.set_xticklabels([f"belief =\n{c}" for c in CATS],fontsize=8)
    ax.set_ylabel("fraction of imagined tiles")
    ax.set_title("Imagined terrain follows the believed category",loc="left",fontsize=10,fontweight="bold")
    ax.legend(fontsize=8,frameon=False)
    fig.tight_layout(); save(fig,"dream_composition.png")

# ---------------------------------------------------- off-manifold (exp3)
def fig_offmanifold():
    t=np.array(MAN["t_values"])
    fig,ax=plt.subplots(figsize=(5.6,3.2))
    ax.plot(t,MAN["linear"]["off_manifold"],color=ACC,lw=2.2,ls="--",marker="o",ms=3,label="linear (chord)")
    ax.plot(t,MAN["manifold"]["off_manifold"],color=GOOD,lw=2.4,marker="o",ms=3,label="manifold (geodesic)")
    ax.set_xlabel("steering progress  t   (rocky → lakes)")
    ax.set_ylabel("distance off the belief manifold")
    ax.set_title("Linear steering leaves the manifold; the geodesic stays on it",loc="left",fontsize=9.5,fontweight="bold")
    ax.legend(fontsize=8,frameon=False)
    fig.tight_layout(); save(fig,"offmanifold.png")

# ---------------------------------------------------- qualitative dream rollout
def fig_dream_rollout():
    seqs=DATA["exp2"]["example_sequences"]
    ncat=len(CATS); H=min(8,len(seqs[CATS[0]]["grids"]))
    fig,axes=plt.subplots(ncat,H,figsize=(H*1.05,ncat*1.15))
    for r,c in enumerate(CATS):
        grids=seqs[c]["grids"]
        for k in range(H):
            ax=axes[r,k]; g=np.array(grids[k])
            ax.imshow(TILE_COLORS[g]/255.,interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([])
            if r==0: ax.set_title(f"t+{k+1}",fontsize=7.5)
        axes[r,0].set_ylabel(c,fontsize=9,color=C[c],rotation=90,labelpad=2,fontweight="bold")
    fig.suptitle("Decoded imagined future ('dream') seeded from each belief",fontsize=10,fontweight="bold",x=.02,ha="left")
    fig.tight_layout(rect=[0,0,1,0.96]); save(fig,"dream_rollout.png")

if __name__=="__main__":
    fig_belief_pca()
    fig_probes_success()
    fig_confusion_door()
    fig_dream_composition()
    fig_offmanifold()
    fig_dream_rollout()
    print("done ->", OUT)


# ---------------------------------------------------- end-of-episode belief PCA (3D)
def fig_belief_pca_end3d():
    import pickle
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    c = pickle.load(open(REPO/"outputs/bridge_tunnel_forkwall/belief_report_raw.pkl","rb"))
    eps = c["episodes"]
    feats, cats, doors = [], [], []
    for e in eps:
        feats.append(np.concatenate([np.asarray(e["stoch_pre"]).reshape(-1),
                                     np.asarray(e["deter_pre"]).reshape(-1)]))
        cats.append(e["category"]); doors.append(e["door"])
    X = np.stack(feats); cats = np.array(cats)
    mu = X.mean(0); Xc = X - mu
    U,S,Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = Xc @ Vt[:3].T                       # (n,3) scores
    ev = (S[:3]**2 / (S**2).sum())
    fig = plt.figure(figsize=(10.6,4.4))
    # (a) 3D scatter colored by category
    ax = fig.add_subplot(1,2,1, projection="3d")
    for cat in CATS:
        m = cats==cat
        ax.scatter(Z[m,0],Z[m,1],Z[m,2],c=C[cat],s=26,alpha=.85,depthshade=True,label=cat,edgecolors="white",linewidths=.3)
        # class-mean marker
        ax.scatter(*Z[m].mean(0),c=C[cat],s=170,marker="*",edgecolors="black",linewidths=.8,zorder=6)
    ax.set_xlabel(f"PC1 ({ev[0]*100:.0f}%)",fontsize=8,labelpad=-6)
    ax.set_ylabel(f"PC2 ({ev[1]*100:.0f}%)",fontsize=8,labelpad=-6)
    ax.set_zlabel(f"PC3 ({ev[2]*100:.0f}%)",fontsize=8,labelpad=-6)
    ax.tick_params(labelsize=6.5,pad=-2); ax.view_init(elev=18,azim=-60)
    ax.set_title("(a) end-of-episode belief, 3-D PCA",loc="left",fontsize=10,fontweight="bold")
    ax.legend(fontsize=8,frameon=False,loc="upper left")
    # (b) PC1-PC2 with door outcome overlaid (top vs bottom) to show the decision axis
    ax2 = fig.add_subplot(1,2,2)
    dmark={"top":"^","bottom":"v","timeout":"x"}
    for cat in CATS:
        for dv in ["top","bottom","timeout"]:
            m=(cats==cat)&(np.array(doors)==dv)
            if m.sum(): ax2.scatter(Z[m,0],Z[m,1],c=C[cat],s=30,marker=dmark[dv],alpha=.8,linewidths=.5,edgecolors="white" if dv!="timeout" else C[cat])
    ax2.set_xlabel(f"PC1 ({ev[0]*100:.0f}%)"); ax2.set_ylabel(f"PC2 ({ev[1]*100:.0f}%)")
    ax2.set_title("(b) same, PC1-PC2 (marker = door reached)",loc="left",fontsize=10,fontweight="bold")
    from matplotlib.lines import Line2D
    handles=[Line2D([0],[0],marker='o',ls='',mfc=C[c],mec='none',label=c) for c in CATS]+\
            [Line2D([0],[0],marker='^',ls='',mfc='#555',mec='none',label='top door'),
             Line2D([0],[0],marker='v',ls='',mfc='#555',mec='none',label='bottom door')]
    ax2.legend(handles=handles,fontsize=7.5,frameon=False,ncol=1,loc="best")
    fig.tight_layout(); save(fig,"belief_pca_end3d.png")


# ---------------------------------------------------- corridor manifold in 3D
def fig_manifold_3d():
    import pickle
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    # recompute a 3-D PCA of the corridor belief cloud (not the planar curve-basis)
    c = pickle.load(open(REPO/"outputs/bridge_tunnel_forkwall/manifold_steer_raw.pkl","rb"))
    traj = c["traj"]
    feats = np.concatenate([traj["rocky"]["feats"], traj["lakes"]["feats"]],0)
    zc = np.concatenate([traj["rocky"]["z"], traj["lakes"]["z"]],0)
    rng=np.random.default_rng(0)
    idx=rng.choice(len(feats),size=min(1200,len(feats)),replace=False)
    X=feats[idx]; zc=zc[idx]; mu=X.mean(0); Xc=X-mu
    U,S,Vt=np.linalg.svd(Xc,full_matrices=False); Z=Xc@Vt[:3].T; ev=S[:3]**2/(S**2).sum()
    # binned means -> manifold polyline in the same basis
    edges=np.linspace(-1,1,16); pts=[]
    for b in range(15):
        m=(zc>=edges[b])&(zc<edges[b+1] if b<14 else zc<=edges[b+1])
        if m.sum()>=3: pts.append(Z[m].mean(0))
    pts=np.array(pts)
    fig=plt.figure(figsize=(6.4,4.8)); ax=fig.add_subplot(111,projection="3d")
    def zcol(v):
        v=np.clip(v,-1,1); a=np.array([124,139,134])/255.
        b=(np.array([235,104,52]) if v<0 else np.array([42,120,214]))/255.; return a+(b-a)*abs(v)
    ax.scatter(Z[:,0],Z[:,1],Z[:,2],c=[zcol(v) for v in zc],s=9,alpha=.45,linewidths=0)
    ax.plot(pts[:,0],pts[:,1],pts[:,2],color=INK,lw=2.6,marker="o",ms=3,zorder=6)
    ax.set_xlabel(f"PC1 ({ev[0]*100:.0f}%)",fontsize=8,labelpad=-6)
    ax.set_ylabel(f"PC2 ({ev[1]*100:.0f}%)",fontsize=8,labelpad=-6)
    ax.set_zlabel(f"PC3 ({ev[2]*100:.0f}%)",fontsize=8,labelpad=-6)
    ax.tick_params(labelsize=6.5,pad=-2); ax.view_init(elev=20,azim=-70)
    ax.set_title("Belief manifold in 3-D (corridor beliefs, coloured by evidence z)",loc="left",fontsize=9.5,fontweight="bold")
    fig.tight_layout(); save(fig,"manifold_3d.png")


if True:
    fig_belief_pca_end3d()
    fig_manifold_3d()
    print("added 3-D + end-of-episode figures")
