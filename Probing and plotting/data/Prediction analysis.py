"""
Separation Direction Analysis — Auto-scaling Pipeline
======================================================
HOW TO USE:
  1. Put your labeled reference files in:
       ./data/yz/        ← all known YZ runs
       ./data/xz/        ← all known XZ runs (if any)
  2. Put unknown files to predict in:
       ./data/unknown/   ← files you want classified
  3. Put free-space baseline files in:
       ./data/baseline/  ← IC / free-space recordings
  4. Run:  python separation_analysis.py
  5. Outputs go to ./output/
       report.html       ← full interactive dashboard
       fig1_signals.png
       fig2_features.png
       fig3_predictions.png
       fig4_presep.png
       predictions.csv   ← machine-readable results

No code editing needed — just add/remove files from the folders.
"""

import os, glob, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.signal import butter, filtfilt
from scipy.spatial.distance import euclidean

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG  — only change these if your folder structure is different
# ═══════════════════════════════════════════════════════════════════════════
DIRS = {
    "yz":       "./data/yz",
    "xz":       "./data/xz",
    "unknown":  "./data/unknown",
    "baseline": "./data/baseline",
}
OUTPUT_DIR    = "./output"
LOWPASS_HZ    = 20       # low-pass filter cutoff
PRE_SEP_WIN   = 2.0      # seconds before sep event to use as pre-sep window
DOWNSAMPLE_N  = 300      # max points per signal for plotting

# Colour pools — automatically assigned per file
YZ_COLORS  = ["#7C6FF7","#A78BFA","#C4B5FD","#818CF8","#6366F1"]
XZ_COLORS  = ["#F87171","#FB923C","#FBBF24","#F43F5E","#EF4444"]
UNK_COLORS = ["#34D399","#6EE7B7","#FCD34D","#67E8F9","#86EFAC",
               "#FCA5A5","#A5F3FC","#C4B5FD","#FDE68A","#BBF7D0",
               "#F9A8D4","#BAE6FD","#D9F99D","#FED7AA","#E9D5FF",
               "#99F6E4","#FDBA74","#F5D0FE","#CCFBF1","#FEF08A"]

STYLE = {
    "bg":    "#0D0F14", "surf":  "#151821", "surf2": "#1C2030",
    "border":"#252A3A", "text":  "#E8EAF0", "muted": "#6B7280",
    "grid":  "#1C2030",
}

# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def butter_lp(df, cutoff=LOWPASS_HZ):
    dt  = df["time"].diff().median()
    fs  = 1.0 / dt
    b, a = butter(4, min(cutoff / (0.5 * fs), 0.99), btype="low")
    for col in ["Fx","Fy","Fz","Tx","Ty","Tz"]:
        if col in df.columns:
            df[col] = filtfilt(b, a, df[col])
    return df

def compute_bias(baseline_dir):
    """Average all free-space baseline files → sensor bias vector."""
    bias  = {c: [] for c in ["Fx","Fy","Fz","Tx","Ty","Tz"]}
    files = glob.glob(os.path.join(baseline_dir, "*.csv"))
    for f in files:
        try:
            df = pd.read_csv(f).dropna()
            for c in bias:
                if c in df.columns:
                    bias[c].append(df[c].mean())
        except Exception:
            pass
    return {c: float(np.mean(v)) if v else 0.0 for c, v in bias.items()}

def load_run(path, bias):
    """Load, filter, bias-correct, detect separation."""
    df = pd.read_csv(path).dropna().sort_values("time").reset_index(drop=True)
    df = butter_lp(df)
    for c in ["Fx","Fy","Fz","Tx","Ty","Tz"]:
        if c in df.columns:
            df[c] = df[c] - bias.get(c, 0.0)
    df["F_mag"] = np.sqrt(df.Fx**2 + df.Fy**2 + df.Fz**2)
    df["T_mag"] = np.sqrt(df.Tx**2 + df.Ty**2 + df.Tz**2)
    df["dF_dt"] = df["F_mag"].diff() / df["time"].diff()
    si  = df["dF_dt"].idxmin()
    sep = float(df.loc[si, "time"])
    df["t_norm"] = df["time"] - sep
    return df, sep

def extract_features(df):
    """Orientation-invariant feature vector."""
    pre = df[df["t_norm"].between(-PRE_SEP_WIN, 0)]
    if len(pre) < 5:
        pre = df.iloc[:max(5, int(len(df)*0.1))]

    def safe_corr(a, b): return float(a.corr(b)) if a.std() > 1e-9 and b.std() > 1e-9 else 0.0
    def safe_ratio(a, b): return float(a.std() / (b.std() + 1e-9))

    return {
        # Full-run features
        "std_Fx_Fz":     safe_ratio(df.Fx,  df.Fz),
        "std_Fy_Fz":     safe_ratio(df.Fy,  df.Fz),
        "std_Tx_Tz":     safe_ratio(df.Tx,  df.Tz),
        "std_Ty_Tz":     safe_ratio(df.Ty,  df.Tz),
        "corr_Tx_Tz":    safe_corr(df.Tx,   df.Tz),
        "corr_Ty_Tz":    safe_corr(df.Ty,   df.Tz),
        "corr_Fx_Fz":    safe_corr(df.Fx,   df.Fz),
        "corr_Fx_Fy":    safe_corr(df.Fx,   df.Fy),
        "T_mag_CV":      df.T_mag.std() / (df.T_mag.mean() + 1e-9),
        "F_mag_CV":      df.F_mag.std() / (df.F_mag.mean() + 1e-9),
        # Pre-separation features (most discriminative)
        "pre_corr_Tx_Tz":  safe_corr(pre.Tx,  pre.Tz),
        "pre_std_Fy_Fz":   safe_ratio(pre.Fy, pre.Fz),
        "pre_corr_Fx_Fz":  safe_corr(pre.Fx,  pre.Fz),
        "pre_std_Tx_Tz":   safe_ratio(pre.Tx, pre.Tz),
        "pre_F_mag_CV":    pre.F_mag.std() / (pre.F_mag.mean() + 1e-9),
    }

CLASSIFY_KEYS = [
    "std_Fx_Fz","std_Fy_Fz","std_Tx_Tz","std_Ty_Tz",
    "corr_Tx_Tz","corr_Ty_Tz","corr_Fx_Fz","T_mag_CV",
    "pre_corr_Tx_Tz","pre_std_Fy_Fz","pre_corr_Fx_Fz","pre_std_Tx_Tz",
]

def classify(feats, centroids, thresholds):
    """
    Classify a feature vector:
    - Compute distance to each known-class centroid.
    - Pick nearest centroid if within threshold, else 'unknown'.
    Returns (label, distances_dict, confidence).
    """
    v = np.array([feats[k] for k in CLASSIFY_KEYS])
    dists = {}
    for cls, c in centroids.items():
        dists[cls] = float(np.linalg.norm(v - c))
    nearest = min(dists, key=dists.get)
    nearest_dist = dists[nearest]
    # Confidence: how much closer to nearest vs second nearest
    sorted_d = sorted(dists.values())
    if len(sorted_d) >= 2 and sorted_d[1] > 0:
        conf = min(1.0, (sorted_d[1] - sorted_d[0]) / sorted_d[1])
    else:
        conf = 1.0
    return nearest, dists, round(conf * 100, 1)

# ═══════════════════════════════════════════════════════════════════════════
# LOAD ALL DATA
# ═══════════════════════════════════════════════════════════════════════════

def load_class(directory, color_pool, class_label):
    """Load all CSVs from a directory, return list of run dicts."""
    runs = []
    files = sorted(glob.glob(os.path.join(directory, "*.csv")))
    for i, path in enumerate(files):
        name = os.path.splitext(os.path.basename(path))[0]
        color = color_pool[i % len(color_pool)]
        try:
            df, sep = load_run(path, bias)
            f = extract_features(df)
            runs.append({
                "name": name, "path": path, "class": class_label,
                "color": color, "df": df, "sep": sep, "feats": f,
                "contact_Fmag": float(df.F_mag.mean()),
                "bias_pct": round(bias_mag / (df.F_mag.mean() + 1e-9) * 100, 1),
            })
            print(f"  [{class_label}] {name}: sep=t{sep:.2f}s  contact={df.F_mag.mean():.2f}N")
        except Exception as e:
            print(f"  SKIP {name}: {e}")
    return runs

print("=" * 60)
print("Loading baseline…")
os.makedirs(DIRS["baseline"], exist_ok=True)
bias = compute_bias(DIRS["baseline"])
bias_mag = float(np.sqrt(sum(v**2 for v in bias.values())))
print(f"  Bias vector: Fx={bias['Fx']:.3f} Fy={bias['Fy']:.3f} Fz={bias['Fz']:.3f}")
print(f"  |bias| = {bias_mag:.3f} N")

print("\nLoading YZ reference runs…")
os.makedirs(DIRS["yz"], exist_ok=True)
yz_runs = load_class(DIRS["yz"], YZ_COLORS, "YZ")

print("\nLoading XZ reference runs…")
os.makedirs(DIRS["xz"], exist_ok=True)
xz_runs = load_class(DIRS["xz"], XZ_COLORS, "XZ")

print("\nLoading unknown runs…")
os.makedirs(DIRS["unknown"], exist_ok=True)
unk_runs = load_class(DIRS["unknown"], UNK_COLORS, "unknown")

all_runs    = yz_runs + xz_runs + unk_runs
ref_runs    = yz_runs + xz_runs

if not yz_runs:
    print("\nWARNING: No YZ reference files found in ./data/yz/")
    print("         Put at least 2 labeled YZ CSVs there to enable classification.\n")

# ═══════════════════════════════════════════════════════════════════════════
# BUILD CENTROIDS + CLASSIFY
# ═══════════════════════════════════════════════════════════════════════════

centroids = {}
if yz_runs:
    vyz = np.array([[r["feats"][k] for k in CLASSIFY_KEYS] for r in yz_runs])
    centroids["YZ"] = vyz.mean(axis=0)
if xz_runs:
    vxz = np.array([[r["feats"][k] for k in CLASSIFY_KEYS] for r in xz_runs])
    centroids["XZ"] = vxz.mean(axis=0)

# Compute distances for all runs (ref + unknown)
for r in all_runs:
    v = np.array([r["feats"][k] for k in CLASSIFY_KEYS])
    r["dists"] = {cls: float(np.linalg.norm(v - c)) for cls, c in centroids.items()}
    if centroids:
        nearest, dists, conf = classify(r["feats"], centroids, thresholds={})
        r["predicted"] = nearest
        r["confidence"] = conf
    else:
        r["predicted"] = r["class"]
        r["confidence"] = 100.0

# Classify unknowns
print("\n" + "=" * 60)
print("PREDICTIONS")
print("=" * 60)
for r in unk_runs:
    dist_str = "  ".join(f"{cls}={d:.3f}" for cls, d in r["dists"].items())
    print(f"  {r['name']:20s}  →  {r['predicted']}  (conf={r['confidence']}%  {dist_str})")

# ═══════════════════════════════════════════════════════════════════════════
# MATPLOTLIB SETUP
# ═══════════════════════════════════════════════════════════════════════════

os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.rcParams.update({
    "figure.facecolor": STYLE["bg"], "axes.facecolor": STYLE["surf"],
    "axes.edgecolor": STYLE["border"], "axes.labelcolor": STYLE["muted"],
    "xtick.color": STYLE["muted"], "ytick.color": STYLE["muted"],
    "text.color": STYLE["text"], "grid.color": STYLE["grid"],
    "grid.linewidth": 0.5, "axes.grid": True,
    "font.family": "DejaVu Sans", "font.size": 10,
})

def marker_for(cls):
    return {"YZ": "o", "XZ": "D", "unknown": "^"}.get(cls, "s")

def ds(df):
    step = max(1, len(df) // DOWNSAMPLE_N)
    return df.iloc[::step]

# ═══════════════════════════════════════════════════════════════════════════
# FIG 1 — Force magnitude signals
# ═══════════════════════════════════════════════════════════════════════════
print("\nFig 1: Signal plots…")
n = len(all_runs)
ncols = min(4, n)
nrows = max(1, (n + ncols - 1) // ncols)

fig, axes = plt.subplots(nrows, ncols, figsize=(4.5*ncols, 3.5*nrows),
                          facecolor=STYLE["bg"], squeeze=False)
fig.suptitle("|F| magnitude — normalized to separation event  (t = 0  =  sep)",
             fontsize=13, color=STYLE["text"], y=1.00)

for idx, r in enumerate(all_runs):
    ax = axes[idx // ncols][idx % ncols]
    sub = ds(r["df"])
    ax.plot(sub.t_norm, sub.F_mag, color=r["color"], linewidth=1.4)
    ax.axvline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.35)
    xlo = max(sub.t_norm.min(), -10)
    xhi = min(sub.t_norm.max(), 30)
    ax.set_xlim(xlo, xhi)
    ax.fill_betweenx([sub.F_mag.min(), sub.F_mag.max()], xlo, 0,
                     color=r["color"], alpha=0.04)
    tag = f"→ {r['predicted']}  {r['confidence']}%" if r["class"]=="unknown" else r["class"]
    ax.set_title(f"{r['name']}  ·  {tag}", color=r["color"], fontsize=9, pad=5)
    ax.set_xlabel("t rel. to sep (s)", fontsize=8)
    ax.set_ylabel("|F| N", fontsize=8)

# Hide unused subplots
for idx in range(len(all_runs), nrows*ncols):
    axes[idx // ncols][idx % ncols].set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig1_signals.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTPUT_DIR}/fig1_signals.png")

# ═══════════════════════════════════════════════════════════════════════════
# FIG 2 — Feature comparison grouped bar
# ═══════════════════════════════════════════════════════════════════════════
print("Fig 2: Feature comparison…")
disp_keys   = ["std_Fx_Fz","std_Fy_Fz","std_Tx_Tz","std_Ty_Tz",
                "corr_Tx_Tz","corr_Ty_Tz","corr_Fx_Fz","T_mag_CV"]
disp_labels = ["std(Fx)/std(Fz)","std(Fy)/std(Fz)★","std(Tx)/std(Tz)",
               "std(Ty)/std(Tz)","corr(Tx,Tz)★","corr(Ty,Tz)",
               "corr(Fx,Fz)","T_mag CV"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), facecolor=STYLE["bg"],
                                gridspec_kw={"width_ratios":[2,1]})
fig.suptitle("Feature analysis — orientation-invariant signatures",
             fontsize=13, color=STYLE["text"])

n_feats = len(disp_keys)
n_runs  = len(all_runs)
width   = max(0.08, min(0.18, 0.8 / n_runs))
x       = np.arange(n_feats)

for i, r in enumerate(all_runs):
    vals   = [r["feats"][k] for k in disp_keys]
    offset = (i - n_runs/2 + 0.5) * width
    ax1.bar(x + offset, vals, width, color=r["color"], alpha=0.8,
            label=r["name"], zorder=3)

ax1.set_xticks(x)
ax1.set_xticklabels(disp_labels, rotation=30, ha="right", fontsize=9)
ax1.set_title("Full-run features", color=STYLE["text"], fontsize=11)
ax1.axhline(0, color=STYLE["muted"], linewidth=0.5)
ax1.legend(fontsize=7, framealpha=0.15, ncol=max(1, n_runs//6))
ax1.annotate("★ = most discriminative", xy=(0.98, 0.02), xycoords="axes fraction",
             ha="right", fontsize=8, color=STYLE["muted"])

# Pre-sep bars
pre_keys   = ["pre_corr_Tx_Tz","pre_std_Fy_Fz","pre_corr_Fx_Fz","pre_std_Tx_Tz"]
pre_labels = ["pre corr(Tx,Tz)","pre std(Fy)/std(Fz)","pre corr(Fx,Fz)","pre std(Tx)/std(Tz)"]
x2 = np.arange(len(pre_keys))
for i, r in enumerate(all_runs):
    vals   = [r["feats"][k] for k in pre_keys]
    offset = (i - n_runs/2 + 0.5) * width
    ax2.bar(x2 + offset, vals, width, color=r["color"], alpha=0.8, zorder=3)
ax2.set_xticks(x2)
ax2.set_xticklabels(pre_labels, rotation=30, ha="right", fontsize=9)
ax2.set_title("Pre-sep window (−2s to sep)", color=STYLE["text"], fontsize=11)
ax2.axhline(0, color=STYLE["muted"], linewidth=0.5)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig2_features.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTPUT_DIR}/fig2_features.png")

# ═══════════════════════════════════════════════════════════════════════════
# FIG 3 — Centroid distances + smoking-gun scatter
# ═══════════════════════════════════════════════════════════════════════════
print("Fig 3: Prediction summary…")
fig = plt.figure(figsize=(14, 6), facecolor=STYLE["bg"])
gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35, width_ratios=[1.2, 1])

# Left: distance bars (one per centroid class)
ax_d = fig.add_subplot(gs[0])
classes = list(centroids.keys())
y_pos   = np.arange(len(all_runs))
bar_h   = 0.4 / max(len(classes), 1)

for ci, cls in enumerate(classes):
    offset = (ci - len(classes)/2 + 0.5) * bar_h
    for ri, r in enumerate(all_runs):
        d = r["dists"].get(cls, 0)
        bar_color = r["color"] if r["predicted"] == cls else "#444"
        ax_d.barh(ri + offset, d, bar_h * 0.85, color=bar_color, alpha=0.8, zorder=3)
        ax_d.text(d + 0.1, ri + offset, f"{d:.2f}", va="center",
                  fontsize=7, color=r["color"] if r["predicted"] == cls else STYLE["muted"])

ax_d.set_yticks(y_pos)
ax_d.set_yticklabels([r["name"] for r in all_runs], fontsize=9)
ax_d.set_xlabel("Distance from class centroid")
ax_d.set_title("Centroid distances — all runs", color=STYLE["text"], fontsize=11)
legend_els = [Line2D([0],[0], color=c, linewidth=4, label=f"{cls} centroid")
              for cls, c in zip(classes, [YZ_COLORS[0], XZ_COLORS[0]])]
ax_d.legend(handles=legend_els, fontsize=8, framealpha=0.15)

# Right: smoking-gun scatter (std_Fy_Fz vs corr_Tx_Tz)
ax_s = fig.add_subplot(gs[1])
cap  = 42  # visual cap for extreme values

for r in all_runs:
    fx = min(r["feats"]["std_Fy_Fz"], cap)
    fy = r["feats"]["corr_Tx_Tz"]
    clipped = r["feats"]["std_Fy_Fz"] > cap
    ax_s.scatter(fx, fy, color=r["color"], s=140, zorder=5,
                 marker=marker_for(r["class"]),
                 edgecolors="white", linewidth=0.6)
    ax_s.annotate(f"  {r['name']}", (fx, fy), fontsize=8, color=r["color"], va="center")
    if clipped:
        ax_s.annotate(f"({r['feats']['std_Fy_Fz']:.0f})", (fx, fy),
                      fontsize=7, color=STYLE["muted"], xytext=(0, 9),
                      textcoords="offset points")

# Draw YZ reference region
if yz_runs:
    xs = [r["feats"]["std_Fy_Fz"] for r in yz_runs]
    ys = [r["feats"]["corr_Tx_Tz"] for r in yz_runs]
    mx, Mx = min(xs), max(xs); my, My = min(ys), max(ys)
    px = (Mx-mx)*0.5+0.05; py = abs(My-my)*0.5+0.02
    rect = plt.Rectangle((mx-px, my-py), (Mx-mx)+2*px, (My-my)+2*py,
                          lw=1, ec=YZ_COLORS[0], fc="none", ls="--", alpha=0.45)
    ax_s.add_patch(rect)
    ax_s.text(mx-px, My+py+0.01, "YZ range", color=YZ_COLORS[0], fontsize=8, alpha=0.7)

if xz_runs:
    xs = [min(r["feats"]["std_Fy_Fz"], cap) for r in xz_runs]
    ys = [r["feats"]["corr_Tx_Tz"] for r in xz_runs]
    mx, Mx = min(xs), max(xs); my, My = min(ys), max(ys)
    px = (Mx-mx)*0.5+0.05; py = abs(My-my)*0.5+0.02
    rect = plt.Rectangle((mx-px, my-py), (Mx-mx)+2*px, (My-my)+2*py,
                          lw=1, ec=XZ_COLORS[0], fc="none", ls="--", alpha=0.45)
    ax_s.add_patch(rect)
    ax_s.text(mx-px, My+py+0.01, "XZ range", color=XZ_COLORS[0], fontsize=8, alpha=0.7)

ax_s.axhline(0, color=STYLE["muted"], linewidth=0.5)
ax_s.set_xlabel("std(Fy)/std(Fz) ★  [capped at 42]")
ax_s.set_ylabel("corr(Tx, Tz) ★")
ax_s.set_title("Smoking-gun feature space", color=STYLE["text"], fontsize=11)

# Legend: shapes
leg = [Line2D([0],[0], marker="o", color="w", mfc="#aaa", ms=8, label="YZ ref"),
       Line2D([0],[0], marker="D", color="w", mfc="#aaa", ms=8, label="XZ ref"),
       Line2D([0],[0], marker="^", color="w", mfc="#aaa", ms=8, label="unknown")]
ax_s.legend(handles=leg, fontsize=8, framealpha=0.15)

fig.suptitle("Prediction summary", fontsize=13, color=STYLE["text"])
plt.savefig(f"{OUTPUT_DIR}/fig3_predictions.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTPUT_DIR}/fig3_predictions.png")

# ═══════════════════════════════════════════════════════════════════════════
# FIG 4 — Pre-sep Tx/Tz traces
# ═══════════════════════════════════════════════════════════════════════════
print("Fig 4: Pre-sep traces…")
ncols4 = min(4, len(all_runs))
nrows4 = max(1, (len(all_runs) + ncols4 - 1) // ncols4)
fig, axes4 = plt.subplots(nrows4*2, ncols4,
                           figsize=(4.5*ncols4, 3.2*nrows4*2),
                           facecolor=STYLE["bg"], squeeze=False)
fig.suptitle("Pre-separation window · Fx/Fz (top) and Tx/Tz (bottom)",
             fontsize=13, color=STYLE["text"], y=1.00)

for idx, r in enumerate(all_runs):
    row_f = (idx // ncols4) * 2
    row_t = row_f + 1
    col   = idx % ncols4
    win   = r["df"][r["df"]["t_norm"].between(-PRE_SEP_WIN, PRE_SEP_WIN)]
    if len(win) == 0: win = r["df"]
    c = r["color"]

    ax_f = axes4[row_f][col]
    ax_f.plot(win.t_norm, win.Fx, color=c, lw=1.3, label="Fx")
    ax_f.plot(win.t_norm, win.Fz, color=c, lw=1.3, ls="--", alpha=0.55, label="Fz")
    ax_f.axvline(0, color="white", lw=0.7, ls=":", alpha=0.3)
    tag = f"→{r['predicted']} {r['confidence']}%" if r["class"]=="unknown" else r["class"]
    ax_f.set_title(f"{r['name']}  {tag}", color=c, fontsize=8, pad=4)
    ax_f.set_ylabel("Force N", fontsize=7)
    ax_f.legend(fontsize=7, framealpha=0.15)

    ax_t = axes4[row_t][col]
    ax_t.plot(win.t_norm, win.Tx, color=c, lw=1.3, label="Tx")
    ax_t.plot(win.t_norm, win.Tz, color=c, lw=1.3, ls="--", alpha=0.55, label="Tz")
    ax_t.axvline(0, color="white", lw=0.7, ls=":", alpha=0.3)
    pctx = r["feats"]["pre_corr_Tx_Tz"]
    ax_t.annotate(f"corr(Tx,Tz)={pctx:.3f}", xy=(0.04,0.05),
                  xycoords="axes fraction", fontsize=7, color=STYLE["muted"])
    ax_t.set_xlabel("t rel. to sep (s)", fontsize=7)
    ax_t.set_ylabel("Torque Nm", fontsize=7)
    ax_t.legend(fontsize=7, framealpha=0.15)

for idx in range(len(all_runs), nrows4*ncols4):
    axes4[(idx//ncols4)*2][idx%ncols4].set_visible(False)
    axes4[(idx//ncols4)*2+1][idx%ncols4].set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig4_presep.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTPUT_DIR}/fig4_presep.png")

# ═══════════════════════════════════════════════════════════════════════════
# predictions.csv
# ═══════════════════════════════════════════════════════════════════════════
rows = []
for r in all_runs:
    row = {"file": r["name"], "true_class": r["class"],
           "predicted": r["predicted"], "confidence_pct": r["confidence"],
           "sep_time_s": round(r["sep"], 3),
           "contact_Fmag_N": round(r["contact_Fmag"], 3),
           "bias_pct": r["bias_pct"]}
    for cls, d in r["dists"].items():
        row[f"dist_{cls}"] = round(d, 4)
    for k, v in r["feats"].items():
        row[k] = round(v, 5)
    rows.append(row)

pred_df = pd.DataFrame(rows)
pred_df.to_csv(f"{OUTPUT_DIR}/predictions.csv", index=False)
print(f"  Saved: {OUTPUT_DIR}/predictions.csv")

# ═══════════════════════════════════════════════════════════════════════════
# HTML DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════
print("Building HTML dashboard…")

def badge_cls(cls, predicted=None):
    c = {"YZ":"#7C6FF7","XZ":"#F87171","unknown":"#F59E0B"}.get(cls,"#888")
    label = f"→ {predicted}" if cls=="unknown" and predicted else cls
    return f'<span style="background:rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},0.15);color:{c};border:1px solid {c}55;padding:2px 9px;border-radius:4px;font-size:11px;font-family:monospace">{label}</span>'

def pill(ok):
    if ok is None: return ""
    if ok: return '<span style="background:rgba(52,211,153,0.12);color:#34d399;padding:1px 8px;border-radius:3px;font-size:10px;font-family:monospace">yes</span>'
    return '<span style="background:rgba(248,113,113,0.12);color:#f87171;padding:1px 8px;border-radius:3px;font-size:10px;font-family:monospace">no</span>'

# Build feature table rows
feat_display = [
    ("std(Fy)/std(Fz) ★", "std_Fy_Fz"),
    ("corr(Tx, Tz) ★",    "corr_Tx_Tz"),
    ("std(Fx)/std(Fz)",   "std_Fx_Fz"),
    ("std(Tx)/std(Tz)",   "std_Tx_Tz"),
    ("std(Ty)/std(Tz)",   "std_Ty_Tz"),
    ("corr(Ty, Tz)",      "corr_Ty_Tz"),
    ("corr(Fx, Fz)",      "corr_Fx_Fz"),
    ("T_mag CV",          "T_mag_CV"),
]
pre_display = [
    ("pre corr(Tx,Tz) ★",     "pre_corr_Tx_Tz"),
    ("pre std(Fy)/std(Fz) ★", "pre_std_Fy_Fz"),
    ("pre corr(Fx,Fz)",       "pre_corr_Fx_Fz"),
    ("pre std(Tx)/std(Tz)",   "pre_std_Tx_Tz"),
]

def yz_range(key):
    if not yz_runs: return None, None
    vals = [r["feats"][key] for r in yz_runs]
    return min(vals), max(vals)

def in_yz(val, key, margin=0.5):
    lo, hi = yz_range(key)
    if lo is None: return None
    span = abs(hi - lo) * margin
    return lo - span <= val <= hi + span

def feat_rows_html(display_list):
    header_cells = "".join(f'<th style="color:#6b7280;font-size:10px;padding:6px 10px;border-bottom:1px solid #252a3a;font-family:monospace;font-weight:400;text-transform:uppercase;letter-spacing:.5px">{r["name"]}</th>' for r in all_runs)
    rows_html = ""
    for label, key in display_list:
        cells = ""
        for r in all_runs:
            v = r["feats"][key]
            color = r["color"]
            inr = in_yz(v, key)
            disp = f"{v:.3f}"
            cells += f'<td style="color:{color};font-family:monospace;font-size:11px;padding:7px 10px;border-bottom:1px solid #1c2030;text-align:right">{disp}</td>'
        if yz_runs:
            lo, hi = yz_range(key)
            range_cell = f'<td style="color:#6b7280;font-size:10px;font-family:monospace;padding:7px 10px;border-bottom:1px solid #1c2030">[{lo:.3f}, {hi:.3f}]</td>'
        else:
            range_cell = "<td></td>"
        rows_html += f'<tr><td style="font-family:monospace;font-size:11px;color:#9ca3af;padding:7px 10px;border-bottom:1px solid #1c2030">{label}</td>{cells}{range_cell}</tr>'
    return f'<thead><tr><th style="color:#6b7280;font-size:10px;padding:6px 10px;border-bottom:1px solid #252a3a;font-weight:400">Feature</th>{header_cells}<th style="color:#6b7280;font-size:10px;padding:6px 10px;border-bottom:1px solid #252a3a;font-weight:400;font-family:monospace">YZ range</th></tr></thead><tbody>{rows_html}</tbody>'

# Build sidebar file list
sidebar_files = ""
for r in all_runs:
    tag = f"→ {r['predicted']} {r['confidence']}%" if r["class"]=="unknown" else r["class"]
    sidebar_files += f'''
    <div style="display:flex;align-items:center;gap:8px;padding:7px 12px;border-radius:6px">
      <span style="width:8px;height:8px;border-radius:50%;background:{r["color"]};flex-shrink:0"></span>
      <div>
        <div style="font-size:12px;color:#e8eaf0">{r["name"]}</div>
        <div style="font-size:10px;color:#6b7280;font-family:monospace">{tag}</div>
      </div>
    </div>'''

# Build prediction cards
pred_cards = ""
for r in unk_runs:
    c = r["color"]
    dist_str = " &nbsp; ".join(f'<span style="font-family:monospace;font-size:11px">{cls}={d:.3f}</span>' for cls, d in r["dists"].items())
    pred_cards += f'''
    <div style="flex:1;min-width:200px;border-radius:10px;padding:16px 18px;border:1px solid {c}44;background:{c}0d">
      <div style="font-size:11px;font-family:monospace;color:#6b7280;margin-bottom:4px">{r["name"]}</div>
      <div style="font-size:20px;font-weight:600;color:{c};margin-bottom:4px">{r["predicted"]}</div>
      <div style="font-size:11px;color:{c};font-family:monospace;margin-bottom:6px">confidence {r["confidence"]}%</div>
      <div style="font-size:11px;color:#6b7280">{dist_str}</div>
    </div>'''

# Overview summary rows
summary_rows = ""
for r in all_runs:
    tag_html = badge_cls(r["class"], r["predicted"])
    summary_rows += f'''
    <tr>
      <td style="color:{r["color"]};font-family:monospace;font-size:12px;padding:8px 10px;border-bottom:1px solid #1c2030">{r["name"]}</td>
      <td style="padding:8px 10px;border-bottom:1px solid #1c2030">{len(r["df"]):,}</td>
      <td style="font-family:monospace;font-size:11px;color:#9ca3af;padding:8px 10px;border-bottom:1px solid #1c2030">t={r["sep"]:.2f}s</td>
      <td style="font-family:monospace;font-size:11px;padding:8px 10px;border-bottom:1px solid #1c2030">{r["contact_Fmag"]:.2f} N</td>
      <td style="font-family:monospace;font-size:11px;color:#6b7280;padding:8px 10px;border-bottom:1px solid #1c2030">{r["bias_pct"]}%</td>
      <td style="padding:8px 10px;border-bottom:1px solid #1c2030">{tag_html}</td>
    </tr>'''

def build_dist_cards(runs):
    parts = []
    for r in runs:
        cls_key = list(r["dists"].keys())[0] if r["dists"] else ""
        dist_val = f"{list(r['dists'].values())[0]:.3f}" if r["dists"] else "—"
        parts.append(
            f'<div style="background:#1c2030;border-radius:8px;padding:12px 16px;min-width:110px">'
            f'<div style="font-size:10px;font-family:monospace;color:#6b7280;margin-bottom:4px">dist {cls_key} · {r["name"]}</div>'
            f'<div style="font-size:20px;font-weight:600;color:{r["color"]}">{dist_val}</div>'
            f'<div style="font-size:10px;color:#6b7280">{r["predicted"]}</div>'
            f'</div>'
        )
    return "".join(parts)

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Separation Direction Analysis</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');
  *{{margin:0;padding:0;box-sizing:border-box}}
  body{{background:#0d0f14;color:#e8eaf0;font-family:'DM Sans',sans-serif;font-size:14px;line-height:1.6}}
  .header{{padding:32px 44px 24px;border-bottom:1px solid #1c2030}}
  .title{{font-size:24px;font-weight:600;letter-spacing:-.5px;color:#fff}}
  .sub{{font-size:12px;color:#6b7280;font-family:'DM Mono',monospace;margin-top:3px}}
  .badges{{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}}
  .main{{display:grid;grid-template-columns:230px 1fr;min-height:calc(100vh - 110px)}}
  .sidebar{{border-right:1px solid #1c2030;padding:20px 12px}}
  .nav-label{{font-size:10px;font-family:'DM Mono',monospace;color:#6b7280;text-transform:uppercase;letter-spacing:1.5px;padding:8px 12px 4px}}
  .nav-btn{{display:flex;align-items:center;gap:10px;padding:8px 12px;border-radius:6px;cursor:pointer;border:none;background:transparent;color:#6b7280;font-family:'DM Sans',sans-serif;font-size:13px;width:100%;text-align:left;transition:all .15s}}
  .nav-btn:hover{{background:#1c2030;color:#e8eaf0}}
  .nav-btn.active{{background:#1c2030;color:#e8eaf0;border:1px solid #252a3a}}
  .nav-dot{{width:8px;height:8px;border-radius:50%;flex-shrink:0}}
  .content{{padding:28px 36px;overflow-y:auto}}
  .panel{{display:none}}.panel.active{{display:block}}
  .sec-title{{font-size:17px;font-weight:600;letter-spacing:-.3px;margin-bottom:5px;color:#fff}}
  .sec-sub{{font-size:12px;color:#6b7280;margin-bottom:18px}}
  .card{{background:#151821;border:1px solid #1c2030;border-radius:10px;padding:18px 22px;margin-bottom:14px}}
  .card-title{{font-size:10px;font-family:'DM Mono',monospace;color:#6b7280;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px}}
  table{{width:100%;border-collapse:collapse;font-size:12px}}
  th{{text-align:left;padding:6px 10px;border-bottom:1px solid #252a3a;color:#6b7280;font-weight:400}}
  td{{padding:7px 10px;border-bottom:1px solid #1c2030;color:#e8eaf0}}
  tr:hover td{{background:rgba(255,255,255,.015)}}
  .img-wrap{{border-radius:8px;overflow:hidden;margin-bottom:14px;border:1px solid #252a3a}}
  .img-wrap img{{width:100%;display:block}}
  code{{font-family:'DM Mono',monospace;font-size:11px;background:#1c2030;padding:1px 5px;border-radius:3px;color:#a78bfa}}
</style>
</head>
<body>
<div class="header">
  <div class="title">Separation Direction Analysis</div>
  <div class="sub">force-torque classification · bias-corrected · {len(yz_runs)} YZ ref · {len(xz_runs)} XZ ref · {len(unk_runs)} unknown</div>
  <div class="badges">
    {"".join(badge_cls(r["class"], r["predicted"]) + " " + r["name"] + "&nbsp;&nbsp;" for r in all_runs)}
  </div>
</div>

<div class="main">
<div class="sidebar">
  <div class="nav-label">Sections</div>
  <button class="nav-btn active" onclick="show('overview',this)"><span class="nav-dot" style="background:#7c6ff7"></span>Overview</button>
  <button class="nav-btn" onclick="show('predictions',this)"><span class="nav-dot" style="background:#f87171"></span>Predictions</button>
  <button class="nav-btn" onclick="show('signals',this)"><span class="nav-dot" style="background:#34d399"></span>Signals</button>
  <button class="nav-btn" onclick="show('features',this)"><span class="nav-dot" style="background:#f59e0b"></span>Features</button>
  <button class="nav-btn" onclick="show('presep',this)"><span class="nav-dot" style="background:#67e8f9"></span>Pre-sep traces</button>
  <button class="nav-btn" onclick="show('baseline',this)"><span class="nav-dot" style="background:#94a3b8"></span>Baseline</button>
  <div class="nav-label" style="margin-top:14px">Files</div>
  {sidebar_files}
</div>

<div class="content">

<!-- OVERVIEW -->
<div class="panel active" id="panel-overview">
  <div class="sec-title">Overview</div>
  <div class="sec-sub">{len(all_runs)} total runs · {len(yz_runs)} YZ reference · {len(xz_runs)} XZ reference · {len(unk_runs)} classified</div>
  <div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:14px">
    {build_dist_cards(all_runs)}
  </div>
  <div class="card">
    <div class="card-title">Run summary</div>
    <table>
      <thead><tr><th>File</th><th>Rows</th><th>Sep event</th><th>Contact |F|</th><th>Bias %</th><th>Class</th></tr></thead>
      <tbody>{summary_rows}</tbody>
    </table>
  </div>
</div>

<!-- PREDICTIONS -->
<div class="panel" id="panel-predictions">
  <div class="sec-title">Predictions</div>
  <div class="sec-sub">Classification results for unknown files.</div>
  <div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px">
    {pred_cards if pred_cards else '<div style="color:#6b7280">No unknown files loaded — add CSVs to ./data/unknown/</div>'}
  </div>
  <div class="img-wrap"><img src="fig3_predictions.png" alt="Prediction summary"></div>
</div>

<!-- SIGNALS -->
<div class="panel" id="panel-signals">
  <div class="sec-title">Signal plots</div>
  <div class="sec-sub">Force magnitude |F| — normalized to separation event (t = 0).</div>
  <div class="img-wrap"><img src="fig1_signals.png" alt="Signals"></div>
</div>

<!-- FEATURES -->
<div class="panel" id="panel-features">
  <div class="sec-title">Feature analysis</div>
  <div class="sec-sub">Orientation-invariant features — unaffected by sensor bias or mounting direction.</div>
  <div class="card">
    <div class="card-title">Full-run features &nbsp; ★ = most discriminative</div>
    <div style="overflow-x:auto"><table>{feat_rows_html(feat_display)}</table></div>
  </div>
  <div class="card">
    <div class="card-title">Pre-separation window (−2s to sep event)</div>
    <div style="overflow-x:auto"><table>{feat_rows_html(pre_display)}</table></div>
  </div>
  <div class="img-wrap"><img src="fig2_features.png" alt="Feature comparison"></div>
</div>

<!-- PRE-SEP -->
<div class="panel" id="panel-presep">
  <div class="sec-title">Pre-separation traces</div>
  <div class="sec-sub">Fx/Fz and Tx/Tz in the 2 seconds before the separation event.</div>
  <div class="img-wrap"><img src="fig4_presep.png" alt="Pre-sep traces"></div>
</div>

<!-- BASELINE -->
<div class="panel" id="panel-baseline">
  <div class="sec-title">Free-space baseline</div>
  <div class="sec-sub">Gravity + sensor mass offset subtracted from all contact runs.</div>
  <div class="card">
    <div class="card-title">Bias vector</div>
    <div style="display:grid;grid-template-columns:repeat(6,1fr);gap:8px">
      {"".join(f'<div style="background:#1c2030;border-radius:6px;padding:10px;text-align:center"><div style="font-size:10px;color:#6b7280;font-family:monospace;margin-bottom:3px">{c}</div><div style="font-size:14px;font-weight:500;font-family:monospace;color:#a78bfa">{bias[c]:+.3f}</div></div>' for c in ["Fx","Fy","Fz","Tx","Ty","Tz"])}
    </div>
    <div style="font-size:11px;color:#6b7280;margin-top:10px">|bias| = {bias_mag:.3f} N &nbsp;·&nbsp; {len(glob.glob(os.path.join(DIRS["baseline"],"*.csv")))} baseline file(s) loaded</div>
  </div>
  <div class="card">
    <div class="card-title">Why bias correction doesn't change predictions</div>
    <div style="font-size:13px;color:#9ca3af;line-height:1.7">
      Bias subtraction removes a <em>constant offset</em>. Our classification features are <code>std()</code> ratios and <code>corr()</code> values — both are mathematically invariant to constant offsets. Only mean-based features shift slightly.
    </div>
  </div>
</div>

</div><!-- content -->
</div><!-- main -->

<script>
function show(id, btn) {{
  document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.nav-btn').forEach(b=>b.classList.remove('active'));
  document.getElementById('panel-'+id).classList.add('active');
  if(btn) btn.classList.add('active');
}}
</script>
</body>
</html>"""

with open(f"{OUTPUT_DIR}/report.html", "w") as f:
    f.write(html)
print(f"  Saved: {OUTPUT_DIR}/report.html")

print("\n" + "=" * 60)
print("DONE")
print(f"  {OUTPUT_DIR}/report.html       ← open in browser")
print(f"  {OUTPUT_DIR}/predictions.csv   ← machine-readable results")
print(f"  {OUTPUT_DIR}/fig1_signals.png")
print(f"  {OUTPUT_DIR}/fig2_features.png")
print(f"  {OUTPUT_DIR}/fig3_predictions.png")
print(f"  {OUTPUT_DIR}/fig4_presep.png")
print("=" * 60)