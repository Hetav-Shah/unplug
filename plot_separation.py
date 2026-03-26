"""
Separation Direction — Visualization Script
=============================================
Run: python plot_separation.py
Outputs: 4 PNG figures saved to ./output/
Requires: pandas, numpy, scipy, matplotlib
"""

import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from scipy.signal import butter, filtfilt

os.makedirs("output", exist_ok=True)

# ── Palette ────────────────────────────────────────────────────────────────
C = {
    "y1":  "#7C6FF7",   # purple  — YZ reference
    "y2":  "#34D399",   # teal    — YZ reference
    "34":  "#F59E0B",   # amber   — predicted YZ
    "45":  "#F87171",   # red     — predicted XZ
    "bg":  "#0D0F14",
    "surf":"#151821",
    "grid":"#1C2030",
    "text":"#E8EAF0",
    "muted":"#6B7280",
}
plt.rcParams.update({
    "figure.facecolor": C["bg"], "axes.facecolor": C["surf"],
    "axes.edgecolor": C["grid"], "axes.labelcolor": C["muted"],
    "xtick.color": C["muted"], "ytick.color": C["muted"],
    "text.color": C["text"], "grid.color": C["grid"],
    "grid.linewidth": 0.5, "axes.grid": True,
    "font.family": "DejaVu Sans", "font.size": 10,
})

# ── Config — UPDATE THESE PATHS ────────────────────────────────────────────
FILES = {
    "y1 (YZ ref)":  "y1.csv",
    "y2 (YZ ref)":  "y2.csv",
    "34 (→ YZ)":    "34.csv",
    "45 (→ XZ)":    "45.csv",
}
IC_FILES = ["initial_condition.csv", "Initial_condition__1_.csv"]

# ── Helpers ────────────────────────────────────────────────────────────────
def butter_lp(df):
    dt = df["time"].diff().median(); fs = 1.0 / dt
    b, a = butter(4, min(20 / (0.5 * fs), 0.99), btype="low")
    for col in ["Fx","Fy","Fz","Tx","Ty","Tz"]:
        df[col] = filtfilt(b, a, df[col])
    return df

def compute_bias(ic_files):
    bias = {c: 0.0 for c in ["Fx","Fy","Fz","Tx","Ty","Tz"]}
    count = 0
    for f in ic_files:
        if not os.path.exists(f): continue
        df = pd.read_csv(f).dropna()
        for c in bias: bias[c] += df[c].mean()
        count += 1
    if count: bias = {c: v/count for c,v in bias.items()}
    return bias

def load(path, bias):
    df = pd.read_csv(path).dropna().sort_values("time").reset_index(drop=True)
    df = butter_lp(df)
    for c in ["Fx","Fy","Fz","Tx","Ty","Tz"]: df[c] -= bias[c]
    df["F_mag"] = np.sqrt(df.Fx**2 + df.Fy**2 + df.Fz**2)
    df["T_mag"] = np.sqrt(df.Tx**2 + df.Ty**2 + df.Tz**2)
    df["dF_dt"] = df["F_mag"].diff() / df["time"].diff()
    si = df["dF_dt"].idxmin()
    sep = df.loc[si, "time"]
    df["t_norm"] = df["time"] - sep
    return df, sep

def feats(df):
    pre = df[df["t_norm"].between(-2, 0)]
    if len(pre) < 5: pre = df.iloc[:20]
    return {
        "std_Fx_Fz":    df.Fx.std() / (df.Fz.std()+1e-9),
        "std_Fy_Fz":    df.Fy.std() / (df.Fz.std()+1e-9),
        "std_Tx_Tz":    df.Tx.std() / (df.Tz.std()+1e-9),
        "std_Ty_Tz":    df.Ty.std() / (df.Tz.std()+1e-9),
        "corr_Tx_Tz":   float(df.Tx.corr(df.Tz)),
        "corr_Ty_Tz":   float(df.Ty.corr(df.Tz)),
        "corr_Fx_Fz":   float(df.Fx.corr(df.Fz)),
        "T_mag_CV":     df.T_mag.std() / df.T_mag.mean(),
        "pre_corr_Tx_Tz": float(pre.Tx.corr(pre.Tz)),
        "pre_std_Fy_Fz":  pre.Fy.std() / (pre.Fz.std()+1e-9),
        "pre_corr_Fx_Fz": float(pre.Fx.corr(pre.Fz)),
        "pre_std_Tx_Tz":  pre.Tx.std() / (pre.Tz.std()+1e-9),
    }

# ── Load all data ──────────────────────────────────────────────────────────
print("Loading data…")
bias = compute_bias(IC_FILES)
print(f"  Bias: Fx={bias['Fx']:.3f}  Fy={bias['Fy']:.3f}  Fz={bias['Fz']:.3f}")

data, fdict, seps = {}, {}, {}
cmap = {"y1 (YZ ref)":C["y1"], "y2 (YZ ref)":C["y2"], "34 (→ YZ)":C["34"], "45 (→ XZ)":C["45"]}
short = {"y1 (YZ ref)":"y1","y2 (YZ ref)":"y2","34 (→ YZ)":"34","45 (→ XZ)":"45"}

for label, path in FILES.items():
    if not os.path.exists(path):
        print(f"  MISSING: {path} — skipping")
        continue
    df, sep = load(path, bias)
    data[label] = df
    fdict[label] = feats(df)
    seps[label] = sep
    print(f"  {label}: {len(df)} rows, sep at t={sep:.2f}s")

# Centroid + distances
feat_keys = ["std_Fx_Fz","std_Fy_Fz","std_Tx_Tz","std_Ty_Tz","corr_Tx_Tz","corr_Ty_Tz","corr_Fx_Fz","T_mag_CV"]
yz_labels = [l for l in data if "YZ" in l]
v = {l: np.array([fdict[l][k] for k in feat_keys]) for l in fdict}
if yz_labels:
    centroid = np.mean([v[l] for l in yz_labels], axis=0)
    dists = {l: float(np.linalg.norm(v[l] - centroid)) for l in v}
else:
    dists = {l: 0.0 for l in v}


# ═══════════════════════════════════════════════════════════════════════════
# FIG 1 — Force magnitude signals (normalized to sep event)
# ═══════════════════════════════════════════════════════════════════════════
print("\nFig 1: Signal plots…")
fig, axes = plt.subplots(2, 2, figsize=(14, 8), facecolor=C["bg"])
fig.suptitle("Force magnitude |F| — normalized to separation event (t = 0)",
             fontsize=13, color=C["text"], y=0.98)

for ax, (label, df) in zip(axes.flat, data.items()):
    color = cmap[label]
    step = max(1, len(df)//400)
    sub = df.iloc[::step]
    ax.plot(sub.t_norm, sub.F_mag, color=color, linewidth=1.4, label="|F|")
    ax.axvline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.fill_betweenx([sub.F_mag.min(), sub.F_mag.max()], -2, 0,
                     color=color, alpha=0.05)
    ax.set_xlim(max(sub.t_norm.min(), -10), min(sub.t_norm.max(), 20))
    ax.set_xlabel("time relative to separation (s)")
    ax.set_ylabel("|F| (N)")
    ax.set_title(f"{label}  ·  sep at t={seps[label]:.2f}s  ·  dist={dists[label]:.3f}",
                 color=color, fontsize=10, pad=6)
    ax.text(0.02, 0.98, "← pre-sep | post-sep →", transform=ax.transAxes,
            fontsize=8, color=C["muted"], va="top")

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("output/fig1_signal_plots.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: output/fig1_signal_plots.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 2 — Feature comparison (grouped bar chart)
# ═══════════════════════════════════════════════════════════════════════════
print("Fig 2: Feature bar chart…")
display_feats = ["std_Fx_Fz","std_Fy_Fz","std_Tx_Tz","std_Ty_Tz",
                 "corr_Tx_Tz","corr_Ty_Tz","corr_Fx_Fz","T_mag_CV"]
feat_labels = ["std(Fx)/std(Fz)","std(Fy)/std(Fz)★","std(Tx)/std(Tz)",
               "std(Ty)/std(Tz)","corr(Tx,Tz)★","corr(Ty,Tz)",
               "corr(Fx,Fz)","T_mag CV"]

labels_order = list(data.keys())
colors_order = [cmap[l] for l in labels_order]
n_feats = len(display_feats)
n_files = len(labels_order)
x = np.arange(n_feats)
width = 0.18

fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor=C["bg"],
                         gridspec_kw={"width_ratios":[2,1]})
fig.suptitle("Feature analysis — orientation-invariant signatures",
             fontsize=13, color=C["text"], y=0.99)

ax = axes[0]
for i, (label, color) in enumerate(zip(labels_order, colors_order)):
    vals = [fdict[label][k] for k in display_feats]
    bars = ax.bar(x + i*width - width*1.5, vals, width, color=color,
                  alpha=0.85, label=short[label], zorder=3)

ax.set_xticks(x)
ax.set_xticklabels(feat_labels, rotation=30, ha="right", fontsize=9)
ax.set_ylabel("Feature value")
ax.set_title("Full-run features (all 4 files)", color=C["text"], fontsize=11)
ax.legend(loc="upper left", framealpha=0.2, fontsize=9)
ax.axhline(0, color=C["muted"], linewidth=0.5)
ax.annotate("★ = smoking-gun features", xy=(0.98, 0.02), xycoords="axes fraction",
            ha="right", fontsize=8, color=C["muted"])

# Pre-sep features
pre_feats_k = ["pre_corr_Tx_Tz","pre_std_Fy_Fz","pre_corr_Fx_Fz","pre_std_Tx_Tz"]
pre_labels  = ["pre corr(Tx,Tz)","pre std(Fy)/std(Fz)","pre corr(Fx,Fz)","pre std(Tx)/std(Tz)"]
ax2 = axes[1]
x2 = np.arange(len(pre_feats_k))
for i, (label, color) in enumerate(zip(labels_order, colors_order)):
    vals = [fdict[label][k] for k in pre_feats_k]
    ax2.bar(x2 + i*width - width*1.5, vals, width, color=color, alpha=0.85, zorder=3)
ax2.set_xticks(x2)
ax2.set_xticklabels(pre_labels, rotation=30, ha="right", fontsize=9)
ax2.set_title("Pre-separation window (−2s to sep)", color=C["text"], fontsize=11)
ax2.axhline(0, color=C["muted"], linewidth=0.5)

plt.tight_layout(rect=[0,0,1,0.97])
plt.savefig("output/fig2_feature_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: output/fig2_feature_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 3 — Distance from YZ centroid + prediction summary
# ═══════════════════════════════════════════════════════════════════════════
print("Fig 3: Prediction summary…")
fig = plt.figure(figsize=(13, 6), facecolor=C["bg"])
gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.2, 1], wspace=0.35)

# Left: distance bar
ax_dist = fig.add_subplot(gs[0])
labels_plot = list(dists.keys())
dist_vals = [dists[l] for l in labels_plot]
colors_plot = [cmap[l] for l in labels_plot]
bars = ax_dist.barh(range(len(labels_plot)), dist_vals, color=colors_plot, alpha=0.85, height=0.5, zorder=3)
ax_dist.set_yticks(range(len(labels_plot)))
ax_dist.set_yticklabels([short[l] for l in labels_plot], fontsize=11)
ax_dist.set_xlabel("Distance from YZ centroid (feature space)")
ax_dist.set_title("YZ centroid distance — classification", color=C["text"], fontsize=11, pad=8)
ax_dist.axvline(1.5, color=C["muted"], linewidth=0.8, linestyle="--", alpha=0.5)
ax_dist.text(1.55, -0.5, "decision\nboundary", fontsize=8, color=C["muted"], va="bottom")
for bar, val, lbl in zip(bars, dist_vals, labels_plot):
    ax_dist.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                 f"{val:.3f}", va="center", fontsize=10, color=cmap[lbl], fontweight="bold")

# Right: smoking-gun scatter
ax_sg = fig.add_subplot(gs[1])
feat_x_k, feat_y_k = "std_Fy_Fz", "corr_Tx_Tz"
for label in data:
    fx = min(fdict[label][feat_x_k], 40)  # cap for display
    fy = fdict[label][feat_y_k]
    clipped = fdict[label][feat_x_k] > 40
    ax_sg.scatter(fx, fy, color=cmap[label], s=180, zorder=5,
                  marker="D" if "XZ" in label else "o",
                  edgecolors="white", linewidth=0.8)
    ax_sg.annotate(f"  {short[label]}", (fx, fy), fontsize=10, color=cmap[label],
                   va="center")
    if clipped:
        ax_sg.annotate(f"({fdict[label][feat_x_k]:.0f}→clipped)", (fx, fy),
                       fontsize=7, color=C["muted"], xytext=(0,10), textcoords="offset points")

# YZ region box
yz_xmin = min(fdict[l][feat_x_k] for l in yz_labels)
yz_xmax = max(fdict[l][feat_x_k] for l in yz_labels)
yz_ymin = min(fdict[l][feat_y_k] for l in yz_labels)
yz_ymax = max(fdict[l][feat_y_k] for l in yz_labels)
margin_x = (yz_xmax - yz_xmin) * 0.5 + 0.1
margin_y = abs(yz_ymax - yz_ymin) * 0.5 + 0.02
rect = plt.Rectangle((yz_xmin - margin_x, yz_ymin - margin_y),
                      (yz_xmax - yz_xmin) + 2*margin_x,
                      (yz_ymax - yz_ymin) + 2*margin_y,
                      linewidth=1, edgecolor=C["y1"], facecolor="none",
                      linestyle="--", alpha=0.5, zorder=2)
ax_sg.add_patch(rect)
ax_sg.text(yz_xmin - margin_x + 0.05, yz_ymax + margin_y + 0.01,
           "YZ range", color=C["y1"], fontsize=8, alpha=0.7)
ax_sg.set_xlabel("std(Fy)/std(Fz)  ★  [capped at 40]")
ax_sg.set_ylabel("corr(Tx, Tz)  ★")
ax_sg.set_title("Smoking-gun feature space", color=C["text"], fontsize=11, pad=8)
ax_sg.axhline(0, color=C["muted"], linewidth=0.5)

fig.suptitle("Prediction summary — separation direction classification",
             fontsize=13, color=C["text"], y=1.01)
plt.savefig("output/fig3_predictions.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: output/fig3_predictions.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 4 — Pre-sep Fx/Fz and Tx/Tz time traces (side by side)
# ═══════════════════════════════════════════════════════════════════════════
print("Fig 4: Pre-sep traces…")
fig, axes = plt.subplots(2, 4, figsize=(16, 7), facecolor=C["bg"], sharex=False)
fig.suptitle("Pre-separation window detail (−2s to +2s around sep event)",
             fontsize=13, color=C["text"], y=0.99)

for col, (label, df) in enumerate(data.items()):
    color = cmap[label]
    win = df[df["t_norm"].between(-2, 2)]
    if len(win) == 0: win = df

    # Fx and Fz
    ax = axes[0][col]
    ax.plot(win.t_norm, win.Fx, color=color, linewidth=1.4, label="Fx")
    ax.plot(win.t_norm, win.Fz, color=color, linewidth=1.4, linestyle="--", alpha=0.6, label="Fz")
    ax.axvline(0, color="white", linewidth=0.8, linestyle=":", alpha=0.4)
    ax.set_title(short[label], color=color, fontsize=11)
    if col == 0:
        ax.set_ylabel("Force (N)\n— Fx  -- Fz", fontsize=8)
    ax.legend(fontsize=7, framealpha=0.2)

    # Tx and Tz
    ax2 = axes[1][col]
    ax2.plot(win.t_norm, win.Tx, color=color, linewidth=1.4, label="Tx")
    ax2.plot(win.t_norm, win.Tz, color=color, linewidth=1.4, linestyle="--", alpha=0.6, label="Tz")
    ax2.axvline(0, color="white", linewidth=0.8, linestyle=":", alpha=0.4)
    ax2.set_xlabel("t relative to sep (s)")
    if col == 0:
        ax2.set_ylabel("Torque (Nm)\n— Tx  -- Tz", fontsize=8)
    ax2.legend(fontsize=7, framealpha=0.2)

# Annotation
for col, label in enumerate(data):
    ptx = fdict[label]["pre_corr_Tx_Tz"]
    pfy = fdict[label]["pre_std_Fy_Fz"]
    axes[1][col].annotate(f"corr(Tx,Tz)={ptx:.3f}", xy=(0.05, 0.05),
                          xycoords="axes fraction", fontsize=8, color=C["muted"])

plt.tight_layout(rect=[0,0,1,0.97])
plt.savefig("output/fig4_presep_traces.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: output/fig4_presep_traces.png")

print("\nAll done. Figures in ./output/")
print("  fig1_signal_plots.png    — |F| over time for all 4 runs")
print("  fig2_feature_comparison.png — grouped bar charts of all features")
print("  fig3_predictions.png     — centroid distance + smoking-gun scatter")
print("  fig4_presep_traces.png   — pre-sep Fx/Fz and Tx/Tz traces")
