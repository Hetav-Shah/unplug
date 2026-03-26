#!/usr/bin/env python3
"""
plot_data.py

Reads the CSV saved by franka_full_game.py and plots everything.

Usage:
    python3 plot_data.py franka_run_1234567890.csv

CSV columns expected:
    timestamp_s, state, pos_z_m, rot_angle_deg,
    fz_N, tz_Nm, v_z_mms, v_rz_degs, event
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker

# ── Load ─────────────────────────────────────────────────────────────────

if len(sys.argv) < 2:
    print("Usage: python3 plot_data.py 2.csv")
    sys.exit(1)

csv_file = sys.argv[1]
df = pd.read_csv(csv_file)

# Separate event rows from normal rows
events = df[df['event'].notna() & (df['event'] != "")].copy()
normal = df[df['event'].isna() | (df['event'] == "")].copy()

print(f"\nLoaded : {csv_file}")
print(f"Rows   : {len(df)}")
print(f"Events : {len(events)}")
print(f"Duration: {df['timestamp_s'].max():.2f} s")
print(f"States seen: {df['state'].unique().tolist()}")
print(f"\nEvents log:")
for _, row in events.iterrows():
    print(f"  t={row['timestamp_s']:.2f}s  [{row['state']}]  {row['event']}")

t = df['timestamp_s']

# State colors
STATE_COLORS = {"PULL": "#1D9E75", "CCW": "#7F77DD", "CW": "#D85A30"}

def shade_states(ax):
    """Shade background by state and mark event lines."""
    prev_state = None
    start_t    = None
    for _, row in df.iterrows():
        if row['state'] != prev_state:
            if prev_state is not None:
                ax.axvspan(start_t, row['timestamp_s'],
                           alpha=0.07,
                           color=STATE_COLORS.get(prev_state, 'gray'))
            start_t    = row['timestamp_s']
            prev_state = row['state']
    if prev_state is not None:
        ax.axvspan(start_t, t.max(),
                   alpha=0.07,
                   color=STATE_COLORS.get(prev_state, 'gray'))
    # Mark events as vertical dashed lines
    for _, row in events.iterrows():
        ax.axvline(row['timestamp_s'],
                   color='red', alpha=0.4,
                   linewidth=0.8, linestyle='--')

patches = [mpatches.Patch(color=c, alpha=0.4, label=s)
           for s, c in STATE_COLORS.items()]

# ═════════════════════════════════════════════════════════════════════════
# PLOT 1 — Full overview: position, rotation, forces, velocities
# ═════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
fig.suptitle(f"Full run overview — {csv_file}", fontsize=13, fontweight='bold')

# 1a — Z position
ax = axes[0]
ax.plot(t, df['pos_z_m'] * 1000, color='#1D9E75', linewidth=1.2)
shade_states(ax)
ax.set_ylabel('Z position (mm)')
ax.legend(handles=patches, loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_title('Z position over time')

# 1b — Rotation angle
ax = axes[1]
ax.plot(t, df['rot_angle_deg'], color='#7F77DD', linewidth=1.2)
shade_states(ax)
ax.set_ylabel('Rotation (deg)')
ax.grid(True, alpha=0.3)
ax.set_title('Rotation angle over time')

# 1c — Forces (fz)
ax = axes[2]
ax.plot(t, df['fz_N'], color='#D85A30', linewidth=1.2, label='fz')
ax.plot(t, df['tz_Nm'], color='#7F77DD', linewidth=1.0,
        alpha=0.7, linestyle='--', label='tz')
ax.axhline(0, color='black', linewidth=0.5, linestyle=':')
shade_states(ax)
ax.set_ylabel('Force / Torque')
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_title('fz (red) and tz (purple dashed)')

# 1d — Commanded velocities
ax = axes[3]
ax.plot(t, df['v_z_mms'], color='#1D9E75', linewidth=1.2, label='v_z (mm/s)')
ax.plot(t, df['v_rz_degs'], color='#7F77DD', linewidth=1.2,
        linestyle='--', label='v_rz (deg/s)')
ax.axhline(0, color='black', linewidth=0.5, linestyle=':')
shade_states(ax)
ax.set_ylabel('Velocity')
ax.set_xlabel('Time (s)')
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_title('Commanded velocities — v_z and v_rz')

plt.tight_layout()
plt.savefig("plot1_overview.png", dpi=150, bbox_inches='tight')
print("\nSaved: plot1_overview.png")

# ═════════════════════════════════════════════════════════════════════════
# PLOT 2 — Admittance validation: fz vs v_z scatter
# ═════════════════════════════════════════════════════════════════════════

pull_data = normal[normal['state'] == 'PULL'].copy()
pull_data = pull_data[pull_data['v_z_mms'] != 0]   # skip zero-velocity rows

if len(pull_data) > 10:
    from scipy import stats
    slope, intercept, r, p, _ = stats.linregress(
        pull_data['fz_N'], pull_data['v_z_mms'])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(pull_data['fz_N'], pull_data['v_z_mms'],
               alpha=0.3, s=8, color='#1D9E75', label='data points')

    import numpy as np
    fz_range = np.linspace(pull_data['fz_N'].min(),
                            pull_data['fz_N'].max(), 100)
    ax.plot(fz_range, slope * fz_range + intercept,
            color='#D85A30', linewidth=2,
            label=f'linear fit  R²={r**2:.4f}')

    ax.axhline(0, color='black', linewidth=0.5, linestyle=':')
    ax.axvline(0, color='black', linewidth=0.5, linestyle=':')
    ax.set_xlabel('fz tared (N)')
    ax.set_ylabel('v_z (mm/s)')
    ax.set_title('Admittance validation — fz vs v_z\n'
                 'Should be a straight line if admittance is working correctly')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plot2_admittance.png", dpi=150, bbox_inches='tight')
    print("Saved: plot2_admittance.png")
    print(f"  Admittance R² = {r**2:.4f}  "
          f"({'good' if r**2 > 0.8 else 'check gains'})")
else:
    print("Not enough pull data for admittance scatter plot")

# ═════════════════════════════════════════════════════════════════════════
# PLOT 3 — State sequence bar
# ═════════════════════════════════════════════════════════════════════════

state_blocks = []
prev_s  = None
start_t = 0
for _, row in df.iterrows():
    if row['state'] != prev_s:
        if prev_s is not None:
            state_blocks.append((prev_s, start_t,
                                  row['timestamp_s'] - start_t))
        start_t = row['timestamp_s']
        prev_s  = row['state']
if prev_s:
    state_blocks.append((prev_s, start_t,
                          df['timestamp_s'].max() - start_t))

fig, ax = plt.subplots(figsize=(15, 2.5))
fig.suptitle("State sequence", fontsize=12, fontweight='bold')

for state, start, dur in state_blocks:
    color = STATE_COLORS.get(state, 'gray')
    ax.barh(0, dur, left=start, height=0.5,
            color=color, alpha=0.85, edgecolor='white', linewidth=0.5)
    if dur > (df['timestamp_s'].max() * 0.02):   # only label if wide enough
        ax.text(start + dur / 2, 0,
                f"{state}\n{dur:.1f}s",
                ha='center', va='center',
                fontsize=8, color='white', fontweight='bold')

ax.set_yticks([])
ax.set_xlabel('Time (s)')
ax.set_xlim(0, df['timestamp_s'].max())
ax.grid(True, axis='x', alpha=0.3)
ax.legend(handles=patches, loc='upper right', fontsize=8)

# Mark events
for _, row in events.iterrows():
    ax.axvline(row['timestamp_s'], color='red',
               alpha=0.6, linewidth=1, linestyle='--')

plt.tight_layout()
plt.savefig("plot3_states.png", dpi=150, bbox_inches='tight')
print("Saved: plot3_states.png")

# ═════════════════════════════════════════════════════════════════════════
# PLOT 4 — fz and tz closeup with event markers and labels
# ═════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
fig.suptitle("Force and torque closeup with events", fontsize=12, fontweight='bold')

ax = axes[0]
ax.plot(t, df['fz_N'], color='#D85A30', linewidth=1.2)
ax.fill_between(t, df['fz_N'], 0,
                where=(df['fz_N'] < 0),
                alpha=0.15, color='#D85A30')
shade_states(ax)
ax.axhline(0, color='black', linewidth=0.5, linestyle=':')
ax.set_ylabel('fz tared (N)')
ax.set_title('Z force — goes negative when object resists pull')
ax.grid(True, alpha=0.3)
for _, row in events.iterrows():
    ax.annotate(row['event'][:30],
                xy=(row['timestamp_s'], row['fz_N']),
                xytext=(0, 12), textcoords='offset points',
                fontsize=6, color='red', rotation=45,
                arrowprops=dict(arrowstyle='->', color='red', lw=0.5))

ax = axes[1]
ax.plot(t, df['tz_Nm'], color='#7F77DD', linewidth=1.2)
ax.fill_between(t, df['tz_Nm'], 0,
                where=(df['tz_Nm'] != 0),
                alpha=0.15, color='#7F77DD')
shade_states(ax)
ax.axhline(0, color='black', linewidth=0.5, linestyle=':')
ax.set_ylabel('tz tared (Nm)')
ax.set_xlabel('Time (s)')
ax.set_title('Z torque — rises when rotation hits a barrier')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plot4_forces_closeup.png", dpi=150, bbox_inches='tight')
print("Saved: plot4_forces_closeup.png")

plt.show()
print("\nDone. All plots saved in the same folder as this script.")