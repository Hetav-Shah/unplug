#!/usr/bin/env python3
"""
Validation and visualisation script for Franka admittance control CSV data.
Run: python3 analyse_run.py franka_run_XXXXXXXXXX.csv
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

# ==========================================================
# Load data
# ==========================================================

csv_path = sys.argv[1] if len(sys.argv) > 1 else "franka_run.csv"
df = pd.read_csv(csv_path)

print(f"\nLoaded: {csv_path}")
print(f"  Rows     : {len(df)}")
print(f"  Duration : {df['timestamp_s'].max():.2f} s")
print(f"  States   : {df['state'].unique().tolist()}")

# Separate normal rows and event rows
events = df[df['event'] != ""].copy()
normal = df[df['event'] == ""].copy()

# State color map
STATE_COLORS = {"PULL": "#1D9E75", "CCW": "#7F77DD", "CW": "#D85A30"}

# ==========================================================
# VALIDATION CHECK 1 — State sequence
# ==========================================================

print("\n--- Validation 1: State sequence ---")
state_changes = df[df['state'] != df['state'].shift()]['state'].tolist()
print(f"  Sequence: {' → '.join(state_changes)}")

expected_pattern = ["PULL", "CCW", "PULL", "CW"]
matches = all(
    state_changes[i % len(expected_pattern)] == expected_pattern[i % len(expected_pattern)]
    for i in range(min(len(state_changes), 8))
)
print(f"  Matches 1→2→1→3 pattern: {'YES' if matches else 'NO'}")

# ==========================================================
# VALIDATION CHECK 2 — Admittance linearity (Z pull)
# ==========================================================

print("\n--- Validation 2: Z admittance linearity ---")
pull_data = normal[normal['state'] == "PULL"].copy()
pull_data = pull_data.dropna(subset=['fz_N', 'v_z_mms'])

if len(pull_data) > 10:
    slope, intercept, r, p, se = stats.linregress(pull_data['fz_N'], pull_data['v_z_mms'])
    print(f"  v_z = {slope:.6f} * fz + {intercept:.6f}")
    print(f"  R²  = {r**2:.4f}  (1.0 = perfect linear admittance)")
    print(f"  p   = {p:.4e}")
else:
    print("  Not enough pull data points")
    slope, intercept, r = 0, 0, 0

# ==========================================================
# VALIDATION CHECK 3 — Stall detection accuracy
# ==========================================================

print("\n--- Validation 3: Stall detection ---")
stall_events = events[events['event'].str.contains("STALLED", na=False)]
print(f"  Stall events detected: {len(stall_events)}")

for _, row in stall_events.iterrows():
    # Check velocity was near zero at stall
    nearby = df[(df['timestamp_s'] >= row['timestamp_s'] - 0.5) &
                (df['timestamp_s'] <= row['timestamp_s'] + 0.1)]
    avg_vz = nearby['v_z_mms'].abs().mean()
    print(f"  t={row['timestamp_s']:.2f}s  state={row['state']}  "
          f"avg|v_z| near stall = {avg_vz:.4f} mm/s  "
          f"({'near zero - correct' if avg_vz < 0.01 else 'still moving - check'})")

# ==========================================================
# VALIDATION CHECK 4 — Goal reached
# ==========================================================

print("\n--- Validation 4: Goal ---")
goal_rows = events[events['event'].str.contains("GOAL_REACHED", na=False)]
if len(goal_rows) > 0:
    print(f"  Goal reached at t={goal_rows.iloc[0]['timestamp_s']:.2f}s  "
          f"z={goal_rows.iloc[0]['pos_z_m']:.4f} m")
else:
    print("  Goal not reached in this run")

# ==========================================================
# FIGURE 1 — Position + state timeline
# ==========================================================

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
fig.suptitle("Franka admittance control — run overview", fontsize=14, fontweight='bold')

t = df['timestamp_s']

# Shade background by state
for ax in axes:
    state_start = None
    prev_state = None
    for _, row in df.iterrows():
        if row['state'] != prev_state:
            if prev_state is not None:
                ax.axvspan(state_start, row['timestamp_s'],
                           alpha=0.08, color=STATE_COLORS.get(prev_state, 'gray'))
            state_start = row['timestamp_s']
            prev_state = row['state']
    if prev_state:
        ax.axvspan(state_start, t.max(),
                   alpha=0.08, color=STATE_COLORS.get(prev_state, 'gray'))

# Mark stall events on all axes
for _, row in stall_events.iterrows():
    for ax in axes:
        ax.axvline(row['timestamp_s'], color='red', alpha=0.4, linewidth=0.8, linestyle='--')

# Plot 1a: Z position
axes[0].plot(t, df['pos_z_m'] * 1000, color='#1D9E75', linewidth=1.2, label='pos_z (mm)')
axes[0].set_ylabel('Z position (mm)')
axes[0].legend(loc='upper left', fontsize=9)
axes[0].grid(True, alpha=0.3)

# Plot 1b: Rotation angle
axes[1].plot(t, df['rot_angle_deg'], color='#7F77DD', linewidth=1.2, label='rotation (deg)')
axes[1].set_ylabel('Rotation (deg)')
axes[1].legend(loc='upper left', fontsize=9)
axes[1].grid(True, alpha=0.3)

# Plot 1c: State as numeric (1/2/3)
state_num = df['state'].map({'PULL': 1, 'CCW': 2, 'CW': 3})
axes[2].step(t, state_num, color='#444441', linewidth=1.5, where='post')
axes[2].set_yticks([1, 2, 3])
axes[2].set_yticklabels(['PULL', 'CCW', 'CW'])
axes[2].set_ylabel('State')
axes[2].set_xlabel('Time (s)')
axes[2].grid(True, alpha=0.3)

# Legend for state shading
patches = [mpatches.Patch(color=c, alpha=0.3, label=s)
           for s, c in STATE_COLORS.items()]
axes[2].legend(handles=patches, loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig("plot1_timeline.png", dpi=150, bbox_inches='tight')
print("\nSaved: plot1_timeline.png")

# ==========================================================
# FIGURE 2 — Force/torque timeline
# ==========================================================

fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
fig.suptitle("Force and torque over time", fontsize=14, fontweight='bold')

for ax in axes:
    state_start = None
    prev_state = None
    for _, row in df.iterrows():
        if row['state'] != prev_state:
            if prev_state is not None:
                ax.axvspan(state_start, row['timestamp_s'],
                           alpha=0.08, color=STATE_COLORS.get(prev_state, 'gray'))
            state_start = row['timestamp_s']
            prev_state = row['state']
    if prev_state:
        ax.axvspan(state_start, t.max(),
                   alpha=0.08, color=STATE_COLORS.get(prev_state, 'gray'))
    for _, row in stall_events.iterrows():
        ax.axvline(row['timestamp_s'], color='red', alpha=0.5,
                   linewidth=0.8, linestyle='--', label='_stall')

axes[0].plot(t, df['fz_N'], color='#D85A30', linewidth=1, label='fz_tared (N)')
axes[0].axhline(0, color='gray', linewidth=0.5, linestyle=':')
axes[0].set_ylabel('fz tared (N)')
axes[0].legend(loc='upper left', fontsize=9)
axes[0].grid(True, alpha=0.3)

axes[1].plot(t, df['torque_mag_Nm'], color='#7F77DD', linewidth=1, label='||τ|| tared (Nm)')
axes[1].axhline(0, color='gray', linewidth=0.5, linestyle=':')
axes[1].set_ylabel('Torque magnitude (Nm)')
axes[1].set_xlabel('Time (s)')
axes[1].legend(loc='upper left', fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plot2_forces.png", dpi=150, bbox_inches='tight')
print("Saved: plot2_forces.png")

# ==========================================================
# FIGURE 3 — Admittance scatter (Z pull)
# ==========================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Admittance validation — force vs velocity", fontsize=14, fontweight='bold')

# Z pull admittance
if len(pull_data) > 10:
    axes[0].scatter(pull_data['fz_N'], pull_data['v_z_mms'],
                    alpha=0.3, s=6, color='#1D9E75', label='data points')
    fz_range = np.linspace(pull_data['fz_N'].min(), pull_data['fz_N'].max(), 100)
    axes[0].plot(fz_range, slope * fz_range + intercept,
                 color='#D85A30', linewidth=2, label=f'fit (R²={r**2:.3f})')
    axes[0].axhline(0, color='gray', linewidth=0.5, linestyle=':')
    axes[0].axvline(0, color='gray', linewidth=0.5, linestyle=':')
    axes[0].set_xlabel('fz tared (N)')
    axes[0].set_ylabel('v_z (mm/s)')
    axes[0].set_title('Z pull admittance')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

# Rotation admittance
rot_data = normal[normal['state'].isin(["CCW", "CW"])].copy()
rot_data = rot_data.dropna(subset=['torque_mag_Nm', 'v_rz_degs'])

if len(rot_data) > 10:
    colors_rot = rot_data['state'].map({'CCW': '#7F77DD', 'CW': '#D85A30'})
    axes[1].scatter(rot_data['torque_mag_Nm'], rot_data['v_rz_degs'],
                    alpha=0.3, s=6, c=colors_rot, label='data')

    slope_r, intercept_r, r_r, _, _ = stats.linregress(
        rot_data['torque_mag_Nm'], rot_data['v_rz_degs'].abs())
    tm_range = np.linspace(0, rot_data['torque_mag_Nm'].max(), 100)
    axes[1].plot(tm_range, slope_r * tm_range + intercept_r,
                 color='black', linewidth=2, label=f'fit (R²={r_r**2:.3f})')
    axes[1].axhline(0, color='gray', linewidth=0.5, linestyle=':')
    axes[1].set_xlabel('torque magnitude (Nm)')
    axes[1].set_ylabel('|v_rz| (deg/s)')
    axes[1].set_title('Rotation admittance')

    ccw_patch = mpatches.Patch(color='#7F77DD', alpha=0.5, label='CCW')
    cw_patch  = mpatches.Patch(color='#D85A30', alpha=0.5, label='CW')
    axes[1].legend(handles=[ccw_patch, cw_patch], fontsize=9)
    axes[1].grid(True, alpha=0.3)

    print(f"\n--- Validation 2b: Rotation admittance linearity ---")
    print(f"  |v_rz| = {slope_r:.6f} * ||τ|| + {intercept_r:.6f}")
    print(f"  R²     = {r_r**2:.4f}")

plt.tight_layout()
plt.savefig("plot3_admittance_scatter.png", dpi=150, bbox_inches='tight')
print("Saved: plot3_admittance_scatter.png")

# ==========================================================
# FIGURE 4 — State sequence bar
# ==========================================================

state_blocks = []
prev_s = None
start_t = 0
for _, row in df.iterrows():
    if row['state'] != prev_s:
        if prev_s is not None:
            state_blocks.append((prev_s, start_t, row['timestamp_s'] - start_t))
        start_t = row['timestamp_s']
        prev_s = row['state']
if prev_s:
    state_blocks.append((prev_s, start_t, df['timestamp_s'].max() - start_t))

fig, ax = plt.subplots(figsize=(14, 3))
fig.suptitle("State sequence timeline", fontsize=14, fontweight='bold')

for i, (state, start, dur) in enumerate(state_blocks):
    color = STATE_COLORS.get(state, 'gray')
    ax.barh(0, dur, left=start, height=0.5, color=color, alpha=0.8, edgecolor='white')
    if dur > 0.5:
        ax.text(start + dur / 2, 0, f"{state}\n{dur:.1f}s",
                ha='center', va='center', fontsize=8, color='white', fontweight='bold')

ax.set_yticks([])
ax.set_xlabel('Time (s)')
ax.set_xlim(0, df['timestamp_s'].max())
ax.grid(True, axis='x', alpha=0.3)

patches = [mpatches.Patch(color=c, alpha=0.8, label=s) for s, c in STATE_COLORS.items()]
ax.legend(handles=patches, loc='upper right', fontsize=9)

for _, row in stall_events.iterrows():
    ax.axvline(row['timestamp_s'], color='red', alpha=0.6, linewidth=1, linestyle='--')

plt.tight_layout()
plt.savefig("plot4_state_sequence.png", dpi=150, bbox_inches='tight')
print("Saved: plot4_state_sequence.png")

# ==========================================================
# Summary report
# ==========================================================

print("\n" + "="*50)
print("SUMMARY")
print("="*50)
print(f"  Total duration       : {df['timestamp_s'].max():.1f} s")
print(f"  Total Z travel       : {(df['pos_z_m'].max() - df['pos_z_m'].min())*1000:.2f} mm")
print(f"  Total rotation       : {df['rot_angle_deg'].max():.2f} deg")
print(f"  State switches       : {len(state_changes) - 1}")
print(f"  Stall events         : {len(stall_events)}")
print(f"  Max |fz|             : {df['fz_N'].abs().max():.3f} N")
print(f"  Max ||torque||       : {df['torque_mag_Nm'].abs().max():.3f} Nm")
print(f"  Z admittance R²      : {r**2:.4f}")
print(f"  Goal reached         : {'YES' if len(goal_rows) > 0 else 'NO'}")
print("="*50)

plt.show()
