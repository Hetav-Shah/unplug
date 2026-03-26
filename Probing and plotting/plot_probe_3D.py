import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---- Load CSV ----
data = pd.read_csv("y2.csv")

# Clean column names
data.columns = data.columns.str.strip()

# Convert to numeric
data = data.apply(pd.to_numeric, errors='coerce')

# ---- Extract data ----
t = data["time"].to_numpy()

px = data["px"].to_numpy()
py = data["py"].to_numpy()
pz = data["pz"].to_numpy()

Fx = data["Fx"].to_numpy()
Fy = data["Fy"].to_numpy()
Fz = data["Fz"].to_numpy()

Tx = data["Tx"].to_numpy()
Ty = data["Ty"].to_numpy()
Tz = data["Tz"].to_numpy()

# ---- Magnitudes ----
Fmag = np.sqrt(Fx**2 + Fy**2 + Fz**2)
Tmag = np.sqrt(Tx**2 + Ty**2 + Tz**2)

# ---- Displacement from start ----
x0, y0, z0 = px[0], py[0], pz[0]

dx = px - x0
dy = py - y0
dz = pz - z0

distance = np.sqrt(dx**2 + dy**2 + dz**2)

# ---- Safe normalization ----
eps = 1e-6
Fmag_safe = np.where(Fmag < eps, eps, Fmag)

Fx_dir = Fx / Fmag_safe
Fy_dir = Fy / Fmag_safe
Fz_dir = Fz / Fmag_safe

# ---- Global axis ranges (FIXED SCALING) ----
force_max = np.max(np.abs([Fx, Fy, Fz]))
torque_max = np.max(np.abs([Tx, Ty, Tz]))
disp_max = np.max(np.abs([dx, dy, dz]))

force_lim = [-force_max * 1.1, force_max * 1.1]
torque_lim = [-torque_max * 1.1, torque_max * 1.1]
disp_lim = [-disp_max * 1.1, disp_max * 1.1]

# ---- Create layout (5 rows × 3 columns) ----
fig = make_subplots(
    rows=5,
    cols=3,
    subplot_titles=(
        "Fx vs Time", "Fy vs Time", "Fz vs Time",
        "Tx vs Time", "Ty vs Time", "Tz vs Time",
        "dx vs Time", "dy vs Time", "dz vs Time",
        "3D Force Cloud", "3D Torque Cloud", "3D Displacement Cloud",
        "Fx/F", "Fy/F", "Fz/F"
    ),
    specs=[
        [{"type":"xy"}, {"type":"xy"}, {"type":"xy"}],
        [{"type":"xy"}, {"type":"xy"}, {"type":"xy"}],
        [{"type":"xy"}, {"type":"xy"}, {"type":"xy"}],
        [{"type":"scene"}, {"type":"scene"}, {"type":"scene"}],
        [{"type":"xy"}, {"type":"xy"}, {"type":"xy"}]
    ]
)

# ------------------------------------------------
# Row 1: Forces
# ------------------------------------------------
fig.add_trace(go.Scatter(x=t, y=Fx), row=1, col=1)
fig.add_trace(go.Scatter(x=t, y=Fy), row=1, col=2)
fig.add_trace(go.Scatter(x=t, y=Fz), row=1, col=3)

# ------------------------------------------------
# Row 2: Torques
# ------------------------------------------------
fig.add_trace(go.Scatter(x=t, y=Tx), row=2, col=1)
fig.add_trace(go.Scatter(x=t, y=Ty), row=2, col=2)
fig.add_trace(go.Scatter(x=t, y=Tz), row=2, col=3)

# ------------------------------------------------
# Row 3: Displacements
# ------------------------------------------------
fig.add_trace(go.Scatter(x=t, y=dx), row=3, col=1)
fig.add_trace(go.Scatter(x=t, y=dy), row=3, col=2)
fig.add_trace(go.Scatter(x=t, y=dz), row=3, col=3)

# ------------------------------------------------
# Row 4: 3D Clouds
# ------------------------------------------------
fig.add_trace(
    go.Scatter3d(
        x=Fx, y=Fy, z=Fz,
        mode='markers',
        marker=dict(size=4, color=distance, colorscale='Viridis',
                    colorbar=dict(title="Displacement"), opacity=0.8)
    ),
    row=4, col=1
)

fig.add_trace(
    go.Scatter3d(
        x=Tx, y=Ty, z=Tz,
        mode='markers',
        marker=dict(size=4, color=Tmag, colorscale='Plasma',
                    colorbar=dict(title="|Torque|"), opacity=0.8)
    ),
    row=4, col=2
)

fig.add_trace(
    go.Scatter3d(
        x=dx, y=dy, z=dz,
        mode='markers',
        marker=dict(size=4, color=Fmag, colorscale='Turbo',
                    colorbar=dict(title="|Force|"), opacity=0.8)
    ),
    row=4, col=3
)

# ------------------------------------------------
# Row 5: Force Direction
# ------------------------------------------------
fig.add_trace(go.Scatter(x=t, y=Fx_dir), row=5, col=1)
fig.add_trace(go.Scatter(x=t, y=Fy_dir), row=5, col=2)
fig.add_trace(go.Scatter(x=t, y=Fz_dir), row=5, col=3)

# ---- Axis labels ----
for col in range(1, 4):
    fig.update_xaxes(title_text="Time (s)", row=1, col=col)
    fig.update_xaxes(title_text="Time (s)", row=2, col=col)
    fig.update_xaxes(title_text="Time (s)", row=3, col=col)
    fig.update_xaxes(title_text="Time (s)", row=5, col=col)

# ---- FIXED Y-RANGES ----
for col in range(1, 4):
    fig.update_yaxes(range=force_lim, row=1, col=col)
    fig.update_yaxes(range=torque_lim, row=2, col=col)
    fig.update_yaxes(range=disp_lim, row=3, col=col)
    fig.update_yaxes(range=[-1.1, 1.1], row=5, col=col)

# Labels
fig.update_yaxes(title_text="Force (N)", row=1, col=1)
fig.update_yaxes(title_text="Torque (Nm)", row=2, col=1)
fig.update_yaxes(title_text="Displacement (m)", row=3, col=1)
fig.update_yaxes(title_text="Normalized", row=5, col=1)

# 3D axis labels
fig.update_scenes(xaxis_title="Fx", yaxis_title="Fy", zaxis_title="Fz", row=4, col=1)
fig.update_scenes(xaxis_title="Tx", yaxis_title="Ty", zaxis_title="Tz", row=4, col=2)
fig.update_scenes(xaxis_title="dx", yaxis_title="dy", zaxis_title="dz", row=4, col=3)

# ---- Layout ----
fig.update_layout(
    title="Probe Data Visualization Dashboard",
    height=1400,
    width=1700,
    showlegend=False
)
print("Plotting started")
fig.show()