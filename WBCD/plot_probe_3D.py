import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---- Load CSV ----
data = pd.read_csv("1.csv")

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

# ---- Create layout (2 rows × 3 columns) ----
fig = make_subplots(
    rows=2,
    cols=3,
    subplot_titles=(
        "Fx vs Time",
        "Fy vs Time",
        "Fz vs Time",
        "3D Force Cloud",
        "3D Displacement Cloud",
        "Force vs Displacement"
    ),
    specs=[
        [{"type":"xy"}, {"type":"xy"}, {"type":"xy"}],
        [{"type":"scene"}, {"type":"scene"}, {"type":"xy"}]
    ]
)

# ------------------------------------------------
# 1️⃣ Fx vs Time
# ------------------------------------------------
fig.add_trace(go.Scatter(x=t, y=Fx, name="Fx"), row=1, col=1)

# ------------------------------------------------
# 2️⃣ Fy vs Time 
# ------------------------------------------------
fig.add_trace(go.Scatter(x=t, y=Fy, name="Fy"), row=1, col=2)

# ------------------------------------------------
# 3️⃣ Fz vs Time
# ------------------------------------------------
fig.add_trace(go.Scatter(x=t, y=Fz, name="Fz"), row=1, col=3)

# ------------------------------------------------
# 4️⃣ 3D Force Cloud
# ------------------------------------------------
fig.add_trace(
    go.Scatter3d(
        x=Fx,
        y=Fy,
        z=Fz,
        mode='markers',
        marker=dict(
            size=3,
            color=distance,
            colorscale='Viridis',
            colorbar=dict(title="Distance")
        ),
        name="Force Cloud"
    ),
    row=2,
    col=1
)

# ------------------------------------------------
# 5️⃣ 3D Displacement Cloud
# ------------------------------------------------
fig.add_trace(
    go.Scatter3d(
        x=px,
        y=py,
        z=pz,
        mode='markers',
        marker=dict(
            size=3,
            color=Fmag,
            colorscale='Turbo',
            colorbar=dict(title="Force")
        ),
        name="Displacement Cloud"
    ),
    row=2,
    col=2
)

# ------------------------------------------------
# 6️⃣ Force vs Displacement
# ------------------------------------------------
fig.add_trace(
    go.Scatter(x=distance, y=Fmag, name="|F| vs dist"),
    row=2,
    col=3
)

# ---- Axis labels ----
fig.update_xaxes(title_text="Time (s)", row=1, col=1)
fig.update_yaxes(title_text="Fx (N)", row=1, col=1)

fig.update_xaxes(title_text="Time (s)", row=1, col=2)
fig.update_yaxes(title_text="Fy (N)", row=1, col=2)

fig.update_xaxes(title_text="Time (s)", row=1, col=3)
fig.update_yaxes(title_text="Fz (N)", row=1, col=3)

fig.update_xaxes(title_text="Displacement (m)", row=2, col=3)
fig.update_yaxes(title_text="Force magnitude (N)", row=2, col=3)

fig.update_scenes(
    xaxis_title="Fx",
    yaxis_title="Fy",
    zaxis_title="Fz",
    row=2,
    col=1
)

fig.update_scenes(
    xaxis_title="X",
    yaxis_title="Y",
    zaxis_title="Z",
    row=2,
    col=2
)

# ---- Layout ----
fig.update_layout(
    title="Probe Data Visualization Dashboard",
    height=900,
    width=1700,
    showlegend=True
)

fig.show()