import matplotlib.pyplot as plt
from collections import deque
import time
import random  # replace with your real data source

# ---- Config ----
window_size = 100   # number of points to show

# ---- Data buffers ----
t_data = deque(maxlen=window_size)

Fx_data = deque(maxlen=window_size)
Fy_data = deque(maxlen=window_size)
Fz_data = deque(maxlen=window_size)

Tx_data = deque(maxlen=window_size)
Ty_data = deque(maxlen=window_size)
Tz_data = deque(maxlen=window_size)

# ---- Setup plot ----
plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Force lines
line_fx, = ax1.plot([], [], label="Fx")
line_fy, = ax1.plot([], [], label="Fy")
line_fz, = ax1.plot([], [], label="Fz")

ax1.set_title("Forces")
ax1.set_ylabel("Force (N)")
ax1.legend()
ax1.grid()

# Torque lines
line_tx, = ax2.plot([], [], label="Tx")
line_ty, = ax2.plot([], [], label="Ty")
line_tz, = ax2.plot([], [], label="Tz")

ax2.set_title("Torques")
ax2.set_ylabel("Torque (Nm)")
ax2.set_xlabel("Time")
ax2.legend()
ax2.grid()

start_time = time.time()

# ---- Main loop ----
while True:
    t = time.time() - start_time

    # 🔴 REPLACE THIS WITH YOUR REAL DATA
    Fx = random.uniform(-10, 10)
    Fy = random.uniform(-10, 10)
    Fz = random.uniform(-10, 10)

    Tx = random.uniform(-2, 2)
    Ty = random.uniform(-2, 2)
    Tz = random.uniform(-2, 2)

    # ---- Store data ----
    t_data.append(t)

    Fx_data.append(Fx)
    Fy_data.append(Fy)
    Fz_data.append(Fz)

    Tx_data.append(Tx)
    Ty_data.append(Ty)
    Tz_data.append(Tz)

    # ---- Update plots ----
    line_fx.set_data(t_data, Fx_data)
    line_fy.set_data(t_data, Fy_data)
    line_fz.set_data(t_data, Fz_data)

    line_tx.set_data(t_data, Tx_data)
    line_ty.set_data(t_data, Ty_data)
    line_tz.set_data(t_data, Tz_data)

    # ---- Rescale axes ----
    ax1.relim()
    ax1.autoscale_view()

    ax2.relim()
    ax2.autoscale_view()

    # ---- Draw ----
    plt.pause(0.01)