import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# === Load both CSVs ===
traj = pd.read_csv("traj.csv")       # Your estimated trajectory
odom = pd.read_csv("odom_sim.csv")   # Ground truth from ROS

# === Normalize column names ===
traj.columns = [c.strip().lower() for c in traj.columns]
odom.columns = [c.strip().lower() for c in odom.columns]

# === Extract likely ground truth fields ===
# Assuming columns: [0]=index, [1]=time, [4]=x, [5]=y, [6]=z, [7–10]=quaternion
odom_time = odom.iloc[:, 0].astype(float).values
odom_x = odom.iloc[:, 4].astype(float).values
odom_y = odom.iloc[:, 5].astype(float).values
odom_qx = odom.iloc[:, 7].astype(float).values
odom_qy = odom.iloc[:, 8].astype(float).values
odom_qz = odom.iloc[:, 9].astype(float).values
odom_qw = odom.iloc[:, 10].astype(float).values

# === Convert quaternion → yaw ===
odom_yaw = np.arctan2(
    2 * (odom_qw * odom_qz + odom_qx * odom_qy),
    1 - 2 * (odom_qy**2 + odom_qz**2)
)

# === Prepare and normalize timestamps ===
odom_time = odom_time.astype(float)
if np.max(odom_time) > 1e6:  # Looks like UNIX timestamps
    odom_time = odom_time - odom_time[0]
else:
    odom_time = odom_time - odom_time[0]

# Ensure monotonic increasing time
if np.any(np.diff(odom_time) <= 0):
    valid = np.diff(odom_time, prepend=odom_time[0] - 1) > 0
    odom_time = odom_time[valid]
    odom_x = odom_x[valid]
    odom_y = odom_y[valid]
    odom_yaw = odom_yaw[valid]

# === Estimated trajectory ===
t_est = traj["time"].astype(float).values
t_est = t_est - t_est[0]

x_est = traj["x"].astype(float).values
y_est = traj["y"].astype(float).values
yaw_est = traj["yaw"].astype(float).values

# === Align timestamp ranges ===
if t_est[-1] > odom_time[-1]:
    t_est = np.clip(t_est, odom_time[0], odom_time[-1])

# === Interpolate ground truth safely ===
interp_x = interp1d(odom_time, odom_x, kind='linear', fill_value='extrapolate')
interp_y = interp1d(odom_time, odom_y, kind='linear', fill_value='extrapolate')
interp_yaw = interp1d(odom_time, odom_yaw, kind='linear', fill_value='extrapolate')

x_gt = interp_x(t_est)
y_gt = interp_y(t_est)
yaw_gt = interp_yaw(t_est)

# === Normalize to start at origin ===
x_est -= x_est[0]
y_est -= y_est[0]
x_gt -= x_gt[0]
y_gt -= y_gt[0]

# === Compute errors ===
ate = np.sqrt(np.mean((x_est - x_gt)**2 + (y_est - y_gt)**2))
yaw_err = np.mean(np.abs(np.rad2deg(yaw_est - yaw_gt)))

print(f"\nAbsolute Trajectory Error (ATE): {ate:.3f} m")
print(f"Mean Yaw Error: {yaw_err:.3f}°")
print(f"Estimated X range: {x_est.min():.4f} → {x_est.max():.4f}")
print(f"Estimated Y range: {y_est.min():.4f} → {y_est.max():.4f}")
print(f"Ground truth X range: {x_gt.min():.4f} → {x_gt.max():.4f}")
print(f"Ground truth Y range: {y_gt.min():.4f} → {y_gt.max():.4f}")

# === Plot trajectories ===
plt.figure(figsize=(7,7))
plt.plot(x_gt, y_gt, 'g-', label="Ground Truth (odom_sim)")
plt.plot(x_est, y_est, 'r--', label="Estimated (traj.csv)")

# Add direction arrows (optional)
skip = max(1, len(x_est)//40)
for i in range(0, len(x_est), skip):
    dx = 0.1 * np.cos(yaw_est[i])
    dy = 0.1 * np.sin(yaw_est[i])
    plt.arrow(x_est[i], y_est[i], dx, dy, head_width=0.03, color='r', alpha=0.6)

for i in range(0, len(x_gt), skip):
    dx = 0.1 * np.cos(yaw_gt[i])
    dy = 0.1 * np.sin(yaw_gt[i])
    plt.arrow(x_gt[i], y_gt[i], dx, dy, head_width=0.03, color='g', alpha=0.6)

plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.legend()
plt.title("Trajectory Comparison (Top-Down View)")
plt.axis("equal")
plt.grid(True)
plt.show()

# === Plot ATE over time ===
pos_error = np.sqrt((x_est - x_gt)**2 + (y_est - y_gt)**2)
plt.figure(figsize=(8,4))
plt.plot(t_est, pos_error, label="ATE [m]")
plt.xlabel("Time [s]")
plt.ylabel("Error [m]")
plt.title("Absolute Trajectory Error Over Time")
plt.grid(True)
plt.legend()
plt.show()
