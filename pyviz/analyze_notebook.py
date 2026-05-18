# %%
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from matplotlib.axes import Axes
import pandas as pd

from filter3.data_loader import (
    load_trial_as_imu_sequence,
    G_SCALAR,
    compute_world_frame_acceleration,
)
from filter3.common import ImuSequence, ViconSequence, StanceSequence

# Optional: discover available trials
# def discover_trials(data_dir: Path) -> list[str]:
#     """Return trial base names that have both MCAP and CSV files."""
#     trial_names: list[str] = []
#     for mcap_path in sorted(data_dir.glob("*.mcap")):
#         base = mcap_path.stem
#         if (data_dir / f"{base}.csv").exists():
#             trial_names.append(base)
#     return trial_names

# available_trials = discover_trials(data_dir)
# print(f"Available trials: {available_trials}")

# %%
# Configuration
data_dir = Path("data")
trial_name = "back_01"  # Change this to your trial name
compensate_lag = True
thresholds = {
    "velocity_z_threshold": 0.07,
    "min_stance_duration": 0.08,
    "velocity_threshold_enter": 0.03,
    "velocity_threshold_leave": 0.08,
}

# Load trial sequences
timestamps_path = data_dir / "timestamps.xlsx"
if not timestamps_path.exists():
    raise FileNotFoundError(f"timestamps.xlsx not found in {data_dir}")

imu_seq, vicon_seq, stance_seq = load_trial_as_imu_sequence(
    mcap_file_path=str(data_dir / f"{trial_name}.mcap"),
    csv_file_path=str(data_dir / f"{trial_name}.csv"),
    timestamps_file_path=str(timestamps_path),
    base_name=trial_name,
    compensate_lag=compensate_lag,
    thresholds=thresholds
)

print(f"Loaded trial: {trial_name}")
print(f"  IMU samples: {len(imu_seq.timestamp)}")
print(f"  Vicon samples: {len(vicon_seq.timestamp)}")
print(f"  Stance samples: {len(stance_seq.timestamp)}")

# %%
# Prepare data for visualization
t = np.asarray(imu_seq.timestamp)

# Compute world-frame accelerations for visualization
# imu_seq.accel is RAW body-frame, need to rotate it for comparison with Vicon
accel_body = np.asarray(imu_seq.accel)
quat_enu = np.asarray(imu_seq.quat)
imu_acc = compute_world_frame_acceleration(
    accel_body, quat_enu, apply_enu_to_wsu=True, remove_gravity=True
)

vicon_acc = np.asarray(vicon_seq.acceleration)
vicon_t = np.asarray(vicon_seq.timestamp)
vicon_velocity_mag = np.asarray(stance_seq.velocity_magnitude)
stance = np.asarray(stance_seq.local_stance)

# Compute dv/dt-based acceleration (also rotate to world frame for comparison)
imu_raw_accel = accel_body.copy()  # Keep raw body-frame for dv comparison

dv_raw_series = np.asarray(imu_seq.dv)
# Rotate dv to world frame for plotting
dv_world_frame = compute_world_frame_acceleration(
    dv_raw_series, quat_enu, apply_enu_to_wsu=True, remove_gravity=True
)

dt = np.diff(t)
if dt.size == 0:
    dt = np.array([1.0], dtype=float)
dt = np.concatenate(([dt[0]], dt))
dt = np.where(dt <= 0, np.nan, dt)

with np.errstate(divide="ignore", invalid="ignore"):
    dv_acc_body = dv_raw_series / dt[:, None]
dv_acc_df = pd.DataFrame(dv_acc_body).ffill().bfill()
dv_acc_body_filled = dv_acc_df.to_numpy(dtype=float)

# Rotate dv/dt-based acceleration to world frame for comparison
dv_equivalent_accel = compute_world_frame_acceleration(
    dv_acc_body_filled, quat_enu, apply_enu_to_wsu=True, remove_gravity=True
)

print(f"Data prepared:")
print(f"  Time range: {t[0]:.2f}s to {t[-1]:.2f}s")
print(f"  Duration: {t[-1] - t[0]:.2f}s")

# %%
# Compute alignment metrics
if t.size > 0:
    sample_periods = np.diff(t)
    median_dt = float(np.median(sample_periods)) if sample_periods.size else float("nan")
    residual = imu_acc - vicon_acc

    # Compute RMSE per axis
    rmse_components = []
    for axis in range(residual.shape[1]):
        axis_residual = residual[:, axis]
        valid = np.isfinite(axis_residual)
        if valid.any():
            rmse_components.append(float(np.sqrt(np.mean(axis_residual[valid] ** 2))))
        else:
            rmse_components.append(float("nan"))
    rmse = np.array(rmse_components, dtype=float)

    # Compute total RMSE
    valid_mask = np.all(np.isfinite(residual), axis=1)
    if valid_mask.any():
        rmse_total = float(np.sqrt(np.mean(residual[valid_mask] ** 2)))
    else:
        rmse_total = float("nan")

    time_diff = np.abs(vicon_t - t)
    max_time_offset = float(np.max(time_diff)) if time_diff.size else float("nan")

    print(f"Alignment Statistics:")
    print(f"  Samples: {t.shape[0]}")
    print(f"  Duration: {t[-1] - t[0]:.2f} s")
    print(f"  Median dt: {median_dt * 1000.0:.2f} ms")
    print(f"  Accel RMSE (m/s²): x={rmse[0]:.3f}, y={rmse[1]:.3f}, z={rmse[2]:.3f}")
    print(f"  Total RMSE: {rmse_total:.3f} m/s²")
    print(f"  Max timestamp mismatch: {max_time_offset * 1e3:.4f} ms")

    # # Calculate dv/dt comparison metrics
    # mse_per_axis = np.mean((imu_raw_accel - dv_equivalent_accel) ** 2, axis=0)
    # dv_rmse_per_axis = np.sqrt(mse_per_axis)
    # print(f"\nAccelerometer vs dv/dt comparison:")
    # print(f"  RMSE x: {dv_rmse_per_axis[0]:.4f} m/s²")
    # print(f"  RMSE y: {dv_rmse_per_axis[1]:.4f} m/s²")
    # print(f"  RMSE z: {dv_rmse_per_axis[2]:.4f} m/s²")
    # print(f"  Total RMSE: {np.sqrt(np.mean(mse_per_axis)):.4f} m/s²")

# %%
# Plot: Acceleration Alignment
axis_labels = ("x", "y", "z")
fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10,8))

for idx, ax in enumerate(axes):
    ax.plot(
        t,
        vicon_acc[:, idx],
        label="Vicon acceleration",
        color="tab:blue",
        linewidth=1.0,
    )
    ax.plot(
        t,
        imu_acc[:, idx],
        label="IMU acceleration",
        color="tab:orange",
        linewidth=0.9,
        alpha=0.9,
    )
    ax.plot(
        t,
        dv_world_frame[:, idx],
        label="IMU Δv",
        color="tab:green",
        linewidth=0.8,
        linestyle="-.",
        alpha=0.8,
    )
    ax.plot(
        t,
        dv_equivalent_accel[:, idx],
        label="IMU Δv / Δt",
        color="tab:purple",
        linewidth=0.8,
        linestyle="--",
        alpha=0.8,
    )
    ax.set_ylabel(f"{axis_labels[idx]} (m/s²)")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

axes[-1].set_xlabel("Time (s)")
axes[0].set_title(f"{trial_name}: acceleration alignment")
axes[0].legend()
fig.tight_layout()
plt.show()

# %%
# Plot: Raw Acceleration Comparison
axis_labels = ("x", "y", "z")
fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 6))

dv_axes = []
for idx, ax in enumerate(axes):
    ax.plot(
        t,
        imu_raw_accel[:, idx],
        label="Raw IMU acceleration",
        color="tab:orange",
        linewidth=0.9,
    )
    ax.plot(
        t,
        dv_equivalent_accel[:, idx],
        label="Raw Δv / Δt",
        color="tab:purple",
        linewidth=0.8,
        linestyle="--",
    )
    ax.set_ylabel(f"{axis_labels[idx]} (m/s²)")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    dv_ax = ax.twinx()
    dv_ax.plot(
        t,
        dv_raw_series[:, idx],
        label="Raw Δv",
        color="tab:green",
        linewidth=0.7,
        linestyle=":",
        alpha=0.8,
    )
    dv_ax.set_ylabel(f"{axis_labels[idx]} Δv (m/s)")
    dv_axes.append(dv_ax)

axes[-1].set_xlabel("Time (s)")
axes[0].set_title(f"{trial_name}: raw acceleration vs Δv/Δt")
# Build combined legend (use first subplot)
handles, labels = axes[0].get_legend_handles_labels()
dv_handles, dv_labels = dv_axes[0].get_legend_handles_labels()
axes[0].legend(handles + dv_handles, labels + dv_labels, loc="upper right")
fig.tight_layout()
plt.show()

# %%
# Plot: Velocity and Position with ZVUP
foot_pos = np.asarray(vicon_seq.foot_position)
foot_vel = np.asarray(vicon_seq.foot_velocity)

fig, axes = plt.subplots(7, 1, figsize=(12, 15), sharex=True)

axes_labels = ['X', 'Y', 'Z']
colors = ['tab:blue', 'tab:orange', 'tab:green']

# Plot foot velocities (first 3 subplots) - this is what stance detection uses
for i, (ax, label, color) in enumerate(zip(axes[:3], axes_labels, colors)):
    ax.plot(t, foot_vel[:, i], color=color, linewidth=1, label=f'Foot Velocity {label}')
    ax.axhline(y=0, color='black', linestyle=':', linewidth=1, alpha=0.7)

    # Zoom in on Z velocity to see stance detection more clearly
    if i == 2:  # Z axis
        ax.set_ylim(-0.25, 0.25)

    # Overlay stance as shaded regions
    y_min, y_max = ax.get_ylim()
    ax.fill_between(t, y_min, y_max, where=stance > 0.5, alpha=0.2, color='red', label='Stance' if i == 0 else '')

    ax.set_ylabel(f'Foot Vel {label} (m/s)')
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    if i == 0:
        ax.legend(loc='upper right')

# Plot foot positions (next 3 subplots)
for i, (ax, label, color) in enumerate(zip(axes[3:6], axes_labels, colors)):
    ax.plot(t, foot_pos[:, i], color=color, linewidth=1, label=f'Foot Position {label}')
    ax.axhline(y=0, color='black', linestyle=':', linewidth=1, alpha=0.7)

    # Overlay stance as shaded regions
    y_min, y_max = ax.get_ylim()
    ax.fill_between(t, y_min, y_max, where=stance > 0.5, alpha=0.2, color='red')

    ax.set_ylabel(f'Foot Pos {label} (m)')
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    if i == 0:
        ax.legend(loc='upper right')

# Plot stance signal directly
axes[6].plot(t, stance, color='red', linewidth=2, label='Stance')
axes[6].fill_between(t, 0, stance, alpha=0.3, color='red')
axes[6].set_ylabel('Stance')
axes[6].set_xlabel('Time (s)')
axes[6].set_ylim([-0.1, 1.1])
axes[6].grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
axes[6].legend(loc='upper right')

fig.suptitle(f"{trial_name}: Velocity and Position with Stance", fontsize=12, fontweight='bold')
fig.tight_layout()
plt.show()
# %%
# Plot: Velocity Magnitude with Stance
fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)

# Plot velocity magnitude
axes[0].plot(t, vicon_velocity_mag, color='tab:blue', linewidth=1.5, label='IMU Velocity Magnitude')

# Overlay stance as shaded regions
y_min, y_max = axes[0].get_ylim()
axes[0].fill_between(t, y_min, y_max, where=stance > 0.5, alpha=0.2, color='red', label='Stance')

axes[0].set_ylabel('Speed (m/s)')
axes[0].grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
axes[0].legend(loc='upper right')
axes[0].set_title(f"{trial_name}: Velocity Magnitude with Stance")

# Plot stance signal directly
axes[1].plot(t, stance, color='red', linewidth=2, label='Stance')
axes[1].fill_between(t, 0, stance, alpha=0.3, color='red')
axes[1].set_ylabel('Stance')
axes[1].set_xlabel('Time (s)')
axes[1].set_ylim([-0.1, 1.1])
axes[1].grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
axes[1].legend(loc='upper right')

fig.tight_layout()
plt.show()
