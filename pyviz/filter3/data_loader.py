"""
Data loader for converting MCAP and CSV data into ImuSequence and ViconSequence formats.

This module provides functions to load IMU data from MCAP files and ground truth
data from CSV (Vicon motion capture) files, synchronize them, and convert to the
ImuSequence and ViconSequence formats used by the JAX-based filter implementations
in filter2/.

Key conventions:
    - Quaternions: (w, x, y, z) order
    - Timestamps: Converted to seconds (relative to first sample)
    - Units: m/s^2 for acceleration, rad/s for angular velocity, meters for position
    - Alignment: ViconSequence is interpolated to match IMU timestamps exactly
    - Data type: float32 for memory efficiency and JAX compatibility
"""

import csv
import numpy as np
import pandas as pd
import polars as pl
import jax.numpy as jnp

from datetime import datetime
from io import StringIO
from scipy.spatial.transform import Rotation as R
from scipy.signal import correlate
from scipy.interpolate import interp1d

from .common import ImuSequence, ViconSequence, StanceSequence


# Constants
G_SCALAR = 9.81  # Gravity in m/s^2
ENU_TO_WSU = R.from_matrix([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])


# Helper functions for loading data
def csv_to_dataframe(path: str, absolute_start: datetime) -> pl.DataFrame:
    """
    Parse Vicon CSV file and return Polars DataFrame.

    Args:
        path: Path to CSV file
        absolute_start: Absolute start timestamp as datetime

    Returns:
        Polars DataFrame with Vicon data
    """
    # Read file with UTF-8-sig to drop any BOM
    with open(path, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()

    # Locate the start indices for each section
    # Some files may not have a Segments section, only Trajectories
    seg_idx = next((i for i, L in enumerate(lines) if L.strip().startswith('Segments')), None)
    traj_idx = next((i for i, L in enumerate(lines) if L.strip().startswith('Trajectories')), None)

    if traj_idx is None:
        raise ValueError("CSV file does not contain 'Trajectories' section")

    def parse_block(start_line: int, end_line: int) -> pl.DataFrame:
        grp_row = lines[start_line + 2].rstrip('\n')
        name_row = lines[start_line + 3].rstrip('\n')
        data_start = start_line + 5
        data_end = end_line

        groups = next(csv.reader([grp_row]))
        names = next(csv.reader([name_row]))

        if len(groups) < len(names):
            groups += [''] * (len(names) - len(groups))
        else:
            groups = groups[:len(names)]

        filled = []
        last = ''
        for g in groups:
            if g.strip():
                last = g.strip()
            filled.append(last)

        block = ''.join(lines[data_start:data_end])

        # Use Polars to read CSV
        df = pl.read_csv(StringIO(block), has_header=False)

        flat = [f"{grp}_{nm}" if grp else nm
                for grp, nm in zip(filled, names)]
        df.columns = flat
        return df

    # Parse Trajectories section (always present)
    traj_df = parse_block(traj_idx, len(lines))

    # If Segments section exists, parse and merge it
    if seg_idx is not None:
        seg_df = parse_block(seg_idx, traj_idx)
        merged = seg_df.join(traj_df, on=['Frame', 'Sub Frame'], how='inner')
    else:
        # No Segments section, just use Trajectories
        merged = traj_df

    # Convert Frame to timedelta and add to absolute_start
    # Frame is at 100Hz, so each frame is 10ms
    merged = merged.with_columns([
        (pl.lit(absolute_start) + pl.duration(milliseconds=pl.col('Frame') * 10)).alias('world_clock_utc')
    ])

    return merged


def mcap_to_dataframe(mcap_file_path: str) -> pl.DataFrame:
    """
    Parse MCAP file and return Polars DataFrame with IMU data.

    Args:
        mcap_file_path: Path to MCAP file

    Returns:
        Polars DataFrame with IMU data
    """
    from mcap.reader import make_reader
    from gen import Packet_pb2  # type: ignore

    records = []
    start_time = None  # Initialize start_time to capture the first message's log_time

    with open(mcap_file_path, "rb") as f:
        reader = make_reader(f)
        for schema, channel, message in reader.iter_messages():
            if start_time is None:
                start_time = message.log_time  # Set start_time to the log_time of the first message
            try:
                pkt = Packet_pb2.Packet()  # type: ignore
                pkt.ParseFromString(message.data)
                if pkt.HasField("imu"):
                    imu_fields = {f"RAW:{field.name}": getattr(pkt.imu, field.name) for field in pkt.imu.DESCRIPTOR.fields}
                    imu_fields.update({
                        "topic": channel.topic,
                        "world_clock_utc": message.log_time,  # Store as int64 nanoseconds
                    })
                    records.append(imu_fields)
                else:
                    print(f"Skipping message on topic {channel.topic}: No imu field present")
            except Exception as e:
                print(f"Failed to decode message on topic {channel.topic}: {e}")

    # Create Polars DataFrame
    df = pl.DataFrame(records)

    # Convert world_clock_utc from nanoseconds to datetime
    if 'world_clock_utc' in df.columns:
        df = df.with_columns([
            pl.from_epoch(pl.col('world_clock_utc'), time_unit='ns').dt.replace_time_zone('UTC').alias('world_clock_utc')
        ])

    return df


def rotate_vector_by_quaternion(
    v_x: float, v_y: float, v_z: float,
    q_w: float, q_x: float, q_y: float, q_z: float
) -> tuple[float, float, float]:
    """Rotate vector from sensor frame to world frame using quaternion."""
    rotation = R.from_quat([q_x, q_y, q_z, q_w])
    rotated_vec = rotation.apply([v_x, v_y, v_z])
    return float(rotated_vec[0]), float(rotated_vec[1]), float(rotated_vec[2])


def rotate_accelerometer_by_quaternion(
    acc_x: float, acc_y: float, acc_z: float,
    q_w: float, q_x: float, q_y: float, q_z: float
) -> tuple[float, float, float]:
    """
    Rotate accelerometer reading by quaternion and remove gravity.

    Args:
        acc_x, acc_y, acc_z: Accelerometer readings in m/s^2
        q_w, q_x, q_y, q_z: Quaternion components (w, x, y, z)

    Returns:
        Tuple of rotated acceleration components with gravity removed
    """
    rot_x, rot_y, rot_z = rotate_vector_by_quaternion(acc_x, acc_y, acc_z, q_w, q_x, q_y, q_z)
    return rot_x, rot_y, rot_z - G_SCALAR


def rotate_delta_velocity_by_quaternion(
    dv_x: float, dv_y: float, dv_z: float,
    q_w: float, q_x: float, q_y: float, q_z: float
) -> tuple[float, float, float]:
    """Rotate delta-velocity vector by quaternion (no gravity removal)."""
    return rotate_vector_by_quaternion(dv_x, dv_y, dv_z, q_w, q_x, q_y, q_z)


def rotate_quaternion_to_coordinate_frame(
    q_w: float, q_x: float, q_y: float, q_z: float,
) -> tuple[float, float, float, float]:
    """
    Rotate quaternion to correct coordinate frame.

    Args:
        q_w, q_x, q_y, q_z: Quaternion components

    Returns:
        Tuple of rotated quaternion components (x, y, z, w)
    """
    # Transform from ENU to WSU frame
    rotation = R.from_quat([q_x, q_y, q_z, q_w])
    combined_rotation = ENU_TO_WSU * rotation
    out = combined_rotation.as_quat()

    # Convert to numpy array to get shape and dtype easily
    arr = np.asarray(out)
    return (float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))


def compute_world_frame_acceleration(
    accel_body: np.ndarray,
    quat: np.ndarray,
    apply_enu_to_wsu: bool = True,
    remove_gravity: bool = True,
) -> np.ndarray:
    """
    Rotate body-frame acceleration to world frame using quaternions.

    Args:
        accel_body: (N, 3) array of body-frame accelerations in m/s^2
        quat: (N, 4) array of quaternions in (w, x, y, z) order
        apply_enu_to_wsu: Whether to apply ENU to WSU frame transformation
        remove_gravity: Whether to remove gravity after rotation

    Returns:
        (N, 3) array of world-frame accelerations in m/s^2
    """
    N = accel_body.shape[0]
    result = np.zeros_like(accel_body)

    for i in range(N):
        # Extract quaternion components (w, x, y, z)
        q_w, q_x, q_y, q_z = quat[i]
        a_x, a_y, a_z = accel_body[i]

        # Apply quaternion rotation from body to world frame
        rotation = R.from_quat([q_x, q_y, q_z, q_w])

        # Apply ENU to WSU transformation if requested
        if apply_enu_to_wsu:
            rotation = ENU_TO_WSU * rotation

        rotated = rotation.apply([a_x, a_y, a_z])

        if remove_gravity:
            # Gravity is in the z-axis (down), so we subtract it
            rotated[2] -= G_SCALAR

        result[i] = rotated

    return result


def estimate_time_lag(
    vicon_df: pl.DataFrame,
    imu_df: pl.DataFrame,
    vicon_time_col: str = 'world_clock_utc',
    imu_time_col: str = 'world_clock_utc'
) -> float:
    """
    Estimate time lag between IMU and Vicon data by minimizing RMSE.

    This method searches through candidate lag values and picks the one that
    minimizes the RMSE between IMU and Vicon accelerations across all axes.
    This is more robust than cross-correlation alone, especially for complex
    motion patterns.

    Args:
        vicon_df: Vicon dataframe with 'imu:ddx', 'imu:ddy', 'imu:ddz' columns
        imu_df: IMU dataframe with 'RAW:a_x/y/z' and 'RAW:qw/qx/qy/qz' columns
        vicon_time_col: Name of time column in vicon_df
        imu_time_col: Name of time column in imu_df

    Returns:
        Estimated lag in seconds (positive means IMU is ahead of Vicon)
    """
    # First get a rough estimate using cross-correlation on z-axis
    # Vicon datetime has microsecond precision, IMU datetime has nanosecond precision
    # Extract integer timestamps in their native units
    vicon_times_us = vicon_df[vicon_time_col].dt.timestamp(time_unit='us').to_numpy()  # type: ignore[attr-defined]
    imu_times_ns = imu_df[imu_time_col].dt.timestamp(time_unit='ns').to_numpy()  # type: ignore[attr-defined]

    # Convert to relative times (seconds since first timestamp) for better numerical stability
    # Convert vicon microseconds to nanoseconds for consistency
    vicon_times_ns = vicon_times_us * 1000
    t0 = min(np.nanmin(vicon_times_ns), np.nanmin(imu_times_ns))
    vicon_times = (vicon_times_ns - t0) / 1e9
    imu_times = (imu_times_ns - t0) / 1e9

    start_time_sec = max(float(np.nanmin(vicon_times)), float(np.nanmin(imu_times)))
    end_time_sec = min(float(np.nanmax(vicon_times)), float(np.nanmax(imu_times)))

    if end_time_sec - start_time_sec < 1:
        print("Warning: Not enough overlap between IMU and Vicon.")
        return 0.0

    # Compute world-frame acceleration from raw IMU data
    accel_body = imu_df.select(['RAW:a_x', 'RAW:a_y', 'RAW:a_z']).to_numpy()  # type: ignore[call-overload]
    quat = imu_df.select(['RAW:qw', 'RAW:qx', 'RAW:qy', 'RAW:qz']).to_numpy()  # type: ignore[call-overload]
    imu_accel_world = compute_world_frame_acceleration(accel_body, quat, remove_gravity=True)

    # Quick cross-correlation for initial estimate
    common_time_sec = np.arange(start_time_sec, end_time_sec, 0.01)
    vicon_interp_z = interp1d(vicon_times.astype(float), vicon_df['imu:ddz'].to_numpy().astype(float),
                             bounds_error=False, fill_value=np.nan)
    imu_interp_z = interp1d(imu_times.astype(float), (imu_accel_world[:, 2] * 1000),
                           bounds_error=False, fill_value=np.nan)

    gt_z = vicon_interp_z(common_time_sec)
    imu_z = imu_interp_z(common_time_sec)
    valid = ~np.isnan(gt_z) & ~np.isnan(imu_z)

    if valid.sum() < 100:
        return 0.0

    gt_z = gt_z[valid]
    imu_z = imu_z[valid]
    min_len = min(len(gt_z), len(imu_z))
    gt_z = gt_z[:min_len]
    imu_z = imu_z[:min_len]

    corr = correlate(gt_z - np.mean(gt_z), imu_z - np.mean(imu_z), mode='full')
    lags = np.arange(-len(gt_z) + 1, len(gt_z))
    xcorr_lag_samples = int(lags[np.argmax(corr)])
    xcorr_lag_sec = float(xcorr_lag_samples) / 100.0

    # Now refine around the cross-correlation estimate using RMSE minimization
    # Search ±300ms around the cross-correlation estimate
    search_range_ms = 300
    search_step_ms = 10

    best_lag_sec = xcorr_lag_sec
    best_rmse = float('inf')

    search_attempted = False
    refinements_tested = 0

    def evaluate_candidate_lag(test_lag_sec: float) -> tuple[float | None, bool]:
        """Return (rmse, True) if candidate is valid, else (None, False)."""
        # Add test time offset to IMU dataframe
        imu_df_test = imu_df.with_columns([  # type: ignore[call-overload]
            (pl.col(imu_time_col) + pl.duration(seconds=test_lag_sec)).alias('test_time')
        ])

        imu_start = imu_df_test['test_time'].min()
        imu_end = imu_df_test['test_time'].max()
        vicon_start = vicon_df[vicon_time_col].min()
        vicon_end = vicon_df[vicon_time_col].max()
        assert imu_start is not None and imu_end is not None, "IMU timestamps should not be None"
        assert vicon_start is not None and vicon_end is not None, "Vicon timestamps should not be None"

        start = max(imu_start, vicon_start)
        end = min(imu_end, vicon_end)

        # Calculate time difference in seconds (end and start are datetime objects)
        time_diff = end - start  # type: ignore  # This is a timedelta
        time_diff_sec = time_diff.total_seconds()  # type: ignore
        if time_diff_sec < 2.0:  # 2 seconds
            return None, False

        imu_overlap = imu_df_test.filter((pl.col('test_time') >= start) & (pl.col('test_time') <= end))
        vicon_overlap = vicon_df.filter((pl.col(vicon_time_col) >= start) & (pl.col(vicon_time_col) <= end))
        if len(imu_overlap) < 100 or len(vicon_overlap) < 100:
            return None, False

        # Convert to numpy for interpolation
        # Use timestamp() to get microseconds/nanoseconds as appropriate
        imu_t_ns = imu_overlap['test_time'].dt.timestamp(time_unit='ns').to_numpy()  # type: ignore[attr-defined]
        vicon_t_us = vicon_overlap[vicon_time_col].dt.timestamp(time_unit='us').to_numpy()  # type: ignore[attr-defined]
        vicon_t_ns: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = vicon_t_us * 1000  # convert us to ns
        first_t_ns = imu_t_ns[0]
        imu_t = (imu_t_ns - first_t_ns) / 1e9
        vicon_t = (vicon_t_ns - first_t_ns) / 1e9

        try:
            vicon_x_interp = interp1d(vicon_t, vicon_overlap['imu:ddx'].to_numpy() / 1000.0,
                                      bounds_error=False, fill_value=np.nan)
            vicon_y_interp = interp1d(vicon_t, vicon_overlap['imu:ddy'].to_numpy() / 1000.0,
                                      bounds_error=False, fill_value=np.nan)
            vicon_z_interp = interp1d(vicon_t, vicon_overlap['imu:ddz'].to_numpy() / 1000.0,
                                      bounds_error=False, fill_value=np.nan)

            vicon_acc = np.stack([
                vicon_x_interp(imu_t),
                vicon_y_interp(imu_t),
                vicon_z_interp(imu_t)
            ], axis=1)

            # Rotate IMU acceleration to world frame on-the-fly
            accel_body_overlap = imu_overlap.select(['RAW:a_x', 'RAW:a_y', 'RAW:a_z']).to_numpy()  # type: ignore[call-overload]
            quat_overlap = imu_overlap.select(['RAW:qw', 'RAW:qx', 'RAW:qy', 'RAW:qz']).to_numpy()  # type: ignore[call-overload]
            imu_acc = compute_world_frame_acceleration(accel_body_overlap, quat_overlap, remove_gravity=True)

            valid_mask = np.all(np.isfinite(vicon_acc), axis=1)
            if valid_mask.sum() < 100:
                return None, False

            residual = imu_acc[valid_mask] - vicon_acc[valid_mask]
            rmse = np.sqrt(np.mean(residual**2))
            return float(rmse), True
        except Exception:
            return None, False

    for offset_ms in range(-search_range_ms, search_range_ms + 1, search_step_ms):
        test_lag_sec = xcorr_lag_sec + (offset_ms / 1000.0)
        search_attempted = True
        rmse, ok = evaluate_candidate_lag(test_lag_sec)
        if not ok or rmse is None:
            continue

        refinements_tested += 1
        if rmse < best_rmse:
            best_rmse = rmse
            best_lag_sec = test_lag_sec

    # Fine refinement around the best coarse lag using 1 ms steps
    fine_range_ms =4
    fine_step_ms = 1
    fine_tests = 0
    for offset_ms in range(-fine_range_ms, fine_range_ms + 1, fine_step_ms):
        test_lag_sec = best_lag_sec + (offset_ms / 1000.0)
        rmse, ok = evaluate_candidate_lag(test_lag_sec)
        if not ok or rmse is None:
            continue
        fine_tests += 1
        if rmse < best_rmse:
            best_rmse = rmse
            best_lag_sec = test_lag_sec

    if search_attempted:
        improvement_ms = (best_lag_sec - xcorr_lag_sec) * 1000
        total_tests = refinements_tested + fine_tests
        print(f"  RMSE refinement: tested {total_tests} offsets, best lag: {best_lag_sec*1000:.0f}ms (xcorr: {xcorr_lag_sec*1000:.0f}ms, adjustment: {improvement_ms:+.0f}ms), RMSE: {best_rmse:.3f}")

    return best_lag_sec


def dataframe_to_vicon_sequence(
    vicon_df: pl.DataFrame,
    imu_timestamps: pl.Series,
    start_time,
    end_time,
    time_col: str = 'world_clock_utc',

) -> ViconSequence:
    """
    Convert Vicon dataframe to ViconSequence format, interpolated to IMU timestamps.

    This ensures that the ViconSequence and ImuSequence are perfectly aligned in time.
    The Vicon data will be interpolated to match the IMU timestamps exactly using
    linear interpolation.

    Units:
        - Positions and velocities converted from mm to meters
        - Accelerations converted from mm/s^2 to m/s^2
        - All arrays are float32 for JAX compatibility

    Args:
        vicon_df: Polars DataFrame with Vicon ground truth data (in mm, mm/s, mm/s^2)
        imu_timestamps: IMU timestamps (Polars Series) to interpolate Vicon data to
        time_col: Name of timestamp column in vicon_df
        start_time: Start time for data range (datetime)
        end_time: End time for data range (datetime)

    Returns:
        ViconSequence with JAX arrays (float32), perfectly aligned with IMU timestamps
        The returned sequence has EXACTLY the same length as imu_timestamps.

    Raises:
        ValueError: If time range is invalid or no Vicon data in range
    """
    # Filter Vicon data to relevant time range
    df = vicon_df.filter((pl.col(time_col) >= start_time) & (pl.col(time_col) <= end_time))

    if len(df) == 0:
        raise ValueError(f"No Vicon data in time range [{start_time}, {end_time}]")

    # Verify that IMU timestamps are within Vicon range
    imu_start = imu_timestamps.min()
    imu_end = imu_timestamps.max()
    vicon_start = df[time_col].min()
    vicon_end = df[time_col].max()

    assert imu_start is not None and imu_end is not None, "IMU timestamps should not be None"
    assert vicon_start is not None and vicon_end is not None, "Vicon timestamps should not be None"

    # Tolerance in seconds (2 milliseconds)
    tolerance_sec = 0.002
    # These are datetime objects from .min()/.max(), so we get timedeltas
    start_diff = imu_start - vicon_start  # type: ignore
    end_diff = imu_end - vicon_end  # type: ignore
    start_diff_sec = start_diff.total_seconds()  # type: ignore
    end_diff_sec = end_diff.total_seconds()  # type: ignore
    out_of_range = (start_diff_sec < -tolerance_sec) or (end_diff_sec > tolerance_sec)

    if out_of_range:
        print(f"Warning: IMU timestamps [{imu_start}, {imu_end}] extend beyond Vicon range [{vicon_start}, {vicon_end}]")
        print(f"  This should not happen with align_with_vicon=True. Interpolation will extrapolate at boundaries.")

    # Convert timestamps to seconds for interpolation
    vicon_time_us = df[time_col].dt.timestamp(time_unit='us').to_numpy()
    imu_time_ns = imu_timestamps.dt.timestamp(time_unit='ns').to_numpy()

    # Convert to same unit (nanoseconds)
    vicon_time_ns = vicon_time_us * 1000
    t0 = min(np.nanmin(vicon_time_ns), np.nanmin(imu_time_ns))
    vicon_time_sec = (vicon_time_ns - t0) / 1e9
    imu_time_sec = (imu_time_ns - t0) / 1e9

    # Helper function to interpolate columns to IMU timestamps
    def interp_to_imu(col_name: str) -> np.ndarray:
        """Interpolate a single column to IMU timestamps."""
        series = df[col_name].to_numpy().astype(float)
        valid = np.isfinite(series)

        if vicon_time_sec.size == 0 or series.size == 0 or valid.sum() < 2:
            return np.full_like(imu_time_sec, np.nan, dtype=float)

        return np.interp(
            imu_time_sec,
            vicon_time_sec[valid],
            series[valid],
            left=np.nan,
            right=np.nan,
        )

    # Interpolate all fields to IMU timestamps (convert mm to meters)
    position = np.stack([
        interp_to_imu('imu:x') / 1000.0,
        interp_to_imu('imu:y') / 1000.0,
        interp_to_imu('imu:z') / 1000.0
    ], axis=1)

    velocity = np.stack([
        interp_to_imu('imu:dx') / 1000.0,
        interp_to_imu('imu:dy') / 1000.0,
        interp_to_imu('imu:dz') / 1000.0
    ], axis=1)

    acceleration = np.stack([
        interp_to_imu('imu:ddx') / 1000.0,
        interp_to_imu('imu:ddy') / 1000.0,
        interp_to_imu('imu:ddz') / 1000.0
    ], axis=1)

    # Ankle data
    ankle_position = np.stack([
        interp_to_imu('ankle:x') / 1000.0 if 'ankle:x' in df.columns else np.zeros(len(imu_time_sec)),
        interp_to_imu('ankle:y') / 1000.0 if 'ankle:y' in df.columns else np.zeros(len(imu_time_sec)),
        interp_to_imu('ankle:z') / 1000.0 if 'ankle:z' in df.columns else np.zeros(len(imu_time_sec))
    ], axis=1)

    ankle_velocity = np.stack([
        interp_to_imu('ankle:dx') / 1000.0 if 'ankle:dx' in df.columns else np.zeros(len(imu_time_sec)),
        interp_to_imu('ankle:dy') / 1000.0 if 'ankle:dy' in df.columns else np.zeros(len(imu_time_sec)),
        interp_to_imu('ankle:dz') / 1000.0 if 'ankle:dz' in df.columns else np.zeros(len(imu_time_sec))
    ], axis=1)

    # Foot data
    foot_position = np.stack([
        interp_to_imu('foot:x') / 1000.0 if 'foot:x' in df.columns else np.zeros(len(imu_time_sec)),
        interp_to_imu('foot:y') / 1000.0 if 'foot:y' in df.columns else np.zeros(len(imu_time_sec)),
        interp_to_imu('foot:z') / 1000.0 if 'foot:z' in df.columns else np.zeros(len(imu_time_sec))
    ], axis=1)

    foot_velocity = np.stack([
        interp_to_imu('foot:dx') / 1000.0 if 'foot:dx' in df.columns else np.zeros(len(imu_time_sec)),
        interp_to_imu('foot:dy') / 1000.0 if 'foot:dy' in df.columns else np.zeros(len(imu_time_sec)),
        interp_to_imu('foot:dz') / 1000.0 if 'foot:dz' in df.columns else np.zeros(len(imu_time_sec))
    ], axis=1)

    return ViconSequence(
        position=jnp.array(position, dtype=jnp.float32),
        velocity=jnp.array(velocity, dtype=jnp.float32),
        acceleration=jnp.array(acceleration, dtype=jnp.float32),
        ankle_position=jnp.array(ankle_position, dtype=jnp.float32),
        ankle_velocity=jnp.array(ankle_velocity, dtype=jnp.float32),
        foot_position=jnp.array(foot_position, dtype=jnp.float32),
        foot_velocity=jnp.array(foot_velocity, dtype=jnp.float32),
        timestamp=jnp.array(imu_time_sec, dtype=jnp.float32)
    )


def dataframe_to_stance_sequence(
    vicon_df: pl.DataFrame,
    imu_timestamps: pl.Series,
    start_time,
    end_time,
    time_col: str = 'world_clock_utc',
    thresholds: dict[str, float] | None = None,
) -> StanceSequence:
    """
    Detect stance phase from Vicon foot velocity data and align with IMU timestamps.

    Stance phase is detected when the foot velocity magnitude falls below a threshold,
    indicating the foot is stationary on the ground.

    Args:
        vicon_df: Polars DataFrame with Vicon ground truth data
        imu_timestamps: IMU timestamps (Polars Series) to interpolate stance data to
        start_time: Start time for data range (datetime)
        end_time: End time for data range (datetime)
        time_col: Name of timestamp column in vicon_df
        velocity_threshold: Velocity threshold in m/s for stance detection (default: 0.05 m/s)

    Returns:
        StanceSequence with stance array and velocity magnitudes aligned with IMU timestamps

    Raises:
        ValueError: If time range is invalid or no Vicon data in range
    """
    # Filter Vicon data to relevant time range
    df = vicon_df.filter((pl.col(time_col) >= start_time) & (pl.col(time_col) <= end_time))

    if len(df) == 0:
        raise ValueError(f"No Vicon data in time range [{start_time}, {end_time}]")

    # Convert timestamps to seconds for interpolation
    vicon_time_us = df[time_col].dt.timestamp(time_unit='us').to_numpy()  # type: ignore[attr-defined]
    imu_time_ns = imu_timestamps.dt.timestamp(time_unit='ns').to_numpy()  # type: ignore[attr-defined]

    # Convert to same unit (nanoseconds)
    vicon_time_ns = vicon_time_us * 1000
    t0 = min(np.nanmin(vicon_time_ns), np.nanmin(imu_time_ns))
    vicon_time_sec = (vicon_time_ns - t0) / 1e9
    imu_time_sec = (imu_time_ns - t0) / 1e9

    # Helper function to interpolate columns to IMU timestamps
    def interp_to_imu(col_name: str) -> np.ndarray:
        """Interpolate a single column to IMU timestamps."""
        series = df[col_name].to_numpy().astype(float)
        valid = np.isfinite(series)

        if vicon_time_sec.size == 0 or series.size == 0 or valid.sum() < 2:
            return np.full_like(imu_time_sec, np.nan, dtype=float)

        return np.interp(
            imu_time_sec,
            vicon_time_sec[valid],
            series[valid],
            left=np.nan,
            right=np.nan,
        )

    # Calculate IMU velocity magnitude from velocity components
    imu_velocity = np.stack([
        interp_to_imu('imu:dx') / 1000.0,
        interp_to_imu('imu:dy') / 1000.0,
        interp_to_imu('imu:dz') / 1000.0
    ], axis=1)
    velocity_magnitude = np.linalg.norm(imu_velocity, axis=1)

    # Ankle velocity is mandatory
    if 'ankle:dx' not in df.columns or 'ankle:dy' not in df.columns or 'ankle:dz' not in df.columns:
        raise ValueError("Missing required ankle velocity columns (ankle:dx, ankle:dy, ankle:dz)")

    ankle_velocity = np.stack([
        interp_to_imu('ankle:dx') / 1000.0,
        interp_to_imu('ankle:dy') / 1000.0,
        interp_to_imu('ankle:dz') / 1000.0
    ], axis=1)
    ankle_velocity_magnitude = np.linalg.norm(ankle_velocity, axis=1)

    # Foot velocity is mandatory
    if 'foot:dx' not in df.columns or 'foot:dy' not in df.columns or 'foot:dz' not in df.columns:
        raise ValueError("Missing required foot velocity columns (foot:dx, foot:dy, foot:dz)")

    # Use foot velocity at native Vicon timestamps for stance detection (convert from mm/s to m/s)
    # This avoids interpolation artifacts affecting the hysteresis logic
    foot_vel_x = df['foot:dx'].to_numpy() / 1000.0
    foot_vel_y = df['foot:dy'].to_numpy() / 1000.0
    foot_vel_z = df['foot:dz'].to_numpy() / 1000.0
    foot_vel_mag = np.sqrt(foot_vel_x**2 + foot_vel_y**2 + foot_vel_z**2)

    # Detect stance: velocity magnitude below threshold
    # Convert to float: 1.0 for stance, 0.0 for no stance

    # stance_vicon = np.zeros_like(foot_vel_mag, dtype=np.float32)
    # in_stance = False
    # for i, vel in enumerate(foot_vel_mag):
    #     if not in_stance and vel < velocity_threshold_enter:
    #         in_stance = True
    #     elif in_stance and vel > velocity_threshold_leave:
    #         in_stance = False
    #     stance_vicon[i] = 1.0 if in_stance else 0.0

    # Thresholds are mandatory - must be provided explicitly
    if thresholds is None:
        raise ValueError(
            "Thresholds parameter is required for stance detection. "
            "Must provide a dictionary with 'velocity_threshold_enter', 'velocity_threshold_leave', "
            "and 'velocity_z_threshold' keys."
        )

    velocity_threshold_enter = thresholds.get('velocity_threshold_enter')
    velocity_threshold_leave = thresholds.get('velocity_threshold_leave')
    velocity_z_threshold = thresholds.get('velocity_z_threshold')
    min_stance_duration = thresholds.get('min_stance_duration', 0.06)

    if velocity_threshold_enter is None:
        raise ValueError("Missing required threshold: 'velocity_threshold_enter'")
    if velocity_threshold_leave is None:
        raise ValueError("Missing required threshold: 'velocity_threshold_leave'")
    if velocity_z_threshold is None:
        raise ValueError("Missing required threshold: 'velocity_z_threshold'")

    stance_mask = np.zeros_like(foot_vel_mag, dtype=bool)
    in_stance = False
    for i, v in enumerate(foot_vel_mag):
        # Z-threshold is a hard constraint - always exit stance if exceeded
        if abs(foot_vel_z[i]) > velocity_z_threshold:
            in_stance = False
        # Enter stance when magnitude is low AND z-velocity is low
        elif not in_stance and (v < velocity_threshold_enter) and (abs(foot_vel_z[i]) < velocity_z_threshold):
            in_stance = True
        # Exit stance when magnitude exceeds leave threshold
        elif in_stance and (v > velocity_threshold_leave):
            in_stance = False
        stance_mask[i] = in_stance

    # Minimum stance duration filter - remove stance phases that are too short
    # Operating at Vicon timestamps (native ground truth)
    dt = np.median(np.diff(vicon_time_sec))
    min_samples = int(min_stance_duration / dt)
    changes = np.diff(stance_mask.astype(int))
    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0] + 1
    if stance_mask[0]: starts = np.insert(starts, 0, 0)
    if stance_mask[-1]: ends = np.append(ends, len(stance_mask))
    for s, e in zip(starts, ends):
        if (e - s) < min_samples:
            stance_mask[s:e] = False

    # Interpolate stance to IMU timestamps
    # Use nearest neighbor interpolation for discrete 0/1 values
    stance_interp = interp1d(
        vicon_time_sec.astype(float),
        stance_mask.astype(float),
        kind='nearest',
        bounds_error=False,
        fill_value=0.0  # Default to not-stance outside range
    )

    stance_imu = stance_interp(imu_time_sec)

    return StanceSequence(
        local_stance=jnp.array(stance_imu, dtype=jnp.float32),
        velocity_magnitude=jnp.array(velocity_magnitude, dtype=jnp.float32),
        ankle_velocity_magnitude=jnp.array(ankle_velocity_magnitude, dtype=jnp.float32),
        timestamp=jnp.array(imu_time_sec, dtype=jnp.float32)
    )


def load_trial_data(
    mcap_file_path: str,
    csv_file_path: str,
    timestamps_file_path: str,
    base_name: str,
    compensate_lag: bool = True
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Load and preprocess IMU and Vicon data for a single trial.

    Both IMU and Vicon data are required for all trials.

    Args:
        mcap_file_path: Path to MCAP file containing IMU data (required)
        csv_file_path: Path to CSV file containing Vicon data (required)
        timestamps_file_path: Path to timestamps Excel file (required)
        base_name: Trial name (required for quaternion frame conversion)
        compensate_lag: Whether to estimate and compensate for time lag

    Returns:
        Tuple of (imu_df, vicon_df) both as pandas DataFrames

    Raises:
        FileNotFoundError: If any required file is missing
        ValueError: If data cannot be loaded or processed
    """
    # Load IMU data (keep as raw, no rotation here)
    imu_df = mcap_to_dataframe(mcap_file_path)

    # Load Vicon data (required)
    # Read timestamps to get trial start time
    timestamps_df = pd.read_excel(timestamps_file_path)
    trial_row = timestamps_df[timestamps_df["Trial"] == base_name]

    if trial_row.empty:
        raise ValueError(f"Trial {base_name} not found in timestamps file")

    date_text = trial_row.iloc[0]["Date"]
    start_text = trial_row.iloc[0]["Start (Text)"]

    # Combine Date and Start (Text) into a single datetime
    datetime_str = f"{date_text.date()} {start_text}"
    local_vicon_t0 = pd.to_datetime(datetime_str, format="%Y-%m-%d %H:%M:%S.%f").tz_localize("US/Eastern")
    utc_vicon_t0 = local_vicon_t0.tz_convert("UTC")

    # Load Vicon CSV data
    vicon_df = csv_to_dataframe(csv_file_path, utc_vicon_t0)

    # Determine the subject prefix (e.g., "Duncan:", "ava:", etc.)
    # Look for IMU marker columns to determine prefix
    subject_prefix = None
    for col in vicon_df.columns:
        if 'LIMULAT_X' in col:
            subject_prefix = col.split('LIMULAT_X')[0]
            break

    if subject_prefix is None:
        raise ValueError("Could not determine subject prefix from Vicon CSV columns")

    # Calculate average position of IMU markers from Vicon (LIMULAT and LIMUMED)
    # These represent the ground truth position/velocity/acceleration of the IMU's location
    # Use 'imu:' prefix to indicate this is the IMU's ground truth (vs ankle, foot, etc.)
    vicon_df = vicon_df.with_columns([
        ((pl.col(f'{subject_prefix}LIMULAT_X') + pl.col(f'{subject_prefix}LIMUMED_X')) / 2).alias('imu:x'),
        ((pl.col(f'{subject_prefix}LIMULAT_Y') + pl.col(f'{subject_prefix}LIMUMED_Y')) / 2).alias('imu:y'),
        ((pl.col(f'{subject_prefix}LIMULAT_Z') + pl.col(f'{subject_prefix}LIMUMED_Z')) / 2).alias('imu:z'),
        ((pl.col(f"{subject_prefix}LIMULAT_X'") + pl.col(f"{subject_prefix}LIMUMED_X'")) / 2).alias('imu:dx'),
        ((pl.col(f"{subject_prefix}LIMULAT_Y'") + pl.col(f"{subject_prefix}LIMUMED_Y'")) / 2).alias('imu:dy'),
        ((pl.col(f"{subject_prefix}LIMULAT_Z'") + pl.col(f"{subject_prefix}LIMUMED_Z'")) / 2).alias('imu:dz'),
        ((pl.col(f"{subject_prefix}LIMULAT_X''") + pl.col(f"{subject_prefix}LIMUMED_X''")) / 2).alias('imu:ddx'),
        ((pl.col(f"{subject_prefix}LIMULAT_Y''") + pl.col(f"{subject_prefix}LIMUMED_Y''")) / 2).alias('imu:ddy'),
        ((pl.col(f"{subject_prefix}LIMULAT_Z''") + pl.col(f"{subject_prefix}LIMUMED_Z''")) / 2).alias('imu:ddz'),
    ])

    # Ankle markers are mandatory - try left ankle first, fall back to right ankle
    ankle_marker = None
    if f'{subject_prefix}LANK_X' in vicon_df.columns:
        ankle_marker = 'LANK'
    else:
        raise ValueError(f"No ankle markers found for subject '{subject_prefix}' (expected LANK or RANK)")

    # Ankle velocity derivatives are mandatory
    if f"{subject_prefix}{ankle_marker}_X'" not in vicon_df.columns:
        raise ValueError(f"Missing ankle velocity derivatives for marker {ankle_marker}")

    ankle_cols = [
        pl.col(f'{subject_prefix}{ankle_marker}_X').alias('ankle:x'),
        pl.col(f'{subject_prefix}{ankle_marker}_Y').alias('ankle:y'),
        pl.col(f'{subject_prefix}{ankle_marker}_Z').alias('ankle:z'),
        pl.col(f"{subject_prefix}{ankle_marker}_X'").alias('ankle:dx'),
        pl.col(f"{subject_prefix}{ankle_marker}_Y'").alias('ankle:dy'),
        pl.col(f"{subject_prefix}{ankle_marker}_Z'").alias('ankle:dz'),
    ]
    vicon_df = vicon_df.with_columns(ankle_cols)

    # Foot markers are mandatory - try left foot first, fall back to right foot
    foot_marker = None
    if f'{subject_prefix}LFOOT_X' in vicon_df.columns:
        foot_marker = 'LFOOT'
    else:
        raise ValueError(f"No foot markers found for subject '{subject_prefix}' (expected LFOOT or RFOOT)")

    # Foot velocity derivatives are mandatory
    if f"{subject_prefix}{foot_marker}_X'" not in vicon_df.columns:
        raise ValueError(f"Missing foot velocity derivatives for marker {foot_marker}")

    foot_cols = [
        pl.col(f'{subject_prefix}{foot_marker}_X').alias('foot:x'),
        pl.col(f'{subject_prefix}{foot_marker}_Y').alias('foot:y'),
        pl.col(f'{subject_prefix}{foot_marker}_Z').alias('foot:z'),
        pl.col(f"{subject_prefix}{foot_marker}_X'").alias('foot:dx'),
        pl.col(f"{subject_prefix}{foot_marker}_Y'").alias('foot:dy'),
        pl.col(f"{subject_prefix}{foot_marker}_Z'").alias('foot:dz'),
    ]
    vicon_df = vicon_df.with_columns(foot_cols)

    # Estimate and compensate for time lag
    if compensate_lag:
        lag_sec = estimate_time_lag(vicon_df, imu_df)
        print(f"Estimated lag: {lag_sec * 1000:.2f} ms")
        imu_df = imu_df.with_columns([
            (pl.col('world_clock_utc') + pl.duration(seconds=lag_sec)).alias("lag_compensated_clock_utc")
        ])
    else:
        imu_df = imu_df.with_columns([
            pl.col('world_clock_utc').alias("lag_compensated_clock_utc")
        ])

    return imu_df, vicon_df


def dataframe_to_imu_sequence(
    imu_df: pl.DataFrame,
    time_col: str = 'world_clock_utc',
    start_time: None = None,
    end_time: None = None
) -> ImuSequence:
    """
    Convert IMU dataframe to ImuSequence format.

    Conventions:
        - Quaternions returned in (w, x, y, z) order
        - Timestamps converted to float32 seconds (relative to first sample)
        - All arrays are float32 for JAX compatibility

    Args:
        imu_df: Polars DataFrame with IMU data (raw quaternion and accel columns)
        time_col: Name of timestamp column to use
        start_time: Optional start time for trimming data (datetime)
        end_time: Optional end time for trimming data (datetime)

    Returns:
        ImuSequence with JAX arrays (all float32)
    """
    # Filter by time range if specified
    df = imu_df
    if start_time is not None:
        df = df.filter(pl.col(time_col) >= start_time)
    if end_time is not None:
        df = df.filter(pl.col(time_col) <= end_time)

    if len(df) == 0:
        raise ValueError("No data in specified time range")

    # Extract arrays and convert to JAX arrays
    # Accelerometer (m/s^2) - keep as raw body-frame data
    accel = jnp.array(df.select(['RAW:a_x', 'RAW:a_y', 'RAW:a_z']).to_numpy(), dtype=jnp.float32)  # type: ignore[call-overload]

    # Gyroscope (rad/s)
    gyro = jnp.array(df.select(['RAW:w_x', 'RAW:w_y', 'RAW:w_z']).to_numpy(), dtype=jnp.float32)  # type: ignore[call-overload]

    # Quaternion (w, x, y, z) - keep as raw quaternions
    quat = jnp.array(df.select(['RAW:qw', 'RAW:qx', 'RAW:qy', 'RAW:qz']).to_numpy(), dtype=jnp.float32)  # type: ignore[call-overload]

    # Delta velocity (m/s)
    dv = jnp.array(df.select(['RAW:dv_x', 'RAW:dv_y', 'RAW:dv_z']).to_numpy(), dtype=jnp.float32)  # type: ignore[call-overload]

    # Delta quaternion (w, x, y, z)
    dq = jnp.array(df.select(['RAW:dq_w', 'RAW:dq_x', 'RAW:dq_y', 'RAW:dq_z']).to_numpy(), dtype=jnp.float32)  # type: ignore[call-overload]

    # Timestamps (convert to seconds as float)
    time_ns = df[time_col].dt.timestamp(time_unit='ns').to_numpy()  # type: ignore[attr-defined]
    t0 = np.nanmin(time_ns)
    timestamps_sec = (time_ns - t0) / 1e9

    timestamp = jnp.array(timestamps_sec, dtype=jnp.float32)

    return ImuSequence(
        accel=accel,
        gyro=gyro,
        quat=quat,
        dv=dv,
        dq=dq,
        timestamp=timestamp
    )


def load_trial_as_imu_sequence(
    mcap_file_path: str,
    csv_file_path: str,
    timestamps_file_path: str,
    base_name: str,
    compensate_lag: bool = True,
    thresholds: dict[str, float] | None = None
) -> tuple[ImuSequence, ViconSequence, StanceSequence]:
    """
    Load trial data and convert to ImuSequence, ViconSequence, and StanceSequence formats.

    This is the main high-level function for loading a complete trial.
    Both IMU and Vicon data are required. The ViconSequence and StanceSequence are
    interpolated to match the IMU timestamps exactly, ensuring perfect alignment.

    **Alignment behavior:**
    - Uses intersection of IMU and Vicon time ranges
    - ViconSequence and StanceSequence are interpolated to match IMU timestamps exactly
    - All sequences have EXACTLY the same length and aligned timestamps
    - Timestamps are relative to first sample (in seconds)

    **Data conventions:**
    - Quaternions: (w, x, y, z) order
    - All positions/velocities in meters and m/s (converted from Vicon mm)
    - Stance values are 0.0 (no stance) or 1.0 (stance)
    - All data as float32 JAX arrays
    - Time lag between IMU and Vicon is estimated and compensated if enabled

    Args:
        mcap_file_path: Path to MCAP file containing IMU data (required)
        csv_file_path: Path to CSV file containing Vicon data (required)
        timestamps_file_path: Path to timestamps Excel file (required)
        base_name: Trial name (required for quaternion frame conversion)
        compensate_lag: Whether to estimate and compensate for time lag (default: True)

    Returns:
        Tuple of (ImuSequence, ViconSequence, StanceSequence).
        All sequences have EXACTLY the same length and aligned timestamps.
        All arrays are float32 for JAX compatibility.

    Raises:
        ValueError: If any required file is missing or data cannot be loaded
    """
    # Load raw data
    imu_df, vicon_df = load_trial_data(
        mcap_file_path=mcap_file_path,
        csv_file_path=csv_file_path,
        timestamps_file_path=timestamps_file_path,
        base_name=base_name,
        compensate_lag=compensate_lag
    )

    # Vicon data must exist
    if vicon_df is None:
        raise ValueError(f"No Vicon data found for trial {base_name}")

    # Determine time column
    time_col = 'lag_compensated_clock_utc' if 'lag_compensated_clock_utc' in imu_df.columns else 'world_clock_utc'

    # Use intersection: only where both IMU and Vicon data exist
    imu_start_time = imu_df[time_col].min()
    imu_end_time = imu_df[time_col].max()
    vicon_start_time = vicon_df['world_clock_utc'].min()
    vicon_end_time = vicon_df['world_clock_utc'].max()
    assert imu_start_time is not None and imu_end_time is not None, "IMU timestamps should not be None"
    assert vicon_start_time is not None and vicon_end_time is not None, "Vicon timestamps should not be None"

    start_time = max(imu_start_time, vicon_start_time)
    end_time = min(imu_end_time, vicon_end_time)

    # Filter IMU data to intersection
    imu_df_filtered = imu_df.filter(
        (pl.col(time_col) >= start_time) & (pl.col(time_col) <= end_time)
    )

    # Trim any IMU samples that would force extrapolation when interpolating Vicon data.
    # Tolerance in nanoseconds (2 milliseconds)
    tolerance_ns = 2_000_000

    # Drop IMU samples that sit outside the available Vicon coverage (with a small tolerance).
    imu_start = imu_df_filtered[time_col][0]
    imu_end = imu_df_filtered[time_col][-1]

    vicon_after_start = vicon_df.filter(pl.col('world_clock_utc') >= imu_start)
    vicon_before_end = vicon_df.filter(pl.col('world_clock_utc') <= imu_end)

    if len(vicon_after_start) == 0 or len(vicon_before_end) == 0:
        raise ValueError(f"No Vicon samples overlap IMU range for trial {base_name}")

    vicon_start = vicon_after_start['world_clock_utc'][0]
    vicon_end = vicon_before_end['world_clock_utc'][-1]

    lower_bound = vicon_start - pl.duration(nanoseconds=tolerance_ns)
    upper_bound = vicon_end + pl.duration(nanoseconds=tolerance_ns)

    # Count samples that will be dropped
    before_len = len(imu_df_filtered)
    imu_df_filtered = imu_df_filtered.filter(
        (pl.col(time_col) >= lower_bound) & (pl.col(time_col) <= upper_bound)
    )
    after_len = len(imu_df_filtered)

    if before_len != after_len:
        dropped = before_len - after_len
        if len(imu_df_filtered) == 0:
            raise ValueError(f"No overlapping data between IMU and Vicon for trial {base_name}")
        print(f"  Dropped {dropped} IMU samples outside Vicon coverage (±{tolerance_ns/1e6:.0f}ms tolerance) to avoid extrapolation.")

    # Align start/end exactly with the supported Vicon range
    start_time = vicon_start
    end_time = vicon_end

    imu_df_filtered = imu_df_filtered.filter(
        (pl.col(time_col) >= start_time) & (pl.col(time_col) <= end_time)
    )

    # Refresh bounds after any trimming
    vicon_interp_start = vicon_start
    vicon_interp_end = vicon_end

    if len(imu_df_filtered) == 0:
        raise ValueError(f"No overlapping data between IMU and Vicon for trial {base_name}")

    # Convert to ImuSequence
    imu_seq = dataframe_to_imu_sequence(
        imu_df=imu_df_filtered,
        time_col=time_col,
        start_time=None,  # Already filtered above
        end_time=None
    )

    # Convert to ViconSequence
    # Vicon is interpolated to match IMU timestamps exactly
    vicon_seq = dataframe_to_vicon_sequence(
        vicon_df=vicon_df,
        imu_timestamps=imu_df_filtered[time_col],
        time_col='world_clock_utc',
        start_time=vicon_interp_start,
        end_time=vicon_interp_end
    )


    # Generate stance sequence
    stance_seq = dataframe_to_stance_sequence(
        vicon_df=vicon_df,
        imu_timestamps=imu_df_filtered[time_col],
        start_time=vicon_interp_start,
        end_time=vicon_interp_end,
        thresholds=thresholds
    )

    # Verify alignment (sanity check)
    assert len(vicon_seq.timestamp) == len(imu_seq.timestamp), \
        f"Alignment error: Vicon has {len(vicon_seq.timestamp)} samples but IMU has {len(imu_seq.timestamp)}"
    assert len(stance_seq.timestamp) == len(imu_seq.timestamp), \
        f"Alignment error: Stance has {len(stance_seq.timestamp)} samples but IMU has {len(imu_seq.timestamp)}"

    return imu_seq, vicon_seq, stance_seq


def load_multiple_trials(
    data_directory: str,
    trial_names: list[str] | None = None,
    thresholds: dict[str,float] | None = None
) -> dict[str, tuple[ImuSequence, ViconSequence, StanceSequence]]:
    """
    Load multiple trials from a data directory.

    Each trial must have both MCAP (IMU) and CSV (Vicon) files.
    Trials without corresponding Vicon data will be skipped with a warning.

    Args:
        data_directory: Path to directory containing data files
        trial_names: Optional list of trial names to load (if None, loads all with matching CSV)

    Returns:
        Dictionary mapping trial names to (ImuSequence, ViconSequence, StanceSequence) tuples.
        All sequences are perfectly time-aligned (same length, same timestamps).
    """
    import os

    timestamps_file = os.path.join(data_directory, "timestamps.xlsx")

    if not os.path.exists(timestamps_file):
        raise FileNotFoundError(f"timestamps.xlsx not found in {data_directory}")

    # Find all MCAP files
    mcap_files = [f for f in os.listdir(data_directory) if f.endswith('.mcap')]

    results: dict[str, tuple[ImuSequence, ViconSequence, StanceSequence]] = {}
    skipped = []

    for mcap_file in mcap_files:
        base_name = mcap_file[:-5]  # Remove .mcap extension

        # Skip if not in requested trials
        if trial_names is not None and base_name not in trial_names:
            continue

        mcap_path = os.path.join(data_directory, mcap_file)
        csv_path = os.path.join(data_directory, f"{base_name}.csv")

        # Check if CSV exists (required!)
        if not os.path.exists(csv_path):
            print(f"Warning: Skipping {base_name} - no corresponding CSV file found")
            skipped.append(base_name)
            continue

        try:
            print(f"Loading trial: {base_name}")
            imu_seq, vicon_seq, stance_seq = load_trial_as_imu_sequence(
                mcap_file_path=mcap_path,
                csv_file_path=csv_path,
                timestamps_file_path=timestamps_file,
                base_name=base_name,
                compensate_lag=True,
                thresholds=thresholds
            )
            results[base_name] = (imu_seq, vicon_seq, stance_seq)
            print(f"  Loaded {len(imu_seq.timestamp)} samples (aligned IMU + Vicon + Stance)")

        except Exception as e:
            print(f"  Error loading {base_name}: {e}")
            skipped.append(base_name)

    if skipped:
        print(f"\nSkipped {len(skipped)} trials: {', '.join(skipped)}")

    return results
