import jax.numpy as jnp
from typing import NamedTuple

# TIME SERIES IO
class ImuSample(NamedTuple):
    """
    Single IMU measurement sample.

    Convention:
        - Quaternions are (w, x, y, z) order
        - Angular velocity in rad/s
        - Linear acceleration in m/s^2
        - Timestamp in seconds (as scalar JAX array)
    """
    accel: jnp.ndarray     # (3,) linear acceleration in m/s^2
    gyro:  jnp.ndarray     # (3,) angular velocity in rad/s
    quat:  jnp.ndarray     # (4,) quaternion (w, x, y, z)
    dv:    jnp.ndarray     # (3,) delta velocity in m/s
    dq:    jnp.ndarray     # (4,) delta quaternion (w, x, y, z)
    timestamp: jnp.ndarray # scalar array for JAX tracing compatibility (seconds)

class ImuSequence(NamedTuple):
    """
    Sequence of IMU measurements over time.

    All arrays should have consistent time dimension T.
    Typically dtype=jnp.float32 for memory efficiency.

    Convention:
        - Quaternions are (w, x, y, z) order
        - Timestamps are in seconds (relative to first sample)
    """
    accel: jnp.ndarray     # (T,3) linear acceleration in m/s^2
    gyro:  jnp.ndarray     # (T,3) angular velocity in rad/s
    quat:  jnp.ndarray     # (T,4) quaternion (w, x, y, z)
    dv:    jnp.ndarray     # (T,3) delta velocity in m/s
    dq:    jnp.ndarray     # (T,4) delta quaternion (w, x, y, z)
    timestamp: jnp.ndarray # (T,) timestamps in seconds

# VICON GROUND TRUTH
class ViconSequence(NamedTuple):
    """Ground truth data from Vicon motion capture system."""
    # IMU position (meters)
    position: jnp.ndarray       # (T,3) x,y,z position of IMU
    velocity: jnp.ndarray       # (T,3) x,y,z velocity of IMU (m/s)
    acceleration: jnp.ndarray   # (T,3) x,y,z acceleration of IMU (m/s^2)

    # Ankle position (meters)
    ankle_position: jnp.ndarray    # (T,3) x,y,z position of ankle
    ankle_velocity: jnp.ndarray    # (T,3) x,y,z velocity of ankle (m/s)

    # Foot position (meters)
    foot_position: jnp.ndarray     # (T,3) x,y,z position of foot
    foot_velocity: jnp.ndarray     # (T,3) x,y,z velocity of foot (m/s)

    timestamp: jnp.ndarray         # (T,) timestamps in seconds


# STANCE DETECTION
class StanceSequence(NamedTuple):
    """Stance phase detection from Vicon foot data."""
    local_stance: jnp.ndarray      # (T,) float array with 0.0 (no stance) or 1.0 (stance)
    
    # Velocity magnitudes (used for stance detection and analysis)
    velocity_magnitude: jnp.ndarray         # (T,) IMU velocity magnitude (m/s)
    ankle_velocity_magnitude: jnp.ndarray   # (T,) ankle velocity magnitude (m/s)
    
    timestamp: jnp.ndarray         # (T,) timestamps in seconds
