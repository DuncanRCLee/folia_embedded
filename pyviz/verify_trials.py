#!/usr/bin/env python3
"""
Verification script to ensure data loading produces consistent metrics.
Run this before and after refactoring to verify no data changes occurred.
"""

from pathlib import Path
import numpy as np

from filter3.data_loader import load_trial_as_imu_sequence

def compute_trial_metrics(trial_name: str, data_dir: Path = Path("data")) -> dict:
    """Compute metrics for a single trial."""
    
    timestamps_path = data_dir / "timestamps.xlsx"
    if not timestamps_path.exists():
        raise FileNotFoundError(f"timestamps.xlsx not found in {data_dir}")
    
    # Load trial with default stance detection thresholds
    thresholds = {
        'velocity_threshold_enter': 0.1,  # m/s
        'velocity_threshold_leave': 0.2,  # m/s
        'velocity_z_threshold': 0.5,      # m/s
        'min_stance_duration': 0.06,      # seconds
    }
    
    imu_seq, vicon_seq, stance_seq = load_trial_as_imu_sequence(
        mcap_file_path=str(data_dir / f"{trial_name}.mcap"),
        csv_file_path=str(data_dir / f"{trial_name}.csv"),
        timestamps_file_path=str(timestamps_path),
        base_name=trial_name,
        compensate_lag=True,
        thresholds=thresholds,
    )
    
    # Extract arrays
    t = np.asarray(imu_seq.timestamp)
    vicon_acc = np.asarray(vicon_seq.acceleration)
    vicon_vel = np.asarray(vicon_seq.velocity)
    vicon_vel_mag = np.asarray(stance_seq.velocity_magnitude)
    ankle_vel_mag = np.asarray(stance_seq.ankle_velocity_magnitude)
    stance = np.asarray(stance_seq.local_stance)
    
    # Compute metrics
    metrics = {
        'trial_name': trial_name,
        'num_samples': len(t),
        'duration': float(t[-1] - t[0]),
        'median_dt': float(np.median(np.diff(t))),
        
        # Vicon acceleration stats
        'vicon_acc_mean': vicon_acc.mean(axis=0).tolist(),
        'vicon_acc_std': vicon_acc.std(axis=0).tolist(),
        
        # Vicon velocity magnitude stats
        'vicon_vel_mag_mean': float(vicon_vel_mag.mean()),
        'vicon_vel_mag_std': float(vicon_vel_mag.std()),
        'vicon_vel_mag_min': float(vicon_vel_mag.min()),
        'vicon_vel_mag_max': float(vicon_vel_mag.max()),
        
        # Ankle velocity magnitude stats
        'ankle_vel_mag_mean': float(ankle_vel_mag.mean()),
        'ankle_vel_mag_std': float(ankle_vel_mag.std()),
        'ankle_vel_mag_min': float(ankle_vel_mag.min()),
        'ankle_vel_mag_max': float(ankle_vel_mag.max()),
        
        # Stance stats
        'stance_mean': float(stance.mean()),
        'stance_sum': float(stance.sum()),
        'num_stance_samples': int((stance > 0.5).sum()),
    }
    
    return metrics


def main():
    """Run verification on a test trial."""
    data_dir = Path("data")
    trial_name = "back_01"  # Use a representative trial
    
    print(f"Verifying trial: {trial_name}")
    print("=" * 60)
    
    try:
        metrics = compute_trial_metrics(trial_name, data_dir)
        
        print(f"\nTrial: {metrics['trial_name']}")
        print(f"  Samples: {metrics['num_samples']}")
        print(f"  Duration: {metrics['duration']:.2f} s")
        print(f"  Median dt: {metrics['median_dt']*1000:.2f} ms")
        
        print(f"\nVicon Acceleration (m/s²):")
        print(f"  Mean: [{metrics['vicon_acc_mean'][0]:.4f}, {metrics['vicon_acc_mean'][1]:.4f}, {metrics['vicon_acc_mean'][2]:.4f}]")
        print(f"  Std:  [{metrics['vicon_acc_std'][0]:.4f}, {metrics['vicon_acc_std'][1]:.4f}, {metrics['vicon_acc_std'][2]:.4f}]")
        
        print(f"\nVicon Velocity Magnitude (m/s):")
        print(f"  Mean: {metrics['vicon_vel_mag_mean']:.4f}")
        print(f"  Std:  {metrics['vicon_vel_mag_std']:.4f}")
        print(f"  Range: [{metrics['vicon_vel_mag_min']:.4f}, {metrics['vicon_vel_mag_max']:.4f}]")
        
        print(f"\nAnkle Velocity Magnitude (m/s):")
        print(f"  Mean: {metrics['ankle_vel_mag_mean']:.4f}")
        print(f"  Std:  {metrics['ankle_vel_mag_std']:.4f}")
        print(f"  Range: [{metrics['ankle_vel_mag_min']:.4f}, {metrics['ankle_vel_mag_max']:.4f}]")
        
        print(f"\nStance Detection:")
        print(f"  Mean: {metrics['stance_mean']:.4f}")
        print(f"  Stance samples: {metrics['num_stance_samples']} / {metrics['num_samples']}")
        
        print("\n" + "=" * 60)
        print("✓ Verification complete")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
