"""
Differentiable training for ESKF filter using JAX.

This module implements gradient-based optimization of filter parameters
by minimizing the Mean Squared Error (MSE) between the predicted trajectory
and ground truth Vicon data.

Key features:
- Fully differentiable filter rollout using JAX
- MSE loss on position trajectory
- Adam optimizer with learning rate scheduling
- Train/validation split with early stopping
- Checkpoint saving for best models
- Support for multiple trials (batch optimization)

Usage:
    python train.py --data-dir data --epochs 100 --lr 0.001
"""

from __future__ import annotations

import argparse
import pickle
from functools import partial
from pathlib import Path
from typing import Iterable, Sequence, Dict, Tuple, Any, NamedTuple

import jax
import jax.numpy as jnp
import optax
from jax import grad, jit, vmap

from analyze import discover_trials, load_trial_sequences
from filter2.common import ImuSequence, ViconSequence, rollout
from filter2.data_loader import (
    TrialSequenceSplit,
    FirstChunkMetrics,
    split_aligned_sequences,
    compute_first_chunk_metrics,
)
from filter2.eskf_midpoint import ESKF15, ESKFParams
from filter2.detectors import (
    create_default_detectors,
    DetectorConfig,
    Detector,
)


class DetectorConfigParams(NamedTuple):
    """Learnable parameters for a single detector."""
    threshold: jnp.ndarray    # scalar
    sharpness: jnp.ndarray    # scalar (typically fixed, but can be learned)
    params: jnp.ndarray       # (P,) detector-specific parameters


class TrainableParams(NamedTuple):
    """
    Learnable parameters for the ESKF filter and detectors.
    
    This structure contains all the parameters that will be optimized
    during training. It's designed to be compatible with JAX's pytree
    system for automatic differentiation.
    
    Note: P0_diag is NOT included here as it's known to be zero (perfect initialization).
    """
    # Filter process noise parameters
    sig_a: jnp.ndarray      # scalar: accelerometer measurement noise
    sig_w: jnp.ndarray      # scalar: gyroscope measurement noise
    sig_aw: jnp.ndarray     # scalar: accelerometer bias random walk
    sig_ww: jnp.ndarray     # scalar: gyroscope bias random walk
    
    # Detector configurations (learnable thresholds and noise params)
    # Each detector has: threshold, sharpness, params array
    detector_configs: Tuple[DetectorConfigParams, ...]


def trainable_to_eskf_params(
    trainable: TrainableParams,
    gravity: jnp.ndarray
) -> ESKFParams:
    """
    Convert trainable parameters to ESKFParams for the filter.
    
    P0_diag is set to zeros because we have perfect initialization from Vicon.
    
    Note: ESKFParams expects float types but JAX will handle scalar arrays fine.
    """
    # P0 is zero because initialization is perfect (from ground truth)
    P0_diag = jnp.zeros(15, dtype=jnp.float32)
    
    return ESKFParams(
        gravity=gravity,
        sig_a=trainable.sig_a,  # Keep as JAX array for tracing
        sig_w=trainable.sig_w,
        sig_aw=trainable.sig_aw,
        sig_ww=trainable.sig_ww,
        P0_diag=P0_diag,
        Q=jnp.empty((0, 0))  # Will be constructed from sigmas in filter
    )


def trainable_to_detectors(
    trainable: TrainableParams,
    detector_fns: Tuple[Any, ...]
) -> Tuple[Detector, ...]:
    """Convert trainable detector parameters to Detector objects."""
    detectors = []
    for det_params, det_fn in zip(trainable.detector_configs, detector_fns):
        cfg = DetectorConfig(
            threshold=det_params.threshold,  # Keep as JAX array for tracing
            sharpness=det_params.sharpness,
            params=det_params.params
        )
        detectors.append(Detector(fn=det_fn, cfg=cfg))
    return tuple(detectors)


def initialize_trainable_params() -> TrainableParams:
    """
    Initialize trainable parameters with reasonable defaults.
    
    These values are based on typical IMU characteristics and
    will be optimized during training.
    
    Note: P0_diag is NOT included as it's known to be zero (perfect initialization).
    """
    # Create default detectors to extract their configurations
    default_detectors = create_default_detectors()
    
    # Extract detector function references and configs
    detector_configs = []
    for det in default_detectors:
        cfg = det.cfg
        detector_configs.append(
            DetectorConfigParams(
                threshold=jnp.array(cfg.threshold, dtype=jnp.float32),
                sharpness=jnp.array(cfg.sharpness, dtype=jnp.float32),
                params=jnp.array(cfg.params, dtype=jnp.float32)
            )
        )
    
    return TrainableParams(
        # Process noise sigmas (initial guesses)
        sig_a=jnp.array(0.1, dtype=jnp.float32),    # accel measurement noise
        sig_w=jnp.array(0.01, dtype=jnp.float32),   # gyro measurement noise
        sig_aw=jnp.array(0.001, dtype=jnp.float32), # accel bias random walk
        sig_ww=jnp.array(0.001, dtype=jnp.float32), # gyro bias random walk
        
        detector_configs=tuple(detector_configs)
    )


def compute_mse_loss(
    filter_positions: jnp.ndarray,
    ground_truth_positions: jnp.ndarray,
    mask: jnp.ndarray | None = None
) -> jnp.ndarray:
    """
    Compute Mean Squared Error between predicted and ground truth positions.
    
    Args:
        filter_positions: (T, 3) predicted positions from filter
        ground_truth_positions: (T, 3) ground truth positions from Vicon
        mask: Optional (T,) boolean mask for valid samples
        
    Returns:
        Scalar MSE loss
    """
    residual = filter_positions - ground_truth_positions
    squared_error = jnp.sum(residual ** 2, axis=-1)  # (T,)
    
    if mask is not None:
        squared_error = jnp.where(mask, squared_error, 0.0)
        return jnp.sum(squared_error) / jnp.maximum(jnp.sum(mask), 1.0)
    else:
        return jnp.mean(squared_error)


def compute_chunk_loss(
    trainable: TrainableParams,
    initial_state,
    chunk_imu: ImuSequence,
    chunk_vicon: jnp.ndarray,
    detector_fns: Tuple[Any, ...],
    gravity: jnp.ndarray,
    eskf_params,
    detectors
) -> jnp.ndarray:
    """Compute loss for a single chunk (differentiable)."""
    from filter2.common import ImuSample, correction_step
    
    filter = ESKF15()
    st = initial_state
    positions = []
    chunk_len = len(chunk_imu.timestamp)
    
    for i in range(chunk_len):
        imu_sample = ImuSample(
            accel=chunk_imu.accel[i],
            gyro=chunk_imu.gyro[i],
            quat=chunk_imu.quat[i],
            dv=chunk_imu.dv[i],
            dq=chunk_imu.dq[i],
            timestamp=chunk_imu.timestamp[i]
        )
        
        if i > 0:
            dt = float(chunk_imu.timestamp[i] - chunk_imu.timestamp[i-1])
        else:
            dt = 0.01  # Default dt for first sample
        
        st = filter.predict(st, imu_sample, dt, eskf_params)
        st, _ = correction_step(filter, st, imu_sample, eskf_params, detectors)
        positions.append(st.pos)
    
    chunk_positions = jnp.stack(positions)
    
    # Compute loss
    mask = jnp.all(jnp.isfinite(chunk_vicon), axis=-1)
    vicon_clean = jnp.where(jnp.isfinite(chunk_vicon), chunk_vicon, 0.0)
    
    return compute_mse_loss(chunk_positions, vicon_clean, mask)


def compute_loss_single_trial(
    trainable: TrainableParams,
    imu_seq: ImuSequence,
    vicon_seq: ViconSequence,
    detector_fns: Tuple[Any, ...],
    gravity: jnp.ndarray,
    init_metrics: FirstChunkMetrics,
    chunk_size: int = 5
) -> jnp.ndarray:
    """
    Compute MSE loss for a single trial.
    
    Args:
        trainable: Trainable parameters
        imu_seq: IMU data sequence
        vicon_seq: Ground truth Vicon sequence
        detector_fns: Tuple of detector functions
        gravity: Gravity vector (3,)
        init_sample_idx: Index of sample to use for initialization
        
    Returns:
        Scalar MSE loss
    """
    # Convert trainable params to filter params
    eskf_params = trainable_to_eskf_params(trainable, gravity)
    detectors = trainable_to_detectors(trainable, detector_fns)
    
    # Create filter instance
    filter = ESKF15()
    
    # Initialize using the specified sample
    init_sample = {
        'accel': imu_seq.accel[init_sample_idx],
        'gyro': imu_seq.gyro[init_sample_idx],
        'quat': imu_seq.quat[init_sample_idx],
    }
    
    # Run filter rollout
    traj, nll = rollout(
        filter=filter,
        params=eskf_params,
        detectors=detectors,
        imu_seq=imu_seq,
        init_dict=init_sample
    )
    
    # Extract positions from trajectory
    # traj is a State15 with fields stacked along first dimension
    filter_positions = traj.pos  # (T, 3)
    
    # Compute MSE loss against ground truth
    # Create mask for valid Vicon data
    mask = jnp.all(jnp.isfinite(vicon_seq.position), axis=-1)
    
    loss = compute_mse_loss(filter_positions, vicon_seq.position, mask)
    
    return loss


def compute_loss_batch(
    trainable: TrainableParams,
    trials: Sequence[Tuple[ImuSequence, ViconSequence]],
    detector_fns: Tuple[Any, ...],
    gravity: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute average loss across multiple trials.
    
    Args:
        trainable: Trainable parameters
        trials: List of (ImuSequence, ViconSequence) tuples
        detector_fns: Tuple of detector functions
        gravity: Gravity vector
        
    Returns:
        Scalar average loss across all trials
    """
    losses = []
    for imu_seq, vicon_seq in trials:
        loss = compute_loss_single_trial(
            trainable, imu_seq, vicon_seq, detector_fns, gravity
        )
        losses.append(loss)
    
    return jnp.mean(jnp.array(losses))


@jit
def train_step(
    trainable: TrainableParams,
    opt_state: Any,
    optimizer: optax.GradientTransformation,
    imu_seq: ImuSequence,
    vicon_seq: ViconSequence,
    detector_fns: Tuple[Any, ...],
    gravity: jnp.ndarray
) -> Tuple[TrainableParams, Any, jnp.ndarray]:
    """
    Single training step with gradient update.
    
    Args:
        trainable: Current trainable parameters
        opt_state: Optimizer state
        optimizer: Optax optimizer
        imu_seq: IMU data
        vicon_seq: Ground truth data
        detector_fns: Detector functions
        gravity: Gravity vector
        
    Returns:
        Updated trainable params, optimizer state, and loss value
    """
    # Compute loss and gradients
    loss, grads = jax.value_and_grad(compute_loss_single_trial)(
        trainable, imu_seq, vicon_seq, detector_fns, gravity
    )
    
    # Apply optimizer update
    updates, opt_state = optimizer.update(grads, opt_state, trainable)
    trainable = optax.apply_updates(trainable, updates)
    
    return trainable, opt_state, loss


def train_epoch(
    trainable: TrainableParams,
    opt_state: Any,
    optimizer: optax.GradientTransformation,
    train_trials: Sequence[Tuple[ImuSequence, ViconSequence]],
    detector_fns: Tuple[Any, ...],
    gravity: jnp.ndarray
) -> Tuple[TrainableParams, Any, float]:
    """
    Train for one epoch over all training trials.
    
    Returns:
        Updated params, optimizer state, and average training loss
    """
    total_loss = 0.0
    
    for imu_seq, vicon_seq in train_trials:
        trainable, opt_state, loss = train_step(
            trainable, opt_state, optimizer,
            imu_seq, vicon_seq, detector_fns, gravity
        )
        total_loss += float(loss)
    
    avg_loss = total_loss / len(train_trials)
    return trainable, opt_state, avg_loss


def validate(
    trainable: TrainableParams,
    val_trials: Sequence[Tuple[ImuSequence, ViconSequence]],
    detector_fns: Tuple[Any, ...],
    gravity: jnp.ndarray
) -> float:
    """
    Evaluate model on validation set.
    
    Returns:
        Average validation loss
    """
    val_loss = compute_loss_batch(trainable, val_trials, detector_fns, gravity)
    return float(val_loss)


def save_checkpoint(
    trainable: TrainableParams,
    epoch: int,
    train_loss: float,
    val_loss: float,
    path: Path
) -> None:
    """Save model checkpoint to disk."""
    checkpoint = {
        'trainable': trainable,
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
    }
    with open(path, 'wb') as f:
        pickle.dump(checkpoint, f)
    print(f"  Saved checkpoint to {path}")


def load_checkpoint(path: Path) -> Dict[str, Any]:
    """Load model checkpoint from disk."""
    with open(path, 'rb') as f:
        return pickle.load(f)


class TrialSummary(NamedTuple):
    """Summary statistics for a single trial split."""
    trial_name: str
    total_pairs: int
    first_chunk_pairs: int
    rest_chunk_pairs: int
    first_chunk_metrics: FirstChunkMetrics


def summarise_trial(trial_name: str, split: TrialSequenceSplit) -> TrialSummary:
    """Calculate summary statistics for the first chunk of a trial."""
    first_pairs = int(len(split.first_imu.timestamp))
    rest_pairs = int(len(split.rest_imu.timestamp))
    total_pairs = first_pairs + rest_pairs

    return TrialSummary(
        trial_name=trial_name,
        total_pairs=total_pairs,
        first_chunk_pairs=first_pairs,
        rest_chunk_pairs=rest_pairs,
        first_chunk_metrics=compute_first_chunk_metrics(split.first_imu),
    )


def process_trials(
    data_dir: Path,
    trials: Sequence[str],
    first_count: int = 50
) -> list[TrialSummary]:
    """Load each trial, split sequences, and build summaries."""
    summaries: list[TrialSummary] = []
    for trial in trials:
        imu_seq, vicon_seq = load_trial_sequences(
            data_dir=data_dir,
            trial_name=trial,
            compensate_lag=True
        )
        split = split_aligned_sequences(imu_seq, vicon_seq, first_count=first_count)
        summaries.append(summarise_trial(trial, split))
    return summaries


def discover_or_validate_trials(
    data_dir: Path,
    requested: Iterable[str] | None
) -> list[str]:
    """Return the list of trials to process, discovering if none provided."""
    if requested:
        return list(requested)
    return discover_trials(data_dir)


def split_train_val(
    trials: Sequence[str],
    val_ratio: float = 0.2
) -> Tuple[list[str], list[str]]:
    """Split trials into training and validation sets."""
    n_trials = len(trials)
    n_val = max(1, int(n_trials * val_ratio))
    
    # Use last trials for validation (deterministic split)
    train_trials = list(trials[:-n_val])
    val_trials = list(trials[-n_val:])
    
    return train_trials, val_trials


def load_all_trials(
    data_dir: Path,
    trial_names: Sequence[str],
    first_count: int = 50
) -> Dict[str, Tuple[ImuSequence, ViconSequence]]:
    """
    Load all specified trials into memory, removing the first chunk for initialization.
    
    Args:
        data_dir: Directory containing trial data
        trial_names: List of trial names to load
        first_count: Number of initial samples to remove (used for initialization/calibration)
    
    Returns:
        Dictionary mapping trial names to (ImuSequence, ViconSequence) tuples with first chunk removed
    """
    trials_data = {}
    for trial_name in trial_names:
        print(f"Loading {trial_name}...")
        imu_seq, vicon_seq = load_trial_sequences(
            data_dir=data_dir,
            trial_name=trial_name,
            compensate_lag=True
        )
        trials_data[trial_name] = (imu_seq, vicon_seq)
    return trials_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ESKF filter using gradient descent on MSE loss."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing trial files (.mcap/.csv/timestamps.xlsx).",
    )
    parser.add_argument(
        "--trial",
        dest="trials",
        action="append",
        default=None,
        help="Trial base name to use (can be repeated). If omitted, all trials are used.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate for Adam optimizer.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction of trials to use for validation.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory to save model checkpoints.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (epochs without improvement).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir
    checkpoint_dir = args.checkpoint_dir
    checkpoint_dir.mkdir(exist_ok=True)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory {data_dir} does not exist")

    # Discover trials
    all_trial_names = discover_or_validate_trials(data_dir, args.trials)
    if not all_trial_names:
        print(f"No trials found in {data_dir}")
        return
    
    print(f"Found {len(all_trial_names)} trials: {', '.join(all_trial_names)}")
    
    # Split into train/val
    train_trial_names, val_trial_names = split_train_val(
        all_trial_names, args.val_ratio
    )
    print(f"\nTrain trials ({len(train_trial_names)}): {', '.join(train_trial_names)}")
    print(f"Val trials ({len(val_trial_names)}): {', '.join(val_trial_names)}")
    
    # Load all trials into memory
    print("\nLoading trials...")
    all_trials_data = load_all_trials(data_dir, all_trial_names)
    
    train_trials = [all_trials_data[name] for name in train_trial_names]
    val_trials = [all_trials_data[name] for name in val_trial_names]
    
    # Initialize trainable parameters
    print("\nInitializing trainable parameters...")
    trainable = initialize_trainable_params()
    
    # Extract detector functions (these are not learned, only their configs)
    default_detectors = create_default_detectors()
    detector_fns = tuple(det.fn for det in default_detectors)
    
    # Define gravity vector (WSU frame: +z is up)
    gravity = jnp.array([0.0, 0.0, -9.81], dtype=jnp.float32)
    
    # Initialize optimizer
    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(trainable)
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(args.epochs):
        # Train for one epoch
        trainable, opt_state, train_loss = train_epoch(
            trainable, opt_state, optimizer,
            train_trials, detector_fns, gravity
        )
        
        # Validate
        val_loss = validate(trainable, val_trials, detector_fns, gravity)
        
        print(f"Epoch {epoch + 1}/{args.epochs}: "
              f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
        
        # Save checkpoint if best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            checkpoint_path = checkpoint_dir / "best_model.pkl"
            save_checkpoint(trainable, epoch, train_loss, val_loss, checkpoint_path)
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break
        
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pkl"
            save_checkpoint(trainable, epoch, train_loss, val_loss, checkpoint_path)
    
    print(f"\nTraining complete! Best validation loss: {best_val_loss:.6f}")
    print(f"Best model saved to {checkpoint_dir / 'best_model.pkl'}")


if __name__ == "__main__":
    main()
