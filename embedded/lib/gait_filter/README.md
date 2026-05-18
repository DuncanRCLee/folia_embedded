# Gait Filter Library - Design Document

## Overview

The Gait Filter library provides **step-to-step position estimation** optimized for terrain classification using data from the Movella MTI-3 AHRS IMU. The filter is designed for embedded deployment on Arduino Portenta H7, providing 20 Hz output for downstream neural network terrain classifiers.

## Design Philosophy

### Problem Statement
- **Goal**: Predict walking gait terrain type (level ground, slopes, stairs) with <1 step latency
- **Key Insight**: Vertical displacement provides the strongest differentiation between terrains
  - Level ground: ~0 net vertical change per step
  - Slopes: Consistent vertical change proportional to horizontal distance
  - Stairs: Large discrete vertical changes (~15-20cm per step)

### IMU Mounting: Shank vs Foot

This filter is designed for **shank-mounted IMU** (on the lower leg), which is more practical for many applications but requires different constraints than foot-mounted sensors:

| Aspect | Foot-Mounted | Shank-Mounted |
|--------|--------------|---------------|
| ZUPT constraint | v = 0 (foot stationary) | v_ankle = 0 (pivot constraint) |
| During stance | IMU is stationary | IMU rotates about ankle |
| Lever arm | Not needed | Required (IMU to ankle distance) |
| Detection | Simple threshold | SHOE detector recommended |

### How Integration Error is Controlled

The filter uses two key mechanisms to bound integration drift:

1. **SHOE Detector (Skog et al.)** - Generalized Likelihood Ratio Test for optimal stance detection
2. **Pivot Constraint** - During stance, ankle velocity is zero while shank rotates about it

#### SHOE Detector

The Stance Hypothesis Optimal Estimation (SHOE) detector uses a GLRT:

```
T = (1/W) × Σ[ ||a_k - g·(a_k/||a_k||)||² / σ_a² + ||ω_k||² / σ_w² ]
```

- First term: acceleration deviation from pure gravity direction
- Second term: gyroscope magnitude (should be ~0 during stance)
- Stance detected when T < γ (threshold)

#### Pivot Constraint

For shank-mounted IMU, during stance the ankle acts as a fixed pivot:

```
v_ankle = v_imu + R × (ω × r_imu_to_ankle) = 0
```

This gives us:
```
v_imu = -R × (ω × r_imu_to_ankle)
```

The filter applies a Kalman correction with this expected velocity rather than assuming v=0.

## MTI-3 Output Selection

The MTI-3 provides several output options. This filter is designed to leverage the optimal combination:

| Output | XDI Code | Usage in Filter |
|--------|----------|-----------------|
| **DeltaV (dv)** | `0x4010` | Primary - Pre-integrated velocity change in body frame |
| **DeltaQ (dq)** | `0x8030` | Primary - Pre-integrated rotation change |
| **Quaternion** | `0x2010` | Orientation from MTI-3 XKF (trusted) |
| **FreeAcceleration** | `0x4030` | Fallback - Gravity-compensated accel in world frame |
| **Acceleration** | `0x4020` | Raw fallback |
| **RateOfTurn** | `0x8020` | ZUPT detection |

### Why dv/dq?

The dv/dq outputs are ideal for foot-mounted IMU applications:

1. **Pre-compensated**: MTI-3 handles coning/sculling compensation internally
2. **Lower noise**: Integration acts as low-pass filter
3. **Drift-bounded**: Per-sample integration error is smaller
4. **Easy rotation**: dq provides clean rotation increments

```
dv[k] ≈ ∫_{t_k}^{t_{k+1}} a(t) dt   (body frame)
dq[k] ≈ exp(0.5 * ∫_{t_k}^{t_{k+1}} ω(t) dt)
```

## Filter Architecture

### State Vector (9D Error State)
```
x = [position(3), velocity(3), orientation_error(3)]
```

The filter uses an **Error-State Kalman Filter (ESKF)** approach:
- Nominal state propagated with measurements
- Error state corrected with ZUPT
- Orientation from MTI-3 XKF is trusted (not estimated)

### Prediction Step (100 Hz)

```cpp
// Rotate dv from body to world frame
Vec3f dv_world = R * dv;

// Remove gravity (dv includes gravity effect)
dv_world += g * dt;

// Update velocity and position
velocity += dv_world;
position += velocity * dt;
```

### Zero Velocity Update (ZUPT)

During detected stance phase:
```cpp
// Measurement: velocity should be zero
y = [0, 0, 0]ᵀ
H = [0₃ₓ₃  I₃ₓ₃  0₃ₓ₃]  // Observing velocity

// Standard Kalman update
K = P * Hᵀ * (H * P * Hᵀ + R)⁻¹
x = x + K * (y - v)
P = (I - K*H) * P
```

### Stance Detection

Uses variance of acceleration magnitude and gyroscope magnitude:
```cpp
bool inStance = (accel_variance < threshold_a) && 
                (gyro_magnitude < threshold_w) &&
                (|accel_mag - g| < 2.0);
```

### Step-Relative Reset

At each heel strike:
1. Generate step summary (for classification features)
2. Reset position to zero
3. Reset position covariance (high confidence at stance)
4. Increment step counter

This ensures:
- **Consistency**: Each step starts from known state
- **Bounded drift**: Errors don't accumulate across steps
- **Classification-ready**: Features are step-normalized

## Output Structure

### Output Frame Options

The filter supports two coordinate frames for position and velocity output:

```cpp
enum class OutputFrame {
    ENU = 0,              // Standard East-North-Up world frame
    HEADING_RELATIVE = 1  // Forward/Lateral/Vertical aligned with walking direction
};
```

**Both frames are always computed and available in the output structure**, regardless of the `output_frame` setting. Choose based on your application:

| Frame | Best For |
|-------|----------|
| **ENU** | GPS fusion, absolute navigation, mapping |
| **Sagittal-Plane-Aligned** | Terrain classification, gait analysis, direction-agnostic ML |

#### Sagittal Plane Alignment

The heading-relative frame provides **stable sagittal plane alignment** by using the **walking direction computed from step displacement** at each heel strike, rather than instantaneous IMU heading:

- **Forward**: Direction of actual walking (from horizontal displacement of previous step)
- **Lateral**: Perpendicular to forward (positive = left), lies in frontal plane
- **Vertical**: Up (unchanged from ENU Z)

This is essential for shank-mounted IMUs because the shank rotates during the gait cycle, but the walking direction remains constant. Using step displacement ensures:

1. **Stability**: Walking direction doesn't change during a step
2. **Accuracy**: Measures actual direction of travel, not where the leg points
3. **Consistency**: Same gait pattern produces same output regardless of compass heading

The transformation:
```
Forward = cos(walking_dir) × East + sin(walking_dir) × North
Lateral = -sin(walking_dir) × East + cos(walking_dir) × North
Vertical = Up (unchanged)
```

### Continuous Output (20 Hz)
```cpp
struct GaitFilterOutput {
    // Step-relative position (ENU world frame)
    float pos_x, pos_y, pos_z;  // meters (East, North, Up)
    
    // Velocity (ENU world frame)
    float vel_x, vel_y, vel_z;  // m/s
    
    // Sagittal-plane-aligned position (walking direction frame)
    float pos_forward;          // Forward position in walking direction (m)
    float pos_lateral;          // Lateral position perpendicular to walking (m, positive = left)
    float pos_vertical;         // Vertical position (m) - same as pos_z
    
    // Sagittal-plane-aligned velocity
    float vel_forward;          // Forward velocity in walking direction (m/s)
    float vel_lateral;          // Lateral velocity (m/s, positive = left)
    float vel_vertical;         // Vertical velocity (m/s) - same as vel_z
    
    // Walking direction used for transformation
    float heading;              // Walking direction angle (radians), from step displacement
    
    // Step metrics
    float step_height;          // max vertical displacement
    float step_length;          // horizontal distance
    float step_duration_ms;
    
    // Orientation
    float pitch, roll, yaw;     // radians
    
    // Phase and quality
    uint8_t phase;              // stance/swing
    float position_uncertainty;
};
```

### Per-Step Summary (at heel strike)
```cpp
struct StepSummary {
    float total_height_change;   // Net vertical (key for terrain!)
    float max_height, min_height;
    float horizontal_distance;
    float swing_time_ms, stance_time_ms;
    float peak_vertical_velocity;
    float height_to_distance_ratio;  // Terrain indicator
};
```

## Terrain Classification Features

The filter provides key features for terrain classification:

| Feature | Level Ground | Slope (Up) | Slope (Down) | Stairs (Up) | Stairs (Down) |
|---------|--------------|------------|--------------|-------------|---------------|
| `total_height_change` | ~0 | +5-15cm | -5-15cm | +15-20cm | -15-20cm |
| `height_to_dist_ratio` | <0.05 | 0.05-0.2 | 0.05-0.2 | >0.3 | >0.3 |
| `max_height` | 5-10cm | 10-15cm | 5-10cm | 15-25cm | 10-15cm |
| `peak_vert_velocity` | 0.3-0.5 | 0.4-0.7 | 0.5-0.8 | 0.6-1.0 | 0.8-1.2 |

## Implementation Notes

### Performance on Portenta H7 (M7 Core @ 480MHz)

The filter is optimized for real-time execution at 100Hz:

| Operation | Estimated Time | Notes |
|-----------|----------------|-------|
| Prediction | ~15-25µs | Optimized covariance using block operations |
| ZUPT correction | ~10-15µs | Only when in stance phase |
| Output generation | ~5µs | Simple copy operations |
| **Total per sample** | **~25-40µs** | Well under 10ms budget |

**CPU Budget**: At 100Hz, each sample has 10ms. The filter uses <0.5% CPU.

### Memory Usage
- Filter state: ~500 bytes (9x9 covariance + state vectors)
- Output buffer: ~64 bytes
- Configuration: ~40 bytes
- Total: **<1KB RAM**

### Optimization Techniques
1. **Block covariance updates**: Exploits sparse F matrix structure
2. **Single-precision floats**: Sufficient accuracy, faster on M7 FPU
3. **Pre-computed constants**: dt², sigma² computed once
4. **In-place operations**: Minimal temporary allocations
5. **Sparse H exploitation**: ZUPT uses H=[0 I 0] structure directly

### Configuration
```cpp
FilterConfig config;
config.dt_imu = 0.01f;      // 100 Hz IMU rate
// Filter outputs at 100Hz. Downstream classifier handles decimation to 20Hz.

// SHOE detector (Skog et al.)
config.use_shoe_detector = true;
config.shoe_sigma_a = 0.5f;      // Accel noise for SHOE (m/s²)
config.shoe_sigma_w = 0.2f;      // Gyro noise for SHOE (rad/s)
config.shoe_gamma = 1e6f;        // Detection threshold (tune this!)
config.shoe_window_size = 5;     // 50ms window

// Pivot constraint for shank-mounted IMU
config.use_pivot_constraint = true;
config.r_imu_to_ankle = Vec3f(0, 0, -0.15f);  // 15cm below IMU
config.pivot_velocity_noise = 0.02f;

// ZUPT tuning
config.zupt_accel_threshold = 0.4f;  // Variance threshold
config.zupt_gyro_threshold = 0.25f;  // rad/s
config.zupt_velocity_noise = 0.005f; // Tight correction

// Noise parameters
config.sigma_accel_noise = 0.08f;    // m/s²/√Hz
config.sigma_gyro_noise = 0.008f;    // rad/s/√Hz
```

## Tuning Guide

### SHOE Detector Threshold (γ)

The `shoe_gamma` threshold is critical for proper stance detection:

| γ Value | Behavior | When to Use |
|---------|----------|-------------|
| 1e5 | Very strict | Slow walking, flat ground |
| 1e6 | Moderate | Normal walking (default) |
| 1e7 | Lenient | Fast walking, stairs |
| 1e8 | Very lenient | Running, variable terrain |

**Tuning procedure:**
1. Log `getLastShoeStatistic()` while walking
2. Observe T values during clear stance phases (T should be low)
3. Observe T values during swing (T should be high)
4. Set γ between these ranges, closer to stance values

### Lever Arm Calibration

The vector from IMU to ankle (`r_imu_to_ankle`) affects pivot constraint accuracy:

```cpp
// Measure the distance from IMU center to ankle joint center
// In body frame: X=forward, Y=left, Z=up
// For typical shank mount with IMU above ankle:
config.r_imu_to_ankle = Vec3f(0.02f, 0.0f, -0.12f);  // 2cm forward, 12cm below
```

**Signs matter!** If ankle is below IMU, z-component should be negative.

### Debugging Integration Drift

If you see excessive drift:

1. **Check ZUPT detection**: Monitor `isZuptActive()` - should be true during stance (~40% of gait cycle)
2. **Check SHOE statistic**: Use `getLastShoeStatistic()` to verify threshold is appropriate
3. **Verify lever arm**: Incorrect lever arm causes systematic errors during pivot constraint
4. **Adjust noise parameters**: Increase `pivot_velocity_noise` if corrections are too aggressive

## Integration with Main Loop

```cpp
#include <gait_filter.hpp>
#include <gait_filter_integration.hpp>

void setup() {
    // ... MTI-3 setup ...
    
    // Option 1: Use defaults (15cm lever arm)
    gait::initGaitFilter();
    
    // Option 2: Specify your lever arm
    // gait::initGaitFilterWithLeverArm(gait::Vec3f(0.02f, 0, -0.12f));
}

void loop() {
    // Read MTI-3
    float* dv = mti->getDeltaV();
    float* dq = mti->getDeltaQ();
    float* quat = mti->getQuaternion();
    
    // Process through filter (100 Hz)
    gait::processGaitFilter(dv, dq, quat, nullptr, mti->getSampleTimeFine());
    
    // Output available every sample (100 Hz)
    if (gait::hasGaitOutput()) {
        auto out = gait::getGaitOutput();
        // Feed to NN classifier input buffer
        
        // Debug: check SHOE statistic
        // Serial.printf("T=%.0f zupt=%d\n", gait::g_gaitFilter->getLastShoeStatistic(), out.zupt_active);
    }
    
    // At heel strike, complete step features available
    if (gait::hasNewStep()) {
        auto step = gait::getStepSummary();
        // Immediate terrain classification possible
    }
}
```

## References

1. Xsens MTI-3 User Manual - dv/dq output description
2. Skog et al. "Zero-Velocity Detection—An Algorithm Evaluation" (2010) - SHOE detector
3. Wagstaff et al. "Improving Foot-Mounted Inertial Navigation Through Real-Time Motion Classification" (2017)
4. Solà "Quaternion kinematics for the error-state Kalman filter" (2017)

## Future Improvements

1. **Adaptive SHOE threshold**: Adjust γ based on walking speed/activity
2. **Soft stance detection**: Use SHOE statistic as confidence weight
3. **Lever arm estimation**: Online calibration of IMU-to-ankle vector
4. **Barometer fusion**: If available, for improved vertical accuracy
