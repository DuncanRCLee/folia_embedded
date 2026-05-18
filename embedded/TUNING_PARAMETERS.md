# System Tuning Parameters Reference

Comprehensive table of all configurable tuning parameters across the embedded system. Last updated: March 17, 2026.

---

## 1. Gait Filter – SHOE Detector (Stance Detection)

| Parameter | Current Value | Units | Description | Reference |
|-----------|---------------|-------|-------------|-----------|
| `sigma_accel_noise` | 0.08 | m/s²/√Hz | Accelerometer process noise for covariance propagation | gait_filter_integration.hpp:50 |
| `sigma_gyro_noise` | 0.008 | rad/s/√Hz | Gyroscope process noise for covariance propagation | gait_filter_integration.hpp:51 |
| `use_shoe_detector` | true | bool | Enable SHOE (Stance Hypothesis OE) foot-flat detector | gait_filter_types.hpp:140 |
| `shoe_sigma_a` | 0.5 | m/s² | Acceleration noise std for SHOE GLRT test statistic | gait_filter_integration.hpp:66 |
| `shoe_sigma_w` | 0.2 | rad/s | Gyroscope noise std for SHOE GLRT test statistic | gait_filter_integration.hpp:67 |
| `shoe_gamma_enter` | 75.0 | (unitless) | SHOE threshold for ENTERING foot-flat (T < γ_enter) | gait_filter_integration.hpp:63 |
| `shoe_gamma_exit` | 125.0 | (unitless) | SHOE threshold for LEAVING foot-flat (T > γ_exit, hysteresis) | gait_filter_integration.hpp:64 |
| `shoe_window_size` | 8 | samples | Window size W for SHOE averaging (max SHOE_BUFFER_SIZE=8) | gait_filter_types.hpp:84 |
| `shoe_debounce_enter` | 3 | samples | Consecutive samples below γ_enter to confirm stance entry (~30ms @ 100Hz) | gait_filter_integration.hpp:65 |
| `shoe_debounce_exit` | 3 | samples | Consecutive samples above γ_exit to confirm stance exit (~30ms @ 100Hz) | gait_filter_integration.hpp:65 |
| `shoe_min_flat_samples` | 30 | samples | Minimum foot-flat duration before exit is allowed (~300ms @ 100Hz) | gait_filter_types.hpp:86 |

---

## 2. Gait Filter – Omega-Sag Detector (Gait Phase)

| Parameter | Current Value | Units | Description | Reference |
|-----------|---------------|-------|-------------|-----------|
| `use_omega_sag_detector` | true | bool | Enable omega-sag sagittal angular velocity detector for heel strike/toe-off | gait_filter_types.hpp:104 |
| `omega_sag_hs_delta_threshold` | 0.3 | rad/s | Minimum \|Δw_sag\| to count as non-zero sign change (deadband) | gait_filter_types.hpp:93 |
| `omega_sag_toe_off_threshold` | -0.5 | rad/s | w_sag below this triggers toe-off detection | gait_filter_types.hpp:94 |
| `omega_sag_toe_off_deriv_threshold` | -0.15 | rad/s/frame | dw_sag/dt must also be below this for toe-off (prevents mid-stance transients) | gait_filter_types.hpp:95 |
| `omega_sag_toe_off_debounce` | 2 | samples | Consecutive frames meeting both toe-off conditions to confirm | gait_filter_types.hpp:96 |
| `omega_sag_refractory_samples` | 10 | samples | Minimum samples between heel-strikes (~100ms @ 100Hz) | gait_filter_types.hpp:97 |
| `omega_sag_min_stance_samples` | 20 | samples | Minimum samples in stance before toe-off is allowed (~200ms @ 100Hz) | gait_filter_types.hpp:98 |
| `omega_sag_sign_window` | 10 | samples | Frames of sign(Δw_sag) to scan for oscillation (max OMEGA_SIGN_BUF_SIZE=16) | gait_filter_types.hpp:99 |
| `omega_sag_quiescence_threshold` | 0.15 | rad/s | \|w_sag\| below this counts as quiescent (stationary) | gait_filter_types.hpp:100 |
| `omega_sag_quiescence_samples` | 50 | samples | Consecutive quiescent frames to auto-enter stance (~500ms @ 100Hz) | gait_filter_types.hpp:101 |
| `omega_sag_hs_neg_amplitude` | -1.5 | rad/s | Min negative w_sag excursion during swing for zero-crossing heel strike | gait_filter_types.hpp:102 |

---

## 3. Gait Filter – ADC Stance Detector (Optional Load Cell)

| Parameter | Current Value | Units | Description | Reference |
|-----------|---------------|-------|-------------|-----------|
| `use_adc_detector` | false | bool | Enable ADC-based strain-gauge stance detection (disabled by default) | gait_filter_types.hpp:119 |
| `adc_stance_enter` | 0.05 | V | Volts above baseline to enter stance (load detected) | gait_filter_types.hpp:115 |
| `adc_stance_exit` | 0.05 | V | Volts above baseline to exit stance (no load) | gait_filter_types.hpp:116 |
| `adc_min_stance_samples` | 20 | samples | Minimum samples in stance before exit check (~200ms @ 100Hz) | gait_filter_types.hpp:117 |

---

## 4. Gait Filter – Pivot Constraint (Shank-Mounted IMU)

| Parameter | Current Value | Units | Description | Reference |
|-----------|---------------|-------|-------------|-----------|
| `use_pivot_constraint` | true | bool | Use ankle pivot constraint instead of legacy v=0 ZUPT | gait_filter_types.hpp:141 |
| `r_imu_to_ankle` | (0, -0.02, -0.08) | m | Vector from IMU center to ankle joint in body frame (body-relative position) | gait_filter_types.hpp:128 |
| `pivot_velocity_noise` | 0.02 | m/s | Ankle velocity measurement noise for pivot constraint Kalman update | gait_filter_integration.hpp:79 |

---

## 5. Gait Filter – Legacy ZUPT (Foot-Mounted Alternative)

| Parameter | Current Value | Units | Description | Reference |
|-----------|---------------|-------|-------------|-----------|
| `use_shoe_detector` | true | bool | Use SHOE detector (if false, simple variance thresholds) | gait_filter_types.hpp:140 |
| `zupt_accel_threshold` | 0.4 | (unitless variance) | Acceleration variance threshold for legacy stance detection | gait_filter_integration.hpp:80 |
| `zupt_gyro_threshold` | 0.25 | rad/s | Gyroscope magnitude threshold for legacy stance detection | gait_filter_integration.hpp:81 |
| `zupt_window_size` | 5 | samples | Window size for legacy ZUPT variance computation (~50ms @ 100Hz) | gait_filter_integration.hpp:82 |
| `zupt_velocity_noise` | 0.005 | m/s | Velocity measurement noise for legacy ZUPT correction | gait_filter_integration.hpp:83 |

---

## 6. Gait Filter – Position & Heading

| Parameter | Current Value | Units | Description | Reference |
|-----------|---------------|-------|-------------|-----------|
| `apply_position_correction` | false | bool | If true, apply position Kalman correction during stance; if false, position is pure integral of velocity | gait_filter_types.hpp:143 |
| `use_toe_off_heading` | true | bool | If true, lock walking direction at toe-off from IMU heading; if false, compute from step displacement | gait_filter_types.hpp:146 |
| `vertical_damping` | 0.02 | (damping factor) | Exponential damping on vertical velocity to reduce drift during swing | gait_filter_types.hpp:149 |
| `step_height_threshold` | 0.03 | m | Minimum height change to count as step (not actively used in current code) | gait_filter_types.hpp:152 |
| `dt_imu` | 0.01 | s | IMU sample period / prediction step size (100 Hz) | gait_filter_types.hpp:154 |

---

## 7. Gait Filter – Covariance Initialization

| Parameter | Current Value | Units | Description | Reference |
|-----------|---------------|-------|-------------|-----------|
| `init_pos_cov` | 0.0001 | m² | Initial position error covariance (diagonal) | gait_filter_integration.hpp:88 |
| `init_vel_cov` | 0.001 | (m/s)² | Initial velocity error covariance (diagonal) | gait_filter_integration.hpp:89 |
| `init_ori_cov` | 0.00001 | rad² | Initial orientation error covariance (diagonal) | gait_filter_integration.hpp:90 |

---

## 8. Terrain Classifier (Edge Impulse)

| Parameter | Current Value | Units | Description | Reference |
|-----------|---------------|-------|-------------|-----------|
| `EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW` | 20 | slices | Number of prediction slices per model window (1000ms / 20 = 50ms stride = 20Hz output) | ei_classifier.hpp:17 |
| `EI_WINDOW_SIZE` | 100 | samples | Model window size (100 samples @ 100Hz = 1000ms) | ei_classifier.hpp:46 |
| `EI_SLICE_SIZE` | 5 | samples | Samples per slice (window / slices = 100 / 20) | ei_classifier.hpp:47 |
| `EI_FEATURES_PER_SAMPLE` | 4 | features | Input features per sample: vel_fwd, vel_z, w_sag, pitch | ei_classifier.hpp:45 |
| `EI_NUM_LABELS` | 6 | (count) | Number of terrain classes: decline, downstairs, idle, incline, level, upstairs | ei_classifier.hpp:48 |
| `EI_DEFAULT_THRESHOLD` | 0.6 | (confidence) | Default confidence threshold for classification output | ei_classifier.hpp:49 |
| `EI_CONFIDENCE_THRESHOLD` | 0.6 | (confidence) | Minimum confidence threshold for valid classification | ei_classifier.hpp:68 |

---

## 9. Gait Prediction Debouncing (Terrain Classifier Output)

| Parameter | Current Value | Units | Description | Reference |
|-----------|---------------|-------|-------------|-----------|
| `GAIT_DEBOUNCE_WINDOW` | 3 | samples | Number of consecutive predictions needed to confirm terrain label (@ 20Hz = 150ms) | control.cpp:232 |
| `GAIT_CONFIDENCE_THRESHOLD` | 80 | % (0-100) | Minimum per-prediction confidence for terrain classification | control.cpp:233 |

---

## 10. Classifier Initialization & Thresholds

| Parameter | Current Value | Units | Description | Reference |
|-----------|---------------|-------|-------------|-----------|
| `confidence_threshold` (api) | 0.6 | float | Global confidence threshold set in classifier initialization | control.cpp:212 |
| `AUTO_CONFIDENCE_THRESHOLD` | 60 | % (0-100) | Minimum confidence for AUTO motor control mode | control.cpp:39 |

---

## 11. Motor Control – Gating Windows

| Parameter | Current Value | Units | Description | Reference |
|-----------|---------------|-------|-------------|-----------|
| `NOLOAD_TIMER_MS` | 200 | ms | Debounce time required in no-load state before confirming no-load condition | control.cpp:46 |
| `NOLOAD_IMMEDIATE_WINDOW_MS` | 400 | ms | Time window after no-load entry during which deferred motor commands can execute immediately | control.cpp:50 |

---

## 12. Motor Hardware – Barrel Cam & Kinematics

| Parameter | Current Value | Units | Description | Reference |
|-----------|---------------|-------|-------------|-----------|
| `MOTOR_BAUD_RATE` | 115200 | baud | Serial communication rate to Vertiq motor | vertiq_motor.hpp:13 |
| `MAX_STATE` | 5 | (count) | Number of discrete motor states (0-4, i.e., 5 total) | vertiq_motor.hpp:14 |
| `CAM_LEAD_MM` | 12.0 | mm | Linear stroke per revolution (barrel cam specification) | vertiq_motor.hpp:17 |
| `LINEAR_STEP_MM` | 2.8 | mm | Linear displacement per motor state transition | vertiq_motor.hpp:18 |
| `ROTATION_PER_STATE` | (computed) | rad | Rotation angle per state: (LINEAR_STEP_MM / CAM_LEAD_MM) × 2π | vertiq_motor.hpp:19 |
| `TRAJ_DURATION` | 0.2 | s | Time to move between adjacent motor states (200ms) | vertiq_motor.hpp:20 |

---

## 13. Communication

| Parameter | Current Value | Units | Description | Reference |
|-----------|---------------|-------|-------------|-----------|
| `BAUDRATE` | 115200 | baud | Serial communication rate (MTI-3 IMU + host debug/telemetry) | main_prelude.hpp:59 |
| `ADC_COUNTS_TO_VOLTS` | 2.048 / 8388608 | V/count | ADS1220 ADC conversion factor (FSR 2.048V, 2^23 counts) | main.cpp:44 |
| `BATCH_SIZE` | 4 | (count) | Number of packets per network batch transmission | main_prelude.hpp:65 |
| `GRAV` | 9.80665 | m/s² | Standard gravity constant for calculations | main_prelude.hpp:48 |
| `MAX_CLASSIFIER_LABELS` | 16 | (count) | Maximum supported terrain classifier output labels | control.hpp:49 |

---

## 14. Step Metrics & Features (Computed)

| Parameter | Current Value | Units | Description | Reference |
|-----------|---------------|-------|-------------|-----------|
| `GRAVITY` | 9.785 | m/s² | Gravity acceleration constant (adjusted per static calibration) | gait_filter.cpp:57 |
| `SHOE_BUFFER_SIZE` | 8 | samples | Circular buffer size for SHOE detector measurement history | gait_filter_types.hpp:10 |
| `OMEGA_SIGN_BUF_SIZE` | 16 | samples | Max ring-buffer size for sign(Δw_sag) history in omega-sag detector | gait_filter_types.hpp:11 |

---

## Summary by Subsystem

### Gait Filter
- **Default Detectors**: SHOE (foot-flat) + Omega-sag (gait phase) + Pivot Constraint
- **ZUPT Style**: Pivot constraint (v_ankle = 0 via rotational kinematics)
- **Output Rate**: 100 Hz (IMU rate); classifier decimates to 20 Hz
- **Key Tuning**: SHOE thresholds (γ_enter, γ_exit), omega-sag thresholds (heel strike, toe-off)

### Terrain Classifier
- **Model Window**: 1000 ms (100 samples @ 100Hz)
- **Prediction Rate**: 20 Hz (5-sample stride)
- **Input Features**: Forward velocity, vertical velocity, sagittal angular rate, shank pitch
- **Debounce**: 3 consecutive predictions @ ≥80% confidence
- **Labels**: 6 terrain types (level, slope up/down, stairs up/down, idle)

### Motor Control
- **States**: 5 discrete positions (0-4) via barrel cam
- **Transition Time**: 200ms per state
- **No-Load Gating**: 200ms debounce + 400ms immediate execution window
- **Auto-Enable Threshold**: ≥60% classifier confidence

### Communication
- **IMU Rate**: 100 Hz
- **Serial Baud**: 115200
- **ADC Resolution**: 23-bit (ADS1220)
- **Batch Size**: 4 packets

---

## Tuning Strategy

1. **SHOE Detector**: Log `shoe_statistic` during walking; set γ_enter between low (stance) values and high (swing) values
2. **Omega-sag**: Adjust `hs_delta_threshold` and `toe_off_threshold` based on walking speed
3. **Motor Timing**: `TRAJ_DURATION` can be reduced (e.g., 0.1s) for faster response or increased for smoother motion
4. **Classifier Confidence**: Raise `GAIT_CONFIDENCE_THRESHOLD` for stricter terrain detection; lower for faster response
5. **ADC Thresholds**: If using load cell, calibrate `adc_stance_enter` and `adc_stance_exit` to your load cell sensitivity
6. **Vertical Damping**: Increase `vertical_damping` (e.g., 0.05) if swing-phase vertical drift is excessive

