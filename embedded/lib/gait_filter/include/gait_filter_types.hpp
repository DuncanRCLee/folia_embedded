#pragma once

/**
 * @file gait_filter_types.hpp
 * @brief Type definitions and data structures for the gait position filter
 * 
 * This file defines the core data structures used by the GaitFilter for
 * step-to-step position estimation optimized for terrain classification.
 */

#ifdef ARDUINO
    #include "ArduinoEigen.h"
#else
    #include <Eigen/Dense>
#endif

#include <cstdint>

namespace gait {

    
static constexpr uint8_t SHOE_BUFFER_SIZE = 8;
static constexpr uint8_t OMEGA_SIGN_BUF_SIZE = 16;  ///< Max ring-buffer size for sign(delta_w_sag)

// ============================================================================
// Basic Type Aliases
// ============================================================================

using Vec3f = Eigen::Vector3f;      // 3D vector (float for memory efficiency)
using Vec4f = Eigen::Vector4f;      // 4D vector (quaternion storage)
using Mat3f = Eigen::Matrix3f;      // 3x3 matrix
using Quatf = Eigen::Quaternionf;   // Quaternion

// State covariance: 9x9 for [position, velocity, orientation_error]
using Mat9f = Eigen::Matrix<float, 9, 9>;

// ============================================================================
// Output Frame Options
// ============================================================================

/**
 * @brief Output coordinate frame selection
 * 
 * Controls how position and velocity are reported:
 * - ENU: Standard East-North-Up world frame (heading-dependent)
 * - HEADING_RELATIVE: Forward/Lateral/Vertical aligned with walking direction
 *   Forward = direction of walking (from step displacement at heel strike)
 *   Lateral = perpendicular to forward (positive = left), in the frontal plane
 *   Vertical = up (same as ENU Z)
 * 
 * This provides stable sagittal plane alignment regardless of IMU orientation.
 */
enum class OutputFrame : uint8_t {
    ENU = 0,              ///< Standard East-North-Up world frame
    HEADING_RELATIVE = 1  ///< Forward/Lateral/Vertical relative to heading
};

// ============================================================================
// Filter Configuration
// ============================================================================

/**
 * @brief Configuration parameters for the gait filter
 */
struct FilterConfig {
    // Process noise standard deviations
    float sigma_accel_noise = 0.1f;     ///< Accelerometer noise (m/s²/√Hz)
    float sigma_gyro_noise = 0.01f;     ///< Gyroscope noise (rad/s/√Hz)
    
    // ========================================================================
    // SHOE Detector Parameters (Skog et al. "Zero-Velocity Detection")
    // ========================================================================
    // SHOE uses a Generalized Likelihood Ratio Test (GLRT) combining:
    // - Acceleration magnitude deviation from gravity
    // - Gyroscope magnitude (should be near zero)
    // The test statistic T is compared against a threshold gamma
    
    float shoe_sigma_a = 0.5f;          ///< Accel noise std for SHOE test (m/s²)
    float shoe_sigma_w = 0.2f;          ///< Gyro noise std for SHOE test (rad/s)
    float shoe_gamma_enter = 75.0f;          ///< SHOE threshold for ENTERING foot flat (T < gamma)
    float shoe_gamma_exit = 125.0f;    ///< SHOE threshold for LEAVING stance (T > gamma_exit)
                                        ///< Using hysteresis prevents rapid toggling
    uint8_t shoe_window_size = SHOE_BUFFER_SIZE;       ///< Window size W for SHOE averaging (samples)
    uint8_t shoe_debounce_enter = 3;    ///< Consecutive samples below gamma to enter stance
    uint8_t shoe_debounce_exit = 3;     ///< Consecutive samples above gamma_exit to leave stance
    uint8_t shoe_min_flat_samples = 30; ///< Min samples foot-flat must be active before exit is allowed (~300ms at 100Hz)

    // ========================================================================
    // Omega-Sag Stance Detector Parameters
    // ========================================================================
    // Detects heel strike via a sharp frame-to-frame spike in sagittal angular
    // velocity (w_sag).  Toe-off is detected when w_sag drops below a negative
    // threshold.  Set use_omega_sag_detector = true to use instead of SHOE.
    float   omega_sag_hs_delta_threshold = 0.3f;   ///< Min |delta_w_sag| to count as a non-zero sign (deadband, rad/s)
    float   omega_sag_toe_off_threshold  = -0.5f;  ///< w_sag below this triggers toe-off (rad/s)
    float   omega_sag_toe_off_deriv_threshold = -0.15f; ///< dw_sag/dt must also be below this for toe-off (rad/s per frame)
    uint8_t omega_sag_toe_off_debounce   = 2;      ///< Consecutive frames meeting both conditions to confirm toe-off
    uint8_t omega_sag_refractory_samples = 10;     ///< Min samples between heel-strikes (~100ms at 100Hz)
    uint8_t omega_sag_min_stance_samples = 20;     ///< Min samples in stance before toe-off is allowed (~200ms)
    uint8_t omega_sag_sign_window        = 10;     ///< Frames of sign(delta_w_sag) to scan for oscillation (max OMEGA_SIGN_BUF_SIZE)
    float   omega_sag_quiescence_threshold = 0.15f; ///< |w_sag| below this counts as quiescent (rad/s)
    uint8_t omega_sag_quiescence_samples = 50;     ///< Consecutive quiescent frames to auto-enter stance (~500ms at 100Hz)
    float   omega_sag_hs_neg_amplitude   = -1.5f;  ///< Min negative w_sag excursion during swing for zero-crossing HS (rad/s)
    bool    use_omega_sag_detector       = true;   ///< If true, use omega-sag detector instead of SHOE

    // ========================================================================
    // ADC Stance Detector Parameters
    // ========================================================================
    // When a strain-gauge ADC is available, use it as the primary stance signal.
    // getRaw() converted to volts (V = counts * 2.048 / 2^23) is passed in.
    // Positive = loaded (inversion applied in driver).
    //   adc_raw  > adc_stance_enter  → load detected  → heel strike / stance
    //   adc_raw  < adc_stance_exit   → no load        → toe-off / swing
    // NOTE: baseline_ is captured at startup under NO-LOAD conditions.
    //       If the foot is loaded at boot the baseline will be wrong.
    float adc_stance_enter  = 0.05f;  ///< Volts above baseline to enter stance (load detected)
    float adc_stance_exit   = 0.05f;  ///< Volts above baseline to exit stance (no hysteresis)
    uint8_t adc_min_stance_samples = 20; ///< Min samples in stance before exit is checked
    bool    use_adc_detector  = false;  ///< Enabled by main loop when ADC is present

    // ========================================================================
    // Pivot Constraint Parameters (for shank-mounted IMU)
    // ========================================================================
    // During stance, ankle acts as pivot point. Constraint:
    //   v_ankle = v_imu + R * (ω × r_imu_to_ankle) = 0
    // where r_imu_to_ankle is the vector from IMU to ankle in body frame
    
    // Vec3f r_imu_to_ankle = Vec3f(0, 0, -0.15f);  ///< IMU to ankle vector in body frame (m)
                                                  ///< Default: 15cm below IMU (typical shank)
    Vec3f r_imu_to_ankle = Vec3f(0, -0.02f, -0.08f);
    float pivot_velocity_noise = 0.02f;  ///< Ankle velocity measurement noise (m/s)
    
    // Legacy ZUPT parameters (kept for compatibility)
    float zupt_accel_threshold = 0.5f;  ///< Accel magnitude variance threshold (legacy)
    float zupt_gyro_threshold = 0.3f;   ///< Gyro magnitude threshold (legacy)
    uint8_t zupt_window_size = 5;       ///< Samples for detection (legacy)
    float zupt_velocity_noise = 0.01f;  ///< Velocity measurement noise (legacy)
    
    // Use SHOE + pivot constraint instead of legacy ZUPT
    bool use_shoe_detector = true;      ///< Use SHOE instead of simple thresholds
    bool use_pivot_constraint = true;   ///< Use pivot constraint instead of v=0
    
    // Position correction control
    bool apply_position_correction = false;  ///< If false, position is pure integral of velocity
                                            ///< (Kalman gain K_p still computed but not applied)
    
    // Walking direction source
    bool use_toe_off_heading = true;    ///< If true, lock walking direction at toe-off from IMU heading
                                        ///< If false, compute walking direction from step displacement at heel strike (1-step lag)
    
    // Vertical velocity damping (helps with drift during swing)
    float vertical_damping = 0.02f;     ///< Damping factor for vertical velocity
    
    // Step detection
    float step_height_threshold = 0.03f; ///< Minimum height change to count as step (m)
    
    // Filter update rate
    float dt_imu = 0.01f;               ///< IMU sample period (100 Hz -> 0.01s)
    // Note: Filter outputs at IMU rate (100 Hz). Downstream classifier handles decimation.
    
    // Output frame selection
    OutputFrame output_frame = OutputFrame::ENU;  ///< Coordinate frame for position/velocity output
    
    // Initial covariance
    float init_pos_cov = 0.001f;        ///< Initial position covariance (m²)
    float init_vel_cov = 0.01f;         ///< Initial velocity covariance (m/s)²
    float init_ori_cov = 0.0001f;       ///< Initial orientation error covariance (rad²)
};

// ============================================================================
// IMU Input Data
// ============================================================================

/**
 * @brief Raw IMU measurement from MTI-3
 * 
 * Supports multiple input modes:
 * - Mode 1: Use pre-integrated dv/dq (recommended, less drift)
 * - Mode 2: Use acceleration + gyroscope (fallback)
 * - Mode 3: Use free acceleration + quaternion (MTI provides gravity-free accel)
 */
struct IMUMeasurement {
    // Pre-integrated outputs (preferred - from MTI-3 internal strapdown)
    Vec3f dv;           ///< Delta velocity (m/s) - integrated acceleration
    Quatf dq;           ///< Delta quaternion - integrated rotation
    bool use_preintegrated = true;  ///< Flag to use dv/dq instead of raw
    
    // Raw sensor outputs (fallback)
    Vec3f accel;        ///< Linear acceleration (m/s²) in sensor frame
    Vec3f gyro;         ///< Angular velocity (rad/s) in sensor frame
    
    // Orientation from MTI-3 onboard filter (always available)
    Quatf orientation;  ///< Body-to-world quaternion from MTI-3 XKF
    
    // Free acceleration (gravity-compensated, in world frame)
    Vec3f free_accel;   ///< Free acceleration from MTI-3 (world frame)
    
    // Timestamp
    uint32_t sample_time_fine;  ///< MTI-3 timestamp (10kHz counter)
    
    // Default constructor
    IMUMeasurement() 
        : dv(Vec3f::Zero())
        , dq(Quatf::Identity())
        , use_preintegrated(true)
        , accel(Vec3f::Zero())
        , gyro(Vec3f::Zero())
        , orientation(Quatf::Identity())
        , free_accel(Vec3f::Zero())
        , sample_time_fine(0)
    {}
};

// ============================================================================
// Filter State
// ============================================================================

/**
 * @brief Gait phase enumeration
 */
enum class GaitPhase : uint8_t {
    UNKNOWN = 0,
    STANCE = 1,         ///< Foot is in contact with ground
    SWING = 2,          ///< Foot is in air
    HEEL_STRIKE = 3,    ///< Transition: swing -> stance
    TOE_OFF = 4         ///< Transition: stance -> swing
};

/**
 * @brief Internal filter state
 */
struct FilterState {
    // Position and velocity (relative to step start, world frame)
    Vec3f position;     ///< Position relative to last stance (m)
    Vec3f velocity;     ///< Velocity in world frame (m/s)
    
    // Orientation (maintained for gravity compensation if not using MTI quaternion)
    Quatf orientation;  ///< Body-to-world quaternion
    
    // Error state covariance (for position, velocity, orientation error)
    Mat9f P;            ///< 9x9 covariance matrix
    
    // Gait phase tracking
    GaitPhase phase;
    uint32_t phase_start_time;  ///< Timestamp when current phase started
    uint32_t samples_in_phase;  ///< Number of IMU samples in current phase
    
    // Step tracking
    uint32_t step_count;        ///< Total steps detected
    Vec3f step_start_position;  ///< World position at start of current step
    
    // SHOE detector buffer (circular) - stores full vectors for GLRT
    Vec3f accel_buffer_vec[SHOE_BUFFER_SIZE];  ///< Recent acceleration vectors (body frame)
    Vec3f gyro_buffer_vec[SHOE_BUFFER_SIZE];   ///< Recent gyroscope vectors (body frame)
    float accel_buffer[SHOE_BUFFER_SIZE];      ///< Recent acceleration magnitudes (legacy)
    float gyro_buffer[SHOE_BUFFER_SIZE];       ///< Recent gyro magnitudes (legacy)
    uint8_t buffer_idx;
    
    // Last SHOE test statistic (for debugging)
    float last_shoe_statistic;
    
    // SHOE debounced stance (used for ZUPT/pivot constraint)
    bool    shoe_stance_active;         ///< Debounced SHOE stance flag (foot-flat window)
    uint8_t shoe_debounce_count;        ///< Consecutive samples meeting SHOE transition condition
    uint8_t shoe_samples_in_flat;       ///< Samples spent in active foot-flat window (for min-duration gate)
    
    // Legacy debounce counter (kept for compatibility)
    uint8_t stance_debounce_count;  ///< Counts consecutive samples meeting transition condition

    // Omega-sag stance detector state
    float   prev_w_sag;                ///< Previous sample's sagittal angular velocity (rad/s)
    bool    omega_stance_active;       ///< Current stance flag from omega-sag detector
    uint8_t omega_refractory;          ///< Refractory countdown after heel strike (samples)
    uint8_t omega_samples_in_stance;   ///< Consecutive samples spent in stance (for min-dwell gate)
    int8_t  delta_sign_buf[OMEGA_SIGN_BUF_SIZE]; ///< Ring buffer of sign(delta_w_sag) for heel-strike detection
    uint8_t delta_sign_idx;            ///< Current write position in delta_sign_buf
    uint8_t omega_quiescence_count;    ///< Consecutive frames with |w_sag| below quiescence threshold
    float   omega_min_wsag_in_swing;   ///< Minimum w_sag observed during current swing phase
    uint8_t omega_toe_off_debounce_count; ///< Consecutive frames w_sag < thresh AND dw/dt < deriv thresh

    // ADC stance detector state
    float   adc_raw_last;          ///< Most recent ADC reading in volts passed via setAdcReading()
    bool    adc_stance_active;     ///< Current stance flag from ADC detector
    uint8_t adc_samples_in_stance; ///< Consecutive samples in ADC-detected stance (for min-dwell gate)

    // Combined (fused) stance detector state (used when ADC is enabled)
    bool    combined_stance_active;     ///< Fused stance flag from all detectors
    bool    prev_omega_for_combined;    ///< Previous omega-sag stance for edge detection
    bool    prev_adc_for_combined;      ///< Previous ADC stance for edge detection

    // Default constructor
    FilterState()
        : position(Vec3f::Zero())
        , velocity(Vec3f::Zero())
        , orientation(Quatf::Identity())
        , P(Mat9f::Identity() * 0.001f)
        , phase(GaitPhase::UNKNOWN)
        , phase_start_time(0)
        , samples_in_phase(0)
        , step_count(0)
        , step_start_position(Vec3f::Zero())
        , buffer_idx(0)
        , last_shoe_statistic(0)
        , shoe_stance_active(false)
        , shoe_debounce_count(0)
        , shoe_samples_in_flat(0)
        , stance_debounce_count(0)
        , prev_w_sag(0.0f)
        , omega_stance_active(false)
        , omega_refractory(0)
        , omega_samples_in_stance(0)
        , delta_sign_idx(0)
        , omega_quiescence_count(0)
        , omega_min_wsag_in_swing(0.0f)
        , omega_toe_off_debounce_count(0)
        , adc_raw_last(0.0f)
        , adc_stance_active(false)
        , adc_samples_in_stance(0)
        , combined_stance_active(false)
        , prev_omega_for_combined(false)
        , prev_adc_for_combined(false)
    {
        for (int i = 0; i < OMEGA_SIGN_BUF_SIZE; i++) delta_sign_buf[i] = 0;
        for (int i = 0; i < SHOE_BUFFER_SIZE; i++) {
            accel_buffer_vec[i] = Vec3f(0, 0, 9.81f);
            gyro_buffer_vec[i] = Vec3f::Zero();
            accel_buffer[i] = 9.81f;  // Initialize near 1g
            gyro_buffer[i] = 0.0f;
        }
    }
};

// ============================================================================
// Filter Output (Streamable)
// ============================================================================

/**
 * @brief Streamable filter output for terrain classification
 * 
 * This structure contains all features useful for gait/terrain classification.
 * Designed to be compact and efficient for streaming at 20Hz.
 */
struct __attribute__((packed)) GaitFilterOutput {
    // Timestamp
    uint32_t timestamp_ms;      ///< Milliseconds since boot
    
    // Step-relative position (world frame, relative to step start)
    float pos_x;                ///< Forward/backward position (m)
    float pos_y;                ///< Left/right position (m)
    float pos_z;                ///< Vertical position (m) - key for terrain!
    
    // Velocity (world frame)
    float vel_x;                ///< Forward velocity (m/s)
    float vel_y;                ///< Lateral velocity (m/s)
    float vel_z;                ///< Vertical velocity (m/s)
    
    // Heading-relative position (forward/lateral/vertical)
    // Aligned with walking direction (sagittal plane), computed from step displacement
    float pos_fwd;              ///< Forward position in walking direction (m)
    float pos_lat;              ///< Lateral position perpendicular to walking (m, positive = left)
    float pos_vertical;         ///< Vertical position (m) - same as pos_z
    
    // Heading-relative velocity (forward/lateral/vertical)
    float vel_fwd;              ///< Forward velocity in walking direction (m/s)
    float vel_lat;              ///< Lateral velocity perpendicular to walking (m/s, positive = left)
    float vel_vertical;         ///< Vertical velocity (m/s) - same as vel_z
    
    // Heading-relative acceleration (forward/lateral)
    float a_fwd;                ///< Forward acceleration in walking direction (m/s²)
    float a_lat;                ///< Lateral acceleration perpendicular to walking (m/s², positive = left)
    
    // Walking direction used for sagittal plane transformation
    float heading;              ///< Walking direction angle (rad), computed from step displacement
    
    // Sagittal plane angular rate (rotation about medial-lateral axis)
    float w_sag;                ///< Angular velocity in sagittal plane (rad/s), positive = forward rotation
    
    // Step metrics (accumulated over current step)
    float step_height;          ///< Max vertical displacement this step (m)
    float step_length;          ///< Horizontal displacement this step (m)
    float step_duration_ms;     ///< Time since step start (ms)
    
    // Orientation (Euler angles for easier interpretation)
    float pitch;                ///< Body pitch angle (rad)
    float roll;                 ///< Body roll angle (rad)
    float yaw;                  ///< Body yaw angle (rad)
    
    // Gait phase
    uint8_t phase;              ///< Current gait phase (GaitPhase enum)
    uint8_t step_count_low;     ///< Lower 8 bits of step count
    
    // Confidence/quality metrics
    float position_uncertainty; ///< Position covariance trace (m²)
    uint8_t zupt_active;        ///< 1 if ZUPT correction active this sample
    uint8_t stance_active;      ///< 1 if stance detected this sample (debounced)
    float shoe_statistic;       ///< Current SHOE test statistic (for debugging)
    uint8_t debounce_counter;   ///< Current debounce counter value (for debugging)
    uint8_t raw_should_enter;   ///< Raw threshold check: T < gamma (for debugging)
    uint8_t raw_should_exit;    ///< Raw threshold check: T > gamma_exit (for debugging)

    // Foot-flat indicator (SHOE-based, for ZUPT window)
    uint8_t foot_flat;           ///< 1 while debounced SHOE stance is active (foot-flat window)

    // Omega-sag stance detector debug
    float   w_sag_delta;         ///< Frame-to-frame |Δw_sag| (rad/s, for debugging)
    uint8_t omega_hs_detected;   ///< 1 if heel-strike impulse detected this frame

    // ADC stance detector debug
    float   adc_raw_out;         ///< Last ADC value in volts fed to stance detector
    uint8_t adc_stance_out;      ///< 1 if ADC reports stance this frame

    // Default constructor
    GaitFilterOutput()
        : timestamp_ms(0)
        , pos_x(0), pos_y(0), pos_z(0)
        , vel_x(0), vel_y(0), vel_z(0)
        , pos_fwd(0), pos_lat(0), pos_vertical(0)
        , vel_fwd(0), vel_lat(0), vel_vertical(0)
        , a_fwd(0), a_lat(0)
        , heading(0)
        , w_sag(0)
        , step_height(0), step_length(0), step_duration_ms(0)
        , pitch(0), roll(0), yaw(0)
        , phase(0), step_count_low(0)
        , position_uncertainty(0)
        , zupt_active(0)
        , stance_active(0)
        , shoe_statistic(0)
        , debounce_counter(0)
        , raw_should_enter(0)
        , raw_should_exit(0)
        , foot_flat(0)
        , w_sag_delta(0.0f)
        , omega_hs_detected(0)
        , adc_raw_out(0.0f)
        , adc_stance_out(0)
    {}
};

// ============================================================================
// Step Summary (for classification features)
// ============================================================================

/**
 * @brief Summary of a completed step for terrain classification
 * 
 * Generated at each heel strike, summarizing the previous step.
 */
struct StepSummary {
    uint32_t step_number;
    
    // Displacement
    float total_height_change;   ///< Net vertical change (m) - positive = upward
    float max_height;            ///< Maximum height during swing (m)
    float min_height;            ///< Minimum height during swing (m)
    float horizontal_distance;   ///< Total horizontal distance (m)
    
    // Timing
    float swing_time_ms;         ///< Duration of swing phase (ms)
    float stance_time_ms;        ///< Duration of stance phase (ms)
    float total_time_ms;         ///< Total step time (ms)
    
    // Velocity
    float peak_vertical_velocity;   ///< Peak vertical velocity during swing (m/s)
    float avg_horizontal_velocity;  ///< Average horizontal velocity (m/s)
    
    // Terrain indicators
    float height_to_distance_ratio; ///< |height_change| / horizontal_distance
    
    // Default constructor
    StepSummary()
        : step_number(0)
        , total_height_change(0), max_height(0), min_height(0), horizontal_distance(0)
        , swing_time_ms(0), stance_time_ms(0), total_time_ms(0)
        , peak_vertical_velocity(0), avg_horizontal_velocity(0)
        , height_to_distance_ratio(0)
    {}
};

} // namespace gait
