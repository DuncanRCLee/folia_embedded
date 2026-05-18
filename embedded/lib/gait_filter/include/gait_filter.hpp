#pragma once

/**
 * @file gait_filter.hpp
 * @brief Step-to-step position estimation filter for gait terrain classification
 * 
 * This filter estimates foot/shank position relative to the start of each step,
 * optimized for terrain classification (level ground, slopes, stairs).
 * 
 * Key features:
 * - Uses MTI-3 dv/dq outputs for minimal integration drift
 * - Zero Velocity Update (ZUPT) during detected stance phases
 * - Step-relative position (resets at heel strike)
 * - Lightweight implementation for Portenta H7 at 20Hz output
 * 
 * Design philosophy:
 * - We don't need global position accuracy (no GPS fusion)
 * - We need CONSISTENT step-to-step measurements for classification
 * - Vertical displacement is most important for terrain differentiation
 * 
 * Usage:
 * @code
 *     gait::GaitFilter filter;
 *     filter.configure(config);
 *     
 *     // In IMU loop (100 Hz):
 *     gait::IMUMeasurement meas;
 *     // ... fill measurement from MTI-3 ...
 *     filter.update(meas);
 *     
 *     // At classification rate (20 Hz):
 *     if (filter.hasNewOutput()) {
 *         gait::GaitFilterOutput out = filter.getOutput();
 *         // ... send to classifier ...
 *     }
 * @endcode
 */

#include "gait_filter_types.hpp"

namespace gait {

class GaitFilter {
public:
    // ========================================================================
    // Construction / Configuration
    // ========================================================================
    
    /**
     * @brief Default constructor with reasonable defaults
     */
    GaitFilter();
    
    /**
     * @brief Constructor with custom configuration
     */
    explicit GaitFilter(const FilterConfig& config);
    
    /**
     * @brief Configure filter parameters
     */
    void configure(const FilterConfig& config);
    
    /**
     * @brief Reset filter to initial state
     * 
     * Call when starting a new walking session or after long pause.
     */
    void reset();
    
    /**
     * @brief Initialize orientation from MTI-3 quaternion
     * 
     * Should be called once at startup with a valid MTI-3 orientation
     * to properly initialize the filter's world frame.
     * 
     * @param quat Body-to-world quaternion from MTI-3
     */
    void initializeOrientation(const Quatf& quat);
    
    /**
     * @brief Initialize with known IMU mounting orientation
     * 
     * For shank-mounted IMU with:
     *   +Y = vertical (up when standing)
     *   +X = right (medial-lateral)
     *   +Z = posterior (backward)
     * 
     * This computes the initial walking direction from the IMU's -Z axis
     * (forward direction) projected onto the horizontal plane, eliminating
     * the need to wait for the first step.
     * 
     * @param quat Body-to-world quaternion from MTI-3
     */
    void initializeForward(const Quatf& quat);
    
    // ========================================================================
    // Main Update Loop
    // ========================================================================
    
    /**
     * @brief Process new IMU measurement
     * 
     * Call this at IMU rate (100 Hz). The filter will internally
     * accumulate measurements and produce output at the configured rate.
     * 
     * @param meas IMU measurement from MTI-3
     */
    void update(const IMUMeasurement& meas);
    
    /**
     * @brief Convenience update from raw MTI-3 arrays
     * 
     * @param dv Delta velocity (3 floats, m/s)
     * @param dq Delta quaternion (4 floats, [w,x,y,z])
     * @param quat Orientation quaternion (4 floats, [w,x,y,z])
     * @param free_acc Free acceleration (3 floats, m/s², world frame)
     * @param sample_time MTI-3 sample time fine
     */
    void update(const float* dv, const float* dq, const float* quat, 
                const float* free_acc, uint32_t sample_time);
    
    // ========================================================================
    // Output Access
    // ========================================================================
    
    /**
     * @brief Check if new output is available
     * 
     * Returns true after every update() call (100 Hz at IMU rate).
     * Downstream classifier should handle decimation to 20 Hz.
     */
    bool hasNewOutput() const { return output_ready_; }
    
    /**
     * @brief Get latest filter output
     * 
     * Clears the output_ready flag.
     */
    GaitFilterOutput getOutput();
    
    /**
     * @brief Get current output without clearing flag
     */
    const GaitFilterOutput& peekOutput() const { return output_; }
    
    /**
     * @brief Check if a new step was just completed
     */
    bool hasNewStep() const { return step_complete_; }
    
    /**
     * @brief Get summary of the last completed step
     * 
     * Clears the step_complete flag.
     */
    StepSummary getStepSummary();
    
    // ========================================================================
    // State Access (for debugging/logging)
    // ========================================================================
    
    /**
     * @brief Get current gait phase
     */
    GaitPhase getPhase() const { return state_.phase; }
    
    /**
     * @brief Get current position (step-relative)
     */
    Vec3f getPosition() const { return state_.position; }
    
    /**
     * @brief Get current velocity
     */
    Vec3f getVelocity() const { return state_.velocity; }
    
    /**
     * @brief Get step count
     */
    uint32_t getStepCount() const { return state_.step_count; }
    
    /**
     * @brief Check if ZUPT is currently active (foot flat)
     */
    bool isZuptActive() const { return zupt_active_; }
    
    /**
     * @brief Check if currently in stance (heel strike to toe off)
     *
     * Unlike isZuptActive() which tracks foot-flat (SHOE), this tracks
     * the broader stance phase from w_sag/ADC/combined detection.
     * Use this for motor state gating.
     */
    bool isInStance() const { return in_stance_; }
    
    /**
     * @brief Get timestamp (ms) of last toe-off event
     * 
     * Returns 0 if no toe-off has been detected yet.
     */
    uint32_t getLastToeOffMs() const { return last_toe_off_ms_; }
    
    /**
     * @brief Get filter configuration (read-only)
     */
    const FilterConfig& getConfig() const { return config_; }
    
    /**
     * @brief Get last SHOE test statistic (for debugging/tuning)
     */
    float getLastShoeStatistic() const { return state_.last_shoe_statistic; }
    
    /**
     * @brief Set the IMU-to-ankle lever arm vector
     * 
     * Call this to configure the pivot constraint for your specific mounting.
     * The vector should point from IMU center to ankle joint center in body frame.
     * 
     * @param r_imu_to_ankle Vector from IMU to ankle in body frame (m)
     */
    void setLeverArm(const Vec3f& r_imu_to_ankle) { config_.r_imu_to_ankle = r_imu_to_ankle; }

    /**
     * @brief Set the most recent ADC reading for stance detection
     *
     * Call this once per IMU update cycle (before update()) when a load-cell /
     * strain-gauge ADC is available.  The value should be the baseline-relative
     * count returned by ADS1220::getRaw() (i.e. 0 at no-load, positive under load).
     * Setting use_adc_detector = true in FilterConfig activates ADC-based stance
     * detection, which takes priority over the omega-sag and SHOE detectors.
     *
     * @param adc_raw  Baseline-relative ADC count (getRaw() from ADS1220)
     */
    void setAdcReading(float adc_raw) { state_.adc_raw_last = adc_raw; }

    /**
     * @brief Enable or disable ADC-based stance detection
     *
     * Call once after initialization when it is known whether a load-cell ADC
     * is present.  When enabled, ADC takes priority over the omega-sag detector.
     *
     * @param en  true = use ADC; false = fall back to omega-sag
     */
    void enableAdcDetector(bool en) { config_.use_adc_detector = en; }
    
    /**
     * @brief Set the output coordinate frame
     * 
     * Controls how position and velocity are reported:
     * - ENU: Standard East-North-Up world frame (heading-dependent)
     * - HEADING_RELATIVE: Forward/Lateral/Vertical relative to body heading
     * 
     * Note: Both ENU and heading-relative values are always computed and available
     * in the output structure. This setting only affects which frame is primary.
     * 
     * @param frame Output coordinate frame selection
     */
    void setOutputFrame(OutputFrame frame) { config_.output_frame = frame; }
    
    /**
     * @brief Get current output frame setting
     */
    OutputFrame getOutputFrame() const { return config_.output_frame; }
    
    /**
     * @brief Get current heading (yaw) angle
     * 
     * Returns the heading extracted from the current orientation quaternion.
     * Useful for heading-relative computations.
     * 
     * @return Heading angle in radians (yaw, counter-clockwise from East/X-axis)
     */
    float getHeading() const;
    
    /**
     * @brief Get position in sagittal-plane-aligned frame (forward/lateral/vertical)
     * 
     * Transforms the ENU position to a frame aligned with the walking direction:
     * - Forward: direction of walking (computed from step displacement)
     * - Lateral: perpendicular to forward (positive = left)
     * - Vertical: up (same as ENU Z)
     * 
     * The walking direction is computed at each heel strike from the horizontal
     * displacement of the completed step, giving stable sagittal plane alignment.
     * 
     * @return Position vector as (forward, lateral, vertical)
     */
    Vec3f getPositionHeadingRelative() const;
    
    /**
     * @brief Get velocity in sagittal-plane-aligned frame (forward/lateral/vertical)
     * 
     * @return Velocity vector as (forward, lateral, vertical)
     */
    Vec3f getVelocityHeadingRelative() const;
    
    /**
     * @brief Get walking direction angle
     * 
     * Returns the walking direction computed from step displacement at heel strike.
     * This provides stable sagittal plane alignment throughout the gait cycle.
     * 
     * @return Walking direction angle in radians (from East/X-axis)
     */
    float getWalkingDirection() const;

private:
    // ========================================================================
    // Internal Methods
    // ========================================================================
    
    /**
     * @brief Predict step using pre-integrated dv/dq
     */
    void predictWithPreintegrated(const IMUMeasurement& meas);
    
    /**
     * @brief Predict step using raw acceleration (fallback)
     */
    void predictWithAcceleration(const IMUMeasurement& meas);
    
    /**
     * @brief Predict step using free acceleration from MTI-3
     */
    void predictWithFreeAcceleration(const IMUMeasurement& meas);
    
    /**
     * @brief Update ZUPT detection buffer with new measurement
     */
    void updateZuptBuffer(const IMUMeasurement& meas);
    
    /**
     * @brief SHOE detector (Skog et al.) — computes foot-flat statistic
     * 
     * Uses a Generalized Likelihood Ratio Test (GLRT) to detect foot flat.
     * Test statistic: T = (1/W) * Σ[ ||a_k - g*a_k/||a_k||||² / σ_a² + ||ω_k||² / σ_w² ]
     * Foot flat detected when T < γ (threshold)
     * 
     * @return true if foot flat detected, false otherwise
     */
    bool detectFF();

    /**
     * @brief Debounced SHOE foot-flat detector for ZUPT/pivot constraint
     *
     * Applies hysteretic debouncing to the raw SHOE statistic to produce a
     * stable foot-flat window.  The result is stored in state_.shoe_stance_active
     * and used exclusively for ZUPT application.  When foot_flat exits (heel off),
     * walking direction is captured.
     *
     * @return true while debounced foot-flat is active
     */
    bool detectFFDebounced();
    
    /**
     * @brief ADC-based stance detector (primary, when use_adc_detector = true)
     *
     * Hysteretic threshold on baseline-relative strain-gauge counts:
     *   Enter stance: adc_raw > adc_stance_enter   (heel strike / loading)
     *   Exit stance:  adc_raw < adc_stance_exit    (toe-off / unloading)
     * A minimum dwell guard prevents spurious exits during mid-stance transients.
     *
     * @return true if stance is active
     */
    bool detectStanceAdc();

    /**
     * @brief Compute current sagittal-plane angular velocity from stored gyro + orientation.
     * Projects world-frame gyro onto the medial-lateral axis aligned with walking direction.
     * @return w_sag in rad/s (positive = forward rotation)
     */
    float computeWsag() const;

    /**
     * @brief Omega-sag stance detector
     *
     * Detects heel strike via a sharp frame-to-frame spike in |Δw_sag|, and
     * toe-off when w_sag crosses below a negative threshold.  A refractory
     * period prevents double-counting the impact oscillation.
     *
     * @param w_sag_current  Current sagittal angular velocity (rad/s)
     * @return true if stance is active
     */
    bool detectStanceOmegaSag(float w_sag_current);

    /**
     * @brief Detect if we're in stance phase (heel strike to toe off)
     *
     * Stance is detected via w_sag and ADC (when enabled), with
     * SHOE foot-flat as a fallback for heel-strike entry if both
     * w_sag and ADC fail.  Toe-off exit uses w_sag or ADC only.
     * When ADC is disabled, uses omega-sag alone.
     *
     * Foot-flat (SHOE) drives ZUPT/position reset independently.
     */
    bool detectStance();
    
    /**
     * @brief Apply pivot constraint ZUPT for shank-mounted IMU
     * 
     * During stance, ankle velocity should be zero:
     *   v_ankle = v_imu + R * (ω × r_imu_to_ankle) = 0
     * 
     * This constrains IMU velocity based on rotation about ankle pivot.
     * 
     * @param gyro Current angular velocity in body frame (rad/s)
     */
    void applyPivotConstraint(const Vec3f& gyro);
    
    /**
     * @brief Apply simple zero velocity update (legacy, for foot-mounted)
     */
    void applyZuptLegacy();
    
    /**
     * @brief Apply Zero Velocity Update correction (dispatches to appropriate method)
     */
    void applyZupt(const IMUMeasurement& meas);
    
    /**
     * @brief Detect gait phase transitions
     */
    /**
     * @brief Update gait phase state machine
     * @param meas Current IMU measurement
     * @param in_stance Result of stance detection (passed to avoid calling detectStance twice)
     */
    void updateGaitPhase(const IMUMeasurement& meas, bool in_stance);
    
    /**
     * @brief Handle heel strike event
     */
    void onHeelStrike();
    
    /**
     * @brief Handle toe-off event
     */
    void onToeOff();
    
    /**
     * @brief Capture walking direction from IMU orientation at heel off
     * (SHOE foot-flat falling edge)
     */
    void captureHeadingAtHeelOff();
    
    /**
     * @brief Update step tracking metrics
     */
    void updateStepMetrics();
    
    /**
     * @brief Generate output packet
     */
    void generateOutput(uint32_t timestamp_ms);
    
    /**
     * @brief Generate step summary
     */
    void generateStepSummary();
    
    /**
     * @brief Convert quaternion to Euler angles (ZYX convention)
     */
    static void quatToEuler(const Quatf& q, float& roll, float& pitch, float& yaw);
    
    /**
     * @brief Extract heading (yaw) from quaternion efficiently
     * 
     * This is more efficient than computing full Euler angles when only
     * heading is needed. Uses the formula for yaw extraction directly.
     * 
     * @param q Orientation quaternion
     * @return Heading angle in radians (yaw)
     */
    static float quatToHeading(const Quatf& q);
    
    /**
     * @brief Transform ENU position/velocity to walking-direction-relative frame
     * 
     * Applies 2D rotation by the walking direction angle to transform from
     * ENU (East-North-Up) to sagittal plane aligned (Forward-Lateral-Vertical).
     * 
     * The transformation is:
     *   forward = cos(walking_dir) * east + sin(walking_dir) * north
     *   lateral = -sin(walking_dir) * east + cos(walking_dir) * north
     *   vertical = up (unchanged)
     * 
     * @param enu_vec Vector in ENU frame
     * @param cos_dir Cosine of walking direction angle
     * @param sin_dir Sine of walking direction angle
     * @return Vector in sagittal-plane-aligned frame (forward, lateral, vertical)
     */
    static Vec3f enuToWalkingFrame(const Vec3f& enu_vec, float cos_dir, float sin_dir);
    
    // ========================================================================
    // Internal State
    // ========================================================================
    
    FilterConfig config_;
    FilterState state_;
    GaitFilterOutput output_;
    StepSummary step_summary_;
    
    // Tracking for current step
    float max_height_this_step_;
    float min_height_this_step_;
    float peak_vertical_vel_this_step_;
    Vec3f step_start_pos_;
    uint32_t step_start_time_ms_;
    uint32_t stance_start_time_ms_;
    uint32_t swing_start_time_ms_;
    
    // Sample tracking
    uint32_t imu_sample_count_;
    
    // Current gyro for pivot constraint
    Vec3f current_gyro_;
    
    // Current free acceleration for output (world frame, gravity-compensated)
    Vec3f current_free_accel_;
    
    // Flags
    bool initialized_;
    bool output_ready_;
    bool step_complete_;
    bool zupt_active_;
    bool in_stance_;           ///< Combined stance (HS to TO) from w_sag/ADC/foot_flat
    uint32_t last_toe_off_ms_; ///< Timestamp of last toe-off (for motor gate)
    bool was_in_stance_;
    bool stance_in_previous_;  ///< Hysteresis state for SHOE detector
    bool shoe_stance_previous_; ///< Previous SHOE foot-flat for heading edge detection
    
    // Walking direction for sagittal plane alignment
    // Computed at toe-off from IMU heading (preferred) or at heel strike from step displacement
    float walking_dir_cos_;    ///< cos(walking_direction)
    float walking_dir_sin_;    ///< sin(walking_direction)
    bool walking_dir_valid_;   ///< True after first step with valid direction
    
    // Toe-off heading capture (for use_toe_off_heading mode)
    float toe_off_heading_;    ///< IMU heading captured at toe-off (rad)
    
    // Gravity vector (world frame)
    // static constexpr float GRAVITY = 9.81f;
    static constexpr float GRAVITY = 9.785f; // Adjusted based on static measurement
    Vec3f gravity_world_;
    
    // ========================================================================
    // Helper functions
    // ========================================================================
    
    /**
     * @brief Compute skew-symmetric matrix from vector (for cross product)
     */
    static Mat3f skew(const Vec3f& v) {
        Mat3f S;
        S << 0, -v.z(), v.y(),
             v.z(), 0, -v.x(),
             -v.y(), v.x(), 0;
        return S;
    }
};

} // namespace gait
