#pragma once

/**
 * @file gait_filter_integration.hpp
 * @brief Example integration of GaitFilter with main.cpp
 * 
 * This header shows how to integrate the gait filter into the main
 * application loop and stream outputs alongside IMU data.
 */

#include "gait_filter.hpp"

namespace gait {

/**
 * @brief Global filter instance (can be allocated in main)
 * Uses static allocation to avoid alignment issues with new/malloc
 */
static GaitFilter& getGaitFilterInstance() {
    // Use static storage with proper alignment - avoids heap allocation issues
    static uint8_t storage[sizeof(GaitFilter)] __attribute__((aligned(16)));
    static bool initialized = false;
    
    if (!initialized) {
        // Placement new - construct GaitFilter in pre-allocated aligned memory
        new (storage) GaitFilter();
        initialized = true;
    }
    
    return *reinterpret_cast<GaitFilter*>(storage);
}

// Convenience reference to the filter instance
#define g_gaitFilter getGaitFilterInstance()

/**
 * @brief Initialize the gait filter with optimal settings for terrain classification
 * 
 * Call this during setup() after IMU initialization.
 * Filter outputs at 100Hz (same as IMU rate). Downstream classifier handles
 * decimation to 20Hz prediction rate.
 * 
 * Uses SHOE detector and pivot constraint for shank-mounted IMU.
 */
inline void initGaitFilter() {
    FilterConfig config;
    
    // Process noise tuned for MTI-3 at 100 Hz
    config.sigma_accel_noise = 0.08f;   // MTI-3 typical noise
    config.sigma_gyro_noise = 0.008f;   // MTI-3 typical noise
    
    // ========================================================================
    // SHOE Detector Parameters (Skog et al.)
    // ========================================================================
    config.use_shoe_detector = true;     // Use SHOE instead of simple thresholds
    config.shoe_gamma_enter = 75.0f;     // Enter stance (Low motion)
    config.shoe_gamma_exit = 125.0f;     // Exit stance (High motion)
    config.shoe_debounce_enter = 3;      // Wait 3 frames (30ms) to confirm stance
    config.shoe_debounce_exit = 3;       // Wait 3 frames (30ms) to confirm swing

    config.shoe_sigma_a = 0.5f;          // Accel noise for SHOE test (m/s²)
    config.shoe_sigma_w = 0.2f;          // Gyro noise for SHOE test (rad/s)
    
    // ========================================================================
    // Pivot Constraint for Shank-Mounted IMU
    // ========================================================================
    config.use_pivot_constraint = true;  // Use pivot constraint instead of v=0
    config.r_imu_to_ankle = Vec3f(0, 0, -0.15f);  // Vector from IMU to ankle (m)
                                                   // Default: 15cm below IMU (typical shank mount)
                                                   // Adjust based on your mounting!
    config.pivot_velocity_noise = 0.02f; // Ankle velocity measurement noise (m/s)
    
    // Position correction control
    config.apply_position_correction = false;   // false: Position = pure integral of velocity
                                                // true: Apply position correction during stance
    
    // Walking direction source
    config.use_toe_off_heading = true;    // Lock direction at toe-off (no lag)
    
    // Legacy ZUPT parameters (used if use_shoe_detector=false)
    config.zupt_accel_threshold = 0.4f;  // Variance threshold
    config.zupt_gyro_threshold = 0.25f;  // rad/s threshold
    config.zupt_window_size = 5;         // 50ms window at 100 Hz
    config.zupt_velocity_noise = 0.005f; // Tight correction during stance
    
    // Vertical damping (helps with drift during swing)
    config.vertical_damping = 0.015f;
    
    // Timing - filter runs at IMU rate
    config.dt_imu = 0.01f;      // 100 Hz IMU
    
    // Initial covariance
    config.init_pos_cov = 0.0001f;
    config.init_vel_cov = 0.001f;
    config.init_ori_cov = 0.00001f;
    
    // Configure the filter (no dynamic allocation needed)
    g_gaitFilter.configure(config);
}

/**
 * @brief Initialize gait filter with optional lever arm and mounting orientation
 * 
 * This is the recommended initialization function. It configures the filter
 * and optionally sets the lever arm and/or initializes walking direction from
 * the IMU's known mounting orientation.
 * 
 * @param r_imu_to_ankle Optional: Vector from IMU center to ankle joint (body frame, meters)
 *                       Pass nullptr to use default (0, 0, -0.15m)
 *                       Typically: (0, 0, -distance) where distance is positive
 * 
 * @param initQuat       Optional: Body-to-world quaternion from MTI-3 [w,x,y,z]
 *                       Pass nullptr to compute walking direction from first step
 *                       If provided, walking direction is computed from IMU's -Z axis
 *                       (forward direction) projected onto horizontal plane
 * 
 * IMU mounting convention (posterior shank):
 *   +Y = vertical (up when standing)
 *   +X = right (medial-lateral axis)
 *   +Z = posterior (backward), so -Z is forward
 * 
 * Usage examples:
 *   initGaitFilterEx();                           // Default config
 *   initGaitFilterEx(&leverArm);                  // Custom lever arm only
 *   initGaitFilterEx(nullptr, quat);              // Mounting orientation only
 *   initGaitFilterEx(&leverArm, quat);            // Both
 */
inline void initGaitFilterEx(const Vec3f* r_imu_to_ankle = nullptr,
                              const float* initQuat = nullptr) {
    // Always configure the filter first
    initGaitFilter();
    
    // Set custom lever arm if provided
    if (r_imu_to_ankle != nullptr) {
        g_gaitFilter.setLeverArm(*r_imu_to_ankle);
    }
    
    // Initialize with mounting orientation if provided
    if (initQuat != nullptr) {
        Quatf q(initQuat[0], initQuat[1], initQuat[2], initQuat[3]);  // [w, x, y, z]
        g_gaitFilter.initializeForward(q);
    }
}

/**
 * @brief Initialize filter with custom lever arm for your specific IMU mounting
 * @deprecated Use initGaitFilterEx() instead
 * 
 * @param r_imu_to_ankle Vector from IMU center to ankle joint (body frame, meters)
 *                       Typically: (0, 0, -distance) where distance is positive
 */
inline void initGaitFilterWithLeverArm(const Vec3f& r_imu_to_ankle) {
    initGaitFilterEx(&r_imu_to_ankle, nullptr);
}

/**
 * @brief Initialize gait filter with known IMU mounting orientation
 * @deprecated Use initGaitFilterEx() instead
 * 
 * @param initQuat Body-to-world quaternion from MTI-3 at initialization
 */
inline void initGaitFilterWithMounting(const float* initQuat) {
    initGaitFilterEx(nullptr, initQuat);
}

/**
 * @brief Set the output coordinate frame for position/velocity
 * 
 * @param frame OutputFrame::ENU for East-North-Up world frame
 *              OutputFrame::HEADING_RELATIVE for forward/lateral/vertical
 * 
 * Note: Both frames are always computed and available in the output.
 * This setting is for user preference and documentation purposes.
 */
inline void setGaitOutputFrame(OutputFrame frame) {
    g_gaitFilter.setOutputFrame(frame);
}

/**
 * @brief Get current output frame setting
 */
inline OutputFrame getGaitOutputFrame() {
    return g_gaitFilter.getOutputFrame();
}

/**
 * @brief Get current walking direction angle
 * 
 * Returns the walking direction computed from step displacement at heel strike.
 * This provides stable sagittal plane alignment throughout the gait cycle.
 * 
 * @return Walking direction in radians
 */
inline float getGaitHeading() {
    return g_gaitFilter.getHeading();
}

/**
 * @brief Get walking direction angle (alias for getGaitHeading)
 * @return Walking direction in radians
 */
inline float getGaitWalkingDirection() {
    return g_gaitFilter.getWalkingDirection();
}

/**
 * @brief Get position in sagittal-plane-aligned frame
 * 
 * Forward = direction of walking (from step displacement)
 * Lateral = perpendicular to walking direction (positive = left)
 * Vertical = up
 * 
 * @return Vec3f(forward, lateral, vertical) in meters
 */
inline Vec3f getGaitPositionHeadingRelative() {
    return g_gaitFilter.getPositionHeadingRelative();
}

/**
 * @brief Get velocity in sagittal-plane-aligned frame
 * @return Vec3f(forward, lateral, vertical) in m/s
 */
inline Vec3f getGaitVelocityHeadingRelative() {
    return g_gaitFilter.getVelocityHeadingRelative();
}

/**
 * @brief Process IMU measurement through the gait filter
 * 
 * Call this in the main loop after reading IMU data.
 * 
 * @param dv Delta velocity from MTI-3 (3 floats)
 * @param dq Delta quaternion from MTI-3 (4 floats, [w,x,y,z])
 * @param quat Orientation quaternion from MTI-3 (4 floats, [w,x,y,z])
 * @param free_acc Free acceleration from MTI-3 (3 floats, optional)
 * @param sample_time MTI-3 sample time fine
 */
inline void processGaitFilter(const float* dv, const float* dq, 
                               const float* quat, const float* free_acc,
                               uint32_t sample_time) {
    // If free_acc not available, create zero array
    float zero_acc[3] = {0, 0, 0};
    const float* acc = free_acc ? free_acc : zero_acc;
    
    g_gaitFilter.update(dv, dq, quat, acc, sample_time);
}

/**
 * @brief Pass the latest baseline-relative ADC reading (in volts) to the stance detector.
 * Call once per IMU cycle, before processGaitFilter(), when ADC is present.
 * @param adc_volts  getRaw() * (2.048f / 8388608.0f) from ADS1220 (positive = loaded)
 */
inline void setGaitFilterAdcReading(float adc_volts) {
    g_gaitFilter.setAdcReading(adc_volts);
}

/**
 * @brief Enable / disable ADC-based stance detection in the gait filter.
 * Call once after initGaitFilterEx() when ADC availability is known.
 * @param en  true = use ADC (primary); false = fall back to omega-sag
 */
inline void setGaitFilterAdcEnabled(bool en) {
    g_gaitFilter.enableAdcDetector(en);
}

/**
 * @brief Check if new filter output is available (at 100 Hz, every IMU sample)
 * 
 * Always returns true after processGaitFilter() is called.
 * Downstream classifier should handle decimation to 20Hz.
 */
inline bool hasGaitOutput() {
    return g_gaitFilter.hasNewOutput();
}

/**
 * @brief Get the latest gait filter output
 */
inline GaitFilterOutput getGaitOutput() {
    return g_gaitFilter.getOutput();
}

/**
 * @brief Check if a step was just completed
 */
inline bool hasNewStep() {
    return g_gaitFilter.hasNewStep();
}

/**
 * @brief Get step summary for terrain classification
 */
inline StepSummary getStepSummary() {
    return g_gaitFilter.getStepSummary();
}

// ============================================================================
// Example integration with main.cpp loop (100Hz filter output)
// ============================================================================

/*
 * In setup():
 *   gait::initGaitFilter();
 *   
 *   // Optional: Use heading-relative output frame
 *   gait::setGaitOutputFrame(gait::OutputFrame::HEADING_RELATIVE);
 * 
 * In loop(), after reading MTI-3 (runs at 100Hz):
 *   float* dv = mti->getDeltaV();
 *   float* dq = mti->getDeltaQ();
 *   float* quat = mti->getQuaternion();
 *   float* freeAcc = mti->getFreeAcceleration();  // if configured
 *   uint32_t sampleTime = mti->getSampleTimeFine();
 *   
 *   gait::processGaitFilter(dv, dq, quat, freeAcc, sampleTime);
 *   
 *   // Output available at 100 Hz (every IMU sample)
 *   if (gait::hasGaitOutput()) {
 *       gait::GaitFilterOutput out = gait::getGaitOutput();
 *       
 *       // ENU frame outputs (always available):
 *       //   out.pos_x, out.pos_y, out.pos_z
 *       //   out.vel_x, out.vel_y, out.vel_z
 *       
 *       // Sagittal-plane-aligned outputs (always available):
 *       // Computed using walking direction from step displacement
 *       //   out.pos_forward  - Position in direction of walking
 *       //   out.pos_lateral  - Position perpendicular to walking (left positive)
 *       //   out.pos_vertical - Vertical position (same as pos_z)
 *       //   out.vel_forward, out.vel_lateral, out.vel_vertical
 *       //   out.heading - Walking direction angle used for transformation
 *       
 *       // Feed to NN classifier input buffer
 *       // Classifier handles decimation to 20Hz predictions
 *   }
 *   
 *   // At each heel strike, step summary is ready
 *   if (gait::hasNewStep()) {
 *       gait::StepSummary step = gait::getStepSummary();
 *       // Immediate terrain classification possible
 *   }
 */

} // namespace gait
