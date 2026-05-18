/**
 * @file gait_filter.cpp
 * @brief Implementation of the step-to-step position filter for gait classification
 */

#include "gait_filter.hpp"
#include <cmath>

#ifdef ARDUINO
    #include <Arduino.h>
    #define GET_TIME_MS() millis()
#else
    #include <chrono>
    static uint32_t GET_TIME_MS() {
        static auto start = std::chrono::steady_clock::now();
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
    }
#endif

namespace gait {

// ============================================================================
// Construction / Configuration
// ============================================================================

GaitFilter::GaitFilter()
    : config_()
    , state_()
    , output_()
    , step_summary_()
    , max_height_this_step_(0)
    , min_height_this_step_(0)
    , peak_vertical_vel_this_step_(0)
    , step_start_pos_(Vec3f::Zero())
    , step_start_time_ms_(0)
    , stance_start_time_ms_(0)
    , swing_start_time_ms_(0)
    , imu_sample_count_(0)
    , current_gyro_(Vec3f::Zero())
    , current_free_accel_(Vec3f::Zero())
    , initialized_(false)
    , output_ready_(false)
    , step_complete_(false)
    , zupt_active_(false)
    , in_stance_(false)
    , last_toe_off_ms_(0)
    , was_in_stance_(false)
    , stance_in_previous_(false)
    , shoe_stance_previous_(false)
    , walking_dir_cos_(1.0f)
    , walking_dir_sin_(0.0f)
    , walking_dir_valid_(false)
    , toe_off_heading_(0.0f)
    , gravity_world_(0, 0, -GRAVITY)
{
    reset();
}

GaitFilter::GaitFilter(const FilterConfig& config)
    : GaitFilter()
{
    configure(config);
}

void GaitFilter::configure(const FilterConfig& config) {
    config_ = config;
    
    // Update covariance based on config
    state_.P.setZero();
    state_.P.block<3,3>(0,0) = Mat3f::Identity() * config_.init_pos_cov;
    state_.P.block<3,3>(3,3) = Mat3f::Identity() * config_.init_vel_cov;
    state_.P.block<3,3>(6,6) = Mat3f::Identity() * config_.init_ori_cov;
}

void GaitFilter::reset() {
    state_ = FilterState();
    output_ = GaitFilterOutput();
    step_summary_ = StepSummary();
    
    max_height_this_step_ = 0;
    min_height_this_step_ = 0;
    peak_vertical_vel_this_step_ = 0;
    step_start_pos_ = Vec3f::Zero();
    step_start_time_ms_ = GET_TIME_MS();
    stance_start_time_ms_ = step_start_time_ms_;
    swing_start_time_ms_ = step_start_time_ms_;
    imu_sample_count_ = 0;
    
    initialized_ = false;
    output_ready_ = false;
    step_complete_ = false;
    zupt_active_ = false;
    in_stance_ = false;
    last_toe_off_ms_ = 0;
    was_in_stance_ = false;
    stance_in_previous_ = false;
    shoe_stance_previous_ = false;
    
    // Reset walking direction (will be computed at first heel strike)
    walking_dir_cos_ = 1.0f;
    walking_dir_sin_ = 0.0f;
    walking_dir_valid_ = false;
    
    // Initialize covariance
    state_.P.setZero();
    state_.P.block<3,3>(0,0) = Mat3f::Identity() * config_.init_pos_cov;
    state_.P.block<3,3>(3,3) = Mat3f::Identity() * config_.init_vel_cov;
    state_.P.block<3,3>(6,6) = Mat3f::Identity() * config_.init_ori_cov;
}

void GaitFilter::initializeOrientation(const Quatf& quat) {
    state_.orientation = quat.normalized();
    initialized_ = true;
    step_start_time_ms_ = GET_TIME_MS();
}

void GaitFilter::initializeForward(const Quatf& quat) {
    // First, do the normal orientation initialization
    initializeOrientation(quat);
    
    // For shank-mounted IMU, forward = body -Z axis
    // Extract horizontal components of body -Z directly from quaternion
    // (more efficient than building full rotation matrix)
    //
    // Body -Z in world frame = negated 3rd column of rotation matrix:
    //   dx = -2(xz + wy)
    //   dy = 2(wx - yz)
    Quatf q = quat.normalized();
    float w = q.w(), x = q.x(), y = q.y(), z = q.z();
    
    float dx = -2.0f * (x*z + w*y);  // East component of forward
    float dy =  2.0f * (w*x - y*z);  // North component of forward
    float horiz_dist = std::sqrt(dx*dx + dy*dy);
    
    if (horiz_dist > 0.01f) {  // Sufficient horizontal projection to determine direction
        walking_dir_cos_ = dx / horiz_dist;
        walking_dir_sin_ = dy / horiz_dist;
        walking_dir_valid_ = true;
    } else {
        // IMU -Z axis is nearly vertical (shank horizontal / leg sticking out)
        // Cannot determine forward direction, fall back to East = forward
        walking_dir_cos_ = 1.0f;
        walking_dir_sin_ = 0.0f;
        walking_dir_valid_ = false;
    }
}

// ============================================================================
// Main Update Loop
// ============================================================================

void GaitFilter::update(const IMUMeasurement& meas) {
    uint32_t now_ms = GET_TIME_MS();
    
    // Initialize orientation from first measurement
    if (!initialized_) {
        initializeOrientation(meas.orientation);
        return;
    }
    
    // Update orientation from MTI-3 (we trust its XKF filter)
    state_.orientation = meas.orientation.normalized();
    
    // Update ZUPT detection buffer
    updateZuptBuffer(meas);
    
    // Prediction step
    if (meas.use_preintegrated) {
        predictWithPreintegrated(meas);
    } else if (meas.free_accel.squaredNorm() > 0.01f) {
        predictWithFreeAcceleration(meas);
    } else {
        predictWithAcceleration(meas);
    }
    
    // Detect stance ONCE per update cycle (heel strike to toe off)
    // detectStance() runs w_sag, ADC, and foot-flat fallback for gait phase.
    // It also runs detectFF/detectFFDebounced internally for foot-flat/ZUPT.
    bool in_stance = detectStance();
    in_stance_ = in_stance;
    
    // Track toe-off timestamp for motor state gating
    if (!in_stance && was_in_stance_) {
        last_toe_off_ms_ = now_ms;
    }
    was_in_stance_ = in_stance;
    
    // Gait phase detection (uses stance result for HS/TO transitions)
    updateGaitPhase(meas, in_stance);
    
    // Apply ZUPT/pivot constraint based on SHOE foot-flat window
    // (independent of stance detection which uses omega-sag / ADC)
    bool foot_flat = state_.shoe_stance_active;
    if (foot_flat) {
        applyZupt(meas);
        zupt_active_ = true;
        
        // Zero position ONCE on foot-flat onset
        if (!stance_in_previous_) {
            state_.position = Vec3f::Zero();
            // Decouple position from velocity so the clamp doesn't corrupt v
            state_.P.block<3,3>(0,3).setZero();
            state_.P.block<3,3>(3,0).setZero();
        }
    } else {
        zupt_active_ = false;
    }
    stance_in_previous_ = foot_flat;
    
    // ========================================================================
    // Update walking direction on foot-flat falling edge (heel off)
    // This is independent of the gait-phase state machine so heading stays
    // correct even when omega-sag or ADC detectors have different timing.
    // ========================================================================
    if (!foot_flat && shoe_stance_previous_) {
        captureHeadingAtHeelOff();
    }
    shoe_stance_previous_ = foot_flat;
    
    // Update step metrics
    updateStepMetrics();
    
    // Increment sample counter
    imu_sample_count_++;
    state_.samples_in_phase++;
    
    // Generate output at IMU rate (100 Hz)
    // Downstream classifier handles decimation to 20 Hz
    generateOutput(now_ms);
    output_ready_ = true;
}

void GaitFilter::update(const float* dv, const float* dq, const float* quat,
                        const float* free_acc, uint32_t sample_time) {
    IMUMeasurement meas;
    
    meas.dv = Vec3f(dv[0], dv[1], dv[2]);
    meas.dq = Quatf(dq[0], dq[1], dq[2], dq[3]);  // Assuming [w,x,y,z] order
    meas.orientation = Quatf(quat[0], quat[1], quat[2], quat[3]);
    meas.free_accel = Vec3f(free_acc[0], free_acc[1], free_acc[2]);
    meas.sample_time_fine = sample_time;
    // Determine predict function to use
    // If true, use pre-integrated dv/dq; else use free acceleration
    meas.use_preintegrated = true;
    
    update(meas);
}

// ============================================================================
// Output Access
// ============================================================================

GaitFilterOutput GaitFilter::getOutput() {
    output_ready_ = false;
    return output_;
}

StepSummary GaitFilter::getStepSummary() {
    step_complete_ = false;
    return step_summary_;
}

// ============================================================================
// Prediction Methods
// ============================================================================

void GaitFilter::predictWithPreintegrated(const IMUMeasurement& meas) {
    // Cache free acceleration for output (already in world frame, gravity-compensated)
    current_free_accel_ = meas.free_accel;
    
    // MTI-3 dv is already integrated acceleration over the sample period
    // dv is in BODY frame, need to rotate to world frame
    
    Mat3f R = state_.orientation.toRotationMatrix();
    
    // Rotate dv from body to world frame
    Vec3f dv_world = R * meas.dv;
    
    // Remove gravity component (dv includes gravity effect)
    // The MTI-3 dv output already has gravity included, so we need to remove it
    dv_world += gravity_world_ * config_.dt_imu;
    // dv_world -= gravity_world_ * config_.dt_imu;
    
    // Update velocity
    state_.velocity += dv_world;
    
    // Apply mild vertical damping to reduce drift
    state_.velocity.z() *= (1.0f - config_.vertical_damping);
    
    // Update position
    state_.position += state_.velocity * config_.dt_imu;
    
    // Optimized covariance prediction for 100Hz
    // Pre-computed constants (dt = 0.01s at 100Hz)
    const float dt = config_.dt_imu;
    const float dt2 = dt * dt;
    const float sigma_a2 = config_.sigma_accel_noise * config_.sigma_accel_noise;
    const float sigma_w2 = config_.sigma_gyro_noise * config_.sigma_gyro_noise;
    
    // Process noise variances (diagonal terms only for speed)
    const float q_pos = sigma_a2 * dt2 * dt / 3.0f;  // Position noise
    const float q_vel = sigma_a2 * dt;               // Velocity noise
    const float q_ori = sigma_w2 * dt;               // Orientation noise
    
    // Simplified covariance prediction exploiting sparsity of F
    // F = [I  dt*I  0]    State transition is simple: p += v*dt
    //     [0   I    0]
    //     [0   0    I]
    // 
    // P' = F*P*F' + Q can be computed efficiently:
    // P'_pp = P_pp + dt*(P_pv + P_vp) + dt²*P_vv + Q_pp
    // P'_pv = P_pv + dt*P_vv
    // P'_vv = P_vv + Q_vv
    // P'_oo = P_oo + Q_oo
    
    // Extract blocks
    Mat3f Ppp = state_.P.block<3,3>(0,0);
    Mat3f Ppv = state_.P.block<3,3>(0,3);
    Mat3f Pvp = state_.P.block<3,3>(3,0);
    Mat3f Pvv = state_.P.block<3,3>(3,3);
    
    // Update position covariance
    state_.P.block<3,3>(0,0) = Ppp + dt * (Ppv + Pvp) + dt2 * Pvv + Mat3f::Identity() * q_pos;
    state_.P.block<3,3>(0,3) = Ppv + dt * Pvv;
    state_.P.block<3,3>(3,0) = Pvp + dt * Pvv;
    
    // Update velocity covariance  
    state_.P.block<3,3>(3,3) = Pvv + Mat3f::Identity() * q_vel;
    
    // Update orientation covariance (independent)
    state_.P.block<3,3>(6,6) += Mat3f::Identity() * q_ori;
    
    // Ensure symmetry (just cross-terms)
    state_.P.block<3,3>(3,0) = state_.P.block<3,3>(0,3).transpose();
}

void GaitFilter::predictWithFreeAcceleration(const IMUMeasurement& meas) {
    // Free acceleration is already in world frame and gravity-compensated
    const float dt = config_.dt_imu;
    const float dt2 = dt * dt;
    
    // Update velocity
    state_.velocity += meas.free_accel * dt;
    
    // Apply mild vertical damping
    state_.velocity.z() *= (1.0f - config_.vertical_damping);
    
    // Update position
    state_.position += state_.velocity * dt + 0.5f * meas.free_accel * dt2;
    
    // Optimized covariance (same structure as preintegrated)
    const float sigma_a2 = config_.sigma_accel_noise * config_.sigma_accel_noise;
    const float sigma_w2 = config_.sigma_gyro_noise * config_.sigma_gyro_noise;
    const float q_pos = sigma_a2 * dt2 * dt / 3.0f;
    const float q_vel = sigma_a2 * dt;
    const float q_ori = sigma_w2 * dt;
    
    Mat3f Ppp = state_.P.block<3,3>(0,0);
    Mat3f Ppv = state_.P.block<3,3>(0,3);
    Mat3f Pvp = state_.P.block<3,3>(3,0);
    Mat3f Pvv = state_.P.block<3,3>(3,3);
    
    state_.P.block<3,3>(0,0) = Ppp + dt * (Ppv + Pvp) + dt2 * Pvv + Mat3f::Identity() * q_pos;
    state_.P.block<3,3>(0,3) = Ppv + dt * Pvv;
    state_.P.block<3,3>(3,0) = Pvp + dt * Pvv;
    state_.P.block<3,3>(3,3) = Pvv + Mat3f::Identity() * q_vel;
    state_.P.block<3,3>(6,6) += Mat3f::Identity() * q_ori;
    state_.P.block<3,3>(3,0) = state_.P.block<3,3>(0,3).transpose();
}

void GaitFilter::predictWithAcceleration(const IMUMeasurement& meas) {
    // Raw acceleration needs gravity removal
    Mat3f R = state_.orientation.toRotationMatrix();
    const float dt = config_.dt_imu;
    const float dt2 = dt * dt;
    
    // Rotate acceleration to world frame
    Vec3f accel_world = R * meas.accel;
    
    // Remove gravity
    Vec3f free_accel = accel_world - gravity_world_;
    
    // Update velocity and position
    state_.velocity += free_accel * dt;
    state_.velocity.z() *= (1.0f - config_.vertical_damping);
    state_.position += state_.velocity * dt + 0.5f * free_accel * dt2;
    
    // Optimized covariance
    const float sigma_a2 = config_.sigma_accel_noise * config_.sigma_accel_noise;
    const float sigma_w2 = config_.sigma_gyro_noise * config_.sigma_gyro_noise;
    const float q_pos = sigma_a2 * dt2 * dt / 3.0f;
    const float q_vel = sigma_a2 * dt;
    const float q_ori = sigma_w2 * dt;
    
    Mat3f Ppp = state_.P.block<3,3>(0,0);
    Mat3f Ppv = state_.P.block<3,3>(0,3);
    Mat3f Pvp = state_.P.block<3,3>(3,0);
    Mat3f Pvv = state_.P.block<3,3>(3,3);
    
    state_.P.block<3,3>(0,0) = Ppp + dt * (Ppv + Pvp) + dt2 * Pvv + Mat3f::Identity() * q_pos;
    state_.P.block<3,3>(0,3) = Ppv + dt * Pvv;
    state_.P.block<3,3>(3,0) = Pvp + dt * Pvv;
    state_.P.block<3,3>(3,3) = Pvv + Mat3f::Identity() * q_vel;
    state_.P.block<3,3>(6,6) += Mat3f::Identity() * q_ori;
    state_.P.block<3,3>(3,0) = state_.P.block<3,3>(0,3).transpose();
}

// ============================================================================
// ZUPT Detection and Correction
// ============================================================================

void GaitFilter::updateZuptBuffer(const IMUMeasurement& meas) {
    // Get acceleration and gyro in body frame
    Vec3f accel_body, gyro_body;
    
    if (meas.use_preintegrated) {
        // dv magnitude normalized by dt gives acceleration
        accel_body = meas.dv / config_.dt_imu;
        
        // Extract gyro from dq: dq ≈ [1, 0.5*ω*dt] for small angles
        // ω ≈ 2 * vec(dq) / dt
        Vec3f dq_vec(meas.dq.x(), meas.dq.y(), meas.dq.z());
        gyro_body = 2.0f * dq_vec / config_.dt_imu;
    } else {
        accel_body = meas.accel;
        gyro_body = meas.gyro;
    }
    
    // Store current gyro for pivot constraint
    current_gyro_ = gyro_body;
    
    // Update circular buffer with vectors (for SHOE)
    // uint8_t idx = state_.buffer_idx % SHOE_BUFFER_SIZE;
    uint8_t idx = state_.buffer_idx;
    state_.accel_buffer_vec[idx] = accel_body;
    state_.gyro_buffer_vec[idx] = gyro_body;
    
    // Also store magnitudes (for legacy detector)
    state_.accel_buffer[idx] = accel_body.norm();
    state_.gyro_buffer[idx] = gyro_body.norm();
    
    // state_.buffer_idx++;
    state_.buffer_idx = (state_.buffer_idx + 1) % SHOE_BUFFER_SIZE;
}

bool GaitFilter::detectFF() {
    // ========================================================================
    // SHOE Foot-Flat Detector (Skog et al.)
    // ========================================================================
    // 
    // Generalized Likelihood Ratio Test (GLRT):
    // Under H0 (stationary): accelerometer measures only gravity, gyro measures zero
    // Under H1 (moving): arbitrary acceleration and rotation
    //
    // Test statistic over window of W samples:
    // T = (1/W) * Σ[ ||a_k - g * (a_k / ||a_k||)||² / σ_a² + ||ω_k||² / σ_w² ]
    //
    // The first term measures deviation from pure gravity direction
    // The second term measures rotation magnitude
    //
    // Stance detected when T < γ (threshold)
    // ========================================================================
    
    const uint8_t W = config_.shoe_window_size;
    if (W > SHOE_BUFFER_SIZE) return false;  // Buffer too small
    
    const float sigma_a2 = config_.shoe_sigma_a * config_.shoe_sigma_a;
    const float sigma_w2 = config_.shoe_sigma_w * config_.shoe_sigma_w;
    const float g = GRAVITY;
    
    float T = 0.0f;
    
    for (uint8_t i = 0; i < W; i++) {
        // uint8_t idx = (state_.buffer_idx - 1 - i) % 8;
        uint8_t idx = (state_.buffer_idx + SHOE_BUFFER_SIZE - 1 - i) % SHOE_BUFFER_SIZE;
        
        Vec3f a_k = state_.accel_buffer_vec[idx];
        Vec3f w_k = state_.gyro_buffer_vec[idx];
        
        // Compute normalized acceleration direction
        float a_norm = a_k.norm();
        if (a_norm < 0.1f) a_norm = 0.1f;  // Prevent division by zero
        
        // Gravity estimate in body frame: g * (a_k / ||a_k||)
        Vec3f g_est = g * (a_k / a_norm);
        
        // Acceleration term: ||a_k - g_est||² / σ_a²
        Vec3f a_diff = a_k - g_est;
        float accel_term = a_diff.squaredNorm() / sigma_a2;
        
        // Gyro term: ||ω_k||² / σ_w²
        float gyro_term = w_k.squaredNorm() / sigma_w2;
        
        T += accel_term + gyro_term;
    }
    
    T /= W;
    
    // Store for debugging
    state_.last_shoe_statistic = T;
    
    // Return raw threshold comparison (debouncing handled in detectFFDebounced())
    return T < config_.shoe_gamma_enter;
}

float GaitFilter::computeWsag() const {
    // Rotate body-frame gyro to world frame using current orientation
    Mat3f R = state_.orientation.toRotationMatrix();
    Vec3f gyro_world = R * current_gyro_;

    // Project onto the medial-lateral axis (perpendicular to walking direction in the horizontal plane)
    // Walking direction: (walking_dir_cos_, walking_dir_sin_, 0)
    // Lateral axis:     (-walking_dir_sin_, walking_dir_cos_, 0)
    // w_sag = component of gyro_world along the lateral axis
    return -walking_dir_sin_ * gyro_world.x() + walking_dir_cos_ * gyro_world.y();
}

bool GaitFilter::detectStanceOmegaSag(float w_sag_current) {
    // =========================================================================
    // Omega-sag stance detector (three complementary heel-strike criteria)
    //
    // HEEL STRIKE detection (any triggers entry):
    //
    //   1. Sign-change criterion (original):
    //      Over the last N frames of delta_w_sag, >= 2 sign changes indicate
    //      an oscillatory impact signature.  Catches sharp heel strikes.
    //
    //   2. Zero-crossing criterion (new):
    //      w_sag crosses from negative to positive AND the minimum w_sag
    //      during the preceding swing was below omega_sag_hs_neg_amplitude.
    //      Catches smooth decelerating heel strikes where the shank simply
    //      reverses direction without a sharp oscillation.
    //
    //   3. Quiescence criterion (new):
    //      |w_sag| stays below quiescence_threshold for quiescence_samples
    //      consecutive frames.  Detects quiet standing / static stance when
    //      the foot was already on the ground before walking started.
    //
    // TOE-OFF detection:
    //   Exit stance when w_sag < omega_sag_toe_off_threshold after the
    //   minimum dwell period, preventing mid-stance chatter.
    // =========================================================================

    // Clamp configurable window to physical buffer size
    const uint8_t W = (config_.omega_sag_sign_window <= OMEGA_SIGN_BUF_SIZE)
                        ? config_.omega_sag_sign_window
                        : OMEGA_SIGN_BUF_SIZE;

    // ------ 1. Sign-change analysis ------
    float delta = w_sag_current - state_.prev_w_sag;

    int8_t sign;
    if      (delta >  config_.omega_sag_hs_delta_threshold) sign =  1;
    else if (delta < -config_.omega_sag_hs_delta_threshold) sign = -1;
    else                                                     sign =  0;

    // Push into ring buffer (write then advance)
    state_.delta_sign_buf[state_.delta_sign_idx] = sign;
    state_.delta_sign_idx = (state_.delta_sign_idx + 1) % OMEGA_SIGN_BUF_SIZE;

    // Count sign changes over the W most-recent entries (oldest first)
    uint8_t sign_changes = 0;
    int8_t  prev_sign    = 0;
    for (uint8_t i = 0; i < W; i++) {
        uint8_t ridx = (state_.delta_sign_idx + OMEGA_SIGN_BUF_SIZE - W + i)
                        % OMEGA_SIGN_BUF_SIZE;
        int8_t  s = state_.delta_sign_buf[ridx];
        if (s != 0) {
            if (prev_sign != 0 && s != prev_sign) {
                sign_changes++;
            }
            prev_sign = s;
        }
    }

    // ------ 2. Zero-crossing analysis ------
    bool zero_cross_hs = false;
    if (!state_.omega_stance_active && state_.omega_refractory == 0) {
        // Track the deepest negative w_sag seen during this swing phase
        if (w_sag_current < state_.omega_min_wsag_in_swing) {
            state_.omega_min_wsag_in_swing = w_sag_current;
        }
        // Negative-to-positive crossing with sufficient preceding excursion
        if (state_.prev_w_sag < 0.0f && w_sag_current >= 0.0f &&
            state_.omega_min_wsag_in_swing < config_.omega_sag_hs_neg_amplitude) {
            zero_cross_hs = true;
        }
    }

    // ------ 3. Quiescence analysis ------
    if (std::abs(w_sag_current) < config_.omega_sag_quiescence_threshold) {
        if (state_.omega_quiescence_count < 255) state_.omega_quiescence_count++;
    } else {
        state_.omega_quiescence_count = 0;
    }
    bool quiescence_hs = (!state_.omega_stance_active &&
                          state_.omega_refractory == 0 &&
                          state_.omega_quiescence_count >= config_.omega_sag_quiescence_samples);

    // ------ 4. SHOE fallback ------
    // If SHOE (debounced) reports foot-flat but omega-sag hasn't triggered,
    // use SHOE as a safety net so stance is never missed entirely.
    bool shoe_fallback_hs = (!state_.omega_stance_active &&
                             state_.omega_refractory == 0 &&
                             state_.shoe_stance_active);

    // ------ State machine ------
    bool hs_detected = false;

    if (state_.omega_refractory > 0) {
        // Within refractory period – suppress re-triggers from ring-down
        state_.omega_refractory--;
        if (state_.omega_stance_active && state_.omega_samples_in_stance < 255) {
            state_.omega_samples_in_stance++;
        }
    } else if (!state_.omega_stance_active) {
        // Swing / unknown phase: check all three heel-strike criteria
        if ((sign_changes >= 2 && zero_cross_hs) || quiescence_hs || shoe_fallback_hs) {
        // if (sign_changes >= 2 || quiescence_hs || shoe_fallback_hs) {
            state_.omega_stance_active     = true;
            // No refractory needed after quiescence or SHOE fallback
            state_.omega_refractory        = (quiescence_hs || shoe_fallback_hs)
                                                ? 0
                                                : config_.omega_sag_refractory_samples;
            state_.omega_samples_in_stance = 0;
            state_.omega_quiescence_count  = 0;
            state_.omega_min_wsag_in_swing = 0.0f;
            // Clear sign buffer to prevent stale re-triggers
            for (uint8_t i = 0; i < OMEGA_SIGN_BUF_SIZE; i++)
                state_.delta_sign_buf[i] = 0;
            hs_detected = true;
        }
    } else {
        // Stance phase: count dwell time
        if (state_.omega_samples_in_stance < 255) {
            state_.omega_samples_in_stance++;
        }
        // Toe-off requires BOTH conditions for N consecutive frames:
        //   1. w_sag < toe_off_threshold  (shank rotating backward)
        //   2. dw_sag/dt < deriv_threshold (still accelerating into toe-off, not recovering)
        // This rejects single-frame transient dips during stair ascent.
        bool toe_off_cond = (state_.omega_samples_in_stance >= config_.omega_sag_min_stance_samples &&
                             w_sag_current < config_.omega_sag_toe_off_threshold &&
                             delta < config_.omega_sag_toe_off_deriv_threshold);
        if (toe_off_cond) {
            state_.omega_toe_off_debounce_count++;
        } else {
            state_.omega_toe_off_debounce_count = 0;
        }
        if (state_.omega_toe_off_debounce_count >= config_.omega_sag_toe_off_debounce) {
            state_.omega_stance_active     = false;
            state_.omega_samples_in_stance = 0;
            state_.omega_toe_off_debounce_count = 0;
            state_.omega_min_wsag_in_swing = 0.0f;   // begin tracking for next swing
            // Clear sign buffer to prevent stale re-triggers on next swing
            for (uint8_t i = 0; i < OMEGA_SIGN_BUF_SIZE; i++)
                state_.delta_sign_buf[i] = 0;
        }
    }

    // Debug output
    output_.w_sag_delta       = delta;
    output_.omega_hs_detected = hs_detected ? 1 : 0;

    state_.prev_w_sag = w_sag_current;
    return state_.omega_stance_active;
}

bool GaitFilter::detectStanceAdc() {
    // =========================================================================
    // ADC-based stance detector
    //
    // Uses baseline-relative strain-gauge voltage (V = counts * 2.048 / 2^23,
    // already inverted so positive = loaded, baseline captured at no-load boot).
    //
    //   Enter stance: adc_v > adc_stance_enter  (loading / heel strike)
    //   Exit stance:  adc_v < adc_stance_exit   (unloading / toe-off)
    //
    // Minimum dwell guard:
    //   Once in stance, exit is blocked for adc_min_stance_samples frames.
    //   This handles brief load drops during mid-stance (e.g. heel-to-toe roll).
    // =========================================================================

    float adc = state_.adc_raw_last;

    if (!state_.adc_stance_active) {
        // Swing / no-load: watch for loading threshold (heel strike)
        if (adc > config_.adc_stance_enter) {
            state_.adc_stance_active     = true;
            state_.adc_samples_in_stance = 0;
        }
    } else {
        // Stance: increment dwell counter
        if (state_.adc_samples_in_stance < 255) {
            state_.adc_samples_in_stance++;
        }
        // Allow exit only after minimum dwell
        if (state_.adc_samples_in_stance >= config_.adc_min_stance_samples &&
            adc < config_.adc_stance_exit) {
            state_.adc_stance_active     = false;
            state_.adc_samples_in_stance = 0;
        }
    }

    output_.adc_raw_out   = adc;
    output_.adc_stance_out = state_.adc_stance_active ? 1 : 0;

    return state_.adc_stance_active;
}

bool GaitFilter::detectFFDebounced() {
    // =========================================================================
    // Debounced SHOE foot-flat detector for ZUPT / pivot constraint
    //
    // Uses hysteretic thresholds (shoe_gamma_enter / shoe_gamma_exit) with
    // configurable debounce counts to produce a stable foot-flat window.
    // Foot-flat drives ZUPT and position reset.  When foot-flat exits
    // (heel off), walking direction is captured.
    // =========================================================================
    float T = state_.last_shoe_statistic;

    if (!state_.shoe_stance_active) {
        // Not in foot-flat: check for entry (T drops below gamma_enter)
        if (T < config_.shoe_gamma_enter) {
            state_.shoe_debounce_count++;
            if (state_.shoe_debounce_count >= config_.shoe_debounce_enter) {
                state_.shoe_stance_active    = true;
                state_.shoe_debounce_count   = 0;
                state_.shoe_samples_in_flat  = 0;
            }
        } else {
            state_.shoe_debounce_count = 0;
        }
    } else {
        // Track how long we have been in the foot-flat window
        if (state_.shoe_samples_in_flat < 255) {
            state_.shoe_samples_in_flat++;
        }
        // In foot-flat: check for exit / heel off (T rises above gamma_exit)
        // Exit is gated: foot-flat must have lasted at least shoe_min_flat_samples
        if (T > config_.shoe_gamma_exit &&
            state_.shoe_samples_in_flat >= config_.shoe_min_flat_samples) {
            state_.shoe_debounce_count++;
            if (state_.shoe_debounce_count >= config_.shoe_debounce_exit) {
                state_.shoe_stance_active   = false;
                state_.shoe_debounce_count  = 0;
                state_.shoe_samples_in_flat = 0;
            }
        } else if (T <= config_.shoe_gamma_exit) {
            state_.shoe_debounce_count = 0;
        }
    }

    return state_.shoe_stance_active;
}

bool GaitFilter::detectStance() {
    // -------------------------------------------------------------------------
    // 1. Always run SHOE foot-flat detection → state_.shoe_stance_active
    //    Foot-flat drives ZUPT/pivot constraint and position reset.
    // -------------------------------------------------------------------------
    detectFF();                         // raw statistic → state_.last_shoe_statistic
    detectFFDebounced();                // debounce → state_.shoe_stance_active

    float T = state_.last_shoe_statistic;
    output_.raw_should_enter = (T < config_.shoe_gamma_enter) ? 1 : 0;
    output_.raw_should_exit  = (T > config_.shoe_gamma_exit)  ? 1 : 0;

    // -------------------------------------------------------------------------
    // 2. Always run omega-sag detector (stance = heel strike to toe off)
    // -------------------------------------------------------------------------
    float w_sag = computeWsag();
    bool omega_stance = detectStanceOmegaSag(w_sag);

    if (!config_.use_adc_detector) {
        // No ADC: use omega-sag alone for stance
        output_.debounce_counter = state_.omega_refractory;
        return omega_stance;
    }

    // -------------------------------------------------------------------------
    // 3. Run ADC detector
    // -------------------------------------------------------------------------
    bool adc_stance = detectStanceAdc();
    bool foot_flat  = state_.shoe_stance_active;

    // -------------------------------------------------------------------------
    // 4. Fused stance detection (ADC enabled)
    //
    //    Heel strike (entry):
    //      Whichever of omega-sag, ADC, or foot-flat (fallback) fires first.
    //      Detected via rising edge of any individual detector.
    //
    //    Toe-off (exit):
    //      Whichever of omega-sag or ADC fires first.
    //      Detected via falling edge of either detector.
    // -------------------------------------------------------------------------

    // Rising edges (heel-strike events)
    bool omega_entered = omega_stance && !state_.prev_omega_for_combined;
    bool adc_entered   = adc_stance   && !state_.prev_adc_for_combined;
    // SHOE foot-flat entry: used as fallback if neither omega nor ADC fired

    // Falling edges (toe-off events)
    bool omega_exited = !omega_stance && state_.prev_omega_for_combined;
    bool adc_exited   = !adc_stance   && state_.prev_adc_for_combined;

    // Update previous-cycle latches
    state_.prev_omega_for_combined = omega_stance;
    state_.prev_adc_for_combined   = adc_stance;

    // Combined state machine
    if (!state_.combined_stance_active) {
        // Entry: any detector transitions into stance, or foot-flat fallback
        if (omega_entered || adc_entered || foot_flat) {
            state_.combined_stance_active = true;
        }
    } else {
        // Exit: either omega-sag or ADC transitions out of stance
        if (omega_exited || adc_exited) {
            state_.combined_stance_active = false;
        }
    }

    output_.debounce_counter = state_.adc_samples_in_stance;
    return state_.combined_stance_active;
}

void GaitFilter::applyPivotConstraint(const Vec3f& gyro) {
    // ========================================================================
    // Pivot Constraint for Shank-Mounted IMU (Wagstaff et al., Nilsson et al.)
    // ========================================================================
    //
    // During stance, the ankle acts as a pivot point with zero velocity:
    //   v_ankle = 0
    //
    // The relationship between IMU velocity and ankle velocity:
    //   v_ankle = v_imu + R * (ω × r)
    // 
    // where:
    //   R = rotation matrix from body to world frame
    //   ω = angular velocity in body frame
    //   r = vector from IMU to ankle in body frame (r_imu_to_ankle)
    //
    // The constraint becomes:
    //   0 = v_imu + R * (ω × r)
    //   v_imu = -R * (ω × r)
    //
    // Measurement model:
    //   z = 0 (ankle velocity)
    //   h(x) = v_imu + R * (ω × r)
    //   H = [0  I  ...] + jacobian of R*(ω×r) w.r.t. error state
    //
    // For simplicity, we use the measurement:
    //   z = -R * (ω × r) = -R * skew(ω) * r = R * skew(r) * ω
    // and compare against current velocity estimate
    // ========================================================================
    
    Mat3f R = state_.orientation.toRotationMatrix();
    Vec3f r = config_.r_imu_to_ankle;
    
    // Compute expected IMU velocity during stance
    // v_expected = -R * (ω × r) = R * (r × ω)
    Vec3f omega_cross_r = gyro.cross(r);  // ω × r in body frame
    Vec3f v_expected = -R * omega_cross_r;  // Transform to world frame
    
    // Measurement: y = v_expected - v_current
    Vec3f y = v_expected - state_.velocity;
    
    // Measurement noise (accounts for lever arm uncertainty and gyro noise)
    const float r_base = config_.pivot_velocity_noise * config_.pivot_velocity_noise;
    // Add noise proportional to angular velocity (lever arm uncertainty)
    float gyro_mag = gyro.norm();
    float r_lever = 0.01f * gyro_mag * gyro_mag;  // Additional noise from lever arm
    Mat3f R_noise = Mat3f::Identity() * (r_base + r_lever);
    
    // Kalman update - H = [0 I 0] (observing velocity)
    Mat3f Pvv = state_.P.block<3,3>(3,3);
    Mat3f S = Pvv + R_noise;
    Mat3f S_inv = S.inverse();
    
    // Kalman gains
    Mat3f Ppv = state_.P.block<3,3>(0,3);
    Mat3f K_p = Ppv * S_inv;
    Mat3f K_v = Pvv * S_inv;
    
    // State correction
    // Position correction is optional (when disabled, position is pure integral of velocity)
    if (config_.apply_position_correction) {
        state_.position += K_p * y;
    }
    state_.velocity += K_v * y;
    
    // Covariance update
    Mat3f I_Kv = Mat3f::Identity() - K_v;
    
    // Update P using Joseph form
    // Note: Covariance update always uses K_p to maintain filter consistency,
    // even when position correction is disabled. This means the covariance
    // will still reflect the "optimal" estimate uncertainty.
    state_.P.block<3,3>(0,0) -= K_p * Ppv.transpose() + Ppv * K_p.transpose() 
                               - K_p * Pvv * K_p.transpose() 
                               + K_p * R_noise * K_p.transpose();
    
    Mat3f new_Ppv = Ppv * I_Kv.transpose() + K_p * R_noise * K_v.transpose();
    state_.P.block<3,3>(0,3) = new_Ppv;
    state_.P.block<3,3>(3,0) = new_Ppv.transpose();
    
    state_.P.block<3,3>(3,3) = I_Kv * Pvv * I_Kv.transpose() 
                              + K_v * R_noise * K_v.transpose();
}

void GaitFilter::applyZuptLegacy() {
    // Original zero-velocity update (velocity = 0)
    const float r = config_.zupt_velocity_noise * config_.zupt_velocity_noise;
    
    Vec3f y = -state_.velocity;
    
    Mat3f Pvv = state_.P.block<3,3>(3,3);
    Mat3f S = Pvv + Mat3f::Identity() * r;
    Mat3f S_inv = S.inverse();
    
    Mat3f Ppv = state_.P.block<3,3>(0,3);
    Mat3f K_p = Ppv * S_inv;
    Mat3f K_v = Pvv * S_inv;
    
    // Position correction is optional (when disabled, position is pure integral of velocity)
    if (config_.apply_position_correction) {
        state_.position += K_p * y;
    }
    state_.velocity += K_v * y;
    
    Mat3f I_Kv = Mat3f::Identity() - K_v;
    
    // Covariance update always uses K_p to maintain filter consistency
    state_.P.block<3,3>(0,0) -= K_p * Ppv.transpose() + Ppv * K_p.transpose() 
                               - K_p * Pvv * K_p.transpose() 
                               + K_p * Mat3f::Identity() * r * K_p.transpose();
    
    Mat3f new_Ppv = Ppv * I_Kv.transpose() + K_p * r * K_v.transpose();
    state_.P.block<3,3>(0,3) = new_Ppv;
    state_.P.block<3,3>(3,0) = new_Ppv.transpose();
    
    state_.P.block<3,3>(3,3) = I_Kv * Pvv * I_Kv.transpose() 
                              + K_v * Mat3f::Identity() * r * K_v.transpose();
}

void GaitFilter::applyZupt(const IMUMeasurement& meas) {
    if (config_.use_pivot_constraint) {
        applyPivotConstraint(current_gyro_);
    } else {
        applyZuptLegacy();
    }
}

// ============================================================================
// Gait Phase Detection
// ============================================================================

void GaitFilter::updateGaitPhase(const IMUMeasurement& meas, bool in_stance) {
    // Store raw detection result for output (useful for debugging)
    output_.stance_active = in_stance ? 1 : 0;
    
    // State machine for gait phase
    switch (state_.phase) {
        case GaitPhase::UNKNOWN:
            // Wait for clear stance detection to initialize
            if (in_stance) {
                state_.phase = GaitPhase::STANCE;
                stance_start_time_ms_ = GET_TIME_MS();
                state_.samples_in_phase = 0;  // Reset counter when entering STANCE
            }
            break;
            
        case GaitPhase::STANCE:
            if (!in_stance && state_.samples_in_phase > 10) {
                // Transition to toe-off
                state_.phase = GaitPhase::TOE_OFF;
                onToeOff();
            }
            break;
            
        case GaitPhase::TOE_OFF:
            // Brief transition phase
            state_.phase = GaitPhase::SWING;
            swing_start_time_ms_ = GET_TIME_MS();
            state_.samples_in_phase = 0;
            break;
            
        case GaitPhase::SWING:
            if (in_stance && state_.samples_in_phase > 10) {
                // Transition to heel strike
                state_.phase = GaitPhase::HEEL_STRIKE;
                onHeelStrike();
            }
            break;
            
        case GaitPhase::HEEL_STRIKE:
            // Brief transition phase
            state_.phase = GaitPhase::STANCE;
            stance_start_time_ms_ = GET_TIME_MS();
            state_.samples_in_phase = 0;
            break;
    }
    
    // Track stance/swing for step detection
    was_in_stance_ = in_stance;
}

void GaitFilter::onHeelStrike() {
    // Generate step summary for the completed step
    generateStepSummary();
    step_complete_ = true;
    
    // Update walking direction based on config
    // Option 1 (use_toe_off_heading = true): Direction was already set at toe-off from IMU heading
    // Option 2 (use_toe_off_heading = false): Compute from step displacement (1-step lag)
    if (!config_.use_toe_off_heading) {
        // Compute walking direction from horizontal step displacement
        // This gives stable sagittal plane alignment for the NEXT step
        float dx = state_.position.x();
        float dy = state_.position.y();
        float horiz_dist = std::sqrt(dx * dx + dy * dy);
        
        // Only update walking direction if we have meaningful displacement
        // (avoids division by zero and noise from small movements)
        constexpr float MIN_STEP_DISTANCE = 0.05f;  // 5cm minimum
        if (horiz_dist > MIN_STEP_DISTANCE) {
            // Normalize to get unit direction vector (cos, sin of walking angle)
            walking_dir_cos_ = dx / horiz_dist;
            walking_dir_sin_ = dy / horiz_dist;
            walking_dir_valid_ = true;
        }
        // If displacement is too small, keep previous walking direction
    }
    
    // Increment step counter
    state_.step_count++;
    
    // Reset step-relative position
    step_start_pos_ = state_.position;
    step_start_time_ms_ = GET_TIME_MS();
    
    // Reset position to zero (step-relative)
    // state_.position = Vec3f::Zero();
    
    // Reset step metrics
    max_height_this_step_ = 0;
    min_height_this_step_ = 0;
    peak_vertical_vel_this_step_ = 0;
    
    // Reset position covariance (we're confident about position at heel strike)
    state_.P.block<3,3>(0,0) = Mat3f::Identity() * config_.init_pos_cov;
}

void GaitFilter::onToeOff() {
    // Mark start of swing phase
    swing_start_time_ms_ = GET_TIME_MS();
    // Note: heading capture is now done on SHOE foot-flat falling edge
    // in update(), not here, so heading works regardless of which stance
    // detector drives the gait phase machine.
}

void GaitFilter::captureHeadingAtHeelOff() {
    // Capture walking direction from IMU orientation at heel off
    // (SHOE foot-flat falling edge).  This is independent of the
    // gait-phase state machine (omega-sag / ADC) so heading stays
    // correct regardless of which detector drives stance.
    if (!config_.use_toe_off_heading) return;

    // For shank-mounted IMU, forward = body -Z axis
    // Body -Z in world frame = negated 3rd column of rotation matrix:
    //   dx = -2(xz + wy)
    //   dy = 2(wx - yz)
    float w = state_.orientation.w();
    float x = state_.orientation.x();
    float y = state_.orientation.y();
    float z = state_.orientation.z();

    float dx = -2.0f * (x*z + w*y);  // East component of forward
    float dy =  2.0f * (w*x - y*z);  // North component of forward
    float horiz_dist = std::sqrt(dx*dx + dy*dy);

    if (horiz_dist > 0.01f) {
        walking_dir_cos_ = dx / horiz_dist;
        walking_dir_sin_ = dy / horiz_dist;
        walking_dir_valid_ = true;
    }
    toe_off_heading_ = std::atan2(walking_dir_sin_, walking_dir_cos_);
}

// ============================================================================
// Step Metrics
// ============================================================================

void GaitFilter::updateStepMetrics() {
    // Track maximum height during step
    if (state_.position.z() > max_height_this_step_) {
        max_height_this_step_ = state_.position.z();
    }
    if (state_.position.z() < min_height_this_step_) {
        min_height_this_step_ = state_.position.z();
    }
    
    // Track peak vertical velocity
    float vz = std::abs(state_.velocity.z());
    if (vz > peak_vertical_vel_this_step_) {
        peak_vertical_vel_this_step_ = vz;
    }
}

void GaitFilter::generateStepSummary() {
    uint32_t now_ms = GET_TIME_MS();
    
    step_summary_.step_number = state_.step_count;
    
    // Displacement metrics
    step_summary_.total_height_change = state_.position.z();
    step_summary_.max_height = max_height_this_step_;
    step_summary_.min_height = min_height_this_step_;
    step_summary_.horizontal_distance = std::sqrt(
        state_.position.x() * state_.position.x() + 
        state_.position.y() * state_.position.y()
    );
    
    // Timing
    step_summary_.swing_time_ms = (float)(stance_start_time_ms_ - swing_start_time_ms_);
    step_summary_.stance_time_ms = (float)(now_ms - stance_start_time_ms_);
    step_summary_.total_time_ms = (float)(now_ms - step_start_time_ms_);
    
    // Velocity metrics
    step_summary_.peak_vertical_velocity = peak_vertical_vel_this_step_;
    if (step_summary_.total_time_ms > 0) {
        step_summary_.avg_horizontal_velocity = step_summary_.horizontal_distance / 
                                                (step_summary_.total_time_ms * 0.001f);
    }
    
    // Terrain indicator
    if (step_summary_.horizontal_distance > 0.01f) {
        step_summary_.height_to_distance_ratio = std::abs(step_summary_.total_height_change) / 
                                                  step_summary_.horizontal_distance;
    } else {
        step_summary_.height_to_distance_ratio = 0;
    }
}

// ============================================================================
// Output Generation
// ============================================================================

void GaitFilter::generateOutput(uint32_t timestamp_ms) {
    output_.timestamp_ms = timestamp_ms;
    
    // Position (step-relative, ENU frame)
    output_.pos_x = state_.position.x();
    output_.pos_y = state_.position.y();
    output_.pos_z = state_.position.z();
    
    // Velocity (ENU frame)
    output_.vel_x = state_.velocity.x();
    output_.vel_y = state_.velocity.y();
    output_.vel_z = state_.velocity.z();
    
    // Compute sagittal-plane-aligned (walking direction relative) position and velocity
    // Uses walking direction computed at heel strike from step displacement
    // This provides stable forward/lateral alignment throughout the gait cycle
    
    // Store the walking direction angle for output (computed from stored cos/sin)
    output_.heading = std::atan2(walking_dir_sin_, walking_dir_cos_);
    
    // Transform position to walking-direction-aligned frame (sagittal plane)
    // Forward = direction of walking, Lateral = perpendicular (positive = left)
    output_.pos_fwd = walking_dir_cos_ * state_.position.x() + walking_dir_sin_ * state_.position.y();
    output_.pos_lat = -walking_dir_sin_ * state_.position.x() + walking_dir_cos_ * state_.position.y();
    output_.pos_vertical = state_.position.z();
    
    // Transform velocity to walking-direction-aligned frame
    output_.vel_fwd = walking_dir_cos_ * state_.velocity.x() + walking_dir_sin_ * state_.velocity.y();
    output_.vel_lat = -walking_dir_sin_ * state_.velocity.x() + walking_dir_cos_ * state_.velocity.y();
    output_.vel_vertical = state_.velocity.z();
    
    // Transform acceleration to walking-direction-aligned frame (same rotation as velocity)
    output_.a_fwd = walking_dir_cos_ * current_free_accel_.x() + walking_dir_sin_ * current_free_accel_.y();
    output_.a_lat = -walking_dir_sin_ * current_free_accel_.x() + walking_dir_cos_ * current_free_accel_.y();
    
    // Sagittal plane angular rate
    // This is the rotation about the medial-lateral axis (perpendicular to walking direction)
    // First transform gyro from body to world frame: ω_world = R * ω_body
    Mat3f R = state_.orientation.toRotationMatrix();
    Vec3f gyro_world = R * current_gyro_;
    
    // The medial-lateral axis in world frame is perpendicular to walking direction
    // Walking direction is (walking_dir_cos_, walking_dir_sin_, 0) in ENU
    // Lateral axis is (-walking_dir_sin_, walking_dir_cos_, 0)
    // Sagittal angular rate = projection of gyro_world onto lateral axis
    output_.w_sag = -walking_dir_sin_ * gyro_world.x() + walking_dir_cos_ * gyro_world.y();
    
    // Step metrics
    output_.step_height = max_height_this_step_ - min_height_this_step_;
    output_.step_length = std::sqrt(
        state_.position.x() * state_.position.x() + 
        state_.position.y() * state_.position.y()
    );
    output_.step_duration_ms = (float)(timestamp_ms - step_start_time_ms_);
    
    // Orientation - use temporary variables to avoid packed struct issues
    float roll_tmp, pitch_tmp, yaw_tmp;
    quatToEuler(state_.orientation, roll_tmp, pitch_tmp, yaw_tmp);
    output_.roll = roll_tmp;
    output_.pitch = pitch_tmp;
    output_.yaw = yaw_tmp;
    
    // Phase
    output_.phase = static_cast<uint8_t>(state_.phase);
    output_.step_count_low = static_cast<uint8_t>(state_.step_count & 0xFF);
    
    // Uncertainty
    output_.position_uncertainty = state_.P(0,0) + state_.P(1,1) + state_.P(2,2);
    
    // ZUPT flag
    output_.zupt_active = zupt_active_ ? 1 : 0;
    
    // Foot-flat: debounced SHOE stance (independent of gait-phase detection)
    output_.foot_flat = state_.shoe_stance_active ? 1 : 0;
    
    // SHOE statistic for debugging (stance_active is set in updateGaitPhase)
    output_.shoe_statistic = state_.last_shoe_statistic;
}

void GaitFilter::quatToEuler(const Quatf& q, float& roll, float& pitch, float& yaw) {
    // ZYX Euler angles (yaw-pitch-roll)
    float w = q.w(), x = q.x(), y = q.y(), z = q.z();
    
    // Roll (x-axis rotation)
    float sinr_cosp = 2.0f * (w * x + y * z);
    float cosr_cosp = 1.0f - 2.0f * (x * x + y * y);
    roll = std::atan2(sinr_cosp, cosr_cosp);
    
    // Pitch (y-axis rotation)
    float sinp = 2.0f * (w * y - z * x);
    if (std::abs(sinp) >= 1.0f) {
        pitch = std::copysign(M_PI / 2.0f, sinp);
    } else {
        pitch = std::asin(sinp);
    }
    
    // Yaw (z-axis rotation)
    float siny_cosp = 2.0f * (w * z + x * y);
    float cosy_cosp = 1.0f - 2.0f * (y * y + z * z);
    yaw = std::atan2(siny_cosp, cosy_cosp);
}

float GaitFilter::quatToHeading(const Quatf& q) {
    // Efficient heading (yaw) extraction from quaternion
    // Only computes the yaw component of ZYX Euler angles
    float w = q.w(), x = q.x(), y = q.y(), z = q.z();
    float siny_cosp = 2.0f * (w * z + x * y);
    float cosy_cosp = 1.0f - 2.0f * (y * y + z * z);
    return std::atan2(siny_cosp, cosy_cosp);
}

Vec3f GaitFilter::enuToWalkingFrame(const Vec3f& enu_vec, float cos_dir, float sin_dir) {
    // Transform from ENU (East-North-Up) to walking-direction-aligned frame
    // (Forward-Lateral-Vertical, i.e., sagittal plane aligned)
    // 
    // In ENU frame:
    //   X = East, Y = North, Z = Up
    // 
    // In walking-direction frame:
    //   Forward = direction of walking (computed from step displacement)
    //   Lateral = perpendicular to forward (positive = left)
    //   Vertical = Up (same as ENU Z)
    //
    // The transformation is a 2D rotation:
    //   forward = cos(walking_dir) * east + sin(walking_dir) * north
    //   lateral = -sin(walking_dir) * east + cos(walking_dir) * north
    //   vertical = up
    
    Vec3f walking_frame;
    walking_frame.x() = cos_dir * enu_vec.x() + sin_dir * enu_vec.y();   // Forward
    walking_frame.y() = -sin_dir * enu_vec.x() + cos_dir * enu_vec.y();  // Lateral
    walking_frame.z() = enu_vec.z();                                      // Vertical
    
    return walking_frame;
}

float GaitFilter::getHeading() const {
    // Return the walking direction (from step displacement), not IMU yaw
    return std::atan2(walking_dir_sin_, walking_dir_cos_);
}

float GaitFilter::getWalkingDirection() const {
    return std::atan2(walking_dir_sin_, walking_dir_cos_);
}

Vec3f GaitFilter::getPositionHeadingRelative() const {
    return enuToWalkingFrame(state_.position, walking_dir_cos_, walking_dir_sin_);
}

Vec3f GaitFilter::getVelocityHeadingRelative() const {
    return enuToWalkingFrame(state_.velocity, walking_dir_cos_, walking_dir_sin_);
}

} // namespace gait
