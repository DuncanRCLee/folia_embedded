/**
 * @file gait_filter_example.cpp
 * @brief Complete example of integrating GaitFilter into main.cpp
 * 
 * This file shows the exact code changes needed to add gait filtering
 * to your existing main loop. The filter runs at 100Hz (same as IMU rate)
 * and outputs position/velocity every sample for NN classifier input.
 * 
 * The downstream classifier handles decimation to 20Hz prediction rate.
 */

// ============================================================================
// STEP 1: Add includes at top of main.cpp
// ============================================================================

#include <gait_filter.hpp>
// OR for the convenience functions:
// #include <gait_filter_integration.hpp>

// ============================================================================
// STEP 2: Add global filter instance (after other globals)
// ============================================================================

gait::GaitFilter* gaitFilter = nullptr;

// ============================================================================
// STEP 3: Initialize in setup() after MTI configuration
// ============================================================================

void initGaitFilterExample() {
    // Configure filter for terrain classification
    gait::FilterConfig config;
    
    // Process noise (tuned for MTI-3)
    config.sigma_accel_noise = 0.08f;
    config.sigma_gyro_noise = 0.008f;
    
    // ZUPT detection thresholds (tuned for walking)
    config.zupt_accel_threshold = 0.4f;  // Accel variance threshold
    config.zupt_gyro_threshold = 0.25f;  // rad/s
    config.zupt_window_size = 5;         // 50ms at 100 Hz
    
    // ZUPT correction strength
    config.zupt_velocity_noise = 0.005f; // Tight correction
    
    // Timing - filter outputs at 100Hz (same as IMU rate)
    config.dt_imu = 0.01f;   // 100 Hz IMU
    // Downstream classifier handles decimation to 20Hz predictions
    
    // Output frame selection (both are always computed)
    // ENU: Standard East-North-Up world frame
    // HEADING_RELATIVE: Forward/Lateral/Vertical relative to heading
    config.output_frame = gait::OutputFrame::HEADING_RELATIVE;  // Optional
    
    gaitFilter = new gait::GaitFilter(config);
    Serial.println(F("Gait filter initialized (100Hz output)"));
}

// ============================================================================
// STEP 4: In loop(), after reading MTI-3 data, process through filter
// ============================================================================

void loopWithGaitFilter() {
    // ... existing code to wait for IMU data ...
    // hal::eventWait(drdy_event, 0xFFFFFFFF);
    // mti->readMessages();
    
    // Get the latest data from the MTi
    float* acc = mti->getAcceleration();
    float* gyr = mti->getRateOfTurn();
    float* quat = mti->getQuaternion();
    float* dv = mti->getDeltaV();
    float* dq = mti->getDeltaQ();
    
    // ========== ADD THIS SECTION ==========
    // Process through gait filter (runs at 100Hz, same as IMU)
    if (gaitFilter != nullptr) {
        // Create measurement structure
        gait::IMUMeasurement meas;
        
        // Pre-integrated outputs (preferred)
        meas.dv = gait::Vec3f(dv[0], dv[1], dv[2]);
        meas.dq = gait::Quatf(dq[0], dq[1], dq[2], dq[3]);  // [w,x,y,z]
        meas.use_preintegrated = true;
        
        // Orientation from MTI-3 (always available)
        meas.orientation = gait::Quatf(quat[0], quat[1], quat[2], quat[3]);
        
        // Raw sensors (for ZUPT detection)
        meas.accel = gait::Vec3f(acc[0], acc[1], acc[2]);
        meas.gyro = gait::Vec3f(gyr[0], gyr[1], gyr[2]);
        
        // Timestamp
        meas.sample_time_fine = mti->getSampleTimeFine();
        
        // Update filter (100 Hz) - produces output every call
        gaitFilter->update(meas);
        
        // Output available every sample (100 Hz) - feed to NN classifier buffer
        if (gaitFilter->hasNewOutput()) {
            gait::GaitFilterOutput out = gaitFilter->getOutput();
            
            // ============================================================
            // OPTION A: Use ENU frame (standard world coordinates)
            // ============================================================
            // out.pos_x, out.pos_y, out.pos_z  - Position (East, North, Up)
            // out.vel_x, out.vel_y, out.vel_z  - Velocity (East, North, Up)
            //
            // Use this when you need absolute direction information or
            // when combining with GPS/compass.
            
            // ============================================================
            // OPTION B: Use heading-relative frame (body-aligned)
            // ============================================================
            // out.pos_forward, out.pos_lateral, out.pos_vertical
            // out.vel_forward, out.vel_lateral, out.vel_vertical
            // out.heading  - Current heading used for transformation
            //
            // Use this when direction of travel doesn't matter, e.g.,
            // terrain classification where "forward" and "sideways" movement
            // patterns are important regardless of compass heading.
            
            // Both sets are always computed - choose based on your application
            
            // Feed to NN classifier input buffer (classifier handles 20Hz decimation)
            // Example with heading-relative frame:
            // classifierBuffer.add(out.pos_forward, out.pos_lateral, out.pos_vertical,
            //                      out.vel_forward, out.vel_lateral, out.vel_vertical, 
            //                      out.pitch, out.roll);
            
            // Or with ENU frame:
            // classifierBuffer.add(out.pos_x, out.pos_y, out.pos_z,
            //                      out.vel_x, out.vel_y, out.vel_z, 
            //                      out.pitch, out.roll);
            
            // Option: Stream via protobuf (100Hz position data)
            // pkt.payload.imu_classifier.pos_z = out.pos_z;
            // pkt.payload.imu_classifier.vel_z = out.vel_z;
            // pkt.payload.imu_classifier.pos_forward = out.pos_forward;
            // pkt.payload.imu_classifier.vel_forward = out.vel_forward;
            
            // Debug print (don't use in production - too fast!)
            // Serial.printf("fwd=%.3f lat=%.3f z=%.3f heading=%.1f°\n", 
            //               out.pos_forward, out.pos_lateral, out.pos_vertical,
            //               out.heading * 180.0f / M_PI);
        }
        
        // Check for completed step (terrain classification opportunity!)
        if (gaitFilter->hasNewStep()) {
            gait::StepSummary step = gaitFilter->getStepSummary();
            
            // Terrain classification based on step metrics
            if (std::abs(step.total_height_change) > 0.12f) {
                // Likely STAIRS
                Serial.printf("STAIRS detected: height=%.3fm\n", step.total_height_change);
            } else if (step.height_to_distance_ratio > 0.08f) {
                // Likely SLOPE
                Serial.printf("SLOPE detected: ratio=%.3f\n", step.height_to_distance_ratio);
            } else {
                // Likely LEVEL GROUND
                Serial.println("Level ground");
            }
        }
    }
    // ========== END ADDED SECTION ==========
    
    // ... rest of existing loop code (protobuf encoding, etc.) ...
}

// ============================================================================
// STEP 5: Optional - Add filter output to protobuf packet
// ============================================================================

// If you want to stream filter outputs, add these fields to your .proto file:
// 
// message IMUFrameClassifier {
//   ... existing fields ...
//   
//   // Gait filter outputs
//   float gait_pos_z = 20;         // Vertical position (step-relative)
//   float gait_vel_z = 21;         // Vertical velocity
//   float gait_step_height = 22;   // Step clearance
//   uint32 gait_phase = 23;        // 1=stance, 2=swing
//   uint32 step_count = 24;
// }

// ============================================================================
// STEP 6: Tuning Notes
// ============================================================================

/*
 * ZUPT THRESHOLD TUNING:
 * - If stance is detected too often (during swing): INCREASE zupt_accel_threshold
 * - If stance is not detected (no ZUPT): DECREASE zupt_accel_threshold
 * - Typical range: 0.2 - 1.0 for walking
 * 
 * VERTICAL DAMPING:
 * - If vertical position drifts during swing: INCREASE vertical_damping
 * - If vertical motion is sluggish: DECREASE vertical_damping
 * - Typical range: 0.01 - 0.05
 * 
 * OUTPUT RATE:
 * - Classifier needs features at consistent rate
 * - 20 Hz gives 5 samples per ~250ms step
 * - Adjust dt_output if classifier expects different rate
 * 
 * DEBUGGING:
 * - Check gaitFilter->isZuptActive() to see when corrections happen
 * - Check gaitFilter->getPhase() for current gait phase
 * - Monitor position_uncertainty for filter health
 */
