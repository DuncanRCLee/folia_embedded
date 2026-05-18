#include "util.hpp"
#include <mti.hpp>
#include <hal.hpp>
#include <cmath>
#include <Arduino.h>

namespace util {

float computeShankPitch(const float* quat, float standingOffset) {
    float qw = quat[0];
    float qx = quat[1];
    float qy = quat[2];
    float qz = quat[3];
    
    // Extract pitch angle (rotation about X-axis) from quaternion
    // Standard roll formula for ZYX Euler convention
    float pitch = atan2f(2.0f * (qw * qx + qy * qz), 1.0f - 2.0f * (qx * qx + qy * qy));
    
    // Subtract standing offset to zero at standing posture
    return pitch - standingOffset;
}

float calibrateStandingPitch(xsens::MTi* mti, hal::Event drdy_event, int numSamples) {
    Serial.println(F("Calibrating shank pitch offset (please stand still)..."));
    hal::sleepMs(2000);  // Wait for IMU to stabilize
    
    // Average pitch over samples
    float pitchSum = 0.0f;
    int sampleCount = 0;
    
    for (int i = 0; i < numSamples; i++) {
        hal::eventWait(drdy_event, 1000);
        mti->readMessages();
        float* quat = mti->getQuaternion();
        
        float pitch = atan2f(2.0f * (quat[0] * quat[1] + quat[2] * quat[3]), 
                             1.0f - 2.0f * (quat[1] * quat[1] + quat[2] * quat[2]));
        pitchSum += pitch;
        sampleCount++;
        hal::sleepMs(10);
    }
    
    float offset = pitchSum / sampleCount;
    Serial.print(F("Standing pitch offset calibrated: "));
    Serial.print(offset * 180.0f / PI);
    Serial.println(F(" degrees"));
    
    return offset;
}

} // namespace util
