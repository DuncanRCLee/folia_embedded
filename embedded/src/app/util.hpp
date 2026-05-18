#pragma once

#include "Wire.h"
#include <hal.hpp>

// Forward declaration
namespace xsens { class MTi; }

// static bool i2cPing(uint8_t addr) {
//   Serial.print(F("Pinging 0x")); Serial.print(addr, HEX); Serial.print(F("... "));
//   Wire.beginTransmission(addr);
//   uint8_t err = Wire.endTransmission(true);
//   Serial.println(err == 0 ? F("ACK") : F("NO-ACK"));
//   return err == 0;
// }

namespace util {

/**
 * @brief Compute shank pitch (sagittal plane rotation) from quaternion
 * 
 * Shank pitch is defined as rotation in the sagittal plane about the medial-lateral axis
 * (IMU X-axis pointing right when mounted on posterior shank with Y-axis vertical).
 * 
 * Sign convention:
 * - 0° = standing posture (calibrated at startup)
 * - Positive = counterclockwise rotation (extension/backward tilt when viewed from right)
 * - Negative = clockwise rotation (flexion/forward tilt when viewed from right)
 * 
 * @param quat Quaternion array [w, x, y, z]
 * @param standingOffset Reference pitch angle during standing (radians), subtracted from result
 * @return Shank pitch angle in radians relative to standing posture
 */
float computeShankPitch(const float* quat, float standingOffset = 0.0f);

/**
 * @brief Calibrate the standing pitch offset for shank pitch measurements
 * 
 * Averages pitch measurements over multiple samples while the user stands still.
 * This establishes the zero reference for shank pitch.
 * 
 * @param mti Pointer to MTi IMU object
 * @param drdy_event Event handle for IMU data ready signal
 * @param numSamples Number of samples to average (default: 100)
 * @return Calibrated standing pitch offset in radians
 */
float calibrateStandingPitch(xsens::MTi* mti, hal::Event drdy_event, int numSamples = 100);

} // namespace util
