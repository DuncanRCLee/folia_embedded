#pragma once

#ifdef ARDUINO
#include <Arduino.h>
#endif
#include <cstdint>
#include <vector>
#include "xbus.hpp"

namespace xsens {
    struct OutputConfigEntry {
      uint16_t id;      // Output ID
      uint16_t rateHz;  // Update rate in Hz
    };

    class MTi {
    public:
        MTi(const Xbus& xbus) : xbus(xbus) {

        }

        bool detect(uint16_t timeout);
        void requestDeviceInfo();
        void getFilterProfile();
        void setFilterProfile(uint8_t profileId);
        void getBaudRate();
        void setBaudRate(uint32_t baudrate);
        void configureOutputs();     // Default configuration (100 Hz for common outputs)
        // Custom configuration: specify XDI outputs and rate (Hz)
        void configureOutputs(const std::vector<uint16_t>& outputs);
        void configureOutputsAdv(const std::vector<OutputConfigEntry>& outputs);
        void reqAlignmentRotation(uint8_t parameter);
        void setAlignmentRotation(uint8_t parameter, float q0, float q1, float q2, float q3);
        // Check if current alignment matches target quaternion (with tolerance)
        bool alignmentMatches(uint8_t param, float q0, float q1, float q2, float q3, float tolerance = 0.001f);
        float* getAlignmentQuaternion() { return xbus.alignQuat; }
        bool isAlignmentQuaternionValid() { return xbus.alignQuatValid; }
        // Orientation reset commands (MID 0xA4)
        void resetOrientation(uint8_t resetCode);
        void resetHeading();       // Reset heading only (code 0x01)
        void resetInclination();   // Reset inclination only (code 0x02) 
        void resetAlignment();     // Reset alignment rotation to identity (code 0x03)
        void defaultHeading();    // Reset heading to default (code 0x05)
        void defaultInclination(); // Reset inclination to default (code 0x06)
        void defaultAlignment();   // Reset alignment to default (code 0x07)
    
        // Factory defaults (MID 0x0E)
        void restoreFactoryDefaults(); // Restore all settings to factory defaults
        
        // Reset device (MID 0x40) - reactivates WakeUp procedure
        void reset();

        void goToConfig();
        void goToMeasurement();

        void readMessages() {
            xbus.readMessages();
        }


        // Accessors
        float*  getHRAcceleration() {return xbus.HR_acc; }
        float*  getHRGyro() {return xbus.HR_gyr; }

        float*  getAcceleration()       { return xbus.acc; }
        float*  getFreeAcceleration()   { return xbus.freeAcc; }
        float*  getRateOfTurn()         { return xbus.gyr; }
        float*  getQuaternion()         { return xbus.quat; }
        float*  getRotation()           { return xbus.rot; }
        float*  getEulerAngles()        { return xbus.euler; }
        float*  getMagnetometer()       { return xbus.mag; }
        float*  getLatLon()             { return xbus.latlon; }
        float*  getDeltaQ()             { return xbus.dq; }
        float*  getDeltaV()             { return xbus.dv; }
        uint32_t getSampleTimeFine()    { return xbus.sampleTimeFine; }

        Xbus xbus;

    private:

        bool deviceInConfigMode() {
            return xbus.configState;
        }
    };

    // XDI message identifiers for configureOutputs()
    static constexpr uint16_t XDI_UTCTime           = 0x1010;
    static constexpr uint16_t XDI_SampleTimeFine    = 0x1060;
    static constexpr uint16_t XDI_Acceleration      = 0x4020;
    static constexpr uint16_t XDI_HR_Acceleration   = 0x4040;
    static constexpr uint16_t XDI_HR_Gyro           = 0x8040;
    static constexpr uint16_t XDI_FreeAcceleration  = 0x4030;
    static constexpr uint16_t XDI_RateOfTurn        = 0x8020;
    static constexpr uint16_t XDI_Quaternion        = 0x2010;
    static constexpr uint16_t XDI_Rotation          = 0x2020;
    static constexpr uint16_t XDI_EulerAngles       = 0x2030;
    static constexpr uint16_t XDI_Magnetometer      = 0xC020;
    static constexpr uint16_t XDI_LatLon            = 0x5040;
    static constexpr uint16_t XDI_DeltaQ            = 0x8030;
    static constexpr uint16_t XDI_DeltaV            = 0x4010;
    static constexpr uint16_t XDI_Velocity          = 0xD010;
}
