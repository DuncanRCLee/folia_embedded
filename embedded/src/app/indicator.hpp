#pragma once

#include <atomic>
#include "Arduino.h"

namespace indicator {

    // System status patterns (RGB LED)
    enum class LedPattern {
        STARTUP,            // Blue fade in/out at 1Hz - during initialization
        IDLE,               // Yellow fade in/out at 1Hz - ready, no TCP client
        CLIENT_CONNECTED,   // Green fade in/out at 1Hz - TCP client connected
        
        // Specific failure modes (Purple with different patterns)
        IMU_FAIL,           // Purple 1 blink - IMU initialization failed
        ADC_FAIL,           // Purple 2 blinks - ADC initialization failed  
        MOTOR_FAIL,         // Purple 3 blinks - Motor initialization failed
        CLASSIFIER_FAIL,    // Purple 4 blinks - Classifier initialization failed
        WIFI_FAIL,          // Purple SOS pattern - WiFi/network failed
    };

    extern std::atomic<LedPattern> currentPattern;
    extern std::atomic<bool> motorMoving;

    // RGB LED status pattern control
    inline void setStartup() { currentPattern.store(LedPattern::STARTUP); }
    inline void setIdle() { currentPattern.store(LedPattern::IDLE); }
    inline void setClientConnected() { currentPattern.store(LedPattern::CLIENT_CONNECTED); }
    inline void setImuFail() { currentPattern.store(LedPattern::IMU_FAIL); }
    inline void setAdcFail() { currentPattern.store(LedPattern::ADC_FAIL); }
    inline void setMotorFail() { currentPattern.store(LedPattern::MOTOR_FAIL); }
    inline void setClassifierFail() { currentPattern.store(LedPattern::CLASSIFIER_FAIL); }
    inline void setWifiFail() { currentPattern.store(LedPattern::WIFI_FAIL); }

    // LED1 (HD59) motor movement control
    inline void setMotorMoving(bool moving) { motorMoving.store(moving); }

    // arg should be std::atomic<bool>* to logging_active
    void indicator_task(void *arg);
}
