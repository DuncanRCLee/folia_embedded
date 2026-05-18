#include "indicator.hpp"
#include "app/main_prelude.hpp"
#include <hal.hpp>

namespace indicator {

std::atomic<LedPattern> currentPattern{LedPattern::STARTUP};
std::atomic<bool> motorMoving{false};

// Blink timing constants (simple on/off, no PWM to avoid hard faults on Portenta RGB LEDs)
static constexpr uint32_t BLINK_ON_MS = 500;
static constexpr uint32_t BLINK_OFF_MS = 500;

// Use digitalWrite instead of analogWrite - Portenta RGB LEDs can cause hard faults with PWM
static inline void rgbOff() {
    digitalWrite(LEDR, HIGH);  // Active low: HIGH = off
    digitalWrite(LEDG, HIGH);
    digitalWrite(LEDB, HIGH);
}

static inline void rgbBlue() {
    digitalWrite(LEDR, HIGH);
    digitalWrite(LEDG, HIGH);
    digitalWrite(LEDB, LOW);   // Active low: LOW = on
}

static inline void rgbYellow() {
    digitalWrite(LEDR, LOW);
    digitalWrite(LEDG, LOW);
    digitalWrite(LEDB, HIGH);
}

static inline void rgbGreen() {
    digitalWrite(LEDR, HIGH);
    digitalWrite(LEDG, LOW);
    digitalWrite(LEDB, HIGH);
}

static inline void rgbPurple() {
    digitalWrite(LEDR, LOW);
    digitalWrite(LEDG, HIGH);
    digitalWrite(LEDB, LOW);
}

// Simple blink helper - blinks a color on/off at ~1Hz
// Returns early if pattern changes
static void blinkColor(char color, std::atomic<LedPattern>& pattern, LedPattern expectedPattern) {
    // Turn on
    if (pattern.load() != expectedPattern) return;
    switch (color) {
        case 'B': rgbBlue(); break;
        case 'Y': rgbYellow(); break;
        case 'G': rgbGreen(); break;
    }
    hal::sleepMs(BLINK_ON_MS);
    
    // Turn off
    if (pattern.load() != expectedPattern) return;
    rgbOff();
    hal::sleepMs(BLINK_OFF_MS);
}

// SOS pattern helper: ... --- ... (3 short, 3 long, 3 short)
static void sosPurple() {
    // S: 3 short blinks (dot = 200ms on, 200ms off)
    for (int i = 0; i < 3; ++i) {
        rgbPurple();
        hal::sleepMs(200);
        rgbOff();
        hal::sleepMs(200);
    }
    hal::sleepMs(400);  // Gap between letters
    
    // O: 3 long blinks (dash = 600ms on, 200ms off)
    for (int i = 0; i < 3; ++i) {
        rgbPurple();
        hal::sleepMs(600);
        rgbOff();
        hal::sleepMs(200);
    }
    hal::sleepMs(400);  // Gap between letters
    
    // S: 3 short blinks
    for (int i = 0; i < 3; ++i) {
        rgbPurple();
        hal::sleepMs(200);
        rgbOff();
        hal::sleepMs(200);
    }
    hal::sleepMs(1000);  // Gap before repeat
}

void indicator_task(void *arg) {
    Serial.println("LED indicator task started");

    std::atomic<bool>* loggingActive = static_cast<std::atomic<bool>*>(arg);

    pinMode(LEDR, OUTPUT);
    pinMode(LEDG, OUTPUT);
    pinMode(LEDB, OUTPUT);
    rgbOff();

    while (true) {
        digitalWrite(LED_HD59, motorMoving.load() ? HIGH : LOW);
        digitalWrite(LED_HD61, loggingActive->load() ? HIGH : LOW);

        // Update RGB LED based on current pattern
        switch (currentPattern.load()) {
            case LedPattern::STARTUP:
                // Blue blink at 1Hz
                blinkColor('B', currentPattern, LedPattern::STARTUP);
                break;

            case LedPattern::IDLE:
                // Yellow blink at 1Hz - ready but no TCP client
                blinkColor('Y', currentPattern, LedPattern::IDLE);
                break;

            case LedPattern::CLIENT_CONNECTED:
                // Green blink at 1Hz - TCP client connected
                blinkColor('G', currentPattern, LedPattern::CLIENT_CONNECTED);
                break;

            case LedPattern::IMU_FAIL:
                // Purple 1 blink pattern - IMU failed
                rgbPurple();
                hal::sleepMs(200);
                rgbOff();
                hal::sleepMs(1800);
                break;

            case LedPattern::ADC_FAIL:
                // Purple 2 blink pattern - ADC failed
                for (int i = 0; i < 2; ++i) {
                    rgbPurple();
                    hal::sleepMs(200);
                    rgbOff();
                    hal::sleepMs(200);
                }
                hal::sleepMs(1400);
                break;

            case LedPattern::MOTOR_FAIL:
                // Purple 3 blink pattern - Motor failed
                for (int i = 0; i < 3; ++i) {
                    rgbPurple();
                    hal::sleepMs(200);
                    rgbOff();
                    hal::sleepMs(200);
                }
                hal::sleepMs(1000);
                break;

            case LedPattern::CLASSIFIER_FAIL:
                // Purple 4 blink pattern - Classifier failed
                for (int i = 0; i < 4; ++i) {
                    rgbPurple();
                    hal::sleepMs(200);
                    rgbOff();
                    hal::sleepMs(200);
                }
                hal::sleepMs(600);
                break;

            case LedPattern::WIFI_FAIL:
                // Purple SOS pattern - WiFi failed
                sosPurple();
                break;
        }
    }
}

}
