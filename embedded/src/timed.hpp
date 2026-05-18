#pragma once
#include "MainPrelude.h"
#include "Arduino.h"
#include <cstdint>

template<uint8_t WHO, typename F> inline void timing(F func) {
    #if DEBUG_MODE
        unsigned long startOuter = micros();
        func();
        unsigned long outerElapsed = micros() - startOuter;

        Serial.print("Timed ");
        Serial.print(WHO);
        Serial.print(" at: ");
        Serial.print(outerElapsed);
        Serial.print(" us (");
        Serial.print(outerElapsed / 1000);
        Serial.println(" ms)");
    #else
        func();
    #endif


}

template<uint8_t WHO, typename F, long INTERVAL_SIZE_US> inline void timed(F func, long now_us) {

    #if DEBUG_MODE
        static long last_us;

        if (now_us - last_us >= INTERVAL_SIZE_US) {
            last_us += INTERVAL_SIZE_US;

            // snapshot start of work
            unsigned long startOuter = micros();
            func();
            unsigned long outerElapsed = micros() - startOuter;

            if (outerElapsed > INTERVAL_SIZE_US) {
            Serial.print("WARNING: ");
            Serial.print(WHO);
            Serial.print(" overrun! took ");
            Serial.print(outerElapsed);
            Serial.print(" us (");
            Serial.print(outerElapsed / 1000);
            Serial.println(" ms)");
            }
        }
    #else
        func();
    #endif

}
