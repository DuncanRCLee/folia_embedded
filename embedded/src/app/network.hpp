#pragma once
#include "Arduino.h"
#include <WiFi.h>

namespace network {

    // ========================================================================
    // WiFi Mode Selection
    // ========================================================================
    // WiFi can operate in two modes:
    //   - AP: Device creates its own WiFi network (default)
    //   - CLIENT: Device connects to an existing WiFi network
    // ========================================================================
    enum class WiFiMode {
        AP,      // Access Point mode - device creates network
        CLIENT   // Client mode - device joins existing network
    };

    // Define command communication states
    enum class CommState {
        STANDBY,
        LOGGING
    };

    // Define the size of the receive buffer for commands
    constexpr uint8_t RXBUF_SIZE = 64;

    // Extern declarations for network components (used by main.cpp mode selection)
    extern WiFiClient tcpClient;
    extern WiFiMode currentWifiMode;

    // WiFi mode selection (interactive, with timeout in milliseconds)
    WiFiMode selectWifiMode(uint32_t timeoutMs = 5000);

    // WiFi configuration functions
    void configureWifiAP();     // Set up as Access Point
    void configureWifiServe();  // Set up as Client (connect to existing network)

    void logging_task(void* arg);
    void console_task(void* arg);
    void monitor_task(void* arg);

    // Print helpers
    void print(const char* s);
    void println(const char* s);

}
