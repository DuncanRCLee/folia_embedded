#include "Arduino.h"
#include <hal.hpp>
#include "imu_packet.hpp"

#include "gen/Packet.pb.h"
#include "pb_encode.h"
#include "WiFi.h"
#include "arduino_secrets.h"
#include "main_prelude.hpp"
#include "network.hpp"
#include <ctype.h>
#include <atomic>

#include "vertiq_motor.hpp"
#include <ads1220.hpp>
#include "indicator.hpp"
#include "ei_classifier.hpp"
#include "control.hpp"


namespace network {

// WiFi components
WiFiUDP udp;
WiFiServer* tcpServer = nullptr;
WiFiClient tcpClient;
IPAddress clientIP;  // Dynamically set when client connects
WiFiMode currentWifiMode = WiFiMode::AP;  // Default to AP mode

// ============================================================================
// WiFi Mode Selection (Interactive)
// ============================================================================
WiFiMode selectWifiMode(uint32_t timeoutMs) {
    Serial.println(F("\n========================================"));
    Serial.println(F("       WiFi Mode Selection"));
    Serial.println(F("========================================"));
    Serial.println(F("  [A] Access Point (AP) mode - DEFAULT"));
    Serial.println(F("  [C] Client mode (connect to network)"));
    Serial.println(F("========================================"));
    Serial.print(F("Select mode within "));
    Serial.print(timeoutMs / 1000);
    Serial.println(F(" seconds (default: AP)..."));

    uint32_t startTime = millis();
    
    while ((millis() - startTime) < timeoutMs) {
        if (Serial.available() > 0) {
            char input = toupper(Serial.read());
            // Clear remaining input buffer
            while (Serial.available()) Serial.read();
            
            if (input == 'A') {
                Serial.println(F("\n>> AP mode selected"));
                currentWifiMode = WiFiMode::AP;
                return WiFiMode::AP;
            } else if (input == 'C') {
                Serial.println(F("\n>> Client mode selected"));
                currentWifiMode = WiFiMode::CLIENT;
                return WiFiMode::CLIENT;
            } else {
                Serial.print(F("Invalid input '"));
                Serial.print(input);
                Serial.println(F("'. Press [A] for AP or [C] for Client."));
            }
        }
        
        // Print countdown every second
        uint32_t elapsed = millis() - startTime;
        uint32_t remaining = (timeoutMs - elapsed) / 1000;
        static uint32_t lastPrint = 0;
        if (remaining != lastPrint && remaining > 0) {
            lastPrint = remaining;
            Serial.print(remaining);
            Serial.print(F("... "));
        }
        
        delay(100);
    }
    
    Serial.println(F("\n\n>> Timeout - defaulting to AP mode"));
    currentWifiMode = WiFiMode::AP;
    return WiFiMode::AP;
}

// ============================================================================
// WiFi AP Mode Configuration
// ============================================================================
void configureWifiAP() {
    Serial.println(F("INFO: Starting WiFi Access Point..."));
    Serial.print(F("SSID: "));
    Serial.println(WIFIAP_SSID);
    Serial.print(F("Channel: "));
    Serial.println(WIFIAP_CHANNEL);

    // Give WiFi module time to initialize (2s to allow recovery after
    // ADC SPI init on the same MCU – the Murata co-processor can be slow)
    delay(2000);

    // Configure static IP for AP mode
    IPAddress apIP(192, 168, 4, 1);
    IPAddress gateway(192, 168, 4, 1);
    IPAddress subnet(255, 255, 255, 0);
    WiFi.config(apIP, gateway, subnet);

    // Start AP mode
    Serial.println(F("Calling WiFi.beginAP()..."));
    int status = WiFi.beginAP(WIFIAP_SSID, WIFIAP_PASS, WIFIAP_CHANNEL);
    Serial.print(F("WiFi.beginAP() returned status: "));
    Serial.println(status);
    
    if (status != WL_AP_LISTENING) {
        Serial.println(F("ERROR: Failed to create Access Point!"));
        Serial.print(F("WiFi status code: "));
        Serial.println(status);
        Serial.println(F("Expected WL_AP_LISTENING (1)"));
        Serial.println(F("Possible causes:"));
        Serial.println(F("  - WiFi module not ready"));
        Serial.println(F("  - Invalid SSID/password"));
        Serial.println(F("  - Channel conflict"));
        indicator::setWifiFail();
        while (1) { delay(1000); } // die with feedback
    }

    // Get AP IP address
    IPAddress ip = WiFi.localIP();
    Serial.println(F("\nINFO: Access Point started successfully!"));
    Serial.print(F("AP IP address: "));
    Serial.println(ip);
    Serial.print(F("TCP port: "));
    Serial.println(CLIENT_PORT_TCP);
    Serial.print(F("UDP port (client should listen): "));
    Serial.println(CLIENT_PORT_UDP);

    // Start TCP server for commands
    tcpServer = new WiFiServer(CLIENT_PORT_TCP);
    tcpServer->begin();

    Serial.println(F("Waiting for client connection..."));
    Serial.print(F("Connect to WiFi network '"));
    Serial.print(WIFIAP_SSID);
    Serial.println(F("' and establish TCP connection."));

    // Wait for TCP client
    indicator::setIdle();
    while (!tcpClient.connected()) {
        tcpClient = tcpServer->accept();
        delay(100);
    }

    clientIP = tcpClient.remoteIP();
    Serial.print(F("Client connected from IP: "));
    Serial.println(clientIP);
    Serial.print(F("UDP data will be sent to: "));
    Serial.print(clientIP);
    Serial.print(F(":"));
    Serial.println(CLIENT_PORT_UDP);
    tcpClient.setTimeout(100);
    indicator::setClientConnected();
}

// ============================================================================
// WiFi Client Mode Configuration (connect to existing network)
// ============================================================================
void configureWifiServe() {
    Serial.println(F("INFO: Connecting to WiFi network..."));
    Serial.print(F("SSID: "));
    Serial.println(SECRET_SSID);

    // Connect to WiFi network
    WiFi.begin(SECRET_SSID, SECRET_PASS);

    // Wait for connection with timeout
    int attempts = 0;
    const int MAX_ATTEMPTS = 30; // 15 seconds at 500ms per attempt
    
    while (WiFi.status() != WL_CONNECTED && attempts < MAX_ATTEMPTS) {
        delay(500);
        Serial.print(".");
        attempts++;
    }
    Serial.println();

    if (WiFi.status() != WL_CONNECTED) {
        Serial.println(F("ERROR: Failed to connect to WiFi network!"));
        Serial.print(F("WiFi status: "));
        Serial.println(WiFi.status());
        indicator::setWifiFail();
        while (1) {} //die
    }

    // Successfully connected
    IPAddress ip = WiFi.localIP();
    Serial.println(F("\nINFO: WiFi connected successfully!"));
    Serial.print(F("IP address: "));
    Serial.println(ip);
    Serial.print(F("TCP port: "));
    Serial.println(CLIENT_PORT_TCP);
    Serial.print(F("UDP port: "));
    Serial.println(CLIENT_PORT_UDP);

    // Start TCP server for commands
    tcpServer = new WiFiServer(CLIENT_PORT_TCP);
    tcpServer->begin();

    Serial.println(F("Waiting for client connection..."));

    // Wait for TCP client
    indicator::setIdle();
    while (!tcpClient.connected()) {
        tcpClient = tcpServer->accept();
        delay(100);
    }

    clientIP = tcpClient.remoteIP();
    Serial.print(F("Client connected: "));
    Serial.println(clientIP);
    tcpClient.setTimeout(100);
    indicator::setClientConnected();
}
void print(const char* s) {
    if (tcpClient && tcpClient.connected()) {
        tcpClient.print(s);
    }
}

void println(const char* s) {
    if (tcpClient && tcpClient.connected()) {
        tcpClient.println(s);
    }
}


static void decodeCommand(uint8_t* rxbuf, std::atomic_bool* logging_active, vertiq::MotorCommandQueue* commandQueue, adc::ADS1220* ads) {
    char cmd = toupper(rxbuf[0]);

    switch (cmd) {
        case 'Q':  // Stop logging
            logging_active->store(false, std::memory_order_relaxed);
            println("Logging disabled");
            break;

        case 'W':  // Start logging
            logging_active->store(true, std::memory_order_relaxed);
            println("Logging enabled");
            break;

        case 'J':  // Calibrate ADC baseline
            if (!ads) {
                println("ADC device not initialized");
                break;
            }
            ads->calibrateBaseline();
            println("ADC baseline recalibrating");
            break;

        case 'M':  // Motor state command: "M <state>"
            if (!commandQueue) {
                println("Motor command queue not initialized");
                break;
            }

            {
                bool valid = (rxbuf[1] == ' ' && rxbuf[2] >= '0' && rxbuf[2] <= '4');
                if (!valid) {
                    println("Invalid format. Use: M <0-4>");
                    break;;
                }

                int state = rxbuf[2] - '0';
                commandQueue->queueStateCommand(state);
                char buf[32];
                snprintf(buf, sizeof(buf), "Motor state %d queued", state);
                println(buf);
            }

            break;
        case 'S':  // Status command
            if (!commandQueue) {
                println("Motor command queue not activated");
                break;
            }
            commandQueue->queueStatusCommand();

            break;

        case 'P':  // PID Kp command
            if (!commandQueue) break;
            if (rxbuf[1] != ' ') break;

            {
                float value = atof((char*)&rxbuf[2]);
                if (value >= 0.0f && value <= 10.0f) {
                    commandQueue->queuePidCommand(cmd, value);
                    char buf[48];
                    snprintf(buf, sizeof(buf), "Set K%c = %.3f queued", tolower(cmd), value);
                    println(buf);
                } else {
                    println("Value out of range (0-10)");
                }
            }

        break;
        case 'D':  // PID Kd command
            if (!commandQueue) break;
            if (rxbuf[1] != ' ') break;

            {
                float value = atof((char*)&rxbuf[2]);
                if (value >= 0.0f && value <= 10.0f) {
                    commandQueue->queuePidCommand(cmd, value);
                    char buf[48];
                    snprintf(buf, sizeof(buf), "Set K%c = %.3f queued", tolower(cmd), value);
                    println(buf);
                } else {
                    println("Value out of range (0-10)");
                }
            }

            break;

        case 'I':  // PID Ki or ADC threshold

            if (rxbuf[1] == ' ' && commandQueue) {
                float value = atof((char*)&rxbuf[2]);
                if (value >= 0.0f && value <= 10.0f) {
                    commandQueue->queuePidCommand('I', value);
                    char buf[48];
                    snprintf(buf, sizeof(buf), "Set Ki = %.3f queued", value);
                    println(buf);
                } else {
                    println("Value out of range (0-10)");
                }
            } else if (ads) {
                int32_t threshold;
                memcpy(&threshold, &rxbuf[2], 4);
                ads->setThreshold(threshold);
            }
            break;

        case 'H':  // Help
            println("\n=== Commands ===");
            println("W       - Start logging");
            println("Q       - Stop logging");
            println("M <0-4> - Motor state");
            println("S       - Motor status");
            println("P/I/D <val> - PID gains");
            println("J       - ADC calibrate");
            println("C       - Classifier status");
            println("N       - Mode status");
            println("O <1-3> - Set mode (1=MANUAL, 2=SEMIAUTO, 3=AUTO)");
            println("L       - Show label-to-motor mapping");
            println("H       - This help");
            break;

        case 'C':  // Classifier status
            ei_classifier::printStatus();
            break;

        case 'N':  // Mode status
            control::printModeStatus();
            break;

        case 'O':  // Set operational mode
            {
                if (rxbuf[1] == ' ' && rxbuf[2] >= '1' && rxbuf[2] <= '3') {
                    int modeNum = rxbuf[2] - '0';
                    control::ControlMode newMode;
                    switch (modeNum) {
                        case 1: newMode = control::ControlMode::MANUAL; break;
                        case 2: newMode = control::ControlMode::SEMIAUTO; break;
                        case 3: newMode = control::ControlMode::AUTO; break;
                        default: newMode = control::ControlMode::MANUAL; break;
                    }
                    control::setMode(newMode);
                } else {
                    println("Invalid format. Use: O <1-3>");
                    println("  1=MANUAL, 2=SEMIAUTO, 3=AUTO");
                }
            }
            break;

        case 'L':  // Show label-to-motor mapping
            control::printLabelMotorMapping();
            break;

        default:
            break;
    }

    memset(rxbuf, 0, RXBUF_SIZE);
}

constexpr size_t HEADER_SIZE = 2;
// BATCH_SIZE is defined in main_prelude.hpp


void console_task(void* arg) {
    static uint8_t rxbuf[RXBUF_SIZE];

    memset(rxbuf, 0, RXBUF_SIZE);
    vertiq::MotorCommandQueue* commandQueue = static_cast<vertiq::MotorCommandQueue*>( ((void**)arg)[0] );
    adc::ADS1220* adcDevice = static_cast<adc::ADS1220*>( ((void**)arg)[1] );
    std::atomic_bool* loggingActive = static_cast<std::atomic_bool*>( ((void**)arg)[2] );

    Serial.println("Console task started");

    while (1) {
        // Wait for valid client connection (monitor task sets this up)
        if (!tcpClient) {
            hal::sleepMs(100);
            continue;
        }

        // Block on TCP read
        int bytesRead = tcpClient.readBytes(rxbuf, 1);
        
        if (bytesRead > 0) {
            // Read rest of available data if any
            if (tcpClient.available() > 0) {
                bytesRead += tcpClient.readBytes(
                    rxbuf + 1,
                    min(tcpClient.available(), (int)(RXBUF_SIZE - 1))
                );
            }
            
            // Process the command
            decodeCommand(rxbuf, loggingActive, commandQueue, adcDevice);
        }
        // If read fails (timeout or error), just loop
        // Monitor task will detect disconnection and clean up
    }
}

void logging_task(void* arg) {
    hal::Queue<IMUPacketSlot, 32>* imuQueue = static_cast<hal::Queue<IMUPacketSlot, 32>*>( ((void**)arg)[0] );
    std::atomic_bool* loggingActive = static_cast<std::atomic_bool*>( ((void**)arg)[1] );

    // Initialize UDP - bind to any available port for sending
    udp.begin(RECV_PORT_UDP);
    
    static uint8_t outgoingBuffer[MTU];
    uint32_t udpSendCount = 0;
    uint32_t lastReportTime = 0;
    static bool firstSend = true;

    // Saved slot: holds a packet that didn't fit in the previous UDP batch
    // so it is sent first in the next batch instead of being dropped.
    static IMUPacketSlot savedSlot;
    static bool hasSavedSlot = false;

    Serial.println("Logging task started");
    Serial.print("UDP initialized, will send to port: ");
    Serial.println(CLIENT_PORT_UDP);

    while (1) {
        size_t head = 0;
        size_t packetCount = 0;

        // Wait for client to be connected before trying to send
        if (!tcpClient || !tcpClient.connected()) {
            // Drain queue but don't send
            IMUPacketSlot slot;
            imuQueue->pop(slot, 0);  // Non-blocking pop to drain
            hasSavedSlot = false;    // discard any saved slot too
            hal::sleepMs(10);
            continue;
        }

        while (packetCount < BATCH_SIZE) {
            IMUPacketSlot slot;

            // Re-use slot that overflowed last batch before popping a new one
            if (hasSavedSlot) {
                slot = savedSlot;
                hasSavedSlot = false;
            } else {
                // Block until packet arrives
                imuQueue->pop(slot, 0xFFFFFFFF);
            }

            // Skip if logging disabled (but still drain queue)
            if (!loggingActive->load(std::memory_order_relaxed)) {
                continue;
            }

            size_t sz = slot.size;

            // Sanity check: drop any single message larger than MTU-2
            if (sz + HEADER_SIZE > MTU) {
                continue;
            }

            // If adding header+message would overflow this UDP packet,
            // save it for the next batch and flush now
            if (head + HEADER_SIZE + sz > MTU) {
                savedSlot = slot;
                hasSavedSlot = true;
                break;
            }

            // Write big-endian length prefix
            outgoingBuffer[head + 0] = uint8_t((sz >> 8) & 0xFF);
            outgoingBuffer[head + 1] = uint8_t((sz) & 0xFF);
            head += HEADER_SIZE;

            // Copy the protobuf bytes
            memcpy(outgoingBuffer + head, slot.data, sz);
            head += sz;

            packetCount++;
        }

        if (head > 0 && clientIP) {
            // Debug: print destination on first send
            if (firstSend) {
                Serial.print("First UDP send to: ");
                Serial.print(clientIP);
                Serial.print(":");
                Serial.println(CLIENT_PORT_UDP);
                firstSend = false;
            }

            if (udp.beginPacket(clientIP, CLIENT_PORT_UDP)) {
                udp.write(outgoingBuffer, head);
                if (udp.endPacket()) {
                    udpSendCount++;

                    // Report every 5 seconds
                    uint32_t now = millis();
                    if (now - lastReportTime >= 5000) {
                        Serial.print("UDP sends: ");
                        Serial.print(udpSendCount);
                        Serial.print(" packets to ");
                        Serial.println(clientIP);
                        lastReportTime = now;
                    }
                } else {
                    Serial.println("ERROR: UDP endPacket failed");
                }
            } else {
                Serial.println("ERROR: UDP beginPacket failed");
            }
        }
    }
}

void monitor_task(void* arg) {
    Serial.println("Monitor task started");
    bool clientConnected = false;
    
    while (1) {
        // 1. Check WiFi connection health (only in Client mode, not AP mode)
        if (currentWifiMode == WiFiMode::CLIENT && WiFi.status() != WL_CONNECTED) {
            Serial.println("WARNING: WiFi disconnected, attempting reconnect...");
            indicator::setWifiFail();
            clientConnected = false;
            
            WiFi.begin(SECRET_SSID, SECRET_PASS);
            int attempts = 0;
            while (WiFi.status() != WL_CONNECTED && attempts < 20) {
                hal::sleepMs(500);
                attempts++;
            }
            
            if (WiFi.status() == WL_CONNECTED) {
                Serial.println("WiFi reconnected");
                Serial.print("IP address: ");
                Serial.println(WiFi.localIP());
            }
        }
        
        // 2. Check client connection status
        bool hasClient = tcpClient && tcpClient.connected();
        
        if (!hasClient) {
            // No client or disconnected - clean up and show yellow LED
            if (clientConnected) {
                Serial.println("Client disconnected (detected by monitor)");
                if (tcpClient) {
                    tcpClient.stop();
                }
                clientConnected = false;
            }
            indicator::setIdle();
            
            // Try to accept new client (non-blocking)
            WiFiClient newClient = tcpServer->accept();
            if (newClient && newClient.connected()) {
                tcpClient = newClient;
                clientIP = tcpClient.remoteIP();
                Serial.print("Client connected: ");
                Serial.println(clientIP);
                clientConnected = true;
                indicator::setClientConnected();
            }
        } else if (!clientConnected) {
            // We have a connected client but didn't know about it - update state
            clientConnected = true;
            indicator::setClientConnected();
        }
        
        hal::sleepMs(1000);  // Check every second
    }
}

}
