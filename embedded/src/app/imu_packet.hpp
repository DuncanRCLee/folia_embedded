#pragma once
#include <cstdint>
#include <cstddef>
#include "gen/Packet.pb.h"

// Maximum size for a single encoded IMU packet
// Named to avoid conflict with IQ Module Communication's MAX_PACKET_SIZE
constexpr size_t IMU_PACKET_BUF_SIZE = imu_Packet_size;

// Structure to hold a pre-encoded protobuf packet
struct IMUPacketSlot {
    uint8_t data[IMU_PACKET_BUF_SIZE];
    size_t size;  // Actual encoded size

    IMUPacketSlot() : size(0) {
        memset(data,0,IMU_PACKET_BUF_SIZE);
    }
};
