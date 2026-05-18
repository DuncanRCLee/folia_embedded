#pragma once
#include "Arduino.h"
#include <atomic>

class Flags {
public:
    // Bit positions for each flag
    enum FlagBits : uint8_t {
        NET_BUFFER_FULL_BIT = 0,
        NET_ENCODE_BAD_BIT = 1,
        NET_UDP_SEND_BAD_BIT = 2,
    };

    Flags() : flags(0) {}

    void set(uint8_t bit) {
        flags.fetch_or(1 << bit, std::memory_order_relaxed);
    }

    void parse_print_clear() {
        uint8_t old_flags = flags.exchange(0, std::memory_order_acquire);

        if (old_flags & (1 << NET_BUFFER_FULL_BIT)) {
            Serial.println("ERR: The UDP buffer was full when a protobuf packet was tried to be written to it.");
        }
        if (old_flags & (1 << NET_ENCODE_BAD_BIT)) {
            Serial.println("ERR: Encoding of the protobuf packet failed.");
        }
        if (old_flags & (1 << NET_UDP_SEND_BAD_BIT)) {
            Serial.println("ERR: Sending the UDP packet failed.");
        }
    }

    bool is_set(uint8_t bit) const {
        return flags.load(std::memory_order_relaxed) & (1 << bit);
    }

private:
    std::atomic<uint8_t> flags;
};
