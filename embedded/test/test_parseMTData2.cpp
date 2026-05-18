#include <unity.h>
#include <cstring>
#include <cmath>

// Mock Arduino functions that xbus.hpp needs
#ifndef ARDUINO
#define digitalRead(x) 0
#define noInterrupts()
#define interrupts()
#define HEX 16
class SerialMock {
public:
    template<typename T>
    void print(T val) {}
    template<typename T>
    void println(T val) {}
    void println() {}
    template<typename T>
    void print(T val, int format) {}
    template<typename T>
    void println(T val, int format) {}
} Serial;
#endif

// Include the actual header from the HAL library
#include <xbus.hpp>

// Helper function to encode float as big-endian bytes
void encodeFloatBE(uint8_t* dest, float value) {
    uint32_t bits;
    memcpy(&bits, &value, 4);
    dest[0] = (bits >> 24) & 0xFF;
    dest[1] = (bits >> 16) & 0xFF;
    dest[2] = (bits >> 8) & 0xFF;
    dest[3] = bits & 0xFF;
}

// Test parsing acceleration data
void test_parse_acceleration() {
    xsens::Xbus xbus(0, 0);  // Dummy address and drdy

    // Build packet: DataID (0x4020) + Length (12) + 3 floats (x=1.0, y=2.0, z=3.0)
    uint8_t packet[15];
    packet[0] = 0x40;  // DataID high byte
    packet[1] = 0x20;  // DataID low byte
    packet[2] = 12;    // Length
    encodeFloatBE(&packet[3], 1.0f);
    encodeFloatBE(&packet[7], 2.0f);
    encodeFloatBE(&packet[11], 3.0f);

    xbus.parseMTData2(packet, 15);

    TEST_ASSERT_FLOAT_WITHIN(0.001, 1.0f, xbus.acc[0]);
    TEST_ASSERT_FLOAT_WITHIN(0.001, 2.0f, xbus.acc[1]);
    TEST_ASSERT_FLOAT_WITHIN(0.001, 3.0f, xbus.acc[2]);
}

// Test parsing quaternion data
void test_parse_quaternion() {
    xsens::Xbus xbus(0, 0);

    // Build packet: DataID (0x2010) + Length (16) + 4 floats (w=1.0, x=0.0, y=0.0, z=0.0)
    uint8_t packet[19];
    packet[0] = 0x20;  // DataID high byte
    packet[1] = 0x10;  // DataID low byte
    packet[2] = 16;    // Length
    encodeFloatBE(&packet[3], 1.0f);
    encodeFloatBE(&packet[7], 0.0f);
    encodeFloatBE(&packet[11], 0.0f);
    encodeFloatBE(&packet[15], 0.0f);

    xbus.parseMTData2(packet, 19);

    TEST_ASSERT_FLOAT_WITHIN(0.001, 1.0f, xbus.quat[0]);
    TEST_ASSERT_FLOAT_WITHIN(0.001, 0.0f, xbus.quat[1]);
    TEST_ASSERT_FLOAT_WITHIN(0.001, 0.0f, xbus.quat[2]);
    TEST_ASSERT_FLOAT_WITHIN(0.001, 0.0f, xbus.quat[3]);
}

void test_fail() {
    TEST_ASSERT_EQUAL(1, 0); // Force a failure
}

// Test parsing gyroscope data
void test_parse_gyroscope() {
    xsens::Xbus xbus(0, 0);

    // Build packet: DataID (0x8020) + Length (12) + 3 floats
    uint8_t packet[15];
    packet[0] = 0x80;  // DataID high byte
    packet[1] = 0x20;  // DataID low byte
    packet[2] = 12;    // Length
    encodeFloatBE(&packet[3], 0.1f);
    encodeFloatBE(&packet[7], 0.2f);
    encodeFloatBE(&packet[11], 0.3f);

    xbus.parseMTData2(packet, 15);

    TEST_ASSERT_FLOAT_WITHIN(0.001, 0.1f, xbus.gyr[0]);
    TEST_ASSERT_FLOAT_WITHIN(0.001, 0.2f, xbus.gyr[1]);
    TEST_ASSERT_FLOAT_WITHIN(0.001, 0.3f, xbus.gyr[2]);
}

// Test parsing multiple data blocks in one packet
void test_parse_multiple_blocks() {
    xsens::Xbus xbus(0, 0);

    // Build packet with both acceleration and gyroscope
    uint8_t packet[30];
    int offset = 0;

    // Acceleration block
    packet[offset++] = 0x40;
    packet[offset++] = 0x20;
    packet[offset++] = 12;
    encodeFloatBE(&packet[offset], 1.0f); offset += 4;
    encodeFloatBE(&packet[offset], 2.0f); offset += 4;
    encodeFloatBE(&packet[offset], 3.0f); offset += 4;

    // Gyroscope block
    packet[offset++] = 0x80;
    packet[offset++] = 0x20;
    packet[offset++] = 12;
    encodeFloatBE(&packet[offset], 0.1f); offset += 4;
    encodeFloatBE(&packet[offset], 0.2f); offset += 4;
    encodeFloatBE(&packet[offset], 0.3f); offset += 4;

    xbus.parseMTData2(packet, offset);

    TEST_ASSERT_FLOAT_WITHIN(0.001, 1.0f, xbus.acc[0]);
    TEST_ASSERT_FLOAT_WITHIN(0.001, 2.0f, xbus.acc[1]);
    TEST_ASSERT_FLOAT_WITHIN(0.001, 3.0f, xbus.acc[2]);
    TEST_ASSERT_FLOAT_WITHIN(0.001, 0.1f, xbus.gyr[0]);
    TEST_ASSERT_FLOAT_WITHIN(0.001, 0.2f, xbus.gyr[1]);
    TEST_ASSERT_FLOAT_WITHIN(0.001, 0.3f, xbus.gyr[2]);
}

// Test parsing packet with MTData2 header
void test_parse_with_header() {
    xsens::Xbus xbus(0, 0);

    uint8_t packet[17];
    packet[0] = 0x36;  // MTData2 header
    packet[1] = 15;    // Block length
    packet[2] = 0x40;  // DataID high byte
    packet[3] = 0x20;  // DataID low byte
    packet[4] = 12;    // Length
    encodeFloatBE(&packet[5], 5.0f);
    encodeFloatBE(&packet[9], 6.0f);
    encodeFloatBE(&packet[13], 7.0f);

    xbus.parseMTData2(packet, 17);

    TEST_ASSERT_FLOAT_WITHIN(0.001, 5.0f, xbus.acc[0]);
    TEST_ASSERT_FLOAT_WITHIN(0.001, 6.0f, xbus.acc[1]);
    TEST_ASSERT_FLOAT_WITHIN(0.001, 7.0f, xbus.acc[2]);
}

// Test parsing high-rate accelerometer data
void test_parse_hr_acc() {
    xsens::Xbus xbus(0, 0);

    uint8_t packet[15];
    packet[0] = 0x40;  // DataID high byte
    packet[1] = 0x40;  // DataID low byte
    packet[2] = 12;    // Length
    encodeFloatBE(&packet[3], 9.8f);
    encodeFloatBE(&packet[7], 0.0f);
    encodeFloatBE(&packet[11], 0.0f);

    xbus.parseMTData2(packet, 15);

    TEST_ASSERT_FLOAT_WITHIN(0.001, 9.8f, xbus.HR_acc[0]);
    TEST_ASSERT_FLOAT_WITHIN(0.001, 0.0f, xbus.HR_acc[1]);
    TEST_ASSERT_FLOAT_WITHIN(0.001, 0.0f, xbus.HR_acc[2]);
}

// Test parsing euler angles
void test_parse_euler() {
    xsens::Xbus xbus(0, 0);

    uint8_t packet[15];
    packet[0] = 0x20;
    packet[1] = 0x30;
    packet[2] = 12;
    encodeFloatBE(&packet[3], 45.0f);   // roll
    encodeFloatBE(&packet[7], 30.0f);   // pitch
    encodeFloatBE(&packet[11], 90.0f);  // yaw

    xbus.parseMTData2(packet, 15);

    TEST_ASSERT_FLOAT_WITHIN(0.001, 45.0f, xbus.euler[0]);
    TEST_ASSERT_FLOAT_WITHIN(0.001, 30.0f, xbus.euler[1]);
    TEST_ASSERT_FLOAT_WITHIN(0.001, 90.0f, xbus.euler[2]);
}

// Test empty/malformed packet handling
void test_parse_empty_packet() {
    xsens::Xbus xbus(0, 0);
    uint8_t packet[2] = {0x00, 0x00};

    // Should not crash
    xbus.parseMTData2(packet, 2);

    // Values should remain NAN
    auto dt = xbus.acc[0];
    TEST_ASSERT_TRUE(std::isnan(dt));
}

// Test packet with length mismatch (boundary check)
void test_parse_truncated_packet() {
    xsens::Xbus xbus(0, 0);

    uint8_t packet[10];
    packet[0] = 0x40;
    packet[1] = 0x20;
    packet[2] = 12;  // Claims 12 bytes but packet is shorter
    encodeFloatBE(&packet[3], 1.0f);

    // Should handle gracefully without crash
    xbus.parseMTData2(packet, 10);

    // Data should not be parsed due to length check
    TEST_ASSERT_TRUE(std::isnan(xbus.acc[0]));
}

// Test DeltaV parsing
void test_parse_deltav() {
    xsens::Xbus xbus(0, 0);

    uint8_t packet[15];
    packet[0] = 0x40;
    packet[1] = 0x10;
    packet[2] = 12;
    encodeFloatBE(&packet[3], 0.05f);
    encodeFloatBE(&packet[7], 0.1f);
    encodeFloatBE(&packet[11], 0.15f);

    xbus.parseMTData2(packet, 15);

    TEST_ASSERT_FLOAT_WITHIN(0.001, 0.05f, xbus.dv[0]);
    TEST_ASSERT_FLOAT_WITHIN(0.001, 0.1f, xbus.dv[1]);
    TEST_ASSERT_FLOAT_WITHIN(0.001, 0.15f, xbus.dv[2]);
}

// Test DeltaQ parsing
void test_parse_deltaq() {
    xsens::Xbus xbus(0, 0);

    uint8_t packet[19];
    packet[0] = 0x80;
    packet[1] = 0x30;
    packet[2] = 16;
    encodeFloatBE(&packet[3], 0.999f);
    encodeFloatBE(&packet[7], 0.001f);
    encodeFloatBE(&packet[11], 0.002f);
    encodeFloatBE(&packet[15], 0.003f);

    xbus.parseMTData2(packet, 19);

    TEST_ASSERT_FLOAT_WITHIN(0.001, 0.999f, xbus.dq[0]);
    TEST_ASSERT_FLOAT_WITHIN(0.001, 0.001f, xbus.dq[1]);
    TEST_ASSERT_FLOAT_WITHIN(0.001, 0.002f, xbus.dq[2]);
    TEST_ASSERT_FLOAT_WITHIN(0.001, 0.003f, xbus.dq[3]);
}

void setUp(void) {
    // Set up before each test
}

void tearDown(void) {
    // Clean up after each test
}

int main(int argc, char **argv) {
    UNITY_BEGIN();

    RUN_TEST(test_parse_acceleration);
    RUN_TEST(test_parse_quaternion);
    RUN_TEST(test_parse_gyroscope);
    RUN_TEST(test_parse_multiple_blocks);
    RUN_TEST(test_parse_with_header);
    RUN_TEST(test_parse_hr_acc);
    RUN_TEST(test_parse_euler);
    RUN_TEST(test_parse_empty_packet);
    RUN_TEST(test_parse_truncated_packet);
    RUN_TEST(test_parse_deltav);
    RUN_TEST(test_parse_deltaq);


    return UNITY_END();
}
