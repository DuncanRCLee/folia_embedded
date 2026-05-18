#pragma once

#include "Arduino.h"
#include <SPI.h>

namespace adc {

    // ADC states
    enum class AdcState : uint8_t {
        STANDBY = 1,
        MEASURE_BASE = 2,
        RUNNING = 3
    };

    // ADS1220 SPI commands
    constexpr uint8_t SPI_MASTER_DUMMY = 0xFF;
    constexpr uint8_t CMD_RESET = 0x06;
    constexpr uint8_t CMD_START = 0x08;
    constexpr uint8_t CMD_WREG = 0x40;
    constexpr uint8_t CMD_RREG = 0x20;

    // Config register addresses
    constexpr uint8_t CONFIG_REG0 = 0x00;
    constexpr uint8_t CONFIG_REG1 = 0x01;
    constexpr uint8_t CONFIG_REG2 = 0x02;
    constexpr uint8_t CONFIG_REG3 = 0x03;

    // Data rates
    constexpr uint8_t DR_20SPS = 0x00;
    constexpr uint8_t DR_45SPS = 0x20;
    constexpr uint8_t DR_90SPS = 0x40;
    constexpr uint8_t DR_175SPS = 0x60;
    constexpr uint8_t DR_330SPS = 0x80;
    constexpr uint8_t DR_600SPS = 0xA0;
    constexpr uint8_t DR_1000SPS = 0xC0;

    // PGA gains
    constexpr uint8_t PGA_GAIN_1 = 0x00;
    constexpr uint8_t PGA_GAIN_2 = 0x02;
    constexpr uint8_t PGA_GAIN_4 = 0x04;
    constexpr uint8_t PGA_GAIN_8 = 0x06;
    constexpr uint8_t PGA_GAIN_16 = 0x08;
    constexpr uint8_t PGA_GAIN_32 = 0x0A;
    constexpr uint8_t PGA_GAIN_64 = 0x0C;
    constexpr uint8_t PGA_GAIN_128 = 0x0E;

    // MUX channels
    constexpr uint8_t MUX_AIN0_AIN1 = 0x00;
    constexpr uint8_t MUX_AIN0_AIN2 = 0x10;
    constexpr uint8_t MUX_AIN0_AIN3 = 0x20;
    constexpr uint8_t MUX_AIN1_AIN2 = 0x30;
    constexpr uint8_t MUX_AIN1_AIN3 = 0x40;
    constexpr uint8_t MUX_AIN2_AIN3 = 0x50;

    class ADS1220 {
    public:
        ADS1220(PinName mosi, PinName miso, PinName sclk, PinName cs, PinName drdy);

        bool begin();
        int8_t loop();

        // Configuration
        void setDataRate(uint8_t rate);
        void setPgaGain(uint8_t gain);
        void selectMuxChannels(uint8_t channels);
        void setContinuousMode();
        void lowSideSwitchClosed();

        // Conversion
        void startConversion();
        int32_t readData();

        // State management
        void calibrateBaseline();
        AdcState getState() const { return state_; }

        // Data access
        int32_t getRaw() const { return raw_; }
        int32_t getBaseline() const { return baseline_; }
        int32_t getThreshold() const { return threshold_; }
        void setThreshold(int32_t t) { threshold_ = t; }

        uint8_t* getConfigRegs();

    private:
        void writeRegister(uint8_t addr, uint8_t value);
        uint8_t readRegister(uint8_t addr);
        void spiCommand(uint8_t cmd);
        void reset();

        MbedSPI spi_;
        PinName csPin_;
        PinName drdyPin_;
        SPISettings spiSettings_;

        uint8_t configReg0_;
        uint8_t configReg1_;
        uint8_t configReg2_;
        uint8_t configReg3_;

        AdcState state_;
        int32_t raw_;
        int32_t baseline_;
        int32_t threshold_;
    };

} // namespace adc
