#include "ads1220.hpp"

namespace adc {

    // Register masks
    constexpr uint8_t REG_CONFIG0_MUX_MASK = 0xF0;
    constexpr uint8_t REG_CONFIG0_PGA_GAIN_MASK = 0x0E;
    constexpr uint8_t REG_CONFIG1_DR_MASK = 0xE0;

    ADS1220::ADS1220(PinName mosi, PinName miso, PinName sclk, PinName cs, PinName drdy)
        : spi_(miso, mosi, sclk)
        , csPin_(cs)
        , drdyPin_(drdy)
        , spiSettings_(1000000, MSBFIRST, SPI_MODE1)
        , configReg0_(0x00)
        , configReg1_(0x04)
        , configReg2_(0x10)
        , configReg3_(0x00)
        , state_(AdcState::STANDBY)
        , raw_(0)
        , baseline_(0)
        , threshold_(0)
    {
    }

    bool ADS1220::begin() {
        pinMode(csPin_, OUTPUT);
        pinMode(drdyPin_, INPUT);

        spi_.begin();
        spi_.beginTransaction(spiSettings_);
        spi_.transfer(0);
        spi_.endTransaction();

        spi_.beginTransaction(spiSettings_);
        delay(100);
        reset();
        delay(100);

        digitalWrite(csPin_, LOW);

        // Default settings
        configReg0_ = 0x00;  // AINP=AIN0, AINN=AIN1, Gain 1, PGA enabled
        configReg1_ = 0x04;  // DR=20 SPS, Mode=Normal, Conv mode=continuous
        configReg2_ = 0x10;  // Vref internal, 50/60Hz rejection
        configReg3_ = 0x00;  // IDAC disabled, DRDY pin only

        writeRegister(CONFIG_REG0, configReg0_);
        writeRegister(CONFIG_REG1, configReg1_);
        writeRegister(CONFIG_REG2, configReg2_);
        writeRegister(CONFIG_REG3, configReg3_);

        delay(100);

        // Verify configuration
        uint8_t r0 = readRegister(CONFIG_REG0);
        uint8_t r1 = readRegister(CONFIG_REG1);
        uint8_t r2 = readRegister(CONFIG_REG2);
        uint8_t r3 = readRegister(CONFIG_REG3);

        bool configOk = (r0 == configReg0_) && (r1 == configReg1_) &&
                        (r2 == configReg2_) && (r3 == configReg3_);

        digitalWrite(csPin_, HIGH);
        delay(100);

        return configOk;
    }

    int8_t ADS1220::loop() {
        int32_t newReading = readData();
        if (newReading == 0) {
            return -1;
        }

        raw_ = newReading;

        if (state_ == AdcState::MEASURE_BASE) {
            baseline_ = raw_;
            state_ = AdcState::RUNNING;
        }

        // Subtract baseline, then negate so that compressive load on the
        // half-bridge strain gauge yields positive counts ("more force =
        // more counts").  The ADS1220 differential reading happens to be
        // negative when the gauge is under load with the current wiring;
        // negation corrects for this polarity.
        raw_ = -(raw_ - baseline_);
        return 0;
    }

    void ADS1220::calibrateBaseline() {
        state_ = AdcState::MEASURE_BASE;
    }

    void ADS1220::setDataRate(uint8_t rate) {
        configReg1_ &= ~REG_CONFIG1_DR_MASK;
        configReg1_ |= rate;
        writeRegister(CONFIG_REG1, configReg1_);
    }

    void ADS1220::setPgaGain(uint8_t gain) {
        configReg0_ &= ~REG_CONFIG0_PGA_GAIN_MASK;
        configReg0_ |= gain;
        writeRegister(CONFIG_REG0, configReg0_);
    }

    void ADS1220::selectMuxChannels(uint8_t channels) {
        configReg0_ &= ~REG_CONFIG0_MUX_MASK;
        configReg0_ |= channels;
        writeRegister(CONFIG_REG0, configReg0_);
    }

    void ADS1220::setContinuousMode() {
        configReg1_ |= (1 << 2);
        writeRegister(CONFIG_REG1, configReg1_);
    }

    void ADS1220::lowSideSwitchClosed() {
        configReg2_ |= (1 << 3);
        writeRegister(CONFIG_REG2, configReg2_);
    }

    void ADS1220::startConversion() {
        spiCommand(CMD_START);
    }

    int32_t ADS1220::readData() {
        uint8_t spiBuf[3];
        int32_t result = 0;
        long bit24;

        digitalWrite(csPin_, LOW);
        delayMicroseconds(100);

        for (int i = 0; i < 3; i++) {
            spiBuf[i] = spi_.transfer(SPI_MASTER_DUMMY);
        }

        delayMicroseconds(100);
        digitalWrite(csPin_, HIGH);

        bit24 = spiBuf[0];
        bit24 = (bit24 << 8) | spiBuf[1];
        bit24 = (bit24 << 8) | spiBuf[2];

        // Convert 24-bit two's complement to 32-bit
        bit24 = (bit24 << 8);
        result = (bit24 >> 8);

        return result;
    }

    uint8_t* ADS1220::getConfigRegs() {
        static uint8_t buf[4];
        buf[0] = readRegister(CONFIG_REG0);
        buf[1] = readRegister(CONFIG_REG1);
        buf[2] = readRegister(CONFIG_REG2);
        buf[3] = readRegister(CONFIG_REG3);
        return buf;
    }

    void ADS1220::writeRegister(uint8_t addr, uint8_t value) {
        digitalWrite(csPin_, LOW);
        spi_.transfer(CMD_WREG | (addr << 2));
        spi_.transfer(value);
        digitalWrite(csPin_, HIGH);
    }

    uint8_t ADS1220::readRegister(uint8_t addr) {
        uint8_t data;
        digitalWrite(csPin_, LOW);
        spi_.transfer(CMD_RREG | (addr << 2));
        data = spi_.transfer(SPI_MASTER_DUMMY);
        digitalWrite(csPin_, HIGH);
        return data;
    }

    void ADS1220::spiCommand(uint8_t cmd) {
        digitalWrite(csPin_, LOW);
        delay(2);
        digitalWrite(csPin_, HIGH);
        delay(2);
        digitalWrite(csPin_, LOW);
        delay(2);
        spi_.transfer(cmd);
        delay(2);
        digitalWrite(csPin_, HIGH);
    }

    void ADS1220::reset() {
        spiCommand(CMD_RESET);
    }

} // namespace adc
