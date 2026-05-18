#pragma once
#include "portable.hpp"

//Definition of opcodes: For more information on opcodes, refer to https://mtidocs.xsens.com/functional-description$mtssp-synchronous-serial-protocol
#define XSENS_CONTROL_PIPE 0x03       //Use this opcode for sending (configuration) commands to the MTi
#define XSENS_STATUS_PIPE 0x04        //Use this opcode for reading the status of the notification/measurement pipes
#define XSENS_NOTIF_PIPE 0x05         //Use this opcode for reading a notification message
#define XSENS_MEAS_PIPE 0x06          //Use this opcode for reading measurement data (MTData2 message)

namespace xsens {

class Xbus {
  public:

    // Constructor for mbed platform (I2C): address is I2C address, drdy is PinName
    #ifdef ARDUINO_ARCH_MBED
    Xbus(uint8_t address, TwoWire& wire, PinName drdy);
    #endif

    // Constructor for Arduino platform (SPI): address is CS pin, drdy is digital pin number
    #ifdef ARDUINO_ARCH_RENESAS
    Xbus(uint8_t address, uint8_t drdy);
    #endif

    // Constructor for test/non-Arduino builds (defined in xbus_test.cpp)
    #if !defined(ARDUINO)
    Xbus(uint8_t address, uint8_t drdy);
    #endif

    enum MesID {WAKEUP = 0x3E, WAKEUPACK = 0x3F, RESET = 0x40, RESETACK = 0x41,
                GOTOCONFIGACK = 0x31, GOTOMEASUREMENTACK = 0x11, REQDID = 0x00, DEVICEID = 0x01, REQPRODUCTCODE = 0x1C,
                PRODUCTCODE = 0x1D, REQFWREV = 0x12, FIRMWAREREV = 0x13, ERROR = 0x42, WARNING = 0x43, OUTPUTCONFIGURATION = 0xC1, MTDATA2 = 0x36,
                REQFILTERPROFILE = 0x64, REQFILTERPROFILEACK = 0x65,
                BAUDRATE = 0x18, BAUDRATEACK = 0x19,
                // Alignment rotation (MID 0xEC/0xED)
                SETALIGNMENTROTATION = 0xEC, ALIGNMENTROTATION = 0xED,
                // Reset orientation (MID 0xA4/0xA5)
                RESETORIENTATION = 0xA4, RESETORIENTATIONACK = 0xA5,
                // Restore factory defaults (MID 0x0E/0x0F)
                RESTOREFACTORYDEF = 0x0E
               };

    enum DataID {
      UTCTIME = 0x1010,
      SAMPLE_TIME_FINE = 0x1060,
      ACCELERATION        = 0x4020,
      FREEACCELERATION    = 0x4030,
      RATEOFTURN          = 0x8020,
      QUATERNION          = 0x2010,
      ROTATION            = 0x2020,
      EULERANGLES         = 0x2030,
      MAGNETOMETER        = 0xC020,
      LATLON              = 0x5040,
      DELTAQ              = 0x8030,
      DELTAV              = 0x4010,
      HR_ACC = 0x4040,
      HR_GYR = 0x8040,
    };

    void sendMessage(uint8_t *message, uint8_t numBytes);
    void readMessages();

    bool read() {
      readPipeStatus();
      if (notificationSize) {                                   //New notification message available to be read
        readPipeNotif();
        parseNotification(datanotif);
      }
      if (measurementSize) {                                    //New measurement packet available to be read
        readPipeMeas();
        parseMTData2(datameas, measurementSize);
        return true;                                            //Return true if new measurements were read
      } else {
        return false;
      }
    }

    bool readFast()
    {


        readPipeStatusFast();
        if (notificationSize) readPipeNotifFast();
        if (measurementSize) readPipeMeasFast();

        if (notificationSize) parseNotification(datanotif);
        if (measurementSize) parseMTData2(datameas, measurementSize);

        return measurementSize;
    }

    bool err = true;

    float HR_acc[3];                     //Used to store latest high rate acc
    float HR_gyr[3];                     //Used to store latest high rate gyro reading
    float acc[3];                     //Used to store latest Acceleration reading
    float freeAcc[3];                 //Used to store latest FreeAcceleration reading
    float gyr[3];                     //Used to store latest RateOfTurn reading
    float quat[4];                    //Used to store latest Quaternion reading
    float rot[9];                     //Used to store latest Rotation Matrix reading
    float euler[3];                   //Used to store latest EulerAngle reading
    float mag[3];                     //Used to store latest Magnetometer reading
    float latlon[2];                  //Used to store latest Latitude/Longitude reading
    float dq[4];                      //Used to store latest DeltaQ reading
    float dv[3];                      //Used to store latest DeltaV reading
    float alignQuat[4];               //Used to store latest Alignment Quaternion reading
    uint8_t alignParam;               //Used to store latest Alignment parameter
    bool alignQuatValid;              //True if the alignment quaternion is valid
    bool wakeUpReceived;              //True if a WakeUp message was received from the MTi
    bool configState;                 //True if MTi is in Config mode, false if MTi is in Measurement mode
    char productCode;                 //Contains the product code (MTi-#) of the connected device

    uint32_t sampleTimeFine;          //Used to store latest SampleTimeFine (10kHz counter)

    void parseMTData2(uint8_t *packet, uint16_t length);

  private:
    uint8_t address;
    #ifdef ARDUINO_ARCH_MBED
    PinName drdy;
    TwoWire& wire;
    #elif defined(ARDUINO_ARCH_RENESAS) || !defined(ARDUINO)
    uint8_t drdy;
    #endif

    volatile uint8_t errorCodeToDrain = 0xFF;
    volatile uint8_t warningCodeToDrain = 0xFF;



    void readPipeStatus();
    void readPipeNotif();
    void readPipeMeas();

    void readPipeStatusFast();
    void readPipeNotifFast();
    void readPipeMeasFast();

    void parse_print_clear();
    void parseNotification(uint8_t* notif);

    /**
     * @brief Copies data from source to destination while swapping the byte order of each 4-byte word.
     * This is used to convert from the big-endian format of the Xsens sensor to the little-endian
     * format of the target processor.
     *
     * @param dest Pointer to the destination array.
     * @param src Pointer to the source array.
     * @param n Number of bytes to copy. Must be a multiple of 4.
     */
    inline void memcpy_swap(void* dest, const void* src, size_t n) {
        uint8_t* d = static_cast<uint8_t*>(dest);
        const uint8_t* s = static_cast<const uint8_t*>(src);
        for (size_t i = 0; i < n / 4; ++i) {
            d[i * 4 + 0] = s[i * 4 + 3];
            d[i * 4 + 1] = s[i * 4 + 2];
            d[i * 4 + 2] = s[i * 4 + 1];
            d[i * 4 + 3] = s[i * 4 + 0];
        }
    }

    uint8_t status[4];                                            //Used to store indicators of the Status Pipe
    uint8_t datanotif[256];                                       //Used to store content read from the Notification Pipe
    uint8_t datameas[256];                                        //Used to store content read from the Measurement Pipe

    uint16_t notificationSize;
    uint16_t measurementSize;
};

}

inline void xsens::Xbus::parseMTData2(uint8_t *packet, uint16_t length)
{
        if (length < 4) return;      // sanity check

        uint8_t *ptr = packet;
        uint8_t *end = packet + length;

        /* Skip the MTData2 header (0x36 LEN) if present */
        if (*ptr == (uint8_t)MesID::MTDATA2) {
            uint8_t blockLen = *(ptr + 1);
            ptr += 2;                         // move past header
            end = ptr + blockLen;             // enforce header LEN
            if (end > packet + length) end = packet + length; // guard
        }

        while (ptr + 3 <= end) {              // need at least idHi idLo LEN
            uint16_t dataId = ((uint16_t)ptr[0] << 8) | ptr[1];
            uint8_t  len    = ptr[2];
            uint8_t *payload = ptr + 3;

            if (payload + len > end) break;   // malformed frame guard

            switch (dataId) {
                case (uint16_t)DataID::SAMPLE_TIME_FINE:
                    // 32-bit unsigned integer, big-endian
                    sampleTimeFine = ((uint32_t)payload[0] << 24) |
                                     ((uint32_t)payload[1] << 16) |
                                     ((uint32_t)payload[2] << 8)  |
                                     ((uint32_t)payload[3]);
                    break;

                case (uint16_t)DataID::HR_ACC:
                    memcpy_swap(HR_acc, payload, 12);
                    break;

                case (uint16_t)DataID::HR_GYR:
                    memcpy_swap(HR_gyr, payload, 12);
                    break;

                case (uint16_t)DataID::ACCELERATION:
                    memcpy_swap(acc, payload, 12);
                    break;

                case (uint16_t)DataID::FREEACCELERATION:
                    memcpy_swap(freeAcc, payload, 12);
                    break;

                case (uint16_t)DataID::RATEOFTURN:
                    memcpy_swap(gyr, payload, 12);
                    break;

                case (uint16_t)DataID::QUATERNION:
                    memcpy_swap(quat, payload, 16);
                    break;

                case (uint16_t)DataID::ROTATION:
                    memcpy_swap(rot, payload, 36);
                    break;

                case (uint16_t)DataID::EULERANGLES:
                    memcpy_swap(euler, payload, 12);
                    break;

                case (uint16_t)DataID::MAGNETOMETER:
                    memcpy_swap(mag, payload, 12);
                    break;

                case (uint16_t)DataID::LATLON:
                    memcpy_swap(latlon, payload, 8);
                    break;

                case (uint16_t)DataID::DELTAQ:
                    memcpy_swap(dq, payload, 16);
                    break;

                case (uint16_t)DataID::DELTAV:
                    memcpy_swap(dv, payload, 12);
                    break;

                default:
                    break;
            }

            ptr = payload + len;              // advance to the next block
        }
}

    inline void xsens::Xbus::parse_print_clear() {
        uint8_t error, warning;

        //noInterrupts();
        error = errorCodeToDrain;
        warning = warningCodeToDrain;
        errorCodeToDrain = 0xFF;
        warningCodeToDrain = 0xFF;
        //interrupts();

        if (error != 0xFF) {
            portable::print("Received an error with code: "); portable::printLine(error);
        }

        if (warning != 0xFF) {
            portable::print("Received an warning with code: "); portable::printLine(warning);
        }
}

inline void xsens::Xbus::parseNotification(uint8_t* notif) {                                           //Parse the most common notification messages
      uint8_t notifID = notif[0];
      switch (notifID) {
        case (uint8_t)MesID::WAKEUP: {
            Serial.println("Received WakeUp message.");
            wakeUpReceived = true;
            break;
          }
        case (uint8_t)MesID::RESETACK: {
            Serial.println("Received ResetACK message.");
            break;
          }
        case (uint8_t)MesID::ERROR: {
            //Serial.println("ERROR");
            errorCodeToDrain = notif[2];
            break;
          }
        case (uint8_t)MesID::WARNING: {
            uint32_t warn = (uint32_t)notif[5] | ((uint32_t)notif[4] << 8);
            warningCodeToDrain = warn;
            break;
          }
        case (uint8_t)MesID::PRODUCTCODE: {
            portable::print("Product code is: ");
            for (int i = 2; i < notificationSize - 1; i++) {
                portable::print(char(notif[i]));
            }
            portable::printLine();
            productCode = char(notif[6]);                                               //Store the product code (MTi-#) for later usage
            break;
          }
        case (uint8_t)MesID::FIRMWAREREV: {
            portable::print("Firmware version is: ");
            portable::print(notif[2]); portable::print("."); portable::print(notif[3]); portable::print("."); portable::printLine(notif[4]);
            break;
          }
        case (uint8_t)MesID::GOTOCONFIGACK: {
            Serial.println("Received GoToConfigACK.");
            configState = true;
            break;
          }
        case (uint8_t)MesID::GOTOMEASUREMENTACK: {
            //Serial.println("Received GoToMeasurementACK.");
            configState = false;
            break;
          }
        case (uint8_t)MesID::OUTPUTCONFIGURATION: {
            //Serial.println("Received SetOutputConfigurationACK.");
            break;
          }
        case (uint8_t)MesID::REQFILTERPROFILEACK: {
            // MID 0x65 is used for both Req and Set acknowledgments
            // notif[0] = 0x65 (MID)
            // notif[1] = LEN
            // For ReqFilterProfile response: LEN >= 2, notif[2-3] = profile ID (big-endian)
            // For SetFilterProfile response: LEN may be 0 or contain status
            
            uint8_t len = notif[1];
            if (len >= 2) {
                // Response to ReqFilterProfile - contains profile ID
                uint8_t profID = notif[3];  // LSB of profile ID (MSB is notif[2])
                const char* profName;
                switch (profID) {
                  // MTI-1 series profiles
                  case 50: profName = "General";            break;
                  case 51: profName = "High_mag_dep";       break;
                  case 52: profName = "Dynamic";            break;
                  case 53: profName = "North_reference";    break;
                  case 54: profName = "VRU_general";        break;
                  default: profName = "UnknownProfile";     break;
                }

                portable::print("Active filter-profile ID: ");
                portable::print(profID);
                portable::print(" → ");
                portable::printLine(profName);
            } else {
                // Response to SetFilterProfile - just acknowledge
                portable::printLine("Filter profile set successfully");
            }
            break;
          }
        case (uint8_t)MesID::BAUDRATEACK: {
            // notif[0] = 0x19
            // notif[1] = LEN (should be 0x01 for a request or 0x00 after a set)
            // notif[2] = baudCode (only if LEN == 1)

            uint8_t len = notif[1];
            if (len == 1) {
              uint8_t code = notif[2];
              uint32_t baud;
              switch (code) {
                case 0x09: baud = 9600;   break;
                case 0x07: baud = 19200;  break;
                case 0x05: baud = 38400;  break;
                case 0x04: baud = 57600;  break;
                case 0x02: baud = 115200; break;
                case 0x01: baud = 230400; break;
                case 0x00: baud = 460800; break;
                case 0x0A: baud = 921600; break;
                default:   baud = 0;      break;
              }
              if (baud) {
                portable::print("Current baudrate: ");
                portable::printLine(baud);
              } else {
                portable::print("Unknown baud-code 0x");
                if (code < 0x10) portable::print('0');
                portable::printLine(code, 16);
              }
            } else {
              portable::printLine("SetBaudRate acknowledged.");
            }
            break;
          }
        case (uint8_t)MesID::ALIGNMENTROTATION: {
            // Response to ReqAlignmentRotation or SetAlignmentRotation (MID 0xED)
            // notif[0] = 0xED
            // notif[1] = LEN (17 for response with quaternion, 0 for Set ACK)
            // notif[2] = parameter (0=RotSensor, 1=RotLocal)
            // notif[3-6] = q0 (w), notif[7-10] = q1 (x), etc. (big-endian floats)
            
            uint8_t len = notif[1];
            if (len == 0x11) {  // 17 bytes = response with quaternion
              uint8_t param = notif[2];
              
              // Extract quaternion (big-endian to float)
              for (int i = 0; i < 4; i++) {
                uint8_t* ptr = &notif[3 + i * 4];
                uint8_t tmp[4] = { ptr[3], ptr[2], ptr[1], ptr[0] };  // swap to little-endian
                memcpy(&alignQuat[i], tmp, 4);
              }
              alignParam = param;
              alignQuatValid = true;
              
              Serial.print("Alignment Rotation (");
              Serial.print(param == 0 ? "RotSensor" : "RotLocal");
              Serial.print("): q=[");
              Serial.print(alignQuat[0], 4); Serial.print(", ");
              Serial.print(alignQuat[1], 4); Serial.print(", ");
              Serial.print(alignQuat[2], 4); Serial.print(", ");
              Serial.print(alignQuat[3], 4); Serial.println("]");
            } else if (len == 0) {
              Serial.println("SetAlignmentRotation acknowledged.");
            } else {
              Serial.print("AlignmentRotation response, len="); Serial.println(len);
            }
            break;
          }
        case (uint8_t)MesID::RESETORIENTATIONACK: {
            // Response to ResetOrientation (MID 0xA5)
            Serial.println("Reset Orientation acknowledged.");
            break;
          }
        default: {
            portable::print("Received undefined notification: ");
            for (int i = 0; i < notificationSize - 1; i++) {
              portable::print(notif[i], 16); portable::print(" ");
            }
            portable::printLine();
            break;
          }
      }
    }
