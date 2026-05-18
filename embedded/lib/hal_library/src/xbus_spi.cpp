#ifdef ARDUINO_ARCH_RENESAS
#include <Arduino.h>
#undef abs
#undef min
#undef max

#include "xbus.hpp"
#include "api/HardwareSPI.h"
#include <SPI.h>
#include "digitalwritefast_unor4.h"

namespace xsens {

inline void csLow(uint8_t pin)  { digitalWriteFast(pin, LOW);  }
inline void csHigh(uint8_t pin) { digitalWriteFast(pin, HIGH); }

Xbus::Xbus(uint8_t address, uint8_t drdy) : address(address), drdy(drdy) {
  for (int i = 0; i < 3; ++i) {
    acc[i]     = NAN;
    freeAcc[i] = NAN;
    gyr[i]     = NAN;
    euler[i]   = NAN;
    mag[i]     = NAN;
    dv[i]      = NAN;
    HR_acc[i]  = NAN;
    HR_gyr[i]  = NAN;
  }
  for (int i = 0; i < 2; ++i) {
    latlon[i] = NAN;
  }
  for (int i = 0; i < 4; ++i) {
    quat[i] = NAN;
    dq[i]   = NAN;
  }
  for (int i = 0; i < 9; ++i) {
    rot[i] = NAN;
  }
}

void Xbus::readMessages() {
    while (digitalRead(drdy)) {
      read();
    }
}

void Xbus::readPipeNotif() {
  uint8_t buffer[] = {XSENS_NOTIF_PIPE, 0xFF, 0xFF, 0xFF};
  SPI.beginTransaction(SPISettings(2000000, MSBFIRST, SPI_MODE3));
  digitalWrite(address, LOW);
  SPI.transfer(buffer, sizeof(buffer));
  for (int i = 0; i < notificationSize; i++) {
    datanotif[i] = SPI.transfer(0xFF);
  }
  digitalWrite(address, HIGH);
  SPI.endTransaction();
}


inline void Xbus::readPipeNotifFast()
{
    uint8_t cmd[4] = { XSENS_NOTIF_PIPE, 0xFF, 0xFF, 0xFF };

    csLow(address);
    SPI.transfer(cmd, sizeof(cmd));                                // select pipe
    memset(datanotif, 0xFF, notificationSize);        // preload dummies
    SPI.transfer(datanotif, notificationSize);        // clock payload
    csHigh(address);

}


void Xbus::readPipeStatus() {
  uint8_t buffer[] = {XSENS_STATUS_PIPE, 0xFF, 0xFF, 0xFF};
  SPI.beginTransaction(SPISettings(2000000, MSBFIRST, SPI_MODE3));
  digitalWrite(address, LOW);
  SPI.transfer(buffer, sizeof(buffer));
  for (int i = 0; i < 4; i++) {
    status[i] = SPI.transfer(0xFF);
  }
  digitalWrite(address, HIGH);
  SPI.endTransaction();

  notificationSize = (uint16_t)status[0] | ((uint16_t)status[1] << 8);
  measurementSize = (uint16_t)status[2] | ((uint16_t)status[3] << 8);
}


inline void Xbus::readPipeStatusFast()
{

    uint8_t cmd[] = { XSENS_STATUS_PIPE, 0xFF, 0xFF, 0xFF };
    memset(status, 0xFF, 4);

    csLow(address);
    SPI.transfer(cmd, 4);
    SPI.transfer(status, 4);
    csHigh(address);

    notificationSize = (uint16_t)status[0] | ((uint16_t)status[1] << 8);
    measurementSize  = (uint16_t)status[2] | ((uint16_t)status[3] << 8);
}


void Xbus::readPipeMeas() {
  uint8_t buffer[] = {XSENS_MEAS_PIPE, 0xFF, 0xFF, 0xFF};
  SPI.beginTransaction(SPISettings(2000000, MSBFIRST, SPI_MODE3));
  digitalWrite(address, LOW);
  SPI.transfer(buffer, sizeof(buffer));
  for (int i = 0; i < measurementSize; i++) {
    datameas[i] = SPI.transfer(0xFF);
  }
  digitalWrite(address, HIGH);
  SPI.endTransaction();
}


inline void Xbus::readPipeMeasFast()
{
    uint8_t cmd[] = { XSENS_MEAS_PIPE, 0xFF, 0xFF, 0xFF };

    csLow(address);
    SPI.transfer(cmd, 4);
    memset(datameas, 0xFF, measurementSize);
    SPI.transfer(datameas, measurementSize);
    csHigh(address);

}

/* dataswapendian() and parseMTData2() unchanged */

void Xbus::sendMessage(uint8_t *message, uint8_t numBytes) {
    //Compute the checksum for the Xbus message to be sent. See https://mtidocs.xsens.com/messages for details.
    uint8_t checksum = 0x01;
    for (int i = 0; i < numBytes; i++) {
      checksum -= message[i];
    }
    message[numBytes] = checksum;                                                                     //Add the checksum at the end of the Xbus message

    uint8_t buffer[] = {XSENS_CONTROL_PIPE, 0xFF, 0xFF, 0xFF};

    SPI.beginTransaction(SPISettings(2000000, MSBFIRST, SPI_MODE3));
    digitalWrite(address, LOW);
    SPI.transfer(buffer, sizeof(buffer));
    SPI.transfer(message, numBytes + 1);
    digitalWrite(address, HIGH);
    SPI.endTransaction();
}

}
#endif
