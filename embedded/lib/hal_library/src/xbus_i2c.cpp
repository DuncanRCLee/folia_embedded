#ifdef ARDUINO_ARCH_MBED
#include <Arduino.h>
#undef abs
#undef min
#undef max

#include "xbus.hpp"

namespace xsens {

 Xbus::Xbus(uint8_t address, TwoWire& wire, PinName drdy) : address(address), drdy(drdy), wire(wire) {
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
  alignQuatValid = false;
  alignParam = 0;
  wakeUpReceived = false;
}

void Xbus::readMessages() {
    while (digitalRead(drdy)) {
      read();
    }
}

void Xbus::readPipeNotif() {
  if (notificationSize == 0) return;
  
  wire.beginTransmission(address);
  wire.write(XSENS_NOTIF_PIPE);
  wire.endTransmission();

  // Bulk read entire notification packet
  size_t bytesRead = wire.requestFrom(address, notificationSize);
  if (bytesRead == notificationSize) {
    wire.readBytes((char*)datanotif, notificationSize);
  }
}

inline void Xbus::readPipeNotifFast() {
    if (notificationSize == 0) return;
    
    wire.beginTransmission(address);
    wire.write(XSENS_NOTIF_PIPE);
    wire.endTransmission(false);
    
    size_t bytesRead = wire.requestFrom(address, notificationSize);
    if (bytesRead == notificationSize) {
      wire.readBytes((char*)datanotif, notificationSize);
    }
}

void Xbus::readPipeStatus() {
  wire.beginTransmission(address);
  wire.write(XSENS_STATUS_PIPE);
  wire.endTransmission();

  size_t bytesRead = wire.requestFrom(address, uint8_t(4));
  if (bytesRead == 4) {
    wire.readBytes((char*)status, 4);
  }

  notificationSize = (uint16_t)status[0] | ((uint16_t)status[1] << 8);
  measurementSize = (uint16_t)status[2] | ((uint16_t)status[3] << 8);
}

inline void Xbus::readPipeStatusFast() {
    wire.beginTransmission(address);
    wire.write(XSENS_STATUS_PIPE);
    wire.endTransmission(false);
    
    size_t bytesRead = wire.requestFrom(address, uint8_t(4));
    if (bytesRead == 4) {
      wire.readBytes((char*)status, 4);
    }
    
    notificationSize = (uint16_t)status[0] | ((uint16_t)status[1] << 8);
    measurementSize = (uint16_t)status[2] | ((uint16_t)status[3] << 8);
}

void Xbus::readPipeMeas() {
  if (measurementSize == 0) return;
  
  wire.beginTransmission(address);
  wire.write(XSENS_MEAS_PIPE);
  wire.endTransmission();

  size_t bytesRead = wire.requestFrom(address, measurementSize);
  if (bytesRead == measurementSize) {
    wire.readBytes((char*)datameas, measurementSize);
  }
}

inline void Xbus::readPipeMeasFast() {
    if (measurementSize == 0) return;
    
    wire.beginTransmission(address);
    wire.write(XSENS_MEAS_PIPE);
    wire.endTransmission(false);
    
    size_t bytesRead = wire.requestFrom(address, measurementSize);
    if (bytesRead == measurementSize) {
      wire.readBytes((char*)datameas, measurementSize);
    }
}

void Xbus::sendMessage(uint8_t *message, uint8_t numBytes) {

  //Compute the checksum for the Xbus message to be sent. See https://mtidocs.xsens.com/messages for details.
  uint8_t checksum = 0x01;
  for (int i = 0; i < numBytes; i++) {
    checksum -= message[i];
  }
  message[numBytes] = checksum;                                                                     //Add the checksum at the end of the Xbus message

  wire.beginTransmission(address);
  wire.write(XSENS_CONTROL_PIPE);                                                                   //Send the opcode before sending the Xbus command
  wire.write(message, numBytes + 1);
  wire.endTransmission();
}


}
#endif
