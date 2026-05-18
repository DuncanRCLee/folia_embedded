#include "xbus.hpp"
#if !defined(ARDUINO)

namespace xsens {

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

// Stub implementations for test builds
void Xbus::readMessages() {}
void Xbus::readPipeNotif() {}
void Xbus::readPipeNotifFast() {}
void Xbus::readPipeStatus() {
  notificationSize = 0;
  measurementSize = 0;
}
void Xbus::readPipeStatusFast() {
  notificationSize = 0;
  measurementSize = 0;
}
void Xbus::readPipeMeas() {}
void Xbus::readPipeMeasFast() {}
void Xbus::sendMessage(uint8_t *message, uint8_t numBytes) {
  (void)message;
  (void)numBytes;
}
// parse_print_clear and parseNotification are inline in header

}
#endif
