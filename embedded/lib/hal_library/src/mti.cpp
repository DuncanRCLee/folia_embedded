#ifdef ARDUINO
#include <Arduino.h>
#undef abs
#undef min
#undef max

#include <cmath>
#include "mti.hpp"

namespace xsens {

bool MTi::detect(uint16_t timeout) {
  //Send goToConfig messages until goToConfigAck is received from the device
  Serial.println("Scanning for MTi.");                                                       //Clear the measurement/notification pipes (without printing) before configuring.
  long int starttime = millis();
  while ((millis() - starttime) < timeout) {
    goToConfig();
    delay(250);
    Serial.println("Reading messages...");

    xbus.readMessages();
    if (deviceInConfigMode()) {
      Serial.println("Device detected.");
      return true;
    }
  }
  Serial.println("Failed to detect device.");
  return false;
}

void MTi::requestDeviceInfo() {
  //Request device info from the MTi using Xbus commands. Refer to the MT Low Level Communication Protocol Document for more information on the commands used here:
  //https://mtidocs.xsens.com/mt-low-level-communication-protocol-documentation

  if (!deviceInConfigMode()) {
    Serial.println("Cannot request device info. Device is not in Config Mode.");
    return;
  }
  xbus.readMessages();                                                                             //Clear the measurement/notification pipes before configuring.
  Serial.println("Requesting device info...");

  uint8_t reqProductCode[] = {0x1C, 0x00};                                                    //reqProductCode Xbus message
  xbus.sendMessage(reqProductCode, sizeof(reqProductCode));
  delay(1000);

  xbus.readMessages();

  uint8_t reqFWRev[] = {0x12, 0x00};                                                          //reqFWRev Xbus message
  xbus.sendMessage(reqFWRev, sizeof(reqFWRev));
  delay(1000);

  xbus.readMessages();
}

void MTi::getFilterProfile() {
  if (!deviceInConfigMode()) {
    Serial.println("Must be in Config Mode to query filter profile.");
    return;
  }

  uint8_t reqFilterProfile[] = { 0x64, 0x00 };
  xbus.sendMessage(reqFilterProfile, sizeof(reqFilterProfile));
  delay(1000);

  xbus.readMessages();
}

void MTi::setFilterProfile(uint8_t profileId) {
  if (!deviceInConfigMode()) {
    Serial.println("Must be in Config Mode to set filter profile.");
    return;
  }

  // MID = SetFilterProfile (0x64), LEN = 0x02, profileId (2 bytes, MSB first)
  uint8_t setFilterProfile[] = { 
    0x64,           // SetFilterProfile MID (not 0x64 which is ReqFilterProfile)
    0x02, 
    0x00,           // MSB (profile IDs are < 256, so MSB is 0)
    profileId       // LSB
  };
  xbus.sendMessage(setFilterProfile, sizeof(setFilterProfile));
  
  Serial.print("Setting filter profile to ID: ");
  Serial.println(profileId);
  
  delay(1000);
  xbus.readMessages();
  
  // Verify the change by querying again
  delay(100);
  getFilterProfile();
}

void MTi::getBaudRate() {
  if (!deviceInConfigMode()) {
    Serial.println("Must be in Config Mode to request baudrate.");
    return;
  }

  // MID = BAUDRATE (0x18), LEN = 0x00
  uint8_t getBaud[] = { static_cast<uint8_t>(Xbus::MesID::BAUDRATE), 0x00 };
  xbus.sendMessage(getBaud, sizeof(getBaud));
  delay(1000);

  xbus.readMessages();
}

void MTi::setBaudRate(uint32_t baudrate) {
  uint8_t baudCode = 0x02;                //Default baudrate is 115200
  switch (baudrate) {
    case 9600:   baudCode = 0x09; break;
    case 19200:  baudCode = 0x07; break;
    case 38400:  baudCode = 0x05; break;
    case 57600:  baudCode = 0x04; break;
    case 115200: baudCode = 0x02; break;
    case 230400: baudCode = 0x01; break;
    case 460800: baudCode = 0x00; break;
    case 921600: baudCode = 0x0A; break;
    default:
      Serial.println("Invalid Baudrate. Defaulting to 115200.");
      baudCode = 0x02;
      break;
  }

  if (!deviceInConfigMode()) {
    Serial.println("Must be in Config Mode to set baudrate.");
    return;
  }

  // MID = BAUDRATE (0x18), LEN = 0x01, payload = baudCode
  uint8_t setBaud[] = {
    static_cast<uint8_t>(Xbus::MesID::BAUDRATE),
    0x01,
    baudCode
  };
  xbus.sendMessage(setBaud, sizeof(setBaud));
  delay(1000);

  xbus.readMessages();
}

void MTi::configureOutputs() {
  //Configure the outputs of the MTi using Xbus commands. Refer to the MT Low Level Communication Protocol Document for more information on the commands used here:
  //https://mtidocs.xsens.com/mt-low-level-communication-protocol-documentation

  if (!deviceInConfigMode()) {
    Serial.println("Cannot configure device. Device is not in Config Mode.");
    return;
  }
  xbus.readMessages();                                                                             //Clear the measurement/notification pipes (without printing) before configuring.

  Serial.println("Configuring Acc, FreeAcc, RateOfTurn & Quaternion @100Hz");
  uint8_t outputConfig[] = {
    // SetOutputConfiguration (0xC0), Data packet size in Hex (N queries x 4 bytes) (0x14 = 20 bytes)
    0xC0, 0x14,
    // Acceleration       (XDI_Acceleration      = 0x4020) @100 Hz
    0x40, 0x20, 0x00, 0x64,
    // Free acceleration  (XDI_FreeAcceleration  = 0x4030) @100 Hz
    0x40, 0x30, 0x00, 0x64,
    // Angular rate       (XDI_RateOfTurn        = 0x8020) @100 Hz
    0x80, 0x20, 0x00, 0x64,
    // EulerAngles        (XDI_Euler             = 0x2030) @100 Hz
    0x20, 0x30, 0x00, 0x64,
    // Quaternion         (XDI_Quaternion        = 0x8030) @100 Hz
    0x20, 0x10, 0x00, 0x64,

    0x80, 0x30, 0x00, 0x64, //dv dq
    0x40, 0x10, 0x00, 0x64
  };
  xbus.sendMessage(outputConfig, sizeof(outputConfig));
  Serial.println("Configured Acc, FreeAcc, RateOfTurn, dv, dq & Quaternion @100Hz");
  delay(1000);
  xbus.readMessages();
}


void MTi::configureOutputsAdv(const std::vector<OutputConfigEntry>& outputs) {
  if (!deviceInConfigMode()) {
    Serial.println("Cannot configure device. Device is not in Config Mode.");
    return;
  }
  xbus.readMessages();

  Serial.println("Configuring with advanced parameters.");

  // Calculate the length of the payload
  size_t payloadBytes = outputs.size() * 4; // 4 bytes for each entry [ID_MSB, ID_LSB, RATE_MSB, RATE_LSB]
  std::vector<uint8_t> configMsg;
  configMsg.reserve(2 + payloadBytes);

  configMsg.push_back(0xC0); // SetOutputConfiguration command
  configMsg.push_back(static_cast<uint8_t>(payloadBytes));

  for (const auto& entry : outputs) {
    // Append ID and rate in big-endian format
    configMsg.push_back(static_cast<uint8_t>((entry.id >> 8) & 0xFF));
    configMsg.push_back(static_cast<uint8_t>(entry.id & 0xFF));
    configMsg.push_back(static_cast<uint8_t>((entry.rateHz >> 8) & 0xFF));
    configMsg.push_back(static_cast<uint8_t>(entry.rateHz & 0xFF));
  }

  xbus.sendMessage(configMsg.data(), configMsg.size());
  Serial.println("Advanced configuration sent.");

  // Let the MTi process it
  delay(1000);
  xbus.readMessages();
}

void MTi::configureOutputs(const std::vector<uint16_t>& outputs) {
  if (!deviceInConfigMode()) {
    Serial.println("Cannot configure device. Device is not in Config Mode.");
    return;
  }
  xbus.readMessages();

  Serial.println("Configuring");

  // build the Xbus payload:
  //   0xC0 (SetOutputConfiguration), length,
  //   then for each output: [ID_MSB, ID_LSB, RATE_MSB, RATE_LSB]
  const uint16_t rateHz = 0x0064;    // 100 decimal == 0x0064
  size_t       payloadBytes = outputs.size() * 4;
  uint8_t      length       = static_cast<uint8_t>(payloadBytes);

  std::vector<uint8_t> msg;
  msg.reserve(2 + payloadBytes);
  msg.push_back(0xC0);
  msg.push_back(length);

  for (auto id : outputs) {

    // ID big-endian
    msg.push_back(static_cast<uint8_t>( (id >> 8) & 0xFF ));
    msg.push_back(static_cast<uint8_t>(  id        & 0xFF ));
    // rate = 100Hz big-endian
    msg.push_back(static_cast<uint8_t>( (rateHz >> 8) & 0xFF ));
    msg.push_back(static_cast<uint8_t>(  rateHz        & 0xFF ));
  }

  // send it off
  xbus.sendMessage(msg.data(), msg.size());
  Serial.println("Configuration sent.");

  // let the MTi process it
  delay(1000);
  xbus.readMessages();

}


void MTi::goToConfig() {
  Serial.println("Entering configuration mode.");
  uint8_t goToConfig[] = {0x30, 0x00};                                                        //goToConfig Xbus message
  xbus.sendMessage(goToConfig, sizeof(goToConfig));
}


void MTi::goToMeasurement() {
  Serial.println("Entering measurement mode.");
  uint8_t goToMeas[] = {0x10, 0x00};                                                          //goToMeasurement Xbus message
  xbus.sendMessage(goToMeas, sizeof(goToMeas));
}

static void floatToBigEndian(float value, uint8_t* bytes) {
  union {
    float f;
    uint8_t b[4];
  } conv;
  conv.f = value;
  // Swap to big-endian (network byte order)
  bytes[0] = conv.b[3];
  bytes[1] = conv.b[2];
  bytes[2] = conv.b[1];
  bytes[3] = conv.b[0];
}

// Request current alignment rotation
// MID = 0xEC, LEN = 0x01, DATA = parameter (0 = RotSensor, 1 = RotLocal)
void MTi::reqAlignmentRotation(uint8_t parameter) {

  if (!deviceInConfigMode()) {
    Serial.println("Must be in Config Mode to request alignment rotation.");
    return;
  }

  uint8_t reqAlignment[] = {
    static_cast<uint8_t>(Xbus::MesID::SETALIGNMENTROTATION),  // 0xEC
    0x01,                                                      // Length
    parameter                                                  // 0=RotSensor, 1=RotLocal
  };
  
  Serial.print("Requesting alignment rotation (");
  Serial.print(parameter == 0 ? "RotSensor" : "RotLocal");
  Serial.println(")...");
  
  xbus.sendMessage(reqAlignment, sizeof(reqAlignment));
  delay(500);
  readMessages();
}

void MTi::setAlignmentRotation(uint8_t parameter, float q0, float q1, float q2, float q3) {
  if (!deviceInConfigMode()) {
    Serial.println("Must be in Config Mode to set alignment rotation.");
    return;
  }

  // MID = SetAlignmentRotation (0xEC), LEN = 0x11 (17 bytes)
  // DATA = parameter (1 byte) + quaternion (4 floats, 16 bytes, big-endian)
  uint8_t setAlignment[20];  // 2 header + 1 param + 16 quat + 1 checksum (handled by sendMessage)
  setAlignment[0] = static_cast<uint8_t>(Xbus::MesID::SETALIGNMENTROTATION);  // 0xEC
  setAlignment[1] = 0x11;  // Length = 17 bytes
  setAlignment[2] = parameter;  // 0=RotSensor, 1=RotLocal
  
  // Convert quaternion components to big-endian
  floatToBigEndian(q0, &setAlignment[3]);
  floatToBigEndian(q1, &setAlignment[7]);
  floatToBigEndian(q2, &setAlignment[11]);
  floatToBigEndian(q3, &setAlignment[15]);
  
  Serial.print("Setting alignment rotation (");
  Serial.print(parameter == 0 ? "RotSensor" : "RotLocal");
  Serial.print("): q=[");
  Serial.print(q0, 4); Serial.print(", ");
  Serial.print(q1, 4); Serial.print(", ");
  Serial.print(q2, 4); Serial.print(", ");
  Serial.print(q3, 4); Serial.println("]");
  
  xbus.sendMessage(setAlignment, 19);  // 19 bytes (checksum added by sendMessage)
  delay(500);
  readMessages();
}

bool MTi::alignmentMatches(uint8_t param, float q0, float q1, float q2, float q3, float tolerance) {
  if (!xbus.alignQuatValid || xbus.alignParam != param) {
    return false;
  }
  
  // Check if each component is within tolerance
  bool matches = (fabs(xbus.alignQuat[0] - q0) <= tolerance) &&
                 (fabs(xbus.alignQuat[1] - q1) <= tolerance) &&
                 (fabs(xbus.alignQuat[2] - q2) <= tolerance) &&
                 (fabs(xbus.alignQuat[3] - q3) <= tolerance);
  
  return matches;
}

void MTi::resetOrientation(uint8_t resetCode) {
  // Reset orientation according to code
  // MID = 0xA4, LEN = 0x02, DATA = resetCode (2 bytes, MSB first)
  // Reset codes from Table 33:
  //   0x0000 (0): Store current settings (only in config mode)
  //   0x0001 (1): Heading reset (NOT supported by GNSS/INS)
  //   0x0003 (3): Object or inclination reset
  //   0x0004 (4): Alignment reset (heading and inclination)
  //   0x0005 (5): Default heading
  //   0x0006 (6): Default inclination
  //   0x0007 (7): Default alignment

  // Code 0x00 (Store current settings) requires Config Mode
  if (resetCode == 0x00 && !deviceInConfigMode()) {
    Serial.println("Cannot store orientation settings. Device must be in Config Mode for resetCode 0x00.");
    return;
  }

  uint8_t resetOri[] = {
    static_cast<uint8_t>(Xbus::MesID::RESETORIENTATION),  // 0xA4
    0x02,       // Length = 2 bytes
    0x00,       // MSB (codes are < 256)
    resetCode   // LSB
  };

  const char* codeName;
  switch (resetCode) {
    case 0x00: codeName = "Store current settings"; break;
    case 0x01: codeName = "Heading reset"; break;
    case 0x03: codeName = "Object or inclination reset"; break;
    case 0x04: codeName = "Alignment reset (heading+inclination)"; break;
    case 0x05: codeName = "Default heading"; break;
    case 0x06: codeName = "Default inclination"; break;
    case 0x07: codeName = "Default Alignment"; break;
    default:   codeName = "Unknown"; break;
  }

  Serial.print("Resetting orientation (");
  Serial.print(codeName);
  Serial.println(")...");

  xbus.sendMessage(resetOri, sizeof(resetOri));
  delay(500);
  readMessages();
}


void MTi::resetHeading() {
  // Reset heading to default
  resetOrientation(0x01);
}

void MTi::resetInclination() {
  // Reset inclination to default
  resetOrientation(0x03);
}

void MTi::resetAlignment() {
  // Reset alignment to default (clears custom RotSensor/RotLocal)
  resetOrientation(0x04);
}

void MTi::defaultHeading() {
  // Reset heading to default factory setting
  resetOrientation(0x05);
}

void MTi::defaultInclination() {
  // Reset inclination to default factory setting
  resetOrientation(0x06);
}

void MTi::defaultAlignment() {
  // Reset alignment to default factory setting
  resetOrientation(0x07);
}

void MTi::restoreFactoryDefaults() {
  // Restore all device settings to factory defaults
  // MID = 0x0E, LEN = 0x00
  // This clears all custom settings including alignment, output config, etc.
  // Must be in Config Mode
  
  if (!deviceInConfigMode()) {
    Serial.println("Must be in Config Mode to restore factory defaults.");
    return;
  }

  uint8_t restoreFactory[] = {
    static_cast<uint8_t>(Xbus::MesID::RESTOREFACTORYDEF),  // 0x0E
    0x00  // No data
  };
  
  Serial.println("Restoring factory defaults...");
  Serial.println("Warning: This will clear ALL custom settings!");
  
  xbus.sendMessage(restoreFactory, sizeof(restoreFactory));
  delay(1000);  // Give device time to process
  readMessages();
  
  Serial.println("Factory defaults restored. Device may need reconfiguration.");
}

void MTi::reset() {
  // Send MID 0x40 (Reset) with no data to reactivate WakeUp procedure
  // WakeUp procedure:
  //   1. Host sends Reset (0x40)
  //   2. Device sends WakeUp (0x3E)
  //   3. Host must send WakeUpAck (0x3F) within 500ms to enter Config State
  //   4. If no WakeUpAck, device enters Measurement State (if DisableAutoMeasurement not set)
  
  uint8_t resetMsg[] = { 
    static_cast<uint8_t>(Xbus::MesID::RESET),  // 0x40
    0x00  // No data
  };
  
  Serial.println("Sending Reset (MID 0x40) to reactivate WakeUp procedure...");
  xbus.sendMessage(resetMsg, sizeof(resetMsg));
  
  // Wait for WakeUp message from device (must respond within 500ms)
  xbus.wakeUpReceived = false;
  unsigned long startTime = millis();
  unsigned long timeout = 500;  // 500ms timeout to detect WakeUp
  
  while (!xbus.wakeUpReceived && (millis() - startTime) < timeout) {
    readMessages();
    delay(10);  // Small delay to prevent busy-waiting
  }
  
  if (xbus.wakeUpReceived) {
    // Send WakeUpAck within 500ms to enter Config State
    uint8_t wakeUpAck[] = {
      static_cast<uint8_t>(Xbus::MesID::WAKEUPACK),  // 0x3F
      0x00  // No data
    };
    
    Serial.println("Sending WakeUpAck (MID 0x3F) to enter Config State...");
    xbus.sendMessage(wakeUpAck, sizeof(wakeUpAck));
    delay(100);
    
    // Read any remaining messages (ResetAck, etc.)
    readMessages();
    
    // Device should now be in Config State
    xbus.configState = true;
    Serial.println("Device in Config State after reset.");
  } else {
    Serial.println("Warning: WakeUp message not received within timeout!");
    Serial.println("Device may have entered Measurement State.");
    xbus.configState = false;
  }
}

}

#endif // ARDUINO
