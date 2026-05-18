#pragma once

// Minimal platform type declarations for HAL headers
// This avoids pulling in full Arduino.h just for type definitions

#ifdef ARDUINO_ARCH_MBED
  // Forward declare MBED types without including full Arduino.h
  // These are defined in the actual MBED framework
  class TwoWire;
  enum PinName : int;
#endif

#ifdef ARDUINO_ARCH_RENESAS
  // Renesas/UNO R4 typically uses standard int types
  // No special forward declarations needed
#endif

// For native/test builds, provide minimal definitions if needed
#if !defined(ARDUINO)
  // Native builds don't need these platform types
#endif
