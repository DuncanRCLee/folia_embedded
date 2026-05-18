#pragma once

#if defined(ARDUINO)
  #include <Arduino.h>
  // Undefine Arduino macros that conflict with C++ standard library
  #undef abs
  #undef min
  #undef max
  #include <math.h>
  #include <string.h>
  #include "Wire.h"
#else
  #include <cstddef>
  #include <cstring>
  #include <cmath>
#endif

// Only include limits where needed, isolated from other headers
#include <limits>

namespace portable {

// ---- NaN ----
// Prefer a typed function instead of the macro NAN.
// Works on both Arduino and native builds.
template <typename T = float>
constexpr T NaN() {
#if defined(ARDUINO)
  // Arduino toolchains can lack <limits> specializations on some cores,
  // but quiet_NaN is generally available. Keep a single codepath if you like.
  return std::numeric_limits<T>::quiet_NaN();
#else
  return std::numeric_limits<T>::quiet_NaN();
#endif
}


inline bool isNaN(float x) {
#if defined(ARDUINO)
  return ::isnan(x);
#else
  using std::isnan;
  return isnan(x);
#endif
}
inline bool isNaN(double x) {
#if defined(ARDUINO)
  return ::isnan(x);
#else
  using std::isnan;
  return isnan(x);
#endif
}

inline void* memCopy(void* dst, const void* src, std::size_t n) {
#if defined(ARDUINO)
  return ::memcpy(dst, src, n);
#else
  return std::memcpy(dst, src, n);
#endif
}

}  // namespace portable

// ---- Portable Print Functions ----
#if !defined(ARDUINO)
  #ifndef _GLIBCXX_IOSTREAM
    #include <iostream>
  #endif
  #ifndef _GLIBCXX_IOMANIP
    #include <iomanip>
  #endif
  #include <bitset>
#endif

namespace portable {

template <typename T>
inline void print(const T& value) {
#if defined(ARDUINO)
  Serial.print(value);
#else
  std::cout << value;
#endif
}

template <typename T>
inline void printLine(const T& value) {
#if defined(ARDUINO)
  Serial.println(value);
#else
  std::cout << value << std::endl;
#endif
}

// Overload for empty line
inline void printLine() {
#if defined(ARDUINO)
  Serial.println();
#else
  std::cout << std::endl;
#endif
}

// Overload for formatted printing (e.g., HEX, BIN, DEC)
template <typename T>
inline void print(const T& value, int format) {
#if defined(ARDUINO)
  Serial.print(value, format);
#else
  if (format == 16) {  // HEX
    std::cout << std::hex << value << std::dec;
  } else if (format == 2) {  // BIN
    // Binary printing for non-Arduino (simple implementation)
    std::cout << std::bitset<sizeof(T) * 8>(value);
  } else {
    std::cout << value;
  }
#endif
}

template <typename T>
inline void printLine(const T& value, int format) {
#if defined(ARDUINO)
  Serial.println(value, format);
#else
  print(value, format);
  std::cout << std::endl;
#endif
}

}  // namespace portable
