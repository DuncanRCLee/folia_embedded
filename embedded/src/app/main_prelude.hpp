#pragma once
#include "Arduino.h"
#ifdef ARDUINO_ARCH_MBED
#include "pinDefinitions.h"
#endif

#define DEBUG_MODE true //1 for debug 2 for real

//PINS - IMU
#define DRDY PinName::PD_4
#define LED_HD59  PA_8
#define LED_HD61  PC_6
#define MTI_ADDR 0x6B
#define nCS 10                      //MTi nCS pin connected to Arduino Digital IO pin 10
#define NN 18

// Folia hardware pins (v5 PCB)
// ADC SPI Pins
#define PIN_ADC_MOSI     PC_3
#define PIN_ADC_MISO     PC_2
#define PIN_ADC_SCLK     PI_1
#define PIN_ADC_CS       PI_0
#define PIN_ADC_DRDY     PC_13

// Motor UART Pins
#define PIN_UART_MOT_RX  PA_0
#define PIN_UART_MOT_TX  PI_9

// External Pins
#define PIN_I2C2_SCL     PH_11
#define PIN_I2C2_SDA     PH_12
#define PIN_UART_RX2     PG_9
#define PIN_UART_TX2     PG_14
#define PIN_GPIO3        PD_5
#define PIN_GPIO4        PE_3
#define PIN_PWM2         PC_7
#define PIN_PWM3         PG_7

// LED Pins (same as LED_HD59/61)
#define PIN_LED1         PA_8
#define PIN_LED2         PC_6

// Loop timing
#define LOOP_DELAY_DEFAULT 10000  // 10ms = 100Hz operation


//KALMAN
const float GRAV = 9.80665f;
const float SI_TO_G = 1.0f / 9.81f;
const float DEG2RAD = M_PI / 180.0f;
const float RAD2DEG = 180.0f / M_PI;

const float sigma_an = 70e-6 * GRAV * sqrt(100);
const float sigma_wn = radians(0.003) * sqrt(100);
const float sigma_aw = (40e-6 * GRAV) / sqrt(3600);
const float sigma_ww = radians(6) / sqrt(3600);

//DATA
const uint32_t BAUDRATE = 115200; //230400
#define MTU 508

// Network batching – keep total UDP payload under the Portenta WiFi stack's
// internal buffer limit (~508 bytes).  The logging_task overflow check
// (head + HEADER_SIZE + sz > MTU) now dynamically caps each batch.
constexpr size_t BATCH_SIZE = 4;
