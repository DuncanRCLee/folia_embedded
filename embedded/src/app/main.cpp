#include "Arduino.h"
#include <WiFi.h>
#include <SPI.h>
#include <xbus.hpp>
#include <mti.hpp>
#include <hal.hpp>
#include <ads1220.hpp>

#include "main_prelude.hpp"
#include "network.hpp"
#include "indicator.hpp"
#include "vertiq_motor.hpp"
#include "ei_classifier.hpp"
#include "control.hpp"
#include "util.hpp"
#include "arduino_secrets.h"

// Gait filter for terrain classification
#include <gait_filter.hpp>
#include <gait_filter_integration.hpp>

#ifdef ARDUINO_ARCH_MBED
    #include "platform/Platform.h"
#endif
#include "imu_packet.hpp"
#include "gen/Packet.pb.h"
#include "pb_encode.h"

hal::Event drdy_event;
xsens::MTi* mti = NULL;

std::atomic<bool> logging_active{false};

// Queue for batching IMU packets (holds 32 pre-encoded packets)
hal::Queue<IMUPacketSlot, 32>* imuQueue = nullptr;
bool failstate = false;

// Shank pitch calibration: reference angle during standing
float g_standingPitchOffset = 0.0f;

// ADS1220 ADC conversion: 24-bit two's complement, Vref = 2.048V
// V = counts * 2.048 / 2^23.  Value is baseline-subtracted and inverted
// in the driver so positive = loaded.
static constexpr float ADC_COUNTS_TO_VOLTS = 2.048f / 8388608.0f;

// Folia components
vertiq::Motor* motor = nullptr;
vertiq::MotorCommandQueue* motorQueue = nullptr;
vertiq::VirtualMotorCommandQueue* virtualMotorQueue = nullptr;
adc::ADS1220* adcDevice = nullptr;

// Enable/disable folia features
bool enableMotor = true;
bool enableAdc = true;

void ingestISR() {
    hal::eventGiveFromISR(drdy_event);
}

void setup() {
    pinMode(LED_HD59, OUTPUT);
    pinMode(LED_HD61, OUTPUT);
    pinMode(DRDY, INPUT);
    digitalWrite(nCS, HIGH);
    hal::schedulerStart();

    hal::sleepMs(2000);
    Serial.begin(BAUDRATE);
    //while(!Serial);

    Serial.println(F("\n=== Folia Driver ===\n"));

    Serial.println("Starting indicator task...");
    hal::taskCreate(indicator::indicator_task, &logging_active, "led_task", 2048, 2);

    Wire.begin();
    Wire.setClock(400000UL);  //Fast i2c
    Wire.setTimeout(50);
    hal::sleepMs(1000);

    mti = new xsens::MTi(xsens::Xbus(MTI_ADDR, Wire, DRDY));

    // if (!i2cPing(MTI_ADDR)) {
    //     indicator::setImuFail();
    //     failstate = true;
    //     Serial.println(F("IMU I2C ping failed!"));
    //     return;
    // }

    if (!mti->detect(1000)) {
        indicator::setImuFail();
        failstate = true;
        Serial.println("Please check your hardware connections.");
        return;
    }

    mti->goToConfig();
    mti->requestDeviceInfo();
    // Restore factory defaults to ensure known state
    // Comment out the following 3 lines if not needed
    // mti->restoreFactoryDefaults();
    // delay(500);
    // mti->reset();
    mti->setFilterProfile(54);
    mti->configureOutputsAdv(
        {
            {xsens::XDI_SampleTimeFine, 100},
            {xsens::XDI_Acceleration, 100},
            {xsens::XDI_RateOfTurn, 100},
            {xsens::XDI_Quaternion, 100},
            {xsens::XDI_DeltaQ, 100},
            {xsens::XDI_DeltaV, 100},
        }
    );
    mti->goToMeasurement();
    Serial.println("MTi configuration complete!");

    // Set up IMU interrupt handling immediately after measurement starts
    drdy_event = hal::eventCreate(false);
    attachInterrupt(digitalPinToInterrupt(DRDY), ingestISR, RISING);

    // Calibrate standing pitch offset
    g_standingPitchOffset = util::calibrateStandingPitch(mti, drdy_event);

    // Initialize ADC (strain gauge)
    if (enableAdc) {
        Serial.println(F("Initializing ADC..."));
        adcDevice = new adc::ADS1220(PIN_ADC_MOSI, PIN_ADC_MISO, PIN_ADC_SCLK,
                                     PIN_ADC_CS, PIN_ADC_DRDY);
        if (!adcDevice->begin()) {
            Serial.println(F("ADC initialization failed!"));
            indicator::setAdcFail();
            failstate = true;
            return;
        }
        adcDevice->setDataRate(adc::DR_1000SPS);
        adcDevice->setPgaGain(adc::PGA_GAIN_64);
        adcDevice->selectMuxChannels(adc::MUX_AIN1_AIN2);
        adcDevice->setContinuousMode();
        adcDevice->lowSideSwitchClosed();
        adcDevice->startConversion();

        // Wait for ADC to stabilize
        delay(1000);
        adcDevice->calibrateBaseline();
        Serial.println(F("ADC ready"));
    }

    // Initialize motor
    if (enableMotor) {
        Serial.println(F("Initializing motor..."));
        motor = new vertiq::Motor(PIN_UART_MOT_RX, PIN_UART_MOT_TX);
        if (motor->setup() != 0) {
            Serial.println(F("Motor initialization failed!"));
            indicator::setMotorFail();
            failstate = true;
            return;
        }
        motorQueue = new vertiq::MotorCommandQueue(*motor);
        Serial.println(F("Motor ready"));
    }

    // ========================================================================
    // Initialize Gait Filter (before mode selection)
    // ========================================================================
    Serial.println(F("Initializing gait filter..."));
    // gait::initGaitFilter();
    // To initialize with mounting orientation only:
    float* initQuat = mti->getQuaternion();  // [w,x,y,z]
    gait::initGaitFilterEx(nullptr, initQuat);
    // To initialize with both custom lever arm and mounting orientation:
    // gait::Vec3f leverArm(0, -0.15f, 0);  // Example lever arm (body frame, meters)
    // gait::initGaitFilterEx(&leverArm, initQuat);

    // Enable ADC-based stance detection if ADC is present
    gait::setGaitFilterAdcEnabled(enableAdc && (adcDevice != nullptr));

    Serial.println(F("Gait filter ready"));

    // ========================================================================
    // Mode Selection (handled by control module) - BEFORE WiFi
    // ========================================================================
    // control::selectMode(2000);  // Defaults to Classifier for now
    control::setMode(control::ControlMode::MANUAL);

    // Always create the virtual motor for AUTO mode logic testing
    virtualMotorQueue = new vertiq::VirtualMotorCommandQueue();
    Serial.println(F("Virtual motor ready"));
    // Reset AUTO motor state machine whenever mode is (re-)selected
    control::resetAutoMotorState();

    // ========================================================================
    // Initialize classifier if needed for selected mode - BEFORE WiFi
    // ========================================================================
    control::initClassifierIfNeeded();

    // ========================================================================
    // WiFi Mode Selection (interactive with 10s timeout, defaults to AP)
    // ========================================================================
    Serial.println(F("Configuring WiFi..."));
    network::WiFiMode wifiMode = network::selectWifiMode(2000);
    
    if (wifiMode == network::WiFiMode::AP) {
        network::configureWifiAP();
    } else {
        network::configureWifiServe();
    };

    // Initialize IMU queue and network task for protobuf streaming
    imuQueue = new hal::Queue<IMUPacketSlot, 32>();
    Serial.println("IMU queue initialized");

    static void* logging_args[2];
    logging_args[0] = imuQueue;
    logging_args[1] = &logging_active;
    hal::taskCreate(network::logging_task, logging_args, "logging_task", 4096, 2);

    static void* console_args[3];
    console_args[0] = motorQueue;
    console_args[1] = adcDevice;
    console_args[2] = &logging_active;
    hal::taskCreate(network::console_task, console_args, "console_task", 4096, 2);

    // Start monitor task for connection health management
    hal::taskCreate(network::monitor_task, nullptr, "monitor_task", 4096, 2);

    // Initialization complete - set idle status (waiting for TCP client)
    // Note: monitor_task will switch to CLIENT_CONNECTED when a client connects
    indicator::setIdle();

    // Print final mode status
    control::printModeStatus();
    Serial.println(F("\n=== System Ready ===\n"));
}

void loop() {
    if (failstate) {
        hal::sleepMs(1000000);
        return;
    }

    // Read ADC if active
    if (enableAdc && adcDevice) {
        adcDevice->loop();
    }

    // Process motor commands
    if (enableMotor && motorQueue) {
        motorQueue->processCommands();
    }

    // Wait for IMU data ready
    hal::eventWait(drdy_event, 0xFFFFFFFF);
    mti->readMessages();

    // Get the latest data from the MTi
    float* acc = mti->getAcceleration();
    float* gyr = mti->getRateOfTurn();
    float* quat = mti->getQuaternion();
    float* dv = mti->getDeltaV();
    float* dq = mti->getDeltaQ();
    float* freeAcc = mti->getFreeAcceleration();

    // ========================================================================
    // Process Gait Filter (always runs, output depends on mode)
    // ========================================================================
    // Pass latest ADC reading (in volts) before update so stance detector can use it
    if (enableAdc && adcDevice) {
        gait::setGaitFilterAdcReading(adcDevice->getRaw() * ADC_COUNTS_TO_VOLTS);
    }
    gait::processGaitFilter(dv, dq, quat, freeAcc, mti->getSampleTimeFine());

    // ========================================================================
    // AUTO Mode: update no-load state machine and apply motor state when safe
    // ========================================================================
    if (control::isAutoMotorEnabled()) {
        bool inStance = gait::g_gaitFilter.isInStance();
        float adcRaw      = (enableAdc && adcDevice) ? adcDevice->getRaw()      * ADC_COUNTS_TO_VOLTS : 0.0f;
        float adcBaseline = (enableAdc && adcDevice) ? adcDevice->getBaseline() * ADC_COUNTS_TO_VOLTS : 0.0f;
        control::updateAutoMotorControl(inStance,
                                         enableAdc && (adcDevice != nullptr),
                                         adcRaw, adcBaseline,
                                         motorQueue,
                                         virtualMotorQueue);
    }

    // Create protobuf packet
    imu_Packet pkt = imu_Packet_init_zero;
    pkt.time_milis = millis();
    pkt.sample_time_fine = mti->getSampleTimeFine();

    // Choose packet type based on mode
    if (control::isGaitDebugEnabled()) {
        // GAIT_DEBUG mode: Stream filter state for tuning
        pkt.which_payload = imu_Packet_imu_gait_tag;

        // Raw IMU data
        memcpy(&pkt.payload.imu_gait.a_x, acc, 3 * sizeof(float));
        memcpy(&pkt.payload.imu_gait.w_x, gyr, 3 * sizeof(float));

        // Get filter output
        gait::GaitFilterOutput out = gait::g_gaitFilter.peekOutput();
        
        // Filter estimated velocity (ENU world frame)
        pkt.payload.imu_gait.vel_x = out.vel_x;
        pkt.payload.imu_gait.vel_y = out.vel_y;
        pkt.payload.imu_gait.vel_z = out.vel_z;
        
        // Filter estimated position (step-relative, ENU world frame)
        pkt.payload.imu_gait.pos_x = out.pos_x;
        pkt.payload.imu_gait.pos_y = out.pos_y;
        pkt.payload.imu_gait.pos_z = out.pos_z;
        
        // Sagittal-plane-aligned velocity (walking direction frame)
        pkt.payload.imu_gait.vel_fwd = out.vel_fwd;
        pkt.payload.imu_gait.vel_lat = out.vel_lat;
        
        // Sagittal-plane-aligned position (walking direction frame)
        pkt.payload.imu_gait.pos_fwd = out.pos_fwd;
        pkt.payload.imu_gait.pos_lat = out.pos_lat;
        
        // Sagittal-plane-aligned acceleration (walking direction frame)
        // pkt.payload.imu_gait.a_fwd = out.a_fwd;
        // pkt.payload.imu_gait.a_lat = out.a_lat;

        // Sagittal plane angular rate (rotation about medial-lateral axis)
        pkt.payload.imu_gait.w_sag = out.w_sag;
        
        // Walking direction angle used for transformation
        pkt.payload.imu_gait.walking_direction = out.heading;
        
        // Shank pitch: sagittal plane rotation relative to standing
        pkt.payload.imu_gait.pitch = util::computeShankPitch(quat, g_standingPitchOffset);
        
        // Stance detection status
        // stance: omega-sag / ADC gait-phase result (HS/TO detection)
        pkt.payload.imu_gait.stance    = out.stance_active;
        // foot_flat: SHOE debounced foot-flat window (drives ZUPT)
        pkt.payload.imu_gait.foot_flat = out.foot_flat;
        pkt.payload.imu_gait.shoe_statistic = out.shoe_statistic;
        pkt.payload.imu_gait.step_count = gait::g_gaitFilter.getStepCount();
        pkt.payload.imu_gait.gait_phase = static_cast<uint32_t>(gait::g_gaitFilter.getPhase());
        
        // Debug fields for stance detection troubleshooting
        pkt.payload.imu_gait.debounce_counter = out.debounce_counter;
        pkt.payload.imu_gait.raw_should_enter = out.raw_should_enter;
        pkt.payload.imu_gait.raw_should_exit  = out.raw_should_exit;

        // ADC data
        if (enableAdc && adcDevice) {
            pkt.payload.imu_gait.adc_raw      = adcDevice->getRaw()      * ADC_COUNTS_TO_VOLTS;
            pkt.payload.imu_gait.adc_baseline = adcDevice->getBaseline() * ADC_COUNTS_TO_VOLTS;
        } else {
            pkt.payload.imu_gait.adc_raw      = 0.0f;
            pkt.payload.imu_gait.adc_baseline = 0.0f;
        }

    } else if (control::isGaitTrainEnabled()) {
        // GAIT_TRAIN mode: Stream reduced filter state for ML training
        pkt.which_payload = imu_Packet_imu_train_tag;

        // Raw IMU data
        memcpy(&pkt.payload.imu_train.a_x, acc, 3 * sizeof(float));
        memcpy(&pkt.payload.imu_train.w_x, gyr, 3 * sizeof(float));

        // Get filter output
        gait::GaitFilterOutput out = gait::g_gaitFilter.peekOutput();
        
        // Filter estimated velocity (ENU world frame)
        pkt.payload.imu_train.vel_x = out.vel_x;
        pkt.payload.imu_train.vel_y = out.vel_y;
        pkt.payload.imu_train.vel_z = out.vel_z;
        
        // Sagittal-plane-aligned velocity (walking direction frame)
        pkt.payload.imu_train.vel_fwd = out.vel_fwd;
        pkt.payload.imu_train.vel_lat = out.vel_lat;
        
        // Filter estimated position (step-relative, ENU world frame)
        pkt.payload.imu_train.pos_x = out.pos_x;
        pkt.payload.imu_train.pos_y = out.pos_y;
        pkt.payload.imu_train.pos_z = out.pos_z;
        
        // Sagittal-plane-aligned position (walking direction frame)
        pkt.payload.imu_train.pos_fwd = out.pos_fwd;
        pkt.payload.imu_train.pos_lat = out.pos_lat;

        // Sagittal plane angular rate (rotation about medial-lateral axis)
        pkt.payload.imu_train.w_sag = out.w_sag;
        
        // Shank pitch: sagittal plane rotation relative to standing
        pkt.payload.imu_train.pitch = util::computeShankPitch(quat, g_standingPitchOffset);

        if (enableAdc && adcDevice) {
            pkt.payload.imu_train.adc_raw      = adcDevice->getRaw()      * ADC_COUNTS_TO_VOLTS;
            pkt.payload.imu_train.adc_baseline = adcDevice->getBaseline() * ADC_COUNTS_TO_VOLTS;
        } else {
            pkt.payload.imu_train.adc_raw      = 0.0f;
            pkt.payload.imu_train.adc_baseline = 0.0f;
        }
        
        // Stance detection status
        // stance: omega-sag / ADC gait-phase result (HS/TO detection)
        pkt.payload.imu_train.stance    = out.stance_active;
        // foot_flat: SHOE debounced foot-flat window (drives ZUPT)
        pkt.payload.imu_train.foot_flat = out.foot_flat;
        pkt.payload.imu_train.shoe_statistic = out.shoe_statistic;

    } else if (control::isClassifierEnabled()) {
        // SEMIAUTO or AUTO mode: Use classifier packet
        pkt.which_payload = imu_Packet_imu_classifier_tag;

        // Populate IMU data
        memcpy(&pkt.payload.imu_classifier.a_x, acc, 3 * sizeof(float));
        memcpy(&pkt.payload.imu_classifier.w_x, gyr, 3 * sizeof(float));
        
        // Get gait filter output for classifier features
        gait::GaitFilterOutput out = gait::g_gaitFilter.peekOutput();
        
        // Compute pitch for classifier input
        float pitch = util::computeShankPitch(quat, g_standingPitchOffset);

        // Populate gait filter outputs for logging
        pkt.payload.imu_classifier.vel_fwd = out.vel_fwd;
        pkt.payload.imu_classifier.vel_lat = out.vel_lat;
        pkt.payload.imu_classifier.vel_z = out.vel_z;
        pkt.payload.imu_classifier.pos_fwd = out.pos_fwd;
        pkt.payload.imu_classifier.pos_lat = out.pos_lat;
        pkt.payload.imu_classifier.pos_z = out.pos_z;
        pkt.payload.imu_classifier.w_sag = out.w_sag;
        pkt.payload.imu_classifier.pitch = pitch;

        // Populate motor state (real motor if enabled, else virtual motor for AUTO testing)
        if (enableMotor && motor) {
            pkt.payload.imu_classifier.motor_state = motor->getStateAsU8();
        } else if (virtualMotorQueue && virtualMotorQueue->getCurrentState() >= 0) {
            pkt.payload.imu_classifier.motor_state =
                static_cast<uint8_t>(virtualMotorQueue->getCurrentState());
        } else {
            pkt.payload.imu_classifier.motor_state = 0;
        }

        if (enableAdc && adcDevice) {
            pkt.payload.imu_classifier.adc_raw      = adcDevice->getRaw()      * ADC_COUNTS_TO_VOLTS;
            pkt.payload.imu_classifier.adc_baseline = adcDevice->getBaseline() * ADC_COUNTS_TO_VOLTS;
        } else {
            pkt.payload.imu_classifier.adc_raw      = 0.0f;
            pkt.payload.imu_classifier.adc_baseline = 0.0f;
        }

        // stance: omega-sag / ADC gait-phase result (HS/TO detection)
        pkt.payload.imu_classifier.stance    = out.stance_active;
        // foot_flat: SHOE debounced foot-flat window (drives ZUPT)
        pkt.payload.imu_classifier.foot_flat = out.foot_flat;
        pkt.payload.imu_classifier.shoe_statistic = out.shoe_statistic;

        // Feed classifier with gait filter outputs: vel_fwd, vel_z, w_sag, pitch
        // Classifier runs at 20Hz (every 50ms / 5 samples)
        if (!control::g_classifierError && ei_classifier::isReady()) {
            // Sample order MUST match EI_CLASSIFIER_FUSION_AXES_STRING and EISample struct definition
            // Need to change the addSample line below if model input features or order change
            ei_classifier::addSample(out.vel_fwd, out.vel_z, out.w_sag, pitch);
            
            if (ei_classifier::hasResult()) {
                ei_classifier::EIResult result = ei_classifier::getResult();
                ei_classifier::clearResult();
                
                uint8_t label = result.predicted_label;
                uint8_t confidence = static_cast<uint8_t>(result.confidence * 100);
                
                control::processClassifierResult(label, confidence, motorQueue);
                
                // Get stable gait prediction using debouncing (ONLY on fresh results)
                uint8_t stable_label = control::getStableGaitPrediction(label, confidence);
                
                pkt.payload.imu_classifier.classifier_label = label;
                pkt.payload.imu_classifier.classifier_confidence = confidence;
                pkt.payload.imu_classifier.classifier_stable_label = stable_label;
                
                // Populate per-class confidence scores
                pkt.payload.imu_classifier.conf_dec = result.scores[0];
                pkt.payload.imu_classifier.conf_dst = result.scores[1];
                pkt.payload.imu_classifier.conf_idl = result.scores[2];
                pkt.payload.imu_classifier.conf_inc = result.scores[3];
                pkt.payload.imu_classifier.conf_lvl = result.scores[4];
                pkt.payload.imu_classifier.conf_ust = result.scores[5];
            } else {
                // Use cached values - don't update debounce buffer
                pkt.payload.imu_classifier.classifier_label = control::g_classifier_label.load(std::memory_order_relaxed);
                pkt.payload.imu_classifier.classifier_confidence = control::g_classifier_confidence.load(std::memory_order_relaxed);
                pkt.payload.imu_classifier.classifier_stable_label = control::getLastStableGaitLabel();
                
                // No per-class scores on cached rows
                pkt.payload.imu_classifier.conf_dec = 0.0f;
                pkt.payload.imu_classifier.conf_dst = 0.0f;
                pkt.payload.imu_classifier.conf_idl = 0.0f;
                pkt.payload.imu_classifier.conf_inc = 0.0f;
                pkt.payload.imu_classifier.conf_lvl = 0.0f;
                pkt.payload.imu_classifier.conf_ust = 0.0f;
            }
        } else {
            // Classifier error
            pkt.payload.imu_classifier.classifier_label = control::g_classifierError ? 0xFE : 0xFF;
            pkt.payload.imu_classifier.classifier_confidence = 0;
            pkt.payload.imu_classifier.classifier_stable_label = 0xFF;  // No stable prediction on error
            pkt.payload.imu_classifier.conf_dec = 0.0f;
            pkt.payload.imu_classifier.conf_dst = 0.0f;
            pkt.payload.imu_classifier.conf_idl = 0.0f;
            pkt.payload.imu_classifier.conf_inc = 0.0f;
            pkt.payload.imu_classifier.conf_lvl = 0.0f;
            pkt.payload.imu_classifier.conf_ust = 0.0f;
        }

    } else {
        // MANUAL mode: Use standard IMU packet (no classifier fields)
        pkt.which_payload = imu_Packet_imu_tag;

        // Populate IMU data
        memcpy(&pkt.payload.imu.a_x, acc, 3 * sizeof(float));
        memcpy(&pkt.payload.imu.w_x, gyr, 3 * sizeof(float));
        memcpy(&pkt.payload.imu.qw, quat, 4 * sizeof(float));
        memcpy(&pkt.payload.imu.dv_x, dv, 3 * sizeof(float));
        memcpy(&pkt.payload.imu.dq_w, dq, 4 * sizeof(float));

        // Populate motor state
        if (enableMotor && motor) {
            pkt.payload.imu.motor_state = motor->getStateAsU8();
        } else {
            pkt.payload.imu.motor_state = 0;
        }

        // Populate ADC data
        if (enableAdc && adcDevice) {
            pkt.payload.imu.adc_raw      = adcDevice->getRaw()      * ADC_COUNTS_TO_VOLTS;
            pkt.payload.imu.adc_baseline = adcDevice->getBaseline() * ADC_COUNTS_TO_VOLTS;
        } else {
            pkt.payload.imu.adc_raw      = 0.0f;
            pkt.payload.imu.adc_baseline = 0.0f;
        }
    }

    // Only encode and queue if logging is active
    if (logging_active.load(std::memory_order_relaxed)) {
        IMUPacketSlot slot;
        pb_ostream_t stream = pb_ostream_from_buffer(slot.data, IMU_PACKET_BUF_SIZE);

        if (pb_encode(&stream, imu_Packet_fields, &pkt)) {
            slot.size = stream.bytes_written;

            // Try to push to queue (non-blocking)
            if (!imuQueue->tryPush(slot)) {
                // Queue is full - data loss
                Serial.println("WARNING: IMU queue full");
            }
        } else {
            Serial.print("ERROR: Protobuf encoding failed: ");
            Serial.println(PB_GET_ERROR(&stream));
        }
    }
}
