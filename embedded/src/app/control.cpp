#include "control.hpp"
#include "Arduino.h"
#include <hal.hpp>
#include "network.hpp"
#include "ei_classifier.hpp"
#include "vertiq_motor.hpp"
#include "indicator.hpp"

namespace control {

// ============================================================================
// State Variables
// ============================================================================

ControlMode g_controlMode = ControlMode::MANUAL;
std::atomic<bool> g_modeSelected{false};
bool g_classifierError = false;
std::atomic<uint8_t> g_classifier_label{0};
std::atomic<uint8_t> g_classifier_confidence{0};

// ============================================================================
// Classifier-to-Motor Mapping
// ============================================================================

// Default mapping for terrain_classifier_v1 labels:
// Labels: "decline", "downstairs", "idle", "incline", "level", "upstairs"
// Motor states: 0=OFF, 1=IDLE, 2=COASTING, 3=DAMPING, 4=SPINNING
static LabelMotorMapping g_labelMotorMap[MAX_CLASSIFIER_LABELS] = {
    {0, 1, "decline"},    // Classifier label 0 -> Stiffness State 3 (assist descent)
    {1, 1, "downstairs"}, // Classifier label 1 -> Stiffness State 3 (assist stairs down)
    {2, 0, "idle"},       // Classifier label 2 -> Stiffness State 0 (standing still)
    {3, 3, "incline"},    // Classifier label 3 -> Stiffness State 4 (assist climbing)
    {4, 2, "level"},      // Classifier label 4 -> Stiffness State 2 (level ground)
    {5, 4, "upstairs"},   // Classifier label 5 -> Stiffness State 4 (assist stairs up)
};
static size_t g_labelMotorMapCount = 6;

// Minimum confidence threshold for AUTO motor control
static constexpr uint8_t AUTO_CONFIDENCE_THRESHOLD = 60;

// ============================================================================
// AUTO Mode – No-Load State Machine Variables
// ============================================================================

// How long after stance exit (no ADC) before confirming no-load [ms]
static constexpr uint32_t NOLOAD_TIMER_MS = 200;

// How long after no-load confirmation that a label change is applied
// immediately vs. deferred to the next no-load period [ms]
static constexpr uint32_t NOLOAD_IMMEDIATE_WINDOW_MS = 400;

// Current state of the no-load detector
static NoLoadState g_noLoadState = NoLoadState::UNKNOWN;

// millis() when stance last exited (used for SWING_WAIT timer)
static uint32_t g_stanceExitTime = 0;

// millis() when no-load was confirmed for the current swing period
static uint32_t g_noLoadConfirmTime = 0;

// Whether we just entered NO_LOAD this iteration (used to apply deferred cmd)
static bool g_justEnteredNoLoad = false;

// The last stable label whose motor state has actually been applied
static uint8_t g_appliedMotorLabel = 0xFF;  // 0xFF = never applied

// Deferred (pending) label and state waiting for the next no-load window
static uint8_t g_pendingMotorLabel = 0xFF;  // 0xFF = nothing pending
static uint8_t g_pendingMotorState = 0xFF;

// ============================================================================
// Mode Selection
// ============================================================================

// Default mode to apply on timeout
static constexpr ControlMode DEFAULT_MODE = ControlMode::MANUAL;

void selectMode(uint32_t timeoutMs) {
    Serial.println(F("\n=== Mode Selection ==="));
    Serial.println(F("Send '1' or 'M' for MANUAL mode (data collection)"));
    Serial.println(F("Send '2' or 'S' for SEMIAUTO mode (classifier + manual motor)"));
    Serial.println(F("Send '3' or 'A' for AUTO mode (classifier + auto motor)"));
    Serial.println(F("Send '4' or 'G' for GAIT_DEBUG mode (filter tuning)"));
    Serial.println(F("Send '5' or 'T' for GAIT_TRAIN mode (ML training data)"));
    Serial.print(F("Default: "));
    Serial.print(getModeName(DEFAULT_MODE));
    Serial.print(F(" in "));
    Serial.print(timeoutMs / 1000);
    Serial.println(F(" seconds...\n"));

    uint32_t modeSelectStart = millis();
    uint32_t lastPrint = 0;

    while (!g_modeSelected.load() && (millis() - modeSelectStart < timeoutMs)) {
        char cmd = 0;
        
        // Check for mode selection command via TCP
        if (network::tcpClient && network::tcpClient.available()) {
            cmd = toupper(network::tcpClient.read());
        }
        
        // Check Serial for mode selection
        if (cmd == 0 && Serial.available()) {
            cmd = toupper(Serial.read());
        }
        
        // Process command
        if (cmd != 0) {
            ControlMode selectedMode = ControlMode::MANUAL;
            bool validCmd = false;
            
            if (cmd == '1' || cmd == 'M') {
                selectedMode = ControlMode::MANUAL;
                validCmd = true;
            } else if (cmd == '2' || cmd == 'S') {
                selectedMode = ControlMode::SEMIAUTO;
                validCmd = true;
            } else if (cmd == '3' || cmd == 'A') {
                selectedMode = ControlMode::AUTO;
                validCmd = true;
            } else if (cmd == '4' || cmd == 'G') {
                selectedMode = ControlMode::GAIT_DEBUG;
                validCmd = true;
            } else if (cmd == '5' || cmd == 'T') {
                selectedMode = ControlMode::GAIT_TRAIN;
                validCmd = true;
            }
            
            if (validCmd) {
                g_controlMode = selectedMode;
                g_modeSelected.store(true);
                
                const char* modeName = getModeName(selectedMode);
                Serial.print(modeName);
                Serial.println(F(" mode selected"));
                
                char buf[64];
                snprintf(buf, sizeof(buf), "%s mode selected", modeName);
                network::println(buf);
            }
        }

        uint32_t elapsed = millis() - modeSelectStart;
        uint32_t remaining = (timeoutMs - elapsed) / 1000;
        if (remaining != lastPrint && remaining > 0) {
            lastPrint = remaining;
            Serial.print(remaining);
            Serial.print(F("... "));
        }
        
        hal::sleepMs(100);
    }

    // Default
    if (!g_modeSelected.load()) {
        g_controlMode = DEFAULT_MODE;
        g_modeSelected.store(true);
        Serial.print(F("Timeout - defaulting to "));
        Serial.print(getModeName(DEFAULT_MODE));
        Serial.println();
        
        char buf[64];
        snprintf(buf, sizeof(buf), "Timeout - defaulting to %s", getModeName(DEFAULT_MODE));
        network::println(buf);
    }
}

ControlMode getMode() {
    return g_controlMode;
}

const char* getModeName(ControlMode mode) {
    switch (mode) {
        case ControlMode::MANUAL:     return "MANUAL";
        case ControlMode::SEMIAUTO:   return "SEMIAUTO";
        case ControlMode::AUTO:       return "AUTO";
        case ControlMode::GAIT_DEBUG: return "GAIT_DEBUG";
        case ControlMode::GAIT_TRAIN: return "GAIT_TRAIN";
        default:                      return "UNKNOWN";
    }
}

bool isClassifierEnabled() {
    return g_controlMode == ControlMode::SEMIAUTO || 
           g_controlMode == ControlMode::AUTO;
}

bool isGaitDebugEnabled() {
    return g_controlMode == ControlMode::GAIT_DEBUG;
}

bool isGaitTrainEnabled() {
    return g_controlMode == ControlMode::GAIT_TRAIN;
}

bool isAutoMotorEnabled() {
    return g_controlMode == ControlMode::AUTO;
}

// ============================================================================
// Classifier Initialization
// ============================================================================

bool initClassifierIfNeeded() {
    if (!isClassifierEnabled()) {
        Serial.println(F("MANUAL mode - classifier disabled"));
        network::println("MANUAL mode - classifier disabled");
        return true;  // Not needed, so "success"
    }

    Serial.println(F("Initializing Edge Impulse classifier..."));
    
#ifdef EI_CLASSIFIER_ENABLED
    if (!ei_classifier::init()) {
        Serial.println(F("ERROR: Classifier initialization failed!"));
        network::println("ERROR: Classifier initialization failed!");
        indicator::setClassifierFail();
        g_classifierError = true;
        return false;
    }
    
    ei_classifier::setConfidenceThreshold(0.6f);
    Serial.println(F("Classifier ready"));
    network::println("Classifier ready");
    return true;
#else
    Serial.println(F("WARNING: Classifier not compiled (EI_CLASSIFIER_ENABLED not defined)"));
    network::println("WARNING: Classifier not compiled - using stub mode");
    // Initialize stub mode for testing
    ei_classifier::init();
    indicator::setClassifierFail();
    g_classifierError = true;  // Mark as error since no real model
    return false;
#endif
}

// ============================================================================
// Stable Gait Detection (Debouncing)
// ============================================================================

// Configuration for stable gait detection
static constexpr size_t GAIT_DEBOUNCE_WINDOW = 3;      // Number of consecutive predictions needed
static constexpr uint8_t GAIT_CONFIDENCE_THRESHOLD = 80; // Minimum confidence (0-100)

// Prediction history entry
struct PredictionEntry {
    uint8_t label;
    uint8_t confidence;
};

// Static buffer for debouncing
static PredictionEntry g_predictionBuffer[GAIT_DEBOUNCE_WINDOW];
static size_t g_bufferIndex = 0;
static size_t g_bufferFilled = 0;
static uint8_t g_stableGaitLabel = 0xFF;  // Current stable prediction (0xFF = no stable prediction)

uint8_t getStableGaitPrediction(uint8_t classifierLabel, uint8_t confidence) {
    // Add new prediction to circular buffer
    g_predictionBuffer[g_bufferIndex].label = classifierLabel;
    g_predictionBuffer[g_bufferIndex].confidence = confidence;
    
    // Advance buffer index
    g_bufferIndex = (g_bufferIndex + 1) % GAIT_DEBOUNCE_WINDOW;
    
    // Track how many entries we've filled
    if (g_bufferFilled < GAIT_DEBOUNCE_WINDOW) {
        g_bufferFilled++;
    }
    
    // Check if buffer is full
    if (g_bufferFilled < GAIT_DEBOUNCE_WINDOW) {
        // Not enough data yet - return current stable state
        return g_stableGaitLabel;
    }
    
    // Check if all predictions in buffer match and meet confidence threshold
    uint8_t candidateLabel = g_predictionBuffer[0].label;
    bool allMatch = true;
    bool allConfident = true;
    
    for (size_t i = 0; i < GAIT_DEBOUNCE_WINDOW; i++) {
        if (g_predictionBuffer[i].label != candidateLabel) {
            allMatch = false;
            break;
        }
        if (g_predictionBuffer[i].confidence < GAIT_CONFIDENCE_THRESHOLD) {
            allConfident = false;
        }
    }
    
    // Update stable prediction only if all conditions met
    if (allMatch && allConfident) {
        // Only log if it's different from current stable state
        if (g_stableGaitLabel != candidateLabel) {
            Serial.print(F("Stable gait change: "));
            Serial.print(g_stableGaitLabel);
            Serial.print(F(" -> "));
            Serial.println(candidateLabel);
        }
        g_stableGaitLabel = candidateLabel;
        return g_stableGaitLabel;  // Return new stable state immediately
    }
    
    // Conditions not met - keep previous stable state
    return g_stableGaitLabel;
}

uint8_t getLastStableGaitLabel() {
    return g_stableGaitLabel;
}

// ============================================================================
// Classifier-to-Motor Processing
// ============================================================================

uint8_t processClassifierResult(uint8_t classifierLabel, uint8_t confidence,
                                 vertiq::MotorCommandQueue* /*motorQueue*/) {
    // Store result for streaming
    g_classifier_label.store(classifierLabel, std::memory_order_relaxed);
    g_classifier_confidence.store(confidence, std::memory_order_relaxed);
    
    // In AUTO mode the actual motor command is issued by updateAutoMotorControl()
    // which runs the no-load state machine.  Nothing more to do here.
    return 0xFF;
}

void setLabelMotorMapping(const LabelMotorMapping* mapping, size_t count) {
    if (count > MAX_CLASSIFIER_LABELS) {
        count = MAX_CLASSIFIER_LABELS;
    }
    
    for (size_t i = 0; i < count; i++) {
        g_labelMotorMap[i] = mapping[i];
    }
    g_labelMotorMapCount = count;
}

uint8_t getMotorStateForLabel(uint8_t classifierLabel) {
    for (size_t i = 0; i < g_labelMotorMapCount; i++) {
        if (g_labelMotorMap[i].classifierLabel == classifierLabel) {
            return g_labelMotorMap[i].motorState;
        }
    }
    return 0xFF;  // No mapping found
}

// ============================================================================
// Status / Debug
// ============================================================================

void printModeStatus() {
    Serial.println(F("\n=== Mode Status ==="));
    Serial.print(F("Mode: "));
    Serial.println(getModeName(g_controlMode));
    
    Serial.print(F("Classifier: "));
    if (!isClassifierEnabled()) {
        Serial.println(F("DISABLED"));
    } else if (g_classifierError) {
        Serial.println(F("ERROR"));
    } else {
        Serial.println(F("ACTIVE"));
    }
    
    Serial.print(F("Motor Control: "));
    Serial.println(isAutoMotorEnabled() ? F("AUTO") : F("MANUAL"));
    
    if (isClassifierEnabled()) {
        Serial.print(F("Last prediction: label="));
        Serial.print(g_classifier_label.load());
        Serial.print(F(" confidence="));
        Serial.print(g_classifier_confidence.load());
        Serial.println(F("%"));
    }
    Serial.println();
    
    // Also send to TCP
    network::println("\n=== Mode Status ===");
    char buf[64];
    snprintf(buf, sizeof(buf), "Mode: %s", getModeName(g_controlMode));
    network::println(buf);
    
    if (!isClassifierEnabled()) {
        network::println("Classifier: DISABLED");
    } else if (g_classifierError) {
        network::println("Classifier: ERROR");
    } else {
        network::println("Classifier: ACTIVE");
    }
    
    snprintf(buf, sizeof(buf), "Motor Control: %s", 
             isAutoMotorEnabled() ? "AUTO" : "MANUAL");
    network::println(buf);
}

void printLabelMotorMapping() {
    Serial.println(F("\n=== Classifier-to-Motor Mapping ==="));
    for (size_t i = 0; i < g_labelMotorMapCount; i++) {
        Serial.print(F("Label "));
        Serial.print(g_labelMotorMap[i].classifierLabel);
        if (g_labelMotorMap[i].labelName) {
            Serial.print(F(" ("));
            Serial.print(g_labelMotorMap[i].labelName);
            Serial.print(F(")"));
        }
        Serial.print(F(" -> Motor state "));
        Serial.println(g_labelMotorMap[i].motorState);
    }
    Serial.println();
    
    // Also send to TCP
    network::println("\n=== Classifier-to-Motor Mapping ===");
    char buf[64];
    for (size_t i = 0; i < g_labelMotorMapCount; i++) {
        snprintf(buf, sizeof(buf), "Label %d (%s) -> Motor %d",
                 g_labelMotorMap[i].classifierLabel,
                 g_labelMotorMap[i].labelName ? g_labelMotorMap[i].labelName : "?",
                 g_labelMotorMap[i].motorState);
        network::println(buf);
    }
}

// ============================================================================
// AUTO Mode – No-Load State Machine Implementation
// ============================================================================

void resetAutoMotorState() {
    g_noLoadState       = NoLoadState::UNKNOWN;
    g_stanceExitTime    = 0;
    g_noLoadConfirmTime = 0;
    g_justEnteredNoLoad = false;
    g_appliedMotorLabel = 0xFF;
    g_pendingMotorLabel = 0xFF;
    g_pendingMotorState = 0xFF;
}

// Helper: queue a motor state to real + virtual motor queues and log it.
static void applyMotorState(uint8_t motorState, uint8_t label,
                             vertiq::MotorCommandQueue* motorQueue,
                             vertiq::VirtualMotorCommandQueue* virtualQueue) {
    const char* labelName = nullptr;
    for (size_t i = 0; i < g_labelMotorMapCount; i++) {
        if (g_labelMotorMap[i].classifierLabel == label) {
            labelName = g_labelMotorMap[i].labelName;
            break;
        }
    }

    char buf[64];
    snprintf(buf, sizeof(buf), "[AUTO] -> state %d (%s)",
             (int)motorState, labelName ? labelName : "?");
    Serial.println(buf);
    network::println(buf);

    if (motorQueue) {
        motorQueue->queueStateCommand(motorState);
    }
    if (virtualQueue) {
        virtualQueue->queueStateCommand(motorState);
        virtualQueue->processCommands();
    }
}

uint8_t updateAutoMotorControl(bool inStance,
                                bool adcEnabled, float adcRaw, float adcBaseline,
                                vertiq::MotorCommandQueue* motorQueue,
                                vertiq::VirtualMotorCommandQueue* virtualQueue) {
    if (!isAutoMotorEnabled()) {
        return 0xFF;
    }

    uint32_t now = millis();
    g_justEnteredNoLoad = false;

    // ------------------------------------------------------------------
    // 1. Update no-load state machine
    // ------------------------------------------------------------------
    NoLoadState prevState = g_noLoadState;

    switch (g_noLoadState) {
        case NoLoadState::UNKNOWN:
            if (inStance) {
                g_noLoadState = NoLoadState::STANCE;
            }
            // Stay UNKNOWN until first stance event.  ADC and omega-sag are
            // event-driven, so there is nothing useful to poll here.
            break;

        case NoLoadState::STANCE:
            if (!inStance) {
                // ADC and omega-sag are event-driven: when they report no-load
                // the foot is genuinely unloaded/airborne, so enter NO_LOAD
                // immediately (next frame).  The 200ms SWING_WAIT timer was a
                // SHOE-specific workaround for its lagging, conservative output
                // and is preserved below for reference only.
                g_noLoadState       = NoLoadState::NO_LOAD;
                g_noLoadConfirmTime = now;
                g_justEnteredNoLoad = true;

                // [ SHOE fallback – commented out ]
                // if (adcEnabled) {
                //     if (adcRaw <= adcBaseline) { NO_LOAD immediately }
                //     else { SWING_WAIT fallback }
                // } else {
                //     // 200ms timer for SHOE lag compensation
                //     g_noLoadState    = NoLoadState::SWING_WAIT;
                //     g_stanceExitTime = now;
                // }
            }
            break;

        case NoLoadState::SWING_WAIT:
            // This state is only reachable when the SHOE fallback above is
            // re-enabled.  With ADC or omega-sag, STANCE transitions directly
            // to NO_LOAD and this block is dead code.
            if (inStance) {
                g_noLoadState = NoLoadState::STANCE;
            } else if (now - g_stanceExitTime >= NOLOAD_TIMER_MS) {
                g_noLoadState       = NoLoadState::NO_LOAD;
                g_noLoadConfirmTime = now;
                g_justEnteredNoLoad = true;
            }
            break;

        case NoLoadState::NO_LOAD:
            if (inStance) {
                g_noLoadState = NoLoadState::STANCE;
            }
            // ADC / omega-sag handle their own load detection internally;
            // no secondary raw-ADC comparison is needed here.
            // [ SHOE fallback – commented out ]
            // else if (adcEnabled && adcRaw > adcBaseline) {
            //     g_noLoadState = NoLoadState::STANCE;
            // }
            break;
    }

    // Log state transitions
    (void)prevState;  // suppress unused warning; transition logging removed

    // ------------------------------------------------------------------
    // 2. Determine current desired motor state from stable gait label
    // ------------------------------------------------------------------
    uint8_t stableLabel = g_stableGaitLabel;
    if (stableLabel == 0xFF) {
        // No stable prediction yet – nothing to do
        return 0xFF;
    }

    uint8_t desiredMotorState = getMotorStateForLabel(stableLabel);
    if (desiredMotorState == 0xFF) {
        return 0xFF;  // No mapping for this label
    }

    bool labelChanged = (stableLabel != g_appliedMotorLabel);

    // ------------------------------------------------------------------
    // 3. Decide whether to apply or defer the motor change
    // ------------------------------------------------------------------
    if (!labelChanged) {
        // Nothing new; but check for a deferred pending command on
        // entering no-load
        if (g_justEnteredNoLoad &&
            g_pendingMotorLabel != 0xFF &&
            g_pendingMotorLabel == stableLabel) {
            applyMotorState(g_pendingMotorState, stableLabel, motorQueue, virtualQueue);
            g_appliedMotorLabel = stableLabel;
            uint8_t applied     = g_pendingMotorState;
            g_pendingMotorLabel = 0xFF;
            g_pendingMotorState = 0xFF;
            return applied;
        }
        return 0xFF;  // No change needed
    }

    // Label has changed – decide timing
    if (g_noLoadState == NoLoadState::NO_LOAD) {
        uint32_t msSinceNoLoad = now - g_noLoadConfirmTime;
        if (g_justEnteredNoLoad || msSinceNoLoad <= NOLOAD_IMMEDIATE_WINDOW_MS) {
            applyMotorState(desiredMotorState, stableLabel, motorQueue, virtualQueue);
            g_appliedMotorLabel = stableLabel;
            g_pendingMotorLabel = 0xFF;
            g_pendingMotorState = 0xFF;
            return desiredMotorState;
        } else {
            // Beyond 300 ms window – defer to next no-load
            g_pendingMotorLabel = stableLabel;
            g_pendingMotorState = desiredMotorState;
        }
    } else {
        // In STANCE or SWING_WAIT – defer label change to next no-load
        g_pendingMotorLabel = stableLabel;
        g_pendingMotorState = desiredMotorState;
    }

    return 0xFF;  // No change applied this iteration
}

AutoMotorDebugInfo getAutoMotorDebugInfo(vertiq::VirtualMotorCommandQueue* virtualQueue) {
    AutoMotorDebugInfo info;
    info.noLoadState        = g_noLoadState;
    info.appliedMotorLabel  = g_appliedMotorLabel;
    info.pendingMotorLabel  = g_pendingMotorLabel;
    info.pendingMotorState  = g_pendingMotorState;
    info.immediateWindowOpen = (g_noLoadState == NoLoadState::NO_LOAD) &&
                               ((millis() - g_noLoadConfirmTime) <= NOLOAD_IMMEDIATE_WINDOW_MS);
    info.msInCurrentState   = (g_noLoadState == NoLoadState::NO_LOAD)
                                ? (millis() - g_noLoadConfirmTime)
                                : 0;
    info.virtualMotorState  = virtualQueue ? virtualQueue->getCurrentState() : -1;
    return info;
}

bool setMode(ControlMode mode) {
    g_controlMode = mode;
    
    const char* modeName = getModeName(mode);
    Serial.print(F("Mode set to: "));
    Serial.println(modeName);
    
    char buf[64];
    snprintf(buf, sizeof(buf), "Mode set to: %s", modeName);
    network::println(buf);
    
    return true;
}

} // namespace control
