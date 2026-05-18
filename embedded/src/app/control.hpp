#pragma once

#include <stdint.h>
#include <stddef.h>
#include <atomic>

// Forward declarations
namespace vertiq { class MotorCommandQueue; class Motor; class VirtualMotorCommandQueue; }

namespace control {

// ============================================================================
// Control Modes
// ============================================================================

/**
 * @brief Operational modes for the Folia system
 * 
 * MANUAL:    Stream IMU/ADC/motor data only. Classifier disabled.
 *            Manual motor control always available.
 * 
 * SEMIAUTO:  Stream all data + classifier predictions. Classifier enabled.
 *            Manual motor control still available.
 * 
 * AUTO:      Stream all data + classifier predictions. Classifier enabled.
 *            Motor state automatically controlled based on classifier output.
 * 
 * GAIT_DEBUG:  Stream gait filter debug data (position, velocity, stance).
 *             For tuning SHOE threshold and testing filter integration.
 * 
 * GAIT_TRAIN:  Stream reduced gait filter data for ML training.
 *             Minimal packet size for efficient data collection.
 */
enum class ControlMode : uint8_t {
    MANUAL = 0,      // Data collection / logging mode
    SEMIAUTO = 1,    // Classifier + manual motor
    AUTO = 2,        // Classifier + automatic motor control
    GAIT_DEBUG = 3,  // Gait filter debug mode
    GAIT_TRAIN = 4   // Gait filter training mode (reduced packet)
};

// ============================================================================
// Classifier-to-Motor Mapping
// ============================================================================

/**
 * @brief Maximum number of classifier labels supported
 */
constexpr size_t MAX_CLASSIFIER_LABELS = 16;

/**
 * @brief Mapping entry from classifier label to motor state
 */
struct LabelMotorMapping {
    uint8_t classifierLabel;   ///< Classifier prediction index
    uint8_t motorState;        ///< Motor state to apply (0-4)
    const char* labelName;     ///< Human-readable label name (optional)
};

// ============================================================================
// AUTO Mode – No-Load Detection State Machine
// ============================================================================

/**
 * @brief States for the no-load (swing phase) detector used in AUTO motor control
 *
 * UNKNOWN    - Initial state before any stance data is available.
 * STANCE     - Foot is in stance (load bearing); motor changes are deferred.
 * SWING_WAIT - Stance just exited; waiting NOLOAD_TIMER_MS before confirming
 *              no-load (used when ADC is disabled).
 * NO_LOAD    - Confirmed swing / no-load; motor changes can be applied.
 */
enum class NoLoadState : uint8_t {
    UNKNOWN    = 0,
    STANCE     = 1,
    SWING_WAIT = 2,
    NO_LOAD    = 3,
};

/**
 * @brief Debug snapshot of the AUTO motor control state machine
 */
struct AutoMotorDebugInfo {
    NoLoadState noLoadState;        ///< Current no-load detector state
    uint8_t     appliedMotorLabel;  ///< Stable label that corresponds to the last applied motor state
    uint8_t     pendingMotorLabel;  ///< Deferred label waiting for next no-load (0xFF = none)
    uint8_t     pendingMotorState;  ///< Motor state for the deferred label (0xFF = none)
    uint32_t    msInCurrentState;   ///< Milliseconds spent in the current no-load state
    bool        immediateWindowOpen;///< True if within the 300 ms immediate-apply window
    int         virtualMotorState;  ///< Current virtual-motor state (-1 = uninitialised)
};

// ============================================================================
// State Variables (extern declarations)
// ============================================================================

extern ControlMode g_controlMode;
extern std::atomic<bool> g_modeSelected;
extern bool g_classifierError;
extern std::atomic<uint8_t> g_classifier_label;
extern std::atomic<uint8_t> g_classifier_confidence;

// ============================================================================
// Public API
// ============================================================================

/**
 * @brief Run mode selection with timeout
 * 
 * Waits for user input via TCP or Serial to select mode.
 * Defaults to MANUAL after timeout.
 * 
 * @param timeoutMs Timeout in milliseconds
 */
void selectMode(uint32_t timeoutMs = 5000);

/**
 * @brief Get current control mode
 */
ControlMode getMode();

/**
 * @brief Get mode name as string
 */
const char* getModeName(ControlMode mode);

/**
 * @brief Check if classifier should be enabled for current mode
 */
bool isClassifierEnabled();

/**
 * @brief Check if gait filter debug mode is enabled
 */
bool isGaitDebugEnabled();

/**
 * @brief Check if gait filter training mode is enabled
 */
bool isGaitTrainEnabled();

/**
 * @brief Check if auto motor control is enabled for current mode
 */
bool isAutoMotorEnabled();

/**
 * @brief Initialize classifier if needed for current mode
 * 
 * @return true if classifier initialized successfully (or not needed)
 */
bool initClassifierIfNeeded();

/**
 * @brief Process classifier result and update motor if in AUTO mode
 * 
 * Call this in the main loop after getting classifier result.
 * 
 * @param classifierLabel Current classifier prediction
 * @param confidence Prediction confidence (0-100)
 * @param motorQueue Motor command queue (for AUTO mode)
 * @return Motor state that was applied (or 0xFF if no change)
 */
uint8_t processClassifierResult(uint8_t classifierLabel, uint8_t confidence,
                                 vertiq::MotorCommandQueue* motorQueue);

/**
 * @brief Drive the AUTO-mode motor state machine.
 *
 * Call once per main-loop iteration (100 Hz), AFTER processClassifierResult /
 * getStableGaitPrediction have been called for any fresh classifier result that
 * arrived this iteration.
 *
 * Implements the following policy:
 *  - No-load is confirmed when the ADC reading falls to baseline (ADC enabled)
 *    or 200 ms after stance exits (ADC disabled), provided stance has not
 *    re-entered.
 *  - When both no-load and a new stable-label-driven motor state are detected,
 *    the state change is applied immediately if within 300 ms of no-load
 *    confirmation; otherwise it is deferred to the next no-load period.
 *  - A pending deferred change is applied at the start of the next confirmed
 *    no-load period only if the stable label still matches.
 *
 * @param inStance        True when combined stance detector (w_sag / ADC / foot-flat fallback) reports stance (heel strike to toe off).
 * @param adcEnabled      True when an ADS1220 is present and running.
 * @param adcRaw          Latest raw ADC reading (ignored when adcEnabled=false).
 * @param adcBaseline     ADC baseline value used for no-load detection.
 * @param motorQueue      Real motor command queue (nullptr if motor disabled).
 * @param virtualQueue    Virtual motor queue for logic testing (nullptr to skip).
 * @return Applied motor state this iteration, or 0xFF if no change was made.
 */
uint8_t updateAutoMotorControl(bool inStance,
                                bool adcEnabled, float adcRaw, float adcBaseline,
                                vertiq::MotorCommandQueue* motorQueue,
                                vertiq::VirtualMotorCommandQueue* virtualQueue = nullptr);

/**
 * @brief Reset the AUTO motor control state machine to its initial state.
 *
 * Call when switching into AUTO mode or after a long pause.
 */
void resetAutoMotorState();

/**
 * @brief Get a debug snapshot of the AUTO motor control state machine.
 */
AutoMotorDebugInfo getAutoMotorDebugInfo(vertiq::VirtualMotorCommandQueue* virtualQueue = nullptr);

/**
 * @brief Get stable gait prediction using debouncing and thresholding
 * 
 * Maintains a buffer of recent predictions and only returns a stable
 * prediction when all buffered predictions agree and meet confidence threshold.
 * Only call this on FRESH classifier results (20Hz), not on cached values.
 * 
 * @param classifierLabel Current classifier prediction (0-255)
 * @param confidence Prediction confidence (0-100)
 * @return Stable gait label, or 0xFF if no stable prediction available
 */
uint8_t getStableGaitPrediction(uint8_t classifierLabel, uint8_t confidence);

/**
 * @brief Get the last stable gait label without updating the debounce buffer
 * 
 * Use this for cached/repeated rows where you don't have a fresh classifier result.
 * 
 * @return Current stable gait label, or 0xFF if no stable prediction yet
 */
uint8_t getLastStableGaitLabel();

/**
 * @brief Set the classifier-to-motor mapping
 * 
 * @param mapping Array of LabelMotorMapping entries
 * @param count Number of entries in array
 */
void setLabelMotorMapping(const LabelMotorMapping* mapping, size_t count);

/**
 * @brief Get motor state for a given classifier label
 * 
 * @param classifierLabel Classifier prediction index
 * @return Motor state (0-4) or 0xFF if no mapping
 */
uint8_t getMotorStateForLabel(uint8_t classifierLabel);

/**
 * @brief Print current mode status to Serial/TCP
 */
void printModeStatus();

/**
 * @brief Print classifier-to-motor mapping to Serial/TCP
 */
void printLabelMotorMapping();

/**
 * @brief Set mode at runtime (for TCP commands)
 * 
 * @param mode New mode to set
 * @return true if mode change was successful
 */
bool setMode(ControlMode mode);

} // namespace control
