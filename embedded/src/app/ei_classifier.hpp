#pragma once

// ============================================================================
// Edge Impulse Model Configuration Override
// Must be defined BEFORE including any Edge Impulse headers
// ============================================================================

/**
 * @brief Number of slices per model window (controls prediction rate)
 * 
 * With 100 samples window at 100Hz (1000ms):
 *   - 20 slices → 5 samples/slice → 50ms stride → 20Hz predictions
 *   - 4 slices  → 25 samples/slice → 250ms stride → 4Hz predictions (default)
 * 
 * This macro MUST be defined before model_metadata.h is included.
 */
#define EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW 20

// Include model metadata AFTER the override - this is safe since it's all #defines
#ifdef EI_CLASSIFIER_ENABLED
#include "model-parameters/model_metadata.h"
#endif

/**
 * @file ei_classifier.hpp
 * @brief Edge Impulse Terrain Classifier Integration
 * 
 * This module integrates the Edge Impulse terrain classifier for the Folia
 * device using run_classifier_continuous() for efficient streaming inference.
 * 
 * Model: terrain_classifier_v1
 * Inputs: vel_fwd, vel_z, w_sag, pitch (from gait_filter)
 * Window: 1000ms (100 samples @ 100Hz)
 * Slice: 50ms (5 samples) → 20Hz output rate
 * Labels: decline, downstairs, idle, incline, level, upstairs
 * 
 * Usage:
 * 1. Call ei_classifier::init() in setup()
 * 2. Call ei_classifier::addSample() each IMU cycle (100Hz) with gait filter outputs
 * 3. Check ei_classifier::hasResult() and getResult() for classification (20Hz)
 */

#include <cstdint>
#include <cstddef>
#include <atomic>

namespace ei_classifier {

// ============================================================================
// Configuration Constants (from model_metadata.h when EI_CLASSIFIER_ENABLED)
// ============================================================================

#ifdef EI_CLASSIFIER_ENABLED

// Use actual values from model metadata
constexpr size_t EI_FEATURES_PER_SAMPLE = EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME;
constexpr size_t EI_WINDOW_SIZE = EI_CLASSIFIER_RAW_SAMPLE_COUNT;
constexpr size_t EI_SLICE_SIZE = EI_CLASSIFIER_SLICE_SIZE;
constexpr size_t EI_SLICE_BUFFER_SIZE = EI_SLICE_SIZE * EI_FEATURES_PER_SAMPLE;
constexpr size_t EI_NUM_LABELS = EI_CLASSIFIER_LABEL_COUNT;
constexpr float EI_DEFAULT_THRESHOLD = EI_CLASSIFIER_THRESHOLD;

#else

// Fallback values for stub mode (must match model when enabled)
constexpr size_t EI_FEATURES_PER_SAMPLE = 4;
constexpr size_t EI_WINDOW_SIZE = 100;
constexpr size_t EI_SLICE_SIZE = 5;  // 100 / 20
constexpr size_t EI_SLICE_BUFFER_SIZE = EI_SLICE_SIZE * EI_FEATURES_PER_SAMPLE;
constexpr size_t EI_NUM_LABELS = 6;
constexpr float EI_DEFAULT_THRESHOLD = 0.6f;

#endif

/**
 * @brief Maximum number of classification labels (for fixed-size array)
 */
constexpr size_t EI_MAX_LABELS = 10;

/**
 * @brief Minimum confidence threshold for valid classification
 */
constexpr float EI_CONFIDENCE_THRESHOLD = EI_DEFAULT_THRESHOLD;

// ============================================================================
// Data Structures
// ============================================================================

/**
 * @brief Input sample for the terrain classifier
 * 
 * Uses gait filter outputs aligned with the sagittal plane.
 * MUST match the order in model_metadata.h: vel_fwd + vel_z + w_sag + pitch
 * Need to change this struct and related code if model input features change
 */
struct EISample {
    float vel_fwd;      ///< Forward velocity in walking direction (m/s)
    float vel_z;        ///< Vertical velocity (m/s)
    float w_sag;        ///< Angular velocity in sagittal plane (rad/s)
    float pitch;        ///< Shank pitch angle relative to standing (rad)
};

/**
 * @brief Classification result
 */
struct EIResult {
    bool valid;                     ///< True if classification was performed
    uint8_t predicted_label;        ///< Index of predicted class
    float confidence;               ///< Confidence score (0.0 - 1.0)
    float scores[EI_MAX_LABELS];    ///< Individual class scores
    size_t num_labels;              ///< Number of labels in model
    uint32_t inference_time_us;     ///< Time taken for inference
    bool anomaly_detected;          ///< True if anomaly detection triggered
    float anomaly_score;            ///< Anomaly score (if model supports it)
};

/**
 * @brief Classifier state
 */
enum class EIState {
    UNINITIALIZED,
    READY,
    COLLECTING,
    INFERENCE_PENDING,
    ERROR
};

// ============================================================================
// Public API
// ============================================================================

/**
 * @brief Initialize the classifier module
 * 
 * Must be called once during setup() before adding samples.
 * 
 * @return true if initialization successful
 */
bool init();

/**
 * @brief Check if classifier is initialized and ready
 */
bool isReady();

/**
 * @brief Get current classifier state
 */
EIState getState();

/**
 * @brief Add a new sample to the input buffer
 * 
 * Call this function every IMU sample (100Hz) with gait filter outputs.
 * Inference runs automatically every EI_INFERENCE_INTERVAL samples (50ms).
 * 
 * @param sample The gait filter sample to add
 * @return true if sample was added successfully
 */
bool addSample(const EISample& sample);

/**
 * @brief Convenience function to add sample from individual values
 * 
 * @param vel_fwd Forward velocity from gait filter (m/s)
 * @param vel_z Vertical velocity from gait filter (m/s)
 * @param w_sag Sagittal plane angular rate from gait filter (rad/s)
 * @param pitch Shank pitch angle (rad)
 * @return true if sample was added successfully
 */
bool addSample(float vel_fwd, float vel_z, float w_sag, float pitch);

/**
 * @brief Check if a new classification result is available
 * 
 * @return true if getResult() will return a new result
 */
bool hasResult();

/**
 * @brief Get the latest classification result
 * 
 * @return EIResult structure with classification data
 */
EIResult getResult();

/**
 * @brief Clear the result flag after reading
 */
void clearResult();

/**
 * @brief Reset the classifier state
 * 
 * Clears the slice buffer and resets the continuous classifier state.
 */
void reset();

/**
 * @brief Get the number of samples collected in the current slice
 */
size_t getSliceSampleCount();

/**
 * @brief Get total samples processed since init
 */
size_t getTotalSampleCount();

/**
 * @brief Get the number of labels in the loaded model
 */
size_t getNumLabels();

/**
 * @brief Get the label name for a given index
 * 
 * @param index Label index (0 to getNumLabels()-1)
 * @return Label name string, or nullptr if index invalid
 */
const char* getLabelName(size_t index);



/**
 * @brief Set confidence threshold for valid classification
 * 
 * @param threshold Minimum confidence (0.0 - 1.0)
 */
void setConfidenceThreshold(float threshold);

/**
 * @brief Get current confidence threshold
 */
float getConfidenceThreshold();

// ============================================================================
// Statistics / Debugging
// ============================================================================

/**
 * @brief Get total number of inferences performed
 */
uint32_t getInferenceCount();

/**
 * @brief Get average inference time in microseconds
 */
uint32_t getAverageInferenceTime();

/**
 * @brief Print classifier status to Serial
 */
void printStatus();

} // namespace ei_classifier
