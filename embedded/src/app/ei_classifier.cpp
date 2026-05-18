/**
 * @file ei_classifier.cpp
 * @brief Edge Impulse Terrain Classifier Implementation
 *
 * Uses run_classifier() with a manually managed circular sliding-window buffer.
 * run_classifier_continuous() is only valid for spectral DSP blocks (MFCC/MFE/
 * spectrogram); raw-flatten models require run_classifier() with the full window.
 *
 * Model: terrain_classifier_v1
 * Inputs: vel_fwd, vel_z, w_sag, pitch
 * Window: 1000ms (100 samples @ 100Hz)
 * Slice:  50ms  (5 samples)  → 20Hz inference trigger
 */

#include "ei_classifier.hpp"
#include "Arduino.h"
#include <cstring>

// Include Edge Impulse runtime headers (model_metadata.h already included via ei_classifier.hpp)
// These contain static variables and must only be in ONE translation unit
#ifdef EI_CLASSIFIER_ENABLED
#include <edge-impulse-sdk/classifier/ei_run_classifier.h>
#include "model-parameters/model_variables.h"
#endif

namespace ei_classifier {

// ============================================================================
// Internal State
// ============================================================================

namespace {
    // -------------------------------------------------------------------------
    // Circular sliding-window buffer (full window, oldest-first readout)
    // Size: EI_WINDOW_SIZE samples × EI_FEATURES_PER_SAMPLE features
    // -------------------------------------------------------------------------
    static constexpr size_t WINDOW_BUF_SIZE = EI_WINDOW_SIZE * EI_FEATURES_PER_SAMPLE;
    float g_window_buf[WINDOW_BUF_SIZE];   // circular buffer storage
    size_t g_window_write_idx = 0;         // next write position (in samples, not floats)
    size_t g_window_samples_filled = 0;    // how many samples written (caps at EI_WINDOW_SIZE)

    // Samples received in the current slice period (0 → EI_SLICE_SIZE trigger)
    size_t g_slice_sample_count = 0;

    // Total samples processed since init
    size_t g_total_sample_count = 0;

    // Flag indicating new result available
    std::atomic<bool> g_result_available{false};

    // Latest classification result
    EIResult g_last_result;

    // Classifier state
    EIState g_state = EIState::UNINITIALIZED;

    // Configuration
    float g_confidence_threshold = EI_CONFIDENCE_THRESHOLD;

    // Statistics
    uint32_t g_inference_count = 0;
    uint64_t g_total_inference_time = 0;

#ifdef EI_CLASSIFIER_ENABLED
    // Edge Impulse signal structure for inference
    signal_t g_signal;

    /**
     * @brief get_data callback – reads from the circular buffer oldest-first.
     *
     * At call time g_window_write_idx points to the NEXT slot to be written,
     * which is also the OLDEST slot when the buffer is full.
     * offset and length are in units of floats (features), not samples.
     */
    int get_window_data(size_t offset, size_t length, float* out_ptr) {
        // Start reading from the oldest sample
        size_t start_sample = (g_window_samples_filled < EI_WINDOW_SIZE)
                              ? 0                   // buffer not yet full – oldest is index 0
                              : g_window_write_idx; // buffer full – oldest is at write head
        size_t start_float = start_sample * EI_FEATURES_PER_SAMPLE + offset;
        for (size_t i = 0; i < length; i++) {
            size_t idx = (start_float + i) % WINDOW_BUF_SIZE;
            out_ptr[i] = g_window_buf[idx];
        }
        return 0;
    }
#endif

    /**
     * @brief Run run_classifier() on the current circular window buffer.
     */
    EIResult performInference() {
        EIResult result;
        memset(&result, 0, sizeof(result));

#ifdef EI_CLASSIFIER_ENABLED
        // Total features in the window passed to run_classifier
        size_t total_features = g_window_samples_filled * EI_FEATURES_PER_SAMPLE;
        if (total_features == 0) {
            result.valid = false;
            return result;
        }

        g_signal.total_length = total_features;
        g_signal.get_data     = &get_window_data;

        ei_impulse_result_t ei_result;
        uint32_t start_time = micros();

        EI_IMPULSE_ERROR err = run_classifier(&g_signal, &ei_result, false);

        uint32_t end_time = micros();
        result.inference_time_us = end_time - start_time;

        if (err != EI_IMPULSE_OK) {
            Serial.print("ERROR: Edge Impulse inference failed: ");
            Serial.println(err);
            result.valid = false;
            return result;
        }
        
        // Copy results - all class confidences are available
        result.valid = true;
        result.num_labels = EI_CLASSIFIER_LABEL_COUNT;
        
        float max_score = 0.0f;
        size_t max_idx = 0;
        
        for (size_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT && i < EI_MAX_LABELS; i++) {
            result.scores[i] = ei_result.classification[i].value;
            if (result.scores[i] > max_score) {
                max_score = result.scores[i];
                max_idx = i;
            }
        }
        
        result.predicted_label = max_idx;
        result.confidence = max_score;
        
        // Check anomaly detection (if model supports it)
#if EI_CLASSIFIER_HAS_ANOMALY
        result.anomaly_detected = (ei_result.anomaly > 0.5f);
        result.anomaly_score = ei_result.anomaly;
#else
        result.anomaly_detected = false;
        result.anomaly_score = 0.0f;
#endif
        
        // Update statistics
        g_inference_count++;
        g_total_inference_time += result.inference_time_us;

#else
        // Stub mode
        result.valid = true;
        result.predicted_label = 0;
        result.confidence = 0.0f;
        result.num_labels = 0;
        result.inference_time_us = 1000;
        result.anomaly_detected = false;
        result.anomaly_score = 0.0f;
        g_inference_count++;
        g_total_inference_time += result.inference_time_us;
#endif

        return result;
    }

} // anonymous namespace

// ============================================================================
// Public API Implementation
// ============================================================================

bool init() {
    memset(g_window_buf, 0, sizeof(g_window_buf));
    g_window_write_idx     = 0;
    g_window_samples_filled = 0;
    g_slice_sample_count   = 0;
    g_total_sample_count   = 0;
    g_result_available.store(false);
    g_inference_count      = 0;
    g_total_inference_time = 0;
    memset(&g_last_result, 0, sizeof(g_last_result));

#ifdef EI_CLASSIFIER_ENABLED
    Serial.println(F("Edge Impulse terrain classifier initializing..."));
    Serial.print(F("Model: "));          Serial.println(EI_CLASSIFIER_PROJECT_NAME);
    Serial.print(F("Inputs: "));         Serial.println(EI_CLASSIFIER_FUSION_AXES_STRING);
    Serial.print(F("Labels: "));         Serial.println(EI_CLASSIFIER_LABEL_COUNT);
    Serial.print(F("Window size: "));
    Serial.print(EI_CLASSIFIER_RAW_SAMPLE_COUNT);
    Serial.print(F(" samples ("));
    Serial.print(EI_CLASSIFIER_RAW_SAMPLE_COUNT * EI_CLASSIFIER_INTERVAL_MS);
    Serial.println(F("ms)"));
    Serial.print(F("Slice size: "));
    Serial.print(EI_SLICE_SIZE);
    Serial.print(F(" samples ("));
    Serial.print(EI_SLICE_SIZE * EI_CLASSIFIER_INTERVAL_MS);
    Serial.println(F("ms) → inference every slice"));
    Serial.print(F("Classes: "));
    for (size_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
        if (i > 0) Serial.print(F(", "));
        Serial.print(ei_classifier_inferencing_categories[i]);
    }
    Serial.println();
    Serial.println(F("Using run_classifier() with circular sliding-window buffer"));
#else
    Serial.println(F("Edge Impulse classifier initialized (STUB MODE)"));
    Serial.println(F("Define EI_CLASSIFIER_ENABLED and add model to lib/ei_model/"));
#endif

    g_state = EIState::READY;
    return true;
}

bool isReady() {
    return g_state == EIState::READY || g_state == EIState::COLLECTING;
}

EIState getState() {
    return g_state;
}

bool addSample(const EISample& sample) {
    if (g_state == EIState::UNINITIALIZED || g_state == EIState::ERROR) {
        return false;
    }

    g_state = EIState::COLLECTING;

    // Write new sample into circular window buffer
    // Feature order must match EI_CLASSIFIER_FUSION_AXES_STRING: vel_fwd, vel_z, w_sag, pitch
    // Need to change the sample lines below if model input features or order change
    size_t base = g_window_write_idx * EI_FEATURES_PER_SAMPLE;
    g_window_buf[base + 0] = sample.vel_fwd;
    g_window_buf[base + 1] = sample.vel_z;
    g_window_buf[base + 2] = sample.w_sag;
    g_window_buf[base + 3] = sample.pitch;

    g_window_write_idx = (g_window_write_idx + 1) % EI_WINDOW_SIZE;
    if (g_window_samples_filled < EI_WINDOW_SIZE) {
        g_window_samples_filled++;
    }

    g_slice_sample_count++;
    g_total_sample_count++;

    // Fire inference every EI_SLICE_SIZE samples (20 Hz)
    if (g_slice_sample_count >= EI_SLICE_SIZE) {
        g_slice_sample_count = 0;

        // Only run once we have at least one full window of data
        if (g_window_samples_filled >= EI_WINDOW_SIZE) {
            g_state = EIState::INFERENCE_PENDING;
            g_last_result = performInference();
            if (g_last_result.valid) {
                g_result_available.store(true);
            }
            g_state = EIState::COLLECTING;
        }
    }

    return true;
}

// Convenience overload to add sample from individual values
// Sample order MUST match EI_CLASSIFIER_FUSION_AXES_STRING and EISample struct definition
// If model input features or order change, need to update this function and EISample struct
bool addSample(float vel_fwd, float vel_z, float w_sag, float pitch) {
    EISample sample;
    sample.vel_fwd = vel_fwd;
    sample.vel_z = vel_z;
    sample.w_sag = w_sag;
    sample.pitch = pitch;
    return addSample(sample);
}

bool hasResult() {
    return g_result_available.load();
}

EIResult getResult() {
    return g_last_result;
}

void clearResult() {
    g_result_available.store(false);
}

void reset() {
    memset(g_window_buf, 0, sizeof(g_window_buf));
    g_window_write_idx      = 0;
    g_window_samples_filled = 0;
    g_slice_sample_count    = 0;
    g_total_sample_count    = 0;
    g_result_available.store(false);

    if (g_state != EIState::ERROR) {
        g_state = EIState::READY;
    }
}

size_t getSliceSampleCount() {
    return g_slice_sample_count;
}

size_t getTotalSampleCount() {
    return g_total_sample_count;
}

size_t getNumLabels() {
#ifdef EI_CLASSIFIER_ENABLED
    return EI_CLASSIFIER_LABEL_COUNT;
#else
    return 0;
#endif
}

const char* getLabelName(size_t index) {
#ifdef EI_CLASSIFIER_ENABLED
    if (index < EI_CLASSIFIER_LABEL_COUNT) {
        return ei_classifier_inferencing_categories[index];
    }
#endif
    return nullptr;
}

void setConfidenceThreshold(float threshold) {
    g_confidence_threshold = threshold;
}

float getConfidenceThreshold() {
    return g_confidence_threshold;
}

uint32_t getInferenceCount() {
    return g_inference_count;
}

uint32_t getAverageInferenceTime() {
    if (g_inference_count == 0) return 0;
    return static_cast<uint32_t>(g_total_inference_time / g_inference_count);
}

void printStatus() {
    Serial.println(F("\n=== Edge Impulse Classifier Status ==="));
    
    Serial.print(F("State: "));
    switch (g_state) {
        case EIState::UNINITIALIZED: Serial.println(F("UNINITIALIZED")); break;
        case EIState::READY: Serial.println(F("READY")); break;
        case EIState::COLLECTING: Serial.println(F("COLLECTING")); break;
        case EIState::INFERENCE_PENDING: Serial.println(F("INFERENCE_PENDING")); break;
        case EIState::ERROR: Serial.println(F("ERROR")); break;
    }
    
    Serial.print(F("Window samples filled: "));
    Serial.print(g_window_samples_filled);
    Serial.print(F(" / "));
    Serial.println(EI_WINDOW_SIZE);

    Serial.print(F("Slice progress: "));
    Serial.print(g_slice_sample_count);
    Serial.print(F(" / "));
    Serial.println(EI_SLICE_SIZE);
    
    Serial.print(F("Total samples: "));
    Serial.println(g_total_sample_count);
    
    Serial.print(F("Confidence threshold: "));
    Serial.println(g_confidence_threshold);
    
    Serial.print(F("Total inferences: "));
    Serial.println(g_inference_count);
    
    Serial.print(F("Avg inference time: "));
    Serial.print(getAverageInferenceTime());
    Serial.println(F(" us"));
    
#ifdef EI_CLASSIFIER_ENABLED
    Serial.print(F("Model: "));
    Serial.println(EI_CLASSIFIER_PROJECT_NAME);
    Serial.print(F("Slice size: "));
    Serial.print(EI_SLICE_SIZE);
    Serial.println(F(" samples"));
    Serial.print(F("Labels: "));
    for (size_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
        if (i > 0) Serial.print(F(", "));
        Serial.print(ei_classifier_inferencing_categories[i]);
    }
    Serial.println();
#else
    Serial.println(F("Model: STUB MODE (no model loaded)"));
#endif
    
    if (g_last_result.valid) {
        Serial.println(F("\nLast result:"));
        Serial.print(F("  Label: "));
        Serial.println(g_last_result.predicted_label);
        Serial.print(F("  Confidence: "));
        Serial.println(g_last_result.confidence);
        Serial.print(F("  Inference time: "));
        Serial.print(g_last_result.inference_time_us);
        Serial.println(F(" us"));
        Serial.println(F("  All scores:"));
        for (size_t i = 0; i < g_last_result.num_labels; i++) {
            Serial.print(F("    "));
            const char* name = getLabelName(i);
            if (name) Serial.print(name);
            else Serial.print(i);
            Serial.print(F(": "));
            Serial.println(g_last_result.scores[i]);
        }
    }
    
    Serial.println(F("=====================================\n"));
}

} // namespace ei_classifier
