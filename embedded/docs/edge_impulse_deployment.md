# Edge Impulse Model Deployment Guide

This guide explains how to export a trained Edge Impulse model and integrate it with the Folia embedded system.

## Operational Modes

The firmware supports three operational modes:

| Mode | Description | Classifier | Motor Control |
|------|-------------|------------|---------------|
| **MANUAL** | Data collection / logging mode | Disabled | Manual only |
| **SEMIAUTO** | Classifier + manual motor | Enabled | Manual only |
| **AUTO** | Full automatic control | Enabled | Automatic (based on classifier) |

### Mode Selection at Startup

On boot, you have 10 seconds to select a mode via TCP or Serial:
- Send `1` or `M` for MANUAL mode
- Send `2` or `S` for SEMIAUTO mode
- Send `3` or `A` for AUTO mode
- **Default**: MANUAL after 10s timeout

### Runtime Mode Commands

| Command | Description |
|---------|-------------|
| `N` | Print current mode status |
| `O <1-3>` | Change mode (1=MANUAL, 2=SEMIAUTO, 3=AUTO) |
| `C` | Print classifier status |
| `L` | Print classifier-to-motor mapping |

### Classifier-to-Motor Mapping (AUTO Mode)

In AUTO mode, classifier predictions are mapped to motor states:

| Classifier Label | Default Motor State | Example Label Name |
|-----------------|--------------------|--------------------|
| 0 | 0 (OFF) | Idle |
| 1 | 1 (IDLE) | Slow |
| 2 | 2 (COASTING) | Norm |
| 3 | 3 (DAMPING) | Fast |
| 4 | 4 (SPINNING) | Upst |
| 5 | 4 (SPINNING) | Dnst |

The mapping can be customized in `control.cpp` via `setLabelMotorMapping()`.

### Classifier Error Codes (in streamed data)

| `classifier_label` | Meaning |
|--------------------|---------|
| 0x00 - 0xFD | Valid classification label |
| 0xFE | Classifier initialization error |
| 0xFF | Classifier not ready |

## Prerequisites

1. A trained model in [Edge Impulse Studio](https://studio.edgeimpulse.com/)
2. Model trained with the correct input features:
   - 6 axes: accelerometer (x,y,z) + gyroscope (x,y,z)
   - Sample rate: 100 Hz
   - Window size: 128 samples (configurable in `ei_classifier.hpp`)

## Export Steps

### 1. Export from Edge Impulse Studio

1. Open your project in Edge Impulse Studio
2. Go to **Deployment** tab
3. Select **Arduino library** as the deployment target
4. Click **Build**
5. Download the generated `.zip` file

### 2. Install the Library

1. Extract the downloaded `.zip` file
2. Copy the entire extracted folder to `lib/ei_model/`
3. Rename the folder to remove version numbers (e.g., `ei_model/`)

Your library structure should look like:
```
lib/
  ei_model/
    library.json (or library.properties)
    src/
      edge-impulse-sdk/
      model-parameters/
      tflite-model/
      ei_classifier_porting.h
      ...
```

### 3. Enable the Classifier

Add the following build flag to your `platformio.ini`:

```ini
[env:portenta_h7]
build_flags = 
    -O3 -LTO -DARDUINO_ARCH_MBED -DPORTENTA
    -DEI_CLASSIFIER_ENABLED
```

### 4. Configure Window Size (if needed)

If your model uses a different window size than 128 samples, update `ei_classifier.hpp`:

```cpp
constexpr size_t EI_WINDOW_SIZE = 128;  // Change to match your model
constexpr size_t EI_FEATURES_PER_SAMPLE = 6;  // accel(3) + gyro(3)
```

### 5. Build and Upload

```bash
pio run -t upload
```

## Runtime Commands

Use the TCP console to check classifier status:

| Command | Description |
|---------|-------------|
| `C` | Print classifier status (model info, inference stats) |
| `W` | Start logging (includes classifier results in stream) |
| `Q` | Stop logging |

## Streamed Data Format

When logging is enabled, each packet includes:

```c
struct imu_IMUFrameHR {
    // ... sensor data ...
    uint32_t classifier_label;      // Predicted class (0 to N-1)
    uint32_t classifier_confidence; // Confidence (0-100%)
};
```

## Memory Considerations

Edge Impulse models require significant resources:

| Resource | Typical Usage |
|----------|---------------|
| Flash | 100-500 KB (depends on model complexity) |
| RAM | 20-100 KB (inference buffer + model state) |

The Portenta H7 has:
- Flash: 2 MB
- RAM: 1 MB (M7 core)

## Troubleshooting

### Buffer Size Mismatch
If you see "Buffer size mismatch" error, ensure:
- `EI_WINDOW_SIZE` matches `EI_CLASSIFIER_RAW_SAMPLE_COUNT`
- `EI_FEATURES_PER_SAMPLE` matches model input features

### Inference Too Slow
If inference takes too long (>100ms at 100Hz):
- Consider quantizing the model (int8)
- Reduce model complexity in Edge Impulse
- Increase `EI_INFERENCE_INTERVAL` to run less frequently

### Model Not Loading
Verify:
1. `EI_CLASSIFIER_ENABLED` is defined
2. Library is in `lib/ei_model/`
3. Library has `library.json` or `library.properties`

## Stub Mode

When `EI_CLASSIFIER_ENABLED` is not defined, the classifier runs in stub mode:
- All functions work but return dummy values
- Useful for testing integration without a model
- No additional memory usage
