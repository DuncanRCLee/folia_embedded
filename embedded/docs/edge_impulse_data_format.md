# Edge Impulse Data Format Documentation

## Overview

This document describes the sensor data streamed by the Folia embedded system for use in Edge Impulse classifier training and on-device inference.

## Streamed Data Fields

The data is streamed as protobuf-encoded packets (`imu_IMUFrameHR`) at **100 Hz** (as configured in `main.cpp`).

### IMU Data (from Xsens MTi)

| Field | Type | Units | Description |
|-------|------|-------|-------------|
| `a_x`, `a_y`, `a_z` | float32 | m/s² | Linear acceleration (3-axis) |
| `w_x`, `w_y`, `w_z` | float32 | rad/s | Angular rate / rate of turn (3-axis) |
| `qw`, `qx`, `qy`, `qz` | float32 | unitless | Orientation quaternion (scalar-first format) |
| `dv_x`, `dv_y`, `dv_z` | float32 | m/s | Delta velocity (integrated acceleration over sample period) |
| `dq_w`, `dq_x`, `dq_y`, `dq_z` | float32 | unitless | Delta quaternion (orientation change over sample period) |

### Folia-Specific Fields

| Field | Type | Units | Description |
|-------|------|-------|-------------|
| `motor_state` | uint32 | enum | Motor state: 0=OFF, 1=IDLE, 2=COASTING, 3=DAMPING, 4=SPINNING |
| `adc_raw` | int32 | counts | Raw ADC reading from strain gauge (ADS1220) |
| `adc_baseline` | int32 | counts | Calibrated baseline for ADC (set during calibration) |

### Classifier Output Fields

| Field | Type | Units | Description |
|-------|------|-------|-------------|
| `classifier_label` | uint32 | index | Predicted class label (0 to N-1) |
| `classifier_confidence` | uint32 | percent | Confidence score (0-100) |

### Timing Fields

| Field | Type | Units | Description |
|-------|------|-------|-------------|
| `time_milis` | int64 | ms | System time in milliseconds |
| `sample_time_fine` | uint32 | 0.1 ms | MTi 10kHz counter (high-resolution timing) |

## Sample Rate

- IMU data: **100 Hz** (configurable in `main.cpp` via `configureOutputsAdv`)
- ADC data: Up to **1000 SPS** (sampled at ADC rate, included in each 100Hz packet)

## Data Format for Edge Impulse

### Recommended Input Features

For CNN-based motion classification, the recommended input features are:

1. **Accelerometer** (3 axes): `a_x`, `a_y`, `a_z`
2. **Gyroscope** (3 axes): `w_x`, `w_y`, `w_z`
3. **ADC** (optional): `adc_raw - adc_baseline` for strain sensing

**Total features per sample**: 6 (accel + gyro) or 7 (with normalized ADC)

### Window Configuration

Edge Impulse CNN models typically use sliding windows:

- **Window size**: 1-2 seconds (100-200 samples at 100Hz)
- **Window stride**: 50-100% overlap recommended
- **Input shape**: `[window_size, num_features]` e.g., `[128, 6]`

### Data Normalization

For Edge Impulse training:
- Accelerometer: Typical range ±8g to ±16g depending on motion
- Gyroscope: Typical range ±500 deg/s to ±2000 deg/s
- ADC: Normalize using `(adc_raw - adc_baseline) / scale_factor`

## Exporting Data for Training

The streamed protobuf packets can be decoded using the corresponding `.proto` file and converted to:
- **CSV format** for Edge Impulse Data Forwarder
- **JSON format** for Edge Impulse Ingestion API
- **CBOR format** for bulk upload

## On-Device Inference

After training in Edge Impulse:
1. Export the model as **C++ library** (Arduino format)
2. Place the generated files in `lib/ei_model/`
3. Use the `ei_classifier` module to run inference

See `src/app/ei_classifier.hpp` for integration details.
