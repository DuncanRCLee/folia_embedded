# Rotation and Data Handling Comparison

## Summary
**The rotation implementations between `data_loader.py` and `analyze.py` are MOSTLY CONSISTENT with one critical difference:**

### Key Difference Found:
1. **Special case handling for `treadmill_02`** in `analyze.py` - this trial bypasses coordinate frame rotation
2. All other rotation logic is consistent

---

## Detailed Comparison

### 1. Quaternion Rotation (Coordinate Frame)

#### `data_loader.py` (Lines 70-89)
```python
def rotate_quaternion_to_coordinate_frame(
    q_w: float, q_x: float, q_y: float, q_z: float,
) -> tuple[float, float, float, float]:
    """Rotate quaternion to correct coordinate frame."""
    rotation = R.from_quat([q_x, q_y, q_z, q_w])
    combined_rotation = ENU_TO_WSU * rotation
    out = combined_rotation.as_quat()
    arr = np.asarray(out)
    return (float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))
```

**Applies:** `ENU_TO_WSU * rotation`
- Converts from ENU (East-North-Up) to WSU (West-South-Up) frame
- Returns quaternion in `(x, y, z, w)` order (scipy convention)
- Used for ALL trials

#### `analyze.py` (Lines 53-56, 82-93)
```python
def rotate_original_quaternion_to_coordinate_frame(q_w, q_x, q_y, q_z):
    rotation = R.from_quat([q_x, q_y, q_z, q_w])
    combined_rotation = enu_to_wsu * rotation
    return combined_rotation.as_quat()

# In augment_data():
if base_name == "treadmill_02":
    # SPECIAL CASE - NO FRAME ROTATION!
    flipped_quat = imu_df.apply(
        lambda row: (row["RAW:qx"], row["RAW:qy"], row["RAW:qz"], row["RAW:qw"]),
        axis=1
    )
else:
    # Normal case - applies frame rotation
    flipped_quat = imu_df.apply(
        lambda row: rotate_original_quaternion_to_coordinate_frame(...),
        axis=1
    )
```

**Difference:**
- `treadmill_02` gets NO coordinate frame transformation (just reorders quaternion)
- All other trials get the same `ENU_TO_WSU` transformation as `data_loader.py`

**Impact:** If you're loading `treadmill_02` data, the quaternions will differ between the two files!

---

### 2. Accelerometer Rotation

#### `data_loader.py` (Lines 44-59)
```python
def rotate_accelerometer_by_quaternion(
    acc_x: float, acc_y: float, acc_z: float,
    q_w: float, q_x: float, q_y: float, q_z: float
) -> tuple[float, float, float]:
    """Rotate accelerometer reading by quaternion and remove gravity."""
    rot_x, rot_y, rot_z = rotate_vector_by_quaternion(acc_x, acc_y, acc_z, q_w, q_x, q_y, q_z)
    return rot_x, rot_y, rot_z - G_SCALAR  # G_SCALAR = 9.81

# rotate_vector_by_quaternion (Lines 34-41):
rotation = R.from_quat([q_x, q_y, q_z, q_w])
rotated_vec = rotation.apply([v_x, v_y, v_z])
return float(rotated_vec[0]), float(rotated_vec[1]), float(rotated_vec[2])
```

#### `analyze.py` (Lines 48-51)
```python
def rotate_accelerometer_by_quaternion(acc_x, acc_y, acc_z, q_w, q_x, q_y, q_z):
    rotation = R.from_quat([q_x, q_y, q_z, q_w])
    rotated_acc = rotation.apply([acc_x, acc_y, acc_z])
    return rotated_acc[0], rotated_acc[1], rotated_acc[2] - g_scalar  # g_scalar = 9.81
```

**Status:** ✅ **IDENTICAL IMPLEMENTATION**
- Both use the same quaternion rotation
- Both subtract gravity from Z-axis: `rot_z - 9.81`
- Same quaternion order: `[q_x, q_y, q_z, q_w]`

---

### 3. Delta Velocity Rotation

#### `data_loader.py` (Lines 62-67)
```python
def rotate_delta_velocity_by_quaternion(
    dv_x: float, dv_y: float, dv_z: float,
    q_w: float, q_x: float, q_y: float, q_z: float
) -> tuple[float, float, float]:
    """Rotate delta-velocity vector by quaternion (no gravity removal)."""
    return rotate_vector_by_quaternion(dv_x, dv_y, dv_z, q_w, q_x, q_y, q_z)
```

#### `analyze.py`
**Status:** ❌ **NOT IMPLEMENTED**
- `analyze.py` does NOT rotate delta velocity
- This data is not used in `analyze.py`'s workflow

---

### 4. Coordinate Frame Constants

#### `data_loader.py` (Line 27)
```python
ENU_TO_WSU = R.from_matrix([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
```

#### `analyze.py` (Line 52)
```python
enu_to_wsu = R.from_matrix([[-1,0,0],[0,-1,0],[0,0,1]])
```

**Status:** ✅ **IDENTICAL**
- Both apply the same transformation matrix
- Flips X and Y, keeps Z unchanged

---

### 5. Time Lag Compensation

#### `data_loader.py` (Lines 92-152)
```python
def estimate_time_lag(vicon_df, imu_df, ...):
    # Uses cross-correlation on z-axis acceleration
    vicon_interp = interp1d(vicon_times, vicon_df['ava:ddz'], ...)
    imu_interp = interp1d(imu_times, (imu_df['rotated:a_z'] * 1000), ...)
    
    corr = correlate(gt_z - np.mean(gt_z), imu_z - np.mean(imu_z), mode='full')
    lags = np.arange(-len(gt_z) + 1, len(gt_z))
    best_lag_samples = int(lags[np.argmax(corr)])
    lag_sec = float(best_lag_samples) / 100.0
    return lag_sec
```

#### `analyze.py` (Lines 139-179)
```python
# In augment_data() function:
vicon_interp = interp1d(vicon_times, vicon_df['ava:ddz'], ...)
imu_interp = interp1d(imu_times, imu_df['rotated:a_z'] * 1000, ...)

corr = correlate(gt_z - np.mean(gt_z), imu_z - np.mean(imu_z), mode='full')
lags = np.arange(-len(gt_z) + 1, len(gt_z))
best_lag_samples = lags[np.argmax(corr)]
lag_sec = best_lag_samples / 100  # 100Hz
```

**Status:** ✅ **IDENTICAL ALGORITHM**
- Both use cross-correlation on z-axis acceleration
- Both interpolate to 100 Hz
- Both use same correlation method
- Minor type difference: `data_loader.py` uses `int()` cast, `analyze.py` doesn't

---

### 6. Vicon Data Processing

#### `data_loader.py` (Lines 384-424 in `load_trial_data`)
```python
# Dynamically detects subject prefix from columns
subject_prefix = None
for col in vicon_df.columns:
    if 'LIMULAT_X' in col:
        subject_prefix = col.split('LIMULAT_X')[0]
        break

# Calculates averaged positions
vicon_df['ava:x'] = (vicon_df[f'{subject_prefix}LIMULAT_X'] + 
                     vicon_df[f'{subject_prefix}LIMUMED_X']) / 2
```

#### `analyze.py` (Lines 115-127)
```python
# Hardcoded to 'ava:' prefix
vicon_df['ava:x'] = (vicon_df['ava:LIMULAT_X'] + vicon_df['ava:LIMUMED_X']) / 2
vicon_df['ava:y'] = (vicon_df['ava:LIMULAT_Y'] + vicon_df['ava:LIMUMED_Y']) / 2
vicon_df['ava:z'] = (vicon_df['ava:LIMULAT_Z'] + vicon_df['ava:LIMUMED_Z']) / 2
```

**Difference:**
- `data_loader.py`: **Dynamic prefix detection** (more flexible)
- `analyze.py`: **Hardcoded 'ava:' prefix** (less flexible)

**Both calculate the same averaged IMU position from lateral and medial markers.**

---

## Potential Issues & Recommendations

### 🔴 Critical Issue: `treadmill_02` Special Case

**Problem:**
```python
# analyze.py has this special case:
if base_name == "treadmill_02":
    # No coordinate frame rotation!
    flipped_quat = imu_df.apply(
        lambda row: (row["RAW:qx"], row["RAW:qy"], row["RAW:qz"], row["RAW:qw"]),
        axis=1
    )
```

**This means:**
- `treadmill_02` quaternions are NOT transformed to WSU frame in `analyze.py`
- All other trials ARE transformed to WSU frame
- `data_loader.py` transforms ALL trials (including `treadmill_02`)

**Recommendation:**
1. Determine if `treadmill_02` truly needs special handling
2. If yes, add the same logic to `data_loader.py`
3. If no, remove the special case from `analyze.py`
4. Add a comment explaining WHY this trial is different

---

### ⚠️ Minor Differences

1. **Subject prefix**: `data_loader.py` auto-detects, `analyze.py` hardcodes
   - **Recommendation:** Use the dynamic detection from `data_loader.py`

2. **Delta velocity**: Only `data_loader.py` rotates it
   - **Status:** OK if `analyze.py` doesn't use delta velocity

3. **Type casting**: Minor differences in float/int conversions
   - **Status:** Negligible impact

---

## Conclusion

**The rotation implementations are functionally identical EXCEPT:**

1. ❌ **`treadmill_02` is handled differently** - needs investigation
2. ✅ All other quaternion rotations are the same
3. ✅ Accelerometer rotations are the same
4. ✅ Gravity removal is the same
5. ✅ Time lag estimation is the same
6. ✅ Coordinate frame transformation (ENU→WSU) is the same

**Action Items:**
- [ ] Investigate why `treadmill_02` needs special handling
- [ ] Sync the special case logic between files or remove it
- [ ] Consider using dynamic subject prefix detection everywhere
