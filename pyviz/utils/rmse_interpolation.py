import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

def calculate_rmse_with_interpolation(left_df: pd.DataFrame, right_df: pd.DataFrame, left_time: pd.Series, right_time: pd.Series, data_col1: str, data_col2: str):
    """
    Calculate RMSE between two datasets with UTC time alignment and interpolation.

    Args:
        left_df (pd.DataFrame): First dataset with data column.
        right_df (pd.DataFrame): Second dataset with data column.
        left_time (pd.Series): Time column for the first dataset.
        right_time (pd.Series): Time column for the second dataset.
        data_col1 (str): Column name for the data in the first dataset.
        data_col2 (str): Column name for the data in the second dataset.

    Returns:
        float: The RMSE between the two datasets after alignment and interpolation.
    """
    # Ensure the time columns are in datetime format
    # dataset1[time_col] = pd.to_datetime(dataset1[time_col])
    # dataset2[time_col] = pd.to_datetime(dataset2[time_col])

    # Find the time intersection
    start_time = max(left_time.min(), right_time.min())
    end_time = min(left_time.max(), right_time.max())

    # Filter datasets to the intersection time range
    filtered_left_df = left_df[(left_time >= start_time) & (left_time <= end_time)]
    filtered_right_df = right_df[(right_time >= start_time) & (right_time <= end_time)]

    if filtered_left_df.empty or filtered_right_df.empty:
        raise ValueError("Filtered datasets have no overlapping time range.")

    # Sort right_time and right_df to ensure proper interpolation
    sorted_indices = np.argsort(right_time)
    right_time = right_time.iloc[sorted_indices]
    filtered_right_df = filtered_right_df.iloc[sorted_indices]

    # Interpolate right_df to match left_df's time points
    interp_func = interp1d(
        right_time.values.astype(np.float64),
        filtered_right_df[data_col2].values,
        kind='linear',
        fill_value="extrapolate",
        assume_sorted=True
    )
    interpolated_data2 = interp_func(left_time.values.astype(np.float64))

    # Calculate RMSE
    rmse = np.sqrt(np.mean((np.array(filtered_left_df[data_col1], dtype=np.float64) - np.array(interpolated_data2, dtype=np.float64)) ** 2))
    return rmse
