import numpy as np
from scipy.stats import median_abs_deviation


def outlier_1D(arr, m=2):
    """
    Identify outliers in a 1D array using the median absolute deviation.

    Values outside +/- m * MAD from the median are considered outliers.

    Args:
        arr: 1D input array
        m: number of MADs defining the outlier threshold, default: 2

    Returns:
        mask: boolean array where True indicates a non-outlier value
    """
    return np.logical_and(
        abs(arr - np.median(arr)) < m * median_abs_deviation(arr),
        abs(arr + np.median(arr)) > m * median_abs_deviation(arr)
        )


def outlier_2D(arr, m=2, n=25, axis=1):
    """
    Identify outlier rows/columns in a 2D array using outlier_1D.

    Columns (along axis 1) with more than n non-outlier values along axis 0 are retained.

    Args:
        arr: 2D input array
        m: number of MADs defining the outlier threshold for outlier_1D, default: 2
        n: minimum number of non-outlier values required per column, default: 25
        axis: axis along which outlier_1D is applied, default: 1

    Returns:
        mask: boolean array where True indicates a non-outlier column
    """
    return np.apply_along_axis(outlier_1D, axis, arr, m=m).sum(axis=0) > (arr.shape[0] - n)