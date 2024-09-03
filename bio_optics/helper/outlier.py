import numpy as np
from scipy.stats import median_abs_deviation


def outlier_1D(arr, m=2):
    """
    Identifies outliers in a 1D array. 
    Outliers are considered all values outside +/- m * mean absolute deviations from the median.
    Returns binary mask where True is NO OUTLIER!
    Requires scipy.stats.median_abs_deviation.
    """
    return np.logical_and(
        abs(arr - np.median(arr)) < m * median_abs_deviation(arr),
        abs(arr + np.median(arr)) > m * median_abs_deviation(arr)
        )
                        
                        
def outlier_2D(arr, m=2, n=25, axis=1):
    """
    Identifies outliers in a 2D array.
    Outliers are considered all 1D arrays along axis 1 that have more than n values identified as outliers along axis 0 using is_outlier_1D().
    Returns binary mask where True is NO OUTLIER!
    Requires scipy.stats.median_abs_deviation.
    """
    return np.apply_along_axis(outlier_1D, axis, arr, m=m).sum(axis=0) > (arr.shape[0] - n) 