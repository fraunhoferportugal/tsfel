from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def signal_window_splitter(signal, window_size, overlap=0):
    """Splits the signal into windows.

    Parameters
    ----------
    signal : nd-array or pandas DataFrame
        input signal
    window_size : int
        number of points of window size
    overlap : float
        percentage of overlap, value between 0 and 1 (exclusive)
        Default: 0
    Returns
    -------
    list
        list of signal windows
    """
    if not isinstance(window_size, int):
        raise SystemExit("window_size must be an integer.")
    step = int(round(window_size)) if overlap == 0 else int(round(window_size * (1 - overlap)))
    if step == 0:
        raise SystemExit(
            "Invalid overlap. " "Choose a lower overlap value.",
        )
    if len(signal) % window_size == 0 and overlap == 0:
        return [signal[i : i + window_size] for i in range(0, len(signal), step)]
    else:
        return [signal[i : i + window_size] for i in range(0, len(signal) - window_size + 1, step)]


def merge_time_series(data, fs_resample, time_unit):
    """Time series data interpolation.

    Parameters
    ----------
    data : dict
        data to interpolate
    fs_resample :
        resample sampling frequency
    time_unit :
        time unit in seconds

    Returns
    -------
    DataFrame
        Interpolated data
    """

    # time interval for interpolation
    sensors_time = np.array([[dn.iloc[0, 0], dn.iloc[-1, 0]] for k, dn in data.items()])
    t0 = np.max(sensors_time[:, 0])
    tn = np.min(sensors_time[:, 1])
    x_new = np.linspace(t0, tn, int((tn - t0) / ((1 / fs_resample) * time_unit)))

    # interpolation
    data_new = np.copy(x_new.reshape(len(x_new), 1))
    header_values = ["time"]
    for k, dn in data.items():
        header_values += [k + str(i) for i in range(1, np.shape(dn)[1])]
        data_new = np.hstack(
            (
                data_new,
                np.array(
                    [interp1d(dn.iloc[:, 0], dn.iloc[:, ax])(x_new) for ax in range(1, np.shape(dn)[1])],
                ).T,
            ),
        )

    return pd.DataFrame(data=data_new[:, 1:], columns=header_values[1:])


def correlated_features(
    features: pd.DataFrame,
    threshold: float = 0.95,
    method: str = "pearson",
    drop_correlated: bool = False,
) -> Union[List[str], Tuple[List[str], pd.DataFrame]]:
    """Identify and optionally remove highly correlated features from a
    DataFrame.

    This function computes the pairwise Pearson correlation of features using
    pandas.corr() and identifies features that have an absolute value of the
    correlation coefficient higher than the specified threshold. Different
    correlation methods supported by such as 'pearson', 'spearman', or
    'kendall'.

    .. deprecated:: 0.1.11

        tsfel.correlated_features will be deprecated in tsfel 0.1.11 and will be
        removed in other upcoming releases. It will be replaced by a future
        DropCorrelated feature class using fit and transform logic.

    Parameters
    ----------
    features : pd.DataFrame
        A DataFrame containing the feature data.
    threshold : float
        The correlation value for removing highly correlated features.
    method : str
        Method to compute correlation. Must be one of {'pearson', 'kendall', 'spearman'}
    drop_correlated : bool:
        If True, drop the highly correlated features from the DataFrame.

    Returns
    -------
    Union[List[str], Tuple[List[str], pd.DataFrame]]:
        - A list of names of highly correlated features.
        - If `drop_correlated` is True, a tuple containing the list of dropped feature names and the updated DataFrame with those features removed.
    """

    corr_matrix = features.corr(method=method).abs()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    # Find index and column name of features with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    if drop_correlated:
        features.drop(to_drop, axis=1, inplace=True)

        return to_drop, features

    else:
        return to_drop
