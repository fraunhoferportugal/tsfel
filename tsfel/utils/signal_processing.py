import numpy as np
import pandas as pd
import pandas_profiling
from scipy.interpolate import interp1d


def signal_window_spliter(signal, window_size, overlap):
    """Splits the signal into windows

    Parameters
    ----------
    signal : nd-array or pandas DataFrame
        input signal
    window_size :
        number of points of window size
    overlap :
        percentage of overlap, value between 0 and 1

    Returns
    -------
    list
        list of signal windows

    """

    step = int(round(window_size)) if overlap == 0 else int(round(window_size * (1 - overlap)))
    return [signal[i:i + window_size] for i in range(0, len(signal) - window_size, step)]


def merge_time_series(data, fs_resample, time_unit):
    """Time series data interpolation

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
    header_values = ['time']
    for k, dn in data.items():
        header_values += [k + str(i) for i in range(1, np.shape(dn)[1])]
        data_new = np.hstack((data_new, np.array([interp1d(dn.iloc[:, 0], dn.iloc[:, ax])(x_new) for ax in range(1, np.shape(dn)[1])]).T))

    return pd.DataFrame(data=data_new[:, 1:], columns=header_values[1:])


def correlation_report(features):
    """Performs a correlation report and removes highly correlated features.

    Parameters
    ----------
    features : DataFrame
        features

    Returns
    -------
    DataFrame
        Uncorrelated features

    """

    profile = pandas_profiling.ProfileReport(features)
    profile.to_file(output_file="CorrelationReport.html")
    inp = str(input('Do you wish to remove correlated features? Enter y/n: '))

    if inp == 'y':
        reject = profile.get_rejected_variables(threshold=0.9)
        if not list(reject):
            print('No features to remove')
        for rej in reject:
            print('Removing ' + str(rej))
            features = features.drop(rej, axis=1)
    return features
