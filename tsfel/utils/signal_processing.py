import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def signal_window_spliter(signal, window_size, overlap):
    """

    :param signal: input signal (array or pandas data frame)
    :param window_size: number of points of window size
    :param overlap: percentage of overlap. Value between 0 and 1
    :return: list of signal windows
    """
    step = int(round(window_size)) if overlap == 0 else int(round(window_size * (1 - overlap)))
    return [signal[i:i + window_size] for i in range(0, len(signal) - window_size, step)]


def time_series_interpolation(data, fs_resample, time_unit):

    x_new = np.linspace(data.iloc[0, 0], data.iloc[-1, 0], int((data.iloc[-1, 0] - data.iloc[0, 0]) / ((1 / fs_resample) * time_unit)))

    data_new = np.array([interp1d(data.iloc[:, 0], data.iloc[:, ax])(x_new) for ax in range(1, np.shape(data)[1])]).T

    return pd.DataFrame(data=data_new)


def merge_time_series(data, fs_resample, time_unit):

    # time interval for interpolation
    sensors_time = np.array([[dn.iloc[0, 0], dn.iloc[-1, 0]] for dn in data])
    t0 = np.max(sensors_time[:, 0])
    tn = np.min(sensors_time[:, 1])
    x_new = np.linspace(t0, tn, int((tn - t0) / ((1 / fs_resample) * time_unit)))

    # interpolation
    data_new = np.copy(x_new.reshape(len(x_new), 1))
    for dn in data:
        data_new = np.hstack((data_new, np.array([interp1d(dn.iloc[:, 0], dn.iloc[:, ax])(x_new) for ax in range(1, np.shape(dn)[1])]).T))

    return pd.DataFrame(data=data_new)
