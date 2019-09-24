import os
import glob
import numbers
import pathlib
import pandas as pd
import numpy as np
from tsfel.utils.signal_processing import merge_time_series, signal_window_spliter


def dataset_features_extractor(main_directory, feat_dict, **kwargs):
    """

    :param main_directory:
    :param search_criteria:
    :param feat_dict:
    :param kwargs:
    :return:
    """

    search_criteria = kwargs.get('search_criteria', None)
    time_unit = kwargs.get('time_unit', 1e9)
    resample_rate = kwargs.get('resample_rate', 30)
    window_size = kwargs.get('window_size', 100)
    overlap = kwargs.get('overlap', 0)
    pre_process = kwargs.get('pre_process', None)
    output_directory = kwargs.get('output_directory', None)

    folders = [f for f in glob.glob(main_directory + "**/", recursive=True)]

    for fl in folders:
        sensor_data = {}
        if search_criteria:
            for c in search_criteria:
                if os.path.isfile(fl + c):
                    key = c.split('.')[0]
                    sensor_data[key] = pd.read_csv(fl+c, header=None)
        else:
            all_files = np.concatenate((glob.glob(fl + '/*.txt'), glob.glob(fl + '/*.csv')))
            for c in all_files:
                key = c.split(os.sep)[-1].split('.')[0]
                sensor_data[key] = pd.read_csv(fl, header=None)

        pp_sensor_data = sensor_data if pre_process is None else pre_process(sensor_data)

        data_new = merge_time_series(pp_sensor_data, resample_rate, time_unit)

        windows = signal_window_spliter(data_new, window_size, overlap)

        features = time_series_features_extractor(feat_dict, windows, fs=resample_rate)

        pathlib.Path(output_directory + fl).mkdir(parents=True, exist_ok=True)
        features.to_csv(output_directory + fl + '/Features.csv', sep=',', encoding='utf-8')

        print('Features saved')


def time_series_features_extractor(dictionary, signal_windows, fs=100):
    """

    :param dictionary: dictionary with selected features from json file
    :param signal_windows: list of signal windows
    :param ts_id: time series id to be concatenated with feature name
    :param fs: sampling frequency
    :return: features values for each window size
    """
    feat_val = pd.DataFrame()
    if isinstance(signal_windows[0], numbers.Real):
        signal_windows = [signal_windows]
    print("*** Feature extraction started ***")
    for wind_sig in signal_windows:
        features = calc_window_features(dictionary, wind_sig, fs)
        feat_val = feat_val.append(features)
    print("*** Feature extraction finished ***")

    return feat_val


def calc_window_features(dictionary, signal_window, fs):
    """
    This function computes features matrix for one window.
    :param dictionary: (json file)
           list of features
    :param signal_window: (pandas DataFrame)
           input from which features are computed, window.
    :param fs: (int)
           sampling frequency    :return: res: (narray-like)
             values of each features for signal.
             nam: (narray-like)
             names of the features
    """
    domain = dictionary.keys()

    # Create global arrays
    func_total = []
    func_names = []
    imports_total = []
    parameters_total = []
    free_total = []

    for atype in domain:
        domain_feats = dictionary[atype].keys()

        for feat in domain_feats:
            # Only returns used functions
            if dictionary[atype][feat]['use'] == 'yes':

                # Read Function Name (generic name)
                func_names += [feat]

                # Read Function (real name of function)
                func_total += [dictionary[atype][feat]['function']]

                # Read Parameters
                parameters_total += [dictionary[atype][feat]['parameters']]

                # Read Free Parameters
                free_total += [dictionary[atype][feat]['free parameters']]

    # Execute imports
    exec("import tsfel")

    # Name of each column to be concatenate with feature name
    if not isinstance(signal_window, pd.DataFrame):
        signal_window = pd.DataFrame(data=signal_window)
    header_names = signal_window.columns.values

    feature_results = []
    feature_names = []

    for ax in range(len(header_names)):
        window = signal_window.iloc[:, ax]
        for i in range(len(func_total)):

            execf = func_total[i] + '(window'

            if parameters_total[i] != '':
                execf += ', ' + parameters_total[i]

            if free_total[i] != '':
                for n, v in free_total[i].items():
                    # TODO: conversion may loose precision (str)
                    execf += ', ' + n + '=' + str(v)

            execf += ')'

            eval_result = eval(execf, locals())

            # Function returns more than one element
            if type(eval_result) == tuple:
                for rr in range(len(eval_result)):
                    if np.isnan(eval_result[0]):
                        eval_result = np.zeros(len(eval_result))
                    feature_results += [eval_result[rr]]
                    feature_names += [str(header_names[ax]) + '_' + func_names[i] + '_' + str(rr)]
            else:
                feature_results += [eval_result]
                feature_names += [str(header_names[ax]) + '_' + func_names[i]]

    feature_results = np.array(feature_results)
    features = pd.DataFrame(data=feature_results.reshape(1, len(feature_results)), columns=feature_names)
    return features


