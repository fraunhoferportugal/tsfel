import os
import sys
import glob
import numbers
import pathlib
import warnings
import importlib
import numpy as np
import pandas as pd
from pathlib import Path
from tsfel.utils.signal_processing import merge_time_series, signal_window_spliter


def dataset_features_extractor(main_directory, feat_dict, verbose=1, **kwargs):
    """Extracts features from a dataset.

    Parameters
    ----------
    main_directory : String
        Input directory
    feat_dict : dict
        Dictionary with features
    verbose : int
        Level of function communication
        (0 or 1 (Default))
    \**kwargs:
    See below:
        * *search_criteria* (``list``) --
            List of file names to compute features. (Example: '/Accelerometer.txt')
            (default: ``None``)

        * *time_unit* (``float``) --
            Time unit
            (default: ``1e9``)

        * *resampling_rate* (``int``) --
            Resampling rate
            (default: ``100``)

        * *window_size* (``int``) --
            Window size in number of samples
            (default: ``100``)

        * *overlap* (``float``) --
            Overlap between 0 and 1
            (default: ``0``)

        * *pre_process* (``function``) --
            Function with pre processing code

            (default: ``None``)

        * *output_directory* (``String``) --
            Output directory
            (default: ``'output_directory', str(Path.home()) + '/tsfel_output'``)

    Returns
    -------
    file
        csv file with the extracted features

    """
    search_criteria = kwargs.get('search_criteria', None)
    time_unit = kwargs.get('time_unit', 1e9)
    resample_rate = kwargs.get('resample_rate', 30)
    window_size = kwargs.get('window_size', 100)
    overlap = kwargs.get('overlap', 0)
    pre_process = kwargs.get('pre_process', None)
    output_directory = kwargs.get('output_directory', str(Path.home()) + '/tsfel_output')
    features_path = kwargs.get('features_path', None)

    if main_directory[-1] != os.sep:
        main_directory = main_directory+os.sep

    folders = [f for f in glob.glob(main_directory + "**/", recursive=True)]

    for fl in folders:
        sensor_data = {}
        if search_criteria:
            for c in search_criteria:
                if os.path.isfile(fl + c):
                    key = c.split('.')[0]
                    sensor_data[key] = pd.read_csv(fl + c, header=None)
        else:
            all_files = np.concatenate((glob.glob(fl + '/*.txt'), glob.glob(fl + '/*.csv')))
            for c in all_files:
                key = c.split(os.sep)[-1].split('.')[0]
                try:
                    data_file = pd.read_csv(c, header=None)
                except pd.io.common.CParserError:
                    continue

                if np.dtype('O') in np.array(data_file.dtypes):
                    continue

                sensor_data[key] = pd.read_csv(c, header=None)

        if not sensor_data:
            continue

        pp_sensor_data = sensor_data if pre_process is None else pre_process(sensor_data)

        data_new = merge_time_series(pp_sensor_data, resample_rate, time_unit)

        windows = signal_window_spliter(data_new, window_size, overlap)

        if features_path:
            features = time_series_features_extractor(feat_dict, windows, fs=resample_rate, verbose=0, features_path=features_path)
        else:
            features = time_series_features_extractor(feat_dict, windows, fs=resample_rate, verbose=0)

        pathlib.Path(output_directory + fl).mkdir(parents=True, exist_ok=True)
        features.to_csv(output_directory + fl + '/Features.csv', sep=',', encoding='utf-8')

    if verbose == 1:
        print('Features files saved in: ', output_directory)


def time_series_features_extractor(dict_features, signal_windows, fs=None, window_spliter=False, verbose=1, **kwargs):
    """Extraction of time series features.

    Parameters
    ----------
    dict_features : dict
        Dictionary with features
    signal_windows: list
        Input from which features are computed, window
    fs : int or None
        Sampling frequency
    window_spliter: bool
        If True computes the signal windows
    verbose : int
        Level of function communication
        (0 or 1 (Default))
    \**kwargs:
    See below:
        * *window_size* (``int``) --
            Window size in number of samples
            (default: ``100``)

        * *overlap* (``float``) --
            Overlap between 0 and 1
            (default: ``0``)

        * *features_path* (``string``) --
            Directory of script with personal features

    Returns
    -------
    DataFrame
        Extracted features

    """

    if verbose == 1:
        print("*** Feature extraction started ***")

    window_size = kwargs.get('window_size', 100)
    overlap = kwargs.get('overlap', 0)
    features_path = kwargs.get('features_path', None)

    feat_val = pd.DataFrame()
    if window_spliter:
        signal_windows = signal_window_spliter(signal_windows, window_size, overlap)

    if isinstance(signal_windows, pd.DataFrame):
        features = calc_window_features(dict_features, signal_windows, fs, features_path=features_path)
        feat_val = feat_val.append(features)
    else:
        for wind_sig in signal_windows:
            if isinstance(wind_sig, numbers.Real):
                features = calc_window_features(dict_features, signal_windows, fs, features_path=features_path)
                feat_val = feat_val.append(features)
                break
            else:
                features = calc_window_features(dict_features, wind_sig, fs, features_path=features_path)
                feat_val = feat_val.append(features)

    if verbose == 1:
        print("*** Feature extraction finished ***")

    return feat_val.reset_index(drop=True)


def calc_window_features(dict_features, signal_window, fs, **kwargs):
    """This function computes features matrix for one window.

    Parameters
    ----------
    dict_features : dict
        Dictionary with features
    signal_window: pandas DataFrame
        Input from which features are computed, window
    fs : int
        Sampling frequency
    \**kwargs:
    See below:
        * *features_path* (``string``) --
            Directory of script with personal features

    Returns
    -------
    pandas DataFrame
        (columns) names of the features
        (data) values of each features for signal

    """

    features_path = kwargs.get('features_path', None)

    # Execute imports
    exec("import tsfel")
    domain = dict_features.keys()

    if features_path:
        sys.path.append(features_path[:-len(features_path.split(os.sep)[-1])-1])
        exec("import "+features_path.split(os.sep)[-1][:-3])
        importlib.reload(sys.modules[features_path.split(os.sep)[-1][:-3]])
        exec("from " + features_path.split(os.sep)[-1][:-3]+" import *")

    # Create global arrays
    func_total = []
    func_names = []
    imports_total = []
    parameters_total = []
    feature_results = []
    feature_names = []

    for _type in domain:
        domain_feats = dict_features[_type].keys()

        for feat in domain_feats:

            # Only returns used functions
            if dict_features[_type][feat]['use'] == 'yes':

                # Read Function Name (generic name)
                func_names = [feat]

                # Read Function (real name of function)
                func_total = [dict_features[_type][feat]['function']]

                # Check for parameters
                if dict_features[_type][feat]['parameters'] != '':
                    param = dict_features[_type][feat]['parameters']

                    # Check assert fs parameter:
                    if 'fs' in param:

                        # Select which fs to use
                        if fs is None:

                            # Check if features dict has default sampling frequency value
                            if type(param['fs']) is int or type(param['fs']) is float:
                                parameters_total = [str(key) + '=' + str(value) for key, value in param.items()]

                                # raise a warning
                                warnings.warn('Using default sampling frequency: '+str(param['fs'])+" Hz.", stacklevel=2)
                            else:
                                raise SystemExit('No sampling frequency assigned.')
                        else:
                            parameters_total = [str(key) + '=' + str(value) for key, value in param.items()
                                                if key not in 'fs']
                            parameters_total += ['fs =' + str(fs)]

                    # feature has no fs parameter
                    else:
                        parameters_total = []
                        for key, value in param.items():
                            if type(value) is str:
                                value = '"'+value+'"'
                            parameters_total.append([str(key) + '=' + str(value)])
                else:
                    parameters_total = ''

                # Name of each column to be concatenate with feature name
                if not isinstance(signal_window, pd.DataFrame):
                    signal_window = pd.DataFrame(data=signal_window)
                header_names = signal_window.columns.values

                for ax in range(len(header_names)):
                    window = signal_window.iloc[:, ax]
                    execf = func_total[0] + '(window'

                    if parameters_total != '':
                        execf += ', ' + str(parameters_total).translate(str.maketrans({'[': '', ']': '', "'": ''}))

                    execf += ')'
                    eval_result = eval(execf, locals())

                    # Function returns more than one element
                    if type(eval_result) == tuple:
                        for rr in range(len(eval_result)):
                            if np.isnan(eval_result[0]):
                                eval_result = np.zeros(len(eval_result))
                            feature_results += [eval_result[rr]]
                            feature_names += [str(header_names[ax]) + '_' + func_names[0] + '_' + str(rr)]
                    else:
                        feature_results += [eval_result]
                        feature_names += [str(header_names[ax]) + '_' + func_names[0]]

    features = pd.DataFrame(data=np.array(feature_results).reshape(1, len(feature_results)),
                            columns=np.array(feature_names))

    return features
