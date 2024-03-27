import glob
import importlib
import multiprocessing as mp
import numbers
import os
import pathlib
import sys
import warnings
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from IPython import get_ipython
from IPython.display import display

from tsfel.utils.progress_bar import display_progress_bar, progress_bar_notebook
from tsfel.utils.signal_processing import merge_time_series, signal_window_splitter


def dataset_features_extractor(main_directory, feat_dict, verbose=1, **kwargs):
    r"""Extracts features from a dataset.

    Parameters
    ----------
    main_directory : String
        Input directory
    feat_dict : dict
        Dictionary with features
    verbose : int
        Verbosity mode. 0 = silent, 1 = progress bar.
        (0 or 1 (Default))
    \**kwargs:
    See below:
        * *search_criteria* (``list``) --
            List of file names to compute features. (Example: 'Accelerometer.txt')
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

        * *features_path* (``string``) --
            Directory of script with personal features

        * *header_names* (``list or array``) --
            Names of each column window

        * *n_jobs* (``int``) --
            The number of jobs to run in parallel. ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors.
            (default: ``None`` in Windows and ``-1`` for other systems)

    Returns
    -------
    file
        csv file with the extracted features
    """
    search_criteria = kwargs.get("search_criteria", None)
    time_unit = kwargs.get("time_unit", 1e9)
    resample_rate = kwargs.get("resample_rate", 30)
    window_size = kwargs.get("window_size", 100)
    overlap = kwargs.get("overlap", 0)
    pre_process = kwargs.get("pre_process", None)
    output_directory = kwargs.get(
        "output_directory",
        str(Path.home()) + "/tsfel_output",
    )
    features_path = kwargs.get("features_path", None)
    names = kwargs.get("header_names", None)

    # Choosing default of n_jobs by operating system
    if sys.platform[:-2] == "win":
        n_jobs_default = None
    else:
        n_jobs_default = -1

    # Choosing default of n_jobs by python interface
    if get_ipython().__class__.__name__ == "ZMQInteractiveShell" or get_ipython().__class__.__name__ == "Shell":
        n_jobs_default = -1

    n_jobs = kwargs.get("n_jobs", n_jobs_default)

    if main_directory[-1] != os.sep:
        main_directory = main_directory + os.sep

    folders = list(glob.glob(main_directory + "**/", recursive=True))

    if folders:
        for fl in folders:
            sensor_data = {}
            if search_criteria:
                for c in search_criteria:
                    if os.path.isfile(fl + c):
                        key = c.split(".")[0]
                        sensor_data[key] = pd.read_csv(fl + c, header=None)
            else:
                all_files = np.concatenate(
                    (glob.glob(fl + "/*.txt"), glob.glob(fl + "/*.csv")),
                )
                for c in all_files:
                    key = c.split(os.sep)[-1].split(".")[0]
                    try:
                        data_file = pd.read_csv(c, header=None)
                    except pd.io.common.CParserError:
                        continue

                    if np.dtype("O") in np.array(data_file.dtypes):
                        continue

                    sensor_data[key] = pd.read_csv(c, header=None)

            if not sensor_data:
                continue

            pp_sensor_data = sensor_data if pre_process is None else pre_process(sensor_data)

            data_new = merge_time_series(pp_sensor_data, resample_rate, time_unit)

            windows = signal_window_splitter(data_new, window_size, overlap)

            if features_path:
                features = time_series_features_extractor(
                    feat_dict,
                    windows,
                    fs=resample_rate,
                    verbose=0,
                    features_path=features_path,
                    header_names=names,
                    n_jobs=n_jobs,
                )
            else:
                features = time_series_features_extractor(
                    feat_dict,
                    windows,
                    fs=resample_rate,
                    verbose=0,
                    header_names=names,
                    n_jobs=n_jobs,
                )

            fl = "/".join(fl.split(os.sep))
            invalid_char = r'<>:"\|?* '
            for char in invalid_char:
                fl = fl.replace(char, "")

            pathlib.Path(output_directory + fl).mkdir(parents=True, exist_ok=True)
            features.to_csv(
                output_directory + fl + "/Features.csv",
                sep=",",
                encoding="utf-8",
            )

        if verbose == 1:
            print("Features files saved in: ", output_directory)
    else:
        raise FileNotFoundError("There is no folder(s) in directory: " + main_directory)


def calc_features(wind_sig, dict_features, fs, **kwargs):
    r"""Extraction of time series features.

    Parameters
    ----------
    wind_sig: list
        Input from which features are computed, window
    dict_features : dict
        Dictionary with features
    fs : float or None
        Sampling frequency
    \**kwargs:
        * *features_path* (``string``) --
            Directory of script with personal features

         * *header_names* (``list or array``) --
            Names of each column window

    Returns
    -------
    DataFrame
        Extracted features
    """

    features_path = kwargs.get("features_path", None)
    names = kwargs.get("header_names", None)
    feat_val = calc_window_features(
        dict_features,
        wind_sig,
        fs,
        features_path=features_path,
        header_names=names,
    )
    feat_val.reset_index(drop=True)

    return feat_val


def time_series_features_extractor(
    dict_features,
    signal_windows,
    fs=None,
    verbose=1,
    **kwargs,
):
    r"""Extraction of time series features.

    Parameters
    ----------
    dict_features : dict
        Dictionary with features
    signal_windows: list
        Input from which features are computed, window
    fs : int or None
        Sampling frequency
    verbose : int
        Verbosity mode. 0 = silent, 1 = progress bar.
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

        * *header_names* (``list or array``) --
            Names of each column window

        * *n_jobs* (``int``) --
            The number of jobs to run in parallel. ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors.
            (default: ``None`` in Windows and ``-1`` for other systems)

    Returns
    -------
    DataFrame
        Extracted features
    """
    if verbose == 1:
        print("*** Feature extraction started ***")

    window_size = kwargs.get("window_size", None)
    overlap = kwargs.get("overlap", 0)
    features_path = kwargs.get("features_path", None)
    names = kwargs.get("header_names", None)

    # Choosing default of n_jobs by operating system
    if sys.platform[:-2] == "win":
        n_jobs_default = None
    else:
        n_jobs_default = -1

    # Choosing default of n_jobs by python interface
    if get_ipython().__class__.__name__ == "ZMQInteractiveShell" or get_ipython().__class__.__name__ == "Shell":
        n_jobs_default = -1

    n_jobs = kwargs.get("n_jobs", n_jobs_default)

    if fs is None:
        warnings.warn(
            "Using default sampling frequency set in configuration file.",
            stacklevel=2,
        )

    if names is not None:
        names = list(names)
    else:
        # Name of each column to be concatenated with feature name
        if isinstance(signal_windows, pd.DataFrame):
            names = signal_windows.columns.values
        elif isinstance(signal_windows[0], pd.DataFrame):
            names = signal_windows[0].columns.values

    if window_size is not None:
        signal_windows = signal_window_splitter(signal_windows, window_size, overlap)

    if len(signal_windows) == 0:
        raise SystemExit(
            "Empty signal windows. Please check window size input parameter.",
        )

    features_final = pd.DataFrame()

    if isinstance(signal_windows, list) and isinstance(signal_windows[0], numbers.Real):
        signal_windows = np.array(signal_windows)

    if len(np.shape(signal_windows)) > 2:
        signal_windows = list(signal_windows)

    # more than one window
    if isinstance(signal_windows, list):
        # Starting the display of progress bar for notebooks interfaces
        if (get_ipython().__class__.__name__ == "ZMQInteractiveShell") or (get_ipython().__class__.__name__ == "Shell"):

            out = display(
                progress_bar_notebook(0, len(signal_windows)),
                display_id=True,
            )
        else:
            out = None

        if isinstance(n_jobs, int):
            # Multiprocessing use
            if n_jobs == -1:
                cpu_count = mp.cpu_count()
            else:
                cpu_count = n_jobs

            pool = mp.Pool(cpu_count)
            features = pool.imap(
                partial(
                    calc_features,
                    dict_features=dict_features,
                    fs=fs,
                    features_path=features_path,
                    header_names=names,
                ),
                signal_windows,
            )

            for i, feat in enumerate(features):
                if verbose == 1:
                    display_progress_bar(i, len(signal_windows), out)
                features_final = pd.concat([features_final, feat], axis=0)

            pool.close()
            pool.join()

        elif n_jobs is None:
            for i, feat in enumerate(signal_windows):
                features_final = pd.concat(
                    [
                        features_final,
                        calc_window_features(
                            dict_features,
                            feat,
                            fs,
                            features_path=features_path,
                            header_names=names,
                        ),
                    ],
                    axis=0,
                )
                if verbose == 1:
                    display_progress_bar(i, len(signal_windows), out)
        else:
            raise SystemExit(
                "n_jobs value is not valid. " "Choose an integer value or None for no multiprocessing.",
            )
    # single window
    else:
        features_final = calc_window_features(
            dict_features,
            signal_windows,
            fs,
            verbose=verbose,
            features_path=features_path,
            header_names=names,
            single_window=True,
        )

    if verbose == 1:
        print("\n" + "*** Feature extraction finished ***")

    # Assuring the same feature extraction order
    features_final = features_final.reindex(sorted(features_final.columns), axis=1)
    return features_final.reset_index(drop=True)


def calc_window_features(
    dict_features,
    signal_window,
    fs,
    verbose=1,
    single_window=False,
    **kwargs,
):
    r"""This function computes features matrix for one window.

    Parameters
    ----------
    dict_features : dict
        Dictionary with features
    signal_window: pandas DataFrame
        Input from which features are computed, window
    fs : float
        Sampling frequency
    verbose : int
        Level of function communication
        (0 or 1 (Default))
    single_window: Bool
        Boolean value for printing the progress bar for only one window feature extraction
    \**kwargs:
    See below:
        * *features_path* (``string``) --
            Directory of script with personal features
        * *header_names* (``list or array``) --
            Names of each column window

    Returns
    -------
    pandas DataFrame
        (columns) names of the features
        (data) values of each features for signal
    """

    features_path = kwargs.get("features_path", None)
    header_names = kwargs.get("header_names", None)

    # To handle object type signals
    signal_window = np.array(signal_window).astype(float)

    single_axis = True if len(signal_window.shape) == 1 else False

    if header_names is None:
        header_names = np.array([0]) if single_axis else np.arange(signal_window.shape[-1])
    else:
        if (len(header_names) != signal_window.shape[-1] and not single_axis) or (
            len(header_names) != 1 and single_axis
        ):
            raise Exception("header_names dimension does not match input columns.")

    # Execute imports
    exec("from tsfel import *")
    domain = dict_features.keys()

    if features_path:
        sys.path.append(features_path[: -len(features_path.split(os.sep)[-1]) - 1])
        exec("import " + features_path.split(os.sep)[-1][:-3])
        importlib.reload(sys.modules[features_path.split(os.sep)[-1][:-3]])
        exec("from " + features_path.split(os.sep)[-1][:-3] + " import *")

    # Create global arrays
    feature_results = []
    feature_names = []

    # Starting the display of progress bar for notebooks interfaces
    # Iterating over features of a single window
    if verbose == 1 and single_window:

        feat_nb = np.hstack([list(dict_features[_type].keys()) for _type in domain])

        if (get_ipython().__class__.__name__ == "ZMQInteractiveShell") or (get_ipython().__class__.__name__ == "Shell"):
            out = display(progress_bar_notebook(0, len(feat_nb)), display_id=True)
        else:
            out = None

        i_feat = -1

    for _type in domain:
        domain_feats = dict_features[_type].keys()

        for feat in domain_feats:

            if verbose == 1 and single_window:
                i_feat = i_feat + 1
                display_progress_bar(i_feat, len(feat_nb), out)

            # Only returns used functions
            if dict_features[_type][feat]["use"] == "yes":

                # Read Function (real name of function)
                func_total = dict_features[_type][feat]["function"]

                if func_total.find("tsfel.") == 0:
                    func_total = func_total.replace("tsfel.", "")

                # Check for parameters
                parameters_total = {}

                if dict_features[_type][feat]["parameters"] != "":
                    parameters_total = dict_features[_type][feat]["parameters"]

                    # Check assert fs parameter:
                    if "fs" in parameters_total:

                        # Select which fs to use
                        if fs is None:
                            # Check if features dict has default sampling frequency value
                            if not (type(parameters_total["fs"]) is int or type(parameters_total["fs"]) is float):
                                raise Exception("No sampling frequency assigned.")
                        else:
                            parameters_total["fs"] = fs

                # Eval feature results
                if single_axis:
                    eval_result = locals()[func_total](
                        signal_window,
                        **parameters_total,
                    )
                    eval_result = np.array([eval_result])

                for ax in range(len(header_names)):
                    sig_ax = signal_window if single_axis else signal_window[:, ax]
                    eval_result_ax = locals()[func_total](sig_ax, **parameters_total)
                    # Function returns more than one element
                    if isinstance(eval_result_ax, tuple):
                        if np.isnan(eval_result_ax[0]):
                            eval_result_ax = np.zeros(len(eval_result_ax))
                        for rr in range(len(eval_result_ax)):
                            feature_results += [eval_result_ax[rr]]
                            feature_names += [
                                str(header_names[ax]) + "_" + feat + "_" + str(rr),
                            ]
                    else:
                        feature_results += [eval_result_ax]
                        feature_names += [str(header_names[ax]) + "_" + feat]

    features = pd.DataFrame(
        data=np.array(feature_results).reshape(1, len(feature_results)),
        columns=np.array(feature_names),
    )

    return features
