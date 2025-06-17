import glob
import importlib
import importlib.util
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
from tsfel.feature_extraction import features as tsfel_features



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


def _calc_features(timeseries, config, fs, **kwargs):
    """Extraction of time series features.

    Parameters
    ----------
    timeseries: list
        The input signal window from which features will be extracted.
    config : dict
        A dictionary containing the settings for feature extraction.
    fs : float, default=None
        Sampling frequency of the input signal.
    \**kwargs:
        Additional keyword arguments, see below:

        * *features_path* (str) --
            Path to a script with custom features.
        
        * *header_names* (list or array-like) --
            Names of each column window.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the extracted features.
    """

    features_path = kwargs.get("features_path", None)
    names = kwargs.get("header_names", None)
    feat_val = calc_window_features(
        config,
        timeseries,
        fs,
        features_path=features_path,
        header_names=names,
    )
    feat_val.reset_index(drop=True)

    return feat_val


def time_series_features_extractor(
    config,
    timeseries,
    fs=None,
    window_size=None,
    overlap=0,
    verbose=1,
    **kwargs,
):
    """Extract features from univariate or multivariate time series.

    Parameters
    ----------
    config : dict
        A dictionary containing the settings for feature extraction.
    timeseries : list, np.ndarray, pd.DataFrame, pd.Series
        The input signal from which features will be extracted.
    fs : float, default=None
        Sampling frequency of the input signal.
    window_size : int or None, optional, default=None
        The size of the windows used to split the input signal, measured in the number of samples.
    overlap : float, optional, default=0
        A value between 0 and 1 that defines the percentage of overlap between consecutive windows.
    n_jobs : int, optional
        The number of jobs to run in parallel. 
            - ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            - ``-1`` means using all available processors.
            - default: ``None`` on Windows, ``-1`` for other systems
    verbose : int, default=1
        The verbosity mode. 0 means silent, and 1 means showing a progress bar.

    **kwargs :
        Additional keyword arguments, see below:

        * *features_path* (str) --
            Path to a script with custom features.
        
        * *header_names* (list or array-like) --
            Names of each column window.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the extracted features, where:
        - Columns represent the names of the features.
        - Rows contain the feature values for each signal window.
    """

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
        if isinstance(timeseries, pd.DataFrame):
            names = timeseries.columns.values
        elif isinstance(timeseries[0], pd.DataFrame):
            names = timeseries[0].columns.values

    if window_size is not None:
        timeseries = signal_window_splitter(timeseries, window_size, overlap)

    if len(timeseries) == 0:
        raise SystemExit(
            "Empty signal windows. Please check window size input parameter.",
        )

    features_final = pd.DataFrame()

    if isinstance(timeseries, list) and isinstance(timeseries[0], numbers.Real):
        timeseries = np.array(timeseries)

    if not isinstance(timeseries, list) and len(np.shape(timeseries)) > 2:
        timeseries = list(timeseries)

    # more than one window
    if isinstance(timeseries, list):
        # Starting the display of progress bar for notebooks interfaces
        if (get_ipython().__class__.__name__ == "ZMQInteractiveShell") or (get_ipython().__class__.__name__ == "Shell"):

            out = display(
                progress_bar_notebook(0, len(timeseries)),
                display_id=True,
            )
        else:
            out = None

        if isinstance(n_jobs, int):
            # Determine number of processes
            cpu_count = mp.cpu_count() if n_jobs == -1 else n_jobs

            ctx = mp.get_context("spawn")
            pool = ctx.Pool(cpu_count)

            features = pool.imap(
                partial(
                    _calc_features,
                    config=config,
                    fs=fs,
                    features_path=features_path,
                    header_names=names,
                ),
                timeseries,
            )

            for i, feat in enumerate(features):
                if verbose == 1:
                    display_progress_bar(i, len(timeseries), out)
                features_final = pd.concat([features_final, feat], axis=0)

            pool.close()
            pool.join()

        elif n_jobs is None:
            for i, feat in enumerate(timeseries):
                features_final = pd.concat(
                    [
                        features_final,
                        calc_window_features(
                            config,
                            feat,
                            fs,
                            features_path=features_path,
                            header_names=names,
                        ),
                    ],
                    axis=0,
                )
                if verbose == 1:
                    display_progress_bar(i, len(timeseries), out)
        else:
            raise SystemExit(
                "n_jobs value is not valid. " "Choose an integer value or None for no multiprocessing.",
            )
    # single window
    else:
        features_final = calc_window_features(
            config,
            timeseries,
            fs,
            verbose=verbose,
            features_path=features_path,
            header_names=names,
            single_window=True,
        )

    # Assuring the same feature extraction order
    features_final = features_final.reindex(sorted(features_final.columns), axis=1)
    return features_final.reset_index(drop=True)


def calc_window_features(
    config,
    window,
    fs,
    verbose=1,
    single_window=False,
    **kwargs,
):
    """Extract features from a univariate or multivariate window.

    Parameters
    ----------
    config : dict
        A dictionary containing the settings for feature extraction.
    window : np.ndarray, pd.DataFrame, pd.Series
        The input signal from which features will be extracted.
    fs : float, default=None
        Sampling frequency of the input signal.
    verbose : int, default=1
        The verbosity mode. 0 means silent, and 1 means showing a progress bar.
    single_window : bool
        If `True`, the progress bar will be shown only for the extraction of features from a single window.

    **kwargs :
        Additional keyword arguments, see below:

        * *features_path* (str) --
            Path to a script with custom features.
        
        * *header_names* (list or array-like) --
            Names of each column window.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the extracted features.
    """
    
    features_path = kwargs.get("features_path", None)
    header_names = kwargs.get("header_names", None)

    # To handle object type signals
    window = np.array(window).astype(float)

    single_axis = True if len(window.shape) == 1 else False

    if header_names is None:
        header_names = np.array([0]) if single_axis else np.arange(window.shape[-1])
    else:
        if (len(header_names) != window.shape[-1] and not single_axis) or (
            len(header_names) != 1 and single_axis
        ):
            raise Exception("header_names dimension does not match input columns.")

    # Execute imports
    exec("from tsfel import *")
    domain = config.keys()

    # Load all available functions
    feature_funcs = load_combined_feature_modules(features_path)

    # Create global arrays
    feature_results = []
    feature_names = []

    # Starting the display of progress bar for notebooks interfaces
    # Iterating over features of a single window
    if verbose == 1 and single_window:

        feat_nb = np.hstack([list(config[_type].keys()) for _type in domain])

        if (get_ipython().__class__.__name__ == "ZMQInteractiveShell") or (get_ipython().__class__.__name__ == "Shell"):
            out = display(progress_bar_notebook(0, len(feat_nb)), display_id=True)
        else:
            out = None

        i_feat = -1

    for _type in domain:
        domain_feats = config[_type].keys()

        for feat in domain_feats:

            if verbose == 1 and single_window:
                i_feat = i_feat + 1
                display_progress_bar(i_feat, len(feat_nb), out)

            # Only returns used functions
            if config[_type][feat]["use"] == "yes":

                # Read Function (real name of function)
                func_total = config[_type][feat]["function"]

                if func_total.find("tsfel.") == 0:
                    func_total = func_total.replace("tsfel.", "")

                # Check for parameters
                parameters_total = {}

                if config[_type][feat]["parameters"] != "":
                    parameters_total = config[_type][feat]["parameters"]

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
                    eval_result = feature_funcs[func_total](
                        window,
                        **parameters_total,
                    )
                    eval_result = np.array([eval_result])

                for ax in range(len(header_names)):
                    sig_ax = window if single_axis else window[:, ax]
                    eval_result_ax = feature_funcs[func_total](sig_ax, **parameters_total)
                    # Function returns more than one element
                    if isinstance(eval_result_ax, tuple):
                        eval_result_ax = (
                            np.zeros(len(eval_result_ax)) if np.isnan(eval_result_ax[0]) else eval_result_ax
                        )
                        for rr, value in enumerate(eval_result_ax):
                            feature_results.append(value)
                            feature_names.append(f"{header_names[ax]}_{feat}_{rr}")

                    elif isinstance(eval_result_ax, dict):
                        names = eval_result_ax["names"]
                        values = eval_result_ax["values"]
                        eval_result_ax = np.zeros(len(values)) if np.isnan(values[0]) else eval_result_ax
                        for name, value in zip(names, values):
                            feature_results.append(value)
                            feature_names.append(f"{header_names[ax]}_{feat}_{name}")
                    else:
                        feature_results += [eval_result_ax]
                        feature_names += [str(header_names[ax]) + "_" + feat]

    features = pd.DataFrame(
        data=np.array(feature_results).reshape(1, len(feature_results)),
        columns=np.array(feature_names),
    )

    return features


def load_combined_feature_modules(features_path=None):
    """
    Load feature functions from both the TSFEL features module and an optional user module.

    Parameters
    ----------
    features_path : str or None
        Path to user-defined Python feature module (.py).
        If None, only the default TSFEL features module is used.

    Returns
    -------
    dict
        A dictionary mapping function names to function objects from both sources.
        User-defined functions override TSFEL ones with the same name.
    """

    # Start with default TSFEL functions
    feature_funcs = {
        name: obj for name, obj in tsfel_features.__dict__.items() if callable(obj)
    }

    if features_path:
        module_name = os.path.splitext(os.path.basename(features_path))[0]
        spec = importlib.util.spec_from_file_location(module_name, features_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module from {features_path}")

        user_module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = user_module
        spec.loader.exec_module(user_module)

        # Add or override with user-defined functions
        user_funcs = {
            name: obj for name, obj in user_module.__dict__.items() if callable(obj)
        }
        feature_funcs.update(user_funcs)

    return feature_funcs