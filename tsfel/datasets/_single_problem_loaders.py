"""Convenience methods to load single problem datasets.

Mainly for testing and supporting the documentation.
"""

import os
import warnings
import zipfile
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import requests

CACHE_ROOT_DIR = os.path.join(os.path.expanduser("~"), ".tsfel")


def _download_dataset(url, cache_dir, filename):
    warnings.warn("Cache folder is empty. Downloading the dataset...", UserWarning)
    Path(os.path.join(cache_dir)).mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        response = requests.get(
            url,
            stream=True,
            allow_redirects=False,
        )

        if response.status_code == 200:
            with open(os.path.join(cache_dir, filename), "wb") as out_file:
                out_file.write(response.content)

    except Exception as e:
        print(f"An error occurred: {e}")


def load_biopluxecg(use_cache=True) -> pd.Series:
    """Load an example single-lead ECG signal colected using BioSignalsPlux.

    Parameters
    ----------
    use_cache : bool, default=True
        If True, caches a local copy of the dataset in the user's home directory.

    Returns
    -------
    X : np.ndarray
        The time series data.

    Notes
    -----
    The signal is sampled at 1000 Hz and has a duration of 10 s.

    Examples
    --------
    >>> from tsfel.datasets import load_biopluxecg
    >>> X = load_biopluxecg()
    """

    REF_URL = (
        "https://raw.githubusercontent.com/hgamboa/novainstrumentation/master/novainstrumentation/data/cleanecg.txt"
    )
    cache_dir = os.path.join(CACHE_ROOT_DIR, "BioPluxECG")

    if not os.path.exists(cache_dir) or not os.listdir(cache_dir) or not use_cache:
        _download_dataset(REF_URL, cache_dir, "biopluxecg.txt")

    X = np.loadtxt(os.path.join(cache_dir, "biopluxecg.txt"))[1]
    X = pd.Series(X, name="LeadII")

    return X


def _get_uci_train_test_splits(
    dataset_dir: str,
    data_modality: List[pd.DataFrame],
    split: str,
) -> (List[pd.DataFrame], np.ndarray):

    raw_signals_split_dir = os.path.join(dataset_dir, "UCI HAR Dataset", split, "Inertial Signals")
    raw_signals_split_label_path = os.path.join(dataset_dir, "UCI HAR Dataset", split, f"y_{split}.txt")
    _, _, filenames = next(os.walk(raw_signals_split_dir), (None, None, []))

    columns = [
        filename[: -len(f"{split}.txt")] if filename.endswith(f"{split}.txt") else filename for filename in filenames
    ]
    filtered_columns = [
        col for col in columns if not data_modality or any(substring in col for substring in data_modality)
    ]

    data = {col: np.loadtxt(os.path.join(raw_signals_split_dir, f"{col}{split}.txt")) for col in columns}
    X = [
        pd.DataFrame({col: data[col][i] for col in filtered_columns}) for i in range(data[filtered_columns[0]].shape[0])
    ]
    y = np.loadtxt(raw_signals_split_label_path)

    return X, y


def get_uci_splits(cache_dir, data_modality):
    return (_get_uci_train_test_splits(cache_dir, data_modality, split) for split in ["train", "test"])


# TODO: Write a parser for this dataset.
def load_ucihar(use_cache=True, data_modality=None) -> (List[pd.DataFrame], np.ndarray, List[pd.DataFrame], np.ndarray):
    """Loads the Human Activity Recognition Using Smartphones dataset from the
    UC Irvine Machine Learning Repository [1]_. Retrieves the raw inertial data
    for both the training and test sets.

    Parameters
    ----------
    use_cache: bool, default=True
        If True, caches a local copy of the dataset in the user's home directory.

    data_modality: None or list of data modalities, default=None
        If set to None, all available data modalities are loaded. Otherwise,
        only the specified modalities are loaded. The supported modalities are
        defined as "body_acc," "body_gyro," and "total_acc".

    Returns
    -------
    X_train : list
        A list of DataFrames containing windows of multivariate time series
        from the training set. The number of channels (columns) in each
        DataFrame depends on the `data_modality`.

    y_train : ndarray
        The corresponding labels for the training set.

    X_test : list
        A list of DataFrames containing windows of multivariate time series
        from the test set. The number of channels (columns) in each
        DataFrame depends on the `data_modality`.

    y_test : ndarray
        The corresponding labels for the test set.

    Notes
    -----
    The signal is sampled at 100 Hz, and it's divided in short fixed-size
    windows.

    .. versionadded:: 0.1.8

    Examples
    --------
    >>> from tsfel.datasets import load_ucihar
    >>> X_train, y_train, X_test, y_test = load_ucihar()

    References
    ----------
    .. [1] Anguita, D., et. al (2013). A Public Domain Dataset for Human
    Activity Recognition using Smartphones. The European Symposium on Artificial
    Neural Networks.
    """

    data_modality = [] if data_modality is None else data_modality
    cache_dir = os.path.join(CACHE_ROOT_DIR, "UCIHAR")
    REF_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"

    if not isinstance(data_modality, list):
        raise ValueError("data_modality must be a list of strings.")

    if not all(
        any(element.startswith(prefix) for prefix in ["body_acc", "body_gyro", "total_acc"])
        for element in data_modality
    ):
        raise ValueError("Elements of the list should be `body_acc`, `body_gyro`, or `total_acc`")

    if not os.path.exists(cache_dir) or not os.listdir(cache_dir) or not use_cache:
        _download_dataset(REF_URL, cache_dir, "ucihar.zip")

        zip_file_path = os.path.join(cache_dir, "ucihar.zip")
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(cache_dir)
            os.remove(zip_file_path)

    (X_train, y_train), (X_test, y_test) = get_uci_splits(cache_dir, data_modality)

    return X_train, y_train, X_test, y_test
