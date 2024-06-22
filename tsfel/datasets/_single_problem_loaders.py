"""Convenience methods to load single problem datasets.

Mainly for testing and supporting the documentation.
"""

import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

CACHE_ROOT_DIR = os.path.expanduser("~/.tsfel")


def _download_dataset(url, cache_dir, filename):
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
        print("Cache folder is empty. Downloading the dataset...")
        Path(os.path.join(cache_dir)).mkdir(
            parents=True,
            exist_ok=True,
        )
        _download_dataset(REF_URL, cache_dir, "biopluxecg.txt")

    X = np.loadtxt(os.path.join(cache_dir, "biopluxecg.txt"))[1]
    X = pd.Series(X, name="LeadII")

    return X


# TODO: Write a parser for this dataset.
def load_ucihar(use_cache=True):
    """Loads the Human Activity Recognition Using Smartphones dataset from the
    UC Irvine Machine Learning Repository [1]_.

    Parameters
    ----------
    use_cache: bool, default=True
        If True, caches a local copy of the dataset in the user's home directory.


    Notes
    -----
    The signal is sampled at 100 Hz and its divided in short fixed-size windows.

    .. versionadded:: 0.1.8

    Examples
    --------
    >>> from tsfel.datasets import load_ucihar
    >>> X = load_ucihar()

    References
    ----------
    .. [1] Anguita, D., Ghio, A., Oneto, L., Parra, X., & Reyes-Ortiz, J.L. (2013). A Public Domain Dataset for Human Activity Recognition using Smartphones. The European Symposium on Artificial Neural Networks.
    """

    REF_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
    cache_dir = os.path.join(CACHE_ROOT_DIR, "UCIHAR")

    if not os.path.exists(cache_dir) or not os.listdir(cache_dir) or not use_cache:
        print("Cache folder is empty. Downloading the dataset...")
        Path(os.path.join(cache_dir)).mkdir(
            parents=True,
            exist_ok=True,
        )
        _download_dataset(REF_URL, cache_dir, "ucihar.zip")

    zip_file_path = os.path.join(cache_dir, "ucihar.zip")
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(cache_dir)
        os.remove(zip_file_path)
