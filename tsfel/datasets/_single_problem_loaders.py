"""Convenience methods to load single problem datasets.

Mainly for testing and supporting the documentation.
"""

import os
import warnings
import zipfile
from pathlib import Path
from remotezip import RemoteZip
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

import numpy as np
import pandas as pd
import requests

CACHE_ROOT_DIR = os.path.join(os.path.expanduser("~"), ".tsfel")


def _download_dataset(url, cache_dir, filename):
    """_Download a dataset and save it in the cache directory."""

    warnings.warn("Cache folder is empty. Downloading the dataset...", UserWarning)
    Path(os.path.join(cache_dir)).mkdir(parents=True, exist_ok=True)

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    file_path = os.path.join(cache_dir, filename)
    temp_path = file_path + ".part"

    try:
        with requests.get(url, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()

            with open(temp_path, "wb") as out_file:
                for chunk in response.iter_content(chunk_size=1024*1024):
                    if chunk:
                        out_file.write(chunk)

        os.replace(temp_path, file_path)

    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise

    return file_path


def _create_session():
    """
    Creates an HTTP sesion with automatic retries when the server temporarily fails or limits requestes.
    """

    retry_strategy = Retry(
        total=8,
        connect=5,
        read=5,
        status=8,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
        respect_retry_after_header=True
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session = requests.Session()
    session.mount("https://", adapter)

    return session


def load_biopluxecg(use_cache=True) -> pd.Series:
    """Load an example single-lead ECG signal collected using BioSignalsPlux.

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
    data_modality: list[pd.DataFrame],
    split: str,
) -> (list[pd.DataFrame], np.ndarray):

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
def load_ucihar(use_cache=True, data_modality=None) -> (list[pd.DataFrame], np.ndarray, list[pd.DataFrame], np.ndarray):
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
       Activity Recognition using Smartphones. The European Symposium on
       Artificial Neural Networks.
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


def _preprocess_aicoshar(df, sensor, common_time, start_time):
    """Interpolate sensor signals to 100 Hz and calculate magnitude."""

    df = df.copy()
    df = df.apply(pd.to_numeric, errors="coerce")

    # Convert timestamps from nanoseconds to seconds
    timestamps = df.iloc[:, 0].to_numpy(dtype=np.float64) / 1e9
    timestamps -= start_time
    values = df.iloc[:, 1:].to_numpy(dtype=np.float64)

    # Interpolate sensor channels
    interpolated = np.column_stack([np.interp(common_time, timestamps, values[:, i]) for i in range(values.shape[1])])

    if values.shape[1] == 3:
        columns = ["x", "y", "z"]
    elif sensor.lower() == "barometer":
        columns = ["pressure"]
    else:
        columns = [f"ch{i+1}" for i in range(values.shape[1])]

    processed = pd.DataFrame(interpolated, columns=[f"{sensor}_{col}" for col in columns])

    if values.shape[1] == 3:
        processed[f"{sensor}_mag"] = np.sqrt(np.sum(interpolated**2, axis=1))

    return processed


def _process_acquisition(sensor_data, as_numpy=False):
    """Process and combine all available sensors in one acquisition."""

    start_time = max(df.iloc[:, 0].min() for df in sensor_data.values()) / 1e9
    end_time = min(df.iloc[:, 0].max() for df in sensor_data.values()) / 1e9
    common_time = np.arange(0, end_time - start_time, 1 / 100)

    processed_sensors = []

    for sensor, df in sensor_data.items():
        processed = _preprocess_aicoshar(df, sensor, common_time, start_time)
        processed_sensors.append(processed)

    result = pd.concat(processed_sensors, axis=1)

    if as_numpy:
        return result.to_numpy(dtype=np.float32)

    return result.astype(np.float32)


def _select_acquisitions(metadata, available_files=None, subset=True):
    """Select acquisitions for training and testing."""

    metadata = metadata.copy()
    metadata["Activity"] = (metadata["AcquisitionID"].str.split("/").str[1].str.rsplit("_", n=1).str[0])

    if subset:
        metadata["AccelerometerPath"] = ("AICOS-HAR/" + metadata["AcquisitionID"] + "/Accelerometer.txt")
        metadata = metadata[metadata["AccelerometerPath"].isin(available_files)].copy()

    train_metadata = metadata[metadata["Fold"] != "test"].copy()
    test_metadata = metadata[metadata["Fold"] == "test"].copy()

    if subset:
        train_metadata = train_metadata.groupby("Activity", group_keys=False).sample(n=500, random_state=42)
        test_metadata = test_metadata.groupby("Activity", group_keys=False).sample(n=500, random_state=42)

    return train_metadata, test_metadata


def _process_aicoshar(source, metadata, subset=False):
    """Process AICOS-HAR and generate training and test splits."""

    X_train, y_train = [], []
    X_test, y_test = [], []

    # Identify available accelerometer files in RemoteZip
    if subset:
        available_files = {name for name in source.namelist() if name.endswith("/Accelerometer.txt")}
    else:
        available_files = None

    # Select acquisitions
    train_metadata, test_metadata = _select_acquisitions(metadata, available_files, subset=subset)

    for selected_metadata, X, y in [(train_metadata, X_train, y_train), (test_metadata, X_test, y_test)]:
        for _, row in selected_metadata.iterrows():
            try:
                if subset:
                    # RemoteZip: only read the accelerometer
                    with source.open(row["AccelerometerPath"]) as file:
                        sensor_data = {"accelerometer": pd.read_csv(file, header=None)}

                    data = _process_acquisition(sensor_data, as_numpy=True)

                else:
                    # Full dataset: read all sensor files
                    acquisition_dir = (source / row["AcquisitionID"])
                    sensor_files = list(acquisition_dir.glob("*.txt"))

                    if not sensor_files:
                        raise FileNotFoundError(f"No sensor files found in {acquisition_dir}")

                    sensor_data = {file.stem.lower(): pd.read_csv(file,header=None) for file in sensor_files}
                    data = _process_acquisition(sensor_data, as_numpy=False)

                X.append(data)
                y.append(row["Activity"])

            except Exception as error:
                warnings.warn(f"Could not process acquisition {row['AcquisitionID']}: {error}")

    return (X_train, np.array(y_train), X_test, np.array(y_test))


def load_aicoshar(download_full: bool = False):
    """Loads the AICOS-HAR human activity recognition dataset from Zenodo [1].
    The signals are resampled at 100Hz.

    Parameters
    ----------
    download_full : bool, default=False
        If False, remotely loads 10 accelerometer acquisitions
        per activity for each train/test split.

        If True, downloads and extracts the complete dataset,
        including all sensors and acquisitions, into the local cache.

    Returns
    -------
    X_train : list
        List of training acquisitions.

    y_train : ndarray
        Training activity labels.

    X_test : list
        List of test acquisitions.

    y_test : ndarray
        Test activity labels.

    Examples
    --------
    >>> from tsfel.datasets._single_problem_loaders import load_aicoshar
    >>> X_train, y_train, X_test, y_test = load_aicoshar()

    References
    ---------
    [1] Barandas, M., Pereira, A.C., Carvalho, C. et al. A human activity recognition 
    dataset spanning multiple devices and body positions with cross-dataset evaluation. 
    Sci Data (2026). https://doi.org/10.1038/s41597-026-08470-y

    """

    REF_URL = "https://zenodo.org/records/19452049/files/AICOS-HAR.zip?download=1"

    if not isinstance(download_full, bool):
        raise ValueError("download_full must be boolean.")

    cache_dir = Path(CACHE_ROOT_DIR) / "AICOSHAR"
    dataset_dir = cache_dir / "AICOS-HAR"
    zip_path = cache_dir / "AICOS-HAR.zip"
    extraction_marker = cache_dir / ".extraction_complete"

    # Option A: download and extract the complete dataset
    if download_full:
        metadata_path = dataset_dir / "metadata.csv"

        if not extraction_marker.exists() or not metadata_path.exists():
            cache_dir.mkdir(parents=True, exist_ok=True)

            if not zipfile.is_zipfile(zip_path):
                _download_dataset(REF_URL, str(cache_dir), zip_path.name)

            with zipfile.ZipFile(zip_path, "r") as zip_file:
                zip_file.extractall(cache_dir)

            extraction_marker.touch()
        metadata = pd.read_csv(metadata_path)
        return _process_aicoshar( dataset_dir, metadata, subset=False)

    # Option B: remotely access only a subset of accelerometer data
    session = _create_session()
    try:
        with RemoteZip(REF_URL, session=session, initial_buffer_size=4 * 1024 * 1024, timeout=(30, 120)) as zip_file:
            with zip_file.open("AICOS-HAR/metadata.csv") as file:
                metadata = pd.read_csv(file)
            return _process_aicoshar(zip_file, metadata, subset=True)

    finally:
        session.close()