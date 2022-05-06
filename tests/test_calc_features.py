import glob
import json
import os
from pathlib import Path

import pandas as pd

import tsfel
from tsfel.feature_extraction.calc_features import (
    dataset_features_extractor,
    time_series_features_extractor,
)
from tsfel.feature_extraction.features_settings import (
    get_features_by_domain,
    get_features_by_tag,
)
from tsfel.utils.add_personal_features import add_feature_json
from tsfel.utils.signal_processing import merge_time_series, signal_window_splitter


# Example of user preprocess sensor data
def pre_process(sensor_data):
    if "Accelerometer" in sensor_data:
        sensor_data["Accelerometer"].iloc[:, 1] = (
            sensor_data["Accelerometer"].iloc[:, 1] * 10
        )
    return sensor_data


ROOT = Path(__file__).resolve().parents[1]

# DATASET DIR
main_directory = (
    ROOT.__str__()
    + os.sep
    + "tests"
    + os.sep
    + "tests_tools"
    + os.sep
    + "test_dataset"
    + os.sep
)

# JSON DIR
tsfel_path_json = (
    tsfel.__path__[0] + os.sep + "feature_extraction" + os.sep + "features.json"
)
personal_path_json = (
    ROOT.__str__()
    + os.sep
    + "tests"
    + os.sep
    + "tests_tools"
    + os.sep
    + "test_features.json"
)
personal_features_path = (
    ROOT.__str__() + os.sep + "tests" + os.sep + "test_personal_features.py"
)

# DEFAULT PARAM for testing
n_jobs = -1
time_unit = 1e9  # seconds
resample_rate = 30  # resample sampling frequency
window_size = 100  # number of points
overlap = 0  # varies between 0 and 1
search_criteria = ["Accelerometer.txt", "Gyroscope.txt"]
output_directory = ROOT.__str__() + os.sep + "output" + os.sep
sensor_data = {}
key = "Accelerometer"

folders = [f for f in glob.glob(main_directory + "**/", recursive=True)]
sensor_data[key] = pd.read_csv(folders[-1] + key + ".txt", header=None)

# add personal feature
# add_feature_json(personal_features_path, personal_path_json)

# Features Dictionary
settings0 = json.load(open(tsfel_path_json))
settings1 = json.load(open(personal_path_json))
settings2 = get_features_by_domain("statistical")
settings3 = get_features_by_domain("temporal")
settings4 = get_features_by_domain("spectral")
settings5 = get_features_by_domain()
settings6 = get_features_by_tag("inertial")
settings7 = get_features_by_tag()

# Signal processing
data_new = merge_time_series(sensor_data, resample_rate, time_unit)
windows = signal_window_splitter(data_new, window_size, overlap)


# time_series_features_extractor

# multi windows and multi axis
# input: list
features0 = time_series_features_extractor(settings5, windows, fs=resample_rate, n_jobs=n_jobs)

# multiple windows and single axis
# input: np.array
features1 = time_series_features_extractor(settings5, data_new.values[:, 0], fs=resample_rate, n_jobs=n_jobs, window_size=window_size, overlap=overlap)
# input: pd.series
features2 = time_series_features_extractor(settings5, data_new.iloc[:, 0], fs=resample_rate, n_jobs=n_jobs, window_size=window_size, overlap=overlap)

# single window and multi axis
# input: pd.DataFrame
features3 = time_series_features_extractor(settings5, data_new, fs=resample_rate, n_jobs=n_jobs)
# input: np.array
features4 = time_series_features_extractor(settings4, data_new.values, fs=resample_rate, n_jobs=n_jobs)

# single window and single axis
# input: pd.Series
features5 = time_series_features_extractor(settings1, data_new.iloc[:, 0], fs=resample_rate, n_jobs=n_jobs)
# input: np.array
features6 = time_series_features_extractor(settings4, data_new.values[:, 0], fs=resample_rate, n_jobs=n_jobs)

# Dataset features extractor
data = dataset_features_extractor(
    main_directory,
    settings1,
    search_criteria=search_criteria,
    time_unit=time_unit,
    resample_rate=resample_rate,
    window_size=window_size,
    overlap=overlap,
    pre_process=pre_process,
    output_directory=output_directory,
    features_path=personal_features_path,
)

print("-----------------------------------OK-----------------------------------")
