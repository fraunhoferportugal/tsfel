import os
import json
import glob
import tsfel
import pandas as pd
from pathlib import Path


# Example of user preprocess sensor data
def pre_process(sensor_data):
    if "Accelerometer" in sensor_data:
        sensor_data["Accelerometer"].iloc[:, 1] = sensor_data["Accelerometer"].iloc[:, 1] * 10
    return sensor_data


# DATASET DIR
main_directory = "tests_tools" + os.sep + "test_dataset" + os.sep

# JSON DIR
# tsfel_path_json = tsfel.__path__[0] + os.sep + "feature_extraction" + os.sep + "features.json"
personal_path_json = "tests_tools" + os.sep + "test_features.json"
personal_features_path = "tests_tools" + os.sep + "test_personal_features.py"

# DEFAULT PARAM for testing
time_unit = 1e9  # seconds
resample_rate = 30  # resample sampling frequency
window_size = 100  # number of points
overlap = 0  # varies between 0 and 1
search_criteria = ["Accelerometer.txt", "Gyroscope.txt"]
output_directory = str(Path.home()) + os.sep + "Documents" + os.sep + "tsfel_output" + os.sep
sensor_data = {}
key = "Accelerometer"

folders = [f for f in glob.glob(main_directory + "**/", recursive=True)]
sensor_data[key] = pd.read_csv(folders[-1] + key + ".txt", header=None)

# add personal feature
# tsfel.add_feature_json(personal_features_path, personal_path_json)

# Features Dictionary
# settings0 = json.load(open(tsfel_path_json))
settings1 = json.load(open(personal_path_json))
settings2 = tsfel.get_features_by_domain("statistical")
settings3 = tsfel.get_features_by_domain("temporal")
settings4 = tsfel.get_features_by_domain("spectral")
settings5 = tsfel.get_features_by_domain()
# settings6 = tsfel.extract_sheet("Features")
# settings7 = tsfel.extract_sheet("Features_test", path_json=personal_path_json)
# settings8 = tsfel.get_features_by_tag("inertial")
# settings10 = tsfel.get_features_by_tag()

# Signal processing
data_new = tsfel.merge_time_series(sensor_data, resample_rate, time_unit)
windows = tsfel.signal_window_splitter(data_new, window_size, overlap)

n_jobs = -1
# time_series_features_extractor

# multi windows and multi axis
# input: list
features0 = tsfel.time_series_features_extractor(settings5, windows, fs=resample_rate, n_jobs=n_jobs)

# multiple windows and single axis
# input: np.array
features1 = tsfel.time_series_features_extractor(settings5, data_new.values[:, 0], fs=resample_rate, n_jobs=n_jobs, window_size=window_size, overlap=overlap)
# input: pd.series
features2 = tsfel.time_series_features_extractor(settings5, data_new.iloc[:, 0], fs=resample_rate, n_jobs=n_jobs, window_size=window_size, overlap=overlap)

# single window and multi axis
# input: pd.DataFrame
features3 = tsfel.time_series_features_extractor(settings5, data_new, fs=resample_rate, n_jobs=n_jobs)

# single window and single axis
# input: pd.Series
features4 = tsfel.time_series_features_extractor(settings1, data_new.iloc[:, 0], fs=resample_rate, n_jobs=n_jobs, features_path=personal_features_path)
# input: np.array
features5 = tsfel.time_series_features_extractor(settings4, data_new.values[:, 0], fs=resample_rate, n_jobs=n_jobs)

# Dataset features extractor
data = tsfel.dataset_features_extractor(
    main_directory,
    settings4,
    search_criteria=search_criteria,
    time_unit=time_unit,
    resample_rate=resample_rate,
    window_size=window_size,
    overlap=overlap,
    pre_process=pre_process,
    output_directory=output_directory,
)
print("-----------------------------------OK-----------------------------------")
