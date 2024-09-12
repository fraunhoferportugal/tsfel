import glob
import json
import os
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from tsfel.feature_extraction.calc_features import dataset_features_extractor, time_series_features_extractor
from tsfel.feature_extraction.features_settings import get_features_by_domain, get_number_features, load_json
from tsfel.utils.add_personal_features import add_feature_json
from tsfel.utils.signal_processing import merge_time_series, signal_window_splitter


# Example of user preprocess sensor data
def pre_process(sensor_data):
    if "Accelerometer" in sensor_data:
        sensor_data["Accelerometer"].iloc[:, 1] = sensor_data["Accelerometer"].iloc[:, 1] * 10
    return sensor_data


# DATASET DIR
main_directory = os.path.join("tests", "tests_tools", "test_dataset", "")

# JSON DIR
# tsfel_path_json = tsfel.__path__[0] + os.sep + "feature_extraction" + os.sep + "features.json"
personal_path_json = os.path.join("tests", "tests_tools", "test_features.json")
personal_features_path = os.path.join(
    "tests",
    "tests_tools",
    "test_personal_features.py",
)

# DEFAULT PARAM for testing
time_unit = 1e9  # seconds
resample_rate = 30  # resample sampling frequency
window_size = 100  # number of points
overlap = 0  # varies between 0 and 1
search_criteria = ["Accelerometer.txt", "Gyroscope.txt"]
output_directory = str(Path.home()) + os.sep + "Documents" + os.sep + "tsfel_output" + os.sep
sensor_data = {}
key = "Accelerometer"
domains = ["statistical", "temporal", "spectral", "fractal"]

folders = list(glob.glob(main_directory + "**/", recursive=True))
sensor_data[key] = pd.read_csv(folders[-1] + key + ".txt", header=None)


# Features Dictionary
# settings0 = json.load(open(tsfel_path_json))
settings2 = get_features_by_domain(domains[0])  # statistical
settings3 = get_features_by_domain(domains[1])  # temporal
settings4 = get_features_by_domain(domains[2])  # spectral
settings5 = get_features_by_domain(domains[3])  # fractal
settings6 = get_features_by_domain()

# Signal processing
data_new = merge_time_series(sensor_data, resample_rate, time_unit)
data_new = data_new[data_new.columns[:-1]]
windows = signal_window_splitter(data_new, window_size, overlap)

n_jobs = 1


class TestCalcFeatures(unittest.TestCase):
    def test_input_list_window_multi_axis_multi(self):
        # multi windows and multi axis
        # input: list
        features0 = time_series_features_extractor(
            settings6,
            windows,
            fs=resample_rate,
            n_jobs=n_jobs,
        )
        np.testing.assert_array_equal(
            features0.shape,
            (16, 468),
        )

    def test_input_array_window_tosplit_axis_single(self):
        # multiple windows and single axis
        # input: np.array
        features1 = time_series_features_extractor(
            settings6,
            data_new.values[:, 0],
            fs=resample_rate,
            n_jobs=n_jobs,
            window_size=window_size,
            overlap=overlap,
        )

        np.testing.assert_array_equal(
            features1.shape,
            (16, 156),
        )

    def test_input_series_window_tosplit_axis_single(self):
        # input: pd.series
        features2 = time_series_features_extractor(
            settings6,
            data_new.iloc[:, 0],
            fs=resample_rate,
            n_jobs=n_jobs,
            window_size=window_size,
            overlap=overlap,
        )

        np.testing.assert_array_equal(
            features2.shape,
            (16, 156),
        )

    def test_input_dataframe_window_single_axis_multi(self):
        # single window and multi axis
        # input: pd.DataFrame
        features3 = time_series_features_extractor(
            settings3,
            data_new,
            fs=resample_rate,
            n_jobs=n_jobs,
        )

        np.testing.assert_array_equal(
            features3.shape,
            (1, 42),
        )

    def test_input_array_window_single_axis_multi(self):
        # input: np.array
        features4 = time_series_features_extractor(
            settings4,
            data_new.values,
            fs=resample_rate,
            n_jobs=n_jobs,
        )

        np.testing.assert_array_equal(
            features4.shape,
            (1, 333),
        )

    def test_input_series_window_single_axis_single(self):
        # single window and single axis
        # input: pd.Series
        features5 = time_series_features_extractor(
            settings2,
            data_new.iloc[:, 0],
            fs=resample_rate,
            n_jobs=n_jobs,
        )

        np.testing.assert_array_equal(
            features5.shape,
            (1, 31),
        )

    def test_input_array_window_single_axis_single(self):
        # input: np.array
        features6 = time_series_features_extractor(
            settings5,
            data_new.values[:, 0],
            fs=resample_rate,
            n_jobs=n_jobs,
        )

        np.testing.assert_array_equal(
            features6.shape,
            (1, 6),
        )

    def test_personal_features(self):
        # personal features
        settings1 = load_json(personal_path_json)
        add_feature_json(personal_features_path, personal_path_json)
        features7 = time_series_features_extractor(
            settings1,
            data_new.values[:, 0],
            fs=resample_rate,
            n_jobs=n_jobs,
            features_path=personal_features_path,
        )

        np.testing.assert_array_equal(
            features7.shape,
            (1, 160),
        )

    def test_get_number_features(self):
        feature_sets_size = [get_number_features(get_features_by_domain(domain)) for domain in domains]
        np.testing.assert_array_equal(
            feature_sets_size,
            [31, 14, 124, 6],  # 184 is the total
        )

    def test_dataset_extractor(self):

        # Dataset features extractor
        dataset_features_extractor(
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


if __name__ == "__main__":
    unittest.main()
