"""Feature extraction must not depend on a Series' observation labels."""

import unittest

import numpy as np
import pandas as pd

import tsfel


class TestSeriesIndex(unittest.TestCase):
    def setUp(self):
        self.config = {
            "statistical": {
                "Mean": {"function": "tsfel.calc_mean", "parameters": "", "use": "yes"},
            },
        }
        self.values = np.array([1.0, 3.0, 5.0, 7.0])

    def extract(self, signal, **kwargs):
        return tsfel.time_series_features_extractor(self.config, signal, fs=100, verbose=0, n_jobs=None, **kwargs)

    def check_series(self, index):
        signal = pd.Series(self.values, index=index)
        original = signal.copy()
        for window_size, overlap, means in [(None, 0, [4.0]), (2, 0, [2.0, 6.0]), (2, 0.5, [2.0, 4.0, 6.0])]:
            with self.subTest(window_size=window_size, overlap=overlap):
                expected = pd.DataFrame({"0_Mean": means})
                actual = self.extract(signal, window_size=window_size, overlap=overlap)
                pd.testing.assert_frame_equal(actual, expected)
                pd.testing.assert_series_equal(signal, original)

    def test_offset_integer_index(self):
        self.check_series(pd.RangeIndex(10, 14))

    def test_nonconsecutive_integer_index(self):
        self.check_series([11, 13, 17, 19])

    def test_default_integer_index(self):
        self.check_series(pd.RangeIndex(4))

    def test_datetime_index(self):
        self.check_series(pd.date_range("2020-01-01", periods=4))

    def test_string_index(self):
        self.check_series(["a", "b", "c", "d"])

    def test_explicit_header_with_offset_index(self):
        signal = pd.Series(self.values, index=range(10, 14))
        actual = self.extract(signal, window_size=2, header_names=["sensor"])
        pd.testing.assert_frame_equal(actual, pd.DataFrame({"sensor_Mean": [2.0, 6.0]}))

    def test_array_and_list_inputs(self):
        for signal in [self.values, self.values.tolist()]:
            with self.subTest(input_type=type(signal)):
                pd.testing.assert_frame_equal(self.extract(signal), pd.DataFrame({"0_Mean": [4.0]}))

    def test_dataframe_window_headers(self):
        signal = pd.DataFrame({"z": self.values, "a": self.values + 10}, index=range(10, 14))
        expected = pd.DataFrame({"a_Mean": [12.0, 16.0], "z_Mean": [2.0, 6.0]})
        pd.testing.assert_frame_equal(self.extract(signal, window_size=2), expected)
        pd.testing.assert_frame_equal(self.extract([signal.iloc[:2], signal.iloc[2:]]), expected)


if __name__ == "__main__":
    unittest.main()
