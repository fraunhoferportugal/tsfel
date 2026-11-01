"""A test suite for the signal processing methods.

The name will likely change after the major refactor.
"""

import unittest

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

import tsfel


class SignalProcessingTestCase(unittest.TestCase):
    """Unit tests for signal processing methods."""

    def test_univariate_correlated_features(self):
        X = self.generate_univariate_correlated_dataset()
        features_name_to_drop, filtered_X = tsfel.correlated_features(X, threshold=0.90, drop_correlated=True)
        np.testing.assert_equal(
            (features_name_to_drop, np.shape(filtered_X)),
            (["Feature_4", "Feature_5", "Feature_7", "Feature_8", "Feature_9"], (1000, 5)),
        )

    def test_no_correlated_features(self):
        X = self.generate_univariate_correlated_dataset(num_features=5, num_redundant=0)
        features_name_to_drop = tsfel.correlated_features(X, threshold=0.90, drop_correlated=False)
        self.assertEqual(features_name_to_drop, [])

    def test_empty_dataframe(self):
        X = pd.DataFrame()
        features_name_to_drop = tsfel.correlated_features(X, threshold=0.90, drop_correlated=False)
        self.assertEqual(features_name_to_drop, [])

    def test_different_thresholds(self):
        X = self.generate_univariate_correlated_dataset()
        thresholds = [0.05, 0.5, 0.95]
        expected_features_to_remove = [5, 5, 3]

        for threshold, expected_n_features in zip(thresholds, expected_features_to_remove):
            features_name_to_drop = tsfel.correlated_features(X, threshold=threshold, drop_correlated=False)
            self.assertIsInstance(features_name_to_drop, list)
            self.assertEqual(len(features_name_to_drop), expected_n_features)

    @staticmethod
    def generate_univariate_correlated_dataset(num_features: int = 10, num_redundant: int = 5) -> pd.DataFrame:
        """Generate a synthetic dataset with correlated features.

        Parameters
        ----------
            num_features: int
                Number of features in the dataset.
            num_redundant: int
                Number of redundant features that are correlated.

        Returns
        -------
            pd.DataFrame: DataFrame with the generated dataset.
        """
        # Generate synthetic data
        X, _ = make_classification(
            n_samples=1000,
            n_features=num_features,
            n_redundant=num_redundant,
            random_state=42,
        )

        # Create a DataFrame with appropriate column names
        column_names = [f"Feature_{i}" for i in range(1, num_features + 1)]
        X_df = pd.DataFrame(X, columns=column_names)

        return X_df

    def test_step_splits_full_windows(self):
        signal = np.arange(1, 6)
        windows = tsfel.signal_window_splitter(signal, window_size=3, step=2)
        self.assertEqual([window.tolist() for window in windows], [[1, 2, 3], [3, 4, 5]])

    def test_step_matches_equivalent_overlap(self):
        signal = np.arange(5.0)
        by_step = tsfel.signal_window_splitter(signal, window_size=3, step=2)
        by_overlap = tsfel.signal_window_splitter(signal, window_size=3, overlap=1 / 3)
        self.assertEqual(len(by_step), len(by_overlap))
        for left, right in zip(by_step, by_overlap):
            np.testing.assert_array_equal(left, right)

    def test_step_rejects_overlap_and_non_positive_values(self):
        signal = np.arange(5.0)
        with self.assertRaises(SystemExit):
            tsfel.signal_window_splitter(signal, window_size=3, overlap=0.5, step=2)
        with self.assertRaises(SystemExit):
            tsfel.signal_window_splitter(signal, window_size=3, step=0)

    def test_extractor_uses_step(self):
        cfg = {"statistical": {"Mean": {"function": "tsfel.calc_mean", "parameters": "", "use": "yes"}}}
        features = tsfel.time_series_features_extractor(
            cfg,
            np.arange(5.0),
            fs=1,
            window_size=3,
            step=2,
            verbose=0,
        )
        self.assertEqual(len(features), 2)
        self.assertIn("0_Mean", features.columns)


if __name__ == "__main__":
    unittest.main()
