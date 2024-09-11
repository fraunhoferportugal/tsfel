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


if __name__ == "__main__":
    unittest.main()
