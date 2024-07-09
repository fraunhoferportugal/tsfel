"""A test suite for the convenient data loader methods."""

import hashlib
import unittest

import numpy as np

import tsfel


class TestDataLoaders(unittest.TestCase):
    def test_load_biopluxecg(self):
        X = tsfel.datasets.load_biopluxecg(use_cache=True)
        checksum = self.calculate_sha256_of_ndarray(X.values)

        return np.testing.assert_string_equal(
            checksum,
            "8e2a2c0f18860b23eb6ebb76b7ceff1cf1fab78f743345fab1f03d315dbc8e21",
        )

    def test_load_ucihar_all(self):
        X_train, y_train, X_test, y_test = tsfel.datasets.load_ucihar(use_cache=True)
        np.testing.assert_equal(
            (len(X_train), len(y_train), len(X_test), len(y_test), X_train[0].shape, X_test[0].shape),
            (7352, 7352, 2947, 2947, (128, 9), (128, 9)),
        )

    def test_load_ucihar_single_data_modality(self):
        X_train, y_train, X_test, y_test = tsfel.datasets.load_ucihar(use_cache=True, data_modality=["body_gyro"])
        np.testing.assert_equal(
            (len(X_train), len(y_train), len(X_test), len(y_test), X_train[0].shape, X_test[0].shape),
            (7352, 7352, 2947, 2947, (128, 3), (128, 3)),
        )

    @staticmethod
    def calculate_sha256_of_ndarray(array: np.ndarray) -> str:
        sha256_hash = hashlib.sha256()
        sha256_hash.update(array.tobytes())

        return sha256_hash.hexdigest()


if __name__ == "__main__":
    unittest.main()
