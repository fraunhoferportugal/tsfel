"""A test suite for the convenient data loader methods."""

import hashlib
import unittest

import numpy as np

import tsfel


class TestDataLoaders(unittest.TestCase):
    def test_load_biopluxecg(self):
        X = tsfel.datasets.load_biopluxecg(use_cache=True)
        checksum = self.calculate_sha256_of_ndarray(X)

        return np.testing.assert_string_equal(
            checksum, "8e2a2c0f18860b23eb6ebb76b7ceff1cf1fab78f743345fab1f03d315dbc8e21"
        )

    @staticmethod
    def calculate_sha256_of_ndarray(array: np.ndarray) -> str:
        sha256_hash = hashlib.sha256()
        sha256_hash.update(array.tobytes())

        return sha256_hash.hexdigest()


if __name__ == "__main__":
    unittest.main()
