import math
import statistics
import unittest

import numpy as np

from tsfel.feature_extraction import features


class TestIntegerDifferences(unittest.TestCase):
    def check_features(self, signal):
        values = [int(value) for value in signal]
        differences = [right - left for left, right in zip(values, values[1:])]
        expected = {
            "mean_diff": sum(differences) / len(differences),
            "mean_abs_diff": sum(abs(value) for value in differences) / len(differences),
            "median_diff": statistics.median(differences),
            "median_abs_diff": statistics.median(abs(value) for value in differences),
            "sum_abs_diff": sum(abs(value) for value in differences),
            "distance": sum(math.hypot(1, value) for value in differences),
            "positive_turning": sum(
                left < middle > right for left, middle, right in zip(values, values[1:], values[2:])
            ),
            "negative_turning": sum(
                left > middle < right for left, middle, right in zip(values, values[1:], values[2:])
            ),
        }
        original = signal.copy()
        for name, wanted in expected.items():
            with self.subTest(dtype=signal.dtype, feature=name):
                actual = getattr(features, name)(signal)
                if name in {"sum_abs_diff", "positive_turning", "negative_turning"}:
                    self.assertEqual(actual, wanted)
                else:
                    tolerance = 1e-6 if signal.dtype == np.float32 else 1e-14
                    np.testing.assert_allclose(actual, wanted, rtol=tolerance, atol=1e-14)
        np.testing.assert_array_equal(signal, original)

    def test_unsigned_samples(self):
        for dtype in [np.uint8, np.uint16, np.uint32, np.uint64]:
            self.check_features(np.array([250, 253, 247, 249], dtype=dtype))

    def test_signed_extrema(self):
        for dtype in [np.int8, np.int16, np.int32, np.int64]:
            limits = np.iinfo(dtype)
            self.check_features(np.array([limits.min, limits.max, limits.min + 1], dtype=dtype))

    def test_small_differences_near_large_integers(self):
        for dtype in [np.int64, np.uint64]:
            largest = np.iinfo(dtype).max
            self.check_features(np.array([largest - 4, largest - 1, largest - 3, largest], dtype=dtype))

    def test_floating_samples(self):
        for dtype in [np.float32, np.float64]:
            self.check_features(np.array([250, 253, 247, 249], dtype=dtype))


if __name__ == "__main__":
    unittest.main()
