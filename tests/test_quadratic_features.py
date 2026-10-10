import unittest

import numpy as np

from tsfel.feature_extraction.features import abs_energy, average_power, rms


class TestQuadraticFeatures(unittest.TestCase):
    def test_integer_samples(self):
        for dtype in [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16]:
            limits = np.iinfo(dtype)
            signal = np.array([limits.min, limits.max, 0, 1], dtype=dtype)
            original = signal.copy()
            energy = float(sum(int(value) ** 2 for value in signal))
            for feature, actual, expected in [
                ("abs_energy", abs_energy(signal), energy),
                ("rms", rms(signal), np.sqrt(energy / len(signal))),
                ("average_power", average_power(signal, 100), energy / 0.03),
            ]:
                with self.subTest(dtype=dtype, feature=feature):
                    np.testing.assert_allclose(actual, expected, rtol=1e-14)
            np.testing.assert_array_equal(signal, original)

    def test_abs_energy_preserves_mask(self):
        for dtype in [np.int16, np.float64]:
            with self.subTest(dtype=dtype):
                signal = np.ma.array([-32768, 30000, 1], mask=[True, False, False], dtype=dtype)
                original = signal.copy()
                np.testing.assert_allclose(abs_energy(signal), 900000001.0)
                np.testing.assert_array_equal(signal.data, original.data)
                np.testing.assert_array_equal(signal.mask, original.mask)

    def test_integer_list(self):
        signal = [-32768, 32767, 0, 1]
        energy = sum(value**2 for value in signal)
        np.testing.assert_allclose(abs_energy(signal), energy)
        np.testing.assert_allclose(rms(signal), np.sqrt(energy / len(signal)))
        np.testing.assert_allclose(average_power(signal, 100), energy / 0.03)

    def test_floating_and_complex_samples(self):
        for dtype in [np.float32, np.float64, np.complex64, np.complex128]:
            with self.subTest(dtype=dtype):
                values = [3 + 4j, 1 - 2j] if np.issubdtype(dtype, np.complexfloating) else [3, -4]
                signal = np.array(values, dtype=dtype)
                original = signal.copy()
                energy = sum(abs(value) ** 2 for value in values)
                squares = sum(value**2 for value in values)
                np.testing.assert_allclose(abs_energy(signal), energy)
                np.testing.assert_allclose(rms(signal), np.sqrt(squares / len(values)))
                np.testing.assert_allclose(average_power(signal, 100), squares / 0.01)
                np.testing.assert_array_equal(signal, original)


if __name__ == "__main__":
    unittest.main()
