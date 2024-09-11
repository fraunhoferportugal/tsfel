import unittest

import colorednoise as cn
import numpy as np

from tsfel.feature_extraction.features import *

# Unit testing for Linux OS
# Implementing signals for testing features

const0 = np.zeros(20)
const1 = np.ones(20)
constNeg = np.ones(20) * (-1)
constF = np.ones(20) * 2.5
lin = np.arange(20)
lin0 = np.linspace(-10, 10, 20)
f = 5
sample = 1000
x = np.arange(0, sample, 1)
Fs = 1000
wave = np.sin(2 * np.pi * f * x / Fs)
np.random.seed(seed=10)
noiseWave = wave + np.random.normal(0, 0.1, 1000)
offsetWave = wave + 2

duration = 5
samples = duration * Fs
whiteNoise = cn.powerlaw_psd_gaussian(0, samples, random_state=10)
pinkNoise = cn.powerlaw_psd_gaussian(1, samples, random_state=10)
brownNoise = cn.powerlaw_psd_gaussian(2, samples, random_state=10)


class TestFeatures(unittest.TestCase):
    # ############################################### STATISTICAL FEATURES ############################################### #
    def test_hist(self):
        np.testing.assert_almost_equal(
            hist_mode(const0, 10),
            0.050000000000000044,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            hist_mode(const1, 10),
            1.05,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            hist_mode(constNeg, 10),
            -0.95,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            hist_mode(constF, 10),
            2.55,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            hist_mode(lin, 10),
            0.95,
        )
        np.testing.assert_almost_equal(
            hist_mode(wave, 10),
            -0.9,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            hist_mode(offsetWave, 10),
            1.1,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            hist_mode(noiseWave, 10),
            -0.8862517157830269,
            decimal=5,
        )

    def test_skewness(self):
        self.assertTrue(np.isnan(skewness(const0)))
        self.assertTrue(np.isnan(skewness(const1)))
        self.assertTrue(np.isnan(skewness(constNeg)))
        self.assertTrue(np.isnan(skewness(constF)))
        np.testing.assert_almost_equal(skewness(lin), 0)
        np.testing.assert_almost_equal(
            skewness(lin0),
            -1.0167718723297815e-16,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            skewness(wave),
            -2.009718347115232e-17,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            skewness(offsetWave),
            9.043732562018544e-16,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            skewness(noiseWave),
            -0.0004854111290521465,
            decimal=5,
        )

    def test_kurtosis(self):
        self.assertTrue(np.isnan(kurtosis(const0)))
        self.assertTrue(np.isnan(kurtosis(const1)))
        self.assertTrue(np.isnan(kurtosis(constNeg)))
        self.assertTrue(np.isnan(kurtosis(constF)))
        np.testing.assert_almost_equal(kurtosis(lin), -1.206015037593985, decimal=2)
        np.testing.assert_almost_equal(kurtosis(lin0), -1.2060150375939847, decimal=2)
        np.testing.assert_almost_equal(kurtosis(wave), -1.501494077162359, decimal=2)
        np.testing.assert_almost_equal(
            kurtosis(offsetWave),
            -1.5014940771623597,
            decimal=2,
        )
        np.testing.assert_almost_equal(
            kurtosis(noiseWave),
            -1.4606204906023366,
            decimal=2,
        )

    def test_mean(self):
        np.testing.assert_almost_equal(calc_mean(const0), 0.0)
        np.testing.assert_almost_equal(calc_mean(const1), 1.0)
        np.testing.assert_almost_equal(calc_mean(constNeg), -1.0)
        np.testing.assert_almost_equal(calc_mean(constF), 2.5)
        np.testing.assert_almost_equal(calc_mean(lin), 9.5)
        np.testing.assert_almost_equal(
            calc_mean(lin0),
            -3.552713678800501e-16,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            calc_mean(wave),
            7.105427357601002e-18,
            decimal=5,
        )
        np.testing.assert_almost_equal(calc_mean(offsetWave), 2.0, decimal=5)
        np.testing.assert_almost_equal(
            calc_mean(noiseWave),
            -0.0014556635615470554,
            decimal=5,
        )

    def test_median(self):
        np.testing.assert_almost_equal(calc_median(const0), 0.0)
        np.testing.assert_almost_equal(calc_median(const1), 1.0)
        np.testing.assert_almost_equal(calc_median(constNeg), -1.0)
        np.testing.assert_almost_equal(calc_median(constF), 2.5)
        np.testing.assert_almost_equal(calc_median(lin), 9.5)
        np.testing.assert_almost_equal(
            calc_median(lin0),
            -3.552713678800501e-16,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            calc_median(wave),
            7.105427357601002e-18,
            decimal=5,
        )
        np.testing.assert_almost_equal(calc_median(offsetWave), 2.0, decimal=5)
        np.testing.assert_almost_equal(
            calc_median(noiseWave),
            0.013846093997438328,
            decimal=5,
        )

    def test_max(self):
        np.testing.assert_almost_equal(calc_max(const0), 0.0)
        np.testing.assert_almost_equal(calc_max(const1), 1.0)
        np.testing.assert_almost_equal(calc_max(constNeg), -1.0)
        np.testing.assert_almost_equal(calc_max(constF), 2.5)
        np.testing.assert_almost_equal(calc_max(lin), 19)
        np.testing.assert_almost_equal(calc_max(lin0), 10.0, decimal=5)
        np.testing.assert_almost_equal(calc_max(wave), 1.0, decimal=5)
        np.testing.assert_almost_equal(
            calc_max(noiseWave),
            1.221757617217142,
            decimal=5,
        )
        np.testing.assert_almost_equal(calc_max(offsetWave), 3.0, decimal=5)

    def test_min(self):
        np.testing.assert_almost_equal(calc_min(const0), 0.0)
        np.testing.assert_almost_equal(calc_min(const1), 1.0)
        np.testing.assert_almost_equal(calc_min(constNeg), -1.0)
        np.testing.assert_almost_equal(calc_min(constF), 2.5)
        np.testing.assert_almost_equal(calc_min(lin), 0)
        np.testing.assert_almost_equal(calc_min(lin0), -10.0, decimal=5)
        np.testing.assert_almost_equal(calc_min(wave), -1.0, decimal=5)
        np.testing.assert_almost_equal(
            calc_min(noiseWave),
            -1.2582533627830566,
            decimal=5,
        )
        np.testing.assert_almost_equal(calc_min(offsetWave), 1.0, decimal=5)

    def test_variance(self):
        np.testing.assert_almost_equal(calc_var(const0), 0.0)
        np.testing.assert_almost_equal(calc_var(const1), 0.0)
        np.testing.assert_almost_equal(calc_var(constNeg), 0.0)
        np.testing.assert_almost_equal(calc_var(constF), 0.0)
        np.testing.assert_almost_equal(calc_var(lin), 33.25)
        np.testing.assert_almost_equal(calc_var(lin0), 36.84210526315789, decimal=5)
        np.testing.assert_almost_equal(calc_var(wave), 0.5, decimal=5)
        np.testing.assert_almost_equal(calc_var(offsetWave), 0.5, decimal=5)
        np.testing.assert_almost_equal(
            calc_var(noiseWave),
            0.5081167177369529,
            decimal=5,
        )

    def test_std(self):
        np.testing.assert_almost_equal(calc_std(const0), 0.0)
        np.testing.assert_almost_equal(calc_std(const1), 0.0)
        np.testing.assert_almost_equal(calc_std(constNeg), 0.0)
        np.testing.assert_almost_equal(calc_std(constF), 0.0)
        np.testing.assert_almost_equal(calc_std(lin), 5.766281297335398)
        np.testing.assert_almost_equal(calc_std(lin0), 6.069769786668839, decimal=5)
        np.testing.assert_almost_equal(calc_std(wave), 0.7071067811865476, decimal=5)
        np.testing.assert_almost_equal(
            calc_std(offsetWave),
            0.7071067811865476,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            calc_std(noiseWave),
            0.7128230620125536,
            decimal=5,
        )

    def test_interq_range(self):
        np.testing.assert_almost_equal(interq_range(const0), 0.0)
        np.testing.assert_almost_equal(interq_range(const1), 0.0)
        np.testing.assert_almost_equal(interq_range(constNeg), 0.0)
        np.testing.assert_almost_equal(interq_range(constF), 0.0)
        np.testing.assert_almost_equal(interq_range(lin), 9.5)
        np.testing.assert_almost_equal(interq_range(lin0), 10.0, decimal=5)
        np.testing.assert_almost_equal(interq_range(wave), 1.414213562373095, decimal=5)
        np.testing.assert_almost_equal(
            interq_range(offsetWave),
            1.414213562373095,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            interq_range(noiseWave),
            1.4277110228590328,
            decimal=5,
        )

    def test_mean_abs_diff(self):
        np.testing.assert_almost_equal(mean_abs_diff(const0), 0.0)
        np.testing.assert_almost_equal(mean_abs_diff(const1), 0.0)
        np.testing.assert_almost_equal(mean_abs_diff(constNeg), 0.0)
        np.testing.assert_almost_equal(mean_abs_diff(constF), 0.0)
        np.testing.assert_almost_equal(mean_abs_diff(lin), 1.0)
        np.testing.assert_almost_equal(
            mean_abs_diff(lin0),
            1.0526315789473684,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            mean_abs_diff(wave),
            0.019988577818740614,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            mean_abs_diff(noiseWave),
            0.10700252903161511,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            mean_abs_diff(offsetWave),
            0.019988577818740614,
            decimal=5,
        )

    def test_mean_abs_deviation(self):
        np.testing.assert_almost_equal(mean_abs_deviation(const0), 0.0)
        np.testing.assert_almost_equal(mean_abs_deviation(const1), 0.0)
        np.testing.assert_almost_equal(mean_abs_deviation(constNeg), 0.0)
        np.testing.assert_almost_equal(mean_abs_deviation(constF), 0.0)
        np.testing.assert_almost_equal(mean_abs_deviation(lin), 5.0)
        np.testing.assert_almost_equal(
            mean_abs_deviation(lin0),
            5.263157894736842,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            mean_abs_deviation(wave),
            0.6365674116287157,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            mean_abs_deviation(noiseWave),
            0.6392749078483896,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            mean_abs_deviation(offsetWave),
            0.6365674116287157,
            decimal=5,
        )

    def test_calc_median_abs_deviation(self):
        np.testing.assert_almost_equal(median_abs_deviation(const0), 0.0)
        np.testing.assert_almost_equal(median_abs_deviation(const1), 0.0)
        np.testing.assert_almost_equal(median_abs_deviation(constNeg), 0.0)
        np.testing.assert_almost_equal(median_abs_deviation(constF), 0.0)
        np.testing.assert_almost_equal(median_abs_deviation(lin), 5.0)
        np.testing.assert_almost_equal(
            median_abs_deviation(lin0),
            5.2631578947368425,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            median_abs_deviation(wave),
            0.7071067811865475,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            median_abs_deviation(offsetWave),
            0.7071067811865475,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            median_abs_deviation(noiseWave),
            0.7068117164205888,
            decimal=5,
        )

    def test_rms(self):
        np.testing.assert_almost_equal(rms(const0), 0.0)
        np.testing.assert_almost_equal(rms(const1), 1.0)
        np.testing.assert_almost_equal(rms(constNeg), 1.0)
        np.testing.assert_almost_equal(rms(constF), 2.5)
        np.testing.assert_almost_equal(rms(lin), 11.113055385446435)
        np.testing.assert_almost_equal(rms(lin0), 6.06976978666884, decimal=5)
        np.testing.assert_almost_equal(rms(wave), 0.7071067811865476, decimal=5)
        np.testing.assert_almost_equal(rms(offsetWave), 2.1213203435596424, decimal=5)
        np.testing.assert_almost_equal(rms(noiseWave), 0.7128245483240299, decimal=5)

    def test_ecdf(self):
        np.testing.assert_almost_equal(
            ecdf(const0),
            (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5),
        )
        np.testing.assert_almost_equal(
            ecdf(const1),
            (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5),
        )
        np.testing.assert_almost_equal(
            ecdf(constNeg),
            (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5),
        )
        np.testing.assert_almost_equal(
            ecdf(constF),
            (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5),
        )
        np.testing.assert_almost_equal(
            ecdf(lin),
            (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5),
        )
        np.testing.assert_almost_equal(
            ecdf(lin0),
            (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5),
        )
        np.testing.assert_almost_equal(
            ecdf(wave),
            (
                0.001,
                0.002,
                0.003,
                0.004,
                0.005,
                0.006,
                0.007,
                0.008,
                0.009,
                0.01,
            ),
        )
        np.testing.assert_almost_equal(
            ecdf(offsetWave),
            (
                0.001,
                0.002,
                0.003,
                0.004,
                0.005,
                0.006,
                0.007,
                0.008,
                0.009,
                0.01,
            ),
        )
        np.testing.assert_almost_equal(
            ecdf(noiseWave),
            (
                0.001,
                0.002,
                0.003,
                0.004,
                0.005,
                0.006,
                0.007,
                0.008,
                0.009,
                0.01,
            ),
        )

    def test_ecdf_percentile(self):
        np.testing.assert_almost_equal(ecdf_percentile(const0), (0, 0))
        np.testing.assert_almost_equal(ecdf_percentile(const1), (1, 1))
        np.testing.assert_almost_equal(ecdf_percentile(constNeg), (-1, -1))
        np.testing.assert_almost_equal(ecdf_percentile(constF), (2.5, 2.5))
        np.testing.assert_almost_equal(ecdf_percentile(lin), (3, 15))
        np.testing.assert_almost_equal(
            ecdf_percentile(lin0),
            (-6.8421053, 5.7894737),
            decimal=7,
        )
        np.testing.assert_almost_equal(ecdf_percentile(wave), (-0.809017, 0.809017))
        np.testing.assert_almost_equal(
            ecdf_percentile(offsetWave),
            (1.1909830056250523, 2.809016994374947),
        )
        np.testing.assert_almost_equal(
            ecdf_percentile(noiseWave),
            (-0.8095410722491809, 0.796916231269631),
        )

    def test_ecdf_percentile_count(self):
        np.testing.assert_almost_equal(ecdf_percentile_count(const0), (0, 0))
        np.testing.assert_almost_equal(ecdf_percentile_count(const1), (1, 1))
        np.testing.assert_almost_equal(ecdf_percentile_count(constNeg), (-1, -1))
        np.testing.assert_almost_equal(ecdf_percentile_count(constF), (2.5, 2.5))
        np.testing.assert_almost_equal(ecdf_percentile_count(lin), (4, 16))
        np.testing.assert_almost_equal(ecdf_percentile_count(lin0), (4, 16))
        np.testing.assert_almost_equal(ecdf_percentile_count(wave), (200, 800))
        np.testing.assert_almost_equal(ecdf_percentile_count(offsetWave), (200, 800))
        np.testing.assert_almost_equal(ecdf_percentile_count(noiseWave), (200, 800))

    # ################################################ TEMPORAL FEATURES ################################################# #
    def test_distance(self):
        np.testing.assert_almost_equal(distance(const0), 19.0)
        np.testing.assert_almost_equal(distance(const1), 19.0)
        np.testing.assert_almost_equal(distance(constNeg), 19.0)
        np.testing.assert_almost_equal(distance(constF), 19.0)
        np.testing.assert_almost_equal(distance(lin), 26.87005768508881)
        np.testing.assert_almost_equal(distance(lin0), 27.586228448267438, decimal=5)
        np.testing.assert_almost_equal(distance(wave), 999.2461809866238, decimal=5)
        np.testing.assert_almost_equal(
            distance(offsetWave),
            999.2461809866238,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            distance(noiseWave),
            1007.8711901383033,
            decimal=5,
        )

    def test_negative_turning(self):
        np.testing.assert_almost_equal(negative_turning(const0), 0.0)
        np.testing.assert_almost_equal(negative_turning(const1), 0.0)
        np.testing.assert_almost_equal(negative_turning(constNeg), 0.0)
        np.testing.assert_almost_equal(negative_turning(constF), 0.0)
        np.testing.assert_almost_equal(negative_turning(lin), 0.0)
        np.testing.assert_almost_equal(negative_turning(lin0), 0.0, decimal=5)
        np.testing.assert_almost_equal(negative_turning(wave), 5, decimal=5)
        np.testing.assert_almost_equal(negative_turning(offsetWave), 5, decimal=5)
        np.testing.assert_almost_equal(negative_turning(noiseWave), 323, decimal=5)

    def test_positive_turning(self):
        np.testing.assert_almost_equal(positive_turning(const0), 0.0)
        np.testing.assert_almost_equal(positive_turning(const1), 0.0)
        np.testing.assert_almost_equal(positive_turning(constNeg), 0.0)
        np.testing.assert_almost_equal(positive_turning(constF), 0.0)
        np.testing.assert_almost_equal(positive_turning(lin), 0.0)
        np.testing.assert_almost_equal(positive_turning(lin0), 0.0, decimal=5)
        np.testing.assert_almost_equal(positive_turning(wave), 5, decimal=5)
        np.testing.assert_almost_equal(positive_turning(offsetWave), 5, decimal=5)
        np.testing.assert_almost_equal(positive_turning(noiseWave), 322, decimal=5)

    def test_centroid(self):
        np.testing.assert_almost_equal(calc_centroid(const0, Fs), 0.0)
        np.testing.assert_almost_equal(calc_centroid(const1, Fs), 0.009499999999999998)
        np.testing.assert_almost_equal(
            calc_centroid(constNeg, Fs),
            0.009499999999999998,
        )
        np.testing.assert_almost_equal(calc_centroid(constF, Fs), 0.0095)
        np.testing.assert_almost_equal(calc_centroid(lin, Fs), 0.014615384615384615)
        np.testing.assert_almost_equal(calc_centroid(lin0, Fs), 0.0095, decimal=5)
        np.testing.assert_almost_equal(
            calc_centroid(wave, Fs),
            0.5000000000000001,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            calc_centroid(offsetWave, Fs),
            0.47126367059427926,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            calc_centroid(noiseWave, Fs),
            0.4996034303128802,
            decimal=5,
        )

    def test_mean_diff(self):
        np.testing.assert_almost_equal(mean_diff(const0), 0.0)
        np.testing.assert_almost_equal(mean_diff(const1), 0.0)
        np.testing.assert_almost_equal(mean_diff(constNeg), 0.0)
        np.testing.assert_almost_equal(mean_diff(constF), 0.0)
        np.testing.assert_almost_equal(mean_diff(lin), 1.0)
        np.testing.assert_almost_equal(mean_diff(lin0), 1.0526315789473684, decimal=5)
        np.testing.assert_almost_equal(
            mean_diff(wave),
            -3.1442201279407477e-05,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            mean_diff(offsetWave),
            -3.1442201279407036e-05,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            mean_diff(noiseWave),
            -0.00010042477181949707,
            decimal=5,
        )

    def test_median_diff(self):
        np.testing.assert_almost_equal(median_diff(const0), 0.0)
        np.testing.assert_almost_equal(median_diff(const1), 0.0)
        np.testing.assert_almost_equal(median_diff(constNeg), 0.0)
        np.testing.assert_almost_equal(median_diff(constF), 0.0)
        np.testing.assert_almost_equal(median_diff(lin), 1.0)
        np.testing.assert_almost_equal(median_diff(lin0), 1.0526315789473684, decimal=5)
        np.testing.assert_almost_equal(
            median_diff(wave),
            -0.0004934396342684,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            median_diff(offsetWave),
            -0.0004934396342681779,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            median_diff(noiseWave),
            -0.004174819648320949,
            decimal=5,
        )

    def test_calc_mean_abs_diff(self):
        np.testing.assert_almost_equal(mean_abs_diff(const0), 0.0)
        np.testing.assert_almost_equal(mean_abs_diff(const1), 0.0)
        np.testing.assert_almost_equal(mean_abs_diff(constNeg), 0.0)
        np.testing.assert_almost_equal(mean_abs_diff(constF), 0.0)
        np.testing.assert_almost_equal(mean_abs_diff(lin), 1.0)
        np.testing.assert_almost_equal(
            mean_abs_diff(lin0),
            1.0526315789473684,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            mean_abs_diff(wave),
            0.019988577818740614,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            mean_abs_diff(offsetWave),
            0.019988577818740614,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            mean_abs_diff(noiseWave),
            0.10700252903161508,
            decimal=5,
        )

    def test_median_abs_diff(self):
        np.testing.assert_almost_equal(median_abs_diff(const0), 0.0)
        np.testing.assert_almost_equal(median_abs_diff(const1), 0.0)
        np.testing.assert_almost_equal(median_abs_diff(constNeg), 0.0)
        np.testing.assert_almost_equal(median_abs_diff(constF), 0.0)
        np.testing.assert_almost_equal(median_abs_diff(lin), 1.0)
        np.testing.assert_almost_equal(
            median_abs_diff(lin0),
            1.0526315789473681,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            median_abs_diff(wave),
            0.0218618462348652,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            median_abs_diff(offsetWave),
            0.021861846234865645,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            median_abs_diff(noiseWave),
            0.08958750592592835,
            decimal=5,
        )

    def test_sum_abs_diff(self):
        np.testing.assert_almost_equal(sum_abs_diff(const0), 0.0)
        np.testing.assert_almost_equal(sum_abs_diff(const1), 0.0)
        np.testing.assert_almost_equal(sum_abs_diff(constNeg), 0.0)
        np.testing.assert_almost_equal(sum_abs_diff(constF), 0.0)
        np.testing.assert_almost_equal(sum_abs_diff(lin), 19)
        np.testing.assert_almost_equal(sum_abs_diff(lin0), 20.0, decimal=5)
        np.testing.assert_almost_equal(
            sum_abs_diff(wave),
            19.968589240921872,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            sum_abs_diff(offsetWave),
            19.968589240921872,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            sum_abs_diff(noiseWave),
            106.89552650258346,
            decimal=5,
        )

    def test_zerocross(self):
        np.testing.assert_almost_equal(zero_cross(const0), 0.0)
        np.testing.assert_almost_equal(zero_cross(const1), 0.0)
        np.testing.assert_almost_equal(zero_cross(constNeg), 0.0)
        np.testing.assert_almost_equal(zero_cross(constF), 0.0)
        np.testing.assert_almost_equal(zero_cross(lin), 1.0)
        np.testing.assert_almost_equal(zero_cross(lin0), 1.0, decimal=5)
        np.testing.assert_almost_equal(zero_cross(wave), 10, decimal=5)
        np.testing.assert_almost_equal(zero_cross(offsetWave), 0.0, decimal=5)
        np.testing.assert_almost_equal(zero_cross(noiseWave), 38, decimal=5)

    def test_autocorr(self):
        np.testing.assert_almost_equal(autocorr(const0), 1)
        np.testing.assert_almost_equal(autocorr(const1), 1)
        np.testing.assert_almost_equal(autocorr(constNeg), 1)
        np.testing.assert_almost_equal(autocorr(constF), 1)
        np.testing.assert_almost_equal(autocorr(lin), 6)
        np.testing.assert_almost_equal(autocorr(lin0), 6)
        np.testing.assert_almost_equal(autocorr(wave), 40)
        np.testing.assert_almost_equal(autocorr(offsetWave), 40)
        np.testing.assert_almost_equal(autocorr(noiseWave), 39)

    def test_auc(self):
        np.testing.assert_almost_equal(auc(const0, Fs), 0.0)
        np.testing.assert_almost_equal(auc(const1, Fs), 0.019)
        np.testing.assert_almost_equal(auc(constNeg, Fs), 0.019)
        np.testing.assert_almost_equal(auc(constF, Fs), 0.0475)
        np.testing.assert_almost_equal(auc(lin, Fs), 0.18050000000000002)
        np.testing.assert_almost_equal(auc(lin0, Fs), 0.09473684210526315)
        np.testing.assert_almost_equal(auc(wave, Fs), 0.6365517062491768)
        np.testing.assert_almost_equal(auc(offsetWave, Fs), 1.998015705379539)
        np.testing.assert_almost_equal(auc(noiseWave, Fs), 0.6375702578824347)

    def test_abs_energy(self):
        np.testing.assert_almost_equal(abs_energy(const0), 0.0)
        np.testing.assert_almost_equal(abs_energy(const1), 20.0)
        np.testing.assert_almost_equal(abs_energy(constNeg), 20.0)
        np.testing.assert_almost_equal(abs_energy(constF), 125.0)
        np.testing.assert_almost_equal(abs_energy(lin), 2470)
        np.testing.assert_almost_equal(abs_energy(lin0), 736.8421052631579)
        np.testing.assert_almost_equal(abs_energy(wave), 500.0)
        np.testing.assert_almost_equal(abs_energy(offsetWave), 4500.0)
        np.testing.assert_almost_equal(abs_energy(noiseWave), 508.11883669335725)

    def test_pk_pk_distance(self):
        np.testing.assert_almost_equal(pk_pk_distance(const0), 0.0)
        np.testing.assert_almost_equal(pk_pk_distance(const1), 0.0)
        np.testing.assert_almost_equal(pk_pk_distance(constNeg), 0.0)
        np.testing.assert_almost_equal(pk_pk_distance(constF), 0.0)
        np.testing.assert_almost_equal(pk_pk_distance(lin), 19)
        np.testing.assert_almost_equal(pk_pk_distance(lin0), 20.0)
        np.testing.assert_almost_equal(pk_pk_distance(wave), 2.0)
        np.testing.assert_almost_equal(pk_pk_distance(offsetWave), 2.0)
        np.testing.assert_almost_equal(pk_pk_distance(noiseWave), 2.4800109800001993)

    def test_slope(self):
        np.testing.assert_almost_equal(slope(const0), 0.0)
        np.testing.assert_almost_equal(slope(const1), -8.935559365603017e-18)
        np.testing.assert_almost_equal(slope(constNeg), 8.935559365603017e-18)
        np.testing.assert_almost_equal(slope(constF), 1.7871118731206033e-17)
        np.testing.assert_almost_equal(slope(lin), 1.0)
        np.testing.assert_almost_equal(slope(lin0), 1.0526315789473686)
        np.testing.assert_almost_equal(slope(wave), -0.0003819408289180587)
        np.testing.assert_almost_equal(slope(offsetWave), -0.00038194082891805853)
        np.testing.assert_almost_equal(slope(noiseWave), -0.00040205425841671337)

    def test_entropy(self):
        np.testing.assert_almost_equal(entropy(const0), 0.0)
        np.testing.assert_almost_equal(entropy(const1), 0.0)
        np.testing.assert_almost_equal(entropy(constNeg), 0.0)
        np.testing.assert_almost_equal(entropy(constF), 0.0)
        np.testing.assert_almost_equal(entropy(lin), 1.0)
        np.testing.assert_almost_equal(entropy(lin0), 1.0)
        np.testing.assert_almost_equal(entropy(wave), 0.9620267810255854)
        np.testing.assert_almost_equal(entropy(offsetWave), 0.8891261649211666)
        np.testing.assert_almost_equal(entropy(noiseWave), 1.0)

    def test_neighbourhood_peaks(self):
        np.testing.assert_almost_equal(neighbourhood_peaks(const0), 0.0)
        np.testing.assert_almost_equal(neighbourhood_peaks(const1), 0.0)
        np.testing.assert_almost_equal(neighbourhood_peaks(constNeg), 0.0)
        np.testing.assert_almost_equal(neighbourhood_peaks(constF), 0.0)
        np.testing.assert_almost_equal(neighbourhood_peaks(lin), 0.0)
        np.testing.assert_almost_equal(neighbourhood_peaks(lin0), 0.0)
        np.testing.assert_almost_equal(neighbourhood_peaks(wave), 5.0)
        np.testing.assert_almost_equal(neighbourhood_peaks(offsetWave), 5.0)
        np.testing.assert_almost_equal(neighbourhood_peaks(noiseWave), 14.0)

    def test_lempel_ziv(self):
        np.testing.assert_almost_equal(lempel_ziv(const0), 0.25)
        np.testing.assert_almost_equal(lempel_ziv(const1), 0.25)
        np.testing.assert_almost_equal(lempel_ziv(constNeg), 0.25)
        np.testing.assert_almost_equal(lempel_ziv(constF), 0.25)
        np.testing.assert_almost_equal(lempel_ziv(lin), 0.4)
        np.testing.assert_almost_equal(lempel_ziv(lin0), 0.4)
        np.testing.assert_almost_equal(lempel_ziv(wave), 0.066)
        np.testing.assert_almost_equal(lempel_ziv(offsetWave), 0.066)
        np.testing.assert_almost_equal(lempel_ziv(noiseWave), 0.079)

    # ################################################ SPECTRAL FEATURES ################################################# #
    def test_max_fre(self):
        np.testing.assert_almost_equal(max_frequency(const0, Fs), 0.0)
        np.testing.assert_almost_equal(max_frequency(const1, Fs), 0.0)
        np.testing.assert_almost_equal(max_frequency(constNeg, Fs), 0.0)
        np.testing.assert_almost_equal(max_frequency(constF, Fs), 0.0)
        np.testing.assert_almost_equal(max_frequency(lin, Fs), 450.0)
        np.testing.assert_almost_equal(max_frequency(lin0, Fs), 450.0, decimal=1)
        np.testing.assert_almost_equal(max_frequency(wave, Fs), 5.0, decimal=1)
        np.testing.assert_almost_equal(max_frequency(offsetWave, Fs), 5.0, decimal=1)
        np.testing.assert_almost_equal(max_frequency(noiseWave, Fs), 465.0, decimal=1)
        np.testing.assert_almost_equal(max_frequency(x, Fs), 345.0, decimal=1)

    def test_med_fre(self):
        np.testing.assert_almost_equal(median_frequency(const0, Fs), 0.0)
        np.testing.assert_almost_equal(median_frequency(const1, Fs), 0.0)
        np.testing.assert_almost_equal(median_frequency(constNeg, Fs), 0.0)
        np.testing.assert_almost_equal(median_frequency(constF, Fs), 0.0)
        np.testing.assert_almost_equal(median_frequency(lin, Fs), 50.0)
        np.testing.assert_almost_equal(median_frequency(lin0, Fs), 150.0, decimal=1)
        np.testing.assert_almost_equal(median_frequency(wave, Fs), 5.0, decimal=1)
        np.testing.assert_almost_equal(median_frequency(offsetWave, Fs), 0.0, decimal=1)
        np.testing.assert_almost_equal(
            median_frequency(noiseWave, Fs),
            146.0,
            decimal=1,
        )
        np.testing.assert_almost_equal(median_frequency(x, Fs), 4.0, decimal=1)

    def test_fund_fre(self):
        np.testing.assert_almost_equal(fundamental_frequency(const0, 1), 0.0)
        np.testing.assert_almost_equal(fundamental_frequency(const1, 1), 0.0)
        np.testing.assert_almost_equal(fundamental_frequency(constNeg, Fs), 0.0)
        np.testing.assert_almost_equal(fundamental_frequency(constF, Fs), 0.0)
        np.testing.assert_almost_equal(fundamental_frequency(lin, Fs), 50.0)
        np.testing.assert_almost_equal(fundamental_frequency(lin0, Fs), 50.0, decimal=1)
        np.testing.assert_almost_equal(
            fundamental_frequency(wave, Fs),
            5.0100200400801596,
            decimal=1,
        )
        np.testing.assert_almost_equal(
            fundamental_frequency(offsetWave, Fs),
            5.0100200400801596,
            decimal=1,
        )
        np.testing.assert_almost_equal(
            fundamental_frequency(noiseWave, Fs),
            5.0100200400801596,
            decimal=1,
        )

    def test_power_spec(self):
        np.testing.assert_almost_equal(max_power_spectrum(const0, Fs), 0.0)
        np.testing.assert_almost_equal(max_power_spectrum(const1, Fs), 0.0)
        np.testing.assert_almost_equal(max_power_spectrum(constNeg, Fs), 0.0)
        np.testing.assert_almost_equal(max_power_spectrum(constF, Fs), 0.0)
        np.testing.assert_almost_equal(
            max_power_spectrum(lin, Fs),
            0.004621506382612649,
        )
        np.testing.assert_almost_equal(
            max_power_spectrum(lin0, Fs),
            0.0046215063826126525,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            max_power_spectrum(wave, Fs),
            0.6666666666666667,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            max_power_spectrum(offsetWave, Fs),
            0.6666666666666667,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            max_power_spectrum(noiseWave, Fs),
            0.6570878541643916,
            decimal=5,
        )

    def test_average_power(self):
        np.testing.assert_almost_equal(average_power(const0, Fs), 0.0)
        np.testing.assert_almost_equal(average_power(const1, Fs), 1052.6315789473686)
        np.testing.assert_almost_equal(average_power(constNeg, Fs), 1052.6315789473686)
        np.testing.assert_almost_equal(average_power(constF, Fs), 6578.9473684210525)
        np.testing.assert_almost_equal(average_power(lin, Fs), 130000.0)
        np.testing.assert_almost_equal(
            average_power(lin0, Fs),
            38781.16343490305,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            average_power(wave, Fs),
            500.5005005005005,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            average_power(offsetWave, Fs),
            4504.504504504504,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            average_power(noiseWave, Fs),
            508.6274641575148,
            decimal=5,
        )

    def test_spectral_centroid(self):
        np.testing.assert_almost_equal(spectral_centroid(const0, Fs), 0.0)
        np.testing.assert_almost_equal(
            spectral_centroid(const1, Fs),
            2.7476856540265033e-14,
        )
        np.testing.assert_almost_equal(
            spectral_centroid(constNeg, Fs),
            2.7476856540265033e-14,
        )
        np.testing.assert_almost_equal(
            spectral_centroid(constF, Fs),
            2.4504208511457478e-14,
        )
        np.testing.assert_almost_equal(
            spectral_centroid(lin, Fs),
            96.7073273343592,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            spectral_centroid(lin0, Fs),
            186.91474845748346,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            spectral_centroid(wave, Fs),
            5.000000000003773,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            spectral_centroid(offsetWave, Fs),
            1.0000000000008324,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            spectral_centroid(noiseWave, Fs),
            181.0147784750644,
            decimal=5,
        )

    def test_spectral_spread(self):
        np.testing.assert_almost_equal(spectral_spread(const0, Fs), 0.0, decimal=5)
        np.testing.assert_almost_equal(
            spectral_spread(const1, Fs),
            2.811883163207112e-06,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            spectral_spread(constNeg, Fs),
            2.811883163207112e-06,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            spectral_spread(constF, Fs),
            2.657703172211011e-06,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            spectral_spread(lin, Fs),
            138.77058121011598,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            spectral_spread(lin0, Fs),
            142.68541769470383,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            spectral_spread(wave, Fs),
            3.585399057660381e-05,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            spectral_spread(offsetWave, Fs),
            2.0000000000692015,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            spectral_spread(noiseWave, Fs),
            165.48999545678365,
            decimal=5,
        )

    def test_spectral_skewness(self):
        np.testing.assert_almost_equal(spectral_skewness(const0, Fs), 0.0, decimal=5)
        np.testing.assert_almost_equal(spectral_skewness(const1, Fs), 0.0, decimal=5)
        np.testing.assert_almost_equal(spectral_skewness(constNeg, Fs), 0.0, decimal=5)
        np.testing.assert_almost_equal(spectral_skewness(constF, Fs), 0.0, decimal=5)
        np.testing.assert_almost_equal(
            spectral_skewness(lin, Fs),
            1.4986055403796703,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            spectral_skewness(lin0, Fs),
            0.8056481576984844,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            spectral_skewness(wave, Fs),
            10757350.436568316,
            decimal=1,
        )
        np.testing.assert_almost_equal(
            spectral_skewness(offsetWave, Fs),
            1.5000000137542306,
            decimal=1,
        )
        np.testing.assert_almost_equal(
            spectral_skewness(noiseWave, Fs),
            0.4126776686583098,
            decimal=1,
        )

    def test_spectral_kurtosis(self):
        np.testing.assert_almost_equal(spectral_kurtosis(const0, Fs), 0.0, decimal=5)
        np.testing.assert_almost_equal(spectral_kurtosis(const1, Fs), 0.0, decimal=5)
        np.testing.assert_almost_equal(spectral_kurtosis(constNeg, Fs), 0.0, decimal=5)
        np.testing.assert_almost_equal(spectral_kurtosis(constF, Fs), 0.0, decimal=5)
        np.testing.assert_almost_equal(
            spectral_kurtosis(lin, Fs),
            4.209140226148914,
            decimal=0,
        )
        np.testing.assert_almost_equal(
            spectral_kurtosis(lin0, Fs),
            2.378341102603641,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            spectral_kurtosis(wave, Fs),
            123562213974218.03,
            decimal=1,
        )
        np.testing.assert_almost_equal(
            spectral_kurtosis(offsetWave, Fs),
            3.2500028252333513,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            spectral_kurtosis(noiseWave, Fs),
            1.7248024846010621,
            decimal=5,
        )

    def test_spectral_slope(self):
        np.testing.assert_almost_equal(spectral_slope(const0, Fs), 0.0)
        np.testing.assert_almost_equal(
            spectral_slope(const1, Fs),
            -0.0009090909090909091,
        )
        np.testing.assert_almost_equal(
            spectral_slope(constNeg, Fs),
            -0.0009090909090909091,
        )
        np.testing.assert_almost_equal(
            spectral_slope(constF, Fs),
            -0.0009090909090909091,
        )
        np.testing.assert_almost_equal(spectral_slope(lin, Fs), -0.0005574279006023302)
        np.testing.assert_almost_equal(
            spectral_slope(lin0, Fs),
            -0.00023672490168659717,
            decimal=1,
        )
        np.testing.assert_almost_equal(
            spectral_slope(wave, Fs),
            -2.3425149700598465e-05,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            spectral_slope(offsetWave, Fs),
            -2.380838323353288e-05,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            spectral_slope(noiseWave, Fs),
            -6.586047565550932e-06,
            decimal=5,
        )

    def test_spectral_decrease(self):
        np.testing.assert_almost_equal(spectral_decrease(const0, Fs), 0.0)
        np.testing.assert_almost_equal(spectral_decrease(const1, Fs), 0.0)
        np.testing.assert_almost_equal(spectral_decrease(constNeg, Fs), 0.0)
        np.testing.assert_almost_equal(spectral_decrease(constF, Fs), 0.0)
        np.testing.assert_almost_equal(spectral_decrease(lin, Fs), -2.2331549790804033)
        np.testing.assert_almost_equal(
            spectral_decrease(lin0, Fs),
            0.49895105698521264,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            spectral_decrease(wave, Fs),
            0.19999999999999687,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            spectral_decrease(offsetWave, Fs),
            -26.97129371996163,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            spectral_decrease(noiseWave, Fs),
            0.06049066756953234,
            decimal=5,
        )

    def test_spectral_roll_on(self):
        np.testing.assert_almost_equal(spectral_roll_on(const0, Fs), 0.0)
        np.testing.assert_almost_equal(spectral_roll_on(const1, Fs), 0.0)
        np.testing.assert_almost_equal(spectral_roll_on(constNeg, Fs), 0.0)
        np.testing.assert_almost_equal(spectral_roll_on(constF, Fs), 0.0)
        np.testing.assert_almost_equal(spectral_roll_on(lin, Fs), 0.0)
        np.testing.assert_almost_equal(spectral_roll_on(lin0, Fs), 50.0, decimal=1)
        np.testing.assert_almost_equal(spectral_roll_on(wave, Fs), 5.0, decimal=5)
        np.testing.assert_almost_equal(spectral_roll_on(offsetWave, Fs), 0.0, decimal=5)
        np.testing.assert_almost_equal(spectral_roll_on(noiseWave, Fs), 5.0, decimal=5)

    def test_spectral_roll_off(self):
        np.testing.assert_almost_equal(spectral_roll_off(const0, Fs), 0.0)
        np.testing.assert_almost_equal(spectral_roll_off(const1, Fs), 0.0)
        np.testing.assert_almost_equal(spectral_roll_off(constNeg, Fs), 0.0)
        np.testing.assert_almost_equal(spectral_roll_off(constF, Fs), 0.0)
        np.testing.assert_almost_equal(spectral_roll_off(lin, Fs), 450.0)
        np.testing.assert_almost_equal(spectral_roll_off(lin0, Fs), 450.0, decimal=5)
        np.testing.assert_almost_equal(spectral_roll_off(wave, Fs), 5.0, decimal=5)
        np.testing.assert_almost_equal(
            spectral_roll_off(offsetWave, Fs),
            5.0,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            spectral_roll_off(noiseWave, Fs),
            465.0,
            decimal=5,
        )

    def test_spectral_distance(self):
        np.testing.assert_almost_equal(spectral_distance(const0, Fs), 0.0)
        np.testing.assert_almost_equal(spectral_distance(const1, Fs), -110.0)
        np.testing.assert_almost_equal(spectral_distance(constNeg, Fs), -110.0)
        np.testing.assert_almost_equal(spectral_distance(constF, Fs), -275.0)
        np.testing.assert_almost_equal(spectral_distance(lin, Fs), -1403.842529396485)
        np.testing.assert_almost_equal(
            spectral_distance(lin0, Fs),
            -377.7289783120891,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            spectral_distance(wave, Fs),
            -122750.0,
            decimal=1,
        )
        np.testing.assert_almost_equal(
            spectral_distance(offsetWave, Fs),
            -623750.0,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            spectral_distance(noiseWave, Fs),
            -125372.23803384512,
            decimal=5,
        )

    def test_spect_variation(self):
        np.testing.assert_almost_equal(spectral_variation(const0, Fs), 1.0)
        np.testing.assert_almost_equal(spectral_variation(const1, Fs), 1.0)
        np.testing.assert_almost_equal(spectral_variation(constNeg, Fs), 1.0)
        np.testing.assert_almost_equal(spectral_variation(constF, Fs), 1.0)
        np.testing.assert_almost_equal(spectral_variation(lin, Fs), 0.04330670010243309)
        np.testing.assert_almost_equal(
            spectral_variation(lin0, Fs),
            0.3930601189429277,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            spectral_variation(wave, Fs),
            0.9999999999999997,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            spectral_variation(offsetWave, Fs),
            0.9999999999999999,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            spectral_variation(noiseWave, Fs),
            0.9775800433849368,
            decimal=5,
        )

    def test_spectral_positive_turning(self):
        np.testing.assert_almost_equal(spectral_positive_turning(const0, Fs), 0.0)
        np.testing.assert_almost_equal(spectral_positive_turning(const1, Fs), 0.0)
        np.testing.assert_almost_equal(spectral_positive_turning(constNeg, Fs), 0.0)
        np.testing.assert_almost_equal(spectral_positive_turning(constF, Fs), 0.0)
        np.testing.assert_almost_equal(spectral_positive_turning(lin, Fs), 0.0)
        np.testing.assert_almost_equal(
            spectral_positive_turning(lin0, Fs),
            1.0,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            spectral_positive_turning(wave, Fs),
            161,
            decimal=0,
        )
        np.testing.assert_almost_equal(
            spectral_positive_turning(offsetWave, Fs),
            161,
            decimal=1,
        )
        np.testing.assert_almost_equal(
            spectral_positive_turning(noiseWave, Fs),
            173.0,
            decimal=1,
        )

    def test_human_range_energy(self):
        np.testing.assert_almost_equal(human_range_energy(const0, Fs), 0.0)
        np.testing.assert_almost_equal(human_range_energy(const1, Fs), 0.0)
        np.testing.assert_almost_equal(human_range_energy(constNeg, Fs), 0.0)
        np.testing.assert_almost_equal(human_range_energy(constF, Fs), 0.0)
        np.testing.assert_almost_equal(human_range_energy(lin, Fs), 0.0)
        np.testing.assert_almost_equal(human_range_energy(lin0, Fs), 0.0)
        np.testing.assert_almost_equal(
            human_range_energy(wave, Fs),
            2.838300923247935e-33,
        )
        np.testing.assert_almost_equal(
            human_range_energy(offsetWave, Fs),
            1.6194431630448383e-33,
        )
        np.testing.assert_almost_equal(
            human_range_energy(noiseWave, Fs),
            4.5026865350839304e-05,
        )

    def test_mfcc(self):
        np.testing.assert_almost_equal(
            mfcc(const0, Fs),
            (
                -1e-08,
                -2.5654632210061364e-08,
                -4.099058125255727e-08,
                -5.56956514302075e-08,
                -6.947048992011573e-08,
                -8.203468073398136e-08,
                -9.313245317896842e-08,
                -1.0253788861142992e-07,
                -1.1005951948899701e-07,
                -1.1554422709759472e-07,
                -1.1888035860690259e-07,
                -1.2000000000000002e-07,
            ),
        )
        np.testing.assert_almost_equal(
            mfcc(const1, Fs),
            (
                0.14096637144714785,
                0.4029720554090289,
                0.2377457745400458,
                0.9307791929462678,
                -0.8138023913445843,
                -0.36127671623673,
                0.17779314470940918,
                1.5842014538963525,
                -5.868875380858009,
                -1.3484207382203723,
                -1.5899059472962034,
                2.9774371742123975,
            ),
        )
        np.testing.assert_almost_equal(
            mfcc(constNeg, Fs),
            (
                0.14096637144714785,
                0.4029720554090289,
                0.2377457745400458,
                0.9307791929462678,
                -0.8138023913445843,
                -0.36127671623673,
                0.17779314470940918,
                1.5842014538963525,
                -5.868875380858009,
                -1.3484207382203723,
                -1.5899059472962034,
                2.9774371742123975,
            ),
        )
        np.testing.assert_almost_equal(
            mfcc(constF, Fs),
            (
                0.1409663714471363,
                0.40297205540906766,
                0.23774577454002216,
                0.9307791929463864,
                -0.8138023913445535,
                -0.3612767162368284,
                0.17779314470931407,
                1.584201453896316,
                -5.868875380858139,
                -1.3484207382203004,
                -1.589905947296293,
                2.977437174212552,
            ),
        )
        np.testing.assert_almost_equal(
            mfcc(lin, Fs),
            (
                63.41077963677539,
                42.33256774689686,
                22.945623346731722,
                -9.267967765468333,
                -30.918618746635172,
                -69.45624761250505,
                -81.74881720705784,
                -112.32234611356338,
                -127.73335353282954,
                -145.3505024599537,
                -152.08439229251312,
                -170.61228411241296,
            ),
        )
        np.testing.assert_almost_equal(
            mfcc(lin0, Fs),
            (
                4.472854975902669,
                9.303621966161266,
                12.815317252229947,
                12.65260020301481,
                9.763110307405048,
                3.627814979708572,
                1.0051648150842092,
                -8.07514557618858,
                -24.79987026383853,
                -36.55749714126207,
                -49.060094200797785,
                -61.45654150658956,
            ),
        )
        np.testing.assert_almost_equal(
            mfcc(wave, Fs),
            (
                115.31298449242963,
                -23.978080415791883,
                64.49711308839377,
                -70.83883973188331,
                -17.4881594184545,
                -122.5191336465161,
                -89.73379214517978,
                -164.5583844690884,
                -153.29482394321641,
                -204.0607944643521,
                -189.9059214788022,
                -219.38937674972897,
            ),
        )
        np.testing.assert_almost_equal(
            mfcc(offsetWave, Fs),
            (
                0.02803261518615674,
                0.21714705316418328,
                0.4010268706527706,
                1.0741653432632032,
                -0.26756380975236493,
                -0.06446520044381611,
                1.2229170142535633,
                2.2173729990650166,
                -5.161787305125577,
                -1.777027230578585,
                -2.2267834681371506,
                1.266610194040295,
            ),
        )
        np.testing.assert_almost_equal(
            mfcc(noiseWave, Fs),
            (
                -59.93874366630627,
                -20.646010360067542,
                -5.9381521505819,
                13.868391975194648,
                65.73380784148053,
                67.65563377433688,
                35.223042940942214,
                73.01746718829553,
                137.50395589362876,
                111.61718917042731,
                82.69709467796633,
                110.67135918512074,
            ),
        )

    def test_power_bandwidth(self):
        np.testing.assert_almost_equal(power_bandwidth(const0, Fs), 0.0)
        np.testing.assert_almost_equal(power_bandwidth(const1, Fs), 0.0)
        np.testing.assert_almost_equal(power_bandwidth(constNeg, Fs), 0.0)
        np.testing.assert_almost_equal(power_bandwidth(constF, Fs), 0.0)
        np.testing.assert_almost_equal(power_bandwidth(lin, Fs), 0.0)
        np.testing.assert_almost_equal(power_bandwidth(lin0, Fs), 0.0)
        np.testing.assert_almost_equal(power_bandwidth(wave, Fs), 2.0)
        np.testing.assert_almost_equal(power_bandwidth(offsetWave, Fs), 2.0)
        np.testing.assert_almost_equal(power_bandwidth(noiseWave, Fs), 2.0)

    def test_fft_mean_coeff(self):
        np.testing.assert_almost_equal(
            spectrogram_mean_coeff(const0, Fs, bins=10)["values"],
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        )
        np.testing.assert_almost_equal(
            spectrogram_mean_coeff(const1, Fs, bins=10)["values"],
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        )
        np.testing.assert_almost_equal(
            spectrogram_mean_coeff(constNeg, Fs, bins=10)["values"],
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        )
        np.testing.assert_almost_equal(
            spectrogram_mean_coeff(constF, Fs, bins=10)["values"],
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        )
        np.testing.assert_almost_equal(
            spectrogram_mean_coeff(lin, Fs, bins=10)["values"],
            (
                0.00408221375370652,
                0.29732082717207287,
                0.04400486791011177,
                0.006686945426272411,
                0.00027732608206304087,
                0.0003337183893114616,
                0.0008722727267959805,
                0.0007221373313148659,
                0.00024061479410220662,
                2.1097101108186473e-07,
            ),
        )
        np.testing.assert_almost_equal(
            spectrogram_mean_coeff(lin0, Fs, bins=10)["values"],
            (
                0.004523228535962903,
                0.3294413597474491,
                0.04875885641009613,
                0.007409357813044217,
                0.00030728651752137475,
                0.0003697710684891545,
                0.0009665071765052403,
                0.0008001521676618994,
                0.00026660919014094884,
                2.337628931654879e-07,
            ),
        )
        np.testing.assert_almost_equal(
            spectrogram_mean_coeff(wave, Fs, bins=10)["values"],
            (
                2.0234880089914443e-06,
                0.0001448004568848076,
                2.1047578415647817e-05,
                3.2022732210152474e-06,
                1.52158292419209e-07,
                1.7741879185514087e-07,
                4.2795757073284126e-07,
                3.5003942541628605e-07,
                1.1626895252132188e-07,
                1.6727906953620535e-10,
            ),
        )
        np.testing.assert_almost_equal(
            spectrogram_mean_coeff(offsetWave, Fs, bins=10)["values"],
            (
                2.0234880089914642e-06,
                0.00014480045688480763,
                2.104757841564781e-05,
                3.2022732210152483e-06,
                1.5215829241920897e-07,
                1.7741879185514156e-07,
                4.27957570732841e-07,
                3.500394254162859e-07,
                1.1626895252132173e-07,
                1.6727906953620255e-10,
            ),
        )
        np.testing.assert_almost_equal(
            spectrogram_mean_coeff(noiseWave, Fs, bins=10)["values"],
            (
                3.2947755935395495e-06,
                0.00014466702099241778,
                3.838265852158549e-05,
                1.6729032217627548e-05,
                1.6879950037320804e-05,
                1.571169205601392e-05,
                1.679718723715948e-05,
                1.810371503556574e-05,
                2.0106126483830693e-05,
                8.91285109135437e-06,
            ),
        )

    def test_lpcc(self):
        np.testing.assert_almost_equal(
            lpcc(const0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        )
        np.testing.assert_almost_equal(
            lpcc(const1),
            (
                (
                    0.04123300285294014,
                    1.0090886040206166,
                    0.5477967308058846,
                    0.38934442309407374,
                    0.3182978464880446,
                    0.28521488787857235,
                    0.2753824147772057,
                    0.2852148878785723,
                    0.3182978464880446,
                    0.3893444230940738,
                    0.5477967308058846,
                    1.0090886040206166,
                )
            ),
        )
        np.testing.assert_almost_equal(
            lpcc(constNeg),
            (
                0.04123300285294014,
                1.0090886040206166,
                0.5477967308058846,
                0.38934442309407374,
                0.3182978464880446,
                0.28521488787857235,
                0.2753824147772057,
                0.2852148878785723,
                0.3182978464880446,
                0.3893444230940738,
                0.5477967308058846,
                1.0090886040206166,
            ),
        )
        np.testing.assert_almost_equal(
            lpcc(constF),
            (
                0.04123300285293459,
                1.0090886040206097,
                0.54779673080588,
                0.3893444230940775,
                0.31829784648804227,
                0.2852148878785703,
                0.2753824147772089,
                0.28521488787857036,
                0.31829784648804227,
                0.3893444230940775,
                0.54779673080588,
                1.0090886040206097,
            ),
        )
        np.testing.assert_almost_equal(
            lpcc(lin),
            (
                0.0008115287870079275,
                0.949808211511519,
                0.469481269387107,
                0.31184557567242355,
                0.24105584503772784,
                0.2081345982990198,
                0.198356078000162,
                0.2081345982990197,
                0.24105584503772784,
                0.31184557567242355,
                0.469481269387107,
                0.9498082115115191,
            ),
        )
        np.testing.assert_almost_equal(
            lpcc(lin0),
            (
                0.14900616258072136,
                0.7120654174490035,
                0.28220640800360736,
                0.13200549895670105,
                0.0674236160580709,
                0.03817662578918231,
                0.029619974765142276,
                0.03817662578918224,
                0.0674236160580709,
                0.13200549895670105,
                0.28220640800360736,
                0.7120654174490035,
            ),
        )
        np.testing.assert_almost_equal(
            lpcc(wave),
            (
                0.3316269831953852,
                2.2936534454791397,
                1.3657365894469182,
                1.0070201818573588,
                0.8468783257961441,
                0.8794963357759213,
                0.6999667686345086,
                0.8794963357759211,
                0.8468783257961441,
                1.0070201818573588,
                1.3657365894469182,
                2.2936534454791397,
            ),
        )
        np.testing.assert_almost_equal(
            lpcc(offsetWave),
            (
                0.6113077446051783,
                1.650942970269406,
                1.191078896704431,
                1.0313503136863278,
                0.9597088652698047,
                0.9261323880465244,
                0.9160979308646922,
                0.9261323880465244,
                0.9597088652698047,
                1.031350313686328,
                1.1910788967044308,
                1.650942970269406,
            ),
        )
        np.testing.assert_almost_equal(
            lpcc(noiseWave),
            (
                0.3907899246825849,
                0.6498327698888337,
                0.7444466184462464,
                0.7833967114468317,
                0.7517838305481447,
                0.7739966761714876,
                0.8210271929385791,
                0.7739966761714876,
                0.7517838305481447,
                0.7833967114468317,
                0.7444466184462464,
                0.6498327698888338,
            ),
        )

    def test_spectral_entropy(self):
        np.testing.assert_almost_equal(spectral_entropy(const0, Fs), 0.0)
        np.testing.assert_almost_equal(spectral_entropy(const1, Fs), 0.0)
        np.testing.assert_almost_equal(spectral_entropy(constNeg, Fs), 0.0)
        np.testing.assert_almost_equal(spectral_entropy(constF, Fs), 0.0)
        np.testing.assert_almost_equal(spectral_entropy(lin, Fs), 0.5983234852309258)
        np.testing.assert_almost_equal(spectral_entropy(lin0, Fs), 0.5745416630615365)
        np.testing.assert_almost_equal(
            spectral_entropy(wave, Fs),
            1.5228376718814352e-29,
        )
        np.testing.assert_almost_equal(
            spectral_entropy(offsetWave, Fs),
            1.783049297437309e-29,
        )
        np.testing.assert_almost_equal(
            spectral_entropy(noiseWave, Fs),
            0.0301141821499739,
        )

    def test_wavelet_entropy(self):
        np.testing.assert_almost_equal(wavelet_entropy(const0, Fs), 0.0)
        np.testing.assert_almost_equal(wavelet_entropy(const1, Fs), 1.944364115732569)
        np.testing.assert_almost_equal(wavelet_entropy(constNeg, Fs), 1.944364115732569)
        np.testing.assert_almost_equal(wavelet_entropy(constF, Fs), 1.944364115732569)
        np.testing.assert_almost_equal(wavelet_entropy(lin, Fs), 1.9788857784433986)
        np.testing.assert_almost_equal(wavelet_entropy(lin0, Fs), 2.034956946804592)
        np.testing.assert_almost_equal(wavelet_entropy(wave, Fs), 1.7273424867024354)
        np.testing.assert_almost_equal(wavelet_entropy(offsetWave, Fs), 1.7956330280777661)
        np.testing.assert_almost_equal(wavelet_entropy(noiseWave, Fs), 2.040089456959981)

    def test_wavelet_abs_mean(self):
        np.testing.assert_almost_equal(
            wavelet_abs_mean(const0, Fs)["values"],
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        )
        np.testing.assert_almost_equal(
            wavelet_abs_mean(const1, Fs)["values"],
            (
                0.07899792805784951,
                0.2398975173116698,
                0.4459805679832682,
                0.6895577839315219,
                0.9667582971198649,
                1.2663728921060002,
                1.5758883935978716,
                1.8718745148169993,
                2.138907471539603,
            ),
        )
        np.testing.assert_almost_equal(
            wavelet_abs_mean(constNeg, Fs)["values"],
            (
                0.07899792805784951,
                0.2398975173116698,
                0.4459805679832682,
                0.6895577839315219,
                0.9667582971198649,
                1.2663728921060002,
                1.5758883935978716,
                1.8718745148169993,
                2.138907471539603,
            ),
        )
        np.testing.assert_almost_equal(
            wavelet_abs_mean(constF, Fs)["values"],
            (
                0.1974948201446238,
                0.5997437932791744,
                1.1149514199581705,
                1.7238944598288044,
                2.4168957427996625,
                3.165932230265,
                3.93972098399468,
                4.6796862870424984,
                5.347268678849009,
            ),
        )
        np.testing.assert_almost_equal(
            wavelet_abs_mean(lin, Fs)["values"],
            (
                0.7109813525202424,
                2.159077655804685,
                4.014235268835318,
                6.206035838005904,
                8.70210674449132,
                11.409384741973945,
                14.237634116552167,
                16.98090677621436,
                19.4954814746231,
            ),
        )
        np.testing.assert_almost_equal(
            wavelet_abs_mean(lin0, Fs)["values"],
            (
                0.041577856872977034,
                0.12626185121702949,
                0.23429487053234785,
                0.36290853615111,
                0.507470608576207,
                0.6538502452979525,
                0.7719006553974893,
                0.8441064363654073,
                0.8675152684243498,
            ),
        )
        np.testing.assert_almost_equal(
            wavelet_abs_mean(wave, Fs)["values"],
            (
                4.957379822130953e-05,
                0.00015010126842380971,
                0.0002786855391736307,
                0.00042636513726987337,
                0.0005926352256626579,
                0.0007734352059476243,
                0.0009591536187513033,
                0.0011540567864786694,
                0.0013593969802381594,
            ),
        )
        np.testing.assert_almost_equal(
            wavelet_abs_mean(offsetWave, Fs)["values"],
            (
                0.0031103433240926713,
                0.00944579942404298,
                0.01756053718576274,
                0.02715605942394034,
                0.03809118969797281,
                0.05008376871179177,
                0.06317116103191377,
                0.0771689867880982,
                0.09218045718872583,
            ),
        )
        np.testing.assert_almost_equal(
            wavelet_abs_mean(noiseWave, Fs)["values"],
            (
                4.49133263252551e-05,
                8.13724192662979e-05,
                0.00033346144050483596,
                0.0005771394603385928,
                0.0007655706842596226,
                0.0009120859349630823,
                0.0010376895846962721,
                0.0011771589229261502,
                0.0013362967355863126,
            ),
        )

    def test_wavelet_std(self):
        np.testing.assert_almost_equal(
            wavelet_std(const0, Fs)["values"],
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        )
        np.testing.assert_almost_equal(
            wavelet_std(const1, Fs)["values"],
            (
                0.1643028877893254,
                0.2750570476702813,
                0.3178759664917355,
                0.2841791231505139,
                0.3353924100329286,
                0.5270543662913602,
                0.7007594702761114,
                0.7929067690446552,
                0.8274973007800435,
            ),
        )
        np.testing.assert_almost_equal(
            wavelet_std(constNeg, Fs)["values"],
            (
                0.1643028877893254,
                0.2750570476702813,
                0.3178759664917355,
                0.2841791231505139,
                0.3353924100329286,
                0.5270543662913602,
                0.7007594702761114,
                0.7929067690446552,
                0.8274973007800435,
            ),
        )
        np.testing.assert_almost_equal(
            wavelet_std(constF, Fs)["values"],
            (
                0.4107572194733135,
                0.6876426191757031,
                0.7946899162293386,
                0.7104478078762853,
                0.8384810250823213,
                1.317635915728401,
                1.751898675690277,
                1.9822669226116378,
                2.068743251950108,
            ),
        )
        np.testing.assert_almost_equal(
            wavelet_std(lin, Fs)["values"],
            (
                2.5500773964136476,
                5.241917897854233,
                8.133127236621021,
                11.234037041323939,
                14.207777056524113,
                16.41502796388351,
                17.54556504615368,
                17.64902126929419,
                17.020996811047862,
            ),
        )
        np.testing.assert_almost_equal(
            wavelet_std(lin0, Fs)["values"],
            (
                2.107175440366379,
                4.719615720034338,
                7.816022665869094,
                11.258654204336576,
                14.234950245434051,
                15.953858109016096,
                16.412086975144252,
                15.974770564870589,
                14.96461788028099,
            ),
        )
        np.testing.assert_almost_equal(
            wavelet_std(wave, Fs)["values"],
            (
                0.0019110445651182074,
                0.009682149832596305,
                0.025576463560125905,
                0.051114581095850424,
                0.08778595797863342,
                0.13620062896860544,
                0.19765669978276515,
                0.2715939983533627,
                0.3603577777529345,
            ),
        )
        np.testing.assert_almost_equal(
            wavelet_std(offsetWave, Fs)["values"],
            (
                0.05109714551769268,
                0.10242370885712056,
                0.15471219203672268,
                0.2092433508248967,
                0.26782384213056326,
                0.3319949781137163,
                0.4042338620921491,
                0.48467139632837664,
                0.5770741023374354,
            ),
        )
        np.testing.assert_almost_equal(
            wavelet_std(noiseWave, Fs)["values"],
            (
                0.08113389135272678,
                0.09399082458459491,
                0.10337522853662447,
                0.11334171048021162,
                0.1319995404268057,
                0.1663714431340742,
                0.21875327203741599,
                0.2871854951094453,
                0.37232441001499883,
            ),
        )

    def test_wavelet_var(self):
        np.testing.assert_almost_equal(
            wavelet_var(const0, Fs)["values"],
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        )
        np.testing.assert_almost_equal(
            wavelet_var(const1, Fs)["values"],
            (
                0.026995438935911654,
                0.07565637947309138,
                0.10104513007305496,
                0.08075777403459496,
                0.11248806870769608,
                0.27778630502678725,
                0.4910638351816562,
                0.6287011443968341,
                0.6847517827982577,
            ),
        )
        np.testing.assert_almost_equal(
            wavelet_var(constNeg, Fs)["values"],
            (
                0.026995438935911654,
                0.07565637947309138,
                0.10104513007305496,
                0.08075777403459496,
                0.11248806870769608,
                0.27778630502678725,
                0.4910638351816562,
                0.6287011443968341,
                0.6847517827982577,
            ),
        )
        np.testing.assert_almost_equal(
            wavelet_var(constF, Fs)["values"],
            (
                0.16872149334944786,
                0.47285237170682104,
                0.6315320629565933,
                0.5047360877162191,
                0.7030504294231003,
                1.7361644064174218,
                3.0691489698853465,
                3.9293821524802133,
                4.279698642489108,
            ),
        )
        np.testing.assert_almost_equal(
            wavelet_var(lin, Fs)["values"],
            (
                6.502894727699807,
                27.477703247844538,
                66.14775864706668,
                126.20358824583832,
                201.86092888789298,
                269.4531430550776,
                307.8468527888098,
                311.48795176399875,
                289.71433244170146,
            ),
        )
        np.testing.assert_almost_equal(
            wavelet_var(lin0, Fs)["values"],
            (
                4.440188336483244,
                22.274772544795244,
                61.09021031337943,
                126.75729449282566,
                202.63380848998295,
                254.52558856261862,
                269.3565988796996,
                255.1932946002558,
                223.9397883028255,
            ),
        )
        np.testing.assert_almost_equal(
            wavelet_var(wave, Fs)["values"],
            (
                3.652091329867838e-06,
                9.374402538084466e-05,
                0.0006541554882424483,
                0.0026127004006042697,
                0.007706374418226394,
                0.01855061133144372,
                0.03906817096901415,
                0.07376329994156637,
                0.12985772798703332,
            ),
        )
        np.testing.assert_almost_equal(
            wavelet_var(offsetWave, Fs)["values"],
            (
                0.0026109182800562606,
                0.010490616136048195,
                0.023935862364807756,
                0.0437827798644308,
                0.07172961041357687,
                0.11022066549272698,
                0.16340501526193463,
                0.23490636241889837,
                0.3330145195885568,
            ),
        )
        np.testing.assert_almost_equal(
            wavelet_var(noiseWave, Fs)["values"],
            (
                0.006582708326036073,
                0.008834275106092093,
                0.010686437874999339,
                0.012846343334580112,
                0.01742387867288791,
                0.02767945709051449,
                0.047852994027075726,
                0.08247550860125724,
                0.13862546629301695,
            ),
        )

    def test_wavelet_energy(self):
        np.testing.assert_almost_equal(
            wavelet_energy(const0, Fs)["values"],
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        )
        np.testing.assert_almost_equal(
            wavelet_energy(const1, Fs)["values"],
            (
                0.182307738654575,
                0.36497561327490674,
                0.547671249100894,
                0.745820160236465,
                1.0232837699083268,
                1.3716729227070505,
                1.7246704219236024,
                2.0328835046844693,
                2.2933985599119477,
            ),
        )
        np.testing.assert_almost_equal(
            wavelet_energy(constNeg, Fs)["values"],
            (
                0.182307738654575,
                0.36497561327490674,
                0.547671249100894,
                0.745820160236465,
                1.0232837699083268,
                1.3716729227070505,
                1.7246704219236024,
                2.0328835046844693,
                2.2933985599119477,
            ),
        )
        np.testing.assert_almost_equal(
            wavelet_energy(constF, Fs)["values"],
            (
                0.4557693466364376,
                0.9124390331872667,
                1.369178122752235,
                1.8645504005911626,
                2.558209424770817,
                3.4291823067676264,
                4.311676054809006,
                5.082208761711174,
                5.733496399779869,
            ),
        )
        np.testing.assert_almost_equal(
            wavelet_energy(lin, Fs)["values"],
            (
                2.6473362482562206,
                5.669155102097631,
                9.069831500123255,
                12.83426932351242,
                16.660960076790722,
                19.990677908601935,
                22.59551012979393,
                24.49161380363657,
                25.880265268522148,
            ),
        )
        np.testing.assert_almost_equal(
            wavelet_energy(lin0, Fs)["values"],
            (
                2.1075855984195266,
                4.721304332477201,
                7.819533515481419,
                11.264501635600086,
                14.243992941185827,
                15.967251131797697,
                16.43022913721847,
                15.997056300337551,
                14.989742194039724,
            ),
        )
        np.testing.assert_almost_equal(
            wavelet_energy(wave, Fs)["values"],
            (
                0.0019116874460376426,
                0.00968331326414813,
                0.02557798181780949,
                0.05111635929753359,
                0.08778795836979632,
                0.13620282498341038,
                0.1976590269749361,
                0.2715964502504272,
                0.3603603418068964,
            ),
        )
        np.testing.assert_almost_equal(
            wavelet_energy(offsetWave, Fs)["values"],
            (
                0.05119172311663272,
                0.10285834561574206,
                0.15570560308177836,
                0.210998178731162,
                0.27051903656893317,
                0.33575146966335573,
                0.4091400870704979,
                0.4907763390189059,
                0.5843900720204608,
            ),
        )
        np.testing.assert_almost_equal(
            wavelet_energy(noiseWave, Fs)["values"],
            (
                0.0811339037840714,
                0.09399085980861496,
                0.10337576636490604,
                0.11334317987658893,
                0.13200176048583787,
                0.16637394324613228,
                0.2187557332431539,
                0.28718790765696783,
                0.3723268080356053,
            ),
        )

    # ################################################ FRACTAL FEATURES ################################################# #

    def test_dfa(self):
        np.testing.assert_almost_equal(dfa(const0), np.nan)
        np.testing.assert_almost_equal(dfa(const1), np.nan)
        np.testing.assert_almost_equal(dfa(constNeg), np.nan)
        np.testing.assert_almost_equal(dfa(constF), np.nan)
        np.testing.assert_almost_equal(dfa(wave), 2.0354620960383225)
        np.testing.assert_almost_equal(dfa(offsetWave), 2.0354620960383234)
        np.testing.assert_almost_equal(dfa(noiseWave), 1.5878329458221712)
        np.testing.assert_almost_equal(dfa(whiteNoise), 0.512887549688051)
        np.testing.assert_almost_equal(dfa(pinkNoise), 1.0162533608512214)
        np.testing.assert_almost_equal(dfa(brownNoise), 1.5183298484325374)

    def test_hurst_exponent(self):
        np.testing.assert_almost_equal(hurst_exponent(const0), np.nan)
        np.testing.assert_almost_equal(hurst_exponent(const1), np.nan)
        np.testing.assert_almost_equal(hurst_exponent(constNeg), np.nan)
        np.testing.assert_almost_equal(hurst_exponent(constF), np.nan)
        np.testing.assert_almost_equal(hurst_exponent(wave), 0.998709262523381)
        np.testing.assert_almost_equal(hurst_exponent(offsetWave), 0.9987092625233801)
        np.testing.assert_almost_equal(hurst_exponent(noiseWave), 1.080805529048927)
        np.testing.assert_almost_equal(hurst_exponent(whiteNoise), 0.5705064906877406)
        np.testing.assert_almost_equal(hurst_exponent(pinkNoise), 0.9225990076703923)
        np.testing.assert_almost_equal(hurst_exponent(brownNoise), 0.9996474734595799)

    def test_higuchi_fractal_dimension(self):
        np.testing.assert_almost_equal(
            higuchi_fractal_dimension(wave),
            1.1116648974914232,
        )
        np.testing.assert_almost_equal(
            higuchi_fractal_dimension(offsetWave),
            1.1116648974914232,
        )
        np.testing.assert_almost_equal(
            higuchi_fractal_dimension(noiseWave),
            1.2787337809642858,
        )
        np.testing.assert_almost_equal(
            higuchi_fractal_dimension(whiteNoise),
            1.9999190166166754,
        )
        np.testing.assert_almost_equal(
            higuchi_fractal_dimension(pinkNoise),
            1.9303823647578682,
        )
        np.testing.assert_almost_equal(
            higuchi_fractal_dimension(brownNoise),
            1.581020130301515,
        )

    def test_maximum_fractal_length(self):
        np.testing.assert_almost_equal(maximum_fractal_length(wave), 1.4082260937055102)
        np.testing.assert_almost_equal(
            maximum_fractal_length(offsetWave),
            1.40822609370551,
        )
        np.testing.assert_almost_equal(
            maximum_fractal_length(noiseWave),
            1.6957110038772847,
        )
        np.testing.assert_almost_equal(
            maximum_fractal_length(whiteNoise),
            3.7493320152840734,
        )
        np.testing.assert_almost_equal(
            maximum_fractal_length(pinkNoise),
            3.542078306170788,
        )
        np.testing.assert_almost_equal(
            maximum_fractal_length(brownNoise),
            2.4274067988328794,
        )

    def test_petrosian_fractal_dimension(self):
        np.testing.assert_almost_equal(petrosian_fractal_dimension(const0), 1.0)
        np.testing.assert_almost_equal(petrosian_fractal_dimension(const1), 1.0)
        np.testing.assert_almost_equal(petrosian_fractal_dimension(constNeg), 1.0)
        np.testing.assert_almost_equal(petrosian_fractal_dimension(constF), 1.0)
        np.testing.assert_almost_equal(petrosian_fractal_dimension(lin), 1.0)
        np.testing.assert_almost_equal(petrosian_fractal_dimension(lin0), 1.0)
        np.testing.assert_almost_equal(
            petrosian_fractal_dimension(wave),
            1.000578238436128,
        )
        np.testing.assert_almost_equal(
            petrosian_fractal_dimension(offsetWave),
            1.000578238436128,
        )
        np.testing.assert_almost_equal(
            petrosian_fractal_dimension(noiseWave),
            1.0343688500384227,
        )
        np.testing.assert_almost_equal(
            petrosian_fractal_dimension(whiteNoise),
            1.0285596700513615,
        )
        np.testing.assert_almost_equal(
            petrosian_fractal_dimension(pinkNoise),
            1.0247825195115237,
        )
        np.testing.assert_almost_equal(
            petrosian_fractal_dimension(brownNoise),
            1.0193455287912367,
        )

    def test_mse(self):
        np.testing.assert_almost_equal(mse(const0), np.nan)
        np.testing.assert_almost_equal(mse(const1), np.nan)
        np.testing.assert_almost_equal(mse(constNeg), np.nan)
        np.testing.assert_almost_equal(mse(constF), np.nan)
        np.testing.assert_almost_equal(mse(wave), 0.08721585110301311)
        np.testing.assert_almost_equal(mse(offsetWave), 0.08721585110301311)
        np.testing.assert_almost_equal(mse(noiseWave), 0.11937683012271358)


if __name__ == "__main__":
    unittest.main()
