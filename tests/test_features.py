from tsfel.feature_extraction.features import *
from numpy.testing import run_module_suite

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


# ############################################### STATISTICAL FEATURES ############################################### #
def test_hist():
    np.testing.assert_almost_equal(hist(const0, 10, 5), (0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0))
    np.testing.assert_almost_equal(hist(const1, 10, 5), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0))
    np.testing.assert_almost_equal(hist(constNeg, 10, 5), (0.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    np.testing.assert_almost_equal(hist(constF, 10, 5), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0))
    np.testing.assert_almost_equal(hist(lin, 10, 5), (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2))
    np.testing.assert_almost_equal(hist(wave, 10, 5), (0.0, 0.0, 0.0, 0.0, 499, 496, 5, 0.0, 0.0, 0.0), decimal=5)
    np.testing.assert_almost_equal(hist(offsetWave, 10, 5), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 499, 496, 5, 0.0), decimal=5)
    np.testing.assert_almost_equal(hist(noiseWave, 10, 5), (0.0, 0.0, 0.0, 48, 446, 450, 56, 0.0, 0.0, 0.0), decimal=5)


def test_skewness():
    np.testing.assert_almost_equal(skewness(const0), 0.0)
    np.testing.assert_almost_equal(skewness(const1), 0.0)
    np.testing.assert_almost_equal(skewness(constNeg), 0.0)
    np.testing.assert_almost_equal(skewness(constF), 0.0)
    np.testing.assert_almost_equal(skewness(lin), 0)
    np.testing.assert_almost_equal(skewness(lin0), -1.0167718723297815e-16, decimal=5)
    np.testing.assert_almost_equal(skewness(wave), -2.009718347115232e-17, decimal=5)
    np.testing.assert_almost_equal(skewness(offsetWave), 9.043732562018544e-16, decimal=5)
    np.testing.assert_almost_equal(skewness(noiseWave), -0.0004854111290521465, decimal=5)


def test_kurtosis():
    np.testing.assert_almost_equal(kurtosis(const0), -3)
    np.testing.assert_almost_equal(kurtosis(const1), -3)
    np.testing.assert_almost_equal(kurtosis(constNeg), -3)
    np.testing.assert_almost_equal(kurtosis(constF), -3.0)
    np.testing.assert_almost_equal(kurtosis(lin), -1.206015037593985, decimal=2)
    np.testing.assert_almost_equal(kurtosis(lin0), -1.2060150375939847, decimal=2)
    np.testing.assert_almost_equal(kurtosis(wave), -1.501494077162359, decimal=2)
    np.testing.assert_almost_equal(kurtosis(offsetWave), -1.5014940771623597, decimal=2)
    np.testing.assert_almost_equal(kurtosis(noiseWave), -1.4606204906023366, decimal=2)


def test_mean():
    np.testing.assert_almost_equal(calc_mean(const0), 0.0)
    np.testing.assert_almost_equal(calc_mean(const1), 1.0)
    np.testing.assert_almost_equal(calc_mean(constNeg), -1.0)
    np.testing.assert_almost_equal(calc_mean(constF), 2.5)
    np.testing.assert_almost_equal(calc_mean(lin), 9.5)
    np.testing.assert_almost_equal(calc_mean(lin0), -3.552713678800501e-16, decimal=5)
    np.testing.assert_almost_equal(calc_mean(wave), 7.105427357601002e-18, decimal=5)
    np.testing.assert_almost_equal(calc_mean(offsetWave), 2.0, decimal=5)
    np.testing.assert_almost_equal(calc_mean(noiseWave), -0.0014556635615470554, decimal=5)


def test_median():
    np.testing.assert_almost_equal(calc_median(const0), 0.0)
    np.testing.assert_almost_equal(calc_median(const1), 1.0)
    np.testing.assert_almost_equal(calc_median(constNeg), -1.0)
    np.testing.assert_almost_equal(calc_median(constF), 2.5)
    np.testing.assert_almost_equal(calc_median(lin), 9.5)
    np.testing.assert_almost_equal(calc_median(lin0), -3.552713678800501e-16, decimal=5)
    np.testing.assert_almost_equal(calc_median(wave), 7.105427357601002e-18, decimal=5)
    np.testing.assert_almost_equal(calc_median(offsetWave), 2.0, decimal=5)
    np.testing.assert_almost_equal(calc_median(noiseWave), 0.013846093997438328, decimal=5)


def test_max():
    np.testing.assert_almost_equal(calc_max(const0), 0.0)
    np.testing.assert_almost_equal(calc_max(const1), 1.0)
    np.testing.assert_almost_equal(calc_max(constNeg), -1.0)
    np.testing.assert_almost_equal(calc_max(constF), 2.5)
    np.testing.assert_almost_equal(calc_max(lin), 19)
    np.testing.assert_almost_equal(calc_max(lin0), 10.0, decimal=5)
    np.testing.assert_almost_equal(calc_max(wave), 1.0, decimal=5)
    np.testing.assert_almost_equal(calc_max(noiseWave), 1.221757617217142, decimal=5)
    np.testing.assert_almost_equal(calc_max(offsetWave), 3.0, decimal=5)


def test_min():
    np.testing.assert_almost_equal(calc_min(const0), 0.0)
    np.testing.assert_almost_equal(calc_min(const1), 1.0)
    np.testing.assert_almost_equal(calc_min(constNeg), -1.0)
    np.testing.assert_almost_equal(calc_min(constF), 2.5)
    np.testing.assert_almost_equal(calc_min(lin), 0)
    np.testing.assert_almost_equal(calc_min(lin0), -10.0, decimal=5)
    np.testing.assert_almost_equal(calc_min(wave), -1.0, decimal=5)
    np.testing.assert_almost_equal(calc_min(noiseWave), -1.2582533627830566, decimal=5)
    np.testing.assert_almost_equal(calc_min(offsetWave), 1.0, decimal=5)


def test_variance():
    np.testing.assert_almost_equal(calc_var(const0), 0.0)
    np.testing.assert_almost_equal(calc_var(const1), 0.0)
    np.testing.assert_almost_equal(calc_var(constNeg), 0.0)
    np.testing.assert_almost_equal(calc_var(constF), 0.0)
    np.testing.assert_almost_equal(calc_var(lin), 33.25)
    np.testing.assert_almost_equal(calc_var(lin0), 36.84210526315789, decimal=5)
    np.testing.assert_almost_equal(calc_var(wave), 0.5, decimal=5)
    np.testing.assert_almost_equal(calc_var(offsetWave), 0.5, decimal=5)
    np.testing.assert_almost_equal(calc_var(noiseWave), 0.5081167177369529, decimal=5)


def test_std():
    np.testing.assert_almost_equal(calc_std(const0), 0.0)
    np.testing.assert_almost_equal(calc_std(const1), 0.0)
    np.testing.assert_almost_equal(calc_std(constNeg), 0.0)
    np.testing.assert_almost_equal(calc_std(constF), 0.0)
    np.testing.assert_almost_equal(calc_std(lin), 5.766281297335398)
    np.testing.assert_almost_equal(calc_std(lin0), 6.069769786668839, decimal=5)
    np.testing.assert_almost_equal(calc_std(wave), 0.7071067811865476, decimal=5)
    np.testing.assert_almost_equal(calc_std(offsetWave), 0.7071067811865476, decimal=5)
    np.testing.assert_almost_equal(calc_std(noiseWave), 0.7128230620125536, decimal=5)


def test_interq_range():
    np.testing.assert_almost_equal(interq_range(const0), 0.0)
    np.testing.assert_almost_equal(interq_range(const1), 0.0)
    np.testing.assert_almost_equal(interq_range(constNeg), 0.0)
    np.testing.assert_almost_equal(interq_range(constF), 0.0)
    np.testing.assert_almost_equal(interq_range(lin), 9.5)
    np.testing.assert_almost_equal(interq_range(lin0), 10.0, decimal=5)
    np.testing.assert_almost_equal(interq_range(wave), 1.414213562373095, decimal=5)
    np.testing.assert_almost_equal(interq_range(offsetWave), 1.414213562373095, decimal=5)
    np.testing.assert_almost_equal(interq_range(noiseWave), 1.4277110228590328, decimal=5)


def test_mean_abs_diff():
    np.testing.assert_almost_equal(mean_abs_diff(const0), 0.0)
    np.testing.assert_almost_equal(mean_abs_diff(const1), 0.0)
    np.testing.assert_almost_equal(mean_abs_diff(constNeg), 0.0)
    np.testing.assert_almost_equal(mean_abs_diff(constF), 0.0)
    np.testing.assert_almost_equal(mean_abs_diff(lin), 1.0)
    np.testing.assert_almost_equal(mean_abs_diff(lin0), 1.0526315789473684, decimal=5)
    np.testing.assert_almost_equal(mean_abs_diff(wave), 0.019988577818740614, decimal=5)
    np.testing.assert_almost_equal(mean_abs_diff(noiseWave), 0.10700252903161511, decimal=5)
    np.testing.assert_almost_equal(mean_abs_diff(offsetWave), 0.019988577818740614, decimal=5)


def test_mean_abs_deviation():
    np.testing.assert_almost_equal(mean_abs_deviation(const0), 0.0)
    np.testing.assert_almost_equal(mean_abs_deviation(const1), 0.0)
    np.testing.assert_almost_equal(mean_abs_deviation(constNeg), 0.0)
    np.testing.assert_almost_equal(mean_abs_deviation(constF), 0.0)
    np.testing.assert_almost_equal(mean_abs_deviation(lin), 5.0)
    np.testing.assert_almost_equal(mean_abs_deviation(lin0), 5.263157894736842, decimal=5)
    np.testing.assert_almost_equal(mean_abs_deviation(wave), 0.6365674116287157, decimal=5)
    np.testing.assert_almost_equal(mean_abs_deviation(noiseWave), 0.6392749078483896, decimal=5)
    np.testing.assert_almost_equal(mean_abs_deviation(offsetWave), 0.6365674116287157, decimal=5)


def test_calc_median_abs_deviation():
    np.testing.assert_almost_equal(median_abs_deviation(const0), 0.0)
    np.testing.assert_almost_equal(median_abs_deviation(const1), 0.0)
    np.testing.assert_almost_equal(median_abs_deviation(constNeg), 0.0)
    np.testing.assert_almost_equal(median_abs_deviation(constF), 0.0)
    np.testing.assert_almost_equal(median_abs_deviation(lin), 5.0)
    np.testing.assert_almost_equal(median_abs_deviation(lin0), 5.2631578947368425, decimal=5)
    np.testing.assert_almost_equal(median_abs_deviation(wave), 0.7071067811865475, decimal=5)
    np.testing.assert_almost_equal(median_abs_deviation(offsetWave), 0.7071067811865475, decimal=5)
    np.testing.assert_almost_equal(median_abs_deviation(noiseWave), 0.7068117164205888, decimal=5)


def test_rms():
    np.testing.assert_almost_equal(rms(const0), 0.0)
    np.testing.assert_almost_equal(rms(const1), 1.0)
    np.testing.assert_almost_equal(rms(constNeg), 1.0)
    np.testing.assert_almost_equal(rms(constF), 2.5)
    np.testing.assert_almost_equal(rms(lin), 11.113055385446435)
    np.testing.assert_almost_equal(rms(lin0), 6.06976978666884, decimal=5)
    np.testing.assert_almost_equal(rms(wave), 0.7071067811865476, decimal=5)
    np.testing.assert_almost_equal(rms(offsetWave), 2.1213203435596424, decimal=5)
    np.testing.assert_almost_equal(rms(noiseWave), 0.7128245483240299, decimal=5)


def test_ecdf():
    np.testing.assert_almost_equal(ecdf(const0), (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5))
    np.testing.assert_almost_equal(ecdf(const1), (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5))
    np.testing.assert_almost_equal(ecdf(constNeg), (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5))
    np.testing.assert_almost_equal(ecdf(constF), (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5))
    np.testing.assert_almost_equal(ecdf(lin), (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5))
    np.testing.assert_almost_equal(ecdf(lin0), (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5))
    np.testing.assert_almost_equal(ecdf(wave), (0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
                                                0.01))
    np.testing.assert_almost_equal(ecdf(offsetWave), (0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
                                                      0.01))
    np.testing.assert_almost_equal(ecdf(noiseWave), (0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
                                                     0.01))


def test_ecdf_percentile():
    np.testing.assert_almost_equal(ecdf_percentile(const0), (0, 0))
    np.testing.assert_almost_equal(ecdf_percentile(const1), (1, 1))
    np.testing.assert_almost_equal(ecdf_percentile(constNeg), (-1, -1))
    np.testing.assert_almost_equal(ecdf_percentile(constF), (2.5, 2.5))
    np.testing.assert_almost_equal(ecdf_percentile(lin), (3, 15))
    np.testing.assert_almost_equal(ecdf_percentile(lin0), (-6.8421053, 5.7894737))
    np.testing.assert_almost_equal(ecdf_percentile(wave), (-0.809017, 0.809017))
    np.testing.assert_almost_equal(ecdf_percentile(offsetWave), (1.1909830056250523, 2.809016994374947))
    np.testing.assert_almost_equal(ecdf_percentile(noiseWave), (-0.8095410722491809, 0.796916231269631))


def test_ecdf_percentile_count():
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
def test_distance():
    np.testing.assert_almost_equal(distance(const0), 19.0)
    np.testing.assert_almost_equal(distance(const1), 19.0)
    np.testing.assert_almost_equal(distance(constNeg), 19.0)
    np.testing.assert_almost_equal(distance(constF), 19.0)
    np.testing.assert_almost_equal(distance(lin), 26.87005768508881)
    np.testing.assert_almost_equal(distance(lin0), 27.586228448267438, decimal=5)
    np.testing.assert_almost_equal(distance(wave), 999.2461809866238, decimal=5)
    np.testing.assert_almost_equal(distance(offsetWave), 999.2461809866238, decimal=5)
    np.testing.assert_almost_equal(distance(noiseWave), 1007.8711901383033, decimal=5)


def test_negative_turning():
    np.testing.assert_almost_equal(negative_turning(const0), 0.0)
    np.testing.assert_almost_equal(negative_turning(const1), 0.0)
    np.testing.assert_almost_equal(negative_turning(constNeg), 0.0)
    np.testing.assert_almost_equal(negative_turning(constF), 0.0)
    np.testing.assert_almost_equal(negative_turning(lin), 0.0)
    np.testing.assert_almost_equal(negative_turning(lin0), 0.0, decimal=5)
    np.testing.assert_almost_equal(negative_turning(wave), 5, decimal=5)
    np.testing.assert_almost_equal(negative_turning(offsetWave), 5, decimal=5)
    np.testing.assert_almost_equal(negative_turning(noiseWave), 323, decimal=5)


def test_positive_turning():
    np.testing.assert_almost_equal(positive_turning(const0), 0.0)
    np.testing.assert_almost_equal(positive_turning(const1), 0.0)
    np.testing.assert_almost_equal(positive_turning(constNeg), 0.0)
    np.testing.assert_almost_equal(positive_turning(constF), 0.0)
    np.testing.assert_almost_equal(positive_turning(lin), 0.0)
    np.testing.assert_almost_equal(positive_turning(lin0), 0.0, decimal=5)
    np.testing.assert_almost_equal(positive_turning(wave), 5, decimal=5)
    np.testing.assert_almost_equal(positive_turning(offsetWave), 5, decimal=5)
    np.testing.assert_almost_equal(positive_turning(noiseWave), 322, decimal=5)


def test_centroid():
    np.testing.assert_almost_equal(calc_centroid(const0, Fs), 0.0)
    np.testing.assert_almost_equal(calc_centroid(const1, Fs), 0.009499999999999998)
    np.testing.assert_almost_equal(calc_centroid(constNeg, Fs), 0.009499999999999998)
    np.testing.assert_almost_equal(calc_centroid(constF, Fs), 0.0095)
    np.testing.assert_almost_equal(calc_centroid(lin, Fs), 0.014615384615384615)
    np.testing.assert_almost_equal(calc_centroid(lin0, Fs), 0.0095, decimal=5)
    np.testing.assert_almost_equal(calc_centroid(wave, Fs), 0.5000000000000001, decimal=5)
    np.testing.assert_almost_equal(calc_centroid(offsetWave, Fs), 0.47126367059427926, decimal=5)
    np.testing.assert_almost_equal(calc_centroid(noiseWave, Fs), 0.4996034303128802, decimal=5)


def test_mean_diff():
    np.testing.assert_almost_equal(mean_diff(const0), 0.0)
    np.testing.assert_almost_equal(mean_diff(const1), 0.0)
    np.testing.assert_almost_equal(mean_diff(constNeg), 0.0)
    np.testing.assert_almost_equal(mean_diff(constF), 0.0)
    np.testing.assert_almost_equal(mean_diff(lin), 1.0)
    np.testing.assert_almost_equal(mean_diff(lin0), 1.0526315789473684, decimal=5)
    np.testing.assert_almost_equal(mean_diff(wave), -3.1442201279407477e-05, decimal=5)
    np.testing.assert_almost_equal(mean_diff(offsetWave), -3.1442201279407036e-05, decimal=5)
    np.testing.assert_almost_equal(mean_diff(noiseWave), -0.00010042477181949707, decimal=5)


def test_median_diff():
    np.testing.assert_almost_equal(median_diff(const0), 0.0)
    np.testing.assert_almost_equal(median_diff(const1), 0.0)
    np.testing.assert_almost_equal(median_diff(constNeg), 0.0)
    np.testing.assert_almost_equal(median_diff(constF), 0.0)
    np.testing.assert_almost_equal(median_diff(lin), 1.0)
    np.testing.assert_almost_equal(median_diff(lin0), 1.0526315789473684, decimal=5)
    np.testing.assert_almost_equal(median_diff(wave), -0.0004934396342684, decimal=5)
    np.testing.assert_almost_equal(median_diff(offsetWave), -0.0004934396342681779, decimal=5)
    np.testing.assert_almost_equal(median_diff(noiseWave), -0.004174819648320949, decimal=5)


def test_calc_mean_abs_diff():
    np.testing.assert_almost_equal(mean_abs_diff(const0), 0.0)
    np.testing.assert_almost_equal(mean_abs_diff(const1), 0.0)
    np.testing.assert_almost_equal(mean_abs_diff(constNeg), 0.0)
    np.testing.assert_almost_equal(mean_abs_diff(constF), 0.0)
    np.testing.assert_almost_equal(mean_abs_diff(lin), 1.0)
    np.testing.assert_almost_equal(mean_abs_diff(lin0), 1.0526315789473684, decimal=5)
    np.testing.assert_almost_equal(mean_abs_diff(wave), 0.019988577818740614, decimal=5)
    np.testing.assert_almost_equal(mean_abs_diff(offsetWave), 0.019988577818740614, decimal=5)
    np.testing.assert_almost_equal(mean_abs_diff(noiseWave), 0.10700252903161508, decimal=5)


def test_median_abs_diff():
    np.testing.assert_almost_equal(median_abs_diff(const0), 0.0)
    np.testing.assert_almost_equal(median_abs_diff(const1), 0.0)
    np.testing.assert_almost_equal(median_abs_diff(constNeg), 0.0)
    np.testing.assert_almost_equal(median_abs_diff(constF), 0.0)
    np.testing.assert_almost_equal(median_abs_diff(lin), 1.0)
    np.testing.assert_almost_equal(median_abs_diff(lin0), 1.0526315789473681, decimal=5)
    np.testing.assert_almost_equal(median_abs_diff(wave), 0.0218618462348652, decimal=5)
    np.testing.assert_almost_equal(median_abs_diff(offsetWave), 0.021861846234865645, decimal=5)
    np.testing.assert_almost_equal(median_abs_diff(noiseWave), 0.08958750592592835, decimal=5)


def test_sum_abs_diff():
    np.testing.assert_almost_equal(sum_abs_diff(const0), 0.0)
    np.testing.assert_almost_equal(sum_abs_diff(const1), 0.0)
    np.testing.assert_almost_equal(sum_abs_diff(constNeg), 0.0)
    np.testing.assert_almost_equal(sum_abs_diff(constF), 0.0)
    np.testing.assert_almost_equal(sum_abs_diff(lin), 19)
    np.testing.assert_almost_equal(sum_abs_diff(lin0), 20.0, decimal=5)
    np.testing.assert_almost_equal(sum_abs_diff(wave), 19.968589240921872, decimal=5)
    np.testing.assert_almost_equal(sum_abs_diff(offsetWave), 19.968589240921872, decimal=5)
    np.testing.assert_almost_equal(sum_abs_diff(noiseWave), 106.89552650258346, decimal=5)


def test_zerocross():
    np.testing.assert_almost_equal(zero_cross(const0), 0.0)
    np.testing.assert_almost_equal(zero_cross(const1), 0.0)
    np.testing.assert_almost_equal(zero_cross(constNeg), 0.0)
    np.testing.assert_almost_equal(zero_cross(constF), 0.0)
    np.testing.assert_almost_equal(zero_cross(lin), 1.0)
    np.testing.assert_almost_equal(zero_cross(lin0), 1.0, decimal=5)
    np.testing.assert_almost_equal(zero_cross(wave), 10, decimal=5)
    np.testing.assert_almost_equal(zero_cross(offsetWave), 0.0, decimal=5)
    np.testing.assert_almost_equal(zero_cross(noiseWave), 38, decimal=5)


def test_autocorr():
    np.testing.assert_almost_equal(autocorr(const0), 0.0)
    np.testing.assert_almost_equal(autocorr(const1), 20.0)
    np.testing.assert_almost_equal(autocorr(constNeg), 20.0)
    np.testing.assert_almost_equal(autocorr(constF), 125.0)
    np.testing.assert_almost_equal(autocorr(lin), 2470.0)
    np.testing.assert_almost_equal(autocorr(lin0), 736.8421052631579, decimal=0)
    np.testing.assert_almost_equal(autocorr(wave), 500.5, decimal=0)
    np.testing.assert_almost_equal(autocorr(offsetWave), 4500.0, decimal=0)
    np.testing.assert_almost_equal(autocorr(noiseWave), 508.6149018530489, decimal=0)


def test_auc():
    np.testing.assert_almost_equal(auc(const0, Fs), 0.0)
    np.testing.assert_almost_equal(auc(const1, Fs), 0.019)
    np.testing.assert_almost_equal(auc(constNeg, Fs), 0.019)
    np.testing.assert_almost_equal(auc(constF, Fs), 0.0475)
    np.testing.assert_almost_equal(auc(lin, Fs), 0.18050000000000002)
    np.testing.assert_almost_equal(auc(lin0, Fs), 0.09473684210526315)
    np.testing.assert_almost_equal(auc(wave, Fs), 0.6365517062491768)
    np.testing.assert_almost_equal(auc(offsetWave, Fs), 1.998015705379539)
    np.testing.assert_almost_equal(auc(noiseWave, Fs), 0.6375702578824347)


def test_abs_energy():
    np.testing.assert_almost_equal(abs_energy(const0), 0.0)
    np.testing.assert_almost_equal(abs_energy(const1), 20.0)
    np.testing.assert_almost_equal(abs_energy(constNeg), 20.0)
    np.testing.assert_almost_equal(abs_energy(constF), 125.0)
    np.testing.assert_almost_equal(abs_energy(lin), 2470)
    np.testing.assert_almost_equal(abs_energy(lin0), 736.8421052631579)
    np.testing.assert_almost_equal(abs_energy(wave), 500.0)
    np.testing.assert_almost_equal(abs_energy(offsetWave), 4500.0)
    np.testing.assert_almost_equal(abs_energy(noiseWave), 508.11883669335725)


def test_pk_pk_distance():
    np.testing.assert_almost_equal(pk_pk_distance(const0), 0.0)
    np.testing.assert_almost_equal(pk_pk_distance(const1), 0.0)
    np.testing.assert_almost_equal(pk_pk_distance(constNeg), 0.0)
    np.testing.assert_almost_equal(pk_pk_distance(constF), 0.0)
    np.testing.assert_almost_equal(pk_pk_distance(lin), 19)
    np.testing.assert_almost_equal(pk_pk_distance(lin0), 20.0)
    np.testing.assert_almost_equal(pk_pk_distance(wave), 2.0)
    np.testing.assert_almost_equal(pk_pk_distance(offsetWave), 2.0)
    np.testing.assert_almost_equal(pk_pk_distance(noiseWave), 2.4800109800001993)


def test_slope():
    np.testing.assert_almost_equal(slope(const0), 0.0)
    np.testing.assert_almost_equal(slope(const1), -8.935559365603017e-18)
    np.testing.assert_almost_equal(slope(constNeg), 8.935559365603017e-18)
    np.testing.assert_almost_equal(slope(constF), 1.7871118731206033e-17)
    np.testing.assert_almost_equal(slope(lin), 1.0)
    np.testing.assert_almost_equal(slope(lin0), 1.0526315789473686)
    np.testing.assert_almost_equal(slope(wave), -0.0003819408289180587)
    np.testing.assert_almost_equal(slope(offsetWave), -0.00038194082891805853)
    np.testing.assert_almost_equal(slope(noiseWave), -0.00040205425841671337)


def test_entropy():
    np.testing.assert_almost_equal(entropy(const0), 0.0)
    np.testing.assert_almost_equal(entropy(const1), 0.0)
    np.testing.assert_almost_equal(entropy(constNeg), 0.0)
    np.testing.assert_almost_equal(entropy(constF), 0.0)
    np.testing.assert_almost_equal(entropy(lin), 1.0)
    np.testing.assert_almost_equal(entropy(lin0), 1.0)
    np.testing.assert_almost_equal(entropy(wave), 0.9620267810255854)
    np.testing.assert_almost_equal(entropy(offsetWave), 0.8890012261845581)
    np.testing.assert_almost_equal(entropy(noiseWave), 1.0)


def test_neighbourhood_peaks():
    np.testing.assert_almost_equal(neighbourhood_peaks(const0), 0.0)
    np.testing.assert_almost_equal(neighbourhood_peaks(const1), 0.0)
    np.testing.assert_almost_equal(neighbourhood_peaks(constNeg), 0.0)
    np.testing.assert_almost_equal(neighbourhood_peaks(constF), 0.0)
    np.testing.assert_almost_equal(neighbourhood_peaks(lin), 0.0)
    np.testing.assert_almost_equal(neighbourhood_peaks(lin0), 0.0)
    np.testing.assert_almost_equal(neighbourhood_peaks(wave), 5.0)
    np.testing.assert_almost_equal(neighbourhood_peaks(offsetWave), 5.0)
    np.testing.assert_almost_equal(neighbourhood_peaks(noiseWave), 14.0)


# ################################################ SPECTRAL FEATURES ################################################# #
def test_max_fre():
    np.testing.assert_almost_equal(max_frequency(const0, Fs), 0.0)
    np.testing.assert_almost_equal(max_frequency(const1, Fs), 0.0)
    np.testing.assert_almost_equal(max_frequency(constNeg, Fs), 0.0)
    np.testing.assert_almost_equal(max_frequency(constF, Fs), 0.0)
    np.testing.assert_almost_equal(max_frequency(lin, Fs), 444.44444444444446)
    np.testing.assert_almost_equal(max_frequency(lin0, Fs), 500.0, decimal=5)
    np.testing.assert_almost_equal(max_frequency(wave, Fs), 5.0100200400801596, decimal=5)
    np.testing.assert_almost_equal(max_frequency(offsetWave, Fs), 5.0100200400801596, decimal=5)
    np.testing.assert_almost_equal(max_frequency(noiseWave, Fs), 464.9298597194388, decimal=5)
    np.testing.assert_almost_equal(max_frequency(x, Fs), 344.689378757515, decimal=1)


def test_med_fre():
    np.testing.assert_almost_equal(median_frequency(const0, Fs), 0.0)
    np.testing.assert_almost_equal(median_frequency(const1, Fs), 0.0)
    np.testing.assert_almost_equal(median_frequency(constNeg, Fs), 0.0)
    np.testing.assert_almost_equal(median_frequency(constF, Fs), 0.0)
    np.testing.assert_almost_equal(median_frequency(lin, Fs), 55.55555555555556)
    np.testing.assert_almost_equal(median_frequency(lin0, Fs), 166.66666666666669, decimal=5)
    np.testing.assert_almost_equal(median_frequency(wave, Fs), 5.0100200400801596, decimal=5)
    np.testing.assert_almost_equal(median_frequency(offsetWave, Fs), 0.0, decimal=5)
    np.testing.assert_almost_equal(median_frequency(noiseWave, Fs), 146.29258517034066, decimal=5)
    np.testing.assert_almost_equal(median_frequency(x, Fs), 4.008016032064128, decimal=1)


def test_fund_fre():
    np.testing.assert_almost_equal(fundamental_frequency(const0, 1), 0.0)
    np.testing.assert_almost_equal(fundamental_frequency(const1, 1), 0.0)
    np.testing.assert_almost_equal(fundamental_frequency(constNeg, Fs), 0.0)
    np.testing.assert_almost_equal(fundamental_frequency(constF, Fs), 0.0)
    np.testing.assert_almost_equal(fundamental_frequency(lin, Fs), 55.55555555555556)
    np.testing.assert_almost_equal(fundamental_frequency(lin0, Fs), 55.55555555555556, decimal=5)
    np.testing.assert_almost_equal(fundamental_frequency(wave, Fs), 5.0100200400801596, decimal=1)
    np.testing.assert_almost_equal(fundamental_frequency(offsetWave, Fs), 5.0100200400801596, decimal=1)
    np.testing.assert_almost_equal(fundamental_frequency(noiseWave, Fs), 5.0100200400801596, decimal=1)


def test_power_spec():
    np.testing.assert_almost_equal(max_power_spectrum(const0, Fs), 0.0)
    np.testing.assert_almost_equal(max_power_spectrum(const1, Fs), 0.0)
    np.testing.assert_almost_equal(max_power_spectrum(constNeg, Fs), 0.0)
    np.testing.assert_almost_equal(max_power_spectrum(constF, Fs), 0.0)
    np.testing.assert_almost_equal(max_power_spectrum(lin, Fs), 0.004621506382612649)
    np.testing.assert_almost_equal(max_power_spectrum(lin0, Fs), 0.0046215063826126525, decimal=5)
    np.testing.assert_almost_equal(max_power_spectrum(wave, Fs), 0.6666666666666667, decimal=5)
    np.testing.assert_almost_equal(max_power_spectrum(offsetWave, Fs), 0.6666666666666667, decimal=5)
    np.testing.assert_almost_equal(max_power_spectrum(noiseWave, Fs), 0.6570878541643916, decimal=5)


def test_total_energy():
    np.testing.assert_almost_equal(total_energy(const0, Fs), 0.0)
    np.testing.assert_almost_equal(total_energy(const1, Fs), 1052.6315789473686)
    np.testing.assert_almost_equal(total_energy(constNeg, Fs), 1052.6315789473686)
    np.testing.assert_almost_equal(total_energy(constF, Fs), 6578.9473684210525)
    np.testing.assert_almost_equal(total_energy(lin, Fs), 130000.0)
    np.testing.assert_almost_equal(total_energy(lin0, Fs), 38781.16343490305, decimal=5)
    np.testing.assert_almost_equal(total_energy(wave, Fs), 500.5005005005005, decimal=5)
    np.testing.assert_almost_equal(total_energy(offsetWave, Fs), 4504.504504504504, decimal=5)
    np.testing.assert_almost_equal(total_energy(noiseWave, Fs), 508.6274641575148, decimal=5)


def test_spectral_centroid():
    np.testing.assert_almost_equal(spectral_centroid(const0, Fs), 0.0)
    np.testing.assert_almost_equal(spectral_centroid(const1, Fs), 2.7476856540265033e-14)
    np.testing.assert_almost_equal(spectral_centroid(constNeg, Fs), 2.7476856540265033e-14)
    np.testing.assert_almost_equal(spectral_centroid(constF, Fs), 2.4504208511457478e-14)
    np.testing.assert_almost_equal(spectral_centroid(lin, Fs), 95.77382394996009)
    np.testing.assert_almost_equal(spectral_centroid(lin0, Fs), 189.7228259594313, decimal=5)
    np.testing.assert_almost_equal(spectral_centroid(wave, Fs), 5.010020040084022, decimal=5)
    np.testing.assert_almost_equal(spectral_centroid(offsetWave, Fs), 1.0020040080169172, decimal=5)
    np.testing.assert_almost_equal(spectral_centroid(noiseWave, Fs), 181.12036927310848, decimal=5)


def test_spectral_spread():
    np.testing.assert_almost_equal(spectral_spread(const0, Fs), 0.0, decimal=5)
    np.testing.assert_almost_equal(spectral_spread(const1, Fs), 2.811883163207112e-06, decimal=5)
    np.testing.assert_almost_equal(spectral_spread(constNeg, Fs), 2.811883163207112e-06, decimal=5)
    np.testing.assert_almost_equal(spectral_spread(constF, Fs), 2.657703172211011e-06, decimal=5)
    np.testing.assert_almost_equal(spectral_spread(lin, Fs), 137.9288076645223, decimal=5)
    np.testing.assert_almost_equal(spectral_spread(lin0, Fs), 140.93247375966078, decimal=5)
    np.testing.assert_almost_equal(spectral_spread(wave, Fs), 3.585399057660381e-05, decimal=5)
    np.testing.assert_almost_equal(spectral_spread(offsetWave, Fs), 2.004008016105514, decimal=5)
    np.testing.assert_almost_equal(spectral_spread(noiseWave, Fs), 165.6402040682083, decimal=5)


def test_spectral_skewness():
    np.testing.assert_almost_equal(spectral_skewness(const0, Fs), 0.0, decimal=5)
    np.testing.assert_almost_equal(spectral_skewness(const1, Fs), 0.0, decimal=5)
    np.testing.assert_almost_equal(spectral_skewness(constNeg, Fs), 0.0, decimal=5)
    np.testing.assert_almost_equal(spectral_skewness(constF, Fs), 0.0, decimal=5)
    np.testing.assert_almost_equal(spectral_skewness(lin, Fs), 1.5090650071326563, decimal=5)
    np.testing.assert_almost_equal(spectral_skewness(lin0, Fs), 0.8140329168647044, decimal=5)
    np.testing.assert_almost_equal(spectral_skewness(wave, Fs), 10643315.707158063, decimal=1)
    np.testing.assert_almost_equal(spectral_skewness(offsetWave, Fs), 1.5000000137542306, decimal=1)
    np.testing.assert_almost_equal(spectral_skewness(noiseWave, Fs), 0.4126776686583098, decimal=1)


def test_spectral_kurtosis():
    np.testing.assert_almost_equal(spectral_kurtosis(const0, Fs), 0.0, decimal=5)
    np.testing.assert_almost_equal(spectral_kurtosis(const1, Fs), 0.0, decimal=5)
    np.testing.assert_almost_equal(spectral_kurtosis(constNeg, Fs), 0.0, decimal=5)
    np.testing.assert_almost_equal(spectral_kurtosis(constF, Fs), 0.0, decimal=5)
    np.testing.assert_almost_equal(spectral_kurtosis(lin, Fs), 4.209140226148914, decimal=0)
    np.testing.assert_almost_equal(spectral_kurtosis(lin0, Fs), 2.4060168768515413, decimal=5)
    np.testing.assert_almost_equal(spectral_kurtosis(wave, Fs), 120959227206031.11, decimal=1)
    np.testing.assert_almost_equal(spectral_kurtosis(offsetWave, Fs), 3.2500028252333513, decimal=5)
    np.testing.assert_almost_equal(spectral_kurtosis(noiseWave, Fs), 1.7251592171239667, decimal=5)


def test_spectral_slope():
    np.testing.assert_almost_equal(spectral_slope(const0, Fs), 0.0)
    np.testing.assert_almost_equal(spectral_slope(const1, Fs), -0.0009818181818181818)
    np.testing.assert_almost_equal(spectral_slope(constNeg, Fs), -0.0009818181818181818)
    np.testing.assert_almost_equal(spectral_slope(constF, Fs), -0.0009818181818181816)
    np.testing.assert_almost_equal(spectral_slope(lin, Fs), -0.0006056882550328839)
    np.testing.assert_almost_equal(spectral_slope(lin0, Fs), -0.00023672490168659717, decimal=1)
    np.testing.assert_almost_equal(spectral_slope(wave, Fs), -2.3425149700598465e-05, decimal=5)
    np.testing.assert_almost_equal(spectral_slope(offsetWave, Fs), -2.380838323353288e-05, decimal=5)
    np.testing.assert_almost_equal(spectral_slope(noiseWave, Fs), -6.586047565550932e-06, decimal=5)


def test_spectral_decrease():
    np.testing.assert_almost_equal(spectral_decrease(const0, Fs), 0.0)
    np.testing.assert_almost_equal(spectral_decrease(const1, Fs), 0.0)
    np.testing.assert_almost_equal(spectral_decrease(constNeg, Fs), 0.0)
    np.testing.assert_almost_equal(spectral_decrease(constF, Fs), 0.0)
    np.testing.assert_almost_equal(spectral_decrease(lin, Fs), -2.255518236004341)
    np.testing.assert_almost_equal(spectral_decrease(lin0, Fs), 0.5195484076294969, decimal=5)
    np.testing.assert_almost_equal(spectral_decrease(wave, Fs), 0.19999999999999687, decimal=5)
    np.testing.assert_almost_equal(spectral_decrease(offsetWave, Fs), -26.963293719961584, decimal=5)
    np.testing.assert_almost_equal(spectral_decrease(noiseWave, Fs), 0.06053938231990085, decimal=5)


def test_spectral_roll_on():
    np.testing.assert_almost_equal(spectral_roll_on(const0, Fs), 0.0)
    np.testing.assert_almost_equal(spectral_roll_on(const1, Fs), 0.0)
    np.testing.assert_almost_equal(spectral_roll_on(constNeg, Fs), 0.0)
    np.testing.assert_almost_equal(spectral_roll_on(constF, Fs), 0.0)
    np.testing.assert_almost_equal(spectral_roll_on(lin, Fs), 0.0)
    np.testing.assert_almost_equal(spectral_roll_on(lin0, Fs), 55.55555555555556, decimal=5)
    np.testing.assert_almost_equal(spectral_roll_on(wave, Fs), 5.0100200400801596, decimal=5)
    np.testing.assert_almost_equal(spectral_roll_on(offsetWave, Fs), 0.0, decimal=5)
    np.testing.assert_almost_equal(spectral_roll_on(noiseWave, Fs), 5.0100200400801596, decimal=5)


def test_spectral_roll_off():
    np.testing.assert_almost_equal(spectral_roll_off(const0, Fs), 0.0)
    np.testing.assert_almost_equal(spectral_roll_off(const1, Fs), 0.0)
    np.testing.assert_almost_equal(spectral_roll_off(constNeg, Fs), 0.0)
    np.testing.assert_almost_equal(spectral_roll_off(constF, Fs), 0.0)
    np.testing.assert_almost_equal(spectral_roll_off(lin, Fs), 444.44444444444446)
    np.testing.assert_almost_equal(spectral_roll_off(lin0, Fs), 500.0, decimal=5)
    np.testing.assert_almost_equal(spectral_roll_off(wave, Fs), 5.0100200400801596, decimal=5)
    np.testing.assert_almost_equal(spectral_roll_off(offsetWave, Fs), 5.0100200400801596, decimal=5)
    np.testing.assert_almost_equal(spectral_roll_off(noiseWave, Fs), 464.9298597194388, decimal=5)


def test_spectral_distance():
    np.testing.assert_almost_equal(spectral_distance(const0, Fs), 0.0)
    np.testing.assert_almost_equal(spectral_distance(const1, Fs), -100)
    np.testing.assert_almost_equal(spectral_distance(constNeg, Fs), -100)
    np.testing.assert_almost_equal(spectral_distance(constF, Fs), -250)
    np.testing.assert_almost_equal(spectral_distance(lin, Fs), -1256.997293357373)
    np.testing.assert_almost_equal(spectral_distance(lin0, Fs), -323.15504563934024, decimal=5)
    np.testing.assert_almost_equal(spectral_distance(wave, Fs), -122500.00000000022, decimal=5)
    np.testing.assert_almost_equal(spectral_distance(offsetWave, Fs), -622500.0, decimal=5)
    np.testing.assert_almost_equal(spectral_distance(noiseWave, Fs), -124832.72310672606, decimal=5)


def test_spect_variation():
    np.testing.assert_almost_equal(spectral_variation(const0, Fs), 1.0)
    np.testing.assert_almost_equal(spectral_variation(const1, Fs), 1.0)
    np.testing.assert_almost_equal(spectral_variation(constNeg, Fs), 1.0)
    np.testing.assert_almost_equal(spectral_variation(constF, Fs), 1.0)
    np.testing.assert_almost_equal(spectral_variation(lin, Fs), 0.04096548417849766)
    np.testing.assert_almost_equal(spectral_variation(lin0, Fs), 0.39913530062615254, decimal=5)
    np.testing.assert_almost_equal(spectral_variation(wave, Fs), 0.9999999999999997, decimal=5)
    np.testing.assert_almost_equal(spectral_variation(offsetWave, Fs), 0.9999999999999999, decimal=5)
    np.testing.assert_almost_equal(spectral_variation(noiseWave, Fs), 0.9775968083533805, decimal=5)


def test_spectral_positive_turning():
    np.testing.assert_almost_equal(spectral_positive_turning(const0, Fs), 0.0)
    np.testing.assert_almost_equal(spectral_positive_turning(const1, Fs), 0.0)
    np.testing.assert_almost_equal(spectral_positive_turning(constNeg, Fs), 0.0)
    np.testing.assert_almost_equal(spectral_positive_turning(constF, Fs), 0.0)
    np.testing.assert_almost_equal(spectral_positive_turning(lin, Fs), 0.0)
    np.testing.assert_almost_equal(spectral_positive_turning(lin0, Fs), 1.0, decimal=5)
    np.testing.assert_almost_equal(spectral_positive_turning(wave, Fs), 155, decimal=0)
    np.testing.assert_almost_equal(spectral_positive_turning(offsetWave, Fs), 158, decimal=1)
    np.testing.assert_almost_equal(spectral_positive_turning(noiseWave, Fs), 172.0, decimal=1)


def test_human_range_energy():
    np.testing.assert_almost_equal(human_range_energy(const0, Fs), 0.0)
    np.testing.assert_almost_equal(human_range_energy(const1, Fs), 0.0)
    np.testing.assert_almost_equal(human_range_energy(constNeg, Fs), 0.0)
    np.testing.assert_almost_equal(human_range_energy(constF, Fs), 0.0)
    np.testing.assert_almost_equal(human_range_energy(lin, Fs), 0.0)
    np.testing.assert_almost_equal(human_range_energy(lin0, Fs), 0.0)
    np.testing.assert_almost_equal(human_range_energy(wave, Fs), 2.838300923247935e-33)
    np.testing.assert_almost_equal(human_range_energy(offsetWave, Fs), 1.6194431630448383e-33)
    np.testing.assert_almost_equal(human_range_energy(noiseWave, Fs), 4.5026865350839304e-05)


def test_mfcc():
    np.testing.assert_almost_equal(mfcc(const0, Fs), (-1e-08, -2.5654632210061364e-08, -4.099058125255727e-08,
                                                      -5.56956514302075e-08, -6.947048992011573e-08,
                                                      -8.203468073398136e-08,
                                                      -9.313245317896842e-08, -1.0253788861142992e-07,
                                                      -1.1005951948899701e-07,
                                                      -1.1554422709759472e-07, -1.1888035860690259e-07,
                                                      -1.2000000000000002e-07))
    np.testing.assert_almost_equal(mfcc(const1, Fs), (0.14096637144714785, 0.4029720554090289, 0.2377457745400458,
                                                      0.9307791929462678, -0.8138023913445843, -0.36127671623673,
                                                      0.17779314470940918, 1.5842014538963525, -5.868875380858009,
                                                      -1.3484207382203723, -1.5899059472962034, 2.9774371742123975))
    np.testing.assert_almost_equal(mfcc(constNeg, Fs), (0.14096637144714785, 0.4029720554090289, 0.2377457745400458,
                                                        0.9307791929462678, -0.8138023913445843, -0.36127671623673,
                                                        0.17779314470940918, 1.5842014538963525, -5.868875380858009,
                                                        -1.3484207382203723, -1.5899059472962034, 2.9774371742123975))
    np.testing.assert_almost_equal(mfcc(constF, Fs), (0.1409663714471363, 0.40297205540906766, 0.23774577454002216,
                                                      0.9307791929463864, -0.8138023913445535, -0.3612767162368284,
                                                      0.17779314470931407, 1.584201453896316, -5.868875380858139,
                                                      -1.3484207382203004, -1.589905947296293, 2.977437174212552))
    np.testing.assert_almost_equal(mfcc(lin, Fs), (63.41077963677539, 42.33256774689686, 22.945623346731722,
                                                   -9.267967765468333, -30.918618746635172, -69.45624761250505,
                                                   -81.74881720705784, -112.32234611356338, -127.73335353282954,
                                                   -145.3505024599537, -152.08439229251312, -170.61228411241296))
    np.testing.assert_almost_equal(mfcc(lin0, Fs), (4.472854975902669, 9.303621966161266, 12.815317252229947,
                                                    12.65260020301481, 9.763110307405048, 3.627814979708572,
                                                    1.0051648150842092, -8.07514557618858, -24.79987026383853,
                                                    -36.55749714126207, -49.060094200797785, -61.45654150658956))
    np.testing.assert_almost_equal(mfcc(wave, Fs), (115.31298449242963, -23.978080415791883, 64.49711308839377,
                                                    -70.83883973188331, -17.4881594184545, -122.5191336465161,
                                                    -89.73379214517978, -164.5583844690884, -153.29482394321641,
                                                    -204.0607944643521, -189.9059214788022, -219.38937674972897))
    np.testing.assert_almost_equal(mfcc(offsetWave, Fs), (0.02803261518615674, 0.21714705316418328, 0.4010268706527706,
                                                          1.0741653432632032, -0.26756380975236493,
                                                          -0.06446520044381611, 1.2229170142535633, 2.2173729990650166,
                                                          -5.161787305125577, -1.777027230578585, -2.2267834681371506,
                                                          1.266610194040295))
    np.testing.assert_almost_equal(mfcc(noiseWave, Fs), (-59.93874366630627, -20.646010360067542, -5.9381521505819,
                                                         13.868391975194648, 65.73380784148053, 67.65563377433688,
                                                         35.223042940942214, 73.01746718829553, 137.50395589362876,
                                                         111.61718917042731, 82.69709467796633, 110.67135918512074))


def test_power_bandwidth():
    np.testing.assert_almost_equal(power_bandwidth(const0, Fs), 0.0)
    np.testing.assert_almost_equal(power_bandwidth(const1, Fs), 0.0)
    np.testing.assert_almost_equal(power_bandwidth(constNeg, Fs), 0.0)
    np.testing.assert_almost_equal(power_bandwidth(constF, Fs), 0.0)
    np.testing.assert_almost_equal(power_bandwidth(lin, Fs), 0.0)
    np.testing.assert_almost_equal(power_bandwidth(lin0, Fs), 0.0)
    np.testing.assert_almost_equal(power_bandwidth(wave, Fs), 2.0)
    np.testing.assert_almost_equal(power_bandwidth(offsetWave, Fs), 2.0)
    np.testing.assert_almost_equal(power_bandwidth(noiseWave, Fs), 2.0)


def test_fft_mean_coeff():
    np.testing.assert_almost_equal(fft_mean_coeff(const0, Fs, nfreq=10),
                                   (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    np.testing.assert_almost_equal(fft_mean_coeff(const1, Fs, nfreq=10),
                                   (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    np.testing.assert_almost_equal(fft_mean_coeff(constNeg, Fs, nfreq=10),
                                   (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    np.testing.assert_almost_equal(fft_mean_coeff(constF, Fs, nfreq=10),
                                   (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    np.testing.assert_almost_equal(fft_mean_coeff(lin, Fs, nfreq=10), (0.00408221375370652, 0.29732082717207287,
                                                                       0.04400486791011177, 0.006686945426272411,
                                                                       0.00027732608206304087, 0.0003337183893114616,
                                                                       0.0008722727267959805, 0.0007221373313148659,
                                                                       0.00024061479410220662, 2.1097101108186473e-07))
    np.testing.assert_almost_equal(fft_mean_coeff(lin0, Fs, nfreq=10), (0.004523228535962903, 0.3294413597474491,
                                                                        0.04875885641009613, 0.007409357813044217,
                                                                        0.00030728651752137475, 0.0003697710684891545,
                                                                        0.0009665071765052403, 0.0008001521676618994,
                                                                        0.00026660919014094884, 2.337628931654879e-07))
    np.testing.assert_almost_equal(fft_mean_coeff(wave, Fs, nfreq=10), (2.0234880089914443e-06, 0.0001448004568848076,
                                                                        2.1047578415647817e-05, 3.2022732210152474e-06,
                                                                        1.52158292419209e-07, 1.7741879185514087e-07,
                                                                        4.2795757073284126e-07, 3.5003942541628605e-07,
                                                                        1.1626895252132188e-07, 1.6727906953620535e-10))
    np.testing.assert_almost_equal(fft_mean_coeff(offsetWave, Fs, nfreq=10), (2.0234880089914642e-06,
                                                                              0.00014480045688480763,
                                                                              2.104757841564781e-05,
                                                                              3.2022732210152483e-06,
                                                                              1.5215829241920897e-07,
                                                                              1.7741879185514156e-07,
                                                                              4.27957570732841e-07,
                                                                              3.500394254162859e-07,
                                                                              1.1626895252132173e-07,
                                                                              1.6727906953620255e-10))
    np.testing.assert_almost_equal(fft_mean_coeff(noiseWave, Fs, nfreq=10), (3.2947755935395495e-06,
                                                                             0.00014466702099241778,
                                                                             3.838265852158549e-05,
                                                                             1.6729032217627548e-05,
                                                                             1.6879950037320804e-05,
                                                                             1.571169205601392e-05,
                                                                             1.679718723715948e-05,
                                                                             1.810371503556574e-05,
                                                                             2.0106126483830693e-05,
                                                                             8.91285109135437e-06))


def test_lpcc():
    np.testing.assert_almost_equal(lpcc(const0), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    np.testing.assert_almost_equal(lpcc(const1), ((0.020164333842602966, 0.9865688990210231, 0.5256081668917854,
                                                   0.36558947821279086, 0.2920451699576349, 0.25507545331173936,
                                                   0.23917931511018226, 0.23917931511018226, 0.25507545331173936,
                                                   0.2920451699576349, 0.36558947821279086, 0.5256081668917854,
                                                   0.9865688990210231)))
    np.testing.assert_almost_equal(lpcc(constNeg), (0.020164333842602966, 0.9865688990210231, 0.5256081668917854,
                                                    0.36558947821279086, 0.2920451699576349, 0.25507545331173936,
                                                    0.23917931511018226, 0.23917931511018226, 0.25507545331173936,
                                                    0.2920451699576349, 0.36558947821279086, 0.5256081668917854,
                                                    0.9865688990210231))
    np.testing.assert_almost_equal(lpcc(constF), (0.020164333842599635, 0.9865688990210177, 0.5256081668917822,
                                                  0.365589478212793, 0.29204516995764224, 0.2550754533117383,
                                                  0.2391793151101857, 0.2391793151101857, 0.2550754533117383,
                                                  0.29204516995764224, 0.365589478212793, 0.5256081668917822,
                                                  0.9865688990210177))
    np.testing.assert_almost_equal(lpcc(lin), (0.009787922299081098, 0.9403087900526141, 0.45515839811652303,
                                               0.2959902388191573, 0.2226794080995073, 0.18587538078947108,
                                               0.17006234165994988, 0.17006234165994988, 0.18587538078947108,
                                               0.2226794080995073, 0.2959902388191573, 0.45515839811652303,
                                               0.9403087900526141))
    np.testing.assert_almost_equal(lpcc(lin0), (0.14693248468111308, 0.7098379679548503, 0.27136979375401815,
                                                0.12066884688694682, 0.054365468824491156, 0.022184966988290034,
                                                0.00867554638640014, 0.00867554638640014, 0.022184966988290034,
                                                0.054365468824491156, 0.12066884688694682, 0.27136979375401815,
                                                0.7098379679548503))
    np.testing.assert_almost_equal(lpcc(wave), (0.27326478573784635, 2.2503511377184005, 1.3120406566259146,
                                                0.9508372630850437, 0.8377303045711273, 0.7195725472552679,
                                                0.715238952271539, 0.715238952271539, 0.7195725472552679,
                                                0.8377303045711273, 0.9508372630850437, 1.3120406566259146,
                                                2.2503511377184005))
    np.testing.assert_almost_equal(lpcc(offsetWave), (0.5435105244008235, 1.5815770053561224, 1.1189968861619681,
                                                      0.9577304362743059, 0.8832739401503552, 0.8458651104475441,
                                                      0.8295606293469393, 0.8295606293469393, 0.8458651104475441,
                                                      0.8832739401503552, 0.9577304362743059, 1.1189968861619681,
                                                      1.5815770053561224))
    np.testing.assert_almost_equal(lpcc(noiseWave), (0.332943861278751, 0.535742501182159, 0.6994290294792235,
                                                     0.699314211544821, 0.6700910097813829, 0.67785538535114,
                                                     0.7162476787322745, 0.7162476787322745, 0.67785538535114,
                                                     0.6700910097813829, 0.699314211544821, 0.6994290294792235,
                                                     0.535742501182159))


def test_spectral_entropy():
    np.testing.assert_almost_equal(spectral_entropy(const0, Fs), 0.0)
    np.testing.assert_almost_equal(spectral_entropy(const1, Fs), 0.0)
    np.testing.assert_almost_equal(spectral_entropy(constNeg, Fs), 0.0)
    np.testing.assert_almost_equal(spectral_entropy(constF, Fs), 0.0)
    np.testing.assert_almost_equal(spectral_entropy(lin, Fs), 0.6006757398806453)
    np.testing.assert_almost_equal(spectral_entropy(lin0, Fs), 0.57319032538303)
    np.testing.assert_almost_equal(spectral_entropy(wave, Fs), 1.5228376718814352e-29)
    np.testing.assert_almost_equal(spectral_entropy(offsetWave, Fs), 1.783049297437309e-29)
    np.testing.assert_almost_equal(spectral_entropy(noiseWave, Fs), 0.030107186831275425)


def test_wavelet_entropy():
    np.testing.assert_almost_equal(wavelet_entropy(const0), 0.0)
    np.testing.assert_almost_equal(wavelet_entropy(const1), 1.9188378548746368)
    np.testing.assert_almost_equal(wavelet_entropy(constNeg), 1.9188378548746368)
    np.testing.assert_almost_equal(wavelet_entropy(constF), 1.9188378548746368)
    np.testing.assert_almost_equal(wavelet_entropy(lin), 1.9648440772467513)
    np.testing.assert_almost_equal(wavelet_entropy(lin0), 2.0713919678725117)
    np.testing.assert_almost_equal(wavelet_entropy(wave), 1.7277528462213683)
    np.testing.assert_almost_equal(wavelet_entropy(offsetWave), 1.7965939302139549)
    np.testing.assert_almost_equal(wavelet_entropy(noiseWave), 2.0467527462416153)


def test_wavelet_abs_mean():
    np.testing.assert_almost_equal(wavelet_abs_mean(const0), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    np.testing.assert_almost_equal(wavelet_abs_mean(const1),
                                   (0.081894185676901, 0.24260084511769256, 0.4653470776794248,
                                    0.8500400580778283, 1.3602249381214044, 1.8378460432593602,
                                    2.2080039502231164, 2.4676456085810874, 2.638131856418627))
    np.testing.assert_almost_equal(wavelet_abs_mean(constNeg),
                                   (0.081894185676901, 0.24260084511769256, 0.4653470776794248,
                                    0.8500400580778283, 1.3602249381214044, 1.8378460432593602,
                                    2.2080039502231164, 2.4676456085810874, 2.638131856418627))
    np.testing.assert_almost_equal(wavelet_abs_mean(constF),
                                   (0.20473546419225214, 0.6065021127942314, 1.1633676941985622,
                                    2.1251001451945712, 3.4005623453035114, 4.5946151081484015,
                                    5.5200098755577915, 6.169114021452717, 6.5953296410465665))
    np.testing.assert_almost_equal(wavelet_abs_mean(lin), (0.7370509925842613, 2.183416725919023, 4.1974435700809565,
                                                           7.744819422931153, 12.504051331233388, 16.982183932901865,
                                                           20.46332353598833, 22.91143100556329, 24.52363151471446))
    np.testing.assert_almost_equal(wavelet_abs_mean(lin0),
                                   (0.0430987066803135, 0.12767505547269026, 0.23510912407745171,
                                    0.3479590829560181, 0.4400900851788993, 0.5024773453284851,
                                    0.5396989380329178, 0.5591602904810937, 0.5669696013289379))
    np.testing.assert_almost_equal(wavelet_abs_mean(wave), (5.138703105035948e-05, 0.00015178141653400073,
                                                            0.00027925117450851024, 0.0004278724786267016,
                                                            0.0005932191214607947, 0.0007717034331954587,
                                                            0.0009601854175466062, 0.0011557903088208192,
                                                            0.0013558175034366186))
    np.testing.assert_almost_equal(wavelet_abs_mean(offsetWave), (0.0032504208945027323, 0.009623752088931016,
                                                                  0.017761411181034453, 0.027372614777691914,
                                                                  0.03826512918833778, 0.050306487368868114,
                                                                  0.06339897203822373, 0.07746693331944604,
                                                                  0.09244971907566273))
    np.testing.assert_almost_equal(wavelet_abs_mean(noiseWave), (4.631139377647647e-05, 7.893225282164063e-05,
                                                                 0.00033257747958655794, 0.0005792253883615155,
                                                                 0.0007699898255271558, 0.0009106252575513913,
                                                                 0.0010387197644970154, 0.0011789334866018457,
                                                                 0.0013341945911985783))


def test_wavelet_std():
    np.testing.assert_almost_equal(wavelet_std(const0), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    np.testing.assert_almost_equal(wavelet_std(const1), (0.1767186264889806, 0.28069306259219023, 0.3235061868750311,
                                                         0.3115893726751135, 0.31446140614407014, 0.3582016825631658,
                                                         0.4133090941627322, 0.4598585090675407, 0.4935514064162697))
    np.testing.assert_almost_equal(wavelet_std(constNeg), (0.1767186264889806, 0.28069306259219023, 0.3235061868750311,
                                                           0.3115893726751135, 0.31446140614407014, 0.3582016825631658,
                                                           0.4133090941627322, 0.4598585090675407, 0.4935514064162697))
    np.testing.assert_almost_equal(wavelet_std(constF), (0.44179656622245145, 0.7017326564804757, 0.8087654671875778,
                                                         0.7789734316877838, 0.7861535153601755, 0.8955042064079146,
                                                         1.0332727354068305, 1.1496462726688517, 1.2338785160406742))
    np.testing.assert_almost_equal(wavelet_std(lin), (2.721791561180164, 5.325234998810811, 8.137581399111415,
                                                      10.529795250703716, 11.836525442245224, 12.296195571788726,
                                                      12.315744378517108, 12.135259348389042, 11.869294506387352))
    np.testing.assert_almost_equal(wavelet_std(lin0), (2.239406940011677, 4.7878443746478245, 7.797954379287043,
                                                       10.418506686200207, 11.746946049852674, 12.045972295386465,
                                                       11.828477896749822, 11.408150997410496, 10.932763618021895))
    np.testing.assert_almost_equal(wavelet_std(wave), (0.001939366875349316, 0.009733675496927717, 0.025635801097107388,
                                                       0.05125305898778544, 0.08783649118731567, 0.13636963970273208,
                                                       0.197613166916789, 0.2721306670702481, 0.360305525758368))
    np.testing.assert_almost_equal(wavelet_std(offsetWave),
                                   (0.05459142980660159, 0.10410347082332229, 0.155831467554863,
                                    0.2101395066938644, 0.268489203478025, 0.33264452641566,
                                    0.4044076212671741, 0.4854392072251105, 0.5771385517659353))
    np.testing.assert_almost_equal(wavelet_std(noiseWave),
                                   (0.08974931069698587, 0.09625674025798765, 0.10445386849293256,
                                    0.11395751571203461, 0.13232763520967267, 0.16659967802754122,
                                    0.2187573594673847, 0.2877270278501564, 0.3722670641715661))


def test_wavelet_var():
    np.testing.assert_almost_equal(wavelet_var(const0), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    np.testing.assert_almost_equal(wavelet_var(const1), (0.031229472948151833, 0.07878859538738324, 0.10465625294642253,
                                                         0.09708793716407076, 0.09888597595410582, 0.128308445391083,
                                                         0.17082440731761822, 0.21146984836182142, 0.24359299077547786))
    np.testing.assert_almost_equal(wavelet_var(constNeg),
                                   (0.031229472948151833, 0.07878859538738324, 0.10465625294642253,
                                    0.09708793716407076, 0.09888597595410582, 0.128308445391083,
                                    0.17082440731761822, 0.21146984836182142, 0.24359299077547786))
    np.testing.assert_almost_equal(wavelet_var(constF), (0.19518420592594893, 0.49242872117114533, 0.654101580915141,
                                                         0.6067996072754422, 0.6180373497131617, 0.8019277836942689,
                                                         1.0676525457351138, 1.3216865522613839, 1.5224561923467361))
    np.testing.assert_almost_equal(wavelet_var(lin), (7.408149302511555, 28.35812779255958, 66.22023102716409,
                                                      110.87658802174253, 140.10333454491848, 151.19642553967668,
                                                      151.67755959697575, 147.26451945266362, 140.88015207935698))
    np.testing.assert_almost_equal(wavelet_var(lin0), (5.014943442972464, 22.923453755846815, 60.808092501441976,
                                                       108.5452815703984, 137.99074149814933, 145.10544854121827,
                                                       139.91288935389912, 130.1459091797181, 119.5253203275432))
    np.testing.assert_almost_equal(wavelet_var(wave),
                                   (3.761143877202169e-06, 9.474443867949103e-05, 0.0006571942978904524,
                                    0.0026268760556054137, 0.007715249184099382, 0.018596678632652963,
                                    0.03905096373888271, 0.07405509996009821, 0.12982007189201397))
    np.testing.assert_almost_equal(wavelet_var(offsetWave),
                                   (0.0029802242083291084, 0.010837532637462314, 0.02428344628030232,
                                    0.044158612273540676, 0.07208645238426431, 0.11065238095429873,
                                    0.16354552413897414, 0.2356512239113438, 0.33308890793448115))
    np.testing.assert_almost_equal(wavelet_var(noiseWave),
                                   (0.008054938770584103, 0.0092653600450937, 0.01091061064313885,
                                    0.012986315387258616, 0.017510603040184203, 0.027755452718880403,
                                    0.04785478232114257, 0.08278684255548469, 0.1385827670669169))


def test_wavelet_energy():
    np.testing.assert_almost_equal(wavelet_energy(const0), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    np.testing.assert_almost_equal(wavelet_energy(const1), (0.19477199643643478, 0.3710037269882903, 0.56674875884399,
                                                            0.9053485723747671, 1.3961009484422982, 1.8724279756816202,
                                                            2.2463539016634275, 2.510128422593423, 2.683902509896041))
    np.testing.assert_almost_equal(wavelet_energy(constNeg), (0.19477199643643478, 0.3710037269882903, 0.56674875884399,
                                                              0.9053485723747671, 1.3961009484422982,
                                                              1.8724279756816202,
                                                              2.2463539016634275, 2.510128422593423, 2.683902509896041))
    np.testing.assert_almost_equal(wavelet_energy(constF), (0.48692999109108687, 0.9275093174707258, 1.4168718971099752,
                                                            2.263371430936918, 3.4902523711057456, 4.6810699392040505,
                                                            5.615884754158569, 6.275321056483556, 6.709756274740101))
    np.testing.assert_almost_equal(wavelet_energy(lin), (2.819821531264169, 5.7554701277638936, 9.156350995411767,
                                                         13.071297407509103, 17.21785800380053, 20.966425462405052,
                                                         23.883575313078858, 25.926785187819767, 27.244974853151422))
    np.testing.assert_almost_equal(wavelet_energy(lin0), (2.2398216316238173, 4.789546395603321, 7.8014978562880115,
                                                          10.424315665491429, 11.75518697346929, 12.056447736534448,
                                                          11.84078393931808, 11.421846147193937, 10.947455177180416))
    np.testing.assert_almost_equal(wavelet_energy(wave),
                                   (0.0019400475520363772, 0.00973485882167256, 0.025637321995655413,
                                    0.051254844946242696, 0.08783849436907175, 0.13637182318514984,
                                    0.19761549963228792, 0.2721331214889804, 0.3603080766970352))
    np.testing.assert_almost_equal(wavelet_energy(offsetWave),
                                   (0.054688110630378595, 0.10454735406375197, 0.15684040935755078,
                                    0.21191477606176637, 0.27120227229148447, 0.3364269959823273,
                                    0.4093469845918956, 0.49158147815928066, 0.584496243351187))
    np.testing.assert_almost_equal(wavelet_energy(noiseWave),
                                   (0.08974932264551803, 0.09625677262091348, 0.10445439794914707,
                                    0.11395898775133596, 0.13232987540429264, 0.16660216672432593,
                                    0.2187598255162308, 0.2877294431226156, 0.37226945502166053))


run_module_suite()
