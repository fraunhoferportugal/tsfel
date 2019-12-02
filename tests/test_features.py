import numpy as np
from tsfel.feature_extraction.features import *
from numpy.testing import assert_array_equal, run_module_suite

# Implementing signals for testing features

const0 = np.zeros(20)
const1 = np.ones(20)
constNeg = np.ones(20)*(-1)
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
    np.testing.assert_equal(hist(const0, 10, 5), (0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0))
    np.testing.assert_equal(hist(const1, 10, 5), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0))
    np.testing.assert_equal(hist(constNeg, 10, 5), (0.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    np.testing.assert_equal(hist(constF, 10, 5),  (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0))
    np.testing.assert_equal(hist(lin, 10, 5), (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2))
    np.testing.assert_almost_equal(hist(wave, 10, 5), (0.0, 0.0, 0.0, 0.0, 499, 496, 5, 0.0, 0.0, 0.0), decimal=5)
    np.testing.assert_almost_equal(hist(offsetWave, 10, 5), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 499, 496, 5, 0.0), decimal=5)
    np.testing.assert_almost_equal(hist(noiseWave, 10, 5), (0.0, 0.0, 0.0, 48, 446, 450, 56, 0.0, 0.0, 0.0), decimal=5)


def test_skewness():
    np.testing.assert_equal(skewness(const0), 0.0)
    np.testing.assert_equal(skewness(const1), 0.0)
    np.testing.assert_equal(skewness(constNeg), 0.0)
    np.testing.assert_equal(skewness(constF), 0.0)
    np.testing.assert_equal(skewness(lin), 0)
    np.testing.assert_almost_equal(skewness(lin0), -1.0167718723297815e-16, decimal=5)
    np.testing.assert_almost_equal(skewness(wave), -2.009718347115232e-17, decimal=5)
    np.testing.assert_almost_equal(skewness(offsetWave), 9.043732562018544e-16, decimal=5)
    np.testing.assert_almost_equal(skewness(noiseWave), -0.0004854111290521465, decimal=5)


def test_kurtosis():
    np.testing.assert_equal(kurtosis(const0), -3)
    np.testing.assert_equal(kurtosis(const1), -3)
    np.testing.assert_equal(kurtosis(constNeg), -3)
    np.testing.assert_equal(kurtosis(constF), -3.0)
    np.testing.assert_almost_equal(kurtosis(lin), -1.206015037593985, decimal=2)
    np.testing.assert_almost_equal(kurtosis(lin0), -1.2060150375939847, decimal=2)
    np.testing.assert_almost_equal(kurtosis(wave), -1.501494077162359, decimal=2)
    np.testing.assert_almost_equal(kurtosis(offsetWave), -1.5014940771623597, decimal=2)
    np.testing.assert_almost_equal(kurtosis(noiseWave), -1.4606204906023366, decimal=2)


def test_mean():
    np.testing.assert_equal(calc_mean(const0), 0.0)
    np.testing.assert_equal(calc_mean(const1), 1.0)
    np.testing.assert_equal(calc_mean(constNeg), -1.0)
    np.testing.assert_equal(calc_mean(constF), 2.5)
    np.testing.assert_equal(calc_mean(lin), 9.5)
    np.testing.assert_almost_equal(calc_mean(lin0), -3.552713678800501e-16, decimal=5)
    np.testing.assert_almost_equal(calc_mean(wave), 7.105427357601002e-18, decimal=5)
    np.testing.assert_almost_equal(calc_mean(offsetWave), 2.0, decimal=5)
    np.testing.assert_almost_equal(calc_mean(noiseWave), -0.0014556635615470554, decimal=5)


def test_median():
    np.testing.assert_equal(calc_median(const0), 0.0)
    np.testing.assert_equal(calc_median(const1), 1.0)
    np.testing.assert_equal(calc_median(constNeg), -1.0)
    np.testing.assert_equal(calc_median(constF), 2.5)
    np.testing.assert_equal(calc_median(lin), 9.5)
    np.testing.assert_almost_equal(calc_median(lin0), -3.552713678800501e-16, decimal=5)
    np.testing.assert_almost_equal(calc_median(wave), 7.105427357601002e-18, decimal=5)
    np.testing.assert_almost_equal(calc_median(offsetWave), 2.0, decimal=5)
    np.testing.assert_almost_equal(calc_median(noiseWave), 0.013846093997438328, decimal=5)


def test_max():
    np.testing.assert_equal(calc_max(const0), 0.0)
    np.testing.assert_equal(calc_max(const1), 1.0)
    np.testing.assert_equal(calc_max(constNeg), -1.0)
    np.testing.assert_equal(calc_max(constF), 2.5)
    np.testing.assert_equal(calc_max(lin), 19)
    np.testing.assert_almost_equal(calc_max(lin0), 10.0, decimal=5)
    np.testing.assert_almost_equal(calc_max(wave), 1.0, decimal=5)
    np.testing.assert_almost_equal(calc_max(noiseWave), 1.221757617217142, decimal=5)
    np.testing.assert_almost_equal(calc_max(offsetWave), 3.0, decimal=5)


def test_min():
    np.testing.assert_equal(calc_min(const0), 0.0)
    np.testing.assert_equal(calc_min(const1), 1.0)
    np.testing.assert_equal(calc_min(constNeg), -1.0)
    np.testing.assert_equal(calc_min(constF), 2.5)
    np.testing.assert_equal(calc_min(lin), 0)
    np.testing.assert_almost_equal(calc_min(lin0), -10.0, decimal=5)
    np.testing.assert_almost_equal(calc_min(wave), -1.0, decimal=5)
    np.testing.assert_almost_equal(calc_min(noiseWave), -1.2582533627830566, decimal=5)
    np.testing.assert_almost_equal(calc_min(offsetWave), 1.0, decimal=5)


def test_variance():
    np.testing.assert_equal(calc_var(const0), 0.0)
    np.testing.assert_equal(calc_var(const1), 0.0)
    np.testing.assert_equal(calc_var(constNeg), 0.0)
    np.testing.assert_equal(calc_var(constF), 0.0)
    np.testing.assert_equal(calc_var(lin), 33.25)
    np.testing.assert_almost_equal(calc_var(lin0), 36.84210526315789, decimal=5)
    np.testing.assert_almost_equal(calc_var(wave), 0.5, decimal=5)
    np.testing.assert_almost_equal(calc_var(offsetWave), 0.5, decimal=5)
    np.testing.assert_almost_equal(calc_var(noiseWave), 0.5081167177369529, decimal=5)


def test_std():
    np.testing.assert_equal(calc_std(const0), 0.0)
    np.testing.assert_equal(calc_std(const1), 0.0)
    np.testing.assert_equal(calc_std(constNeg), 0.0)
    np.testing.assert_equal(calc_std(constF), 0.0)
    np.testing.assert_equal(calc_std(lin), 5.766281297335398)
    np.testing.assert_almost_equal(calc_std(lin0), 6.069769786668839, decimal=5)
    np.testing.assert_almost_equal(calc_std(wave), 0.7071067811865476, decimal=5)
    np.testing.assert_almost_equal(calc_std(offsetWave), 0.7071067811865476, decimal=5)
    np.testing.assert_almost_equal(calc_std(noiseWave), 0.7128230620125536, decimal=5)


def test_interq_range():
    np.testing.assert_equal(interq_range(const0), 0.0)
    np.testing.assert_equal(interq_range(const1), 0.0)
    np.testing.assert_equal(interq_range(constNeg), 0.0)
    np.testing.assert_equal(interq_range(constF), 0.0)
    np.testing.assert_equal(interq_range(lin), 9.5)
    np.testing.assert_almost_equal(interq_range(lin0), 10.0, decimal=5)
    np.testing.assert_almost_equal(interq_range(wave), 1.414213562373095, decimal=5)
    np.testing.assert_almost_equal(interq_range(offsetWave), 1.414213562373095, decimal=5)
    np.testing.assert_almost_equal(interq_range(noiseWave), 1.4277110228590328, decimal=5)


def test_mean_abs_diff():
    np.testing.assert_equal(mean_abs_diff(const0), 0.0)
    np.testing.assert_equal(mean_abs_diff(const1), 0.0)
    np.testing.assert_equal(mean_abs_diff(constNeg), 0.0)
    np.testing.assert_equal(mean_abs_diff(constF), 0.0)
    np.testing.assert_equal(mean_abs_diff(lin), 1.0)
    np.testing.assert_almost_equal(mean_abs_diff(lin0), 1.0526315789473684, decimal=5)
    np.testing.assert_almost_equal(mean_abs_diff(wave), 0.019988577818740614, decimal=5)
    np.testing.assert_almost_equal(mean_abs_diff(noiseWave), 0.10700252903161511, decimal=5)
    np.testing.assert_almost_equal(mean_abs_diff(offsetWave), 0.019988577818740614, decimal=5)


def test_mean_abs_deviation():
    np.testing.assert_equal(mean_abs_deviation(const0), 0.0)
    np.testing.assert_equal(mean_abs_deviation(const1), 0.0)
    np.testing.assert_equal(mean_abs_deviation(constNeg), 0.0)
    np.testing.assert_equal(mean_abs_deviation(constF), 0.0)
    np.testing.assert_equal(mean_abs_deviation(lin), 5.0)
    np.testing.assert_almost_equal(mean_abs_deviation(lin0), 5.263157894736842, decimal=5)
    np.testing.assert_almost_equal(mean_abs_deviation(wave), 0.6365674116287157, decimal=5)
    np.testing.assert_almost_equal(mean_abs_deviation(noiseWave), 0.6392749078483896, decimal=5)
    np.testing.assert_almost_equal(mean_abs_deviation(offsetWave), 0.6365674116287157, decimal=5)


def test_calc_median_abs_deviation():
    np.testing.assert_equal(median_abs_deviation(const0), 0.0)
    np.testing.assert_equal(median_abs_deviation(const1), 0.0)
    np.testing.assert_equal(median_abs_deviation(constNeg), 0.0)
    np.testing.assert_equal(median_abs_deviation(constF), 0.0)
    np.testing.assert_equal(median_abs_deviation(lin), 5.0)
    np.testing.assert_almost_equal(median_abs_deviation(lin0), 5.2631578947368425, decimal=5)
    np.testing.assert_almost_equal(median_abs_deviation(wave), 0.7071067811865475, decimal=5)
    np.testing.assert_almost_equal(median_abs_deviation(offsetWave), 0.7071067811865475, decimal=5)
    np.testing.assert_almost_equal(median_abs_deviation(noiseWave), 0.7068117164205888, decimal=5)


def test_rms():
    np.testing.assert_equal(rms(const0), 0.0)
    np.testing.assert_equal(rms(const1), 1.0)
    np.testing.assert_equal(rms(constNeg), 1.0)
    np.testing.assert_equal(rms(constF), 2.5)
    np.testing.assert_equal(rms(lin), 11.113055385446435)
    np.testing.assert_almost_equal(rms(lin0), 6.06976978666884, decimal=5)
    np.testing.assert_almost_equal(rms(wave), 0.7071067811865476, decimal=5)
    np.testing.assert_almost_equal(rms(offsetWave), 2.1213203435596424, decimal=5)
    np.testing.assert_almost_equal(rms(noiseWave), 0.7128245483240299, decimal=5)


# ################################################ TEMPORAL FEATURES ################################################# #
def test_distance():
    np.testing.assert_equal(distance(const0), 19.0)
    np.testing.assert_equal(distance(const1), 19.0)
    np.testing.assert_equal(distance(constNeg), 19.0)
    np.testing.assert_equal(distance(constF), 19.0)
    np.testing.assert_equal(distance(lin), 26.87005768508881)
    np.testing.assert_almost_equal(distance(lin0), 27.586228448267438, decimal=5)
    np.testing.assert_almost_equal(distance(wave), 999.2461809866238, decimal=5)
    np.testing.assert_almost_equal(distance(offsetWave), 999.2461809866238, decimal=5)
    np.testing.assert_almost_equal(distance(noiseWave), 1007.8711901383033, decimal=5)


def test_minpeaks():
    np.testing.assert_equal(minpeaks(const0), 0.0)
    np.testing.assert_equal(minpeaks(const1), 0.0)
    np.testing.assert_equal(minpeaks(constNeg), 0.0)
    np.testing.assert_equal(minpeaks(constF), 0.0)
    np.testing.assert_equal(minpeaks(lin), 0.0)
    np.testing.assert_almost_equal(minpeaks(lin0), 0.0, decimal=5)
    np.testing.assert_almost_equal(minpeaks(wave), 5, decimal=5)
    np.testing.assert_almost_equal(minpeaks(offsetWave), 5, decimal=5)
    np.testing.assert_almost_equal(minpeaks(noiseWave), 323, decimal=5)


def test_maxpeaks():
    np.testing.assert_equal(maxpeaks(const0), 0.0)
    np.testing.assert_equal(maxpeaks(const1), 0.0)
    np.testing.assert_equal(maxpeaks(constNeg), 0.0)
    np.testing.assert_equal(maxpeaks(constF), 0.0)
    np.testing.assert_equal(maxpeaks(lin), 0.0)
    np.testing.assert_almost_equal(maxpeaks(lin0), 0.0, decimal=5)
    np.testing.assert_almost_equal(maxpeaks(wave), 5, decimal=5)
    np.testing.assert_almost_equal(maxpeaks(offsetWave), 5, decimal=5)
    np.testing.assert_almost_equal(maxpeaks(noiseWave), 322, decimal=5)


def test_calc_centroid():
    np.testing.assert_equal(calc_centroid(const0, Fs), 0.0)
    np.testing.assert_equal(calc_centroid(const1, Fs), 0.009499999999999998)
    np.testing.assert_equal(calc_centroid(constNeg, Fs), 0.009499999999999998)
    np.testing.assert_equal(calc_centroid(constF, Fs), 0.0095)
    np.testing.assert_equal(calc_centroid(lin, Fs), 0.014615384615384615)
    np.testing.assert_almost_equal(calc_centroid(lin0, Fs), 0.0095, decimal=5)
    np.testing.assert_almost_equal(calc_centroid(wave, Fs), 0.5000000000000001, decimal=5)
    np.testing.assert_almost_equal(calc_centroid(offsetWave, Fs), 0.47126367059427926, decimal=5)
    np.testing.assert_almost_equal(calc_centroid(noiseWave, Fs), 0.4996034303128802, decimal=5)


def test_mean_diff():
    np.testing.assert_equal(mean_diff(const0), 0.0)
    np.testing.assert_equal(mean_diff(const1), 0.0)
    np.testing.assert_equal(mean_diff(constNeg), 0.0)
    np.testing.assert_equal(mean_diff(constF), 0.0)
    np.testing.assert_equal(mean_diff(lin), 1.0)
    np.testing.assert_almost_equal(mean_diff(lin0), 1.0526315789473684, decimal=5)
    np.testing.assert_almost_equal(mean_diff(wave), -3.1442201279407477e-05, decimal=5)
    np.testing.assert_almost_equal(mean_diff(offsetWave), -3.1442201279407036e-05, decimal=5)
    np.testing.assert_almost_equal(mean_diff(noiseWave), -0.00010042477181949707, decimal=5)


def test_median_diff():
    np.testing.assert_equal(median_diff(const0), 0.0)
    np.testing.assert_equal(median_diff(const1), 0.0)
    np.testing.assert_equal(median_diff(constNeg), 0.0)
    np.testing.assert_equal(median_diff(constF), 0.0)
    np.testing.assert_equal(median_diff(lin), 1.0)
    np.testing.assert_almost_equal(median_diff(lin0), 1.0526315789473684, decimal=5)
    np.testing.assert_almost_equal(median_diff(wave), -0.0004934396342684, decimal=5)
    np.testing.assert_almost_equal(median_diff(offsetWave), -0.0004934396342681779, decimal=5)
    np.testing.assert_almost_equal(median_diff(noiseWave), -0.004174819648320949, decimal=5)


def test_calc_mean_abs_diff():
    np.testing.assert_equal(mean_abs_diff(const0), 0.0)
    np.testing.assert_equal(mean_abs_diff(const1), 0.0)
    np.testing.assert_equal(mean_abs_diff(constNeg), 0.0)
    np.testing.assert_equal(mean_abs_diff(constF), 0.0)
    np.testing.assert_equal(mean_abs_diff(lin), 1.0)
    np.testing.assert_almost_equal(mean_abs_diff(lin0), 1.0526315789473684, decimal=5)
    np.testing.assert_almost_equal(mean_abs_diff(wave), 0.019988577818740614, decimal=5)
    np.testing.assert_almost_equal(mean_abs_diff(offsetWave), 0.019988577818740614, decimal=5)
    np.testing.assert_almost_equal(mean_abs_diff(noiseWave), 0.10700252903161508, decimal=5)


def test_median_abs_diff():
    np.testing.assert_equal(median_abs_diff(const0), 0.0)
    np.testing.assert_equal(median_abs_diff(const1), 0.0)
    np.testing.assert_equal(median_abs_diff(constNeg), 0.0)
    np.testing.assert_equal(median_abs_diff(constF), 0.0)
    np.testing.assert_equal(median_abs_diff(lin), 1.0)
    np.testing.assert_almost_equal(median_abs_diff(lin0), 1.0526315789473681, decimal=5)
    np.testing.assert_almost_equal(median_abs_diff(wave), 0.0218618462348652, decimal=5)
    np.testing.assert_almost_equal(median_abs_diff(offsetWave), 0.021861846234865645, decimal=5)
    np.testing.assert_almost_equal(median_abs_diff(noiseWave), 0.08958750592592835, decimal=5)


def test_sum_abs_diff():
    np.testing.assert_equal(sum_abs_diff(const0), 0.0)
    np.testing.assert_equal(sum_abs_diff(const1), 0.0)
    np.testing.assert_equal(sum_abs_diff(constNeg), 0.0)
    np.testing.assert_equal(sum_abs_diff(constF), 0.0)
    np.testing.assert_equal(sum_abs_diff(lin), 19)
    np.testing.assert_almost_equal(sum_abs_diff(lin0), 20.0, decimal=5)
    np.testing.assert_almost_equal(sum_abs_diff(wave), 19.968589240921872, decimal=5)
    np.testing.assert_almost_equal(sum_abs_diff(offsetWave), 19.968589240921872, decimal=5)
    np.testing.assert_almost_equal(sum_abs_diff(noiseWave), 106.89552650258346, decimal=5)


def test_zerocross():
    np.testing.assert_equal(zero_cross(const0), 0.0)
    np.testing.assert_equal(zero_cross(const1), 0.0)
    np.testing.assert_equal(zero_cross(constNeg), 0.0)
    np.testing.assert_equal(zero_cross(constF), 0.0)
    np.testing.assert_equal(zero_cross(lin), 1.0)
    np.testing.assert_almost_equal(zero_cross(lin0), 1.0, decimal=5)
    np.testing.assert_almost_equal(zero_cross(wave), 10, decimal=5)
    np.testing.assert_almost_equal(zero_cross(offsetWave), 0.0, decimal=5)
    np.testing.assert_almost_equal(zero_cross(noiseWave), 38, decimal=5)


def test_autocorr():
    np.testing.assert_equal(autocorr(const0), 0.0)
    np.testing.assert_equal(autocorr(const1), 20.0)
    np.testing.assert_equal(autocorr(constNeg), 20.0)
    np.testing.assert_equal(autocorr(constF), 125.0)
    np.testing.assert_equal(autocorr(lin), 2470.0)
    np.testing.assert_almost_equal(autocorr(lin0), 736.8421052631579, decimal=0)
    np.testing.assert_almost_equal(autocorr(wave), 500.5, decimal=0)
    np.testing.assert_almost_equal(autocorr(offsetWave), 4500.0, decimal=0)
    np.testing.assert_almost_equal(autocorr(noiseWave), 508.6149018530489, decimal=0)


def test_auc():
    np.testing.assert_equal(auc(const0, Fs), 0.0)
    np.testing.assert_equal(auc(const1, Fs), 9.518999999999998)
    np.testing.assert_equal(auc(constNeg, Fs), -9.518999999999998)
    np.testing.assert_equal(auc(constF, Fs), 23.797500000000003)
    np.testing.assert_equal(auc(lin, Fs), 95.171)
    np.testing.assert_equal(auc(lin0, Fs), 4.989999999999997)
    np.testing.assert_equal(auc(wave, Fs), 3.1410759074645966e-05)
    np.testing.assert_equal(auc(offsetWave, Fs), 1000.998031410759)
    np.testing.assert_equal(auc(noiseWave, Fs), -0.7958996038449087)


def test_abs_energy():
    np.testing.assert_equal(abs_energy(const0), 0.0)
    np.testing.assert_equal(abs_energy(const1), 20.0)
    np.testing.assert_equal(abs_energy(constNeg), 20.0)
    np.testing.assert_equal(abs_energy(constF), 125.0)
    np.testing.assert_equal(abs_energy(lin), 2470)
    np.testing.assert_equal(abs_energy(lin0), 736.8421052631579)
    np.testing.assert_equal(abs_energy(wave), 500.0)
    np.testing.assert_equal(abs_energy(offsetWave), 4500.0)
    np.testing.assert_equal(abs_energy(noiseWave), 508.11883669335725)


def test_pk_pk_distance():
    np.testing.assert_equal(pk_pk_distance(const0), 0.0)
    np.testing.assert_equal(pk_pk_distance(const1), 0.0)
    np.testing.assert_equal(pk_pk_distance(constNeg), 0.0)
    np.testing.assert_equal(pk_pk_distance(constF), 0.0)
    np.testing.assert_equal(pk_pk_distance(lin), 19)
    np.testing.assert_equal(pk_pk_distance(lin0), 20.0)
    np.testing.assert_equal(pk_pk_distance(wave), 2.0)
    np.testing.assert_equal(pk_pk_distance(offsetWave), 2.0)
    np.testing.assert_equal(pk_pk_distance(noiseWave), 2.4800109800001993)


def test_slope():
    np.testing.assert_equal(slope(const0), 0.0)
    np.testing.assert_equal(slope(const1), -8.935559365603017e-18)
    np.testing.assert_equal(slope(constNeg), 8.935559365603017e-18)
    np.testing.assert_equal(slope(constF), 1.7871118731206033e-17)
    np.testing.assert_equal(slope(lin), 1.0)
    np.testing.assert_equal(slope(lin0), 1.0526315789473686)
    np.testing.assert_equal(slope(wave), -0.0003819408289180587)
    np.testing.assert_equal(slope(offsetWave), -0.00038194082891805853)
    np.testing.assert_equal(slope(noiseWave), -0.00040205425841671337)


def test_entropy():
    np.testing.assert_equal(entropy(const0), 0.0)
    np.testing.assert_equal(entropy(const1), 0.0)
    np.testing.assert_equal(entropy(constNeg),0.0)
    np.testing.assert_equal(entropy(constF), 0.0)
    np.testing.assert_equal(entropy(lin), 0.994983274605318)
    np.testing.assert_equal(entropy(lin0), 0.994983274605318)
    np.testing.assert_equal(entropy(wave), 0.9972021515128497)
    np.testing.assert_equal(entropy(offsetWave), 0.99720215151285)
    np.testing.assert_equal(entropy(noiseWave), 0.9957000733996481)


# ################################################ SPECTRAL FEATURES ################################################# #
def test_max_fre():
    np.testing.assert_equal(max_frequency(const0, Fs), 0.0)
    np.testing.assert_equal(max_frequency(const1, Fs), 0.0)
    np.testing.assert_equal(max_frequency(constNeg, Fs), 0.0)
    np.testing.assert_equal(max_frequency(constF, Fs), 0.0)
    np.testing.assert_equal(max_frequency(lin, Fs), 444.44444444444446)
    np.testing.assert_almost_equal(max_frequency(lin0, Fs), 500.0, decimal=5)
    np.testing.assert_almost_equal(max_frequency(wave, Fs), 5.0100200400801596, decimal=5)
    np.testing.assert_almost_equal(max_frequency(offsetWave, Fs), 5.0100200400801596, decimal=5)
    np.testing.assert_almost_equal(max_frequency(noiseWave, Fs), 464.9298597194388, decimal=5)
    np.testing.assert_almost_equal(max_frequency(x, Fs),  344.689378757515, decimal=1)


def test_med_fre():
    np.testing.assert_equal(median_frequency(const0, Fs), 0.0)
    np.testing.assert_equal(median_frequency(const1, Fs), 0.0)
    np.testing.assert_equal(median_frequency(constNeg, Fs), 0.0)
    np.testing.assert_equal(median_frequency(constF, Fs), 0.0)
    np.testing.assert_equal(median_frequency(lin, Fs), 55.55555555555556)
    np.testing.assert_almost_equal(median_frequency(lin0, Fs), 166.66666666666669, decimal=5)
    np.testing.assert_almost_equal(median_frequency(wave, Fs), 5.0100200400801596, decimal=5)
    np.testing.assert_almost_equal(median_frequency(offsetWave, Fs), 0.0, decimal=5)
    np.testing.assert_almost_equal(median_frequency(noiseWave, Fs), 146.29258517034066, decimal=5)
    np.testing.assert_almost_equal(median_frequency(x, Fs), 4.008016032064128, decimal=1)


def test_fund_fre():
    np.testing.assert_equal(fundamental_frequency(const0, 1), 0.0)
    np.testing.assert_equal(fundamental_frequency(const1, 1), 0.0)
    np.testing.assert_equal(fundamental_frequency(constNeg, Fs), 0.0)
    np.testing.assert_equal(fundamental_frequency(constF, Fs), 0.0)
    np.testing.assert_equal(fundamental_frequency(lin, Fs), 0.0)
    np.testing.assert_almost_equal(fundamental_frequency(lin0, Fs), 0.0, decimal=5)
    np.testing.assert_almost_equal(fundamental_frequency(wave, Fs), 5.0100200400801596, decimal=1)
    np.testing.assert_almost_equal(fundamental_frequency(offsetWave, Fs), 5.0100200400801596, decimal=1)
    np.testing.assert_almost_equal(fundamental_frequency(noiseWave, Fs), 5.0100200400801596, decimal=1)


def test_power_spec():
    np.testing.assert_equal(max_power_spectrum(const0, Fs), 0.0)
    np.testing.assert_equal(max_power_spectrum(const1, Fs), 0.0)
    np.testing.assert_equal(max_power_spectrum(constNeg, Fs), 0.0)
    np.testing.assert_equal(max_power_spectrum(constF, Fs), 0.0)
    np.testing.assert_equal(max_power_spectrum(lin, Fs), 0.004621506382612649)
    np.testing.assert_almost_equal(max_power_spectrum(lin0, Fs), 0.0046215063826126525, decimal=5)
    np.testing.assert_almost_equal(max_power_spectrum(wave, Fs), 0.6666666666666667, decimal=5)
    np.testing.assert_almost_equal(max_power_spectrum(offsetWave, Fs), 0.6666666666666667, decimal=5)
    np.testing.assert_almost_equal(max_power_spectrum(noiseWave, Fs),0.6570878541643916, decimal=5)


def test_total_energy():
    np.testing.assert_equal(total_energy(const0, Fs), 0.0)
    np.testing.assert_equal(total_energy(const1, Fs), 1052.6315789473686)
    np.testing.assert_equal(total_energy(constNeg, Fs), 1052.6315789473686)
    np.testing.assert_equal(total_energy(constF, Fs), 6578.9473684210525)
    np.testing.assert_equal(total_energy(lin, Fs), 130000.0)
    np.testing.assert_almost_equal(total_energy(lin0, Fs), 38781.16343490305, decimal=5)
    np.testing.assert_almost_equal(total_energy(wave, Fs), 500.5005005005005, decimal=5)
    np.testing.assert_almost_equal(total_energy(offsetWave, Fs), 4504.504504504504, decimal=5)
    np.testing.assert_almost_equal(total_energy(noiseWave, Fs), 508.6274641575148, decimal=5)


def test_spectral_centroid():
    np.testing.assert_equal(spectral_centroid(const0, Fs), 0.0)
    np.testing.assert_equal(spectral_centroid(const1, Fs), 2.7476856540265033e-14)
    np.testing.assert_equal(spectral_centroid(constNeg, Fs), 2.7476856540265033e-14)
    np.testing.assert_equal(spectral_centroid(constF, Fs), 2.4504208511457478e-14)
    np.testing.assert_equal(spectral_centroid(lin, Fs), 95.77382394996009)
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
    np.testing.assert_almost_equal(spectral_skewness(const1, Fs), 118013206.35142924, decimal=5)
    np.testing.assert_almost_equal(spectral_skewness(constNeg, Fs), 118013206.35142924, decimal=5)
    np.testing.assert_almost_equal(spectral_skewness(constF, Fs), 125831896.77783316, decimal=5)
    np.testing.assert_almost_equal(spectral_skewness(lin, Fs), 1.5090650071326563, decimal=5)
    np.testing.assert_almost_equal(spectral_skewness(lin0, Fs), 0.8140329168647044, decimal=5)
    np.testing.assert_almost_equal(spectral_skewness(wave, Fs), 10619659.012776615, decimal=1)
    np.testing.assert_almost_equal(spectral_skewness(offsetWave, Fs), 1.5000000137542306, decimal=1)
    np.testing.assert_almost_equal(spectral_skewness(noiseWave, Fs), 0.4126776686583098, decimal=1)


def test_spectral_kurtosis():
    np.testing.assert_almost_equal(spectral_kurtosis(const0, Fs), 0.0, decimal=5)
    np.testing.assert_almost_equal(spectral_kurtosis(const1, Fs), 1.522441238512017e+16, decimal=5)
    np.testing.assert_almost_equal(spectral_kurtosis(constNeg, Fs), 1.522441238512017e+16, decimal=5)
    np.testing.assert_almost_equal(spectral_kurtosis(constF, Fs), 1.7332972436843942e+16, decimal=5)
    np.testing.assert_almost_equal(spectral_kurtosis(lin, Fs), 4.209140226148914, decimal=0)
    np.testing.assert_almost_equal(spectral_kurtosis(lin0, Fs), 2.4060168768515413, decimal=5)
    np.testing.assert_almost_equal(spectral_kurtosis(wave, Fs),  120474686747609.03, decimal=1)
    np.testing.assert_almost_equal(spectral_kurtosis(offsetWave, Fs), 3.2500028252333513, decimal=5)
    np.testing.assert_almost_equal(spectral_kurtosis(noiseWave, Fs), 1.7251592171239667, decimal=5)


def test_spectral_slope():
    np.testing.assert_equal(spectral_slope(const0, Fs), 0.0)
    np.testing.assert_equal(spectral_slope(const1, Fs), -0.0009818181818181818)
    np.testing.assert_equal(spectral_slope(constNeg, Fs), -0.0009818181818181818)
    np.testing.assert_equal(spectral_slope(constF, Fs), -0.0009818181818181816)
    np.testing.assert_equal(spectral_slope(lin, Fs), -0.0006056882550328839)
    np.testing.assert_almost_equal(spectral_slope(lin0, Fs), -0.00023672490168659717, decimal=1)
    np.testing.assert_almost_equal(spectral_slope(wave, Fs), -2.3425149700598465e-05, decimal=5)
    np.testing.assert_almost_equal(spectral_slope(offsetWave, Fs), -2.380838323353288e-05, decimal=5)
    np.testing.assert_almost_equal(spectral_slope(noiseWave, Fs), -6.586047565550932e-06, decimal=5)


def test_spectral_decrease():
    np.testing.assert_equal(spectral_decrease(const0, Fs), 0.0)
    np.testing.assert_equal(spectral_decrease(const1, Fs), -2.397654526981306e+16)
    np.testing.assert_equal(spectral_decrease(constNeg, Fs), -2.397654526981306e+16)
    np.testing.assert_equal(spectral_decrease(constF, Fs), -2.66984481893762e+16)
    np.testing.assert_equal(spectral_decrease(lin, Fs), -2.255518236004341)
    np.testing.assert_almost_equal(spectral_decrease(lin0, Fs), 0.5195484076294969, decimal=5)
    np.testing.assert_almost_equal(spectral_decrease(wave, Fs), 0.19999999999999687, decimal=5)
    np.testing.assert_almost_equal(spectral_decrease(offsetWave, Fs), -26.963293719961584, decimal=5)
    np.testing.assert_almost_equal(spectral_decrease(noiseWave, Fs), 0.06053938231990085, decimal=5)


def test_spectral_roll_on():
    np.testing.assert_equal(spectral_roll_on(const0, Fs), 0.0)
    np.testing.assert_equal(spectral_roll_on(const1, Fs), 0.0)
    np.testing.assert_equal(spectral_roll_on(constNeg, Fs), 0.0)
    np.testing.assert_equal(spectral_roll_on(constF, Fs), 0.0)
    np.testing.assert_equal(spectral_roll_on(lin, Fs), 0.0)
    np.testing.assert_almost_equal(spectral_roll_on(lin0, Fs), 55.55555555555556, decimal=5)
    np.testing.assert_almost_equal(spectral_roll_on(wave, Fs), 5.0100200400801596, decimal=5)
    np.testing.assert_almost_equal(spectral_roll_on(offsetWave, Fs), 0.0, decimal=5)
    np.testing.assert_almost_equal(spectral_roll_on(noiseWave, Fs), 5.0100200400801596, decimal=5)


def test_spectral_roll_off():
    np.testing.assert_equal(spectral_roll_off(const0, Fs), 0.0)
    np.testing.assert_equal(spectral_roll_off(const1, Fs), 0.0)
    np.testing.assert_equal(spectral_roll_off(constNeg, Fs), 0.0)
    np.testing.assert_equal(spectral_roll_off(constF, Fs), 0.0)
    np.testing.assert_equal(spectral_roll_off(lin, Fs), 444.44444444444446)
    np.testing.assert_almost_equal(spectral_roll_off(lin0, Fs), 500.0, decimal=5)
    np.testing.assert_almost_equal(spectral_roll_off(wave, Fs), 5.0100200400801596, decimal=5)
    np.testing.assert_almost_equal(spectral_roll_off(offsetWave, Fs), 5.0100200400801596, decimal=5)
    np.testing.assert_almost_equal(spectral_roll_off(noiseWave, Fs), 464.9298597194388, decimal=5)


def test_spectral_distance():
    np.testing.assert_equal(spectral_distance(const0, Fs), 0.0)
    np.testing.assert_equal(spectral_distance(const1, Fs), -100)
    np.testing.assert_equal(spectral_distance(constNeg, Fs), -100)
    np.testing.assert_equal(spectral_distance(constF, Fs), -250)
    np.testing.assert_equal(spectral_distance(lin, Fs), -1256.997293357373)
    np.testing.assert_almost_equal(spectral_distance(lin0, Fs), -323.15504563934024, decimal=5)
    np.testing.assert_almost_equal(spectral_distance(wave, Fs), -122500.00000000022, decimal=5)
    np.testing.assert_almost_equal(spectral_distance(offsetWave, Fs), -622500.0, decimal=5)
    np.testing.assert_almost_equal(spectral_distance(noiseWave, Fs), -124832.72310672606, decimal=5)


def test_spect_variation():
    np.testing.assert_equal(spectral_variation(const0, Fs), 1.0)
    np.testing.assert_equal(spectral_variation(const1, Fs), 1.0)
    np.testing.assert_equal(spectral_variation(constNeg, Fs), 1.0)
    np.testing.assert_equal(spectral_variation(constF, Fs), 1.0)
    np.testing.assert_equal(spectral_variation(lin, Fs), 0.04096548417849766)
    np.testing.assert_almost_equal(spectral_variation(lin0, Fs), 0.39913530062615254, decimal=5)
    np.testing.assert_almost_equal(spectral_variation(wave, Fs), 0.9999999999999997, decimal=5)
    np.testing.assert_almost_equal(spectral_variation(offsetWave, Fs), 0.9999999999999999, decimal=5)
    np.testing.assert_almost_equal(spectral_variation(noiseWave, Fs), 0.9775968083533805, decimal=5)


def test_spectral_maxpeaks():
    np.testing.assert_equal(spectral_maxpeaks(const0, Fs), 0.0)
    np.testing.assert_equal(spectral_maxpeaks(const1, Fs), 4.0)
    np.testing.assert_equal(spectral_maxpeaks(constNeg, Fs), 4.0)
    np.testing.assert_equal(spectral_maxpeaks(constF, Fs), 4.0)
    np.testing.assert_equal(spectral_maxpeaks(lin, Fs), 0.0)
    np.testing.assert_almost_equal(spectral_maxpeaks(lin0, Fs), 1.0, decimal=5)
    np.testing.assert_almost_equal(spectral_maxpeaks(wave, Fs), 166, decimal=0)
    np.testing.assert_almost_equal(spectral_maxpeaks(offsetWave, Fs), 157, decimal=1)
    np.testing.assert_almost_equal(spectral_maxpeaks(noiseWave, Fs), 172.0, decimal=1)


def test_human_range_energy():
    np.testing.assert_equal(human_range_energy(const0, Fs), 0.0)
    np.testing.assert_equal(human_range_energy(const1, Fs), 0.0)
    np.testing.assert_equal(human_range_energy(constNeg, Fs), 0.0)
    np.testing.assert_equal(human_range_energy(constF, Fs), 0.0)
    np.testing.assert_equal(human_range_energy(lin, Fs), 0.0)
    np.testing.assert_almost_equal(human_range_energy(lin0, Fs), 0.0)
    np.testing.assert_almost_equal(human_range_energy(wave, Fs), 2.838300923247935e-33)
    np.testing.assert_almost_equal(human_range_energy(offsetWave, Fs), 1.6194431630448383e-33)
    np.testing.assert_almost_equal(human_range_energy(noiseWave, Fs), 4.5026865350839304e-05)


def test_mfcc():
    np.testing.assert_equal(mfcc(const0, Fs), (-1e-08, -2.5654632210061364e-08, -4.099058125255727e-08,
                                               -5.56956514302075e-08, -6.947048992011573e-08, -8.203468073398136e-08,
                                               -9.313245317896842e-08, -1.0253788861142992e-07, -1.1005951948899701e-07,
                                               -1.1554422709759472e-07, -1.1888035860690259e-07,
                                               -1.2000000000000002e-07))
    np.testing.assert_equal(mfcc(const1, Fs), (0.14096637144714785, 0.4029720554090289, 0.2377457745400458,
                                               0.9307791929462678, -0.8138023913445843, -0.36127671623673,
                                               0.17779314470940918, 1.5842014538963525, -5.868875380858009,
                                               -1.3484207382203723, -1.5899059472962034, 2.9774371742123975))
    np.testing.assert_equal(mfcc(constNeg, Fs), (0.14096637144714785, 0.4029720554090289, 0.2377457745400458,
                                                 0.9307791929462678, -0.8138023913445843, -0.36127671623673,
                                                 0.17779314470940918, 1.5842014538963525, -5.868875380858009,
                                                 -1.3484207382203723, -1.5899059472962034, 2.9774371742123975))
    np.testing.assert_equal(mfcc(constF, Fs), (0.1409663714471363, 0.40297205540906766, 0.23774577454002216,
                                               0.9307791929463864, -0.8138023913445535, -0.3612767162368284,
                                               0.17779314470931407, 1.584201453896316, -5.868875380858139,
                                               -1.3484207382203004, -1.589905947296293, 2.977437174212552))
    np.testing.assert_equal(mfcc(lin, Fs), (63.41077963677539, 42.33256774689686, 22.945623346731722,
                                            -9.267967765468333, -30.918618746635172, -69.45624761250505,
                                            -81.74881720705784, -112.32234611356338, -127.73335353282954,
                                            -145.3505024599537, -152.08439229251312, -170.61228411241296))
    np.testing.assert_almost_equal(mfcc(lin0, Fs), (4.472854975902669, 9.303621966161266, 12.815317252229947,
                                                    12.65260020301481, 9.763110307405048, 3.627814979708572,
                                                    1.0051648150842092, -8.07514557618858, -24.79987026383853,
                                                    -36.55749714126207, -49.060094200797785, -61.45654150658956))
    np.testing.assert_almost_equal(mfcc(wave, Fs), (115.31298449242963,-23.978080415791883, 64.49711308839377,
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
    np.testing.assert_equal(power_bandwidth(const0, Fs), 0.0)
    np.testing.assert_equal(power_bandwidth(const1, Fs), 0.0)
    np.testing.assert_equal(power_bandwidth(constNeg, Fs), 0.0)
    np.testing.assert_equal(power_bandwidth(constF, Fs), 0.0)
    np.testing.assert_equal(power_bandwidth(lin, Fs), 0.0)
    np.testing.assert_almost_equal(power_bandwidth(lin0, Fs), 0.0)
    np.testing.assert_almost_equal(power_bandwidth(wave, Fs), 2.0)
    np.testing.assert_almost_equal(power_bandwidth(offsetWave, Fs), 2.0)
    np.testing.assert_almost_equal(power_bandwidth(noiseWave, Fs), 2.0)


def test_fft_mean_coeff():
    np.testing.assert_equal(fft_mean_coeff(const0, Fs, nfreq=10), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    np.testing.assert_equal(fft_mean_coeff(const1, Fs, nfreq=10), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    np.testing.assert_equal(fft_mean_coeff(constNeg, Fs, nfreq=10), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    np.testing.assert_equal(fft_mean_coeff(constF, Fs, nfreq=10), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    np.testing.assert_equal(fft_mean_coeff(lin, Fs, nfreq=10), (0.00408221375370652, 0.29732082717207287,
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
    np.testing.assert_equal(lpcc(const0), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    np.testing.assert_equal(lpcc(const1), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    np.testing.assert_equal(lpcc(constNeg), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    np.testing.assert_equal(lpcc(constF), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    np.testing.assert_equal(lpcc(lin), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    np.testing.assert_almost_equal(lpcc(lin0), (0.017793342850434657, 0.12419699587050197, 0.17985773867565555,
                                                0.13749027713829948, 0.14521059821841656, 0.14362411136332903,
                                                0.14403924127165643, 0.14362411136332903, 0.14521059821841656,
                                                0.13749027713829948, 0.17985773867565555, 0.12419699587050197))
    np.testing.assert_almost_equal(lpcc(wave), (8.08705689884851e-07, 0.10193422882411193, 0.0051922525746904875,
                                                0.0003496693593067946, 2.355214618130234e-05, 1.2419899263690914e-06,
                                                3.091008802744081e-06, 1.2419899263690914e-06, 2.355214618130234e-05,
                                                0.0003496693593067946, 0.0051922525746904875, 0.10193422882411193))
    np.testing.assert_almost_equal(lpcc(offsetWave), (8.087054868870942e-07, 0.10193422882503231, 0.005192252575236202,
                                                      0.0003496693583308415, 2.3552147454092374e-05,
                                                      1.241991615337501e-06, 3.0910069449505212e-06,
                                                      1.241991615337501e-06, 2.3552147454092374e-05,
                                                      0.0003496693583308415, 0.005192252575236202, 0.10193422882503231))
    np.testing.assert_almost_equal(lpcc(noiseWave), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))


run_module_suite()
