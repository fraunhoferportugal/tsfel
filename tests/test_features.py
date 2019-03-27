from numpy.testing import assert_array_equal, run_module_suite
import numpy as np
from tsfel import *


const0 = np.zeros(20)
const1 = np.ones(20)
constNeg = np.ones(20)*(-1)
constF = np.ones(20) * 2.5
lin = np.arange(20)
lin0 = np.linspace(-10,10,20)
f = 5
sample = 1000
x = np.arange(0, sample, 1)
Fs = 1000
wave = np.sin(2 * np.pi * f * x / Fs)
np.random.seed(seed=10)
noiseWave = wave + np.random.normal(0,0.1,1000)
offsetWave = wave + 2


#### STATISTICAL FEATURES ####
def test_skew():
    np.testing.assert_equal(skew(const0), 0.0)
    np.testing.assert_equal(skew(const1), 0.0)
    np.testing.assert_equal(skew(constNeg), 0.0)
    np.testing.assert_equal(skew(constF), 0.0)
    np.testing.assert_equal(skew(lin), 0)
    np.testing.assert_almost_equal(skew(lin0), -1.0167718723297815e-16, decimal=5)
    np.testing.assert_almost_equal(skew(wave), -2.009718347115232e-17, decimal=5)
    np.testing.assert_almost_equal(skew(offsetWave), 9.043732562018544e-16, decimal=5)
    np.testing.assert_almost_equal(skew(noiseWave), -0.0004854111290521465, decimal=5)


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
    np.testing.assert_equal(np.mean(const0), 0.0)
    np.testing.assert_equal(np.mean(const1), 1.0)
    np.testing.assert_equal(np.mean(constNeg), -1.0)
    np.testing.assert_equal(np.mean(constF), 2.5)
    np.testing.assert_equal(np.mean(lin), 9.5)
    np.testing.assert_almost_equal(np.mean(lin0), -3.552713678800501e-16, decimal=5)
    np.testing.assert_almost_equal(np.mean(wave), 7.105427357601002e-18, decimal=5)
    np.testing.assert_almost_equal(np.mean(offsetWave), 2.0, decimal=5)
    np.testing.assert_almost_equal(np.mean(noiseWave), -0.0014556635615470554, decimal=5)


def test_median():
    np.testing.assert_equal(np.median(const0), 0.0)
    np.testing.assert_equal(np.median(const1), 1.0)
    np.testing.assert_equal(np.median(constNeg), -1.0)
    np.testing.assert_equal(np.median(constF), 2.5)
    np.testing.assert_equal(np.median(lin), 9.5)
    np.testing.assert_almost_equal(np.median(lin0), -3.552713678800501e-16, decimal=5)
    np.testing.assert_almost_equal(np.median(wave), 7.105427357601002e-18, decimal=5)
    np.testing.assert_almost_equal(np.median(offsetWave), 2.0, decimal=5)
    np.testing.assert_almost_equal(np.median(noiseWave), 0.013846093997438328, decimal=5)


def test_hist():
    np.testing.assert_equal(hist(const0, 10, 5), (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0))
    np.testing.assert_equal(hist(const1, 10, 5), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0))
    np.testing.assert_equal(hist(constNeg, 10, 5), (0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    np.testing.assert_equal(hist(constF, 10, 5),  (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0))
    np.testing.assert_equal(hist(lin, 10, 5), (0.0, 0.0, 0.0, 0.0, 0.0, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.3333333333333333))
    np.testing.assert_almost_equal(hist(wave, 10, 5), (0.   , 0.   , 0.   , 0.   , 0.499, 0.496, 0.005, 0.   , 0.   ,
       0.   ), decimal=5)
    np.testing.assert_almost_equal(hist(offsetWave, 10, 5), (0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.499, 0.496, 0.005,
       0.  ), decimal=5)
    np.testing.assert_almost_equal(hist(noiseWave, 10, 5), (0.   , 0.   , 0.   , 0.048, 0.446, 0.45 , 0.056, 0.   , 0.   ,
       0.   ), decimal=5)


def test_max():
    np.testing.assert_equal(np.max(const0), 0.0)
    np.testing.assert_equal(np.max(const1), 1.0)
    np.testing.assert_equal(np.max(constNeg), -1.0)
    np.testing.assert_equal(np.max(constF), 2.5)
    np.testing.assert_equal(np.max(lin), 19)
    np.testing.assert_almost_equal(np.max(lin0), 10.0, decimal=5)
    np.testing.assert_almost_equal(np.max(wave), 1.0, decimal=5)
    np.testing.assert_almost_equal(np.max(noiseWave), 1.221757617217142, decimal=5)
    np.testing.assert_almost_equal(np.max(offsetWave), 3.0, decimal=5)


def test_min():
    np.testing.assert_equal(np.min(const0), 0.0)
    np.testing.assert_equal(np.min(const1), 1.0)
    np.testing.assert_equal(np.min(constNeg), -1.0)
    np.testing.assert_equal(np.min(constF), 2.5)
    np.testing.assert_equal(np.min(lin), 0)
    np.testing.assert_almost_equal(np.min(lin0), -10.0, decimal=5)
    np.testing.assert_almost_equal(np.min(wave), -1.0, decimal=5)
    np.testing.assert_almost_equal(np.min(noiseWave), -1.2582533627830566, decimal=5)
    np.testing.assert_almost_equal(np.min(offsetWave), 1.0, decimal=5)


def test_variance():
    np.testing.assert_equal(np.var(const0), 0.0)
    np.testing.assert_equal(np.var(const1), 0.0)
    np.testing.assert_equal(np.var(constNeg), 0.0)
    np.testing.assert_equal(np.var(constF), 0.0)
    np.testing.assert_equal(np.var(lin), 33.25)
    np.testing.assert_almost_equal(np.var(lin0), 36.84210526315789, decimal=5)
    np.testing.assert_almost_equal(np.var(wave), 0.5, decimal=5)
    np.testing.assert_almost_equal(np.var(offsetWave), 0.5, decimal=5)
    np.testing.assert_almost_equal(np.var(noiseWave), 0.5081167177369529, decimal=5)


def test_std():
    np.testing.assert_equal(np.std(const0), 0.0)
    np.testing.assert_equal(np.std(const1), 0.0)
    np.testing.assert_equal(np.std(constNeg), 0.0)
    np.testing.assert_equal(np.std(constF), 0.0)
    np.testing.assert_equal(np.std(lin), 5.766281297335398)
    np.testing.assert_almost_equal(np.std(lin0), 6.069769786668839, decimal=5)
    np.testing.assert_almost_equal(np.std(wave), 0.7071067811865476, decimal=5)
    np.testing.assert_almost_equal(np.std(offsetWave), 0.7071067811865476, decimal=5)
    np.testing.assert_almost_equal(np.std(noiseWave), 0.7128230620125536, decimal=5)


def test_int_range():
    np.testing.assert_equal(interq_range(const0), 0.0)
    np.testing.assert_equal(interq_range(const1), 0.0)
    np.testing.assert_equal(interq_range(constNeg), 0.0)
    np.testing.assert_equal(interq_range(constF), 0.0)
    np.testing.assert_equal(interq_range(lin), 9.5)
    np.testing.assert_almost_equal(interq_range(lin0), 10.0, decimal=5)
    np.testing.assert_almost_equal(interq_range(wave), 1.414213562373095, decimal=5)
    np.testing.assert_almost_equal(interq_range(offsetWave), 1.414213562373095, decimal=5)
    np.testing.assert_almost_equal(interq_range(noiseWave), 1.4277110228590328, decimal=5)


def test_calc_meanad():
    np.testing.assert_equal(calc_meanad(const0), 0.0)
    np.testing.assert_equal(calc_meanad(const1), 0.0)
    np.testing.assert_equal(calc_meanad(constNeg), 0.0)
    np.testing.assert_equal(calc_meanad(constF), 0.0)
    np.testing.assert_equal(calc_meanad(lin), 5.0)
    np.testing.assert_almost_equal(calc_meanad(lin0), 5.263157894736842, decimal=5)
    np.testing.assert_almost_equal(calc_meanad(wave), 0.6365674116287159, decimal=5)
    np.testing.assert_almost_equal(calc_meanad(noiseWave), 0.6392749078483896, decimal=5)
    np.testing.assert_almost_equal(calc_meanad(offsetWave), 0.6365674116287157, decimal=5)


def test_calc_medad():
    np.testing.assert_equal(calc_medad(const0), 0.0)
    np.testing.assert_equal(calc_medad(const1), 0.0)
    np.testing.assert_equal(calc_medad(constNeg), 0.0)
    np.testing.assert_equal(calc_medad(constF), 0.0)
    np.testing.assert_equal(calc_medad(lin), 5.0)
    np.testing.assert_almost_equal(calc_medad(lin0), 5.2631578947368425, decimal=5)
    np.testing.assert_almost_equal(calc_medad(wave), 0.7071067811865475, decimal=5)
    np.testing.assert_almost_equal(calc_medad(offsetWave), 0.7071067811865475, decimal=5)
    np.testing.assert_almost_equal(calc_medad(noiseWave), 0.7068117164205888, decimal=5)


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


#### TEMPORAL FEATURES ####
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


def test_centroid():
    np.testing.assert_equal(centroid(const0, Fs), 0.0)
    np.testing.assert_equal(centroid(const1, Fs), 0.009)
    np.testing.assert_equal(centroid(constNeg, Fs), 0.009)
    np.testing.assert_equal(centroid(constF, Fs), 0.009)
    np.testing.assert_equal(centroid(lin, Fs), 0.013571428571428573)
    np.testing.assert_almost_equal(centroid(lin0, Fs), 0.00708955223880597, decimal=5)
    np.testing.assert_almost_equal(centroid(wave, Fs), 0.45, decimal=5)
    np.testing.assert_almost_equal(centroid(offsetWave, Fs), 0.44999999999999996, decimal=5)
    np.testing.assert_almost_equal(centroid(noiseWave, Fs), 0.4494737253547235, decimal=5)


def test_calc_meandiff():
    np.testing.assert_equal(calc_meandiff(const0), 0.0)
    np.testing.assert_equal(calc_meandiff(const1), 0.0)
    np.testing.assert_equal(calc_meandiff(constNeg), 0.0)
    np.testing.assert_equal(calc_meandiff(constF), 0.0)
    np.testing.assert_equal(calc_meandiff(lin), 1.0)
    np.testing.assert_almost_equal(calc_meandiff(lin0), 1.0526315789473684, decimal=5)
    np.testing.assert_almost_equal(calc_meandiff(wave), -3.1442201279407477e-05, decimal=5)
    np.testing.assert_almost_equal(calc_meandiff(offsetWave), -3.1442201279407036e-05, decimal=5)
    np.testing.assert_almost_equal(calc_meandiff(noiseWave), -0.00010042477181949707, decimal=5)


def test_calc_meddiff():
    np.testing.assert_equal(calc_meddiff(const0), 0.0)
    np.testing.assert_equal(calc_meddiff(const1), 0.0)
    np.testing.assert_equal(calc_meddiff(constNeg), 0.0)
    np.testing.assert_equal(calc_meddiff(constF), 0.0)
    np.testing.assert_equal(calc_meddiff(lin), 1.0)
    np.testing.assert_almost_equal(calc_meddiff(lin0), 1.0526315789473684, decimal=5)
    np.testing.assert_almost_equal(calc_meddiff(wave), -0.0004934396342684, decimal=5)
    np.testing.assert_almost_equal(calc_meddiff(offsetWave), -0.0004934396342681779, decimal=5)
    np.testing.assert_almost_equal(calc_meddiff(noiseWave), -0.004174819648320949, decimal=5)


def test_calc_meanadiff():
    np.testing.assert_equal(calc_meanadiff(const0), 0.0)
    np.testing.assert_equal(calc_meanadiff(const1), 0.0)
    np.testing.assert_equal(calc_meanadiff(constNeg), 0.0)
    np.testing.assert_equal(calc_meanadiff(constF), 0.0)
    np.testing.assert_equal(calc_meanadiff(lin), 1.0)
    np.testing.assert_almost_equal(calc_meanadiff(lin0), 1.0526315789473684, decimal=5)
    np.testing.assert_almost_equal(calc_meanadiff(wave), 0.019988577818740614, decimal=5)
    np.testing.assert_almost_equal(calc_meanadiff(offsetWave), 0.019988577818740614, decimal=5)
    np.testing.assert_almost_equal(calc_meanadiff(noiseWave), 0.10700252903161508, decimal=5)


def test_calc_medadiff():
    np.testing.assert_equal(calc_medadiff(const0), 0.0)
    np.testing.assert_equal(calc_medadiff(const1), 0.0)
    np.testing.assert_equal(calc_medadiff(constNeg), 0.0)
    np.testing.assert_equal(calc_medadiff(constF), 0.0)
    np.testing.assert_equal(calc_medadiff(lin), 1.0)
    np.testing.assert_almost_equal(calc_medadiff(lin0), 1.0526315789473681, decimal=5)
    np.testing.assert_almost_equal(calc_medadiff(wave), 0.0218618462348652, decimal=5)
    np.testing.assert_almost_equal(calc_medadiff(offsetWave), 0.021861846234865645, decimal=5)
    np.testing.assert_almost_equal(calc_medadiff(noiseWave), 0.08958750592592835, decimal=5)


def test_calc_sadiff():
    np.testing.assert_equal(calc_sadiff(const0), 0.0)
    np.testing.assert_equal(calc_sadiff(const1), 0.0)
    np.testing.assert_equal(calc_sadiff(constNeg), 0.0)
    np.testing.assert_equal(calc_sadiff(constF), 0.0)
    np.testing.assert_equal(calc_sadiff(lin), 19)
    np.testing.assert_almost_equal(calc_sadiff(lin0), 20.0, decimal=5)
    np.testing.assert_almost_equal(calc_sadiff(wave), 19.968589240921872, decimal=5)
    np.testing.assert_almost_equal(calc_sadiff(offsetWave), 19.968589240921872, decimal=5)
    np.testing.assert_almost_equal(calc_sadiff(noiseWave), 106.89552650258346, decimal=5)


def test_zeroCross():
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


#### TEMPORAL FEATURES ####
def test_max_fre():
    np.testing.assert_equal(max_frequency(const0, Fs), 55.55555555555556)
    np.testing.assert_equal(max_frequency(const1, Fs), 444.44444444444446)
    np.testing.assert_equal(max_frequency(constNeg, Fs), 444.44444444444446)
    np.testing.assert_equal(max_frequency(constF, Fs), 444.44444444444446)
    np.testing.assert_equal(max_frequency(lin, Fs), 500.0)
    np.testing.assert_almost_equal(max_frequency(lin0, Fs), 500.0, decimal=5)
    np.testing.assert_almost_equal(max_frequency(wave, Fs), 5.0100200400801596, decimal=5)
    np.testing.assert_almost_equal(max_frequency(offsetWave, Fs), 5.0100200400801596, decimal=5)
    np.testing.assert_almost_equal(max_frequency(noiseWave, Fs), 464.9298597194388, decimal=5)
    np.testing.assert_almost_equal(max_frequency(x, Fs),  390.7815631262525, decimal=1)


def test_med_fre():
    np.testing.assert_equal(median_frequency(const0, Fs), 55.55555555555556)
    np.testing.assert_equal(median_frequency(const1, Fs), 222.22222222222223)
    np.testing.assert_equal(median_frequency(constNeg, Fs), 222.22222222222223)
    np.testing.assert_equal(median_frequency(constF, Fs), 222.22222222222223)
    np.testing.assert_equal(median_frequency(lin, Fs), 166.66666666666669)
    np.testing.assert_almost_equal(median_frequency(lin0, Fs), 166.66666666666669, decimal=5)
    np.testing.assert_almost_equal(median_frequency(wave, Fs), 5.0100200400801596, decimal=5)
    np.testing.assert_almost_equal(median_frequency(offsetWave, Fs), 5.0100200400801596, decimal=5)
    np.testing.assert_almost_equal(median_frequency(noiseWave, Fs), 146.29258517034066, decimal=5)
    np.testing.assert_almost_equal(median_frequency(x, Fs), 19.03807615230461, decimal=1)


def test_fund_fre():
    np.testing.assert_equal(fundamental_frequency(const0, 1), 0.0)
    np.testing.assert_equal(fundamental_frequency(const1, 1), 0.0)
    np.testing.assert_equal(fundamental_frequency(constNeg, Fs), 0.0)
    np.testing.assert_equal(fundamental_frequency(constF, Fs), 0.0)
    np.testing.assert_equal(fundamental_frequency(lin, Fs), 0.0)
    np.testing.assert_almost_equal(fundamental_frequency(lin0, Fs), 0.0, decimal=5)
    np.testing.assert_almost_equal(fundamental_frequency(wave, Fs), 4.008016032064128, decimal=1)
    np.testing.assert_almost_equal(fundamental_frequency(offsetWave, Fs), 3.006012024048096, decimal=1)
    np.testing.assert_almost_equal(fundamental_frequency(noiseWave, Fs), 39.07815631262525, decimal=1)


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
    np.testing.assert_equal(spectral_centroid(const1, Fs), 232.87645373386948)
    np.testing.assert_equal(spectral_centroid(constNeg, Fs), 232.87645373386948)
    np.testing.assert_equal(spectral_centroid(constF, Fs), 231.25899007425215)
    np.testing.assert_equal(spectral_centroid(lin, Fs), 189.72282595943128)
    np.testing.assert_almost_equal(spectral_centroid(lin0, Fs), 189.7228259594313, decimal=5)
    np.testing.assert_almost_equal(spectral_centroid(wave, Fs), 5.010020040084022, decimal=5)
    np.testing.assert_almost_equal(spectral_centroid(offsetWave, Fs), 5.010020040084512, decimal=5)
    np.testing.assert_almost_equal(spectral_centroid(noiseWave, Fs), 181.26672828206048, decimal=5)


def test_spectral_spread():
    np.testing.assert_almost_equal(spectral_spread(const0, Fs), 0.0, decimal=5)
    np.testing.assert_almost_equal(spectral_spread(const1, Fs), 12780.633762829042, decimal=5)
    np.testing.assert_almost_equal(spectral_spread(constNeg, Fs), 12780.633762829042, decimal=5)
    np.testing.assert_almost_equal(spectral_spread(constF, Fs), 13180.136854206703, decimal=5)
    np.testing.assert_almost_equal(spectral_spread(lin, Fs), 19861.962160017465, decimal=5)
    np.testing.assert_almost_equal(spectral_spread(lin0, Fs), 19861.962160017472, decimal=5)
    np.testing.assert_almost_equal(spectral_spread(wave, Fs), 0.0, decimal=5)
    np.testing.assert_almost_equal(spectral_spread(offsetWave, Fs), 1.4368545161461419e-09, decimal=5)
    np.testing.assert_almost_equal(spectral_spread(noiseWave, Fs), 27432.31810321154, decimal=5)


def test_spectral_skewness():
    np.testing.assert_almost_equal(spectral_skewness(const0, Fs), 0.0, decimal=5)
    np.testing.assert_almost_equal(spectral_skewness(const1, Fs), 0.47000214356667996, decimal=5)
    np.testing.assert_almost_equal(spectral_skewness(constNeg, Fs), 0.47000214356667996, decimal=5)
    np.testing.assert_almost_equal(spectral_skewness(constF, Fs), 0.5161486069831996, decimal=5)
    np.testing.assert_almost_equal(spectral_skewness(lin, Fs), 0.8140329168647059, decimal=5)
    np.testing.assert_almost_equal(spectral_skewness(lin0, Fs), 0.8140329168647046, decimal=5)
    np.testing.assert_almost_equal(spectral_skewness(wave, Fs), 10624606.203489874, decimal=1)
    np.testing.assert_almost_equal(spectral_skewness(offsetWave, Fs), 9940161.672715995, decimal=1)
    np.testing.assert_almost_equal(spectral_skewness(noiseWave, Fs), 0.41151531285467985, decimal=1)


def test_spectral_kurtosis():
    np.testing.assert_almost_equal(spectral_kurtosis(const0, Fs), 0.0, decimal=5)
    np.testing.assert_almost_equal(spectral_kurtosis(const1, Fs), 2.046321609772459, decimal=5)
    np.testing.assert_almost_equal(spectral_kurtosis(constNeg, Fs), 2.046321609772459, decimal=5)
    np.testing.assert_almost_equal(spectral_kurtosis(constF, Fs), 2.0107996983500787, decimal=5)
    np.testing.assert_almost_equal(spectral_kurtosis(lin, Fs), 2.0107996983500787, decimal=0)
    np.testing.assert_almost_equal(spectral_kurtosis(lin0, Fs), 2.4060168768515413, decimal=5)
    np.testing.assert_almost_equal(spectral_kurtosis(wave, Fs),  120631070963747.72, decimal=1)
    np.testing.assert_almost_equal(spectral_kurtosis(offsetWave, Fs), 105458851126694.56, decimal=5)
    np.testing.assert_almost_equal(spectral_kurtosis(noiseWave, Fs), 1.7244872082677942, decimal=5)


def test_spectral_slope():
    np.testing.assert_equal(spectral_slope(const0, Fs), 0.0)
    np.testing.assert_equal(spectral_slope(const1, Fs), -5.721690608887972e-19)
    np.testing.assert_equal(spectral_slope(constNeg, Fs), -5.721690608887972e-19)
    np.testing.assert_equal(spectral_slope(constF, Fs), -1.330865622866306e-18)
    np.testing.assert_equal(spectral_slope(lin, Fs), -0.09209918800721191)
    np.testing.assert_almost_equal(spectral_slope(lin0, Fs), -0.09209918800721191, decimal=1)
    np.testing.assert_almost_equal(spectral_slope(wave, Fs), -0.011807228915662668, decimal=5)
    np.testing.assert_almost_equal(spectral_slope(offsetWave, Fs), -0.011807228915662668, decimal=5)
    np.testing.assert_almost_equal(spectral_slope(noiseWave, Fs), -0.011996993466635138, decimal=5)


def test_spectral_decrease():
    np.testing.assert_equal(spectral_decrease(const0, Fs), 0.0)
    np.testing.assert_equal(spectral_decrease(const1, Fs), -1.092141531796704)
    np.testing.assert_equal(spectral_decrease(constNeg, Fs), -1.092141531796704)
    np.testing.assert_equal(spectral_decrease(constF, Fs), -1.0677129498080251)
    np.testing.assert_equal(spectral_decrease(lin, Fs), -0.41194423707863714)
    np.testing.assert_almost_equal(spectral_decrease(lin0, Fs), -0.4119442370786371, decimal=5)
    np.testing.assert_almost_equal(spectral_decrease(wave, Fs), 0.333333333333328, decimal=5)
    np.testing.assert_almost_equal(spectral_decrease(offsetWave, Fs), 0.3333333333333272, decimal=5)
    np.testing.assert_almost_equal(spectral_decrease(noiseWave, Fs), 0.09171304789096776, decimal=5)


def test_spectral_roll_on():
    np.testing.assert_equal(spectral_roll_on(const0, Fs), 55.55555555555556)
    np.testing.assert_equal(spectral_roll_on(const1, Fs), 111.11111111111111)
    np.testing.assert_equal(spectral_roll_on(constNeg, Fs), 111.11111111111111)
    np.testing.assert_equal(spectral_roll_on(constF, Fs), 111.11111111111111)
    np.testing.assert_equal(spectral_roll_on(lin, Fs), 55.55555555555556)
    np.testing.assert_almost_equal(spectral_roll_on(lin0, Fs), 55.55555555555556, decimal=5)
    np.testing.assert_almost_equal(spectral_roll_on(wave, Fs), 5.0100200400801596, decimal=5)
    np.testing.assert_almost_equal(spectral_roll_on(offsetWave, Fs), 5.0100200400801596, decimal=5)
    np.testing.assert_almost_equal(spectral_roll_on(noiseWave, Fs), 5.0100200400801596, decimal=5)


def test_spectral_roll_off():
    np.testing.assert_equal(spectral_roll_off(const0, Fs), 55.55555555555556)
    np.testing.assert_equal(spectral_roll_off(const1, Fs), 444.44444444444446)
    np.testing.assert_equal(spectral_roll_off(constNeg, Fs), 444.44444444444446)
    np.testing.assert_equal(spectral_roll_off(constF, Fs), 444.44444444444446)
    np.testing.assert_equal(spectral_roll_off(lin, Fs), 500.0)
    np.testing.assert_almost_equal(spectral_roll_off(lin0, Fs), 500.0, decimal=5)
    np.testing.assert_almost_equal(spectral_roll_off(wave, Fs), 5.0100200400801596, decimal=5)
    np.testing.assert_almost_equal(spectral_roll_off(offsetWave, Fs), 5.0100200400801596, decimal=5)
    np.testing.assert_almost_equal(spectral_roll_off(noiseWave, Fs), 464.9298597194388, decimal=5)


def test_spectral_curve_distance():
    np.testing.assert_equal(curve_distance(const0, Fs), 0.0)
    np.testing.assert_equal(curve_distance(const1, Fs), -3.0871200587084623e-15)
    np.testing.assert_equal(curve_distance(constNeg, Fs), -3.0871200587084623e-15)
    np.testing.assert_equal(curve_distance(constF, Fs), -7.0852193832076304e-15)
    np.testing.assert_equal(curve_distance(lin, Fs), -403.8425293964846)
    np.testing.assert_almost_equal(curve_distance(lin0, Fs), -425.0973993647206, decimal=5)
    np.testing.assert_almost_equal(curve_distance(wave, Fs), -122750.0000000002, decimal=5)
    np.testing.assert_almost_equal(curve_distance(offsetWave, Fs), -122750.0000000002, decimal=5)
    np.testing.assert_almost_equal(curve_distance(noiseWave, Fs), -125369.50449385124, decimal=5)


def test_spect_variation():
    np.testing.assert_equal(spect_variation(const0, Fs), 1.0)
    np.testing.assert_equal(spect_variation(const1, Fs), 1.0)
    np.testing.assert_equal(spect_variation(constNeg, Fs), 1.0)
    np.testing.assert_equal(spect_variation(constF, Fs), 0.9872320416232951)
    np.testing.assert_equal(spect_variation(lin, Fs), 0.7225499818571235)
    np.testing.assert_almost_equal(spect_variation(lin0, Fs), 0.7225499818571237, decimal=5)
    np.testing.assert_almost_equal(spect_variation(wave, Fs), 1.0, decimal=5)
    np.testing.assert_almost_equal(spect_variation(offsetWave, Fs), 1.0, decimal=5)
    np.testing.assert_almost_equal(spect_variation(noiseWave, Fs), 0.9961527995674907, decimal=5)


def test_spectral_maxpeaks():
    np.testing.assert_equal(spectral_maxpeaks(const0, Fs), 0.0)
    np.testing.assert_equal(spectral_maxpeaks(const1, Fs), 4.0)
    np.testing.assert_equal(spectral_maxpeaks(constNeg, Fs), 4.0)
    np.testing.assert_equal(spectral_maxpeaks(constF, Fs), 4.0)
    np.testing.assert_equal(spectral_maxpeaks(lin, Fs), 0.0)
    np.testing.assert_almost_equal(spectral_maxpeaks(lin0, Fs), 0.0, decimal=5)
    np.testing.assert_almost_equal(spectral_maxpeaks(wave, Fs), 163, decimal=0)
    np.testing.assert_almost_equal(spectral_maxpeaks(offsetWave, Fs), 160, decimal=1)
    np.testing.assert_almost_equal(spectral_maxpeaks(noiseWave, Fs), 171.0, decimal=1)

run_module_suite()
