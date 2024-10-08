import warnings

import scipy.signal
from statsmodels.tsa.stattools import acf

from tsfel.constants import FEATURES_MIN_SIZE
from tsfel.feature_extraction.features_utils import *

warning_flag = False
warning_msg = (
    "The fractal features will not be calculated and will be replaced with 'nan' because the length of the input signal is smaller than the required minimum of "
    + str(FEATURES_MIN_SIZE)
    + " data points."
)

# ############################################# TEMPORAL DOMAIN ##################################################### #


@set_domain("domain", "temporal")
def autocorr(signal):
    """Calculates the first 1/e crossing of the autocorrelation function (ACF).
    The adjusted ACF is calculated using the `statsmodels.tsa.stattools.acf`.
    Following the recommendations for long time series (size > 450), we use the
    FFT convolution. This feature measures the first time lag at which the
    autocorrelation function drops below 1/e (= 0.3679).

    Feature computational cost: 2

    Parameters
    ----------
    signal : nd-array
        Input from which autocorrelation is computed

    Returns
    -------
    int
        The first time lag at which the ACF drops below 1/e (= 0.3679).
    """
    n = len(signal)
    threshold = 0.36787944117144233  # 1 / np.exp(1)

    # For constant input signals, the ACF remains constant, and the expected values for all lags other than
    # lag 0 will be zero. We standardize that (1/e) occurs at lag 1.
    if np.all(signal == signal[0]):
        return 1

    a = acf(signal, adjusted=True, fft=n > 450, nlags=(int(n / 3)))[1:]
    indices = np.where(a < threshold)[0]
    first1e_acf = indices[0] + 1 if indices.size > 0 else None

    return first1e_acf


@set_domain("domain", "temporal")
def calc_centroid(signal, fs):
    """Computes the centroid along the time axis.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which centroid is computed
    fs: int
        Signal sampling frequency

    Returns
    -------
    float
        Temporal centroid
    """

    time = compute_time(signal, fs)

    energy = np.array(signal) ** 2

    t_energy = np.dot(np.array(time), np.array(energy))
    energy_sum = np.sum(energy)

    if energy_sum == 0 or t_energy == 0:
        centroid = 0
    else:
        centroid = t_energy / energy_sum

    return centroid


@set_domain("domain", "temporal")
@set_domain("tag", "emg")
def negative_turning(signal):
    """Computes number of negative turning points of the signal.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which minimum number of negative turning points are counted
    Returns
    -------
    float
        Number of negative turning points
    """
    diff_sig = np.diff(signal)
    array_signal = np.arange(len(diff_sig[:-1]))
    negative_turning_pts = np.where((diff_sig[array_signal] < 0) & (diff_sig[array_signal + 1] > 0))[0]

    return len(negative_turning_pts)


@set_domain("domain", "temporal")
@set_domain("tag", "emg")
def positive_turning(signal):
    """Computes number of positive turning points of the signal.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which  positive turning points are counted

    Returns
    -------
    float
        Number of positive turning points
    """
    diff_sig = np.diff(signal)

    array_signal = np.arange(len(diff_sig[:-1]))

    positive_turning_pts = np.where((diff_sig[array_signal + 1] < 0) & (diff_sig[array_signal] > 0))[0]

    return len(positive_turning_pts)


@set_domain("domain", "temporal")
def mean_abs_diff(signal):
    """Computes mean absolute differences of the signal.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which mean absolute deviation is computed

    Returns
    -------
    float
        Mean absolute difference result
    """
    return np.mean(np.abs(np.diff(signal)))


@set_domain("domain", "temporal")
def mean_diff(signal):
    """Computes mean of differences of the signal.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which mean of differences is computed

    Returns
    -------
    float
        Mean difference result
    """
    return np.mean(np.diff(signal))


@set_domain("domain", "temporal")
def median_abs_diff(signal):
    """Computes median absolute differences of the signal.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which median absolute difference is computed

    Returns
    -------
    float
        Median absolute difference result
    """
    return np.median(np.abs(np.diff(signal)))


@set_domain("domain", "temporal")
def median_diff(signal):
    """Computes median of differences of the signal.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which median of differences is computed

    Returns
    -------
    float
        Median difference result
    """
    return np.median(np.diff(signal))


@set_domain("domain", "temporal")
def distance(signal):
    """Computes signal traveled distance.

    Calculates the total distance traveled by the signal
    using the hypotenuse between 2 datapoints.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which distance is computed

    Returns
    -------
    float
        Signal distance
    """
    diff_sig = np.diff(signal).astype(float)
    return np.sum([np.sqrt(1 + diff_sig**2)])


@set_domain("domain", "temporal")
def sum_abs_diff(signal):
    """Computes sum of absolute differences of the signal.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which sum absolute difference is computed

    Returns
    -------
    float
        Sum absolute difference result
    """
    return np.sum(np.abs(np.diff(signal)))


@set_domain("domain", "temporal")
@set_domain("tag", ["audio", "emg"])
def zero_cross(signal):
    """Computes Zero-crossing rate of the signal.

    Corresponds to the total number of times that the signal changes from
    positive to negative or vice versa.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which the zero-crossing rate are computed

    Returns
    -------
    int
        Number of times that signal value cross the zero axis
    """
    return len(np.where(np.diff(np.sign(signal)))[0])


@set_domain("domain", "temporal")
def slope(signal):
    """Computes the slope of the signal.

    Slope is computed by fitting a linear equation to the observed data.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which linear equation is computed

    Returns
    -------
    float
        Slope
    """
    t = np.linspace(0, len(signal) - 1, len(signal))

    return np.polyfit(t, signal, 1)[0]


@set_domain("domain", "temporal")
def auc(signal, fs):
    """Computes the area under the curve of the signal computed with trapezoid
    rule.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which the area under the curve is computed
    fs : float
        Sampling Frequency
    Returns
    -------
    float
        The area under the curve value
    """
    t = compute_time(signal, fs)

    return np.sum(0.5 * np.diff(t) * np.abs(np.array(signal[:-1]) + np.array(signal[1:])))


@set_domain("domain", "temporal")
def neighbourhood_peaks(signal, n=10):
    """Computes the number of peaks from a defined neighbourhood of the signal.

    Reference: Christ, M., Braun, N., Neuffer, J. and Kempa-Liehr A.W. (2018). Time Series FeatuRe Extraction on basis
     of Scalable Hypothesis tests (tsfresh -- A Python package). Neurocomputing 307 (2018) 72-77

    Parameters
    ----------
    signal : nd-array
         Input from which the number of neighbourhood peaks is computed
    n :  int
        Number of peak's neighbours to the left and to the right

    Returns
    -------
    int
        The number of peaks from a defined neighbourhood of the signal
    """
    signal = np.array(signal)
    subsequence = signal[n:-n]
    # initial iteration
    peaks = (subsequence > np.roll(signal, 1)[n:-n]) & (subsequence > np.roll(signal, -1)[n:-n])
    for i in np.arange(2, n + 1):
        peaks &= subsequence > np.roll(signal, i)[n:-n]
        peaks &= subsequence > np.roll(signal, -i)[n:-n]
    return np.sum(peaks)


@set_domain("domain", "temporal")
def lempel_ziv(signal, threshold=None):
    """Computes the Lempel-Ziv's (LZ) complexity index, normalized by the
    signal's length.

    Parameters
    ----------
    signal : np.ndarray
        Input signal.
    amp_thres : float, optional
        Amplitude Threshold for the binarisation. If None, the mean of the signal is used.

    Returns
    -------
    lz_index : float
        Lempel-Ziv complexity index
    """
    if threshold is None:
        threshold = np.mean(signal)

    binary_signal = (signal > threshold).astype(int)
    string_binary_signal = "".join(map(str, binary_signal))
    lz_index = calc_lempel_ziv_complexity(string_binary_signal)

    return lz_index


# ############################################ STATISTICAL DOMAIN #################################################### #
@set_domain("domain", "statistical")
@set_domain("tag", "audio")
def abs_energy(signal):
    """Computes the absolute energy of the signal.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which the area under the curve is computed

    Returns
    -------
    float
        Absolute energy
    """
    return np.sum(np.abs(signal) ** 2)


@set_domain("domain", "statistical")
@set_domain("tag", "audio")
def average_power(signal, fs):
    """Computes the average power of the signal.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Signal from which average power is computed
    fs : float
        Sampling frequency

    Returns
    -------
    float
        Average power
    """
    time = compute_time(signal, fs)

    return np.sum(np.array(signal) ** 2) / (time[-1] - time[0])


@set_domain("domain", "statistical")
@set_domain("tag", "eeg")
def entropy(signal, prob="standard"):
    """Computes the entropy of the signal using the Shannon Entropy.

    Description in Article:
    Regularities Unseen, Randomness Observed: Levels of Entropy Convergence
    Authors: Crutchfield J. Feldman David

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which entropy is computed
    prob : string
        Probability function (kde or gaussian functions are available)

    Returns
    -------
    float
        The normalized entropy value
    """

    if prob == "standard":
        value, counts = np.unique(signal, return_counts=True)
        p = counts / counts.sum()
    elif prob == "kde":
        p = kde(signal)
    elif prob == "gauss":
        p = gaussian(signal)

    if np.sum(p) == 0:
        return 0.0

    # Handling zero probability values
    p = p[np.where(p != 0)]

    # If probability all in one value, there is no entropy
    if np.log2(len(signal)) == 1:
        return 0.0
    elif np.sum(p * np.log2(p)) / np.log2(len(signal)) == 0:
        return 0.0
    else:
        return -np.sum(p * np.log2(p)) / np.log2(len(signal))


@set_domain("domain", "statistical")
def hist_mode(signal, nbins=10):
    """Compute the mode of a histogram using a given number of (linearly spaced)
    bins.

    Feature computational cost: 1

    Parameters
    ----------
    signal : np.ndarray
        Input signal from which the histogram is computed.
    nbins : int
        The number of equal-width bins in the given range, by default 10.

    Returns
    -------
    float
        The mode of the histogram (the midpoint of the bin with the highest
        count).
    """

    hist_values, bin_edges = np.histogram(signal, bins=nbins)
    max_bin_idx = np.argmax(hist_values)
    mode_value = (bin_edges[max_bin_idx] + bin_edges[max_bin_idx + 1]) / 2.0

    return mode_value


@set_domain("domain", "statistical")
def interq_range(signal):
    """Computes interquartile range of the signal.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which interquartile range is computed

    Returns
    -------
    float
        Interquartile range result
    """
    return np.percentile(signal, 75) - np.percentile(signal, 25)


@set_domain("domain", "statistical")
def kurtosis(signal):
    """Computes kurtosis of the signal.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which kurtosis is computed

    Returns
    -------
    float
        Kurtosis result
    """
    return scipy.stats.kurtosis(signal)


@set_domain("domain", "statistical")
def skewness(signal):
    """Computes skewness of the signal.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which skewness is computed

    Returns
    -------
    int
        Skewness result
    """
    return scipy.stats.skew(signal)


@set_domain("domain", "statistical")
def calc_max(signal):
    """Computes the maximum value of the signal.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
       Input from which max is computed

    Returns
    -------
    float
        Maximum result
    """
    return np.max(signal)


@set_domain("domain", "statistical")
def calc_min(signal):
    """Computes the minimum value of the signal.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which min is computed

    Returns
    -------
    float
        Minimum result
    """
    return np.min(signal)


@set_domain("domain", "statistical")
@set_domain("tag", "inertial")
def calc_mean(signal):
    """Computes mean value of the signal.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which mean is computed.

    Returns
    -------
    float
        Mean result
    """
    return np.mean(signal)


@set_domain("domain", "statistical")
def calc_median(signal):
    """Computes median of the signal.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which median is computed

    Returns
    -------
    float
        Median result
    """
    return np.median(signal)


@set_domain("domain", "statistical")
def mean_abs_deviation(signal):
    """Computes mean absolute deviation of the signal.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which mean absolute deviation is computed

    Returns
    -------
    float
        Mean absolute deviation result
    """
    return np.mean(np.abs(signal - np.mean(signal, axis=0)), axis=0)


@set_domain("domain", "statistical")
def median_abs_deviation(signal):
    """Computes median absolute deviation of the signal.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which median absolute deviation is computed

    Returns
    -------
    float
        Mean absolute deviation result
    """
    return scipy.stats.median_abs_deviation(signal, scale=1)


@set_domain("domain", "statistical")
@set_domain("tag", ["inertial", "emg"])
def rms(signal):
    """Computes root mean square of the signal.

    Square root of the arithmetic mean (average) of the squares of the original values.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which root mean square is computed

    Returns
    -------
    float
        Root mean square
    """
    return np.sqrt(np.sum(np.array(signal) ** 2) / len(signal))


@set_domain("domain", "statistical")
def calc_std(signal):
    """Computes standard deviation (std) of the signal.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which std is computed

    Returns
    -------
    float
        Standard deviation result
    """
    return np.std(signal)


@set_domain("domain", "statistical")
def calc_var(signal):
    """Computes variance of the signal.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
       Input from which var is computed

    Returns
    -------
    float
        Variance result
    """
    return np.var(signal)


@set_domain("domain", "statistical")
def pk_pk_distance(signal):
    """Computes the peak to peak distance.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which peak to peak is computed

    Returns
    -------
    float
        peak to peak distance
    """
    return np.abs(np.max(signal) - np.min(signal))


@set_domain("domain", "statistical")
def ecdf(signal, d=10):
    """Computes the values of ECDF (empirical cumulative distribution function)
    along the time axis.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which ECDF is computed
    d: integer
        Number of ECDF values to return

    Returns
    -------
    float
        The values of the ECDF along the time axis
    """
    _, y = calc_ecdf(signal)
    if len(signal) <= d:
        return tuple(y)
    else:
        return tuple(y[:d])


@set_domain("domain", "statistical")
def ecdf_slope(signal, p_init=0.5, p_end=0.75):
    """Computes the slope of the ECDF between two percentiles. Possibility to
    return infinity values.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which ECDF is computed
    p_init : float
        Initial percentile
    p_end : float
        End percentile

    Returns
    -------
    float
        The slope of the ECDF between two percentiles
    """
    signal = np.array(signal)
    # check if signal is constant
    if np.sum(np.diff(signal)) == 0:
        return np.inf
    else:
        x_init, x_end = ecdf_percentile(signal, percentile=[p_init, p_end])
        return (p_end - p_init) / (x_end - x_init)


@set_domain("domain", "statistical")
def ecdf_percentile(signal, percentile=None):
    """Computes the percentile value of the ECDF.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which ECDF is computed
    percentile: list
        Percentile value to be computed

    Returns
    -------
    float
        The input value(s) of the ECDF
    """
    if percentile is None:
        percentile = [0.2, 0.8]
    signal = np.array(signal)
    if isinstance(percentile, str):
        percentile = safe_eval_string(percentile)
    if isinstance(percentile, (float, int)):
        percentile = [percentile]

    # calculate ecdf
    x, y = calc_ecdf(signal)

    if len(percentile) > 1:
        # check if signal is constant
        if np.sum(np.diff(signal)) == 0:
            return tuple(np.repeat(signal[0], len(percentile)))
        else:
            return tuple([x[y <= p].max() for p in percentile])
    else:
        # check if signal is constant
        if np.sum(np.diff(signal)) == 0:
            return signal[0]
        else:
            return x[y <= percentile].max()


@set_domain("domain", "statistical")
def ecdf_percentile_count(signal, percentile=None):
    """Computes the cumulative sum of samples that are less than the
    percentile.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which ECDF is computed
    percentile: list
        Percentile threshold

    Returns
    -------
    float
        The cumulative sum of samples
    """
    if percentile is None:
        percentile = [0.2, 0.8]

    signal = np.array(signal)
    if isinstance(percentile, str):
        percentile = safe_eval_string(percentile)
    if isinstance(percentile, (float, int)):
        percentile = [percentile]

    # calculate ecdf
    x, y = calc_ecdf(signal)

    if len(percentile) > 1:
        # check if signal is constant
        if np.sum(np.diff(signal)) == 0:
            return tuple(np.repeat(signal[0], len(percentile)))
        else:
            return tuple([x[y <= p].shape[0] for p in percentile])
    else:
        # check if signal is constant
        if np.sum(np.diff(signal)) == 0:
            return signal[0]
        else:
            return x[y <= percentile].shape[0]


# ############################################## SPECTRAL DOMAIN ##################################################### #


@set_domain("domain", "spectral")
def spectral_distance(signal, fs):
    """Computes the signal spectral distance.

    Distance of the signal's cumulative sum of the FFT elements to
    the respective linear regression.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Signal from which spectral distance is computed
    fs : float
        Sampling frequency

    Returns
    -------
    float
        spectral distance
    """
    f, fmag = calc_fft(signal, fs)

    cum_fmag = np.cumsum(fmag)

    # Computing the linear regression
    points_y = np.linspace(0, cum_fmag[-1], len(cum_fmag))

    return np.sum(points_y - cum_fmag)


@set_domain("domain", "spectral")
def fundamental_frequency(signal, fs):
    """Computes fundamental frequency of the signal.

    The fundamental frequency integer multiple best explain
    the content of the signal spectrum.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which fundamental frequency is computed
    fs : float
        Sampling frequency

    Returns
    -------
    f0: float
       Predominant frequency of the signal
    """
    signal = signal - np.mean(signal)
    f, fmag = calc_fft(signal, fs)

    # Finding big peaks, not considering noise peaks with low amplitude

    bp = scipy.signal.find_peaks(fmag, height=max(fmag) * 0.3)[0]

    # # Condition for offset removal, since the offset generates a peak at frequency zero
    bp = bp[bp != 0]
    if not list(bp):
        f0 = 0
    else:
        # f0 is the minimum big peak frequency
        f0 = f[min(bp)]

    return f0


@set_domain("domain", "spectral")
def max_power_spectrum(signal, fs):
    """Computes maximum power spectrum density of the signal.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which maximum power spectrum is computed
    fs : float
        Sampling frequency

    Returns
    -------
    nd-array
        Max value of the power spectrum density
    """
    if np.std(signal) == 0:
        return float(max(scipy.signal.welch(signal, fs, nperseg=len(signal))[1]))
    else:
        return float(max(scipy.signal.welch(signal / np.std(signal), fs, nperseg=len(signal))[1]))


@set_domain("domain", "spectral")
def max_frequency(signal, fs):
    """Computes maximum frequency of the signal.

    Feature computational cost: 2

    Parameters
    ----------
    signal : nd-array
        Input from which maximum frequency is computed
    fs : float
        Sampling frequency

    Returns
    -------
    float
        0.95 of maximum frequency using cumsum
    """
    f, fmag = calc_fft(signal, fs)
    cum_fmag = np.cumsum(fmag)

    try:
        ind_mag = np.where(cum_fmag > cum_fmag[-1] * 0.95)[0][0]
    except IndexError:
        ind_mag = np.argmax(cum_fmag)

    return f[ind_mag]


@set_domain("domain", "spectral")
def median_frequency(signal, fs):
    """Computes median frequency of the signal.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which median frequency is computed
    fs: int
        Sampling frequency

    Returns
    -------
    f_median : int
       0.50 of maximum frequency using cumsum.
    """
    f, fmag = calc_fft(signal, fs)
    cum_fmag = np.cumsum(fmag)
    try:
        ind_mag = np.where(cum_fmag > cum_fmag[-1] * 0.50)[0][0]
    except IndexError:
        ind_mag = np.argmax(cum_fmag)
    f_median = f[ind_mag]

    return f_median


@set_domain("domain", "spectral")
@set_domain("tag", "audio")
def spectral_centroid(signal, fs):
    """Barycenter of the spectrum.

    Description and formula in Article:
    The Timbre Toolbox: Extracting audio descriptors from musicalsignals
    Authors Peeters G., Giordano B., Misdariis P., McAdams S.

    Feature computational cost: 2

    Parameters
    ----------
    signal : nd-array
        Signal from which spectral centroid is computed
    fs: int
        Sampling frequency

    Returns
    -------
    float
        Centroid
    """
    f, fmag = calc_fft(signal, fs)
    if not np.sum(fmag):
        return 0
    else:
        return np.dot(f, fmag / np.sum(fmag))


@set_domain("domain", "spectral")
def spectral_decrease(signal, fs):
    """Represents the amount of decreasing of the spectra amplitude.

    Description and formula in Article:
    The Timbre Toolbox: Extracting audio descriptors from musicalsignals
    Authors Peeters G., Giordano B., Misdariis P., McAdams S.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Signal from which spectral decrease is computed
    fs : float
        Sampling frequency

    Returns
    -------
    float
        Spectral decrease
    """
    f, fmag = calc_fft(signal, fs)

    fmag_band = fmag[1:]
    len_fmag_band = np.arange(2, len(fmag) + 1)

    # Sum of numerator
    soma_num = np.sum((fmag_band - fmag[0]) / (len_fmag_band - 1), axis=0)

    if not np.sum(fmag_band):
        return 0
    else:
        # Sum of denominator
        soma_den = 1 / np.sum(fmag_band)

        # Spectral decrease computing
        return soma_den * soma_num


@set_domain("domain", "spectral")
def spectral_kurtosis(signal, fs):
    """Measures the flatness of a distribution around its mean value.

    Description and formula in Article:
    The Timbre Toolbox: Extracting audio descriptors from musicalsignals
    Authors Peeters G., Giordano B., Misdariis P., McAdams S.

    Feature computational cost: 2

    Parameters
    ----------
    signal : nd-array
        Signal from which spectral kurtosis is computed
    fs : float
        Sampling frequency

    Returns
    -------
    float
        Spectral Kurtosis
    """
    f, fmag = calc_fft(signal, fs)
    if not spectral_spread(signal, fs):
        return 0
    else:
        spect_kurt = ((f - spectral_centroid(signal, fs)) ** 4) * (fmag / np.sum(fmag))
        return np.sum(spect_kurt) / (spectral_spread(signal, fs) ** 4)


@set_domain("domain", "spectral")
def spectral_skewness(signal, fs):
    """Measures the asymmetry of a distribution around its mean value.

    Description and formula in Article:
    The Timbre Toolbox: Extracting audio descriptors from musicalsignals
    Authors Peeters G., Giordano B., Misdariis P., McAdams S.

    Feature computational cost: 2

    Parameters
    ----------
    signal : nd-array
        Signal from which spectral skewness is computed
    fs : float
        Sampling frequency

    Returns
    -------
    float
        Spectral Skewness
    """
    f, fmag = calc_fft(signal, fs)
    spect_centr = spectral_centroid(signal, fs)

    if not spectral_spread(signal, fs):
        return 0
    else:
        skew = ((f - spect_centr) ** 3) * (fmag / np.sum(fmag))
        return np.sum(skew) / (spectral_spread(signal, fs) ** 3)


@set_domain("domain", "spectral")
def spectral_spread(signal, fs):
    """Measures the spread of the spectrum around its mean value.

    Description and formula in Article:
    The Timbre Toolbox: Extracting audio descriptors from musicalsignals
    Authors Peeters G., Giordano B., Misdariis P., McAdams S.

    Feature computational cost: 2

    Parameters
    ----------
    signal : nd-array
        Signal from which spectral spread is computed.
    fs : float
        Sampling frequency

    Returns
    -------
    float
        Spectral Spread
    """
    f, fmag = calc_fft(signal, fs)
    spect_centroid = spectral_centroid(signal, fs)

    if not np.sum(fmag):
        return 0
    else:
        return np.dot(((f - spect_centroid) ** 2), (fmag / np.sum(fmag))) ** 0.5


@set_domain("domain", "spectral")
def spectral_slope(signal, fs):
    """Computes the spectral slope.

    Spectral slope is computed by finding constants m and b of the function aFFT = mf + b, obtained by linear regression
    of the spectral amplitude.

    Description and formula in Article:
    The Timbre Toolbox: Extracting audio descriptors from musicalsignals
    Authors Peeters G., Giordano B., Misdariis P., McAdams S.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Signal from which spectral slope is computed
    fs : float
        Sampling frequency

    Returns
    -------
    float
        Spectral Slope
    """
    f, fmag = calc_fft(signal, fs)
    sum_fmag = fmag.sum()
    dot_ff = (f * f).sum()
    sum_f = f.sum()
    len_f = len(f)

    if not ([f]) or (sum_fmag == 0):
        return 0
    else:
        if not (len_f * dot_ff - sum_f**2):
            return 0
        else:
            num_ = (1 / sum_fmag) * (len_f * np.sum(f * fmag) - sum_f * sum_fmag)
            denom_ = len_f * dot_ff - sum_f**2
            return num_ / denom_


@set_domain("domain", "spectral")
def spectral_variation(signal, fs):
    """Computes the amount of variation of the spectrum along time.

    Spectral variation is computed from the normalized cross-correlation between two consecutive amplitude spectra.

    Description and formula in Article:
    The Timbre Toolbox: Extracting audio descriptors from musicalsignals
    Authors Peeters G., Giordano B., Misdariis P., McAdams S.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Signal from which spectral variation is computed.
    fs : float
        Sampling frequency

    Returns
    -------
    float
        Spectral Variation
    """
    f, fmag = calc_fft(signal, fs)

    sum1 = np.sum(np.array(fmag)[:-1] * np.array(fmag)[1:])
    sum2 = np.sum(np.array(fmag)[1:] ** 2)
    sum3 = np.sum(np.array(fmag)[:-1] ** 2)

    if not sum2 or not sum3:
        variation = 1
    else:
        variation = 1 - (sum1 / ((sum2**0.5) * (sum3**0.5)))

    return variation


@set_domain("domain", "spectral")
def spectral_positive_turning(signal, fs):
    """Computes number of positive turning points of the fft magnitude signal.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which the number of positive turning points of the fft magnitude are computed
    fs : float
        Sampling frequency

    Returns
    -------
    float
        Number of positive turning points
    """
    f, fmag = calc_fft(signal, fs)
    diff_sig = np.diff(fmag)

    array_signal = np.arange(len(diff_sig[:-1]))

    positive_turning_pts = np.where((diff_sig[array_signal + 1] < 0) & (diff_sig[array_signal] > 0))[0]

    return len(positive_turning_pts)


@set_domain("domain", "spectral")
@set_domain("tag", "audio")
def spectral_roll_off(signal, fs):
    """Computes the spectral roll-off of the signal.

    The spectral roll-off corresponds to the frequency where 95% of the signal magnitude is contained
    below of this value.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Signal from which spectral roll-off is computed
    fs : float
        Sampling frequency

    Returns
    -------
    float
        Spectral roll-off
    """
    f, fmag = calc_fft(signal, fs)
    cum_ff = np.cumsum(fmag)
    value = 0.95 * (np.sum(fmag))

    return f[np.where(cum_ff >= value)[0][0]]


@set_domain("domain", "spectral")
def spectral_roll_on(signal, fs):
    """Computes the spectral roll-on of the signal.

    The spectral roll-on corresponds to the frequency where 5% of the signal magnitude is contained
    below of this value.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Signal from which spectral roll-on is computed
    fs : float
        Sampling frequency

    Returns
    -------
    float
        Spectral roll-on
    """
    f, fmag = calc_fft(signal, fs)
    cum_ff = np.cumsum(fmag)
    value = 0.05 * (np.sum(fmag))

    return f[np.where(cum_ff >= value)[0][0]]


@set_domain("domain", "spectral")
@set_domain("tag", "inertial")
def human_range_energy(signal, fs):
    """Computes the human range energy ratio.

    The human range energy ratio is given by the ratio between the energy
    in frequency 0.6-2.5Hz and the whole energy band.

    Feature computational cost: 2

    Parameters
    ----------
    signal : nd-array
        Signal from which human range energy ratio is computed
    fs : float
        Sampling frequency

    Returns
    -------
    float
        Human range energy ratio
    """
    f, fmag = calc_fft(signal, fs)

    allenergy = np.sum(fmag**2)

    if allenergy == 0:
        # For handling the occurrence of Nan values
        return 0.0

    hr_energy = np.sum(fmag[np.argmin(np.abs(0.6 - f)) : np.argmin(np.abs(2.5 - f))] ** 2)

    ratio = hr_energy / allenergy

    return ratio


@set_domain("domain", "spectral")
@set_domain("tag", ["audio", "emg"])
def mfcc(signal, fs, pre_emphasis=0.97, nfft=512, nfilt=40, num_ceps=12, cep_lifter=22):
    """Computes the MEL cepstral coefficients.

    It provides the information about the power in each frequency band.

    Implementation details and description on:
    https://www.kaggle.com/ilyamich/mfcc-implementation-and-tutorial
    https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html#fnref:1

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which MEL coefficients is computed
    fs : float
        Sampling frequency
    pre_emphasis : float
        Pre-emphasis coefficient for pre-emphasis filter application
    nfft : int
        Number of points of fft
    nfilt : int
        Number of filters
    num_ceps: int
        Number of cepstral coefficients
    cep_lifter: int
        Filter length

    Returns
    -------
    nd-array
        MEL cepstral coefficients
    """
    filter_banks = filterbank(signal, fs, pre_emphasis, nfft, nfilt)

    mel_coeff = scipy.fft.dct(filter_banks, type=2, axis=0, norm="ortho")[1 : (num_ceps + 1)]  # Keep 2-13

    mel_coeff -= np.mean(mel_coeff, axis=0) + 1e-8

    # liftering
    ncoeff = len(mel_coeff)
    n = np.arange(ncoeff)
    lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)  # cep_lifter = 22 from python_speech_features library

    mel_coeff *= lift

    return tuple(mel_coeff)


@set_domain("domain", "spectral")
def power_bandwidth(signal, fs):
    """Computes power spectrum density bandwidth of the signal.

    It corresponds to the width of the frequency band in which 95% of its power is located.

    Description in article:
    Power Spectrum and Bandwidth Ulf Henriksson, 2003 Translated by Mikael Olofsson, 2005

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which the power bandwidth computed
    fs : float
        Sampling frequency

    Returns
    -------
    float
        Occupied power in bandwidth
    """
    # Computing the power spectrum density
    if np.std(signal) == 0:
        freq, power = scipy.signal.welch(signal, fs, nperseg=len(signal))
    else:
        freq, power = scipy.signal.welch(signal / np.std(signal), fs, nperseg=len(signal))

    if np.sum(power) == 0:
        return 0.0

    # Computing the lower and upper limits of power bandwidth
    cum_power = np.cumsum(power)
    f_lower = freq[np.where(cum_power >= cum_power[-1] * 0.95)[0][0]]

    cum_power_inv = np.cumsum(power[::-1])
    f_upper = freq[np.abs(np.where(cum_power_inv >= cum_power[-1] * 0.95)[0][0] - len(power) + 1)]

    # Returning the bandwidth in terms of frequency

    return np.abs(f_upper - f_lower)


@set_domain("domain", "spectral")
def spectrogram_mean_coeff(signal, fs, bins=32):
    """Calculates the average power spectral density (PSD) for each frequency
    throughout the entire signal duration provided by the spectrogram.

    The values represent the average power spectral density computed on frequency bins. The feature name refers to the
    frequency bin where the PSD was taken. Each bin is ``fs`` / (``bins`` * 2 - 2) Hz wide. The method relies on the
    `scipy.signal.spectrogram` and except for ``nperseg`` and ``fs``, all the other parameters are set to its defaults.

    Feature computational cost: 1

    Parameters
    ----------
    signal : array_like
        Input from which the spectrogram average power spectral density coefficients are computed.
    fs : float
        Sampling frequency of the ``signal``.
    bins : int, optional
        The number of frequency bins.

    Returns
    -------
    nd-array
        The power spectral density for each frequency bin averaged along the entire signal duration.

    Notes
    -----
    The optimal number of frequency bins depend on the task at hand. Using a
    higher number of bins with low sampling frequencies may result in excessive
    frequency resolution and the loss of valuable coarse-grained information.
    The default value should be suitable for most cases when working with the
    default sampling frequency. The number of frequency bins must be modified
    in the feature configuration file.

    .. versionadded:: 0.1.7
    """

    if bins > len(signal) // 2 + 1:
        bins = len(signal) // 2 + 1

    frequencies, _, Sxx = scipy.signal.spectrogram(signal, fs, nperseg=bins * 2 - 2)
    Sxx_mean = Sxx.mean(1)
    f_keys = np.round(frequencies, 2).astype(str)

    return {"names": [f + "Hz" for f in f_keys], "values": Sxx_mean}


@set_domain("domain", "spectral")
@set_domain("tag", "audio")
def lpcc(signal, n_coeff=12):
    """Computes the linear prediction cepstral coefficients.

    Implementation details and description in:
    http://www.practicalcryptography.com/miscellaneous/machine-learning/tutorial-cepstrum-and-lpccs/

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from linear prediction cepstral coefficients are computed
    n_coeff : int
        Number of coefficients

    Returns
    -------
    nd-array
        Linear prediction cepstral coefficients
    """
    # 12-20 cepstral coefficients are sufficient for speech recognition
    lpc_coeffs = lpc(signal, n_coeff)

    if np.sum(lpc_coeffs) == 0:
        return tuple(np.zeros(len(lpc_coeffs)))

    # Power spectrum
    powerspectrum = np.abs(np.fft.fft(lpc_coeffs)) ** 2
    lpcc_coeff = np.fft.ifft(np.log(powerspectrum))

    return tuple(np.abs(lpcc_coeff))


@set_domain("domain", "spectral")
@set_domain("tag", "eeg")
def spectral_entropy(signal, fs):
    """Computes the spectral entropy of the signal based on Fourier transform.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which spectral entropy is computed
    fs : float
        Sampling frequency

    Returns
    -------
    float
        The normalized spectral entropy value
    """
    # Removing DC component
    sig = signal - np.mean(signal)

    f, fmag = calc_fft(sig, fs)

    power = fmag**2

    if power.sum() == 0:
        return 0.0

    prob = np.divide(power, power.sum())

    prob = prob[prob != 0]

    # If probability all in one value, there is no entropy
    if prob.size == 1:
        return 0.0

    return -np.multiply(prob, np.log2(prob)).sum() / np.log2(prob.size)


@set_domain("domain", "spectral")
@set_domain("tag", "eeg")
def wavelet_entropy(signal, fs, wavelet="mexh", max_width=10):
    """Computes CWT entropy of the signal.

    Implementation details in:
    https://dsp.stackexchange.com/questions/13055/how-to-calculate-cwt-shannon-entropy
    B.F. Yan, A. Miyamoto, E. Bruhwiler, Wavelet transform-based modal parameter identification considering uncertainty

    Parameters
    ----------
    signal : nd-array
        Input from which CWT is computed
    fs: int
        Signal sampling frequency
    wavelet : string
        Wavelet to use, defaults to "mexh" which represents the mexican hat wavelet (Ricker wavelet)
    max_width : int
        Maximum width to use for transformation, defaults to 10

    Returns
    -------
    float
        wavelet entropy
    """
    if np.sum(signal) == 0:
        return 0.0

    max_width = int(max_width)

    widths = np.arange(1, max_width)

    coeffs, _ = continuous_wavelet_transform(signal=signal, fs=fs, wavelet=wavelet, widths=widths)
    energy_scale = np.sum(np.abs(coeffs), axis=1)
    t_energy = np.sum(energy_scale)
    prob = energy_scale / t_energy
    w_entropy = -np.sum(prob * np.log(prob))

    return w_entropy


@set_domain("domain", "spectral")
@set_domain("tag", ["eeg", "ecg"])
def wavelet_abs_mean(signal, fs, wavelet="mexh", max_width=10):
    """Computes CWT absolute mean value of each wavelet scale.

    Parameters
    ----------
    signal : nd-array
        Input from which CWT is computed
    fs: int
        Signal sampling frequency
    wavelet : string
        Wavelet to use, defaults to "mexh" which represents the mexican hat wavelet (Ricker wavelet)
    max_width : int
        Maximum width to use for transformation, defaults to 10

    Returns
    -------
    nd-array
        CWT absolute mean value
    """
    max_width = int(max_width)
    widths = np.arange(1, max_width)

    coeffs, frequencies = continuous_wavelet_transform(signal=signal, fs=fs, wavelet=wavelet, widths=widths)
    f_keys = np.round(frequencies, 2).astype(str)

    return {"names": [f + "Hz" for f in f_keys], "values": np.abs(np.mean(coeffs, axis=1))}


@set_domain("domain", "spectral")
@set_domain("domain", "eeg")
def wavelet_std(signal, fs, wavelet="mexh", max_width=10):
    """Computes CWT std value of each wavelet scale.

    Parameters
    ----------
    signal : nd-array
        Input from which CWT is computed
    fs: int
        Signal sampling frequency
    wavelet : string
        Wavelet to use, defaults to "mexh" which represents the mexican hat wavelet (Ricker wavelet)
    max_width : int
        Maximum width to use for transformation, defaults to 10

    Returns
    -------
    nd-array
        CWT std
    """
    max_width = int(max_width)
    widths = np.arange(1, max_width)

    coeffs, frequencies = continuous_wavelet_transform(signal=signal, fs=fs, wavelet=wavelet, widths=widths)
    f_keys = np.round(frequencies, 2).astype(str)

    return {"names": [f + "Hz" for f in f_keys], "values": np.std(coeffs, axis=1)}


@set_domain("domain", "spectral")
@set_domain("tag", "eeg")
def wavelet_var(signal, fs, wavelet="mexh", max_width=10):
    """Computes CWT variance value of each wavelet scale.

    Parameters
    ----------
    signal : nd-array
        Input from which CWT is computed
    fs: int
        Signal sampling frequency
    wavelet : string
        Wavelet to use, defaults to "mexh" which represents the mexican hat wavelet (Ricker wavelet)
    max_width : int
        Maximum width to use for transformation, defaults to 10

    Returns
    -------
    nd-array
        CWT variance
    """
    max_width = int(max_width)
    widths = np.arange(1, max_width)

    coeffs, frequencies = continuous_wavelet_transform(signal=signal, fs=fs, wavelet=wavelet, widths=widths)
    f_keys = np.round(frequencies, 2).astype(str)

    return {"names": [f + "Hz" for f in f_keys], "values": np.var(coeffs, axis=1)}


@set_domain("domain", "spectral")
@set_domain("tag", "eeg")
def wavelet_energy(signal, fs, wavelet="mexh", max_width=10):
    """Computes CWT energy of each wavelet scale.

    Implementation details:
    https://stackoverflow.com/questions/37659422/energy-for-1-d-wavelet-in-python

    Parameters
    ----------
    signal : nd-array
        Input from which CWT is computed
    fs: int
        Signal sampling frequency
    wavelet : string
        Wavelet to use, defaults to "mexh" which represents the mexican hat wavelet (Ricker wavelet)
    max_width : int
        Maximum width to use for transformation, defaults to 10

    Returns
    -------
    nd-array
        CWT energy
    """
    max_width = int(max_width)
    widths = np.arange(1, max_width)

    coeffs, frequencies = continuous_wavelet_transform(signal=signal, fs=fs, wavelet=wavelet, widths=widths)
    f_keys = np.round(frequencies, 2).astype(str)

    return {"names": [f + "Hz" for f in f_keys], "values": np.sqrt(np.sum(coeffs**2, axis=1) / np.shape(coeffs)[1])}


# ############################################## FRACTAL DOMAIN ##################################################### #
@set_domain("domain", "fractal")
def dfa(signal):
    """Computes the Detrended Fluctuation Analysis (DFA) of the signal.

    Parameters
    ----------
    signal : np.ndarray
        Input signal.

    Returns
    -------
    alpha_dfa : float
        Scaling exponent in DFA.
    """
    global warning_flag

    if np.var(signal) == 0 and np.all(signal == signal[0]):
        return np.nan

    n = len(signal)

    if n < FEATURES_MIN_SIZE:
        if not warning_flag:
            warnings.warn(warning_msg, UserWarning)
            warning_flag = True
        return np.nan

    accumulated_signal = np.cumsum(signal - np.mean(signal))
    windows = set(np.linspace(4, n // 10, n // 2, dtype=int))
    fluct = np.zeros(len(windows))

    for idx, window in enumerate(windows):
        fluct[idx] = np.sqrt(np.mean(calc_rms(accumulated_signal, window) ** 2))

    i_plateau = find_plateau(np.log(fluct))
    fluct = fluct[0:i_plateau]
    windows = list(windows)[0:i_plateau]

    coeffs = np.polyfit(np.log(windows), np.log(fluct), 1)
    alpha_dfa = coeffs[0]

    return alpha_dfa


@set_domain("domain", "fractal")
def hurst_exponent(signal):
    """Computes the Hurst exponent of the signal through the Rescaled range
    (R/S) analysis.

    Parameters
    ----------
    signal : np.ndarray
        Input signal.

    Returns
    -------
    h_exp : float
        Hurst exponent.
    """
    global warning_flag

    if np.var(signal) == 0 and np.all(signal == signal[0]):
        return np.nan

    n = len(signal)

    if n < FEATURES_MIN_SIZE:
        if not warning_flag:
            warnings.warn(warning_msg, UserWarning)
            warning_flag = True
        return np.nan

    lags = set(np.linspace(4, n // 10, n // 2, dtype=int))
    rs = [compute_rs(signal, lag) for lag in lags]

    n_values = np.array(list(lags))[np.isfinite(rs)]
    rs = np.array(rs)[np.isfinite(rs)]

    coeffs = np.polyfit(np.log10(n_values), np.log10(rs), 1)
    h_exp = coeffs[0]

    return h_exp


@set_domain("domain", "fractal")
def higuchi_fractal_dimension(signal):
    """Computes the fractal dimension of a signal using Higuchi's method (HFD).

    Parameters
    ----------
    signal : np.ndarray
        Input signal.

    Returns
    -------
    hfd : float
       Fractal dimension.
    """
    global warning_flag

    n = len(signal)

    if n < FEATURES_MIN_SIZE:
        if not warning_flag:
            warnings.warn(warning_msg, UserWarning)
            warning_flag = True
        return np.nan

    k_values, lk = calc_lengths_higuchi(signal)

    coeffs = np.polyfit(np.log(1 / k_values), np.log(lk), 1)
    hfd = coeffs[0]

    return hfd


@set_domain("domain", "fractal")
def maximum_fractal_length(signal):
    """Computes the Maximum Fractal Length (MFL) of the signal, which is the
    average length at the smallest scale, measured from the logarithmic plot
    determining FD. The Higuchi's method is used.

    Parameters
    ----------
    signal : np.ndarray
        Input signal.

    Returns
    -------
    mfl : float
       Maximum Fractal Length.
    """
    global warning_flag

    n = len(signal)

    if n < FEATURES_MIN_SIZE:
        if not warning_flag:
            warnings.warn(warning_msg, UserWarning)
            warning_flag = True
        return np.nan

    k_values, lk = calc_lengths_higuchi(signal)

    coeffs = np.polyfit(np.log10(1 / k_values), np.log10(lk), 1)
    trendpoly = np.poly1d(coeffs)
    mfl_value = trendpoly(0)

    return mfl_value


@set_domain("domain", "fractal")
def petrosian_fractal_dimension(signal):
    """Computes the Petrosian Fractal Dimension of a signal.

    Parameters
    ----------
    signal : np.ndarray
        Input signal.

    Returns
    -------
    pfd : float
       Petrosian Fractal Dimension.
    """
    n = len(signal)
    diff_signal = np.diff(np.sign(np.diff(signal)))
    num_sign_changes = np.sum(diff_signal != 0)

    pfd = np.log10(n) / (np.log10(n) + np.log10(n / (n + 0.4 * num_sign_changes)))

    return pfd


@set_domain("domain", "fractal")
def mse(signal, m=3, maxscale=None, tolerance=None):
    """Computes the Multiscale entropy (MSE) of the signal, that performs the
    entropy analysis over multiple time scales.

    Parameters
    ----------
    signal : np.ndarray
        Input signal.
    m : int
        Embedding dimension for the sample entropy, defaults to 3.
    maxscale : int
        Maximum scale factor, defaults to 1/13 of the length of the input signal.
    tolerance : float
        Tolerance value, defaults to 0.2 times the standard deviation of the input signal.

    Returns
    -------
    mse_area : np.ndarray
        Normalized area under the MSE curve.
    """
    global warning_flag

    if np.var(signal) == 0 and np.all(signal == signal[0]):
        return np.nan

    n = len(signal)

    if n < FEATURES_MIN_SIZE:
        if not warning_flag:
            warnings.warn(warning_msg, UserWarning)
            warning_flag = True
        return np.nan

    if tolerance is None:
        tolerance = 0.2 * np.std(signal)

    if maxscale is None:
        maxscale = n // (10 + 3)

    mse_values = np.array([sample_entropy(coarse_graining(signal, i + 1), m, tolerance) for i in np.arange(maxscale)])
    mse_values_finite = mse_values[np.isfinite(mse_values)]
    mse_area = np.trapz(mse_values_finite) / len(mse_values_finite)

    return mse_area
