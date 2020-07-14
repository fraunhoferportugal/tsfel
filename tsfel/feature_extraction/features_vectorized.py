import scipy.signal
from tsfel.feature_extraction.features_utils import *
import tsfel.feature_extraction.features as fts

# ############################################# TEMPORAL DOMAIN ##################################################### #


@set_domain("domain", "temporal")
def autocorr(signal):
    """Computes autocorrelation of the signal.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which autocorrelation is computed

    Returns
    -------
    float
        Cross correlation of 1-dimensional sequence

    """
    signal = np.array(signal)
    return vectorized(fts.autocorr, signal)


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

    energy = signal ** 2

    t_energy = np.sum(time * energy, axis=-1)
    energy_sum = np.sum(energy, axis=-1)
    return divideWithZero(t_energy, energy_sum)


@set_domain("domain", "temporal")
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

    return vectorized(fts.negative_turning, signal)


@set_domain("domain", "temporal")
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
    return vectorized(fts.positive_turning, signal)



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
    return np.mean(np.abs(np.diff(signal)), axis=-1)


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
    return np.mean(np.diff(signal), axis=-1)


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
    return np.median(np.abs(np.diff(signal)), axis=-1)



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
    return np.median(np.diff(signal), axis=-1)


@set_domain("domain", "temporal")
def distance(signal):
    """Computes signal traveled distance.

    Calculates the total distance traveled by the signal
    using the hipotenusa between 2 datapoints.

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
    return np.sum(np.sqrt(np.diff(signal) ** 2 + 1), axis=-1)



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
    return np.sum(np.abs(np.diff(signal)), axis=-1)



@set_domain("domain", "temporal")
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
    return np.sum(np.abs(np.diff(np.sign(signal))) == 2, axis=-1)



@set_domain("domain", "temporal")
def total_energy(signal, fs):
    """Computes the total energy of the signal.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Signal from which total energy is computed
    fs : int
        Sampling frequency

    Returns
    -------
    float
        Total energy

    """
    return np.sum(signal ** 2, axis=-1) / (np.ma.size(signal, axis=-1) / fs - 1./fs)



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
    return vectorized(fts.slope, signal)



@set_domain("domain", "temporal")
def auc(signal, fs):
    """Computes the area under the curve of the signal computed with trapezoid rule.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which the area under the curve is computed
    fs : int
        Sampling Frequency
    Returns
    -------
    float
        The area under the curve value

    """
    t = compute_time(signal, fs)
    return np.sum(np.diff(t, axis=-1) * signal[..., :-1] + signal[..., 1:] / 2, axis=-1)



@set_domain("domain", "temporal")
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
    return np.sum(signal ** 2, axis=-1)



@set_domain("domain", "temporal")
def pk_pk_distance(signal):
    """Computes the peak to peak distance.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which the area under the curve is computed

    Returns
    -------
    float
        peak to peak distance

    """
    return np.abs(np.max(signal, axis=-1) - np.min(signal, axis=-1))



@set_domain("domain", "temporal")
def entropy(signal, prob='standard'):
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

    if prob == 'gauss':
        p = gaussian(signal)
    elif prob == 'kde':
        p = kde(signal)
    else:
        raise Exception("Unknown prob estimator")

    return __entropy(p)



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
    return vectorized(fts.neighbourhood_peaks, signal)


# ############################################ STATISTICAL DOMAIN #################################################### #


@set_domain("domain", "statistical")
def hist(signal, nbins=10, r=1):
    """Computes histogram of the signal.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from histogram is computed
    nbins : int
        The number of equal-width bins in the given range
    r : float
        The lower(-r) and upper(r) range of the bins

    Returns
    -------
    nd-array
        The values of the histogram

    """
    return vectorized(fts.hist, signal)



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
    return np.percentile(signal, 75, axis=-1) - np.percentile(signal, 25, axis=-1)



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
    return scipy.stats.kurtosis(signal, axis=-1)


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
    return scipy.stats.skew(signal, axis=-1)



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
    return np.max(signal, axis=-1)



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
    return np.min(signal, axis=-1)



@set_domain("domain", "statistical")
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
    return np.mean(signal, axis=-1)



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
    return np.median(signal, axis=-1)



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
    return np.mean(np.abs(signal - matchLastDimByRepeat(np.mean(signal, axis=-1), signal)), axis=-1)



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
    return scipy.stats.median_absolute_deviation(signal, scale=1, axis=-1)



@set_domain("domain", "statistical")
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
    return np.sqrt(np.mean(np.square(signal), axis=-1))



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
    return np.std(signal, axis=-1)



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
    return np.var(signal, axis=-1)



@set_domain("domain", "statistical")
def ecdf(signal, d=10):
    """Computes the values of ECDF (empirical cumulative distribution function) along the time axis.

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
    return vectorized(fts.ecdf, signal)



@set_domain("domain", "statistical")
def ecdf_slope(signal, p_init=0.5, p_end=0.75):
    """Computes the slope of the ECDF between two percentiles.
    Possibility to return infinity values.

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
    return vectorized(fts.ecdf_slope, signal)



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
    return vectorized(fts.ecdf_percentile, signal)



@set_domain("domain", "statistical")
def ecdf_percentile_count(signal, percentile=None):
    """Computes the cumulative sum of samples that are less than the percentile.

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
    return vectorized(fts.ecdf_percentile_count, signal)



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
    fs : int
        Sampling frequency

    Returns
    -------
    float
        spectral distance

    """
    _, fmag = calc_fft(signal, fs)

    cum_fmag = np.cumsum(fmag, axis=-1)

    # Compute the linear regression
    # TODO: there must be a nicer version than this transpose...
    points_y = np.linspace(0, cum_fmag[...,-1], np.ma.size(cum_fmag, axis=-1))
    points_y = points_y.transpose(np.append(np.arange(1, signal.ndim), 0))

    return np.sum(points_y - cum_fmag, axis=-1)



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
    fs : int
        Sampling frequency

    Returns
    -------
    f0: float
       Predominant frequency of the signal

    """
    return vectorized(fts.fundamental_frequency, signal)



@set_domain("domain", "spectral")
def max_power_spectrum(signal, fs):
    """Computes maximum power spectrum density of the signal.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which maximum power spectrum is computed
    fs : scalar
        Sampling frequency

    Returns
    -------
    nd-array
        Max value of the power spectrum density

    """
    return vectorized(fts.max_power_spectrum, signal)



@set_domain("domain", "spectral")
def max_frequency(signal, fs):
    """Computes maximum frequency of the signal.

    Feature computational cost: 2

    Parameters
    ----------
    signal : nd-array
        Input from which maximum frequency is computed
    fs : int
        Sampling frequency

    Returns
    -------
    float
        0.95 of maximum frequency using cumsum
    """
    f, fmag = calc_fft(signal, fs)

    cum_fmag = np.cumsum(fmag, axis=-1)
    expanded = matchLastDimByRepeat(
        np.take(cum_fmag, -1, axis=-1), cum_fmag)

    try:
        ind_mag = np.argmax(
            np.array(np.asarray(cum_fmag > expanded * 0.95)), axis=-1)
    except IndexError:
        ind_mag = np.argmax(cum_fmag, axis=-1)

    ind_mag = np.expand_dims(ind_mag, axis=-1)
    return np.squeeze(np.take(f, ind_mag), axis=-1)



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

    cum_fmag = np.cumsum(fmag, axis=-1)
    expanded = matchLastDimByRepeat(
        np.take(cum_fmag, -1, axis=-1), cum_fmag)

    try:
        ind_mag = np.argmax(
            np.array(np.asarray(cum_fmag > expanded * 0.5)), axis=-1)
    except IndexError:
        ind_mag = np.argmax(cum_fmag, axis=-1)

    ind_mag = np.expand_dims(ind_mag, axis=-1)
    return np.squeeze(np.take(f, ind_mag), axis=-1)


@set_domain("domain", "spectral")
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
    summedFmag = matchLastDimByRepeat(np.sum(fmag, axis=-1), fmag)
    return np.sum(f * divideWithZero(fmag, summedFmag), axis=-1)




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
    fs : int
        Sampling frequency

    Returns
    -------
    float
        Spectral decrease

    """
    _, fmag = calc_fft(signal, fs)

    fmag_band = fmag[..., 1:]
    len_fmag_band = np.arange(1, np.ma.size(fmag, axis=-1))

    # Sum of numerator
    soma_num = np.sum(
        (fmag_band - matchLastDimByRepeat(fmag[..., 0], fmag_band)) / len_fmag_band, axis=-1)

    return divideWithZero(1, np.sum(fmag_band, axis=-1)) * soma_num



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
    fs : int
        Sampling frequency

    Returns
    -------
    float
        Spectral Kurtosis

    """
    f, fmag = calc_fft(signal, fs)
    spect_centr = spectral_centroid(signal, fs)
    spread = spectral_spread(signal, fs)
    summedFmag = matchLastDimByRepeat(np.sum(fmag, axis=-1), fmag)

    spect_kurt = ((f - matchLastDimByRepeat(spect_centr, f)) ** 4) * (fmag / summedFmag)
    return divideWithZero(np.sum(spect_kurt, axis=-1), spread ** 4)




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
    fs : int
        Sampling frequency

    Returns
    -------
    float
        Spectral Skewness

    """
    f, fmag = calc_fft(signal, fs)
    spect_centr = spectral_centroid(signal, fs)
    summedFmag = matchLastDimByRepeat(np.sum(fmag, axis=-1), fmag)

    skew = ((f - matchLastDimByRepeat(spect_centr, f)) ** 3) * (fmag / summedFmag)
    return divideWithZero(np.sum(skew, axis=-1), spectral_spread(signal, fs) ** 3)




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
    fs : int
        Sampling frequency

    Returns
    -------
    float
        Spectral Spread

    """
    f, fmag = calc_fft(signal, fs)
    spect_centroid = spectral_centroid(signal, fs)
    helper = (f - matchLastDimByRepeat(spect_centroid, f)) ** 2
    summedFmag = matchLastDimByRepeat(np.sum(fmag, axis=-1), fmag)
    return np.sum(helper * divideWithZero(fmag, summedFmag), axis=-1) ** 0.5



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
    fs : int
        Sampling frequency

    Returns
    -------
    float
        Spectral Slope

    """
    f, fmag = calc_fft(signal, fs)
    num_ = divideWithZero(1, np.sum(fmag, axis=-1)) * (np.ma.size(f, axis=-1) * np.sum(f * fmag, axis=-1) - np.sum(f, axis=-1) * np.sum(fmag, axis=-1))
    denom_ = np.ma.size(f, axis=-1) * np.sum(f * f, axis=-1) - np.sum(f, axis=-1) ** 2
    return divideWithZero(num_, denom_)




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
    fs : int
        Sampling frequency

    Returns
    -------
    float
        Spectral Variation

    """
    _, fmag = calc_fft(signal, fs)

    sum1 = np.sum(fmag[..., :-1] * fmag[..., 1:], axis=-1)
    sum2 = np.sum(fmag[..., 1:] ** 2, axis=-1)
    sum3 = np.sum(fmag[...,:-1] ** 2, axis=-1)

    return 1 - divideWithZero(sum1, ((sum2 ** 0.5) * (sum3 ** 0.5)))



@set_domain("domain", "spectral")
def spectral_positive_turning(signal, fs):
    """Computes number of positive turning points of the fft magnitude signal.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which the number of positive turning points of the fft magnitude are computed
    fs : int
        Sampling frequency

    Returns
    -------
    float
        Number of positive turning points

    """
    return vectorized(fts.spectral_positive_turning, signal)



@set_domain("domain", "spectral")
def spectral_roll_off(signal, fs):
    """Computes the spectral roll-off of the signal.

    The spectral roll-off corresponds to the frequency where 95% of the signal magnitude is contained
    below of this value.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Signal from which spectral roll-off is computed
    fs : int
        Sampling frequency

    Returns
    -------
    float
        Spectral roll-off

    """
    f, fmag = calc_fft(signal, fs)
    cum_fmag = np.cumsum(fmag, axis=-1)
    value = matchLastDimByRepeat(0.95 * np.sum(fmag, axis=-1), cum_fmag)
    ind_mag = np.argmax(np.array(np.asarray(cum_fmag > value)), axis=-1)
    ind_mag = np.expand_dims(ind_mag, axis=-1)
    return np.squeeze(np.take(f, ind_mag), axis=-1)



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
    fs : int
        Sampling frequency

    Returns
    -------
    float
        Spectral roll-on

    """
    f, fmag = calc_fft(signal, fs)
    cum_fmag = np.cumsum(fmag, axis=-1)
    value = matchLastDimByRepeat(0.05 * np.sum(fmag, axis=-1), cum_fmag)
    ind_mag = np.argmax(np.array(np.asarray(cum_fmag >= value)), axis=-1)
    ind_mag = np.expand_dims(ind_mag, axis=-1)
    return np.squeeze(np.take(f, ind_mag), axis=-1)



@set_domain("domain", "spectral")
def human_range_energy(signal, fs):
    """Computes the human range energy ratio.

    The human range energy ratio is given by the ratio between the energy
    in frequency 0.6-2.5Hz and the whole energy band.

    Feature computational cost: 2

    Parameters
    ----------
    signal : nd-array
        Signal from which human range energy ratio is computed
    fs : int
        Sampling frequency

    Returns
    -------
    float
        Human range energy ratio

    """
    f, fmag = calc_fft(signal, fs)

    allenergy = np.sum(fmag ** 2, axis=-1)

    hr_energy = np.sum(fmag[..., np.argmin(
        np.abs(0.6 - f[..., :])):np.argmin(np.abs(2.5 - f[..., :]))] ** 2, axis=-1)

    return divideWithZero(hr_energy, allenergy)



@set_domain("domain", "spectral")
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
    fs : int
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
    return vectorized(fts.mfcc, signal)



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
    fs : int
        Sampling frequency

    Returns
    -------
    float
        Occupied power in bandwidth

    """
    return vectorized(fts.power_bandwidth, signal)



@set_domain("domain", "spectral")
def fft_mean_coeff(signal, fs, nfreq=256):
    """Computes the mean value of each spectrogram frequency.

    nfreq can not be higher than half signal length plus one.
    When it does, it is automatically set to half signal length plus one.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which fft mean coefficients are computed
    fs : int
        Sampling frequency
    nfreq : int
        The number of frequencies

    Returns
    -------
    nd-array
        The mean value of each spectrogram frequency

    """
    nfreq = min(nfreq, np.ma.size(signal, axis=-1) // 2 + 1)

    return np.mean(scipy.signal.spectrogram(signal, fs, nperseg=nfreq * 2 - 2, axis=-1)[2], axis=-1)




@set_domain("domain", "spectral")
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
    return vectorized(fts.lpcc, signal)



@set_domain("domain", "spectral")
def spectral_entropy(signal, fs):
    """Computes the spectral entropy of the signal based on Fourier transform.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which spectral entropy is computed
    fs : int
        Sampling frequency

    Returns
    -------
    float
        The normalized spectral entropy value

    """
    # TODO: this norm is copied form original, but feels weird, check if necessary
    sig = signal - np.expand_dims(np.mean(signal, axis=-1), axis=-1)
    f, fmag = calc_fft(sig, fs)

    power = fmag ** 2
    prob = power / np.expand_dims(np.sum(power, axis=-1), axis=-1)

    return __entropy(prob)



@set_domain("domain", "spectral")
def wavelet_entropy(signal, function=scipy.signal.ricker, widths=np.arange(1, 10)):
    """Computes CWT entropy of the signal.

    Implementation details in:
    https://dsp.stackexchange.com/questions/13055/how-to-calculate-cwt-shannon-entropy
    B.F. Yan, A. Miyamoto, E. Bruhwiler, Wavelet transform-based modal parameter identification considering uncertainty

    Feature computational cost: 2

    Parameters
    ----------
    signal : nd-array
        Input from which CWT is computed
    function :  wavelet function
        Default: scipy.signal.ricker
    widths :  nd-array
        Widths to use for transformation
        Default: np.arange(1,10)

    Returns
    -------
    float
        wavelet entropy

    """
    return vectorized(fts.wavelet_entropy, signal)



@set_domain("domain", "spectral")
def wavelet_abs_mean(signal, function=scipy.signal.ricker, widths=np.arange(1, 10)):
    """Computes CWT absolute mean value of each wavelet scale.

    Feature computational cost: 2

    Parameters
    ----------
    signal : nd-array
        Input from which CWT is computed
    function :  wavelet function
        Default: scipy.signal.ricker
    widths :  nd-array
        Widths to use for transformation
        Default: np.arange(1,10)

    Returns
    -------
    tuple
        CWT absolute mean value

    """
    return vectorized(fts.wavelet_abs_mean, signal)


@set_domain("domain", "spectral")
def wavelet_std(signal, function=scipy.signal.ricker, widths=np.arange(1, 10)):
    """Computes CWT std value of each wavelet scale.

    Feature computational cost: 2

    Parameters
    ----------
    signal : nd-array
        Input from which CWT is computed
    function :  wavelet function
        Default: scipy.signal.ricker
    widths :  nd-array
        Widths to use for transformation
        Default: np.arange(1,10)

    Returns
    -------
    tuple
        CWT std

    """
    return vectorized(fts.negativewavelet_std_turning, signal)


@set_domain("domain", "spectral")
def wavelet_var(signal, function=scipy.signal.ricker, widths=np.arange(1, 10)):
    """Computes CWT variance value of each wavelet scale.

    Feature computational cost: 2

    Parameters
    ----------
    signal : nd-array
        Input from which CWT is computed
    function :  wavelet function
        Default: scipy.signal.ricker
    widths :  nd-array
        Widths to use for transformation
        Default: np.arange(1,10)

    Returns
    -------
    tuple
        CWT variance

    """
    return vectorized(fts.wavelet_var, signal)


@set_domain("domain", "spectral")
def wavelet_energy(signal, function=scipy.signal.ricker, widths=np.arange(1, 10)):
    """Computes CWT energy of each wavelet scale.

    Implementation details:
    https://stackoverflow.com/questions/37659422/energy-for-1-d-wavelet-in-python

    Feature computational cost: 2

    Parameters
    ----------
    signal : nd-array
        Input from which CWT is computed
    function :  wavelet function
        Default: scipy.signal.ricker
    widths :  nd-array
        Widths to use for transformation
        Default: np.arange(1,10)

    Returns
    -------
    tuple
        CWT energy

    """
    return vectorized(fts.wavelet_energy, signal)

