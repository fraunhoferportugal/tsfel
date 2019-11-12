import scipy
import numpy as np
from tsfel.feature_extraction.features_utils import *


def set_domain(key, value):
    def decorate_func(func):
        setattr(func, key, value)
        return func
    return decorate_func

# ############################################# TEMPORAL DOMAIN ##################################################### #


@set_domain("domain", "temporal")
def autocorr(signal):
    """Computes autocorrelation of the signal.

    Parameters
    ----------
    signal : nd-array
        Input from which autocorrelation is computed

    Returns
    -------
    float
        Cross correlation of 1-dimensional sequence

    """

    return float(np.correlate(signal, signal))


@set_domain("domain", "temporal")
def calc_centroid(signal, fs):
    """Computes the centroid along the time axis.

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

    energy = np.array(signal)**2

    t_energy = np.dot(np.array(time), np.array(energy))
    energy_sum = np.sum(energy)

    if energy_sum == 0 or t_energy == 0:
        centroid = 0
    else:
        centroid = t_energy / energy_sum

    return centroid


@set_domain("domain", "temporal")
def minpeaks(signal):
    """Computes number of minimum peaks of the signal.

    Parameters
    ----------
    signal : nd-array
        Input from which minimum number of peaks is counted
    Returns
    -------
    float
        Minimum number of peaks

    """
    diff_sig = np.diff(signal)

    return np.sum([1 for nd in range(len(diff_sig[:-1])) if ((diff_sig[nd] < 0) and (diff_sig[nd + 1] > 0))])


@set_domain("domain", "temporal")
def maxpeaks(signal):
    """Computes number of maximum peaks of the signal.

    Parameters
    ----------
    signal : nd-array
        Input from which maximum number of peaks is counted

    Returns
    -------
    float
        Maximum number of peaks

    """
    diff_sig = np.diff(signal)

    return np.sum([1 for nd in range(len(diff_sig[:-1])) if ((diff_sig[nd+1] < 0) and (diff_sig[nd] > 0))])


@set_domain("domain", "temporal")
def mean_abs_diff(signal):
    """Computes mean absolute differences of the signal.

   Parameters
   ----------
   signal : nd-array
       Input from which mean absolute deviation is computed

   Returns
   -------
   float
       Mean absolute difference result

   """

    return np.mean(abs(np.diff(signal)))


@set_domain("domain", "temporal")
def mean_diff(signal):
    """Computes mean of differences of the signal.

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

   Parameters
   ----------
   signal : nd-array
       Input from which median absolute difference is computed

   Returns
   -------
   float
       Median absolute difference result

   """

    return np.median(abs(np.diff(signal)))


@set_domain("domain", "temporal")
def median_diff(signal):
    """Computes median of differences of the signal.

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
    using the hipotenusa between 2 datapoints.

    Parameters
    ----------
    signal : nd-array
        Input from which distance is computed

    Returns
    -------
    float
        Signal distance

    """
    diff_sig = np.diff(signal)
    return np.sum([np.sqrt(1+df**2) for df in diff_sig])


@set_domain("domain", "temporal")
def sum_abs_diff(signal):
    """Computes sum of absolute differences of the signal.

   Parameters
   ----------
   signal : nd-array
       Input from which sum absolute difference is computed

   Returns
   -------
   float
       Sum absolute difference result

   """

    return np.sum(abs(np.diff(signal)))


@set_domain("domain", "temporal")
def zero_cross(signal):
    """Computes Zero-crossing rate of the signal.

    Corresponds to the total number of times that the signal changes from
    positive to negative or vice versa.

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
def total_energy(signal, fs):
    """Computes the total energy of the signal.

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

    time = compute_time(signal, fs)

    return np.sum(np.array(signal)**2)/(time[-1]-time[0])


@set_domain("domain", "temporal")
def temporal_variance(signal, fs):
    """ Computes the temporal variance of the signal.

    Parameters
    ----------
    signal: nd-array
        Input from which temporal variance is computed
    fs: int
        Sampling frequency

    Returns
    -------
    float
        Temporal variance

    """

    time = compute_time(signal, fs)

    if np.sum(signal) == 0:
        return 0.0

    p = time / np.sum(time)

    return np.sum(p * (signal - np.sum(p * signal)) ** 2)


@set_domain("domain", "temporal")
def temporal_deviation(signal, fs):
    """Computes the temporal standard deviation of the signal.

    Parameters
    ----------
    signal : nd-array
        Input from which temporal deviation is computed
    fs : int
        Sampling frequency

    Returns
    -------
    float
        Temporal standard deviation

    """

    t_var = temporal_variance(signal, fs)

    return np.sqrt(t_var)


# ############################################ STATISTICAL DOMAIN #################################################### #


@set_domain("domain", "statistical")
def hist(signal, nbins=10, r=1):
    """Computes histogram of the signal.

    Parameters
    ----------
    signal : nd-array
        Input from histogram is computed
    nbins : int
        The number of equal-width bins in the givel range
    r : float
        The lower(-r) and upper(r) range of the bins

    Returns
    -------
    nd-array
        The values of the histogram

    """

    histsig, bin_edges = np.histogram(signal, bins=nbins, range=[-r, r])  # TODO:subsampling parameter

    return tuple(histsig)


@set_domain("domain", "statistical")
def interq_range(signal):
    """Computes interquartile range of the signal.

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
def calc_mean(signal):
    """Computes mean value of the signal.

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

    Parameters
    ----------
    signal : nd-array
        Input from which mean absolute deviation is computed

    Returns
    -------
    float
        Mean absolute deviation result

    """

    return np.mean(abs(signal-np.mean(signal, axis=0)), axis=0)


@set_domain("domain", "statistical")
def median_abs_deviation(signal):
    """Computes median absolute deviation of the signal.

    Parameters
    ----------
    signal : nd-array
        Input from which median absolute deviation is computed

    Returns
    -------
    float
        Mean absolute deviation result

    """

    return scipy.stats.median_absolute_deviation(signal, scale=1)


@set_domain("domain", "statistical")
def rms(signal):
    """Computes root mean square of the signal.

    Square root of the arithmetic mean (average) of the squares of the original values.

    Parameters
    ----------
    signal : nd-array
        Input from which root mean square is computed

    Returns
    -------
    float
        Root mean square

    """

    return np.sqrt(np.sum(np.array(signal)**2)/len(signal))


@set_domain("domain", "statistical")
def calc_std(signal):
    """Computes standard deviation (std) of the signal.

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
def slope(signal):
    """Computes the slope of the signal.

    Slope is computed by fitting a linear equation to the observed data.

    Parameters
    ----------
    signal : nd-array
        Input from which linear equation is computed

    Returns
    -------
    float
        Slope

    """
    t = np.linspace(0, len(signal)-1, len(signal))

    return np.polyfit(t, signal, 1)[0]


# ############################################## SPECTRAL DOMAIN ##################################################### #

@set_domain("domain", "spectral")
def spectral_distance(signal, fs):
    """Computes the signal spectral distance.

    Distance of the signal's cumulative sum of the FFT elements to
    the respective linear regression.

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

    f, fmag = calc_fft(signal, fs)

    cum_fmag = np.cumsum(fmag)

    # Computing the linear regression
    points_y = np.linspace(0, cum_fmag[-1], len(cum_fmag))

    return np.sum(points_y-cum_fmag)


@set_domain("domain", "spectral")
def fundamental_frequency(signal, fs):
    """Computes fundamental frequency of the signal.

    The fundamental frequency integer multiple best explain
    the content of the signal spectrum.

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

    signal = signal - np.mean(signal)
    f, fmag = calc_fft(signal, fs)

    # Condition for offset removal, since the offset generates a peak at frequency zero
    try:
        # With 0.1 the offset frequency is discarded
        cond = np.where(f > 0.1)[0][0]
    except IndexError:
        cond = 0

    # Finding big peaks, not considering noise peaks with low amplitude
    bp = scipy.signal.find_peaks(fmag[cond:], threshold=10)[0]
    if not list(bp):
        f0 = 0
    else:
        # add cond for correcting the indices position
        bp = bp + cond
        # f0 is the minimum big peak frequency
        f0 = f[min(bp)]

    return f0


@set_domain("domain", "spectral")
def max_power_spectrum(signal, fs):
    """Computes maximum power spectrum density of the signal.

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

    if np.std(signal) == 0:
        return float(max(scipy.signal.welch(signal, int(fs), nperseg=len(signal))[1]))
    else:
        return float(max(scipy.signal.welch(signal/np.std(signal), int(fs), nperseg=len(signal))[1]))


@set_domain("domain", "spectral")
def max_frequency(signal, fs):
    """Computes maximum frequency of the signal.

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
    cum_fmag = np.cumsum(fmag)

    try:
        ind_mag = np.where(cum_fmag > cum_fmag[-1]*0.95)[0][0]
    except IndexError:
        ind_mag = np.argmax(cum_fmag)

    return f[ind_mag]


@set_domain("domain", "spectral")
def median_frequency(signal, fs):
    """Computes median frequency of the signal.

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
def spectral_centroid(signal, fs):
    """Barycenter of the spectrum.

    Description and formula in Article:
    The Timbre Toolbox: Extracting audio descriptors from musicalsignals
    Authors Peeters G., Giordano B., Misdariis P., McAdams S.

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
        return np.dot(f, fmag/np.sum(fmag))


@set_domain("domain", "spectral")
def spectral_decrease(signal, fs):
    """Represents the amount of decreasing of the spectra amplitude.

    Description and formula in Article:
    The Timbre Toolbox: Extracting audio descriptors from musicalsignals
    Authors Peeters G., Giordano B., Misdariis P., McAdams S.

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

    f, fmag = calc_fft(signal, fs)

    fmag_band = fmag[1:]
    len_fmag_band = np.arange(2, len(fmag)+1)

    # Sum of numerator
    soma_num = np.sum((fmag_band-fmag[0])/(len_fmag_band-1), axis=0)

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

    if not np.sum(fmag):
        return 0
    else:
        return np.dot(((f-spect_centroid)**2), (fmag / np.sum(fmag)))**0.5


@set_domain("domain", "spectral")
def spectral_slope(signal, fs):
    """Computes the spectral slope.

    Spectral slope is computed by finding constants m and b of the function aFFT = mf + b, obtained by linear regression
    of the spectral amplitude.

    Description and formula in Article:
    The Timbre Toolbox: Extracting audio descriptors from musicalsignals
    Authors Peeters G., Giordano B., Misdariis P., McAdams S.

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

    if not(list(f)) or (np.sum(fmag) == 0):
        return 0
    else:
        if not (len(f) * np.dot(f, f) - np.sum(f) ** 2):
            return 0
        else:
            num_ = (1/np.sum(fmag))*(len(f) * np.dot(f, fmag) - np.sum(f) * np.sum(fmag))
            denom_ = (len(f) * np.dot(f, f) - np.sum(f) ** 2)
            return num_ / denom_


@set_domain("domain", "spectral")
def spectral_variation(signal, fs):
    """Computes the amount of variation of the spectrum along time.

    Spectral variation is computed from the normalized cross-correlation between two consecutive amplitude spectra.

    Description and formula in Article:
    The Timbre Toolbox: Extracting audio descriptors from musicalsignals
    Authors Peeters G., Giordano B., Misdariis P., McAdams S.

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

    f, fmag = calc_fft(signal, fs)

    sum1 = np.sum(np.array(fmag)[:-1]*np.array(fmag)[1:])
    sum2 = np.sum(np.array(fmag)[1:]**2)
    sum3 = np.sum(np.array(fmag)[:-1]**2)

    if not sum2 or not sum3:
        variation = 1
    else:
        variation = 1-(sum1/((sum2**0.5)*(sum3**0.5)))

    return variation


@set_domain("domain", "spectral")
def spectral_maxpeaks(signal, fs):
    """Computes number of maximum spectral peaks of the signal.

    Parameters
    ----------
    signal : nd-array
        Input from which the number of maximum spectral peaks is computed
    fs : int
        Sampling frequency

    Returns
    -------
    float
        Total number of maximum spectral peaks

    """

    f, fmag = calc_fft(signal, fs)
    diff_sig = np.diff(fmag)

    return np.sum([1 for nd in range(len(diff_sig[:-1])) if ((diff_sig[nd+1] < 0) and (diff_sig[nd] > 0))])


@set_domain("domain", "spectral")
def spectral_roll_off(signal, fs):
    """Computes the spectral roll-off of the signal.

    The spectral roll-off corresponds to the frequency where 95% of the signal magnitude is contained
    below of this value.

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
    cum_ff = np.cumsum(fmag)
    value = 0.95*(sum(fmag))

    return f[np.where(cum_ff >= value)[0][0]]


@set_domain("domain", "spectral")
def spectral_roll_on(signal, fs):
    """Computes the spectral roll-on of the signal.

    The spectral roll-on corresponds to the frequency where 5% of the signal magnitude is contained
    below of this value.

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
    cum_ff = np.cumsum(fmag)
    value = 0.05*(sum(fmag))

    return f[np.where(cum_ff >= value)[0][0]]


@set_domain("domain", "spectral")
def power_bandwidth(signal, fs):
    """Computes power spectrum density bandwidth of the signal.

    It corresponds to the width of the frequency band in which 95% of its power is located.

    Description in article:
    Power Spectrum and Bandwidth Ulf Henriksson, 2003 Translated by Mikael Olofsson, 2005

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

    # Computing the power spectrum density
    if np.std(signal) == 0:
        freq, power = scipy.signal.welch(signal, fs, nperseg=len(signal))
    else:
        freq, power = scipy.signal.welch(signal/np.std(signal), fs, nperseg=len(signal))

    if np.sum(power) == 0:
        return 0.0

    # Computing the lower and upper limits of power bandwidth
    cum_power = np.cumsum(power)
    f_lower = freq[np.where(cum_power >= cum_power[-1]*0.95)[0][0]]

    cum_power_inv = np.cumsum(power[::-1])
    f_upper = freq[abs(np.where(cum_power_inv >= cum_power[-1]*0.95)[0][0]-len(power)+1)]

    # Returning the bandwidth in terms of frequency

    return abs(f_upper-f_lower)


@set_domain("domain", "spectral")
def human_range_energy(signal, fs):
    """Computes the human range energy ratio.

    The human range energy ratio is given by the ratio between the energy
    in frequency 0.6-2.5Hz and the whole energy band.

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

    f, fmag = calc_fft(signal,fs)

    allenergy = np.sum(fmag**2)

    if allenergy == 0:
        # For handling the occurrence of Nan values
        return 0.0

    hr_energy = np.sum(fmag[np.argmin(abs(0.6 - f)):np.argmin(abs(2.5 - f))] ** 2)

    ratio = hr_energy/allenergy

    return ratio


@set_domain("domain", "spectral")
def hist_fft(signal, fs, nbins=10, r=1):
    """Computes the histogram of the fft values.

    The human range energy ratio is given by the ratio between the energy
    in frequency 0.6-2.5Hz and the whole energy band.

    Parameters
    ----------
    signal : nd-array
        Input from histogram is computed
    fs : int
        Sampling frequency
    nbins : int
        The number of equal-width bins in the given range
    r : float
        The lower(-r) and upper(r) range of the bins

    Returns
    -------
    nd-array
        The values of the histogram for fft magnitude

    """

    f, fmag = calc_fft(signal, fs)

    return hist(fmag, nbins, r)


@set_domain("domain", "spectral")
def lpcc(signal, n_coeff=12):
    """Computes the linear prediction cepstral coefficients.

    Implementation details and description in:
    http://www.practicalcryptography.com/miscellaneous/machine-learning/tutorial-cepstrum-and-lpccs/

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
        return tuple(np.zeros(n_coeff))

    # Power spectrum
    powerspectrum = np.abs(np.fft.fft(lpc_coeffs)) ** 2
    lpcc_coeff = np.fft.ifft(np.log(powerspectrum))

    return tuple(abs(lpcc_coeff))


@set_domain("domain", "spectral")
def mfcc(signal, fs, pre_emphasis=0.97, nfft=512, nfilt=40, num_ceps=12, cep_lifter=22):
    """Computes the MEL cepstral coefficients.

    It provides the information about the power in each frequency band.

    Implementation details and description on:
    https://www.kaggle.com/ilyamich/mfcc-implementation-and-tutorial
    https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html#fnref:1

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

    filter_banks = filterbank(signal, fs, pre_emphasis, nfft, nfilt)

    mel_coeff = scipy.fftpack.dct(filter_banks, type=2, axis=0, norm='ortho')[1:(num_ceps+1)]  # Keep 2-13

    mel_coeff -= (np.mean(mel_coeff, axis=0) + 1e-8)

    # liftering
    ncoeff = len(mel_coeff)
    n = np.arange(ncoeff)
    lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)  # cep_lifter = 22 from python_speech_features library

    mel_coeff *= lift

    return tuple(mel_coeff)
