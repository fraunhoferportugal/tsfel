import novainstrumentation as ni
import numpy as np
import scipy


def set_domain(key, value):
    def decorate_func(func):
        setattr(func, key, value)
        return func
    return decorate_func

# ####################################  TEMPORAL DOMAIN  ############################################################# #


def _compute_time(sign, fs):
    """Creates the signal correspondent time array.
    """
    time = range(len(sign))
    time = [float(x)/fs for x in time]
    return time


def _signal_energy(sign, time):
    """Computes the energy of the signal. For that, first is made the segmentation of the signal in 10 windows
    and after it's considered that the energy of the signal is the sum of all calculated points in each window.
    Parameters
    ----------
    sign: ndarray
        input from which max frequency is computed.
    Returns
    -------
    energy: float list
       signal energy.
    time_energy: float list
        signal time energy
    """

    window_len = len(sign)

    # window for energy calculation
    if window_len < 10:
        window = 1
    else:
        window = window_len//10 # each window of the total signal will have 10 windows

    energy = np.zeros(window_len//window)
    time_energy = np.zeros(window_len//window)

    i = 0
    for a in range(0, len(sign) - window, window):
        energy[i] = np.sum(np.array(sign[a:a+window])**2)
        interval_time = time[int(a+(window//2))]
        time_energy[i] = interval_time
        i += 1

    return list(energy), list(time_energy)


@set_domain("domain", "temporal")
def autocorr(sig):
    """Compute autocorrelation along the specified axis.

    Parameters
    ----------
    sig: ndarray
        input from which autocorrelation is computed.

    Returns
    -------
    corr: float
        Cross correlation of 1-dimensional sequence.
    """
    return float(np.correlate(sig, sig))


@set_domain("domain", "temporal")
def centroid(sign, fs):
    """Computes the centroid along the time axis.
    ----------
    sign: ndarray
        input from which max frequency is computed.
    fs: int
        signal sampling frequency.
    Returns
    -------
    centroid: float
        temporal centroid
    """

    time = _compute_time(sign, fs)

    energy, time_energy = _signal_energy(sign, time)

    total_energy = np.dot(np.array(time_energy),np.array(energy))
    energy_sum = np.sum(energy)

    if energy_sum == 0 or total_energy == 0:
        centroid = 0
    else:
        centroid = total_energy / energy_sum
    return centroid


@set_domain("domain", "temporal")
def minpeaks(sig):
    """Compute number of minimum peaks along the specified axes.

    Parameters
    ----------
    sig: ndarray

    Returns
    -------
     float
        min number of peaks

    """
    diff_sig = np.diff(sig)

    return np.sum([1 for nd in range(len(diff_sig[:-1])) if (diff_sig[nd]<0 and diff_sig[nd + 1]>0)])


@set_domain("domain", "temporal")
def calc_meanadiff(sig):
    """Compute mean absolute differences along the specified axes.

   Parameters
   ----------
   input: ndarray
       input from which mean absolute deviation is computed.

   Returns
   -------
   mad: int
      mean absolute difference result.
   """

    return np.mean(abs(np.diff(sig)))


@set_domain("domain", "temporal")
def calc_meandiff(sig):
    """Compute mean of differences along the specified axes.

   Parameters
   ----------
   input: ndarray
       input from which mean absolute deviation is computed.

   Returns
   -------
   mad: int
      mean absolute difference result.
   """

    return np.mean(np.diff(sig))


@set_domain("domain", "temporal")
def calc_medadiff(sig):
    """Compute median absolute differences along the specified axes.

   Parameters
   ----------
   input: ndarray
       input from which mean absolute deviation is computed.

   Returns
   -------
   mad: int
      mean absolute difference result.
   """

    return np.median(abs(np.diff(sig)))


@set_domain("domain", "temporal")
def calc_meddiff(sig):
    """Compute mean of differences along the specified axes.

   Parameters
   ----------
   input: ndarray
       input from which mean absolute deviation is computed.

   Returns
   -------
   mad: int
      mean absolute difference result.
   """

    return np.median(np.diff(sig))


@set_domain("domain", "temporal")
def calc_medad(sig):
    """Compute mean absolute deviation along the specified axes.
   Parameters
   ----------
   input: ndarray
       input from which mean absolute deviation is computed.
   Returns
   -------
   mad: int
      mean absolute deviation result.
   """
    m = np.median(sig)
    diff = [abs(x - m) for x in sig]

    return np.median(diff)


@set_domain("domain", "temporal")
def maxpeaks(sig):
    """Compute number of peaks along the specified axes.

    Parameters
    ----------
    sig: ndarray
        input from histogram is computed.
    type: string
        can be 'all', 'max', and 'min', and expresses which peaks are going to be accounted
    Returns
    -------
    num_p: float
        total number of peaks

    """
    diff_sig = np.diff(sig)

    return np.sum([1 for nd in range(len(diff_sig[:-1])) if (diff_sig[nd+1]<0 and diff_sig[nd]>0)])


@set_domain("domain", "temporal")
def distance(sig):
    """ Calculates the total distance traveled by the signal,
    using the hipotenusa between 2 datapoints
    Parameters
    ----------
    s: array-like
      the input signal.
    Returns
    -------
    signal distance: if the signal was straightened distance
    """
    df_sig = np.diff(sig)
    return np.sum([np.sqrt(1+df**2) for df in df_sig])


@set_domain("domain", "temporal")
def calc_sadiff(sig):
    """Compute sum of absolute differences along the specified axes.

   Parameters
   ----------
   input: ndarray
       input from which sum absolute diff is computed.

   Returns
   -------
   mad: int
      sum absolute difference result.
   """

    return np.sum(abs(np.diff(sig)))


@set_domain("domain", "temporal")
def zero_cross(sig):
    """Compute Zero-crossing rate along the specified axis.
         total number of times that the signal changes from positive to negative or vice versa, normalized by the window length.
    Parameters
    ----------
    sig: ndarray
        input from which the zero-crossing rate are computed.

    Returns
    -------
    count_vector: int
        number of times that signal value cross the zero axe.
    """
    #return np.where(ny.diff(ny.sign(sig)))[0]

    return len(np.where(np.diff(np.sign(sig)))[0])


# ###################################### STATISTICAL DOMAIN #############################################################


@set_domain("domain", "statistical")
def hist(sig, nbins=10, r=1):
    """Compute histogram along the specified axes.

    Parameters
    ----------
    sig: ndarray
        input from histogram is computed.
    nbins: int
     the number of equal-width bins in the givel range.
    rang: float
        the lower(-r) and upper(r) range of the bins.
    Returns
    -------
    histsig: ndarray
        the values of the histogram

    bin_edges: ndarray
        the bin_edges, 'len(hist)+1'

    """

    histsig, bin_edges = np.histogram(sig, bins=nbins, range=[-r, r], density=True)  # TODO:subsampling parameter

    # bin_edges = bin_edges[:-1]
    # bin_edges += (bin_edges[1]-bin_edges[0])/2.

    return tuple(histsig)


@set_domain("domain", "statistical")
def calc_iqr(sig):
     """Compute interquartile range along the specified axes.

    Parameters
    ----------
    input: ndarray
        input from which interquartile range is computed.

    Returns
    -------
    iqr: int
       interquartile range result.
    """

     return np.percentile(sig, 75) - np.percentile(sig, 25)


@set_domain("domain", "statistical")
def calc_kurtosis(sig):
     """Compute kurtosis along the specified axes.

    Parameters
    ----------
    input: ndarray
        input from which kurtosis is computed.

    Returns
    -------
    k: int
       kurtosis result.
    """
     return scipy.stats.kurtosis(sig)


@set_domain("domain", "statistical")
def calc_skewness(sig):
     """Compute skewness along the specified axes.

    Parameters
    ----------
    input: ndarray
        input from which skewness is computed.

    Returns
    -------
    s: int
       skewness result.
    """
     return scipy.stats.skew(sig)


@set_domain("domain", "statistical")
def calc_max(sig):
    """Compute max (max) along the specified axes.

   Parameters
   ----------
   input: ndarray
       input from which max is computed.

   Returns
   -------
   max_value: int
      max result.
   """
    return np.max(sig)


@set_domain("domain", "statistical")
def calc_mean(sig):
     """Compute mean value along the specified axes.

    Parameters
    ----------
    input: ndarray
        input from which mean is computed.

    Returns
    -------
    m: int
       mean result.
    """
     return np.mean(sig)


@set_domain("domain", "statistical")
def calc_meanad(sig):
     """Compute mean absolute deviation along the specified axes.

    Parameters
    ----------
    input: ndarray
        input from which mean absolute deviation is computed.

    Returns
    -------
    mad: int
       mean absolute deviation result.
    """
     m = np.mean(sig)
     diff = [abs(x-m) for x in sig]

     return np.mean(diff)


@set_domain("domain", "statistical")
def calc_median(sig):
    """Compute median (median) along the specified axes.

   Parameters
   ----------
   input: ndarray
       input from which median is computed.

   Returns
   -------
   median_value: int
      median result.
   """
    return np.median(sig)


@set_domain("domain", "statistical")
def calc_min(sig):
    """Compute min (min) along the specified axes.

   Parameters
   ----------
   input: ndarray
       input from which min is computed.

   Returns
   -------
   min_value: int
      min result.
   """
    return np.min(sig)


@set_domain("domain", "statistical")
def rms(sig):
     """Compute root mean square along the specified axes.

    Parameters
    ----------
    input: ndarray
        input from which root mean square is computed.

    Returns
    -------
    rms: int
       square root of the arithmetic mean (average) of the squares of the original values.
    """

     return np.sqrt(np.sum(np.array(sig)**2)/len(sig))


@set_domain("domain", "statistical")
def calc_std(sig):
     """Compute standard deviation (std) along the specified axes.

    Parameters
    ----------
    input: ndarray
        input from which std is computed.

    Returns
    -------
    std_value: int
       std result.
    """
     return np.std(sig)


@set_domain("domain", "statistical")
def calc_var(sig):
    """Compute variance (var) along the specified axes.

   Parameters
   ----------
   input: ndarray
       input from which var is computed.

   Returns
   -------
   var_value: int
      var result.
   """
    return np.var(sig)


# ############################################ SPECTRAL DOMAIN ####################################################### #


def _plotfft(s, fmax):
    """ This functions computes the fft of a signal, returning the frequency
    and their magnitude values.
    Parameters
    ----------
    s: array-like
      the input signal.
    fmax: int
      the sampling frequency.
    doplot: boolean
      a variable to indicate whether the plot is done or not.
    Returns
    -------
    f: array-like
      the frequency values (xx axis)
    fs: array-like
      the amplitude of the frequency values (yy axis)
    """

    fs = abs(np.fft.fft(s))
    f = np.linspace(0, fmax // 2, len(s) // 2)
    return (f[1:len(s) // 2].copy(), fs[1:len(s) // 2].copy())


def _big_peaks(s, th, min_peak_distance=5, peak_return_percentage=0.1):
    pp = []
    if not list(s):
        return pp
    else:
        p = ni.peaks(s, th)
        if not list(p):
            return pp
        else:
            p = ni.clean_near_peaks(s, p, min_peak_distance)
            if not list(p):
                return pp
            else:
                ars = np.argsort(s[p])
                pp = p[ars]

                num_peaks_to_return = int(np.ceil(len(p) * peak_return_percentage))

                pp = pp[-num_peaks_to_return:]

            return pp


@set_domain("domain", "spectral")
def curve_distance(sign, fs):
    """Euclidean distance of the signal's cumulative sum of the FFT elements to the respective linear regression.

    Parameters
    ----------
    sign: ndarray
        signal from which spectral slope is computed.
    fs: int
        sampling frequency of the signal
    Returns
    -------
    curve distance: float
        curve distance
    """
    f, ff = _plotfft(sign, fs)
    cum_ff = np.cumsum(ff)
    points_y = np.linspace(0, cum_ff[-1], len(cum_ff))

    return np.sum(points_y-cum_ff)


@set_domain("domain", "spectral")
def fundamental_frequency(s, fs):
    # TODO: review fundamental frequency to guarantee that f0 exists
    # suggestion peak level should be bigger
    # TODO: explain code
    """Compute fundamental frequency along the specified axes.
    Parameters
    ----------
    s: ndarray
        input from which fundamental frequency is computed.
    FS: int
        sampling frequency
    Returns
    -------
    f0: int
       its integer multiple best explain the content of the signal spectrum.
    """

    s = s - np.mean(s)
    f, _fs = _plotfft(s, fs)

    _fs = _fs[1:len(_fs) // 2]
    f = f[1:len(f) // 2]
    try:
        cond = np.where(f > 0.5)[0][0]
    except:
        cond = 0

    bp = _big_peaks(_fs[cond:], 0)
    if not list(bp):
        f0 = 0
    else:
        bp = bp + cond
        f0 = f[min(bp)]
    return f0


@set_domain("domain", "spectral")
def linear_regression(sign):
    """fits a liner equation to the observed data

    Parameters
    ----------
    sig: ndarray
        input from histogram is computed.
    fs: int
        sampling frequency of the signal
    Returns
    -------
    num_p: float
        slope

    """
    t = np.linspace(0, 5, len(sign))

    return np.polyfit(t, sign, 1)[0]


@set_domain("domain", "spectral")
def max_power_spectrum(sig, fs):
    """Compute power spectrum density along the specified axes.

    Parameters
    ----------
    sig: ndarray
        input from which cepstral coefficients are computed.
    FS: scalar
        sampling frequency
    Returns
    -------
    max_power: ndarray
        max value of the power spectrum.
    peak_freq: ndarray
        max frequency corresponding to the elements in power spectrum.

    """

    if np.std(sig) == 0:
        return float(max(scipy.signal.welch(sig, int(fs), nperseg=len(sig))[1]))
    else:
        return float(max(scipy.signal.welch(sig/np.std(sig), int(fs), nperseg=len(sig))[1]))


@set_domain("domain", "spectral")
def max_frequency(sig, fs):
    """Compute max frequency along the specified axes.

    Parameters
    ----------
    sig: ndarray
        input from which max frequency is computed.
    FS: int
        sampling frequency
    Returns
    -------
    f_max: int
       0.95 of max_frequency using cumsum.
    """

    f, _fs = _plotfft(sig, fs)
    t = np.cumsum(_fs)

    try:
        ind_mag = np.where(t > t[-1]*0.95)[0][0]
    except:
        ind_mag = np.argmax(t)
    f_max = f[ind_mag]

    return f_max


@set_domain("domain", "spectral")
def median_frequency(sig, fs):
    """Compute median frequency along the specified axes.
    Parameters
    ----------
    sig: ndarray
        input from which median frequency is computed.
    FS: int
        sampling frequency
    Returns
    -------
    f_max: int
       0.50 of max_frequency using cumsum.
    """

    f, _fs = _plotfft(sig, fs)
    t = np.cumsum(_fs)
    try:
        ind_mag = np.where(t > t[-1] * 0.50)[0][0]
    except:
        ind_mag = np.argmax(t)
    f_median = f[ind_mag]

    return f_median


@set_domain("domain", "spectral")
def spectral_centroid(sign, fs):
    """Barycenter of the spectrum.

    Parameters
    ----------
    sign: ndarray
        signal from which spectral slope is computed.
    fs: int
        sampling frequency of the signal
    Returns
    -------
    spread: float
        centroid
    """
    f, ff = _plotfft(sign, fs)
    if not np.sum(ff):
        return 0
    else:
        return np.dot(f, ff/np.sum(ff))


@set_domain("domain", "spectral")
def spectral_decrease(sign, fs):
    """Represents the amount of decraesing of the spectra amplitude.

    Parameters
    ----------
    sign: ndarray
        signal from which spectral slope is computed.
    fs: int
        sampling frequency of the signal
    Returns
    -------
        spectrak decrease
    """
    f, ff = _plotfft(sign, fs)

    k = len(ff)
    soma_num = 0
    for a in range(2, k):
        soma_num = soma_num + ((ff[a]-ff[1])/(a-1))

    ff2 = ff[2:]
    if not np.sum(ff2):
        return 0
    else:
        soma_den = 1 / np.sum(ff2)
        return soma_den * soma_num


@set_domain("domain", "spectral")
def spectral_kurtosis(sign, fs):
    """Measures the flatness of a distribution around its mean value. Computed from the 4th order moment.

    Parameters
    ----------
    sign: ndarray
        signal from which spectral slope is computed.
    fs: int
        sampling frequency of the signal
    Returns
    -------
    kurtosis: float
        kurtosis
    """
    f, ff = _plotfft(sign, fs)
    if not spectral_spread(sign, fs):
        return 0
    else:
        spect_kurt = ((f - spectral_centroid(sign, fs)) ** 4) * (ff / np.sum(ff))
        return np.sum(spect_kurt) / (spectral_spread(sign, fs) ** 2)


@set_domain("domain", "spectral")
def spectral_maxpeaks(sign, fs):
    """Compute number of peaks along the specified axes.

    Parameters
    ----------
    sig: ndarray
        input from histogram is computed.
    fs: int
        sampling frequency of the signal
    Returns
    -------
    num_p: float
        total number of peaks

    """
    f, ff = _plotfft(sign, fs)
    diff_sig = np.diff(ff)

    return np.sum([1 for nd in range(len(diff_sig[:-1])) if (diff_sig[nd+1]<0 and diff_sig[nd]>0)])


@set_domain("domain", "spectral")
def spectral_roll_off(sign, fs):
    """Compute the spectral roll-off of the signal, i.e., the frequency where 95% of the signal energy is contained
    below of this value.

    Parameters
    ----------
    sign: ndarray
        signal from which spectral slope is computed.
    fs: int
        sampling frequency of the signal
    Returns
    -------
    roll_off: float
        spectral roll-off
    """
    output = 0
    f, ff = _plotfft(sign, fs)
    cum_ff = np.cumsum(ff)
    value = 0.95*(sum(ff))

    for i in range(len(ff)):
        if cum_ff[i] >= value:
            output = f[i]
            break
    return output


@set_domain("domain", "spectral")
def spectral_roll_on(sign, fs):
    """Compute the spectral roll-on of the signal, i.e., the frequency where 5% of the signal energy is contained
    below of this value.

    Parameters
    ----------
    sign: ndarray
        signal from which spectral slope is computed.
    fs: int
        sampling frequency of the signal
    Returns
    -------
    roll_off: float
        spectral roll-on
    """
    output = 0
    f, ff = _plotfft(sign, fs)
    cum_ff = np.cumsum(ff)
    value = 0.05*(sum(ff))

    for i in range(len(ff)):
        if cum_ff[i] >= value:
            output = f[i]
            break
    return output


@set_domain("domain", "spectral")
def spectral_skewness(sign, fs):
    """Measures the asymmetry of a distribution around its mean value. Computed from the 3rd order moment.

    Parameters
    ----------
    sign: ndarray
        signal from which spectral slope is computed.
    fs: int
        sampling frequency of the signal
    Returns
    -------
    skeness: float
        skeness
    """
    f, ff = _plotfft(sign, fs)
    spect_centr = spectral_centroid(sign, fs)
    if not spectral_spread(sign, fs):
        return 0
    else:
        skew = ((f - spect_centr) ** 3) * (ff / np.sum(ff))
        return np.sum(skew) / (spectral_spread(sign, fs) ** (3 / 2))


@set_domain("domain", "spectral")
def spectral_slope(sign, fs):
    """Computes the constants m and b of the function aFFT = mf + b, obtained by linear regression of the
    spectral amplitude.

    Parameters
    ----------
    sign: ndarray
        signal from which spectral slope is computed.
    fs: int
        sampling frequency of the signal
    Returns
    -------
    m: float
        slope
    b: float
        y-intercept
    """
    f, ff = _plotfft(sign, fs)
    if not(list(f)):
        return 0
    else:
        if not (len(f) * np.dot(f, f) - np.sum(f) ** 2):
            return 0
        else:
            return (len(f) * np.dot(f, ff) - np.sum(f) * np.sum(ff)) / (len(f) * np.dot(f, f) - np.sum(f) ** 2)


@set_domain("domain", "spectral")
def spectral_spread(sign, fs):
    """Measures the spread of the spectrum around its mean value.

    Parameters
    ----------
    sign: ndarray
        signal from which spectral slope is computed.
    fs: int
        sampling frequency of the signal
    Returns
    -------
    spread: float
        spread
    """
    f, ff = _plotfft(sign, fs)
    spect_centr = spectral_centroid(sign, fs)
    if not np.sum(ff):
        return 0
    else:
        return np.dot(((f-spect_centr)**2), (ff / np.sum(ff)))


@set_domain("domain", "spectral")
def spectral_variation(sign, fs):
    """Amount of variation of the spectrum along time. Computed from the normalized cross-correlation between two consecutive amplitude spectra.

    Parameters
    ----------
    sign: ndarray
        signal from which spectral slope is computed.
    fs: int
        sampling frequency of the signal
    Returns
    -------
    variation: float

    """
    f, ff = _plotfft(sign, fs)
    energy, freq = _signal_energy(ff, f)

    sum1 = 0
    sum2 = 0
    sum3 = 0
    for a in range(len(energy)-1):
        sum1 = sum1+(energy[a-1]*energy[a])
        sum2 = sum2+(energy[a-1]**2)
        sum3 = sum3+(energy[a]**2)

    if not sum2 or not sum3:
        variation = 1
    else:
        variation = 1-((sum1)/((sum2**0.5)*(sum3**0.5)))

    return variation


@set_domain("domain", "spectral")
def total_energy(sign, fs):
    """
    Compute the acc_total power, using the given windowSize and value time in samples

    """
    time = _compute_time(sign, fs)

    return np.sum(np.array(sign)**2)/(time[-1]-time[0])


# TODO unit tests for the following features
def variance(sign, FS):
    """ Measures how far the numbers are spread out.

    Parameters
    ----------
    sig: ndarray
        input from histogram is computed.
    fs: int
        sampling frequency of the signal
    Returns
    -------
    num_p: float
        signal variance

    """
    time = _compute_time(sign, FS)
    soma_den = 0
    soma_num = 0
    for z in range(0, len(sign)):
        soma_num = soma_num + (time[z]*((sign[z]*np.mean(sign))**2))
        soma_den = soma_den + time[z]

    return soma_num/soma_den


def deviation(sign, FS):
    """Temporal deviation.

    Parameters
    ----------
    sig: ndarray
        input from histogram is computed.
    fs: int
        sampling frequency of the signal
    Returns
    -------
    deviation: float
        temporal deviation

    """
    time = _compute_time(sign, FS)
    soma_den = 0
    soma_num = 0
    for z in range(0, len(sign)-1):
        soma_num = soma_num+(time[z+1]*(sign[z+1]-sign[z]))
        soma_den = soma_den+time[z+1]

    return soma_num/soma_den


def ceps_coeff(sig, coefNumber):
    """Compute cepstral coefficients along the specified axes.

    Parameters
    ----------
    sig: ndarray
        input from which cepstral coefficients are computed.
    coefNumber:
    Returns
    -------
    cc: ndarray

    """

    est = lpc_coef(sig, coefNumber)
    cc = lpcar2cc(est)
    if len(cc) == 1:
        cc = float(cc)
    else:
        cc = tuple(cc)

    return cc


def power_bandwidth(sig, FS, samples):
    """Compute power spectrum density bandwidth along the specified axes.

    Parameters
    ----------
    sig: ndarray
        input from which cepstral coefficients are computed.
    FS: scalar
        sampling frequency
    samples: int
        number of bands
    Returns
    -------
    bandwidth: ndarray
        power in bandwidth
    """
    bd = []
    bdd = []
    freq, power = scipy.signal.welch(sig/np.std(sig), FS, nperseg=len(sig))

    for i in range(len(power)):
        bd += [float(power[i])]

    bdd += bd[:samples]

    return tuple(bdd)


def trfbank(fs, nfft, lowfreq, linsc, logsc, nlinfilt, nlogfilt):
    """Compute triangular filterbank for MFCC computation."""
    # Total number of filters
    nfilt = nlinfilt + nlogfilt

    #------------------------
    # Compute the filter bank
    #------------------------
    # Compute start/middle/end points of the triangular filters in spectral domain
    freqs = np.zeros(nfilt+2) #modified
    freqs[:nlinfilt] = lowfreq + np.arange(nlinfilt) * linsc
    freqs[nlinfilt:] = freqs[nlinfilt-1] * logsc ** np.arange(1, nlogfilt + 3)

    heights = 2./(freqs[2:] - freqs[0:-2])

    # Compute filterbank coeff (in fft domain, in bins)
    fbank = np.zeros((nfilt, nfft))

    # FFT bins (in Hz)
    nfreqs = np.arange(nfft) / (1. * nfft) * fs
    for i in range(nfilt):
        low = freqs[i]
        cen = freqs[i+1] #modified
        hi = freqs[i+2] #modified

        lid = np.arange(np.floor(low * nfft / fs) + 1,
                        np.floor(cen * nfft / fs) + 1, dtype=int)
        lslope = heights[i] / (cen - low)
        rid = np.arange(np.floor(cen * nfft / fs) + 1,
                        np.floor(hi * nfft / fs) + 1, dtype=int)
        rslope = heights[i] / (hi - cen)
        fbank[i][lid] = lslope * (nfreqs[lid] - low)
        fbank[i][rid] = rslope * (hi - nfreqs[rid])

    return fbank, freqs


def mfcc(sign, fs, num_coeffs=26):
    """

    :param sign:
    :param fs:
    :param num_coeffs:
    :return:
    """

    f, ff = _plotfft(sign, fs)

    sp = np.square(ff) / len(f)

    mel_freqs = np.linspace(0, 2595 * np.log10(1 + f[-1] / 700.), num_coeffs + 2)
    freqs = 700 * (np.power(10, (mel_freqs / 2595)) - 1)

    indexes = np.array(freqs * len(f) / freqs[-1], np.int32)
    filter_results = np.empty(0)

    for i in range(0, num_coeffs):
        bin1 = indexes[i]
        bin2 = indexes[i + 2]
        peak = indexes[i + 1]

        if bin1 != bin2 != peak:
            a = sp[bin1:bin2]
            b = _mfcc_filter(bin2 - bin1, peak - bin1)
            filter_results = np.append(filter_results, np.dot(a, b))
        else:
            filter_results = np.append(filter_results, [0])
    out = scipy.fftpack.dct(filter_results, type=2, axis=0, norm='ortho')

    return out


def _mfcc_filter(width, peak):
    if width % 2 != 0:
        out = np.linspace(0, 1, peak)
        out = np.append(out, np.linspace(1. - 1. / (width - peak), 0, width - peak))
    else:
        out = np.linspace(0, 1, peak)
        out = np.append(out, np.linspace(1, 0, int(width - peak)))

    return out


def fast_fourier_transform(sig):
    """
    Computes the one-dimensional discrete Fourier Transform.
    :param sig:ndarray
        input from which cepstral coefficients are computed.
    :return: fft_ndarray
    """
    fft = np.fft.fft(sig)

    return fft


def index_highest_fft(sig):
    """Computes the index of the highest Fast Fourier Transform using an one-dimensional discrete Fourier Transform for real input.

    Parameters
    ---------
    sig: ndarray
        input from which cepstral coefficients are computed.
    Returns
    ---------
    h_fft: int64

    """
    fft = fast_fourier_transform(sig)
    h_fft = np.argmax(fft)

    return h_fft


def ratio_1st_2nd_highest_fft_values(sig):
    """
    Computes the ratio between the first and second highest FFT values.
    :param sig:ndarray
    :return:r_fft: float
    """

    fft = fast_fourier_transform(sig)
    fft_1st = np.max(fft)
    fft_2nd = np.max(np.delete(fft, np.where(fft == fft_1st)[0]))
    r_fft =  fft_1st/fft_2nd

    return r_fft