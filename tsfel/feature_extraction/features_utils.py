import scipy
import numpy as np
from decorator import decorator

def set_domain(key, value):
    def decorate_func(func):
        setattr(func, key, value)
        return func

    return decorate_func

@decorator
def vectorize(fn, signal, *args, **kwargs):
    res = np.array([fn(ts, *args, **kwargs) for ts in signal.reshape(-1, signal.shape[-1])])
    if res.size == np.prod(signal.shape[:-1]):
        return res.reshape(signal.shape[:-1])
    else:
        return res.reshape((*signal.shape[:-1], -1))


def match_last_dim_by_value_repeat(values, wts):
    return np.repeat(np.expand_dims(values, axis=-1), np.ma.size(wts, axis=-1), axis=-1)


def tile_last_dim_to_match_shape(arr, targetArr):
    return np.tile(arr, np.shape(targetArr)[:-1] + (1,))


def compute_time(signal, fs):
    """Creates the signal correspondent time array.

    Parameters
    ----------
    signal: nd-array
        Input from which the time is computed.
    fs: int
        Sampling Frequency

    Returns
    -------
    time : float list
        Signal time

    """

    return tile_last_dim_to_match_shape(np.arange(0, np.ma.size(signal, axis=-1) / fs, 1./fs), signal)


def devide_keep_zero(a, b, out=np.zeros_like):
    return np.divide(a, b, out=out(b), where=b != 0)


def calc_fft(signal, sf):
    """ This functions computes the fft of a signal.

    Assumption
    ----------
    This is a well behaved ndarray in which each dimension has the same length.
    willma.ill fail / the wrong dimension is used, as numpy will (likely) flatten all mismatching dimensions below

    Parameters
    ----------
    signal : nd-array
        The input signal from which fft is computed
    sf : int
        Sampling frequency

    Returns
    -------
    f: nd-array
        Frequency values (xx axis)
    fmag: nd-array
        Amplitude of the frequency values (yy axis)

    """

    fmag = np.abs(np.fft.fft(signal, axis=-1))
    signalLength = np.ma.size(signal, axis=-1) // 2
    f = np.linspace(0, sf // 2, signalLength)

    # as we already assumed they all have the same length and the same sf, we can just bring f to the same shape as the fmag return value

    fmag_ret = fmag[..., :signalLength]
    f_ret = tile_last_dim_to_match_shape(f, fmag_ret)

    return f_ret, fmag_ret




def filterbank(signal, fs, pre_emphasis=0.97, nfft=512, nfilt=40):
    """Computes the MEL-spaced filterbank.

    It provides the information about the power in each frequency band.

    Implementation details and description on:
    https://www.kaggle.com/ilyamich/mfcc-implementation-and-tutorial
    https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html#fnref:1

    Parameters
    ----------
    signal : nd-array
        Input from which filterbank is computed
    fs : int
        Sampling frequency
    pre_emphasis : float
        Pre-emphasis coefficient for pre-emphasis filter application
    nfft : int
        Number of points of fft
    nfilt : int
        Number of filters

    Returns
    -------
    nd-array
        MEL-spaced filterbank

    """

    # Signal is already a window from the original signal, so no frame is needed.
    # According to the references it is needed the application of a window function such as
    # hann window. However if the signal windows don't have overlap, we will lose information,
    # as the application of a hann window will overshadow the windows signal edges.

    # pre-emphasis filter to amplify the high frequencies

    emphasized_signal = np.append(np.array(signal)[0], np.array(signal[1:]) - pre_emphasis * np.array(signal[:-1]))

    # Fourier transform and Power spectrum
    mag_frames = np.absolute(np.fft.rfft(emphasized_signal, nfft))  # Magnitude of the FFT

    pow_frames = ((1.0 / nfft) * (mag_frames ** 2))  # Power Spectrum

    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (fs / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz
    filter_bin = np.floor((nfft + 1) * hz_points / fs)

    fbank = np.zeros((nfilt, int(np.floor(nfft / 2 + 1))))
    for m in range(1, nfilt + 1):

        f_m_minus = int(filter_bin[m - 1])  # left
        f_m = int(filter_bin[m])  # center
        f_m_plus = int(filter_bin[m + 1])  # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - filter_bin[m - 1]) / (filter_bin[m] - filter_bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (filter_bin[m + 1] - k) / (filter_bin[m + 1] - filter_bin[m])

    # Area Normalization
    # If we don't normalize the noise will increase with frequency because of the filter width.
    enorm = 2.0 / (hz_points[2:nfilt + 2] - hz_points[:nfilt])
    fbank *= enorm[:, np.newaxis]

    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # dB

    return filter_banks


def autocorr_norm(signal):
    """Computes the autocorrelation.

    Implementation details and description in:
    https://ccrma.stanford.edu/~orchi/Documents/speaker_recognition_report.pdf

    Parameters
    ----------
    signal : nd-array
        Input from linear prediction coefficients are computed

    Returns
    -------
    nd-array
        Autocorrelation result

    """

    variance = np.var(signal)
    signal = np.copy(signal - signal.mean())
    r = scipy.signal.correlate(signal, signal)[-len(signal):]

    if (signal == 0).all():
        return np.zeros(len(signal))

    acf = r / variance / len(signal)

    return acf


def create_symmetric_matrix(acf, n_coeff=12):
    """Computes a symmetric matrix.

    Implementation details and description in:
    https://ccrma.stanford.edu/~orchi/Documents/speaker_recognition_report.pdf

    Parameters
    ----------
    acf : nd-array
        Input from which a symmetric matrix is computed
    n_coeff : int
        Number of coefficients

    Returns
    -------
    nd-array
        Symmetric Matrix

    """

    smatrix = np.empty((n_coeff, n_coeff))
    xx = np.arange(n_coeff)
    j = np.tile(xx, n_coeff)
    i = np.repeat(xx, n_coeff)
    smatrix[i, j] = acf[np.abs(i - j)]

    return smatrix


def lpc(signal, n_coeff=12):
    """Computes the linear prediction coefficients.

    Implementation details and description in:
    https://ccrma.stanford.edu/~orchi/Documents/speaker_recognition_report.pdf

    Parameters
    ----------
    signal : nd-array
        Input from linear prediction coefficients are computed
    n_coeff : int
        Number of coefficients

    Returns
    -------
    nd-array
        Linear prediction coefficients

    """

    if signal.ndim > 1:
        raise ValueError("Only 1 dimensional arrays are valid")
    if n_coeff > signal.size:
        raise ValueError("Input signal must have a length >= n_coeff")

    # Calculate LPC with Yule-Walker
    acf = np.correlate(signal, signal, 'full')

    r = np.zeros(n_coeff+1, 'float32')
    # Assuring that works for all type of input lengths
    nx = np.min([n_coeff+1, len(signal)])
    r[:nx] = acf[len(signal)-1:len(signal)+n_coeff]

    smatrix = create_symmetric_matrix(r[:-1], n_coeff)

    if np.sum(smatrix) == 0:
        return tuple(np.zeros(n_coeff+1))

    lpc_coeffs = np.dot(np.linalg.inv(smatrix), -r[1:])

    return tuple(np.concatenate(([1.], lpc_coeffs)))



def create_xx(features):
    """Computes the range of features amplitude for the probability density function calculus.
    Parameters
    ----------
    features : nd-array
        Input features
    Returns
    -------
    nd-array
        range of features amplitude
    """

    min_f = np.min(features, axis=-1)
    max_f = np.abs(np.max(features, axis=-1))
    max_f = np.where(min_f != max_f, max_f, max_f + 10)

    return np.linspace(min_f, max_f, np.ma.size(features, axis=-1)) \
            .transpose(np.append(np.arange(1, features.ndim), 0))

def kde(features):
    """Computes the probability density function of the input signal using a Gaussian KDE (Kernel Density Estimate)

    Parameters
    ----------
    features : nd-array
        Input from which probability density function is computed

    Returns
    -------
    nd-array
        probability density values

    """
    features_ = np.copy(features)
    xx = create_xx(features_)

    if min(features_) == max(features_):
        noise = np.random.randn(len(features_)) * 0.0001
        features_ = np.copy(features_ + noise)

    kernel = scipy.stats.gaussian_kde(features_, bw_method='silverman')

    return np.array(kernel(xx) / np.sum(kernel(xx)))


def gaussian(features):
    """Computes the probability density function of the input signal using a Gaussian function
    Parameters
    ----------
    features : nd-array
        Input from which probability density function is computed
    Returns
    -------
    nd-array
        probability density values
    """

    xx = create_xx(features)
    std_value = np.expand_dims(np.std(features, axis=-1), axis=-1)
    mean_value = np.expand_dims(np.mean(features, axis=-1), axis=-1)

    pdf_gauss = scipy.stats.norm.pdf(xx, mean_value, std_value)

    return np.where(std_value == 0, 0.0, \
                   np.array(pdf_gauss / np.expand_dims(np.sum(pdf_gauss, axis=-1), axis=-1)))

def entropy_vectorized(p):
    normTerm = np.log2(np.count_nonzero(p, axis=-1))

    # this is the vectorized form of the if sum == 0 case used by tsfel
    # unfortunately only the parts where the condition is met is not feasable,
    #     as we would need to use list comprehension and the like
    return np.where(np.sum(p, axis=-1) == 0, 0, \
                    # calculate entropy
                   - np.sum(np.where(p==0, 0, p * np.log2(p)), axis=-1) / normTerm)


def wavelet(signal, function=scipy.signal.ricker, widths=np.arange(1, 10)):
    """Computes CWT (continuous wavelet transform) of the signal.

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
    nd-array
        The result of the CWT along the time axis
        matrix with size (len(widths),len(signal))

    """

    if isinstance(function, str):
        function = eval(function)

    if isinstance(widths, str):
        widths = eval(widths)

    cwt = scipy.signal.cwt(signal, function, widths)

    return cwt


def calc_ecdf(signal):
    """Computes the ECDF of the signal.

      Parameters
      ----------
      signal : nd-array
          Input from which ECDF is computed
      Returns
      -------
      nd-array
        Sorted signal and computed ECDF.

      """
    return np.sort(signal), np.arange(1, len(signal)+1)/len(signal)

