import numpy as np
import scipy
from sklearn.neighbors import KDTree


def set_domain(key, value):
    def decorate_func(func):
        setattr(func, key, value)
        return func

    return decorate_func


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

    return np.arange(0, len(signal)) / fs


def calc_fft(signal, fs):
    """This functions computes the fft of a signal.

    Parameters
    ----------
    signal : nd-array
        The input signal from which fft is computed
    fs : float
        Sampling frequency

    Returns
    -------
    f: nd-array
        Frequency values (xx axis)
    fmag: nd-array
        Amplitude of the frequency values (yy axis)
    """

    fmag = np.abs(np.fft.rfft(signal))
    f = np.fft.rfftfreq(len(signal), d=1 / fs)

    return f.copy(), fmag.copy()


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
    fs : float
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

    emphasized_signal = np.append(
        np.array(signal)[0],
        np.array(signal[1:]) - pre_emphasis * np.array(signal[:-1]),
    )

    # Fourier transform and Power spectrum
    mag_frames = np.absolute(
        np.fft.rfft(emphasized_signal, nfft),
    )  # Magnitude of the FFT

    pow_frames = (1.0 / nfft) * (mag_frames**2)  # Power Spectrum

    low_freq_mel = 0
    high_freq_mel = 2595 * np.log10(1 + (fs / 2) / 700)  # Convert Hz to Mel
    mel_points = np.linspace(
        low_freq_mel,
        high_freq_mel,
        nfilt + 2,
    )  # Equally spaced in Mel scale
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)  # Convert Mel to Hz
    filter_bin = np.floor((nfft + 1) * hz_points / fs)

    fbank = np.zeros((nfilt, int(np.floor(nfft / 2 + 1))))
    for m in np.arange(1, nfilt + 1):

        f_m_minus = int(filter_bin[m - 1])  # left
        f_m = int(filter_bin[m])  # center
        f_m_plus = int(filter_bin[m + 1])  # right

        for k in np.arange(f_m_minus, f_m):
            fbank[m - 1, k] = (k - filter_bin[m - 1]) / (filter_bin[m] - filter_bin[m - 1])
        for k in np.arange(f_m, f_m_plus):
            fbank[m - 1, k] = (filter_bin[m + 1] - k) / (filter_bin[m + 1] - filter_bin[m])

    # Area Normalization
    # If we don't normalize the noise will increase with frequency because of the filter width.
    enorm = 2.0 / (hz_points[2 : nfilt + 2] - hz_points[:nfilt])
    fbank *= enorm[:, np.newaxis]

    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(
        filter_banks == 0,
        np.finfo(float).eps,
        filter_banks,
    )  # Numerical Stability
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
    r = scipy.signal.correlate(signal, signal)[-len(signal) :]

    if (signal == 0).all():
        return np.zeros(len(signal))

    acf = r / variance / len(signal)

    return acf


def create_symmetric_matrix(acf, order=11):
    """Computes a symmetric matrix.

    Implementation details and description in:
    https://ccrma.stanford.edu/~orchi/Documents/speaker_recognition_report.pdf

    Parameters
    ----------
    acf : nd-array
        Input from which a symmetric matrix is computed
    order : int
        Order

    Returns
    -------
    nd-array
        Symmetric Matrix
    """

    smatrix = np.empty((order, order))
    xx = np.arange(order)
    j = np.tile(xx, order)
    i = np.repeat(xx, order)
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

    # Calculate the order based on the number of coefficients
    order = n_coeff - 1

    # Calculate LPC with Yule-Walker
    acf = np.correlate(signal, signal, "full")

    r = np.zeros(order + 1, "float32")
    # Assuring that works for all type of input lengths
    nx = np.min([order + 1, len(signal)])
    r[:nx] = acf[len(signal) - 1 : len(signal) + order]

    smatrix = create_symmetric_matrix(r[:-1], order)

    if np.sum(smatrix) == 0:
        return tuple(np.zeros(order + 1))

    lpc_coeffs = np.dot(np.linalg.inv(smatrix), -r[1:])

    return tuple(np.concatenate(([1.0], lpc_coeffs)))


def create_xx(features):
    """Computes the range of features amplitude for the probability density
    function calculus.

    Parameters
    ----------
    features : nd-array
        Input features

    Returns
    -------
    nd-array
        range of features amplitude
    """

    features_ = np.copy(features)

    if max(features_) < 0:
        max_f = -max(features_)
        min_f = min(features_)
    else:
        min_f = min(features_)
        max_f = max(features_)

    if min(features_) == max(features_):
        xx = np.linspace(min_f, min_f + 10, len(features_))
    else:
        xx = np.linspace(min_f, max_f, len(features_))

    return xx


def kde(features):
    """Computes the probability density function of the input signal using a
    Gaussian KDE (Kernel Density Estimate)

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

    kernel = scipy.stats.gaussian_kde(features_, bw_method="silverman")

    return np.array(kernel(xx) / np.sum(kernel(xx)))


def gaussian(features):
    """Computes the probability density function of the input signal using a
    Gaussian function.

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
    std_value = np.std(features_)
    mean_value = np.mean(features_)

    if std_value == 0:
        return 0.0
    pdf_gauss = scipy.stats.norm.pdf(xx, mean_value, std_value)

    return np.array(pdf_gauss / np.sum(pdf_gauss))


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
    return np.sort(signal), np.arange(1, len(signal) + 1) / len(signal)


def coarse_graining(signal, scale):
    """Applies a coarse-graining process to a time series: for a given scale factor, it splits
    the signal into non-overlapping windows and averages the data points.

    Parameters
    ----------
    signal : np.ndarray
        Input signal.
    scale : int
        Scale factor, determines the length of the non-overlapping windows.

    Returns
    -------
    coarsegrained_signal : np.ndarray
        Coarse-grained signal.
    """

    n = len(signal)
    windows = n // scale
    usable_n = windows * scale

    windowed_signal = np.reshape(signal[0:usable_n], (windows, scale))
    coarsegrained_signal = np.nanmean(windowed_signal, axis=1)

    return coarsegrained_signal


def get_templates(signal, m=3):
    """Helper function for the sample entropy calculation. Divides a signal
    into templates vectors of length m.

    Parameters
    ----------
    signal : np.ndarray
        Input signal.
    m : int
        Embedding dimension that defines the length of the template vectors, defaults to 3.

    Returns
    -------
    np.ndarray
        Array of template vectors.
    """
    return np.array([signal[i : i + m] for i in np.arange(len(signal) - m + 1)])


def sample_entropy(signal, m, tolerance):
    """Computes the sample entropy of a signal.

    Parameters
    ----------
    signal : np.ndarray
        Input signal.
    m : int
        Embedding dimension that defines the length of the template vectors, defaults to 3.
    tolerance : float
        Tolerance value, defaults to 0.2 times the standard deviation of the input signal.

    Returns
    -------
    float
        Sample Entropy of a signal.
    """
    templates_B = get_templates(signal, m)
    templates_B = templates_B[:-1]
    kdtree_B = KDTree(templates_B, metric="chebyshev")
    count_B = kdtree_B.query_radius(templates_B, tolerance, count_only=True).astype(
        np.float64,
    )
    proportion_B = np.mean((count_B - 1) / (templates_B.shape[0] - 1))

    templates_A = get_templates(signal, m + 1)
    kdtree_A = KDTree(templates_A, metric="chebyshev")
    count_A = kdtree_A.query_radius(templates_A, tolerance, count_only=True).astype(
        np.float64,
    )
    proportion_A = np.mean((count_A - 1) / (templates_A.shape[0] - 1))

    if proportion_B > 0 and proportion_A > 0:
        return -np.log(proportion_A / proportion_B)
    return np.nan


def calc_rms(signal, window):
    """Windowed Root Mean Square (RMS) with linear detrending.

    Parameters
    ----------
    signal: nd-array
        Signal
    window: int
        Length of the window in which RMS will be calculated

    Returns
    -------
    rms : nd-array
        RMS data in each window with length len(signal)//window
    """
    num_windows = len(signal) // window
    rms = np.zeros(num_windows)

    for idx in np.arange(num_windows):
        start_idx = idx * window
        end_idx = start_idx + window
        windowed_signal = signal[start_idx:end_idx]

        coeff = np.polyfit(np.arange(window), windowed_signal, 1)
        detrended_window = windowed_signal - np.polyval(coeff, np.arange(window))
        rms[idx] = np.sqrt(np.mean(detrended_window**2))

    return rms


def compute_rs(signal, lag):
    """Computes the average rescaled range for a window of length lag.

    Parameters
    ----------
    signal : np.ndarray
        Input signal.
    lag : int
        Window length.

    Returns
    -------
    float
        Average R/S.
    """
    n = len(signal)

    windowed_signal = np.reshape(signal[: n - n % lag], (-1, lag))
    mean_windows = np.mean(windowed_signal, axis=1)
    accumulated_windowed_signal = np.cumsum(
        windowed_signal - np.reshape(mean_windows, (-1, 1)),
        axis=1,
    )

    r = np.max(accumulated_windowed_signal, axis=1) - np.min(
        accumulated_windowed_signal,
        axis=1,
    )
    s = np.std(windowed_signal, axis=1)

    rs = np.divide(r, s)

    return np.mean(rs)


def calc_lengths_higuchi(signal):
    """Computes the lengths for different subdivisions, using the Higuchi's
    method.

    Parameters
    ----------
    signal : np.ndarray
        Input signal.

    Returns
    -------
    lk : nd-array
        Length of curve for different subdivisions
    """
    n = len(signal)
    k_values = np.arange(1, n // 10)
    lk = []

    for k in k_values:
        lmk = 0
        for m in np.arange(1, k + 1):
            sum_length = 0
            for i in np.arange(1, (n - m) // k + 1):
                sum_length += abs(signal[m + i * k - 1] - signal[m + (i - 1) * k - 1])
            lmk += (sum_length * (n - 1)) / (((n - m) // k) * k**2)
        lk.append(lmk / k)

    return k_values, lk


def calc_lempel_ziv_complexity(sequence):
    """Manual implementation of the Lempel-Ziv complexity.

    It is defined as the number of different substrings encountered as
    the stream is viewed from begining to the end.

    Reference:
    https://github.com/Naereen/Lempel-Ziv_Complexity/blob/master/src/lempel_ziv_complexity.py

    Parameters
    ----------
    sequence : string
        Binarised signal, as a string of characters

    Returns
    -------
        LZ index
    """

    sub_strings = set()

    ind = 0
    inc = 1
    while True:
        if ind + inc > len(sequence):
            break
        sub_str = sequence[ind : ind + inc]
        if sub_str in sub_strings:
            inc += 1
        else:
            sub_strings.add(sub_str)
            ind += inc
            inc = 1

    return len(sub_strings) / len(sequence)


def find_plateau(y, threshold=0.1, consecutive_points=5):
    """Finds a plateau (if it exists).

    Parameters
    ----------
    y : np.ndarray
        Array of y-axis values.
    threshold : float
        Slope threshold to consider as a plateau (default is 0.1).
    consecutive_points: int
        Number of consecutive points with a small derivative to consider as a plateau (default is 5).

    Returns
    -------
        Index of the beggining of the plateau if it is found, length of y otherwise.
    """

    dy = np.diff(y)

    for i in np.arange(len(dy) - consecutive_points + 1):
        if np.all(np.abs(dy[i : i + consecutive_points]) < threshold):
            plateau_value = np.mean(y[i : i + consecutive_points])
            if plateau_value > np.mean(y):
                return i

    # No plateau found
    return len(y)
