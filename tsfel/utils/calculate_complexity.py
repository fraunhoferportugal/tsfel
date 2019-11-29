import time
import json
import numpy as np
from scipy.optimize import curve_fit
from tsfel.feature_extraction.features_settings import load_json
from tsfel.feature_extraction.calc_features import calc_window_features


# curves
def n_squared(x, no):
    """The model function"""
    return no * x ** 2


def n_nlog(x, no):
    """The model function"""
    return no * x * np.log(x)


def n_linear(x, no):
    """The model function"""
    return no * x


def n_log(x, no):
    """The model function"""
    return no * np.log(x)


def n_constant(x, no):
    """The model function"""
    return np.zeros(len(x)) + no


def find_best_curve(t, signal):
    """Finds the best curve.

    Parameters
    ----------
    t : nd-array
        Log space
    signal : nd-array
        Mean execution time array

    Returns
    -------
    str
        Best fit curve name

    """

    all_chisq = []
    list_curves = [n_squared, n_nlog, n_linear, n_log, n_constant]
    all_curves = []
    # Model parameters
    stdev = 2
    sig = np.zeros(len(signal)) + stdev

    # Fit the curve
    for curve in list_curves:
        start = 1
        popt, pcov = curve_fit(curve, t, signal, sigma=sig, p0=start, absolute_sigma=True)

        # Compute chi square
        nexp = curve(t, *popt)
        r = signal - nexp
        chisq = np.sum((r / stdev) ** 2)
        all_chisq.append(chisq)
        all_curves.append(nexp)

    idx_best = np.argmin(all_chisq)

    curve_name = str(list_curves[idx_best])
    idx1 = curve_name.find("n_")
    idx2 = curve_name.find("at")
    curve_name = curve_name[idx1 + 2:idx2 - 1]

    return curve_name


def compute_complexity(feature, domain, json_path, **kwargs):
    """Computes the feature complexity.

    Parameters
    ----------
    feature : string
        Feature name
    domain : string
        Feature domain
    json_path: json
        Features json file
    \**kwargs:
    See below:
        * *features_path* (``string``) --
            Directory of script with personal features

    Returns
    -------
    int
        Feature complexity

    Writes complexity in json file

    """

    dictionary = load_json(json_path)

    features_path = kwargs.get('features_path', None)

    # The inputs from this function should be replaced by a dictionary
    one_feat_dict = {domain: {feature: dictionary[domain][feature]}}

    t = np.logspace(3.0, 5.0, 6)
    signal, s = [], []
    f = 0.05
    x = np.arange(0, t[-1] + 1, 1)
    fs = 100
    wave = np.sin(2 * np.pi * f * x / fs)

    for ti in t:
        for _ in range(20):

            start = time.time()
            calc_window_features(one_feat_dict, wave[:int(ti)], fs, features_path=features_path)
            end = time.time()

            s += [end - start]

        signal += [np.mean(s)]

    curve_name = find_best_curve(t, signal)
    dictionary[domain][feature]['complexity'] = curve_name

    with open(json_path, "w") as write_file:
        json.dump(dictionary, write_file, indent=4, sort_keys=True)

    if curve_name == 'constant' or curve_name == 'log':
        return 1
    elif curve_name == 'linear':
        return 2
    elif curve_name == 'nlog' or curve_name == 'squared':
        return 3
    else:
        return 0
