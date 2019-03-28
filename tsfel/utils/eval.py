import numpy as np
from scipy.optimize import curve_fit
from tsfel.utils.read_json import compute_dictionary
from tsfel.utils.read_json import one_extract
import time
import json
import tsfel


###########################################
# curves
def n_Squared(x, No):
    """The model function"""
    return No * (x) ** 2


def n_Nlog(x, No):
    """The model function"""
    return No * x * np.log(x)


def n_Linear(x, No):
    """The model function"""
    return No * (x)


def n_Log(x, No):
    """The model function"""
    return No * np.log(x)


def n_Constant(x, No):
    """The model function"""
    return np.zeros(len(x)) + No


###########################################a


def find_best_curve(t, signal):

    all_chisq = []
    list_curves = [n_Squared, n_Nlog, n_Linear, n_Log, n_Constant]
    all_curves = []
    # Model parameters
    stdev = 2
    sig = np.zeros(len(signal)) + stdev

    # Fit the curve
    for curve in list_curves:
        start = 1
        popt, pcov = curve_fit(curve, t, signal, sigma=sig, p0=start, absolute_sigma=True)

        # Compute chi square
        Nexp = curve(t, *popt)
        r = signal - Nexp
        chisq = np.sum((r / stdev) ** 2)
        df = len(signal) - 2

        all_chisq.append(chisq)
        all_curves.append(Nexp)

    idx_best = np.argmin(all_chisq)

    curve_name = str(list_curves[idx_best])
    idx1 = curve_name.find("n_")
    idx2 = curve_name.find("at")
    curve_name = curve_name[idx1 + 2:idx2 - 1]

    return np.min(all_chisq), curve_name

    # Plot the data with error bars along with the fit result


def compute_complexity(feat, domain, filename):
    DEFAULT = {'use': 'yes', 'metric': 'euclidean', 'free parameters': '', 'number of features': 1, 'parameters': ''}
    dictionary = compute_dictionary(filename, DEFAULT)
    t = np.logspace(3.0, 5.0, 6)
    signal,s = [], []
    f = 0.05
    x = np.arange(0, t[-1]+1, 1)
    Fs = 100
    wave = np.sin(2 * np.pi * f * x / Fs)
    for ti in t:
        for _ in range(20):
            start = time.time()
            res = one_extract(dictionary[domain][feat], wave[:int(ti)], Fs)
            end = time.time()
            s += [end-start]
        #print(np.mean(s))
        signal += [np.mean(s)]

    chisq, curve_name = find_best_curve(t, signal)
    dictionary[domain][feat]['Complexity'] = curve_name

    with open(filename, "w") as write_file:
        js = json.dump(dictionary, write_file, indent=4, sort_keys=True)

    if curve_name == 'Constant' or curve_name == 'Log':
        return 1
    elif curve_name == 'Linear':
        return 2
    elif curve_name == 'Nlog' or curve_name == 'Squared':
        return 3
    else:
        return 0

# find_best_curve(signal)
