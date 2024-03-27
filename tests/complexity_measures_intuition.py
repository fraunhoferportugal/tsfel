"""This module offers helper functions designed to facilitate testing and
enhance understanding of features that quantify repetitiveness in dynamic
systems."""

import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf

from tests.tests_tools.test_data.complexity_datasets import load_complexities_datasets


def plot_dfa(info, scale, fluctuations, ax):
    """Plot log-log visualization of the detrended fluctuation analysis
    (DFA)."""

    polyfit = np.polyfit(np.log2(scale), np.log2(fluctuations), 1)
    fluctfit = 2 ** np.polyval(polyfit, np.log2(scale))
    ax.loglog(scale, fluctuations, "o", c="#90A4AE")
    ax.loglog(
        scale,
        fluctfit,
        c="#E91E63",
        label=r"$\alpha$ = {:.3f}".format(info["Alpha"]),
    )

    return ax


def plot_multiscale_entropy(info, ax):
    """Plots the entropy values for each scale factor."""

    ax.plot(
        info["Scale"][np.isfinite(info["Value"])],
        info["Value"][np.isfinite(info["Value"])],
        color="#FF9800",
    )

    return ax


def get_first_acf_time_constant(signal):
    """Captures the approximate scale of the autocorrelation function by
    measuring the first time lag at which the autocorrelation function (acf)
    drops below 1/e = 0.3679."""
    n = signal.size
    maxlag = int(n / 4)
    threshold = 0.36787944117144233  # 1 / np.exp(1)

    acf_vec = acf(signal, adjusted=True, fft=n > 400, nlags=maxlag)[
        1:
    ]  # n > 400 empirically selected based on performance tests
    idxs = np.where(acf_vec < threshold)[0]
    first_lag = idxs[0] + 1 if idxs.size > 0 else maxlag

    return first_lag


dataset = load_complexities_datasets()

f, ax = plt.subplots(4, len(dataset), figsize=(15, 5))
for i, (k, _v) in enumerate(dataset.items()):
    signal = dataset[k]

    # First row (raw data)
    ax[0, i].plot(signal, "k")
    ax[0, i].set_title(k)
    ax[0, i].spines[["left", "top", "right"]].set_visible(False)
    ax[0, i].tick_params(left=False, right=False, labelleft=False)

    # Second row (first_acf_time_constant)
    plot_acf(signal, ax[1, i], adjusted=True, fft=len(signal) > 1250, title="")
    ax[1, i].axhline(y=(1 / np.e), color="r", linestyle="--")
    ax[1, i].spines[["top", "right"]].set_visible(False)
    ax[1, i].set_title("$\\tau = %d$" % get_first_acf_time_constant(signal))

    # Third row (dfa)
    # TODO: Change to our own implementation instead of relying on neurokit2
    _, info = nk.fractal_dfa(signal)
    plot_dfa(info, info["scale"], info["Fluctuations"], ax[2, i])
    ax[2, i].set_title("$\\alpha = %0.2f$" % info["Alpha"])
    ax[2, i].set_axis_off()

    # Forth row (mse)
    mse, info = nk.entropy_multiscale(signal)
    plot_multiscale_entropy(info, ax[3, i])
    ax[3, i].set_title("mse = %0.2f" % mse)
    ax[3, i].set_ylim([0, 2.5])
    ax[3, i].spines[["top", "right"]].set_visible(False)

f.tight_layout()
plt.show(block=False)
