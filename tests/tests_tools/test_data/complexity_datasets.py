"""This module provides functions for creating or downloading representative
data from several dataset sources. These functions are designed to enhance the
understanding of complexity measures related to dynamical systems.

The implementation for generating colored noise is sourced from the
'colorednoise' PyPI package, with credit to Felix Patzelt.
"""

import os
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd
from numpy import integer, newaxis, sqrt
from numpy import sum as npsum
from numpy.fft import irfft, rfftfreq
from numpy.random import Generator, RandomState, default_rng


def powerlaw_psd_gaussian(
    exponent: float,
    size: Union[int, Iterable[int]],
    fmin: float = 0.0,
    random_state: Optional[Union[int, Generator, RandomState]] = None,
):
    """Gaussian (1/f)**beta noise.

    Based on the algorithm in:
    Timmer, J. and Koenig, M.:
    On generating power law noise.
    Astron. Astrophys. 300, 707-710 (1995)

    Normalised to unit variance

    Parameters:
    -----------

    exponent : float
        The power-spectrum of the generated noise is proportional to

        S(f) = (1 / f)**beta
        flicker / pink noise:   exponent beta = 1
        brown noise:            exponent beta = 2

        Furthermore, the autocorrelation decays proportional to lag**-gamma
        with gamma = 1 - beta for 0 < beta < 1.
        There may be finite-size issues for beta close to one.

    shape : int or iterable
        The output has the given shape, and the desired power spectrum in
        the last coordinate. That is, the last dimension is taken as time,
        and all other components are independent.

    fmin : float, optional
        Low-frequency cutoff.
        Default: 0 corresponds to original paper.

        The power-spectrum below fmin is flat. fmin is defined relative
        to a unit sampling rate (see numpy's rfftfreq). For convenience,
        the passed value is mapped to max(fmin, 1/samples) internally
        since 1/samples is the lowest possible finite frequency in the
        sample. The largest possible value is fmin = 0.5, the Nyquist
        frequency. The output for this value is white noise.

    random_state :  int, numpy.integer, numpy.random.Generator, numpy.random.RandomState,
                    optional
        Optionally sets the state of NumPy's underlying random number generator.
        Integer-compatible values or None are passed to np.random.default_rng.
        np.random.RandomState or np.random.Generator are used directly.
        Default: None.

    Returns
    -------
    out : array
        The samples.


    Examples:
    ---------

    # generate 1/f noise == pink noise == flicker noise
    """

    # Make sure size is a list so we can iterate it and assign to it.
    if isinstance(size, (integer, int)):
        size = [size]
    elif isinstance(size, Iterable):
        size = list(size)
    else:
        raise ValueError("Size must be of type int or Iterable[int]")

    # The number of samples in each time series
    samples = size[-1]

    # Calculate Frequencies (we asume a sample rate of one)
    # Use fft functions for real output (-> hermitian spectrum)
    f = rfftfreq(samples)  # type: ignore # mypy 1.5.1 has problems here

    # Validate / normalise fmin
    if 0 <= fmin <= 0.5:
        fmin = max(fmin, 1.0 / samples)  # Low frequency cutoff
    else:
        raise ValueError("fmin must be chosen between 0 and 0.5.")

    # Build scaling factors for all frequencies
    s_scale = f
    ix = npsum(s_scale < fmin)  # Index of the cutoff
    if ix and ix < len(s_scale):
        s_scale[:ix] = s_scale[ix]
    s_scale = s_scale ** (-exponent / 2.0)

    # Calculate theoretical output standard deviation from scaling
    w = s_scale[1:].copy()
    w[-1] *= (1 + (samples % 2)) / 2.0  # correct f = +-0.5
    sigma = 2 * sqrt(npsum(w**2)) / samples

    # Adjust size to generate one Fourier component per frequency
    size[-1] = len(f)

    # Add empty dimension(s) to broadcast s_scale along last
    # dimension of generated random power + phase (below)
    dims_to_add = len(size) - 1
    s_scale = s_scale[(newaxis,) * dims_to_add + (Ellipsis,)]

    # prepare random number generator
    normal_dist = _get_normal_distribution(random_state)

    # Generate scaled random power + phase
    sr = normal_dist(scale=s_scale, size=size)
    si = normal_dist(scale=s_scale, size=size)

    # If the signal length is even, frequencies +/- 0.5 are equal
    # so the coefficient must be real.
    if not (samples % 2):
        si[..., -1] = 0
        sr[..., -1] *= sqrt(2)  # Fix magnitude

    # Regardless of signal length, the DC component must be real
    si[..., 0] = 0
    sr[..., 0] *= sqrt(2)  # Fix magnitude

    # Combine power + corrected phase to Fourier components
    s = sr + 1j * si

    # Transform to real time series & scale to unit variance
    y = irfft(s, n=samples, axis=-1) / sigma

    return y


def _get_normal_distribution(
    random_state: Optional[Union[int, Generator, RandomState]],
):
    normal_dist = None
    if isinstance(random_state, (integer, int)) or random_state is None:
        random_state = default_rng(random_state)
        normal_dist = random_state.normal
    elif isinstance(random_state, (Generator, RandomState)):
        normal_dist = random_state.normal
    else:
        raise ValueError(
            "random_state must be one of integer, numpy.random.Generator, " "numpy.random.Randomstate",
        )
    return normal_dist


def _get_data_from_url_column(url: str):
    return pd.read_csv(url, usecols=[1]).values.flatten()


def _get_data_from_url_ucr(url: str):
    return pd.read_csv(url, header=None).iloc[0, :][1:].values


def _get_data_from_url_plux(url: str):
    return np.loadtxt(url)[1]


COLORED_NOISE_SAMPLES = 2**12


def load_complexities_datasets():
    metadata = {
        "white": {"exponent": 0},
        "pink": {"exponent": 1},
        "brownian": {"exponent": 2},
        "ecg": {
            "url": "https://raw.githubusercontent.com/hgamboa/novainstrumentation/master/novainstrumentation/data/cleanecg.txt",
            "func": _get_data_from_url_plux,
        },
        "airpax": {
            "url": "https://raw.githubusercontent.com/AileenNielsen/TimeSeriesAnalysisWithPython/master/data/AirPassengers.csv",
            "func": _get_data_from_url_column,
        },
        "earthquake": {
            "url": "https://raw.githubusercontent.com/AileenNielsen/TimeSeriesAnalysisWithPython/master/data/Earthquakes.csv",
            "func": _get_data_from_url_ucr,
        },
        "50words": {
            "url": "https://raw.githubusercontent.com/AileenNielsen/TimeSeriesAnalysisWithPython/master/data/50words_TEST.csv",
            "func": _get_data_from_url_ucr,
        },
    }

    dataset = {
        key: (
            powerlaw_psd_gaussian(metadata[key]["exponent"], COLORED_NOISE_SAMPLES)
            if "exponent" in metadata[key]
            else metadata[key]["func"](metadata[key]["url"])
        )
        for key in metadata
    }

    return dataset
