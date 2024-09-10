[![Documentation Status](https://readthedocs.org/projects/tsfel/badge/?version=dev)](https://tsfel.readthedocs.io/en/latest/?badge=latest)
[![license](https://img.shields.io/badge/license-BSD%203--Clause-green?logo=style)](https://github.com/aeon-toolkit/aeon/blob/main/LICENSE)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/tsfel)
[![!pypi](https://img.shields.io/pypi/v/tsfel?logo=pypi&color=blue)](https://pypi.org/project/tsfel/)
[![Downloads](https://pepy.tech/badge/tsfel)](https://pepy.tech/project/tsfel)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fraunhoferportugal/tsfel/blob/master/notebooks/TSFEL_HAR_Example.ipynb)
[![Slack](https://img.shields.io/badge/slack-join_here-lightgreen.svg?logo=slack)](https://join.slack.com/t/tsfel/shared_invite/zt-2pnrdi1s4-kCROK6R1mSP3D2MnSXUSZg)

# Time Series Feature Extraction Library

<p align="center">
    <a href="https://readthedocs.org/projects/tsfel/badge/?version=latest"><img src="https://raw.githubusercontent.com/fraunhoferportugal/tsfel/master/docs/imgs/tsfel_feature_sets.png" width="100%" alt="tsfel logo" /></a>
</p>

## Intuitive time series feature extraction

`TSFEL` is an open-source Python library for time series analysis. It centralizes a large and powerful feature set of
several feature extraction methods from statistical, temporal, spectral, and fractal domains.

The documentation is available [here](https://github.com/fraunhoferportugal/tsfel/issues).

You can install `TSFEL` via pip using the following:

```python
pip install tsfel
```

A release on conda-forge is coming soon.

## Getting started

Below is a quick example of how to use `TSFEL` for time series feature extraction:

```python
import tsfel

# Loads a 10 s single lead ECG
data = tsfel.datasets.load_biopluxecg()

# Set up the default configuration using using the statistical, temporal and spectral feature sets. 
cfg = tsfel.get_features_by_domain()

# Extract features
X = tsfel.time_series_features_extractor(cfg, data)
```

For a more detailed walk-through — including input/output data formats, extraction routine configuration, and how to
implement your custom features — refer to the documentation here.

## Highlights

- **Intuitive, fast deployment, and reproducible**: Easily configure your feature extraction pipeline and store the
  configuration file to ensure reproducibility.
- **Computational complexity evaluation**: Estimate the computational time required for feature extraction in advance.
- **Comprehensive documentation**: Each feature extraction method is accompanied by a detailed explanation.
- **Unit tested**: We provide an extensive suite of unit tests for each feature to ensure accurate and reliable feature
  calculation.
- **Easily extended**: Adding new features is straightforward, and we encourage contributions of custom features to the
  community.

## Available features

`TSFEL` automatically extracts more than 65 distinct features across statistical, temporal, spectral, and fractal
domains.

#### Statistical domain

| Features                  | Computational Cost |
|---------------------------|:------------------:|
| Absolute energy           |         1          |
| Average power             |         1          |
| ECDF                      |         1          |
| ECDF Percentile           |         1          |
| ECDF Percentile Count     |         1          |
| Entropy                   |         1          |
| Histogram                 |         1          |
| Interquartile range       |         1          |
| Kurtosis                  |         1          |
| Max                       |         1          |
| Mean                      |         1          |
| Mean absolute deviation   |         1          |
| Median                    |         1          |
| Median absolute deviation |         1          |
| Min                       |         1          |
| Root mean square          |         1          |
| Skewness                  |         1          |
| Standard deviation        |         1          |
| Variance                  |         1          |

#### Temporal domain

| Features                | Computational Cost |
|-------------------------|:------------------:|
| Area under the curve    |         1          |
| Autocorrelation         |         2          |
| Centroid                |         1          |
| Lempel-Ziv-Complexity\* |         2          |
| Mean absolute diff      |         1          |
| Mean diff               |         1          |
| Median absolute diff    |         1          |
| Median diff             |         1          |
| Negative turning points |         1          |
| Peak to peak distance   |         1          |
| Positive turning points |         1          |
| Signal distance         |         1          |
| Slope                   |         1          |
| Sum absolute diff       |         1          |
| Zero crossing rate      |         1          |
| Neighbourhood peaks     |         1          |

\* Disabled by default due to its longer execution time compared to other features.

#### Spectral domain

| Features                         | Computational Cost |
|----------------------------------|:------------------:|
| FFT mean coefficient             |         1          |
| Fundamental frequency            |         1          |
| Human range energy               |         1          |
| LPCC                             |         1          |
| MFCC                             |         1          |
| Max power spectrum               |         1          |
| Maximum frequency                |         1          |
| Median frequency                 |         1          |
| Power bandwidth                  |         1          |
| Spectral centroid                |         2          |
| Spectral decrease                |         1          |
| Spectral distance                |         1          |
| Spectral entropy                 |         1          |
| Spectral kurtosis                |         2          |
| Spectral positive turning points |         1          |
| Spectral roll-off                |         1          |
| Spectral roll-on                 |         1          |
| Spectral skewness                |         2          |
| Spectral slope                   |         1          |
| Spectral spread                  |         2          |
| Spectral variation               |         1          |
| Wavelet absolute mean            |         2          |
| Wavelet energy                   |         2          |
| Wavelet standard deviation       |         2          |
| Wavelet entropy                  |         2          |
| Wavelet variance                 |         2          |

#### Fractal domain

| Features                             | Computational Cost |
|--------------------------------------|:------------------:|
| Detrended fluctuation analysis (DFA) |         3          |
| Higuchi fractal dimension            |         3          |
| Hurst exponent                       |         3          |
| Maximum fractal length               |         3          |
| Multiscale entropy (MSE)             |         1          |
| Petrosian fractal dimension          |         1          |

_Fractal domain features are typically applied to relatively longer signals to capture meaningful patterns, and it's
usually
unnecessary to previously divide the signal into shorter windows. Therefore, this domain is disabled in the default
feature
configuration files._

## Support & General discussion

For bug reports, please use the [GitHub issue tracker](https://github.com/fraunhoferportugal/tsfel/issues). To make
feature requests, share ideas, engage in general discussions, or receive announcements, you're welcome to join
our [Slack community](https://join.slack.com/t/tsfel/shared_invite/zt-2pnrdi1s4-kCROK6R1mSP3D2MnSXUSZg).

## Citing

If you use `TSFEL` in your work, please cite the following publication:

Barandas, Marília and Folgado, Duarte, et al. "*TSFEL: Time Series Feature Extraction Library.*" SoftwareX 11 (
2020). [https://doi.org/10.1016/j.softx.2020.100456](https://doi.org/10.1016/j.softx.2020.100456)

## Acknowledgements

We gratefully acknowledge the financial support received from the Center for Responsible AI and the Total Integrated and
Predictive Manufacturing System Platform for Industry 4.0 projects.
