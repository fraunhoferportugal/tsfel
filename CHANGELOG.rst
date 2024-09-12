=========
Changelog
=========

Version 0.1.9
=============
- Changes
    - Replaced the ``Histogram`` feature with ``Histogram mode`` (`#167 <https://github.com/fraunhoferportugal/tsfel/pull/167>`_)

- Improvements
    - The ``correlated_features`` method now supports returning a filtered feature vector (`#163 <https://github.com/fraunhoferportugal/tsfel/pull/163>`_)

- Documentation
    - Set up a Slack community invite (`#164 <https://github.com/fraunhoferportugal/tsfel/issues/164>`_)
    - Enhanced documentation for ``the time_series_feature_extraction`` and ``get_features_by_domain functions``
    - Updated the main example notebook on Human Activity Recognition (`#159 <https://github.com/fraunhoferportugal/tsfel/issues/159>`_)

Version 0.1.8
=============
- New Features
    - Added a new Datasets module with convenient methods to load single-problem datasets (`#156 <https://github.com/fraunhoferportugal/tsfel/pull/156>`_)
    - Improved the ``get_features_by_domain`` method, allowing easier selection of multiple feature domain combinations

- Improvements
    - Significantly reduced the computational time for the LPCC feature (`#156 <https://github.com/fraunhoferportugal/tsfel/pull/153>`_)
    - Resolved deprecation issues with SciPy Wavelets by switching to PyWavelets for features that rely on wavelets (`#147 <https://github.com/fraunhoferportugal/tsfel/pull/147>`_)
    - Renamed the ``fft_mean_coefficient`` feature to ``spectrogram_mean_coefficient`` for descriptive correctness (`#145 <https://github.com/fraunhoferportugal/tsfel/pull/145>`_)

- Bugfixes
    - Fixed a bug causing a circular import issue (`#154 <https://github.com/fraunhoferportugal/tsfel/pull/154>`_)
    - Fixed a ResourceWarning when loading the feature configuration file (`#152 <https://github.com/fraunhoferportugal/tsfel/pull/152>`_)
    - Removed the use of ``eval`` (`#150 <https://github.com/fraunhoferportugal/tsfel/pull/150>`_)

- Documentation
    - Major documentation updates, including detailed explanations of the expected input and output data formats

Version 0.1.7
=============
- New features
    - Implemented the Lempel-Ziv-Complexity in the temporal domain (`#146 <https://github.com/fraunhoferportugal/tsfel/pull/146>`_)
    - Added the fractal domain with the following features (`#144 <https://github.com/fraunhoferportugal/tsfel/pull/144>`_):
        - Detrended fluctuation analysis (DFA)
        - Higuchi fractal dimension
        - Hurst exponent
        - Maximum fractal length
        - Multiscale entropy (MSE)
        - Petrosian fractal dimension

- Changes
    - Changed the ``autocorrelation`` logic. It now measures the first lag below (1/e) from the ACF (`#142 <https://github.com/fraunhoferportugal/tsfel/issues/142>`_).

Version 0.1.6
=============
- Changes
    - Feature ``total energy`` changed name to ``average power``
    - Features ``peak to peak``, ``absolute energy`` and ``entropy`` are now classified as statistical

- Bugfixes
    - Fixed a bug on numpy bool usage (`#133 <https://github.com/fraunhoferportugal/tsfel/issues/133>`_)
    - Fixed a bug on features' header names

- Improvements
    - Correlated features are now computed using absolute value
    - Unit tests improvements
    - Refactoring of some code sections and overall improved stability\


Version 0.1.5
=============
-  Bugfixes
   - Fixed a bug on scipy function median_absolute_deviation to median_abs_deviation (`#128 <https://github.com/fraunhoferportugal/tsfel/pull/128>`_)
   - Fixed on pandas function df.append to pd.concat (`#120 <https://github.com/fraunhoferportugal/tsfel/pull/120>`_)


Version 0.1.4
=============
- Bugfixes
    - Fixed a bug on the progress bar not being displayed if the signal is passed already divided into windows (`#49 <https://github.com/fraunhoferportugal/tsfel/issues/49>`_)
    - Fixed a bug on the ``distance`` feature (`#54 <https://github.com/fraunhoferportugal/tsfel/issues/54>`_)
    - Fixed a bug raising zero division in the ECDF slope feature (`#57 <https://github.com/fraunhoferportugal/tsfel/pull/57>`_)
    - Fixed a bug when adding customised features using the JSON
    - Fixed a bug on LPC was returning inconsistent values (`#58 <https://github.com/fraunhoferportugal/tsfel/pull/58>`_)
    - Fixed a bug on normalised autocorrelation (`#64 <https://github.com/fraunhoferportugal/tsfel/pull/64>`_)

- Improvements
    - Refactoring of some code sections and overall improved stability
    - The documentation has been improved and a FAQ section was created
    - The ``window_splitter`` parameter is now deprecated. If the user selected a ``window_size`` it is assumed that the signal must be divided into windows.
    - Unit tests improvements

- New features
    - Added to return the size of the feature vector from the configuration dictionary (`#50 <https://github.com/fraunhoferportugal/tsfel/issues/50>`_)


Version 0.1.3
=============
- Bugfixes
    - Bug fixes on computational complexity calculation (`#15 <https://github.com/fraunhoferportugal/tsfel/pull/15>`_)
    - Fixed an error on `lpcc` feature (`#38 <https://github.com/fraunhoferportugal/tsfel/pull/38>`_)
    - Removed `entropy` warning (`#38 <https://github.com/fraunhoferportugal/tsfel/pull/38>`_)

- Improvements
    - Code cleaning on (`TSFEL_HAR_Example.ipynb <https://github.com/fraunhoferportugal/tsfel/blob/development/notebooks/TSFEL_HAR_Example.ipynb>`_)
    - `ecdf` code cleaning and computational optimization
    - Overlap value is now optional and set to default as 0
    - Unit test improvements
    - Nomenclature review of peak-related features

- New features:
    - Added new tutorials based on Jupyter notebooks (`#19 <https://github.com/fraunhoferportugal/tsfel/issues/19>`_)
    - Added progress bar during feature extraction (`#16 <https://github.com/fraunhoferportugal/tsfel/issues/16>`_)
    - Implemented multiprocessing. The `n_jobs` kwarg selects the number of CPUs to be scheduled (`#30 <https://github.com/fraunhoferportugal/tsfel/pull/30>`_)
    - Added the `neighbourhood_peaks` feature


Version 0.1.1
=============

- Added new features
    - Empirical cumulative distribution function
    - Empirical cumulative distribution function percentile
    - Empirical cumulative distribution function slope
    - Empirical cumulative distribution function percentile count
    - Spectral entropy
    - Wavelet entropy
    - Wavelet absolute mean
    - Wavelet standard deviation
    - Wavelet variance
    - Wavelet energy

- Minor fixes for Google Colab


Version 0.1.0
=============

- Release of TSFEL with documentation
