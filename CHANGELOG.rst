=========
Changelog
=========


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
