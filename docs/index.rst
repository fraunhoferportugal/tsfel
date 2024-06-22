Welcome to TSFEL documentation!
===============================

.. image:: imgs/tsfel_logo.png
    :align: center
    :scale: 35 %
    :alt: TSFEL!

Time Series Feature Extraction Library (TSFEL for short) is a Python package for feature extraction on time series data. It provides exploratory feature extraction tasks on time series without requiring significant programming effort. ``TSFEL`` automatically extracts over 60 different features on the statistical, temporal, spectral and fractal domains. ``TSFEL`` was built by data scientists for data scientists!

Highlights
==========

- **Intuitive, fast deployment and reproducible**: interactive UI for feature selection and customization
- **Computational complexity evaluation**: estimate the computational effort before extracting features
- **Comprehensive documentation**: each feature extraction method has a detailed explanation
- **Unit tested**: we provide unit tests for each feature
- **Easily extended**: adding new features is easy and we encourage you to contribute with your custom features

Installation
============

Installation can be easily done with ``pip``:

.. code:: bash

    $ pip install tsfel

Get started
===========

The code below extracts all the available features on an example dataset.

.. code:: python

    import tsfel
    import pandas as pd

    # load dataset
    data = tsfel.datasets.load_biopluxecg()

    # Retrieves a pre-defined feature configuration file to extract the temporal, statistical and spectral feature sets
    cfg = tsfel.get_features_by_domain()

    # Extract features
    X = tsfel.time_series_features_extractor(cfg, data)

How to cite TSFEL?
==================

.. admonition:: Note

   Used TSFEL in your research? Please cite us in your publication!
   Click :ref:`here<authors>` for further details.

Contents
========

.. toctree::
   :maxdepth: 2

   Get Started <descriptions/get_started>
   Feature List <descriptions/feature_list>
   Personalised Features <descriptions/personal>
   Frequently Asked Questions <descriptions/faq>
   Module Reference <descriptions/modules>
   Authors <authors>
   Changelog <changelog>
   License <license>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
