Welcome to TSFEL documentation!
===============================

.. image:: imgs/tsfel_logo.png
    :align: center
    :scale: 35 %
    :alt: TSFEL!

Time Series Feature Extraction Library (TSFEL for short) is a Python package for feature extraction on time series data. It aims to provide exploratory feature extraction tasks on time series without requiring significant programming effort. TSFEL automatically extracts over 50 different features on the statistical, temporal and spectral domains. TSFEL was built by data scientists for data scientists! 

Highlights
==========

- **Intuitive, fast deployment and reproducible**: interactive UI for feature selection and customization
- **Computational complexity evaluation**: estimate the computational effort before extracting features
- **Comprehensive documentation**: each feature extraction method has a detailed explanation
- **Unit tested**: we provide unit tests for each feature
- **Easily extended**: adding new features is easy and we encourage you to contribute with your custom features

Contents
========

.. toctree::
   :maxdepth: 2

   Get Started <descriptions/get_started>
   Feature List <descriptions/feature_list>
   Contributing <descriptions/contributing>
   Authors <authors>
   Changelog <changelog>
   License <license>

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
    df = pd.read_csv('Dataset.txt')

    # Retrieves a pre-defined feature configuration file to extract all available features 
    cfg = tsfel.get_all_features()

    # Extract features
    X = tsfel.time_series_features_extractor(cfg, df) 

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
