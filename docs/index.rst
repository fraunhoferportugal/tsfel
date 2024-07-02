Welcome to TSFEL documentation!
===============================

.. image:: imgs/tsfel_logo.png
    :align: center
    :scale: 35 %
    :alt: TSFEL!

**Time Series Feature Extraction Library (TSFEL)** is a Python package for efficient feature extraction from time series data. It offers a comprehensive set of feature extraction routines without requiring extensive programming effort. ``TSFEL`` automatically extracts over 65 features spanning statistical, temporal, spectral, and fractal domains.

The ``TSFEL`` project began in 2019 intending to centralize development in feature extraction methods for time series data, applicable across various fields including healthcare and industry. ``TSFEL`` is currently being used in academic and industrial projects, demonstrating its wide-ranging applicability. Built by data scientists for data scientists, ``TSFEL`` aims to streamline and enhance feature extraction processes.

Highlights
==========

- **Intuitive, fast deployment, and reproducible**: Easily configure your feature extraction pipeline and store the configuration file to ensure reproducibility.
- **Computational complexity evaluation**: Estimate the computational time required for feature extraction in advance.
- **Comprehensive documentation**: Each feature extraction method is accompanied by a detailed explanation.
- **Unit tested**: We provide an extensive suite of unit tests for each feature to ensure accurate and reliable feature calculation.
- **Easily extended**: Adding new features is straightforward, and we encourage contributions of custom features to the community.

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
