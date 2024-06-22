===========
Get Started
===========

Overview
--------

Time series are passed as inputs for the main ``TSFEL`` extraction method, either as variables previously loaded in memory or stored in files within a dataset. Since ``TSFEL`` can handle multidimensional time series, a set of preprocessing methods is applied to ensure adequate signal quality and time series synchronization, enabling accurate window calculation.

After feature extraction, the results are saved using a standard schema, making them compatible with most classification and data mining platforms. Each line corresponds to a window, with the results of the feature extraction methods stored in the corresponding columns.

.. image:: ../imgs/tsfel_pipeline.png
    :align: center
    :scale: 25 %
    :alt: TSFEL!

Extract from time series stored in DataFrames, Series or ndarrays
-----------------------------------------------------------------

Let's start by downloading some data. The following method returns a single-lead electrocardiogram (ECG) recorded at 100 Hz for a duration of 10 seconds.

.. code:: python

    import tsfel
    import pandas as pd

    data = tsfel.datasets.load_biopluxecg()    # A single-lead ECG collected during 10 s at 100 Hz.

Let us look to the input data structure:

.. code:: python

    data.head()
    0    2.898565
    1    2.462342
    2    -0.513560
    3    -5.263333
    4    -8.934970
    dtype: float64

Our input data is a ``Series`` named *LeadII*. Note that ``TSFEL`` can also handle multidimensional time series. In such cases, you will need to include the additional time series as a ``DataFrame`` or ``ndarray``.

Now that we have the input data, we are ready for the feature extraction step. ``TSFEL`` relies on dictionaries to set up the configuration for extraction. We provide a set of template configuration dictionaries that can be used out of the box.

In this example, we will use the configuration that extracts all available features from the temporal, statistical, and spectral sets.

.. code:: python

    cfg = tsfel.get_features_by_domain() # Extracts the temporal, statistical and spectral feature sets.
    X = tsfel.time_series_feature_extractor(cfg, data, fs=100)
    X.shape    # (1, 165)

We now have ``X`` as the extracted feature vector, composed of 165 features calculated for the entire length of the input data.

Alternatively, if we are interested in performing window splitting before the feature extraction, we can divide the input data into shorter equal-length windows of size 100 (corresponding to 1 second).

.. code:: python

    cfg = tsfel.get_features_by_domain()    # Extracts the temporal, statistical and spectral feature sets.
    X = tsfel.time_series_feature_extractor(cfg, data, fs=100, window_size=100)    # Performs window splitting before feature extraction
    X.shape    # (10, 165)


Extract from time series stored in datasets
-------------------------------------------

In the previous section, we observed how TSFEL can be used for feature extraction on time series stored in memory. The process of training machine learning models requires significant amounts of data. Time series datasets are often organised in a multitude of different schemas defined by the entities who collected and curated the data.
TSFEL provides a method to increase flexibility when extracting features over multiple files stored in datasets. We provide below a list of assumptions when using this method and how TSFEL handles it:

* **Time series are stored on different file locations**

  * TSFEL crawls over a given dataset root directory and extracts features from all text files which match filenames provided by the user


* **Files store time series in delimited format**

  * TSFEL expects that the first column must contain the timestamp and following columns contain the time series values.


* **Files might not be syncronised in time**

  * TSFEL handles this assumption by conducting a linear interpolation to ensure all the time series are syncronised in time before feature extraction. The resampling frequency is set by the user.


The following code block extracts features on data residing over ``main_directory``, from all files named ``Accelerometer.txt``. Timestamps were recorded in nanoseconds and the resampling frequency is set to 100 Hz.

.. code:: python

  import tsfel

  main_directory = '/my_root_dataset_directory/'        # The root directory of the dataset
  output_directory = '/my_output_feature_directory/'    # The resulted file from the feature extraction will be saved on this directory

  data = tsfel.dataset_features_extractor(
                        main_directory, tsfel.get_features_by_domain(), search_criteria="Accelerometer.txt",
                        time_unit=1e-9, resample_rate=100, window_size=250,
                        output_directory=output_directory
         )

Set up the feature extraction config file
------------------------------------------
One of the main advantages of TSFEL is providing a large number of time series features out-of-the-box. Nevertheless, there are occasions where you might not be interested in extracting the complete set. Examples comprise scenarios where the models will be deployed in low-power embedded devices, or you simply want to be more specific in what features are extracted.

TSFEL divides the available features into three domains: statistical, temporal and spectral. The two methods to extract features explained above expect a configuration file - ``feat_dict`` - a dictionary containing which features and hyperparameters will be used.

Bellow, we list four examples to set up the configuration dictionary.

.. code:: python

  import tsfel

  cfg_file = tsfel.get_features_by_domain()               # All features will be extracted.
  cgf_file = tsfel.get_features_by_domain("statistical")  # All statistical domain features will be extracted
  cgf_file = tsfel.get_features_by_domain("temporal")     # All temporal domain features will be extracted
  cgf_file = tsfel.get_features_by_domain("spectral")     # All spectral domain features will be extracted
  cgf_file = tsfel.get_features_by_domain("fractal")      # All fractal domain features will be extracted

In case you want a customised set of features or a combination of features from several domains, you can need to edit the configuration dictionary (JSON). You must edit the value of the key ``use`` to ``yes`` or ``no`` as appropriate. You can load any of the previous configuration dictionaries and set to ``"use": "no"`` the features you are not interested in or edit a dictionary manually or programmatically and set the ``use`` as ``yes`` or ``no`` as appropriate. An example file is available  `here <https://github.com/fraunhoferportugal/tsfel/blob/development/tsfel/feature_extraction/features.json/>`_.
