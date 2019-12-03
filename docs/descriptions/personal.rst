=====================
Personalised features
=====================

TSFEL provides a comprehensive set of features that can be applied in time series across several domains.

We also provide flexibility for users who desire to add personalised features to those that are already available in the library.
Personalised features are saved locally and users who want to share their implementations can create a pull request. Contributions are more than welcome and greatly appreciated!

1. Implement your feature
-------------------------

Create a new Python file for your features in any directory.

An example of a feature implementation format is shown below.

.. code:: python

    from tsfel.features_utils import set_domain

    @set_domain("domain", "temporal")
    def custom_feature(signal, parameters):
        """Description of your feature.

        Parameters
        ----------
        signal:
            The time series to calculate the feature of.
        param:
            Parameters of your feature (optional)
        Returns
        -------
        float
            Calculated feature

        """
        # Feature implementation

        return feature

The available domains are *statistical*, *temporal* and *spectral*. Also, note that TSFEL uses the ``fs`` variable to represent the sampling frequency parameter. Use this notation to take full advantage of the library for features which require the sampling frequency.

2. Add the new feature to features.json
---------------------------------------

After implementing your feature, use the ``add_feature_json`` function from ``tsfel.add_personal_features`` to store your personal feature information. This method adds the metadata from previously implemented feature to a JSON file which is used as configurator in TSFEL.

A JSON example file is available `here <https://github.com/fraunhoferportugal/tsfel/blob/development/tsfel/feature_extraction/features.json>`_.

.. code:: python

    def add_feature_json(features_path, json_path):
    """Adds new feature to features.json.

    Parameters
    ----------
    features_path: string
        Personal Python module directory containing new features implementation.

    json_path: string
        Personal .json file directory containing existing features from TSFEL.
        New customised features will be added to file in this directory.

    """

3. Extract your feature
-----------------------

The newly implemented feature is ready to be extracted. You must pass as an argument the path of the script where the implemented features in step 1 reside (``features_path``). TSFEL will do the rest for you.

.. code:: python

	features = tsfel.time_series_features_extractor(dict_features, signal_windows, fs=None, window_spliter=False, personal_dir=my_dir)


4. Create a pull request (optional)
-----------------------------------

If you would like to contribute to TSFEL growth, remember to add a pull request on our GitHub page with your feature implementation.

To add your feature to TSFEL complete the following steps:

1. Fork TSFEL's Development branch

2. Add your feature implementation to `/tsfel/feature_extraction/features.py <https://github.com/fraunhoferportugal/tsfel/blob/development/tsfel/feature_extraction/features.py>`_

3. Add documentation for the new feature in `/tsfel/feature_extraction/features.json <https://github.com/fraunhoferportugal/tsfel/blob/development/tsfel/feature_extraction/features.json>`_

4. Develop unit tests for the new feature in `/tests/test_features.py <https://github.com/fraunhoferportugal/tsfel/blob/development/tests/test_features.py>`_

5. Create a pull request in `GitHub <hhttps://github.com/fraunhoferportugal/tsfel>`_
