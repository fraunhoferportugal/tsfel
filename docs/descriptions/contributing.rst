============
Contributing
============

TSFEL provides a comprehensive set of features that can be applied in time series across several domains.

We also provide flexibility for users who desire to add personalised features to those that are already available in the library.
Personalised features are saved locally and users who want to share their implementations are welcomed to create a pull request. Contributions are more than welcome and greatly appreciated!

1. Implement your feature
-------------------------

Create a new Python file for your features in any directory. 

An example of a feature function implementation format is shown below.

.. code:: python

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

**Note:** TSFEL uses the ``fs`` variable to represent the sampling frequency parameter. Use this notation to take full advantage of the library for features which require the sampling frequency.

2. Add the new feature to features.json
---------------------------------------

After implementing your feature, use ``add_feature_json`` function, from ``tsfel.add_personal_features``, to store your personal feature information.
Make sure you have a copy of the ``feature.json`` file and use its directory to fill the *json_path* parameter.

.. code:: python

    def add_feature_json(domain, json_path, func, feat=''):
        """Adds new feature to features.json.

        Parameters
        ----------
        domain : str
	    Feature domain
        json_path: json
	    Personal .json file containing existing features from TSFEL.
	    New customised features will be added to this file.
        func: func
	    Feature function
        feat: str
	    Feature name (optional)

        Returns
        -------
        dict
	    Features dictionary with a new feature added.

        """

3. Extract your feature
----------------------- 

The newly implemented feature is ready to be extracted. You must pass as an argument the directory of the script where the implemented features in step 1 resides (``my_dir``). TSFEL will do the rest for you.

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

