=====================
Personalised features
=====================

You can add your own features to TSFEL and extract them together with the built-in ones.
It takes three steps.

1. Write your feature
---------------------

Create a Python file, for example ``my_features.py``, with one function per feature:

.. code-block:: python

    import numpy as np

    from tsfel import set_domain


    @set_domain("domain", "statistical")
    def crest_factor(signal):
        """Computes the ratio between the signal's peak and its RMS value."""
        return np.max(np.abs(signal)) / np.sqrt(np.mean(signal**2))

A few rules:

- The first argument is the signal, as a one-dimensional NumPy array.
- The ``@set_domain`` decorator sets the domain: ``"statistical"``, ``"temporal"``,
  ``"spectral"``, ``"fractal"``, or a name of your own.
- If the feature needs the sampling frequency, call the argument ``fs``.
- Give every other argument a default value, for example ``threshold=0.5``.
- Use a function name that TSFEL doesn't already use, or yours replaces the built-in feature.

2. Register it
--------------

Copy TSFEL's configuration file and add your feature to it:

.. code-block:: python

    import shutil
    from pathlib import Path

    import tsfel

    config_dir = Path(tsfel.__file__).parent / "feature_extraction"
    shutil.copy(config_dir / "features.json", "my_features.json")

    tsfel.add_feature_json("my_features.py", "my_features.json")

TSFEL prints ``Feature crest_factor was added.`` Run ``add_feature_json`` again whenever you
add or rename a feature.

3. Extract it
-------------

Load your configuration with ``json_path``, and pass your Python file as ``features_path``:

.. code-block:: python

    cfg = tsfel.get_features_by_domain(
        "statistical", json_path="my_features.json"
    )
    X = tsfel.time_series_features_extractor(
        cfg, signal, fs=100, features_path="my_features.py"
    )

Your feature appears in ``X`` next to the built-in ones, as ``0_crest_factor``.

.. note::

    If you split the signal into windows (``window_size``) in a script, put the extraction
    under ``if __name__ == "__main__":``. TSFEL processes windows in parallel, and without
    it the script can hang.

Share it with others
--------------------

If your feature could be useful to others, add it to TSFEL: put the function in
``tsfel/feature_extraction/features.py``, add an entry to
``tsfel/feature_extraction/features.json`` following the existing ones, add a test to
``tests/test_features.py``, and open a pull request against the ``dev`` branch on
`GitHub <https://github.com/fraunhoferportugal/tsfel>`_.
