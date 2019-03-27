from numpy.testing import assert_array_equal, run_module_suite
import numpy as np
import pandas as pd
import tsfel

# Get dictionary
FEATURES_JSON = 'test_features.json'
DEFAULT = {'use': 'yes', 'metric': 'euclidean', 'free parameters': '', 'number of features': 1, 'parameters': ''}
cfg_file = tsfel.compute_dictionary(FEATURES_JSON, DEFAULT)


def test_get_features():
    # Get features
    signal = np.ones(100)

    signal_windows = tsfel.window_spliter(signal, 20, 0.5)

    X_train = tsfel.extract_features(signal_windows, 'x', cfg_file, fs=20, filename='test_get_features_result.csv')

    # test_features = pd.read_csv('test_get_features_result.csv', sep=',', index_col=0)
    # pd.testing.assert_frame_equal(X_train, test_features)

# run_module_suite()
