from numpy.testing import assert_array_equal, run_module_suite
import numpy as np
import pandas as pd
from TSFEL import*

# Get dictionary
FEATURES_JSON = 'TSFEL/tests/test_features.json'
DEFAULT = {'use': 'yes', 'metric': 'euclidean', 'free parameters': '', 'number of features': 1, 'parameters': ''}
cfg_file = compute_dictionary(FEATURES_JSON, DEFAULT)


def test_get_features():
    # Get features
    const1 = np.ones(20)

    X_train = extract_features(const1, 'const1', cfg_file, segment=True)
    # X_train.to_csv('TSFEL/tests/test_get_features_result', sep=',')

    test_features = pd.read_csv('TSFEL/tests/test_get_features_result', sep=',', index_col=0)
    pd.testing.assert_frame_equal(X_train, test_features)

#if __name__ == "__main__":
    #run_module_suite()
run_module_suite()