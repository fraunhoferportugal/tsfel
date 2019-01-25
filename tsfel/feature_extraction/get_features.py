import pandas as pd
import numpy as np
#import src.feature_extraction.utils.read_json as rj
from TSFEL.tsfel.utils.read_json import feat_extract

def extract_features(sig, label, cfg, segment=True, window_size=5):
    """ Performs a forward feature selection.
    Parameters
    ----------
    X_train: array-like
      train set features
    X_test: array-like
      test set features
    y_train: array-like
      train set labels
    y_test: array-like
      test set labels
    y_test: array-like
      test set labels
    features_descrition: list of strings
      list with extracted features names
    classifier: object
      classifier object
    Returns
    -------
    signal distance: if the signal was straightened distance
    """
    feat_val = None
    labels = None
    print("*** Feature extraction started ***")
    if segment:
        sig = [sig[i:i + window_size] for i in range(0, len(sig), window_size)]
    for wind_idx, wind_sig in enumerate(sig):
        row_idx, labels = feat_extract(cfg, wind_sig, label)
        if wind_idx == 0:
            feat_val = row_idx
        else:
            feat_val = np.vstack((feat_val, row_idx))
    feat_val = np.array(feat_val)
    d = {str(lab): feat_val[:,idx] for idx, lab in enumerate(labels)}
    df = pd.DataFrame(data=d)
    df.to_csv('TSFEL/tsfel/utils/Features.csv', sep=',', encoding='utf-8', index_label="Sample")
    print("*** Feature extraction finished ***")

    return df
