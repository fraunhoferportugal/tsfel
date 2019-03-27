import pandas as pd
import numpy as np
from tsfel.utils.read_json import feat_extract


def window_spliter(signal, window_size, overlap):
    """

    :param signal: input signal
    :param window_size: number of points of window size
    :param overlap: percentage of overlap. Value between 0 and 1
    :return: list of signal windows
    """
    step = int(round(window_size)) if overlap == 0 else int(round(window_size * (1 - overlap)))
    return [signal[i:i + window_size] for i in range(0, len(signal) - window_size, step)]


def extract_features(sig, label, cfg, fs, filename='Features.csv'):
    """

    :param sig: list of signal windows
    :param label: label name to be concatenated with feature name
    :param cfg: dictionary with selected features from json file
    :param fs: sampling frequency
    :param filename: optional parameter; filename for the output file with features values
    :return: features values for each window size
    """
    feat_val = None
    feature_label = None
    print("*** Feature extraction started ***")
    for wind_idx, wind_sig in enumerate(sig):
        row_idx, feature_label = feat_extract(cfg, wind_sig, label, FS=fs)
        feat_val = row_idx if wind_idx == 0 else np.vstack((feat_val, row_idx))
    feat_val = np.array(feat_val)
    d = {str(lab): feat_val[:,idx] for idx, lab in enumerate(feature_label)}
    df = pd.DataFrame(data=d)

    df.to_csv(filename, sep=',', encoding='utf-8', index_label="Sample")
    print("*** Feature extraction finished ***")

    return df
