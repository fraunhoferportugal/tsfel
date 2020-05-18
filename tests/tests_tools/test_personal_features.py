from tsfel.feature_extraction.features_utils import set_domain
import numpy as np


@set_domain("domain", "statistical")
def new_feature(signal):
    """Computes a new feature
    Parameters
    ----------
    signal : nd-array
        Input from which new feature is computed

    Returns
    -------
    float
        new feature
    """
    return np.mean(signal)-np.std(signal)


@set_domain("domain", "statistical")
def new_feature_with_parameter(signal, weight=0.5):
    """A new feature
    Parameters
    ----------
    signal : nd-array
        Input from which new feature is computed
    weight : float
        float percentage
    Returns
    -------
    float
        new feature with parameter
    """
    return np.mean(signal)-np.std(signal)*weight
