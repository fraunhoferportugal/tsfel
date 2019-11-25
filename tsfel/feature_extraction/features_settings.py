import json
import tsfel


def load_json(filename):
    """Loads the json file given by filename.

    Parameters
    ----------
    filename : json
        Json file

    Returns
    -------
    Dict
        Dictionary

    """

    return json.load(open(filename))


def get_features_by_domain(domain=None, filename=None):
    """Creates a dictionary with the features settings by domain.

    Parameters
    ----------
    domain : string
        Available domains: "statistical"; "spectral"; "temporal"
        If domain equals None, then the features settings from all domains are returned.
    filename : str
        Directory of json file. Default: package features.json directory

    Returns
    -------
    Dict
        Dictionary with the features settings

    """

    if domain not in ['statistical', 'temporal', 'spectral', None]:
        raise SystemExit('No valid domain. Choose: statistical, temporal, spectral or None (for all feature settings).')

    if filename is None:
        filename = tsfel.__path__[0]+"/feature_extraction/features.json"

    dict_features = load_json(filename)
    if domain is None:
        return dict_features
    else:
        return {domain: dict_features[domain]}



