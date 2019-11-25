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


def get_features_by_domain(domain, filename=None):
    """Creates a dictionary with the features settings by domain.

    Parameters
    ----------
    domain : string
        Available domains: "statistical"; "spectral"; "temporal"
    filename : str
        Directory of json file. Default: package features.json directory

    Returns
    -------
    Dict
        Dictionary with the features settings by domain

    """

    if filename is None:
        filename = tsfel.__path__[0]+"/feature_extraction/features.json"

    dict_features = load_json(filename)
    settings = dict_features[domain]

    return settings


def get_all_features():
    """Creates a dictionary with the features settings from all domains.

    Returns
    -------
    Dict
        Dictionary with the features settings from all domains

    """

    settings = {'statistical': get_features_by_domain('statistical')['statistical'],
                'temporal': get_features_by_domain('temporal')['temporal'],
                'spectral': get_features_by_domain('spectral')['spectral']}

    return settings


