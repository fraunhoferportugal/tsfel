import json
import tsfel


def load_json(json_path):
    """Loads the json file given by filename.

    Parameters
    ----------
    json_path : string
        Json path

    Returns
    -------
    Dict
        Dictionary

    """

    return json.load(open(json_path))


def get_features_by_domain(domain=None, json_path=None):
    """Creates a dictionary with the features settings by domain.

    Parameters
    ----------
    domain : string
        Available domains: "statistical"; "spectral"; "temporal"
        If domain equals None, then the features settings from all domains are returned.
    json_path : string
        Directory of json file. Default: package features.json directory

    Returns
    -------
    Dict
        Dictionary with the features settings

    """

    if json_path is None:
        json_path = tsfel.__path__[0]+"/feature_extraction/features.json"

        if domain not in ['statistical', 'temporal', 'spectral', None]:
            raise SystemExit(
                'No valid domain. Choose: statistical, temporal, spectral or None (for all feature settings).')

    dict_features = load_json(json_path)
    if domain is None:
        return dict_features
    else:
        return {domain: dict_features[domain]}





