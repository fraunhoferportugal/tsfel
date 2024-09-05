import json

import numpy as np

import tsfel
from tsfel.feature_extraction.features_utils import safe_eval_string


def load_json(json_path):
    """A convenient method that wraps the built-in `json.load`. This method
    might be handy to load customized feature configuration files.

    Parameters
    ----------
    json_path : file-like object, string, or pathlib.Path.
        The json file to read.

    Returns
    -------
    dict
        Data stored in the file.
    """
    with open(json_path) as f:
        out = json.load(f)

    return out


def get_features_by_domain(domain=None, json_path=None):
    """Creates a dictionary with the features settings by domain.

    Parameters
    ----------        
    domain : str, list of str, or None, default=None
        Specifies which feature domains to include in the dictionary.
            - 'statistical', 'temporal', 'spectral', 'fractal': Includes the corresponding feature domain.
            - 'all': Includes all available feature domains.
            - list of str: A combination of the above strings, e.g., ['statistical', 'temporal'].
            - None: By default, includes the 'statistical', 'temporal', and 'spectral' domains.
    json_path : string
        Directory of json file. Default: package features.json directory

    Returns
    -------
    Dict
        Dictionary with the features settings
    """

    valid_domains = ["statistical", "temporal", "spectral", "fractal", "all"]

    if json_path is None:
        json_path = tsfel.__path__[0] + "/feature_extraction/features.json"

        if isinstance(domain, str) and domain not in valid_domains:
            raise ValueError(
                f"Domain {domain} is invalid. Please choose from `statistical`, `temporal`, `spectral`, `fractal` or `all`.",
            )
        elif isinstance(domain, list) and not np.all([d in valid_domains for d in domain]):
            raise ValueError(
                "At least one invalid domain was provided. Please choose from `statistical`, `temporal`, `spectral`, `fractal` or `all`.",
            )
        elif not isinstance(domain, (str, list)) and domain is not None:
            raise TypeError(
                "The 'domain' argument must be a string or a list of strings.",
            )

    dict_features = load_json(json_path)
    if domain is None:
        return dict_features
    else:
        if domain == "all":
            domain = ["statistical", "temporal", "spectral", "fractal"]

        if isinstance(domain, str):
            domain = [domain]

        d_feat = {}
        for d in domain:
            if d == "fractal":
                for k in dict_features[d]:
                    dict_features[d][k]["use"] = "yes"
            d_feat.update({d: dict_features[d]})
        return d_feat


def get_features_by_tag(tag=None, json_path=None):
    """Creates a dictionary with the features settings by tag.

    Parameters
    ----------
    tag : string
        Available tags: "audio"; "inertial", "ecg"; "eeg"; "emg".
        If tag equals None then, all available features are returned.
    json_path : string
        Directory of json file. Default: package features.json directory

    Returns
    -------
    Dict
        Dictionary with the features settings
    """
    if json_path is None:
        json_path = tsfel.__path__[0] + "/feature_extraction/features.json"

        if tag not in ["audio", "inertial", "ecg", "eeg", "emg", None]:
            raise SystemExit(
                "No valid tag. Choose: audio, inertial, ecg, eeg, emg or None.",
            )
    features_tag = {}
    dict_features = load_json(json_path)
    if tag is None:
        return dict_features
    else:
        for domain in dict_features:
            features_tag[domain] = {}
            for feat in dict_features[domain]:
                if dict_features[domain][feat]["use"] == "no":
                    continue
                # Check if tag is defined
                try:
                    js_tag = dict_features[domain][feat]["tag"]
                    if isinstance(js_tag, list):
                        if any(tag in js_t for js_t in js_tag):
                            features_tag[domain].update(
                                {feat: dict_features[domain][feat]},
                            )
                    elif js_tag == tag:
                        features_tag[domain].update({feat: dict_features[domain][feat]})
                except KeyError:
                    continue
        # To remove empty dicts
        return {d: features_tag[d] for d in list(features_tag.keys()) if bool(features_tag[d])}


def get_number_features(dict_features):
    """Count the total number of features based on input parameters of each
    feature.

    Parameters
    ----------
    dict_features : dict
        Dictionary with features settings

    Returns
    -------
    int
        Feature vector size
    """
    number_features = 0
    for domain in dict_features:
        for feat in dict_features[domain]:
            if dict_features[domain][feat]["use"] == "no":
                continue
            n_feat = dict_features[domain][feat]["n_features"]

            if isinstance(n_feat, int):
                number_features += n_feat
            else:
                n_feat_param = dict_features[domain][feat]["parameters"][n_feat]
                if isinstance(n_feat_param, int):
                    number_features += n_feat_param
                else:
                    n_feat_param_list = safe_eval_string(n_feat_param)
                    number_features += len(n_feat_param_list)

    return number_features
