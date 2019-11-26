import os
import json
import inspect
from tsfel.utils.calculate_complexity import compute_complexity


def add_feature_json(domain, json_path, func, feat=''):
    """Adds new feature to features.json.

    Parameters
    ----------
    domain : str
        Feature domain
    json_path: json
        Personal .json file containing existing features from TSFEL.
        New customised features will be added to this file.
    func: func
        Feature function
    feat: str
        Feature name (optional)

    Returns
    -------
    dict
        Features dictionary with new feature added.

    """

    if domain not in ['statistical', 'temporal', 'spectral']:
        raise SystemExit('No valid domain. Choose: statistical, temporal or spectral.')

    # print(os.path.abspath(inspect.getfile(func)))
    personal_dir = os.path.abspath(inspect.getfile(func))
    # Access to personal features.json
    feat_json = json.load(open(json_path))

    # Feature specifications
    # Name
    name = func.__name__
    # Description
    if func.__doc__ is not None:
        descrip = func.__doc__.split("\n")[0]
    else:
        descrip = ""
    # Feature usage
    use = "yes"
    # Feature function arguments
    param_name = inspect.getfullargspec(func)[0]
    # Check if feature name is given
    if feat == '':
        feat = name

    # Access feature parameters
    if param_name != "":
        # Retrieve default values of arguments
        spec = inspect.getfullargspec(func)
        defaults = dict(zip(spec.args[::-1], (spec.defaults or ())[::-1]))
        defaults.update(spec.kwonlydefaults or {})

        for p in param_name[1:]:
            if p not in list(defaults.keys()):
                if p is 'fs':
                    # Assigning a default value for fs if not given
                    defaults[p] = 100
                else:
                    defaults[p] = None
        if len(defaults) == 0:
            defaults = ""
    else:
        defaults = ""

    new_feature = {"description": descrip,
                   "parameters": defaults,
                   "function": name,
                   "use": use
                   }
    # Check if domain exists
    try:
        feat_json[domain][feat] = new_feature
    except KeyError:
        feat_json[domain] = {feat: new_feature}

    # Write new feature on json file
    with open(json_path, "w") as fout:
        json.dump(feat_json, fout, indent=" ")

    # Calculate feature complexity
    compute_complexity(feat, domain, json_path, personal_dir=personal_dir)

    return feat_json

