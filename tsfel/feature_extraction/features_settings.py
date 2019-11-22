import json
import tsfel
import inspect


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


def get_features_by_domain(domain):
    """Creates a dictionary with the features settings by domain.

    Parameters
    ----------
    domain : string
        Available domains: "statistical"; "spectral"; "temporal"

    Returns
    -------
    Dict
        Dictionary with the features settings by domain

    """

    domain = domain.lower()
    settings = {domain: {}}
    for fname, f in tsfel.features.__dict__.items():
        if getattr(f, "domain", None) == domain:
            settings[domain][fname] = {}
            args = inspect.getfullargspec(f).args
            defaults = inspect.getfullargspec(f).defaults

            if defaults is None:
                iterator = args[1:] if len(args) > 1 else ''
                param = ''
            else:
                iterator = '' if len(args) - len(defaults) == 1 else args[1:-len(defaults)]
                param = {str(fparam): defaults[i] for i, fparam in enumerate(args[-len(defaults):])}

            all_param = ''
            for param in iterator:
                all_param += param + ','

            if param == 'fs':
                settings[domain][fname]['parameters'] = {str(param): None}
            else:
                settings[domain][fname]['parameters'] = param
            settings[domain][fname]['function'] = 'tsfel.' + fname
            settings[domain][fname]['use'] = 'yes'

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


