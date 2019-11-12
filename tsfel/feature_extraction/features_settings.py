import json
import tsfel
import inspect


def compute_dictionary(features_json, default):
    """
    This function computes the dictionary
    :param features_json: (json file)
           list of features
    :param default: (json file)
           default features
    :return: dictionary: (json file)
             complete dictionary
    """
    # TODO: REMOVE

    data = json.load(open(features_json))

    domain = data.keys()
    for atype in domain:
        domain_feats = data[atype].keys()
        for feat in domain_feats:
            # Concatenate two dictionaries
            data[atype][feat] = dict(list(default.items()) + list(data[atype][feat].items()))

    return data


def load_user_settings(filename):
    return json.load(open(filename))


def get_features_by_domain(domain):
    domain = domain.lower()
    settings = {domain: {}}
    for fname, f in tsfel.features.__dict__.items():
        if getattr(f, "domain", None) == domain:
            settings[domain][fname] = {}
            args = inspect.getfullargspec(f).args
            defaults = inspect.getfullargspec(f).defaults

            if defaults is None:
                iterator = args[1:] if len(args) > 1 else ''
                free_param = ''
            else:
                iterator = '' if len(args) - len(defaults) == 1 else args[1:-len(defaults)]
                free_param = {str(fparam): defaults[i] for i, fparam in enumerate(args[-len(defaults):])}

            all_param = ''
            for param in iterator:
                all_param += param + ','

            settings[domain][fname]['fs'] = all_param[:-1]
            settings[domain][fname]['free parameters'] = free_param
            settings[domain][fname]['function'] = 'tsfel.' + fname
            settings[domain][fname]['use'] = 'yes'

    return settings


def get_all_features():
    settings = {'statistical': get_features_by_domain('statistical')['statistical'],
                'temporal': get_features_by_domain('temporal')['temporal'],
                'spectral': get_features_by_domain('spectral')['spectral']}
    return settings