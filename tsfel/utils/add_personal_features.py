import importlib
import inspect
import json
import os
import sys
import warnings
from inspect import getmembers, isfunction

from tsfel.feature_extraction.features_settings import load_json
from tsfel.utils.calculate_complexity import compute_complexity


def add_feature_json(features_path, json_path):
    """Adds new feature to features.json.

    Parameters
    ----------
    features_path: string
        Personal Python module directory containing new features implementation.

    json_path: string
        Personal .json file directory containing existing features from TSFEL.
        New customised features will be added to file in this directory.

    """

    sys.path.append(features_path[:-len(features_path.split(os.sep)[-1]) - 1])
    exec("import " + features_path.split(os.sep)[-1][:-3])

    # Reload module containing the new features
    importlib.reload(sys.modules[features_path.split(os.sep)[-1][:-3]])
    exec("import " + features_path.split(os.sep)[-1][:-3] + " as pymodule")

    # Functions from module containing the new features
    functions_list = [o for o in getmembers(locals()['pymodule']) if isfunction(o[1])]
    function_names = [fname[0] for fname in functions_list]

    # Check if @set_domain was declared on features module
    vset_domain = False

    for fname, f in list(locals()['pymodule'].__dict__.items()):

        if getattr(f, "domain", None) is not None:

            vset_domain = True

            # Access to personal features.json
            feat_json = load_json(json_path)

            # Assign domain and tag
            domain = getattr(f, "domain", None)
            tag = getattr(f, "tag", None)

            # Feature specifications
            # Description
            if f.__doc__ is not None:
                descrip = f.__doc__.split("\n")[0]
            else:
                descrip = ""
            # Feature usage
            use = "yes"
            # Feature function arguments
            args_name = inspect.getfullargspec(f)[0]

            # Access feature parameters
            if args_name != "":
                # Retrieve default values of arguments
                spec = inspect.getfullargspec(f)
                defaults = dict(zip(spec.args[::-1], (spec.defaults or ())[::-1]))
                defaults.update(spec.kwonlydefaults or {})

                for p in args_name[1:]:
                    if p not in list(defaults.keys()):
                        if p == 'fs':
                            # Assigning a default value for fs if not given
                            defaults[p] = 100
                        else:
                            defaults[p] = None
                if len(defaults) == 0:
                    defaults = ""
            else:
                defaults = ""

            # Settings of new feature
            new_feature = {"description": descrip,
                           "parameters": defaults,
                           "function": fname,
                           "use": use
                           }

            # Check if domain exists
            try:
                feat_json[domain][fname] = new_feature
            except KeyError:
                feat_json[domain] = {fname: new_feature}

            # Insert tag if it is declared
            if tag is not None:
                feat_json[domain][fname]['tag'] = tag

            # Write new feature on json file
            with open(json_path, "w") as fout:
                json.dump(feat_json, fout, indent=" ")

            # Calculate feature complexity
            compute_complexity(fname, domain, json_path, features_path=features_path)
            print('Feature '+str(fname)+' was added.')

    if vset_domain is False:
        warnings.warn('No features were added. Please declare @set_domain.', stacklevel=2)


