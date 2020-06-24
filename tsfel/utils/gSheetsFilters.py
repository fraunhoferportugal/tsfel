import ast
import warnings

import gspread
import numpy as np
from oauth2client.service_account import ServiceAccountCredentials

import tsfel
from tsfel.feature_extraction.features_settings import load_json
from tsfel.utils.calculate_complexity import compute_complexity


def filter_features(features, filters):
    """Filtering features based on Google sheet.

    Parameters
    ----------
    features : dict
        Dictionary with features
    filters : dict
        Filters from Google sheets

    Returns
    -------
    dict
        Filtered features

    """
    features_all = list(np.concatenate([list(features[dk].keys()) for dk in sorted(features.keys())]))
    list_shown, feat_shown = list(features.keys()), features_all
    cost_shown = features_all
    if filters['2'] != {}:
        list_hidden = filters['2']['hiddenValues']
        list_shown = [dk for dk in features.keys() if dk not in list_hidden]
    if filters['1'] != {}:
        feat_hidden = filters['1']['hiddenValues']
        feat_shown = [ff for ff in features_all if ff not in feat_hidden]
    if filters['3'] != {}:
        cost_numbers = filters['3']['hiddenValues']
        cost_hidden = list(np.concatenate([['constant', 'log'] if int(cn) == 1 else
                                           ['squared', 'nlog'] if int(cn) == 3 else ['linear']
                                           if int(cn) == 2 else ['unknown'] for cn in cost_numbers]))
        cost_shown = []
        for dk in features.keys():
            cost_shown += [ff for ff in features[dk].keys() if features[dk][ff]['complexity'] not in cost_hidden]
    features_filtered = list(np.concatenate([list(features[dk].keys())
                                             for dk in sorted(features.keys()) if dk in list_shown]))
    features_filtered = [ff for ff in features_filtered if ff in feat_shown]
    features_filtered = [cc for cc in features_filtered if cc in cost_shown]

    return features_filtered


def extract_sheet(gsheet_name, **kwargs):
    """Interaction between features.json and Google sheets.

    Parameters
    ----------
    gsheet_name : str
        Google Sheet name
    \**kwargs:
    See below:
        * *path_json* (``string``) --
            Json path
    Returns
    -------
    dict
        Features

    """
    # Path to Tsfel
    lib_path = tsfel.__path__

    # Access features.json
    path_json = kwargs.get('path_json', lib_path[0] + '/feature_extraction/features.json')

    # Read features.json into a dictionary of features and parameters
    dict_features = load_json(path_json)

    # Number of features from json file
    len_json = 0
    for domain in list(dict_features.keys()):
        len_json += len(dict_features[domain].keys())

    # Access Google sheet
    # Scope and credentials using the content of client_secret.json file
    scope = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name(lib_path[0] + '/utils/client_secret.json', scope)

    # Create a gspread client authorizing it using those credentials
    client = gspread.authorize(creds)

    # and pass it to the spreadsheet name, getting access to sheet1
    confManager = client.open(gsheet_name)
    sheet = confManager.sheet1
    metadata = confManager.fetch_sheet_metadata()

    # Reading from Google Sheet
    # Features
    list_of_features = sheet.col_values(2)[4:]

    try:
        filters = metadata['sheets'][sheet.id]['basicFilter']['criteria']
        list_filt_features = filter_features(dict_features, filters)
    except KeyError:
        print('No filters running. Check Google Sheet filters.')
        list_filt_features = list_of_features.copy()

    use_or_not = ['TRUE' if lf in list_filt_features else 'FALSE' for lf in list_of_features]

    assert len(list_of_features) <= (len_json), \
        "To insert a new feature, please add it to data/features.json with the code in src/utils/features.py"

    # adds a new feature in Google sheet if it is missing from features.json
    if len(list_of_features) < (len_json):

        # new feature was added
        for domain in dict_features.keys():
            for feat in dict_features[domain].keys():
                if feat not in list_of_features:
                    feat_dict = dict_features[domain][feat]
                    param = ''
                    fs = 'no'

                    # Read parameters from features.json
                    if feat_dict['parameters']:
                        param = feat_dict['parameters'].copy()
                        if 'fs' in feat_dict['parameters']:
                            fs = 'yes'
                            param.pop('fs')
                            if len(param) == 0:
                                param = ''

                    curve = feat_dict['complexity']
                    curves_all = ['linear', 'log', 'squared', 'nlog', 'constant']
                    complexity = compute_complexity(feat, domain,
                                                    path_json) if curve not in curves_all else 1 if curve in [
                        'constant', 'log'] else 2 if curve == 'linear' else 3
                    new_feat = ['', feat, domain, complexity, fs, str(param),
                                feat_dict['description']]

                    # checks if the Google sheet has no features
                    if sheet.findall(domain) == []:
                        idx_row = 4
                    else:
                        idx_row = sheet.findall(domain)[-1].row

                    # Add new feature at the end of feature domain
                    sheet.insert_row(new_feat, idx_row + 1)
                    print(feat + " feature was added to Google Sheet.")

        # Update list of features and domains from Google sheet
        list_of_features = sheet.col_values(2)[4:]

        # Update filtered features from Google sheet. Check if filters exist.
        try:
            filters = metadata['sheets'][sheet.id]['basicFilter']['criteria']
            list_filt_features = filter_features(dict_features, filters)
        except KeyError:
            list_filt_features = list_of_features.copy()

        use_or_not = ['TRUE' if lf in list_filt_features else 'FALSE' for lf in list_of_features]

    assert 'TRUE' in use_or_not, 'Please select a feature to extract!' + '\n'

    # Reading from Google Sheet
    # Domain
    list_domain = sheet.col_values(3)[4:]
    # Parameters and fs
    gs_param_list = sheet.col_values(6)[4:]
    gs_fs_list = sheet.col_values(5)[4:]
    # Check for invalid fs parameter
    try:
        gs_fs = int(sheet.cell(4, 9).value)
    except ValueError:
        warnings.warn('Invalid sampling frequency. Setting a default 100Hz sampling frequency.')
        gs_fs = 100
        sheet.update_cell(4, 9, str(gs_fs))

    # Fix for empty cells in parameters column
    if len(gs_param_list) < len(list_of_features):
        empty = [''] * (len(list_of_features) - len(gs_param_list))
        gs_param_list = gs_param_list + empty

    # Update dict of features with changes from Google sheet
    for ii, feature in enumerate(list_of_features):
        domain = list_domain[ii]
        try:
            if use_or_not[ii] == 'TRUE':
                dict_features[domain][feature]['use'] = 'yes'
                # Check features parameters from Google sheet
                if gs_param_list[ii] != '':
                    if dict_features[domain][feature]['parameters'] == '' or ('fs' in list(
                            dict(dict_features[domain][feature]['parameters'])) and len(list(
                            dict(dict_features[domain][feature]['parameters']))) == 1):
                        warnings.warn('The ' + feature + ' feature does not require parameters.')
                    else:
                        try:
                            param_sheet = ast.literal_eval(gs_param_list[ii])
                            if not isinstance(param_sheet, dict):
                                warnings.warn('Invalid parameter format. Using the following parameters for ' + feature + ' feature: '
                                              + str(dict_features[domain][feature]['parameters']))
                            else:
                                # update dic of features based on Google sheet
                                dict_features[domain][feature]['parameters'] = param_sheet
                        except ValueError:
                            warnings.warn('Invalid parameter format. Using the following parameters for ' + feature + ' feature: '
                                          + str(dict_features[domain][feature]['parameters']))
                elif dict_features[domain][feature]['parameters'] != '' and ('fs' not in list(
                        dict(dict_features[domain][feature]['parameters'])) or len(list(
                        dict(dict_features[domain][feature]['parameters']))) != 1):
                    warnings.warn('Using the following parameters for ' + feature + ' feature: '
                                  + str(dict_features[domain][feature]['parameters']))
                # Check features that use sampling frequency parameter
                if gs_fs_list[ii] != 'no':
                    # update dict of features based on Google sheet fs
                    dict_features[domain][feature]['parameters']['fs'] = gs_fs

            else:
                dict_features[domain][feature]['use'] = 'no'
        except KeyError:
            print('Unknown domain at cell', int(ii + 5))

    return dict_features
