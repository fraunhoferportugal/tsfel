import ast
import json

import gspread
import numpy as np
from oauth2client.service_account import ServiceAccountCredentials
from tsfel.utils.calculate_complexity import compute_complexity

import tsfel


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
        cost_hidden = list(np.concatenate([['Constant', 'Log'] if int(cn) == 1 else
                                           ['Squared', 'Nlog'] if int(cn) == 3 else ['Linear']
                                           if int(cn) == 2 else ['Unknown'] for cn in cost_numbers]))
        cost_shown = []
        for dk in features.keys():
            cost_shown += [ff for ff in features[dk].keys() if features[dk][ff]['Complexity'] not in cost_hidden]
    features_filtered = list(np.concatenate([list(features[dk].keys())
                                             for dk in sorted(features.keys()) if dk in list_shown]))
    features_filtered = [ff for ff in features_filtered if ff in feat_shown]
    features_filtered = [cc for cc in features_filtered if cc in cost_shown]

    return features_filtered


def extract_sheet(gsheet_name):
    """Interaction between features.json and Google sheets.

    Parameters
    ----------
    gsheet_name : str
        Google Sheet name

    Returns
    -------
    dict
        Features

    """
    # path to Tsfel
    lib_path = tsfel.__path__

    # Access features.json

    path_json = lib_path[0] + '/feature_extraction/features.json'
    default = {'use': 'yes', 'metric': 'euclidean', 'free parameters': '', 'number of features': 1}

    # Read features.json into a dictionary of features and parameters
    dict_features = json.load(open(path_json))

    for _type in dict_features.keys():
        domain_feats = dict_features[_type].keys()
        for feat in domain_feats:
            # Concatenate two dictionaries
            dict_features[_type][feat] = dict(list(default.items()) + list(dict_features[_type][feat].items()))

    len_stat = len(dict_features['Statistical'].keys())
    len_temp = len(dict_features['Temporal'].keys())
    len_spec = len(dict_features['Spectral'].keys())

    # Access Google sheet

    # Scope and credentials using the content of client_secret.json file
    scope = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name(lib_path[0] + '/utils/client_secret.json', scope)

    # create a gspread client authorizing it using those credentials.
    client = gspread.authorize(creds)

    # and pass it to the spreadsheet name and getting access to sheet1
    confManager = client.open(gsheet_name)
    sheet = confManager.sheet1
    metadata = confManager.fetch_sheet_metadata()

    # Features from Google sheet
    list_of_features = sheet.col_values(2)[4:]
    list_domain = sheet.col_values(3)[4:]
    filters = metadata['sheets'][sheet.id]['basicFilter']['criteria']
    list_filt_features = filter_features(dict_features, filters)

    use_or_not = ['TRUE' if lf in list_filt_features else 'FALSE' for lf in list_of_features]

    assert len(list_of_features) <= (len_spec + len_stat + len_temp), \
        "To insert a new feature, please add it to data/features.json with the code in src/utils/features.py"

    # adds a new feature in Google sheet if it is missing from features.json
    if len(list_of_features) < (len_spec + len_stat + len_temp):

        # new feature was added
        for domain in dict_features.keys():
            for feat in dict_features[domain].keys():
                if feat not in list_of_features:
                    feat_dict = dict_features[domain][feat]
                    param = ''
                    if feat_dict['free parameters']:
                        param = str({"nbins": [10], 'r': [1]})
                    if feat_dict['fs'] != '':
                        param = str({"fs": 100})
                    curve = feat_dict['Complexity']
                    curves_all = ['Linear', 'Log', 'Square', 'Nlog', 'Constant']
                    complexity = compute_complexity(feat, domain,
                                                    path_json) if curve not in curves_all else 1 if curve in [
                        'Constant', 'Log'] else 2 if curve == 'Linear' else 3
                    new_feat = ['', feat, domain, complexity, param,
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
        list_domain = sheet.col_values(3)[4:]

        # Update filtered features from Google sheet
        filters = metadata['sheets'][sheet.id]['basicFilter']['criteria']
        list_filt_features = filter_features(dict_features, filters)
        use_or_not = ['TRUE' if lf in list_filt_features else 'FALSE' for lf in list_of_features]

    assert 'TRUE' in use_or_not, 'Please select a feature to extract!' + '\n'

    # Update dict of features with changes from Google sheet
    for ii, feature in enumerate(list_of_features):
        domain = list_domain[ii]
        if use_or_not[ii] == 'TRUE':
            dict_features[domain][feature]['use'] = 'yes'

            # Check for histogram free parameters: nbins and r
            if feature == 'Histogram':
                if sheet.cell(ii + 5, 5).value != '':
                    val = sheet.cell(ii + 5, 5).value

                    # update dic of features based on Google sheet histogram parameters
                    dict_features[domain][feature]['free parameters'] = \
                        {'nbins': ast.literal_eval(val)['nbins'], "r": ast.literal_eval(val)['r']}

            # Check features that use sampling frequency parameter
            if dict_features[domain][feature]['fs'] != '':
                if (sheet.cell(4, 8).value != '') or (type(sheet.cell(4, 8).value) == str):
                    # update dict of features based on Google sheet fs
                    val = sheet.cell(4, 8).value
                    dict_features[domain][feature]['fs'] = val

                    # update fs parameter in Google sheet
                    param = str({"fs": int(val)})
                    sheet.update_cell(ii + 5, 5, param)
        else:
            dict_features[domain][feature]['use'] = 'no'

    return dict_features
