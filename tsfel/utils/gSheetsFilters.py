import gspread
from oauth2client.service_account import ServiceAccountCredentials
import numpy as np
import tsfel
import ast
from tsfel.feature_extraction.features_settings import compute_dictionary
from tsfel.utils.calculate_complexity import compute_complexity

def filter_features(dic, filters):
    """
    @brief: Filtering features based on feature selection from Google sheet
    :param dic: dict, dictionary with features
    :param filters: dict, dictionary with filters from Google sheets
    :return: dict, filtered dictionary
    """
    features_all = list(np.concatenate([list(dic[dk].keys()) for dk in sorted(dic.keys())]))
    list_shown, feat_shown = list(dic.keys()), features_all
    cost_shown = features_all
    if filters['2'] != {}:
        list_hidden = filters['2']['hiddenValues']
        list_shown = [dk for dk in dic.keys() if dk not in list_hidden]
    if filters['1'] != {}:
        feat_hidden = filters['1']['hiddenValues']
        feat_shown = [ff for ff in features_all if ff not in feat_hidden]
    if filters['3'] != {}:
        cost_numbers = filters['3']['hiddenValues']
        cost_hidden = list(np.concatenate([['Constant','Log'] if int(cn) == 1 else ['Squared','Nlog'] if int(cn) == 3 else ['Linear'] if int(cn) == 2 else ['Unknown'] for cn in cost_numbers]))
        cost_shown = []
        for dk in dic.keys():
            cost_shown += [ff for ff in dic[dk].keys() if dic[dk][ff]['Complexity'] not in cost_hidden]
    features_filtered = list(np.concatenate([list(dic[dk].keys()) for dk in sorted(dic.keys()) if dk in list_shown]))
    features_filtered = [ff for ff in features_filtered if ff in feat_shown]
    features_filtered = [cc for cc in features_filtered if cc in cost_shown]

    return features_filtered

def extract_sheet(gSheetName):
    """
    @brief: Interaction between features.json and Google sheets.
    :param gSheetName: string, Google Sheet name
    :return: dict, dictionary of features
    """
    # path to tsfel library
    lib_path = tsfel.__path__

    # Access to features.json

    FEATURES_JSON = lib_path[0] + '/feature_extraction/features.json'
    DEFAULT = {'use': 'yes', 'metric': 'euclidean', 'free parameters': '', 'number of features': 1}

    # Dict of features and parameters
    DICTIONARY = compute_dictionary(FEATURES_JSON, DEFAULT)

    len_stat = len(DICTIONARY['Statistical'].keys())
    len_temp = len(DICTIONARY['Temporal'].keys())
    len_spec = len(DICTIONARY['Spectral'].keys())

    # Access to Google sheet

    # Scope and credentials using the content of client_secret.json file
    scope = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name(lib_path[0] + '/utils/client_secret.json', scope)
    # create a gspread client authorizing it using those credentials.
    client = gspread.authorize(creds)
    # and pass it to the spreadsheet name and getting access to sheet1
    confManager = client.open(gSheetName)
    sheet = confManager.sheet1
    metadata = confManager.fetch_sheet_metadata()

    # Features from gsheets
    list_of_features = sheet.col_values(2)[4:]
    list_domain = sheet.col_values(3)[4:]
    filters = metadata['sheets'][sheet.id]['basicFilter']['criteria']
    list_filt_features = filter_features(DICTIONARY, filters)

    use_or_not = ['TRUE' if lf in list_filt_features else 'FALSE' for lf in list_of_features]

    assert len(list_of_features) <= (len_spec + len_stat + len_temp), \
        "To insert a new feature, please add it to data/features.json with the code in src/utils/features.py"

    # adds a new feature in gsheets if it is missing from features.json
    if len(list_of_features) < (len_spec + len_stat + len_temp):
        # new feature was added
        for domain in DICTIONARY.keys():
            for feat in DICTIONARY[domain].keys():
                if feat not in list_of_features:
                    feat_dict = DICTIONARY[domain][feat]
                    param = ''
                    if feat_dict['free parameters']:
                        param = str({"nbins": [10], 'r': [1]})
                    if feat_dict['fs'] != '':
                        param = str({"fs": 100})
                    curve = feat_dict['Complexity']
                    curves_all = ['Linear', 'Log', 'Square', 'Nlog', 'Constant']
                    complexity = compute_complexity(feat, domain,
                                                    FEATURES_JSON) if curve not in curves_all else 1 if curve in [
                        'Constant', 'Log'] else 2 if curve == 'Linear' else 3
                    new_feat = ['', feat, domain, complexity, param,
                                feat_dict['description']]
                    # checks if the gsheets has no features
                    if sheet.findall(domain) == []:
                        idx_row = 4
                    else:
                        idx_row = sheet.findall(domain)[-1].row

                    # Add new feature at the end of feature domain
                    sheet.insert_row(new_feat, idx_row + 1)
                    print(feat + " was added to gsheet.")

        # Update list of features an domain from gsheets
        list_of_features = sheet.col_values(2)[4:]
        list_domain = sheet.col_values(3)[4:]
        # Update filtered features from gsheets
        filters = metadata['sheets'][sheet.id]['basicFilter']['criteria']
        list_filt_features = filter_features(DICTIONARY, filters)
        use_or_not = ['TRUE' if lf in list_filt_features else 'FALSE' for lf in list_of_features]

    assert 'TRUE' in use_or_not, 'Please select a feature to extract!' + '\n'

    # Update dict of features with changes from gsheets
    for ii, feature in enumerate(list_of_features):
        domain = list_domain[ii]
        if use_or_not[ii] == 'TRUE':
            DICTIONARY[domain][feature]['use'] = 'yes'

            # Check for histogram free parameters: nbins and r
            if feature == 'Histogram':
                if sheet.cell(ii + 5, 5).value != '':
                    val = sheet.cell(ii + 5, 5).value
                    # update dic of features based on gsheets histogram parameters
                    DICTIONARY[domain][feature]['free parameters'] = {
                'nbins': ast.literal_eval(val)['nbins'], "r": ast.literal_eval(val)['r']}

            # Check features that use sampling frequency parameter
            if DICTIONARY[domain][feature]['fs'] != '':
                if (sheet.cell(4, 8).value != '') or (type(sheet.cell(4, 8).value) == str):
                    # update dict of features based on gsheet fs
                    val = sheet.cell(4, 8).value
                    DICTIONARY[domain][feature]['fs'] = val
                    # update fs parameter in gsheet
                    param = str({"fs": float(val)})
                    sheet.update_cell(ii + 5, 5, param)
        else:
            DICTIONARY[domain][feature]['use'] = 'no'
    return DICTIONARY
