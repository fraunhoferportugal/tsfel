from tsfel.feature_extraction.features_settings import load_user_settings, get_features_by_domain, get_all_features

FEATURES_JSON = 'features.json'

settings0 = load_user_settings(FEATURES_JSON)

settings1 = get_features_by_domain('Statistical')

settings2 = get_features_by_domain('Temporal')

settings3 = get_features_by_domain('Spectral')

settings4 = get_all_features()

