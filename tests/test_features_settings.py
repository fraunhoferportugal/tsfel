import tsfel


FEATURES_JSON = 'features.json'

settings0 = tsfel.load_user_settings(FEATURES_JSON)

settings1 = tsfel.get_features_by_domain('Statistical')

settings2 = tsfel.get_features_by_domain('Temporal')

settings3 = tsfel.get_features_by_domain('Spectral')

settings4 = tsfel.get_all_features()

