import tsfel


FEATURES_JSON = tsfel.__path__ + '/feature_extraction/features.json'

settings0 = tsfel.load_user_settings(FEATURES_JSON)

settings1 = tsfel.get_features_by_domain('statistical')

settings2 = tsfel.get_features_by_domain('temporal')

settings3 = tsfel.get_features_by_domain('spectral')

settings4 = tsfel.get_all_features()

settings5 = tsfel.extract_sheet('Features')