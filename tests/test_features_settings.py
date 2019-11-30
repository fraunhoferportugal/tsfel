import tsfel


FEATURES_JSON = tsfel.__path__[0] + '/feature_extraction/features.json'

settings0 = tsfel.load_json(FEATURES_JSON)

settings1 = tsfel.get_features_by_domain('statistical')

settings2 = tsfel.get_features_by_domain('temporal')

settings3 = tsfel.get_features_by_domain('spectral')

settings4 = tsfel.get_features_by_domain(None)

settings5 = tsfel.extract_sheet('Features')