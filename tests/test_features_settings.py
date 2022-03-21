import tsfel

FEATURES_JSON = tsfel.__path__[0] + "/feature_extraction/features.json"

settings0 = tsfel.load_json(FEATURES_JSON)

settings1 = tsfel.get_features_by_domain("statistical")

settings2 = tsfel.get_features_by_domain("temporal")

settings3 = tsfel.get_features_by_domain("spectral")

settings4 = tsfel.get_features_by_domain("fractal")

settings5 = tsfel.get_features_by_domain(None)

settings6 = tsfel.extract_sheet("Features")

settings7 = tsfel.get_features_by_tag("audio")

settings8 = tsfel.get_features_by_tag("inertial")

settings9 = tsfel.get_features_by_tag("ecg")

settings10 = tsfel.get_features_by_tag("eeg")

settings11 = tsfel.get_features_by_tag("emg")

settings12 = tsfel.get_features_by_tag(None)
