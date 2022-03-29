import tsfel
from tsfel.feature_extraction.features_settings import get_features_by_domain, get_features_by_tag, load_json

FEATURES_JSON = tsfel.__path__[0] + "/feature_extraction/features.json"

settings0 = load_json(FEATURES_JSON)

settings1 = get_features_by_domain("statistical")
settings2 = get_features_by_domain("temporal")
settings3 = get_features_by_domain("spectral")
settings4 = get_features_by_domain(None)
# settings5 = tsfel.extract_sheet('Features')
settings6 = get_features_by_tag("audio")
settings7 = get_features_by_tag("inertial")
settings8 = get_features_by_tag("ecg")
settings9 = get_features_by_tag("eeg")
settings10 = get_features_by_tag("emg")
settings11 = get_features_by_tag(None)
