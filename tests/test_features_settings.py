import tsfel
from tsfel.feature_extraction.features_settings import (
    get_features_by_domain,
    get_features_by_tag,
    load_json,
)

FEATURES_JSON = tsfel.__path__[0] + "/feature_extraction/features.json"

settings0 = load_json(FEATURES_JSON)

settings01 = get_features_by_domain("statistical")
settings02 = get_features_by_domain("temporal")
settings03 = get_features_by_domain("spectral")
settings04 = get_features_by_domain(None)
settings05 = get_features_by_tag("audio")
settings06 = get_features_by_tag("inertial")
settings07 = get_features_by_tag("ecg")
settings08 = get_features_by_tag("eeg")
settings09 = get_features_by_tag("emg")
settings10 = get_features_by_tag(None)
