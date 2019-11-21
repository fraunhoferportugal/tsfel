import json
import glob
import tsfel
import pandas as pd
from pathlib import Path


# Example of user preprocess sensor data
def pre_process(sensor_data):
    if 'Accelerometer' in sensor_data:
        sensor_data['Accelerometer'].iloc[:, 1] = sensor_data['Accelerometer'].iloc[:, 1] * 0
    return sensor_data

# User dataset
main_directory = str(Path.home()) + "/Documents/Datasets/HumanPatterns/"
# Different ways to extract feature dictionary
path_json = tsfel.__path__[0] + '/feature_extraction/features.json'
settings0 = json.load(open(path_json))
settings1 = tsfel.get_features_by_domain('statistical')
settings2 = tsfel.get_features_by_domain('temporal')
settings3 = tsfel.get_features_by_domain('spectral')
settings4 = tsfel.get_all_features()
settings5 = tsfel.extract_sheet('Features')


# Feature Extraction using dataset_feature_extractor
time_unit = 1e9  # seconds
resample_rate = 30  # resample sampling frequency
window_size = 100  # number of points
overlap = 0  # varies between 0 and 1
search_criteria = ['Accelerometer.txt', 'Gyroscope.txt']
output_directory = str(Path.home()) + '/Documents/tsfel_output'
data = tsfel.dataset_features_extractor(main_directory, settings2, search_criteria=search_criteria, time_unit=time_unit,
                                        resample_rate=resample_rate, window_size=window_size, overlap=overlap,
                                        pre_process=pre_process, output_directory=output_directory)

# Feature Extraction using time_series_features_extractor
folders = [f for f in glob.glob(main_directory + "**/", recursive=True)]
sensor_data = {}
key = 'Accelerometer'
sensor_data[key] = pd.read_csv(folders[-1] + key + '.txt', header=None)

data_new = tsfel.merge_time_series(sensor_data, resample_rate, time_unit)

windows = tsfel.signal_window_spliter(data_new, window_size, overlap)

features = tsfel.time_series_features_extractor(settings2, windows, fs=resample_rate)

