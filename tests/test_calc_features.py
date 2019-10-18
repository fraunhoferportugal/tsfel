import numpy as np
import pandas as pd
import tsfel
import json
from pathlib import Path


def pre_process(sensor_data):
    if 'Accelerometer' in sensor_data:
        sensor_data['Accelerometer'].iloc[:, 1] = sensor_data['Accelerometer'].iloc[:, 1] * 0
    return sensor_data


main_directory = str(Path.home()) + "/Documents/Datasets/HumanPatterns/"

# feat_dict = json.load(open("features.json"))
# settings1 = tsfel.get_features_by_domain('Statistical')
settings2 = tsfel.get_features_by_domain('Temporal')
# settings3 = tsfel.get_features_by_domain('Spectral')
# settings4 = tsfel.get_all_features()

# kwargs
time_unit = 1e9  # seconds
resample_rate = 30  # resample sampling frequency
window_size = 100  # number of points
overlap = 0  # varies between 0 and 1
search_criteria = ['Accelerometer.txt', 'Gyroscope.txt']
output_directory = str(Path.home()) + '/Documents/tsfel_output'
data = tsfel.dataset_features_extractor(main_directory, settings2, search_criteria=search_criteria, time_unit=time_unit,
                                        resample_rate=resample_rate, window_size=window_size, overlap=overlap,
                                        pre_process=pre_process, output_directory=output_directory)
