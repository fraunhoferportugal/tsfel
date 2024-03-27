import os
import shutil
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
import scipy.io as sio
from tqdm import tqdm

module_path = os.path.dirname(__file__)
data_path = os.path.join(module_path, "cached_datasets")


class Empirical1000Dataset:
    """A convenience class to access the Empirical1000 dataset.

    When using one this dataset, please cite [1]_.

    Parameters
    ----------
    use_cache : bool (default: True)
        Whether a cached version of the dataset should be used, if one is found.
        The dataset is always cached upon loading, and this parameter only determines
        whether the cached version shall be refreshed upon loading.

    extract_features : bool (default: False)
        Whether a feature extraction routine should be called after loading the data files.
        TODO: Implement this functionality.

    Notes
    -----
        To optimize running times it is recommended to use `use_cache=True` (default) in order
        to only experience a time once downloading and work on a cached version of the dataset afterward.


    References
    ----------
    .. [1] Fulcher, B. D., Lubba, C. H., Sethi, S. S., & Jones, N. S. (2020).
    A self-organizing, living library of time-series data. Scientific data, 7(1), 213.
    """

    def __init__(self, use_cache=True, extract_features=False):
        self.use_cache = use_cache

        self.raw = {}
        self.features = {}  # An add-on to this class will contain a featurized data representation.

        self.__dataset_url = "https://figshare.com/ndownloader/articles/5436136/versions/10"
        self.cache_folder_path = data_path
        self.__empirical1000_folder_path = os.path.join(
            self.cache_folder_path,
            "Empirical1000",
        )

        if not os.path.exists(self.cache_folder_path) or not os.listdir(self.cache_folder_path) or not use_cache:
            print("Cache folder is empty. Downloading the Empirical1000 dataset...")
            Path(os.path.join(self.cache_folder_path, "Empirical1000")).mkdir(
                parents=True,
                exist_ok=True,
            )
            self.__download_dataset()

        self.__data_files = sio.loadmat(
            os.path.join(self.__empirical1000_folder_path, "INP_1000ts.mat"),
        )
        self.metadata = pd.read_csv(
            os.path.join(self.__empirical1000_folder_path, "hctsa_timeseries-info.csv"),
        )

        for i, name in enumerate(self.metadata["Name"]):
            self.raw[i] = EmpiricalTimeSeries(
                name,
                self.__data_files["timeSeriesData"][0][i].flatten(),
            )
            if extract_features:
                tqdm_iterator = tqdm(
                    total=len(self.metadata["Name"]),
                    desc="Extracting features.",
                )
                tqdm_iterator.update(1)

    def __download_dataset(self):
        try:
            # Send a request to the dataset URL with allow_redirects=False to handle redirection
            response = requests.get(
                self.__dataset_url,
                stream=True,
                allow_redirects=False,
            )

            # Check if the request was successful (status code 200)
            if response.status_code == 200:
                total_size = int(response.headers.get("content-length", 0)) or None
                filename = self.__extract_filename(response)

                with tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    desc="Downloading Empirical1000 dataset",
                    dynamic_ncols=True,
                ) as pbar:
                    with open(
                        os.path.join(self.cache_folder_path, filename),
                        "wb",
                    ) as out_file:
                        shutil.copyfileobj(response.raw, out_file)
                        pbar.update(os.path.getsize(out_file.name))

                zip_file_path = os.path.join(self.cache_folder_path, filename)
                with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
                    zip_ref.extractall(self.__empirical1000_folder_path)

                # A sanitizing routine that deletes non-relevant files.
                os.remove(zip_file_path)
                extracted_files = os.listdir(self.__empirical1000_folder_path)
                timeseries_files = [
                    file for file in extracted_files if "timeseries-info" in file.lower() or "inp" in file.lower()
                ]

                for file in extracted_files:
                    if file not in timeseries_files:
                        os.remove(os.path.join(self.__empirical1000_folder_path, file))

                print(f"Dataset downloaded and saved to: {self.cache_folder_path}")
            else:
                print(
                    f"Failed to download dataset. Status code: {response.status_code}",
                )

        except Exception as e:
            print(f"An error occurred: {e}")

    @staticmethod
    def __extract_filename(response):
        # Try to extract the filename from the Content-Disposition header.
        # If the header is not present, extract the filename from the URL
        content_disposition = response.headers.get("Content-Disposition")
        if content_disposition and "filename=" in content_disposition:
            filename = content_disposition.split("filename=")[1].strip("\";'")
        else:
            filename = os.path.basename(urlparse(response.url).path)

        return filename

    def get_by_len(self, l):
        query = np.nonzero(self.metadata["Length"] == l)[0]
        filtered_data = {key: self.raw[key] for key in query if key in self.raw}

        return filtered_data


class EmpiricalTimeSeries:
    """The object representing a time series from the Empirical1000 dataset.

    Attributes:
        name (str): The name of the time series.
        sig (array-like): The time series data.
        len (int): The length of the time series.

    Methods:
        __init__(self, name, sig):
            Initializes a new EmpiricalTimeSeries object.
    """

    def __init__(self, name, sig):
        """Initializes a new EmpiricalTimeSeries object.

        Parameters:
            name (str): The name or identifier of the time series.
            sig (list or array-like): The time series data.
        """
        self.name = name
        self.sig = sig
        self.len = len(sig)


class FeaturizedEmpiricalTimeSeries(EmpiricalTimeSeries):
    def __init__(self, empirical_time_series):
        super().__init__(empirical_time_series.name, empirical_time_series.sig)
        self.example_feature = self.get_feature()

    @staticmethod
    def get_feature(self):
        return -1
