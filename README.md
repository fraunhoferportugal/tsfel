[![Documentation Status](https://readthedocs.org/projects/tsfel/badge/?version=latest)](https://tsfel.readthedocs.io/en/latest/?badge=latest)
[![license](https://img.shields.io/badge/License-BSD%203-brightgreen)](https://github.com/fraunhoferportugal/tsfel/blob/master/LICENSE.txt)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/tsfel)
![PyPI](https://img.shields.io/pypi/v/tsfel)
[![Downloads](https://pepy.tech/badge/tsfel)](https://pepy.tech/project/tsfel)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fraunhoferportugal/tsfel/blob/master/notebooks/TSFEL_HAR_Example.ipynb)

# Time Series Feature Extraction Library
## Intuitive time series feature extraction
This repository hosts the **TSFEL - Time Series Feature Extraction Library** python package. TSFEL assists researchers on exploratory feature extraction tasks on time series without requiring significant programming effort.

Users can interact with TSFEL using two methods:
##### Online
It does not requires installation as it relies on Google Colabs and a user interface provided by Google Sheets

##### Offline
Advanced users can take full potential of TSFEL by installing as a python package
```python
pip install tsfel
```

## Includes a comprehensive number of features
TSFEL is optimized for time series and **automatically extracts over 60 different features on the statistical, temporal and spectral domains.**

## Functionalities
* **Intuitive, fast deployment and reproducible**: interactive UI for feature selection and customization
* **Computational complexity evaluation**: estimate the computational effort before extracting features
* **Comprehensive documentation**: each feature extraction method has a detailed explanation
* **Unit tested**: we provide unit tests for each feature
* **Easily extended**: adding new features is easy and we encourage you to contribute with your custom features

## Get started
The code below extracts all the available features on an example dataset file.

```python
import tsfel
import pandas as pd

# load dataset
df = pd.read_csv('Dataset.txt')

# Retrieves a pre-defined feature configuration file to extract all available features
cfg = tsfel.get_features_by_domain()

# Extract features
X = tsfel.time_series_features_extractor(cfg, df)
```

## Acknowledgements
We would like to acknowledge the financial support obtained from the project Total Integrated and Predictive Manufacturing System Platform for Industry 4.0, co-funded by Portugal 2020, framed under the COMPETE 2020 (Operational Programme  Competitiveness and Internationalization) and European Regional Development Fund (ERDF) from European Union (EU), with operation code POCI-01-0247-FEDER-038436.
