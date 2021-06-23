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

## Available features

#### Statistical domain
| Features                   | Computational Cost |
|----------------------------|:------------------:|
| ECDF                       |          1         |
| ECDF Percentile            |          1         |
| ECDF Percentile Count      |          1         |
| Histogram                  |          1         |
| Interquartile range        |          1         |
| Kurtosis                   |          1         |
| Max                        |          1         |
| Mean                       |          1         |
| Mean absolute deviation    |          1         |
| Median                     |          1         |
| Median absolute deviation  |          1         |
| Min                        |          1         |
| Root mean square           |          1         |
| Skewness                   |          1         |
| Standard deviation         |          1         |
| Variance                   |          1         |


#### Temporal domain
| Features                   | Computational Cost |
|----------------------------|:------------------:|
| Absolute energy            |          1         |
| Area under the curve       |          1         |
| Autocorrelation            |          1         |
| Centroid                   |          1         |
| Entropy                    |          1         |
| Mean absolute diff         |          1         |
| Mean diff                  |          1         |
| Median absolute diff       |          1         |
| Median diff                |          1         |
| Negative turning points    |          1         |
| Peak to peak distance      |          1         |
| Positive turning points    |          1         |
| Signal distance            |          1         |
| Slope                      |          1         |
| Sum absolute diff          |          1         |
| Total energy               |          1         |
| Zero crossing rate         |          1         |
| Neighbourhood peaks        |          1         |


#### Spectral domain
| Features                          | Computational Cost |
|-----------------------------------|:------------------:|
| FFT mean coefficient              |          1         |
| Fundamental frequency             |          1         |
| Human range energy                |          2         |
| LPCC                              |          1         |
| MFCC                              |          1         |
| Max power spectrum                |          1         |
| Maximum frequency                 |          1         |
| Median frequency                  |          1         |
| Power bandwidth                   |          1         |
| Spectral centroid                 |          2         |
| Spectral decrease                 |          1         |
| Spectral distance                 |          1         |
| Spectral entropy                  |          1         |
| Spectral kurtosis                 |          2         |
| Spectral positive turning points  |          1         |
| Spectral roll-off                 |          1         |
| Spectral roll-on                  |          1         |
| Spectral skewness                 |          2         |
| Spectral slope                    |          1         |
| Spectral spread                   |          2         |
| Spectral variation                |          1         |
| Wavelet absolute mean             |          2         |
| Wavelet energy                    |          2         |
| Wavelet standard deviation        |          2         |
| Wavelet entropy                   |          2         |
| Wavelet variance                  |          2         |


## Citing
When using TSFEL please cite the following publication:

Barandas, Marília and Folgado, Duarte, et al. "*TSFEL: Time Series Feature Extraction Library.*" SoftwareX 11 (2020). [https://doi.org/10.1016/j.softx.2020.100456](https://doi.org/10.1016/j.softx.2020.100456)

## Acknowledgements
We would like to acknowledge the financial support obtained from the project Total Integrated and Predictive Manufacturing System Platform for Industry 4.0, co-funded by Portugal 2020, framed under the COMPETE 2020 (Operational Programme  Competitiveness and Internationalization) and European Regional Development Fund (ERDF) from European Union (EU), with operation code POCI-01-0247-FEDER-038436.
