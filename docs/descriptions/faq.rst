===
FAQ
===


    * **Does TSFEL work on Windows?**

        Yes. By default we disabled multiprocessing in Windows. The multiprocessing in Windows was not completely stable. If you want to enable you can se ``n_jobs`` variable accordingly at your own risk.


    * **Is it mandatory to pass the sampling rate?**

        No. However, please note that some spectral features rely on the sampling rate to be calculated correctly. Therefore, if you have access to the sampling rate, it is a good practice to pass the correct value. The default sampling rate value is set to 100 Hz. In case you do not have access to the sampling rate, you might refrain from using spectral features.


    * **Does TSFEL work with unevenly sampled data?**

        TSFEL does not handle unevenly sampled data by default. Some of the features (e.g. spectral domain) require that you state the sampling frequency to compute the feature values correctly. A workaround to analyse unevenly spaced time series is to transform the data into equally spaced observations using interpolation (e.g. linear interpolation). After that you could use TSFEL directly on this converted equally spaced data.


    * **Does TSFEL allow to extract features from multi-dimensional time series with variable lengths?**

        Yes, it is possible, indeed. That's actually one of the functionalities that weren't adequately addressed by similar packages when we started the development of TSFEL. I recommend that the time series be stored in a data file and processed using the dataset_features_extractor. We are still updating the documentation of this functionality. In the meantime, you can read Section 2.2.1. Data ingestion and preprocessing of `TSFEL publication <https://www.sciencedirect.com/science/article/pii/S2352711020300017/>`_ which addresses that topic.


    * **Do I have examples showcasing a complete pipeline using TSFEL?**

       Sure, we provide several notebooks with examples of complete classification pipelines using TSFEL. The notebooks are available `here <https://github.com/fraunhoferportugal/tsfel/tree/master/notebooks/>`_. If you want to share a notebook with additional pipelines, please feel free to reach us.


    * **Why should I use TSFEL?**

       TSFEL assists researchers on exploratory feature extraction tasks on time series without requiring significant programming effort. All the features used in TSFEL have been delicately implemented and are unit tested. TSFEL also has an enthusiastic team of maintainers that would be happy to help you addressing doubts and solving issues.


    * **How can I give the authors credit for their work?**

       If you used TSFEL we would be appreciated if you star our `GitHub <https://github.com/fraunhoferportugal/tsfel/>`_ repository. In case you use TSFEL during your research, you would be happy if you can `cite our work <https://www.sciencedirect.com/science/article/pii/S2352711020300017/>`_.