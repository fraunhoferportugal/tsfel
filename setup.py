import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tsfel",
    version="0.0.2",
    author="Fraunhofer Portugal",
    description="Library for time series feature extraction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fraunhoferportugal/tsfel/",
    package_data={'tsfel': ['utils/features.json', 'utils/client_secret.json']},
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['scipy', 'pandas', 'matplotlib', 'numpy'],
)
