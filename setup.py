import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="TSFDLIB",
    version="0.0.113",
    author="Fraunhofer",
    author_email="TSFDlib@gmail.com",
    description="Library for time series feature extraction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TSFDlib/TSFDlib",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)