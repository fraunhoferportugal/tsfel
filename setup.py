import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt', 'r') as f:
    install_reqs = [
        s for s in [
            line.strip(' \n') for line in f
        ] if not s.startswith('#') and s != ''
    ]

setuptools.setup(
    name="tsfel",
    version="0.1.5",
    author="Fraunhofer Portugal",
    description="Library for time series feature extraction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fraunhoferportugal/tsfel/",
    package_data={'tsfel': ['feature_extraction/features.json', 'utils/client_secret.json']},
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    install_requires=install_reqs
)
