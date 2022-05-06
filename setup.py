from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent


with open("README.md") as fh:
    long_description = fh.read()


def find_requirements(filename):
    with (ROOT / "requirements" / filename).open() as f:
        return [
            s
            for s in [line.strip(" \n") for line in f]
            if not s.startswith("#") and s != ""
        ]


install_reqs = find_requirements("requirements.txt")
docs_require = find_requirements("requirements-docs.txt")
test_requires = find_requirements("requirements-test.txt")
dev_requires = find_requirements("requirements-dev.txt")

setup(
    name="tsfel",
    version="0.2.0",
    author="Fraunhofer Portugal",
    description="Library for time series feature extraction.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fraunhoferportugal/tsfel",
    package_data={"tsfel": ["feature_extraction/features.json"]},
    packages=find_packages(),
    setup_requires=["setuptools>=47.1.1"],
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    install_requires=install_reqs,
    extras_require={
        "dev": dev_requires,
        "docs": docs_require,
        "test": test_requires,
    },
)
