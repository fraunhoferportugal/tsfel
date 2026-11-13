from importlib.metadata import PackageNotFoundError, version

from tsfel.constants import *
from tsfel.datasets import *
from tsfel.feature_extraction import *
from tsfel.utils import *

try:
    __version__ = version("tsfel")
except PackageNotFoundError:
    __version__ = "unknown"
