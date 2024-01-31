from tsfel import *
from pkg_resources import get_distribution
import warnings

__version__ = get_distribution('tsfel').version
warnings.filterwarnings("once", category=UserWarning, module='')
