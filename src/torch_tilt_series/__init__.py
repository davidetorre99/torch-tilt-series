"""tilt series"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("torch-tilt-series")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "davide torre"
__email__ = "davidetorre99@gmail.com"
