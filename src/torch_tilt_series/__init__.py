"""Tilt series data structure, projection and subtilt extraction for cryo-ET."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("torch-tilt-series")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Marten Chaillet, Davide Torre"
__email__ = "martenchaillet@gmail.com, davidetorre99@gmail.com"

from torch_tilt_series.tilt_series import TiltSeries

__all__ = ["TiltSeries"]
