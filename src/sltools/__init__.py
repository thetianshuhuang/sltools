"""sltools: A collection of Slurm tools."""

import importlib.metadata

try:
    __version__ = importlib.metadata.version("sltools")
except importlib.metadata.PackageNotFoundError:  # Running from a source tree.
    __version__ = "unknown"
