"""
.. include:: ../README.md
.. include:: ../USAGE.md
.. include:: ../CONTRIBUTING.md
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("attrici")
except PackageNotFoundError:
    # package is not installed
    pass
