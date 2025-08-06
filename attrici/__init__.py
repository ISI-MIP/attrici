"""
.. include:: ../README.md
.. include:: ../USAGE.md
.. include:: ../OVERVIEW.md
.. include:: ../BOOTSTRAPPING.md
.. include:: ../DEVELOPMENT.md
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("attrici")
except PackageNotFoundError:  # pragma: no cover
    # package is not installed
    pass
