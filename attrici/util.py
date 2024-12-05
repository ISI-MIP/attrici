import importlib.metadata
import sys
import time
from datetime import datetime

from loguru import logger

import attrici


def get_data_provenance_metadata(**other_metadata):
    """Assemble metadata for data provenance

    Returns
    -------
    dictionary
        Dictionary with provenance information
    """
    res = {
        "attrici_version": attrici.__version__,
        "attrici_packages": "\n".join(
            sorted(
                {
                    f"{dist.name}=={dist.version}"
                    for dist in importlib.metadata.distributions()
                },
                key=str.casefold,
            )
        ),
        "attrici_python_version": sys.version,
        "created_at": datetime.now().isoformat(),
    }
    res.update(other_metadata)
    return res


def timeit(func):
    def wrapped(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.info(
            "Function '{}' executed in {:.1f} minutes",
            func.__name__,
            (end - start) / 60,
        )
        return result

    return wrapped
