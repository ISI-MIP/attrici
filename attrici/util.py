import time

from loguru import logger


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
