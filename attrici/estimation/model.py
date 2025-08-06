from dataclasses import dataclass
from typing import Callable

from func_timeout import FunctionTimedOut, func_timeout
from joblib import Memory
from loguru import logger


class Parameter:
    pass


class AttriciGLM:
    @dataclass
    class PredictorDependentParam(Parameter):
        link: Callable
        modes: int

    @dataclass
    class PredictorIndependentParam(Parameter):
        link: Callable
        modes: int


class Model:
    def fit_cached(self, defining_data_inputs, cache_dir=None, timeout=None, **kwargs):
        memory = Memory(cache_dir, verbose=0)

        def fit(inputs):
            return self.fit(**kwargs)

        def cache_validation_callback(*args):
            logger.info("Using cached results")
            return True

        def fit_cached():
            return memory.cache(
                fit,
                cache_validation_callback=cache_validation_callback,
            )(defining_data_inputs)

        if timeout is not None:
            try:
                return func_timeout(timeout, fit_cached)
            except FunctionTimedOut:
                return None
        return fit_cached()
