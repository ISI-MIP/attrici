from dataclasses import dataclass
from typing import Callable


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
    pass
