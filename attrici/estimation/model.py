from dataclasses import dataclass
from typing import Callable


class AttriciGLM:
    @dataclass
    class PredictorDependentParam:
        link: Callable
        modes: int

    @dataclass
    class PredictorIndependentParam:
        link: Callable
        modes: int


class Model:
    pass
