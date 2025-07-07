from dataclasses import dataclass
from typing import Callable


class Parameter:
    pass


class AttriciGLM:
    @dataclass
    class ParametersDependentOnLongTermPredictor(Parameter):
        link: Callable
        modes: int

    @dataclass
    class ParametersIndependentOfLongTermPredictor(Parameter):
        link: Callable
        modes: int


class Model:
    pass
