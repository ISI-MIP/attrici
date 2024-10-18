import collections
import collections.abc
import inspect

import numpy as np

# monkey patch for newer numpy versions
# TODO remove once pymc3 is replaced
if not hasattr(np, "asscalar"):
    np.asscalar = np.ndarray.item

np.bool = bool

if not hasattr(collections, "Iterable"):
    collections.Iterable = collections.abc.Iterable

if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec

import pymc3 as pm  # noqa
import theano.tensor as tt  # noqa
