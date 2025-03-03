import numpy as np
import pytest

from attrici.variables import check_bounds


def test_check_bounds_no_bounds():
    data = np.array([1, 2, 3])
    check_bounds(data)


def test_check_bounds_lower_bound_valid():
    data = np.array([1, 2, 3])
    check_bounds(data, lower=0)


def test_check_bounds_upper_bound_valid():
    data = np.array([1, 2, 3])
    check_bounds(data, upper=4)


def test_check_bounds_both_bounds_valid():
    data = np.array([1, 2, 3])
    check_bounds(data, lower=0, upper=4)


def test_check_bounds_lower_bound_invalid():
    data = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        check_bounds(data, lower=2)


def test_check_bounds_upper_bound_invalid():
    data = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        check_bounds(data, upper=2)


def test_check_bounds_both_bounds_invalid():
    data = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        check_bounds(data, lower=2, upper=2)


def test_check_bounds_empty_data():
    data = np.array([])
    check_bounds(data)


def test_check_bounds_single_element_data():
    data = np.array([1])
    check_bounds(data, lower=0, upper=2)
