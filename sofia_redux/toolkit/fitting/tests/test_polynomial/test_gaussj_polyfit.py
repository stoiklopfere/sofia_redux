# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.toolkit.fitting.polynomial import gaussj_polyfit


@pytest.fixture
def data1d():
    coeffs = [1, 1e-1, 1e-3]
    x = np.arange(10).astype(float)
    y = coeffs[0] + coeffs[1] * x + coeffs[2] * x ** 2
    return np.array([x, y]), coeffs


@pytest.fixture
def data2d():
    y, x = np.mgrid[:5, :5]
    # define a plane
    # This should be pretty easy to distinguish what goes where
    # z = 0 + (0.1 * x) + (0.01 * y) + (0.0001 * xy) + (0.000001 * x^2)
    x, y = x.ravel(), y.ravel()
    z = 1e-7 + (0.1 * x) + (0.01 * y) + (1e-3 * x * y) + (1e-6 * x ** 2)
    return np.array([x, y, z])


def test_1d(data1d):
    samples, expected = data1d
    info = {}
    c = gaussj_polyfit(samples, 2, info=info)
    assert np.allclose(c, expected)
    assert np.allclose(info['exponents'], np.arange(3)[:, None])


def test_2d(data2d):
    samples = data2d
    info = {}
    c = gaussj_polyfit(samples, 2, info=info)
    assert np.allclose(info['exponents'], [
        [0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [0, 2]])
    assert np.allclose(c, [1e-7, 0.1, 1e-6, 1e-2, 1e-3, 0])
    c = gaussj_polyfit(samples, [2, 1], info=info)
    assert np.allclose(info['exponents'], [
        [0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]])
    assert np.allclose(c, [1e-7, 0.1, 1e-6, 1e-2, 1e-3, 0])


def test_bad_values(data2d):
    samples = data2d
    c = gaussj_polyfit(samples * np.nan, 2)
    assert np.isnan(c).all()
