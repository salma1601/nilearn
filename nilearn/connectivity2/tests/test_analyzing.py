from math import sqrt

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from nose.tools import assert_almost_equal

from nilearn.connectivity2.analyzing import (_compute_distance,
                                             _compute_geo_variance,
                                             _compute_variance,
                                             compute_spreading,
                                             split_sample,
                                             assert_is_outlier)


def test_compute_distance():
    # Check distance between the same matrices is zero
    matrix1 = np.array([[3., 1.], [1., 2.]])
    assert_almost_equal(
        _compute_distance(matrix1, matrix1, distance_type='euclidean'), 0.)
    assert_almost_equal(
        _compute_distance(matrix1, matrix1, distance_type='geometric'), 0.)

    # Euclidian distance
    matrix2 = np.array([[3., 0.], [0., 2.]])
    assert_almost_equal(
        _compute_distance(matrix1, matrix2, distance_type='euclidean'),
        sqrt(2))

    # Geometric distance
    matrix1 = np.diag([1., 2., 3.])
    matrix2 = np.diag([3., 4., 3.])
    assert_almost_equal(
        _compute_distance(matrix1, matrix2, distance_type='geometric'),
        np.linalg.norm(np.log([2., 3.])))


def test_compute_variance():
    matrix = np.array([[3, 2], [2, 3]])
    assert_array_almost_equal(_compute_variance(matrix, np.array([matrix])),
                              [0.])
    matrices = np.dstack((np.eye(2), np.array([[3, 0], [0, 2]]))).T
    assert_array_almost_equal(_compute_variance(matrix, matrices),
                              [12.5])


def test_compute_geo_variance():
    matrix = np.diag([1., 2., 3.])
    assert_array_almost_equal(
        _compute_geo_variance(matrix, np.array([matrix])), [0.])
    matrices = np.dstack((np.eye(3), np.diag([3., 4., 3.]))).T
    assert_array_almost_equal(_compute_geo_variance(matrix, matrices),
                              np.linalg.norm(np.log([2., 3.])) ** 2)


def test_compute_spreading():
    matrix1 = np.array([[3, 2], [2, 3]])
    matrix2 = np.eye(2)
    matrix3 = np.array([[3, 0], [0, 2]])
    matrices = np.dstack((matrix1, matrix2, matrix3)).T
    assert_array_almost_equal([12.5, 10.5, 7],
                              compute_spreading(matrices))


def test_split_sample():
    lower_half, higher_half = split_sample(np.arange(10)[:, np.newaxis])
    assert_array_equal(lower_half, np.arange(5))
    assert_array_equal(higher_half, np.arange(5, 10))


def test_assert_is_outlier():
    inliers = [np.diag([1., 2.]),
               np.diag([1., .2]),
               np.diag([.1, .5])]
    outlier = np.diag([1.5, 2.1])
    assert_is_outlier(outlier, inliers)
