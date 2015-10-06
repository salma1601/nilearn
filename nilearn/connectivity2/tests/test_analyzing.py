import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from nilearn.connectivity2.analyzing import (_compute_variance,
                                             compute_spreading,
                                             split_sample)


def test_compute_variance():
    matrix = np.array([[3, 2], [2, 3]])
    assert_array_almost_equal(_compute_variance(matrix, np.array([matrix])),
                              [0.])
    matrices = np.dstack((np.eye(2), np.array([[3, 0], [0, 2]]))).T
    assert_array_almost_equal(_compute_variance(matrix, matrices),
                              [12.5])


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
