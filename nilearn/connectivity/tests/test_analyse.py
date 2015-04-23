# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 07:19:01 2015

@author: sb238920
"""
# Standard library imports

# Related third party imports
from math import sqrt

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from nose.tools import assert_raises, assert_equal, assert_is_instance, \
    assert_greater, assert_greater_equal

def test_compute_distances():
    arrays = np.array([[  1,   1.],
                       [  0.,   1.],
                       [  1.,   0.]])
    matrix = np.array([[0., sqrt(2)], [sqrt(2), 0.]])
    assert_array_equal(compute_distances(arrays), matrix)
    