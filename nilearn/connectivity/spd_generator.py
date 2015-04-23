from embedding import map_sym, vec_to_sym, geometric_mean, sym_to_vec
from nilearn._utils.extmath import is_spd
from sklearn.covariance import EmpiricalCovariance


def sample_wishart(scale_mat, dof, rand_gen=None):
    """
    Returns a sample from the Wishart distribution, the distribution of the
    sample covariance matrix for a sample from a multivariate normal
    distribution.

    Parameters
    ==========
    scale_mat : (n_features, n_features) numpy.ndarray
        Scale matrix. Raise an error if not square symmetric positive definite.

    dof : float
        Degrees of freedom. Raise an error if dof < n_features.

    rand_gen : numpy.random.RandomState or None, optional
        Random generator to use for generation.

    Returns
    =======
    out : numpy.ndarray
        The drawn sample, shape is (n_features, n_features).
    """
    if scale_mat.ndim != 2 or scale_mat.shape[0] != scale_mat.shape[-1]:
        raise ValueError("Expected square scale matrix")
    if not is_spd(scale_mat):
        raise ValueError("Expected symmetric positive definite scale matrix")
    if dof < scale_mat.shape[0]:
        raise ValueError("Expected more than {0} degrees of freedom".format(
                                                    scale_mat.shape[0] - 1))

    if rand_gen is None:
        rand_gen = np.random.RandomState(0)

    n_features = scale_mat.shape[0]
    chol = np.linalg.cholesky(scale_mat)

    # use matlab's heuristic for choosing between the two different sampling
    # schemes
    if (dof <= 81 + n_features) and (dof == round(dof)):
        X = np.dot(chol, rand_gen.normal(size=(n_features, dof)))
    else:
        A = np.diag(np.sqrt(rand_gen.chisquare(dof - np.arange(0, n_features),
                                               size=n_features)))
        A[np.tri(n_features, k=-1, dtype=bool)] = rand_gen.normal(
            size=(n_features * (n_features - 1) / 2.))
        X = np.dot(chol, A)

    return np.dot(X, X.T)


def sample_spd_normal(mean, cov=None, rand_gen=None):
    """Draw random spd matrix from Gaussian distribution on the manifold.

    Parameters
    ==========
    mean : (n_features, n_features) numpy.ndarray
        Mean of the (n_features, n_features) matrix distribution. Raise an
        error if not square symmetric positive definite.
    cov : (p, p) numpy.ndarray
        Covariance matrix on the manifold. Expected shape related to the shape
        of mean by the relation M = n_features * (n_features + 1) / 2.
        Raise an error if not correct shape or not symmetric positive definite.

    Returns
    =======
    out : numpy.ndarray
        The drawn sample, shape is (n_features, n_features).
    """
    if rand_gen is None:
        rand_gen = np.random.RandomState(0)

    if mean.ndim != 2 or mean.shape[0] != mean.shape[-1]:
        raise ValueError("Expected square mean matrix")

    if not is_spd(mean,  decimal=7):
        raise ValueError("Expected symmetric positive definite mean matrix")

    p = mean.shape[0] * (mean.shape[0] + 1) / 2
    if cov is None:
        cov = np.eye(p)
    elif cov.ndim != 2 or cov.shape[0] != cov.shape[-1]:
        raise ValueError("Expected square covariance matrix")
    elif cov.shape[0] != p:
        raise ValueError("Shapes of mean and covariance matrices are not"
                         "compatible, expected covariance of shape "
                         "({0}, {0})".format(p))
    elif is_spd(cov, decimal=7):
        raise ValueError("Expected symmetric positive definite covariance"
                         "matrix")

    tangent = rand_gen.multivariate_normal(np.zeros(
        cov.shape[0]), cov)
    mean_sqrt = map_sym(np.sqrt, mean)
    tangent = vec_to_sym(tangent)
    tangent_exp = map_sym(np.exp, tangent)
    out = mean_sqrt.dot(tangent_exp).dot(mean_sqrt)
    return out


def spd_covariance(tangents, isometry=False):
    """
    tangents : array like, shape n_matrices, n_features, n_features
    """
    if not np.any(tangents):
        raise ValueError('Empty list')

    vectors = [sym_to_vec(tangent, isometry=isometry) for tangent in tangents]
    estimator = EmpiricalCovariance(assume_centered=False)
    estimator.fit(np.array(vectors))
    cov = estimator.covariance_

    # Insure positive definiteness
    if not is_spd(cov, decimal=7):
        epsilon = np.abs(np.min(np.linalg.eigvalsh(cov)))
        print 'adding 2 x {} to the diagonal'.format(epsilon)
        p = cov.shape[0]
        cov += 2 * epsilon * np.eye(p)
    return cov


def whiten(matrices, isometry=False):
    cov = spd_covariance(matrices)
    cov_sqrt = map_sym(np.sqrt, cov)
    cov_sqrt_inv = map_sym(lambda x: 1 / x, cov_sqrt)
    w_matrices = []
    for matrix in matrices:
        vector = sym_to_vec(matrix, isometry=isometry)
        w_vector = cov_sqrt_inv.dot(vector)
        w_matrices.append(vec_to_sym(w_vector, isometry=isometry))
    return np.array(w_matrices)


def compute_cov_mask(size):
    mask = np.zeros((size, size), dtype=bool)
    np.fill_diagonal(mask, True)
    vector_mask = sym_to_vec(mask, isometry=False)
    cov_mask = vector_mask * vector_mask[:, np.newaxis]
    return cov_mask


def spds_to_wishart(spds, dof=None, rand_gen=None):
    """Generates samples for Wishart distribution with mean and dof extracted
    from given data.

    Parameters
    ==========
    spds : array like
    """
    if rand_gen is None:
        rand_gen = np.random.RandomState(0)

    n_samples = len(spds)
    if n_samples == 0:
        raise ValueError("empty list")
    if dof is None:
        dof = spds[0].shape[0]

    scale_mat = np.mean(spds, axis=0) / dof
    if not is_spd(scale_mat, decimal=7):
        raise ValueError("mean of matrices is not spd")

    wishart_spds = [sample_wishart(scale_mat, dof=dof, rand_gen=rand_gen) for
                    n in range(n_samples)]
    return wishart_spds




def spds_to_normal(mean, cov=None, n_subjects=40, rand_gen=None):
    if rand_gen is None:
        rand_gen = np.random.RandomState(0)

    spds = []
    if False:
        for n in range(0, n_subjects):
            for k in range(n_subjects):
                spd = sample_spd_normal(mean, cov=cov, rand_gen=rand_gen)
                spds.append(spd)
    from nilearn.connectivity.embedding import map_sym, vec_to_sym
    from nilearn._utils.extmath import is_spd
    mean_sqrt = map_sym(np.sqrt, mean)
    p = mean.shape[0] * (mean.shape[0] + 1) / 2
    if cov is None:
        cov =  np.eye(p) * dispersion
    for n in range(n_subjects):
        tangent = rand_gen.multivariate_normal(np.zeros(cov.shape[0]), cov)
        tangent = vec_to_sym(tangent, isometry=True)  # TODO: change to True
        tangent_exp = map_sym(np.exp, tangent)
        spd = mean_sqrt.dot(tangent_exp).dot(mean_sqrt)
        if not is_spd(spd):
            epsilon = np.abs(np.min(np.linalg.eigvalsh(spd)))
            print epsilon
            spd += 2 * epsilon * np.eye(mean.shape[0])
        spds.append(spd)
    return spds


import copy
import warnings
from math import sqrt, exp, log, cosh, sinh

import numpy as np
from scipy import linalg
from numpy.testing import assert_array_almost_equal, assert_array_equal
from nose.tools import assert_raises, assert_equal, assert_is_instance, \
    assert_greater, assert_greater_equal

from nilearn._utils.extmath import is_spd
from nilearn.connectivity.embedding import check_mat, map_sym, map_eig, \
    geometric_mean, grad_geometric_mean, sym_to_vec, vec_to_sym, \
    prec_to_partial, CovEmbedding, cov_to_corr



from nilearn.connectivity.tests.test_embedding_tmp2 import sample_wishart,\
    sample_spd_normal
from nilearn.connectivity.embedding import geometric_mean


def synthetic_data_wishart(sigma, n_samples=201, n_subjects=40):
    rand_gen = np.random.RandomState(0)
    spds = []
    for k in range(n_subjects):
#            spd = random_wishart(sigma, dof=n_samples, rand_gen=rand_gen)
        spd = sample_wishart(sigma, dof=n_samples, rand_gen=rand_gen)
#            if cov_to_corr(spd)[0, 1] - prec_to_partial(np.linalg.inv(spd))[0, 1] > 0.2 \
#                and (cov_to_corr(spd)[0, 1] - prec_to_partial(np.linalg.inv(spd))[0, 1] < 0.5):
        spds.append(spd)
    if not spds:
        print 'no'

    return spds


def synthetic_data_manifold(mean, cov=None, n_subjects=40, dispersion=0.01):
    rand_gen = np.random.RandomState(0)  # TODO: variable random state?
    spds = []
    if False:
        for n in range(0, n_subjects):
            for k in range(n_subjects):
                spd = sample_spd_normal(mean, cov=cov, rand_gen=rand_gen)
                spds.append(spd)
    from nilearn.connectivity.embedding import map_sym, vec_to_sym
    from nilearn._utils.extmath import is_spd
    mean_sqrt = map_sym(np.sqrt, mean)
    p = mean.shape[0] * (mean.shape[0] + 1) / 2
    if cov is None:
        cov =  np.eye(p) * dispersion
    for n in range(n_subjects):
        tangent = rand_gen.multivariate_normal(np.zeros(cov.shape[0]), cov)
        tangent = vec_to_sym(tangent, isometry=True)  # TODO: change to True
        tangent_exp = map_sym(np.exp, tangent)
        spd = mean_sqrt.dot(tangent_exp).dot(mean_sqrt)
        if not is_spd(spd):
            epsilon = np.abs(np.min(np.linalg.eigvalsh(spd)))
            print epsilon
            spd += 2 * epsilon * np.eye(mean.shape[0])
        spds.append(spd)
    return spds



def test_check_mat():
    """Test check_mat function"""
    non_square = np.ones((2, 3))
    assert_raises(ValueError, check_mat, non_square, 'square')

    non_sym = np.array([[0, 1], [0, 0]])
    assert_raises(ValueError, check_mat, non_sym, 'symmetric')

    non_spd = np.ones((3, 3))
    assert_raises(ValueError, check_mat, non_spd, 'spd')


def test_map_sym():
    """Test map_sym function"""
    # Test on exp map
    sym = np.ones((2, 2))
    sym_exp = exp(1.) * np.array([[cosh(1.), sinh(1.)], [sinh(1.), cosh(1.)]])
    assert_array_almost_equal(map_sym(np.exp, sym), sym_exp)

    # Test on sqrt map
    spd_sqrt = np.array([[2., -1., 0.], [-1., 2., -1.], [0., -1., 2.]])
    spd = spd_sqrt.dot(spd_sqrt)
    assert_array_almost_equal(map_sym(np.sqrt, spd), spd_sqrt)

    # Test on log map
    spd = np.array([[1.25, 0.75], [0.75, 1.25]])
    spd_log = np.array([[0., log(2.)], [log(2.), 0.]])
    assert_array_almost_equal(map_sym(np.log, spd), spd_log)


def test_geometric_mean_couple():
    """Test geometric_mean function for two matrices"""
    n_features = 7
    spd1 = np.ones((n_features, n_features))
    spd1 = spd1.dot(spd1) + n_features * np.eye(n_features)
    spd2 = np.tril(np.ones((n_features, n_features)))
    spd2 = spd2.dot(spd2.T)
    vals_spd2, vecs_spd2 = np.linalg.eigh(spd2)
    spd2_sqrt = map_eig(np.sqrt, vals_spd2, vecs_spd2)
    spd2_inv_sqrt = map_eig(np.sqrt, 1. / vals_spd2, vecs_spd2)
    geo = spd2_sqrt.dot(map_sym(np.sqrt,
        spd2_inv_sqrt.dot(spd1).dot(spd2_inv_sqrt))).dot(spd2_sqrt)
    assert_array_almost_equal(geometric_mean([spd1, spd2]), geo)


def test_geometric_mean_diagonal():
    """Test geometric_mean function for diagonal matrices"""
    n_matrices = 20
    n_features = 5
    diags = []
    for k in range(n_matrices):
        diag = np.eye(n_features)
        diag[k % n_features, k % n_features] = 1e4 + k
        diag[(n_features - 1) // (k + 1), (n_features - 1) // (k + 1)] = \
            (k + 1) * 1e-4
        diags.append(diag)
    geo = np.prod(np.array(diags), axis=0) ** (1 / float(len(diags)))
    assert_array_almost_equal(geometric_mean(diags), geo)


def test_geometric_mean_geodesic():
    """Test geometric_mean function for single geodesic matrices"""
    n_matrices = 10
    n_features = 6
    sym = np.arange(n_features) / np.linalg.norm(np.arange(n_features))
    sym = sym * sym[:, np.newaxis]
    times = np.arange(n_matrices)
    non_singular = np.eye(n_features)
    non_singular[1:3, 1:3] = np.array([[-1, -.5], [-.5, -1]])
    spds = []
    for time in times:
        spds.append(non_singular.dot(map_sym(np.exp, time * sym)).dot(
            non_singular.T))
    geo = non_singular.dot(map_sym(np.exp, times.mean() * sym)).dot(
        non_singular.T)
    assert_array_almost_equal(geometric_mean(spds), geo)


def random_diagonal(p, v_min=1., v_max=2., rand_gen=None):
    """Generate a random diagonal matrix.

    Parameters
    ----------
    p : int
        The first dimension of the array.

    v_min : float, optional (default to 1.)
        Minimal element.

    v_max : float, optional (default to 2.)
        Maximal element.

    rand_gen: numpy.random.RandomState or None, optional
        Random generator to use for generation.

    Returns
    -------
    output : numpy.ndarray, shape (p, p)
        A diagonal matrix with the given minimal and maximal elements.

    """
    if rand_gen is None:
        rand_gen = np.random.RandomState(0)

    diag = rand_gen.rand(p) * (v_max - v_min) + v_min
    diag[diag == np.amax(diag)] = v_max
    diag[diag == np.amin(diag)] = v_min
    return np.diag(diag)


def random_spd(p, eig_min, cond, rand_gen=None):
    """Generate a random symmetric positive definite matrix.

    Parameters
    ----------
    p : int
        The first dimension of the array.

    eig_min : float
        Minimal eigenvalue.

    cond : float
        Condition number, defined as the ratio of the maximum eigenvalue to the
        minimum one.

    rand_gen: numpy.random.RandomState or None, optional
        Random generator to use for generation.

    Returns
    -------
    ouput : numpy.ndarray, shape (p, p)
        A symmetric positive definite matrix with the given minimal eigenvalue
        and condition number.
    """
    if rand_gen is None:
        rand_gen = np.random.RandomState(0)

    mat = rand_gen.randn(p, p)
    unitary, _ = linalg.qr(mat)
    diag = random_diagonal(p, v_min=eig_min, v_max=cond * eig_min,
                           rand_gen=rand_gen)
    return unitary.dot(diag).dot(unitary.T)


def random_non_singular(p, sing_min=1., sing_max=2., rand_gen=None):
    """Generate a random nonsingular matrix.

    Parameters
    ----------
    p : int
        The first dimension of the array.

    sing_min : float, optional (default to 1.)
        Minimal singular value.

    sing_max : float, optional (default to 2.)
        Maximal singular value.

    Returns
    -------
    output : numpy.ndarray, shape (p, p)
        A nonsingular matrix with the given minimal and maximal singular
        values.
    """
    if rand_gen is None:
        rand_gen = np.random.RandomState(0)

    diag = random_diagonal(p, v_min=sing_min, v_max=sing_max,
                           rand_gen=rand_gen)
    mat1 = rand_gen.randn(p, p)
    mat2 = rand_gen.randn(p, p)
    unitary1, _ = linalg.qr(mat1)
    unitary2, _ = linalg.qr(mat2)
    return unitary1.dot(diag).dot(unitary2.T)


#######################################################
#   TODO: to remove
def generate_3d_spd(a_min=.5, a_max=2, b_min=.5, b_max=2, c_min=.5, c_max=2,
                    d_min=1., f_min=1., rand_gen=None):
    """Generate a 3x3 spd matrix M_{i, j} s.t. corr(M)_{1,3} > partial(M)_{1,3}
    M = np.array([[a, d, e], [d, b, f], [e, f, c]])
    """
    if rand_gen is None:
        rand_gen = np.random.RandomState(0)

    # Generate diagonal
    v_min = np.array([a_min, b_min, c_min])
    v_max = np.array([a_max, b_max, c_max])
    a, b, c = v_min + rand_gen.rand(3) * (v_max - v_min)

    # Generate  2nd diagonal
    d_max = sqrt(a * b)   # 1st minor positive
    f_max = sqrt(b * c)   # 3rd minor positive
    if d_min > d_max or f_min > f_max:
        d_min = max(d_max - 0.1, 0.)
        f_min = max(f_max - 0.1, 0.)
#        print('changed minimal bounds for d or f')

    v_min = np.array([d_min, f_min])
    v_max = np.array([d_max, f_max])
    d, f = v_min + rand_gen.rand(2) * (v_max - v_min)
    d_corr = d / d_max
    f_corr = f / f_max
    delta = (1 - d_corr ** 2) * (1 - f_corr ** 2)
    e_min = sqrt(a * c) * d_corr * f_corr * (1 - sqrt(delta)) / (
        d_corr * f_corr + f_corr / d_corr * (1 - d_corr ** 2) + \
        d_corr / f_corr * (1 - f_corr ** 2))
    e_max = sqrt(a * c) * (d_corr * f_corr + sqrt(delta))
    e_min = sqrt(a * c) * d_corr * f_corr * (1 - sqrt(delta)) / (
        d_corr ** 2 + f_corr ** 2 - d_corr ** 2 * f_corr ** 2)
    e_max = sqrt(a * c) * d_corr * f_corr * (1 + sqrt(delta)) / (
        d_corr ** 2 + f_corr ** 2 - d_corr ** 2 * f_corr ** 2)
    e_max = min(e_max, d * c / f)
    if e_min > e_max:
        raise ValueError("Choose other bounds")
    else:
        e = e_min + rand_gen.rand(1) * (e_max - e_min)

#    print d_corr
#    print - (e * f - c * d) / sqrt((b * c - f ** 2) * (a * c - e ** 2))
    return np.array([[a, d, e], [d, b, f], [e, f, c]])


def test_generate_3d_spd():
    for n in range(0, 100):
#        print n
        rand_gen = np.random.RandomState(n)
        m = generate_3d_spd(a_min=.1, a_max=20, b_min=3, b_max=5, c_min=.5,
                            c_max=1, d_min=1., f_min=1., rand_gen=rand_gen)
        assert(is_spd(m))
        corr = cov_to_corr(m)
        partial = prec_to_partial(np.linalg.inv(m))
        assert_greater(corr[0, 1], partial[0, 1])
        assert_greater(partial[0, 1], 0)


def test_bounds():
    n_features = 3
    n_samples = 201
    n_matrices = 40
    for n in range(0, 40):
        print n 
        rand_gen = np.random.RandomState(n)
        spds = []
        for k in range(n_matrices):
            spd = generate_3d_spd(rand_gen=rand_gen)
            spd = random_wishart(sigma, dof=n_samples, rand_gen=None)
            spd = sample_wishart(sigma, dof, rand_gen=None)
            if cov_to_corr(spd)[0, 1] - prec_to_partial(np.linalg.inv(spd))[0, 1] > 0.2 \
                and (cov_to_corr(spd)[0, 1] - prec_to_partial(np.linalg.inv(spd))[0, 1] < 0.5):
                spds.append(spd)
        if not spds:
            print 'no'
        geo = geometric_mean(spds)
        corrs = [cov_to_corr(spd) for spd in spds]
        parts = [prec_to_partial(np.linalg.inv(spd)) for spd in spds]
        print "==============================="
        print [(corr[0,1], part[0,1]) for corr, part in zip(corrs, parts)]
        print np.mean(spds, axis=0)
        print "-------------------------"
        print geo
        print "-------------------------"
        print np.mean(parts, axis=0)
        print "============================="
        assert_greater(np.mean(corrs, axis=0)[0, 1], cov_to_corr(geo)[0, 1])
        assert_greater(cov_to_corr(geo)[0, 1], np.mean(parts, axis=0)[0, 1])

######################################################


def test_geometric_mean_properties():
    """Test geometric_mean function for random spd matrices
    """
    n_matrices = 40
    n_features = 4 ## 15
    from random import randint ###
    rand_gen = np.random.RandomState(randint(0, 100)) ### 0
    spds = []
    for k in range(n_matrices):
        spds.append(random_spd(n_features, eig_min=1., cond=10.,
                               rand_gen=rand_gen))
    input_spds = copy.copy(spds)
    geo = geometric_mean(spds)

    # Generic
    assert_is_instance(spds, list)
    for spd, input_spd in zip(spds, input_spds):
        assert_array_equal(spd, input_spd)
    assert(is_spd(geo))
##########################################
#   TODO : to remove
    a_corr = np.mean([cov_to_corr(spd) for spd in spds], axis=0) ###
    g_corr = cov_to_corr(geo) ###
    corr3 = prec_to_partial(np.mean([np.linalg.inv(spd) for spd in spds],
                                     axis=0)) ###
    print "ar corr \n", a_corr
    print "geo corr \n", g_corr
    print "part corr \n", corr3

    def g_larger_a(th):
        return (g_corr - a_corr) > th
    
    
    def a_larger_p(th):
        return (a_corr - corr3) > th
    
    
    def g_lower_a(th):
        return (-g_corr + a_corr) > th
    
    
    def a_lower_p(th):
        return (-a_corr + corr3) > th
##############################################    
    th = 1e-3
#    assert(np.any(np.logical_and(g_larger_a(th), a_larger_p(th))) == False)
#    assert(np.any(np.logical_and(g_lower_a(th), a_lower_p(th))) == False)
    from scipy.stats import gmean
    assert_greater(gmean([np.linalg.det(spd[:2, :2]) for spd in spds]),
                   np.linalg.det(geo[:2, :2]))
    a_corr_sq = np.mean([cov_to_corr(spd) ** 2 for spd in spds], axis=0) ###
    g_corr_sq = cov_to_corr(geo) ** 2
    off_diagonal = np.triu(np.abs(a_corr_sq), 1) - np.triu(np.abs(g_corr_sq), 1)
    np.fill_diagonal(off_diagonal, 1)
    print "========="
#    assert_greater(np.amax(gmean([np.diag(spd) for spd in spds]) - np.diag(geo)),
#                   0.)
#    assert_greater_equal(np.min(off_diagonal), 0.)

    # Invariance under reordering
    spds.reverse()
    spds.insert(0, spds[1])
    spds.pop(2)
    assert_array_almost_equal(geometric_mean(spds), geo)

    # Invariance under congruent transformation
    non_singular = random_non_singular(n_features, rand_gen=rand_gen)
    spds_cong = [non_singular.dot(spd).dot(non_singular.T) for spd in spds]
    assert_array_almost_equal(geometric_mean(spds_cong),
                              non_singular.dot(geo).dot(non_singular.T))

    # Invariance under inversion
    spds_inv = [linalg.inv(spd) for spd in spds]
    init = linalg.inv(np.mean(spds, axis=0))
    assert_array_almost_equal(geometric_mean(spds_inv, init=init),
                              linalg.inv(geo))

    # Gradient norm is decreasing
    grad_norm = grad_geometric_mean(spds)
    difference = np.diff(grad_norm)
    assert_greater_equal(0., np.amax(difference))

    # Check warning if gradient norm in the last step is less than
    # tolerance
    max_iter = 1
    tol = 1e-10
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        geo = geometric_mean(spds, max_iter=max_iter, tol=tol)
        assert_equal(len(w), 1)
    grad_norm = grad_geometric_mean(spds, max_iter=max_iter, tol=tol)
    assert_equal(len(grad_norm), max_iter)
    assert_greater(grad_norm[-1], tol)

    # Evaluate convergence. A warning is printed if tolerance is not reached
    for p in [.5, 1.]:  # proportion of badly conditionned matrices
        spds = []
        for k in range(int(p * n_matrices)):
            spds.append(random_spd(n_features, eig_min=1e-2, cond=1e6,
                                   rand_gen=rand_gen))
        for k in range(int(p * n_matrices), n_matrices):
            spds.append(random_spd(n_features, eig_min=1., cond=10.,
                                   rand_gen=rand_gen))
        if p < 1:
            max_iter = 30
        else:
            max_iter = 60
        geo = geometric_mean(spds, max_iter=max_iter, tol=1e-5)


def test_geometric_mean_checks():
    n_features = 5

    # Non square input matrix
    mat1 = np.ones((n_features, n_features + 1))
    assert_raises(ValueError, geometric_mean, [mat1])

    # Input matrices of different shapes
    mat1 = np.eye(n_features)
    mat2 = np.ones((n_features + 1, n_features + 1))
    assert_raises(ValueError, geometric_mean, [mat1, mat2])

    # Non spd input matrix
    assert_raises(ValueError, geometric_mean, [mat2])


def test_sym_to_vec():
    """Test sym_to_vec function"""
    # Check output value is correct
    sym = np.ones((3, 3))
    vec = np.array([1., sqrt(2), 1., sqrt(2),  sqrt(2), 1.])
    assert_array_almost_equal(sym_to_vec(sym), vec)
    mask_sym = sym > 0
    mask_vec = np.ones(6, dtype=bool)
    assert_array_equal(sym_to_vec(mask_sym, isometry=False), mask_vec)

    # Check vec_to_sym is the inverse function of sym_to_vec
    n_features = 19
    rand_gen = np.random.RandomState(0)
    m = rand_gen.rand(n_features, n_features)
    sym = m + m.T
    vec = sym_to_vec(sym)
    assert_array_almost_equal(vec_to_sym(vec), sym)
    syms = np.asarray([sym, 2. * sym, 0.5 * sym])
    vecs = sym_to_vec(syms)
    assert_array_almost_equal(vec_to_sym(vecs), syms)

    vec = sym_to_vec(sym, isometry=False)
    assert_array_almost_equal(vec_to_sym(vec, isometry=False),
                              sym)
    assert_array_almost_equal(vec[..., -n_features:], sym[..., -1, :])
    vecs = sym_to_vec(syms, isometry=False)
    assert_array_almost_equal(vec_to_sym(vecs, isometry=False),
                              syms)
    assert_array_almost_equal(vecs[..., -n_features:], syms[..., -1, :])


def test_vec_to_sym():
    """Test vec_to_sym function"""
    # Check error if unsuitable size
    vec = np.ones(31)
    assert_raises(ValueError, vec_to_sym, vec)

    # Check output value is correct
    vec = np.ones(6, )
    sym = np.array([[sqrt(2), 1., 1.], [1., sqrt(2), 1.],
                    [1., 1., sqrt(2)]]) / sqrt(2)
    assert_array_almost_equal(vec_to_sym(vec), sym)
    mask_vec = vec > 0
    mask_sym = np.ones((3, 3), dtype=bool)
    assert_array_equal(vec_to_sym(mask_vec, isometry=False), mask_sym)

    # Check sym_to_vec is the inverse function of vec_to_sym
    n = 41
    p = n * (n + 1) / 2
    rand_gen = np.random.RandomState(0)
    vec = rand_gen.rand(p)
    sym = vec_to_sym(vec)
    assert_array_almost_equal(sym_to_vec(sym), vec)
    sym = vec_to_sym(vec, isometry=False)
    assert_array_almost_equal(sym_to_vec(sym, isometry=False),
                              vec)
    vecs = np.asarray([vec, 2. * vec, 0.5 * vec])
    syms = vec_to_sym(vecs)
    assert_array_almost_equal(sym_to_vec(syms), vecs)
    syms = vec_to_sym(vecs, isometry=False)
    assert_array_almost_equal(sym_to_vec(syms, isometry=False),
                              vecs)


def test_prec_to_partial():
    """Test prec_to_partial function"""
    prec = np.array([[2., -1., 1.], [-1., 2., -1.], [1., -1., 1.]])
    partial = np.array([[1., .5, -sqrt(2.) / 2.], [.5, 1., sqrt(2.) / 2.],
                        [-sqrt(2.) / 2., sqrt(2.) / 2., 1.]])
    assert_array_almost_equal(prec_to_partial(prec), partial)


def test_fit_transform():
    """Test fit_transform method for class CovEmbedding"""
    n_subjects = 10
    n_features = 49
    n_samples = 200

    # Generate signals and compute empirical covariances
    covs = []
    signals = []
    rand_gen = np.random.RandomState(0)
    for k in range(n_subjects):
        signal = rand_gen.randn(n_samples, n_features)
        signals.append(signal)
        signal -= signal.mean(axis=0)
        covs.append((signal.T).dot(signal) / n_samples)

    input_covs = copy.copy(covs)
    for kind in ["correlation", "tangent", "precision", "partial correlation"]:
        estimators = {'kind': kind}
        cov_embedding = CovEmbedding(**estimators)
        covs_transformed = cov_embedding.fit_transform(signals)

        # Generic
        assert_is_instance(covs_transformed, np.ndarray)
        assert_equal(len(covs_transformed), len(covs))

        for k, vec in enumerate(covs_transformed):
            assert_equal(vec.size, n_features * (n_features + 1) / 2)
            assert_array_equal(input_covs[k], covs[k])
            cov_new = vec_to_sym(vec)
            assert(is_spd(covs[k]))

            # Positive definiteness if expected and output value checks
            if estimators["kind"] == "tangent":
                assert_array_almost_equal(cov_new, cov_new.T)
                geo_sqrt = map_sym(np.sqrt, cov_embedding.mean_cov_)
                assert(is_spd(geo_sqrt))
                assert(is_spd(cov_embedding.whitening_))
                assert_array_almost_equal(
                cov_embedding.whitening_.dot(geo_sqrt), np.eye(n_features))
                assert_array_almost_equal(geo_sqrt.dot(
                    map_sym(np.exp, cov_new)).dot(geo_sqrt), covs[k])
                corr1 = cov_to_corr(np.mean(covs, axis=0)) ###
                corr2 = cov_to_corr(cov_embedding.mean_cov_) ###
                corr3 = prec_to_partial(np.mean([np.linalg.inv(spd) for spd in covs],
                                                 axis=0)) ###
                assert_greater(0., np.min(corr1 - corr2)) ###
                assert_greater(0., np.min(corr2 - corr3)) ###

            if estimators["kind"] == "precision":
                assert(is_spd(cov_new))
                assert_array_almost_equal(cov_new.dot(covs[k]),
                                          np.eye(n_features))
            if estimators["kind"] == "correlation":
                assert(is_spd(cov_new))
                d = np.sqrt(np.diag(np.diag(covs[k])))
                assert_array_almost_equal(d.dot(cov_new).dot(d), covs[k])
            if estimators["kind"] == "partial correlation":
                prec = linalg.inv(covs[k])
                d = np.sqrt(np.diag(np.diag(prec)))
                assert_array_almost_equal(d.dot(cov_new).dot(d), -prec +\
                    2 * np.diag(np.diag(prec)))