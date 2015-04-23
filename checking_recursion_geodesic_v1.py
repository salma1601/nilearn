"""
Compare the performance of geometric mean gradient descent algorithm
"""
import warnings
import numpy as np

from numpy.testing import assert_array_equal, assert_array_almost_equal
from scipy import linalg
from scipy.stats import gmean

from nilearn.connectivity.embedding import grad_geometric_mean, \
    geometric_mean, map_sym, map_eig
from nilearn.connectivity.tests.test_embedding import random_spd, random_diagonal


def generate_spds(n_matrices=40, n_features=39, p=0.):
    rand_gen = np.random.RandomState(0)
    spds = []
    for k in range(int(p * n_matrices)):
        spds.append(random_spd(n_features, eig_min=1e-2, cond=1e6,
                               rand_gen=rand_gen))
    for k in range(int(p * n_matrices), n_matrices):
        spds.append(random_spd(n_features, eig_min=1., cond=10.,
                               rand_gen=rand_gen))
    return spds


# Number of subjects reduction
def reduce_subjects(n_matrices=40, n_features=39):
    """Test geometric_mean function for random spd matrices
    """
    rand_gen = np.random.RandomState(0)

    for p in [0., .5, 1.]:  # proportion of badly conditionned matrices
        spds = []
        for k in range(int(p * n_matrices)):
            spds.append(random_spd(n_features, eig_min=1e-2, cond=1e6,
                                   rand_gen=rand_gen))
        for k in range(int(p * n_matrices), n_matrices):
            spds.append(random_spd(n_features, eig_min=1., cond=10.,
                                   rand_gen=rand_gen))
        print '-- proportion of bad conditioning is ', p
        print ' computing mean'
        g_geo = grad_geometric_mean(spds, max_iter=max_iter, tol=tol)
        print ' reducing subjects'
        spds2 = np.array(spds)
        means = .5 * (spds2[::2, ...] + spds2[1::2, ...])
        spds2 = [spd for spd in means]
        geo2 = geometric_mean(spds2, max_iter=max_iter, tol=tol)
        assert_array_almost_equal(np.mean(spds2, axis=0), np.mean(spds,
                                  axis=0))
        g_geo2 = grad_geometric_mean(spds2, max_iter=max_iter, tol=tol)
        print ' recomputing mean'
        g_geo = grad_geometric_mean(spds, max_iter=max_iter, tol=tol,
                                    init=geo2)
    return len(g_geo2), len(g_geo)


# Preconditionning
def precond(spds):
    multipliers = []
    medians = []
    n_features = spds[0].shape[0]
    spds2 = []
    for spd in spds:
        vals, vecs = np.linalg.eigh(spd)
        idx = vals.argsort()
        assert_array_equal(idx, np.arange(n_features))
        vals = vals[idx]
        vecs = vecs[:, idx]
        multipliers.append(vals[0])
        spds2.append(spd)  # / vals[0]
        medians.append(vals[n_features / 2])
    factor = np.amin(medians)
    d = np.ones(n_features)
    d[n_features / 2:] /= factor
    spds2 = [np.diag(d).dot(spd).dot(np.diag(d)) for spd in spds2]
    return d, multipliers, spds2


def geo_mean_three(spd1, spd2, spd3, max_iter=50, tol=1e-7):
    vals_spd2, vecs_spd2 = np.linalg.eigh(spd2)
    spd2_sqrt = map_eig(np.sqrt, vals_spd2, vecs_spd2)
    spd2_inv_sqrt = map_eig(np.sqrt, 1. / vals_spd2, vecs_spd2)
    geo12 = spd2_sqrt.dot(map_sym(np.sqrt,
        spd2_inv_sqrt.dot(spd1).dot(spd2_inv_sqrt))).dot(spd2_sqrt)

    # geodesic from geo12 to spd3
    vals_spd3, vecs_spd3 = np.linalg.eigh(spd3)
    spd3_sqrt = map_eig(np.sqrt, vals_spd3, vecs_spd3)
    spd3_inv_sqrt = map_eig(np.sqrt, 1. / vals_spd3, vecs_spd3)
    speed = spd3_sqrt.dot(map_sym(np.log, spd3_inv_sqrt.dot(geo12).dot(
        spd3_inv_sqrt))).dot(spd3_sqrt)
    geo = spd3
    t = .01
    norm_old = np.inf
    mats = [spd1, spd2, spd3]
    for n in range(max_iter):
        # Computation of the gradient
        vals_geo, vecs_geo = linalg.eigh(geo)
        geo_inv_sqrt = map_eig(np.sqrt, 1. / vals_geo, vecs_geo)
        whitened_mats = [geo_inv_sqrt.dot(mat).dot(geo_inv_sqrt)
            for mat in mats]
        logs = [map_sym(np.log, w_mat) for w_mat in whitened_mats]
        logs_mean = np.mean(logs, axis=0)  # Covariant derivative is
                                           # - geo.dot(logms_mean)

        if np.any(np.isnan(logs_mean)):
            raise FloatingPointError("Nan value after logarithm operation.")
        norm = np.linalg.norm(logs_mean)  # Norm of the covariant derivative on
                                          # the tangent space at point geo

        # Update of the minimizer
        geo = spd3_sqrt.dot(map_sym(np.exp, t * spd3_inv_sqrt.dot(speed).dot(
            spd3_inv_sqrt))).dot(spd3_sqrt)

        # Update the norm and the step size
        if norm < norm_old:
            norm_old = norm
        if norm > norm_old:
            t = t / 2.
            norm = norm_old
        if tol is not None and norm / geo.size < tol:
            break

    if tol is not None and norm / geo.size >= tol:
        warnings.warn("Maximum number of iterations {0} reached without " \
                      "getting to the requested tolerance level " \
                      "{1}.".format(max_iter, tol))
    return geo

def geometric_mean_lin(mats, init=None, max_iter=10, tol=1e-7):
    # Initialization
    mats = np.array(mats)
    if init is None:
        geo = np.mean(mats, axis=0)
    else:
        geo = init

    norm_old = np.inf
    step = 1.

    # Gradient descent
    for n in range(max_iter):
        # Computation of the gradient
        vals_geo, vecs_geo = linalg.eigh(geo)
        geo_inv_sqrt = map_eig(np.sqrt, 1. / vals_geo, vecs_geo)
        whitened_mats = [geo_inv_sqrt.dot(mat).dot(geo_inv_sqrt)
            for mat in mats]
        logs = [map_sym(np.log, w_mat) for w_mat in whitened_mats]
        logs_mean = np.mean(logs, axis=0)  # Covariant derivative is
                                           # - geo.dot(logms_mean)

        if np.any(np.isnan(logs_mean)):
            raise FloatingPointError("Nan value after logarithm operation.")
        norm = np.linalg.norm(logs_mean)  # Norm of the covariant derivative on
                                          # the tangent space at point geo

        # Update of the minimizer
        geo = geo + logs_mean * step

        # Update the norm and the step size
        if norm < norm_old:
            norm_old = norm
        if norm > norm_old:
            step = step / 2.
            norm = norm_old
        if tol is not None and norm / geo.size < tol:
            break
    print n
    if tol is not None and norm / geo.size >= tol:
        warnings.warn("Maximum number of iterations {0} reached without " \
                      "getting to the requested tolerance level " \
                      "{1}.".format(max_iter, tol))
    return geo


if __name__ == "__precond__":
#    for n_matrices in range(2, 200, 100):
    for p in [0., .5, 1.]:  # proportion of badly conditionned matrices
        spds = generate_spds(n_matrices=40, n_features=5, p=p)
        if p < 1:
            max_iter = 30
        else:
            max_iter = 60
        tol = 1e-5
        geo = geometric_mean(spds, max_iter=max_iter, tol=tol)
        d, multipliers, spds2 = precond(spds)
        geo2 = geometric_mean(spds2, max_iter=max_iter, tol=tol)
        assert_array_almost_equal(np.diag(1. / d).dot(geo2).dot(np.diag(
            1. / d)) * gmean(multipliers), geo)


if __name__ == "__main__":
#    for n_matrices in range(2, 200, 100):
    for p in [0., .5, 1.]:  # proportion of badly conditionned matrices
        spds = generate_spds(n_matrices=40, n_features=5, p=p)
        if p < 1:
            max_iter = 30
        else:
            max_iter = 60
        tol = 1e-5
        geo = geometric_mean(spds, max_iter=max_iter, tol=tol)
        geo2 = geometric_mean_lin(spds, max_iter=max_iter, tol=tol)
        assert_array_almost_equal(geo2, geo)