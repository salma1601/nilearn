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
from nilearn.connectivity.tests.test_embedding import random_spd, random_diagonal, \
    random_non_singular

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


def geometric_mean_n(mats, init=None, max_iter=10, tol=1e-7):
    # Shape and symmetry positive definiteness checks
    n_features = mats[0].shape[0]
    for mat in mats:
        if mat.shape[0] != n_features:
            raise ValueError("Matrices are not of the same shape.")

    # Initialization
    mats = np.array(mats)
    if init is None:
        geo = np.mean(mats, axis=0)
    else:
        if init.shape[0] != n_features:
            raise ValueError("Initialization has not the correct shape.")
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
        vals_log, vecs_log = linalg.eigh(logs_mean - np.diag(np.diag(
            logs_mean)))
        geo_sqrt = map_eig(np.sqrt, vals_geo, vecs_geo)
        geo = geo_sqrt.dot(map_eig(np.exp, vals_log * step, vecs_log)).dot(
            geo_sqrt)  # Move along the geodesic with step size step
        np.fill_diagonal(geo, 1.)

        # Update the norm and the step size
        if norm < norm_old:
            norm_old = norm
        if norm > norm_old:
            print "slow down"
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


def geometric_mean_adaptative(mats, init=None, max_iter=10, tol=1e-7):
    # Shape and symmetry positive definiteness checks
    n_features = mats[0].shape[0]
    for mat in mats:
        if mat.shape[0] != n_features:
            raise ValueError("Matrices are not of the same shape.")

    # Initialization
    mats = np.array(mats)
    if init is None:
        geo = np.mean(mats, axis=0)
    else:
        if init.shape[0] != n_features:
            raise ValueError("Initialization has not the correct shape.")
        geo = init

    norm_old = np.inf
    step = 1.
    entered = False
    # Gradient descent
    for n in range(max_iter):
        # Computation of the gradient
        vals_geo, vecs_geo = linalg.eigh(geo)
        geo_sqrt = map_eig(np.sqrt, vals_geo, vecs_geo)
        geo_inv_sqrt = map_eig(np.sqrt, 1. / vals_geo, vecs_geo)
        whitened_mats = [geo_inv_sqrt.dot(mat).dot(geo_inv_sqrt)
            for mat in mats]
        conds = [np.linalg.cond(mat) for mat in spds]
        cond = gmean(conds)
        whitened_conds = [np.linalg.cond(mat) for mat in whitened_mats]
        whitened_cond = gmean(whitened_conds)
        print cond
        print whitened_cond
        if whitened_cond < cond and n > 1 and entered == False:
            print " adapting "
            entered = True
            whitened_init = geo_inv_sqrt.dot(init).dot(geo_inv_sqrt)
            geo2 = geometric_mean_adaptative(whitened_mats,
                                             max_iter=max_iter - 1, tol=tol,
                                             init=whitened_init)
            init = geo_sqrt.dot(geo2).dot(geo_sqrt)
        else:
            logs = [map_sym(np.log, w_mat) for w_mat in whitened_mats]
            logs_mean = np.mean(logs, axis=0)  # Covariant derivative is
                                               # - geo.dot(logms_mean)

            if np.any(np.isnan(logs_mean)):
                raise FloatingPointError("Nan value after logarithm operation.")
            norm = np.linalg.norm(logs_mean)  # Norm of the covariant derivative on
                                              # the tangent space at point geo
    
            # Update of the minimizer
            vals_log, vecs_log = linalg.eigh(logs_mean)
            geo_sqrt = map_eig(np.sqrt, vals_geo, vecs_geo)
            geo = geo_sqrt.dot(map_eig(np.exp, vals_log * step, vecs_log)).dot(
                geo_sqrt)  # Move along the geodesic with step size step

            # Update the norm and the step size
            if norm < norm_old:
                norm_old = norm
            if norm > norm_old:
                step = step / 2.
                norm = norm_old
            if tol is not None and norm / geo.size < tol:
                break
            init = geo

    print n
    if tol is not None and norm / geo.size >= tol:
        warnings.warn("Maximum number of iterations {0} reached without " \
                      "getting to the requested tolerance level " \
                      "{1}.".format(max_iter, tol))
    return geo


def geometric_mean_stoch(mats, init=None, max_iter=10, tol=1e-7):
    # Shape and symmetry positive definiteness checks
    n_features = mats[0].shape[0]
    for mat in mats:
        if mat.shape[0] != n_features:
            raise ValueError("Matrices are not of the same shape.")

    # Initialization
    mats = np.array(mats)
    if init is None:
        geo = np.mean(mats, axis=0)
    else:
        if init.shape[0] != n_features:
            raise ValueError("Initialization has not the correct shape.")
        geo = init

    norm_old = np.inf
    step = 1.
    # Gradient descent
    for n in range(max_iter):
        # Computation of the gradient
        vals_geo, vecs_geo = linalg.eigh(geo)
        geo_sqrt = map_eig(np.sqrt, vals_geo, vecs_geo)
        geo_inv_sqrt = map_eig(np.sqrt, 1. / vals_geo, vecs_geo)
        n_matrices = len(mats)
        rand_gen = np.random.RandomState(0)
        kept = rand_gen.randint(n_matrices)
        w_mat = geo_inv_sqrt.dot(mats[kept]).dot(geo_inv_sqrt)
        logs_mean = map_sym(np.log, w_mat)  # Covariant derivative is
                                         # - geo.dot(logms_mean)

        if np.any(np.isnan(logs_mean)):
            raise FloatingPointError("Nan value after logarithm operation.")

        # Update of the minimizer
        vals_log, vecs_log = linalg.eigh(logs_mean)
        geo_sqrt = map_eig(np.sqrt, vals_geo, vecs_geo)
        geo = geo_sqrt.dot(map_eig(np.exp, vals_log * step, vecs_log)).dot(
            geo_sqrt)  # Move along the geodesic with step size step

        # Update the norm and the step size
        if n % 10 == 0:
            whitened_mats = [geo_inv_sqrt.dot(mat).dot(geo_inv_sqrt)
                for mat in mats]
            logs = [map_sym(np.log, w_mat) for w_mat in whitened_mats]
            logs_mean = np.mean(logs, axis=0)  # Covariant derivative is
            norm = np.linalg.norm(logs_mean)  # Norm of the covariant derivative on
                                              # the tangent space at point geo
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


def geodesic(P0, P1, time):
    from scipy import linalg
    from scipy.linalg import logm, expm
    return P0.dot(expm(time * logm(linalg.inv(P0).dot(P1))))


def parallel_transport(A, B, C, t, tau):
    from scipy import linalg
    from scipy.linalg import logm, expm
    tran1 = expm(-t * logm(linalg.inv(A).dot(B)))
    tran2 = expm(t * logm(linalg.inv(A).dot(C)))
    tran = geodesic(A, B, t).dot(expm(tau * logm(tran1.dot(tran2))))
    return tran


def search_grid(A, B, C, G, eps=1e-2, start1=0., end1=1.,step1=.1,
                start2=0., end2=1., step2=.1):
    found = False
    diff = np.inf
    for i in np.arange(start1, end1, step1):
        for j in np.arange(start2, end2, step2):
            print diff
            M = parallel_transport(A, B, C, t=i, tau=j)
            if diff > np.amax(abs(M - G)):
                diff = np.amax(abs(M - G))
                time = i
                tau = j
            if np.amax(abs(M - G)) < eps:
                found = True
                break

    return diff, time, tau, found

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


if __name__ == "__adapt__":
    n_features = 5
    for p in [0., .5, 1.]:  # proportion of badly conditionned matrices
        spds = generate_spds(n_matrices=40, n_features=5, p=p)
        if p < 1:
            max_iter = 30
        else:
            max_iter = 60
        tol = 1e-5
        geo = geometric_mean(spds, max_iter=max_iter, tol=tol)
        print "adaptative"
        geo2 = geometric_mean_adaptative(spds, max_iter=max_iter, tol=tol)
        assert_array_almost_equal(geo2, geo)

if __name__ == "__stoch__":
    n_features = 39
    n_matrices = 40
    warnings.simplefilter("always")

    for p in [0., 0., 1.]:  # proportion of badly conditionned matrices
        spds = generate_spds(n_matrices=40, n_features=n_features, p=p)
        if p < 1:
            max_iter = 30
        else:
            max_iter = 60
        tol = 1e-5
        print "normal"
        geo = geometric_mean(spds, max_iter=max_iter, tol=tol)
        geo_n = geometric_mean_stoch(spds, max_iter=2 * max_iter, tol=tol)


if __name__ == "__main__":
    n_features = 3
    n_matrices = 3
    for p in [0., .5, 1.]:  # proportion of badly conditionned matrices
#        spds = generate_spds(n_matrices=3, n_features=n_features, p=p)
        spds = []
        for n in range(3):
            spds.append(random_diagonal(n_features))
        spds = generate_spds(n_matrices=3, n_features=n_features, p=p)
        if p < 1:
            max_iter = 30
        else:
            max_iter = 60
        tol = 1e-5
        print "normal"
        geo = geometric_mean(spds, max_iter=max_iter, tol=tol)
        diff, time, tau, found = search_grid(spds[0], spds[1], spds[2], geo, eps=1e-2)