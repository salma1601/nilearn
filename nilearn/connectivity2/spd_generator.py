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
########################################################
    # original code

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


subject_n = 0
n_samples = np.mean([subject.shape[0] for subject in subjects])
n_samples = int(n_samples)
spds1 = synthetic_data_wishart(mean_matrices[0] / n_samples,
                               n_samples=n_samples, n_subjects=n_subjects)

# Play with the covariance matrix
n = mean_matrices[2].shape[0]
p = n * (n + 1) / 2
block_cov = np.eye(p)
block_cov[:p / 2, :p / 2] *= .1
geo_sqrt = map_sym(np.sqrt, mean_matrices[2])
geo_sqrt_inv = map_sym(lambda x: 1/x, geo_sqrt)
cov = geo_sqrt_inv.dot(mean_matrices[0]).dot(geo_sqrt_inv)
cov = map_sym(np.log, cov)

from nilearn.connectivity.embedding import sym_to_vec
cov = np.zeros((p, p))
mean_vector = np.zeros((p,))
for matrix in all_matrices[2]:
    vector = sym_to_vec(matrix)
    mean_vector += vector / n_subjects
    cov += vector * vector[:, np.newaxis] / (n_subjects - 1)
cov -= mean_vector * mean_vector[:, np.newaxis] * (n_subjects + 1) / (n_subjects - 1)
epsilon = np.abs(np.min(np.linalg.eigvalsh(cov)))
cov += 2 * epsilon * np.eye(p)

from nilearn.connectivity.spd_generator import spd_covariance, compute_cov_mask
cov = spd_covariance(all_matrices[2], isometry=True)
mask = compute_cov_mask(n)
spds2 = synthetic_data_manifold(mean_matrices[2],
                               n_subjects=40, cov=cov)
for spds, distribution in zip([spds1, spds2], ['wishart, dof={}'.format(
                                    n_samples), 'gaussian manifold']):
    geo = geometric_mean(spds)
    corrs = [cov_to_corr(spd) for spd in spds]
    partials = [prec_to_partial(np.linalg.inv(spd)) for spd in spds]
    plot_matrix(geo, "gmean, " + distribution)
    plot_matrix(mean_matrices[2], "gmean, data")
    plot_matrix(np.mean(spds, axis=0), "amean, " + distribution)
    plot_matrix(mean_matrices[0], "amean, data")
    plot_matrix(np.mean(spds, axis=0) - geo, "amean - geo, " + distribution)
    plt.show()
    #plot_matrix(cov_to_corr(geo), "corr(gmean), whishart")
    #plot_matrix(np.mean(corrs, axis=0), "amean of corrs, whishart")
    #plot_matrix(np.mean(partials, axis=0), "amean of partials, whishart")
    plot_matrix(np.mean(corrs, axis=0) - cov_to_corr(geo),
                "mean of corrs - corr(gmean), " + distribution)
    plot_matrix(mean_matrices[3] - cov_to_corr(mean_matrices[2]),
                "mean of corrs - corr(gmean), data")
    plot_matrix(np.mean(corrs, axis=0) - np.mean(partials, axis=0),
                "mean of corrs - mean of partials, " + distribution)
    plot_matrix(mean_matrices[3] - np.mean(partials, axis=0),
                "mean of corrs - mean of partial corrs, data")
    plot_matrix(cov_to_corr(geo) - mean_matrices[4],
                "corr(gmean) - mean of partial corrs, " + distribution)
    plot_matrix(cov_to_corr(mean_matrices[2]) - mean_matrices[4],
                "corr(gmean) - mean of partial corrs, data")
    plt.show()

for spds, distribution, color in zip([spds1, spds2],
                                     ['wishart', 'gaussian manifold'],
                                     ['b', 'r']):
    geo = geometric_mean(spds)
    corrs = [cov_to_corr(spd) for spd in spds]
    partials = [prec_to_partial(np.linalg.inv(spd)) for spd in spds]
    dist_plot, = plt.plot((np.mean(corrs, axis=0) - cov_to_corr(geo)).ravel(),
                          (np.mean(corrs, axis=0) - np.mean(
                           partials, axis=0)).ravel(),
                          color + '.', label=distribution + ' distribution')
adhd_plot, = plt.plot((mean_matrices[3] - cov_to_corr(
                      mean_matrices[2])).ravel(),
                      (mean_matrices[3] - mean_matrices[4]).ravel(), 'g.',
                      label='ADHD dataset')
plt.xlabel('corrs - corr(gmean)')
plt.ylabel('corrs - partials')
plt.legend()
plt.title('differences between connectivity measures over regions, {}'.format(
        estimators[n_estimator][0]))
from scipy.stats import pearsonr
r, pval = pearsonr((mean_matrices[3] - cov_to_corr(mean_matrices[2])).ravel(),
                   (mean_matrices[3] - mean_matrices[4]).ravel())
print('pearson corr = {}, pval = {}'.format(r, pval))
plt.show()
