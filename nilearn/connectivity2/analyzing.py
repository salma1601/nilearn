import numpy as np

from sklearn.covariance import LedoitWolf
from nilearn.connectivity import CovEmbedding, vec_to_sym


def compute_connectivity(subjects, measures=['covariance', 'precision',
                                             'tangent', 'correlation',
                                             'partial correlation'],
                         cov_estimator=LedoitWolf()):
    """ Computes the individual and group connectivity matrices for given
    subjects."""
    all_matrices = {}
    mean_matrices = {}
    for measure in measures:
        cov_embedding = CovEmbedding(cov_estimator=cov_estimator, kind=measure)
        matrices = vec_to_sym(
            cov_embedding.fit_transform(subjects))
        all_matrices[measure] = matrices
        if measure == 'tangent':
            mean = cov_embedding.mean_cov_
        else:
            mean = matrices.mean(axis=0)
        mean_matrices[measure] = mean
    return all_matrices, mean_matrices


def _compute_variance(matrix, matrices):
    """Computes the average distance from each matrix to the others.

    Parameters
    ==========
    matrices : numpy.ndarray, shape (n_matrices, n_features, n_features)
        The input matrices.

    Returns
    =======
    variance : float
        The average distance from each matrix to the others.
    """
    distances = [np.linalg.norm(matrix - m) ** 2 for m in matrices]
    return np.mean(distances)


def compute_spreading(matrices):
    """Computes the average distance from each matrix to the others.

    Parameters
    ==========
    matrices : numpy.ndarray, shape (n_matrices, n_features, n_features)
        The input matrices.

    Returns
    =======
    spreading : list, length n_matrices
        The average distance from each matrix to the others.
    """
    variances = []
    n_matrices = matrices.shape[0]
    # TODO: remove for loop
    for matrix_n, matrix in enumerate(matrices):
        indices = range(n_matrices)
        indices.remove(matrix_n)
        variances.append(_compute_variance(matrix, matrices[indices, ...]))
    return variances


def split_sample(sample, metric='norm 2'):
    """Separating the higher half of a sample array from the lower half with
    respect to a given metric.

    Parameters
    ----------
    sample : numpy.ndarray, shape (n_observations, n_features)
        The sample to split

    metric : one of {'norm2', 'norm 1'}

    Returns
    -------
    halfs : tuple of 1D numpy.ndarray
        The splietted sample (lower half, higher half)
    """
    if metric == 'norm 2':
        values = np.linalg.norm(sample, axis=-1)
        median = np.median(values)
        halfs = (np.where(values <= median)[0], np.where(values > median)[0])
    return halfs