import numpy as np
from scipy import linalg

from sklearn.covariance import LedoitWolf
from nilearn.connectivity import ConnectivityMeasure
from nilearn.connectivity.connectivity_matrices import (_form_symmetric,
                                                        _map_eigenvalues)


def compute_connectivity(subjects, measures=['covariance', 'precision',
                                             'tangent', 'correlation',
                                             'partial correlation'],
                         cov_estimator=LedoitWolf()):
    """ Computes the individual and group connectivity matrices for given
    subjects."""
    all_matrices = {}
    mean_matrices = {}
    for measure in measures:
        cov_embedding = ConnectivityMeasure(cov_estimator=cov_estimator,
                                            kind=measure)
        matrices = cov_embedding.fit_transform(subjects)
        all_matrices[measure] = matrices
        if measure == 'tangent':
            mean = cov_embedding.mean_cov_
        else:
            mean = matrices.mean(axis=0)
        mean_matrices[measure] = mean
    return all_matrices, mean_matrices


def _compute_distance(matrix1, matrix2, distance_type='euclidean'):
    """Computes the average distance between 2 matrices.

    Parameters
    ==========
    matrix1 : numpy.ndarray, shape (n_features, n_features)
        The first matrix.

    matrix2 : numpy.ndarray, shape (n_features, n_features)
        The second matrix.

    distance_type : one of {euclidean, geometric}
        The distance type to use.

    Returns
    =======
    distance : float
        The distance from each matrix to the others.
    """
    if distance_type == 'euclidean':
        distance = np.linalg.norm(matrix1 - matrix2)
    elif distance_type == 'geometric':
        vals, vecs = linalg.eigh(matrix1)
        matrix1_inv_sqrt = _form_symmetric(np.sqrt, 1. / vals, vecs)
        displacement_matrix = _map_eigenvalues(
            np.log, matrix1_inv_sqrt.dot(matrix2).dot(matrix1_inv_sqrt))
        distance = np.linalg.norm(displacement_matrix)
    else:
        raise ValueError('expected distance type {0} or {1}, got {2} type '
                         .format('euclidean', 'geometric', distance_type))
    return distance


def compute_pairwise_distances(matrices, distance_type='euclidean'):
    """Computes the pairwise distances from each matrix to the others.

    Parameters
    ==========
    matrices : numpy.ndarray, shape (n_matrices, n_features, n_features)
        The input matrices.

    Returns
    =======
    distance_matrix : numpy.ndarray, shape (n_matrices, n_matrices)
        The pairwise distances.
    """
    n_matrices = matrices.shape[0]
    matrix_distances = np.zeros((n_matrices, n_matrices))
    # TODO: remove for loop
    for i, j in zip(*np.triu_indices(n_matrices, 1)):
        matrix_distances[i, j] = _compute_distance(matrices[i], matrices[j],
                                                   distance_type=distance_type)
    matrix_distances = matrix_distances + matrix_distances.T
    return matrix_distances


def _compute_variance(matrix, matrices):
    """Computes the average squared distance from one matrix to a list of
    matrices.

    Parameters
    ==========
    matrices : numpy.ndarray, shape (n_matrices, n_features, n_features)
        The input matrices.

    Returns
    =======
    variance : float
        The average squared distance from each matrix to the others.
    """
    distances = [np.linalg.norm(matrix - m) ** 2 for m in matrices]
    return np.mean(distances)


def _compute_std(matrix, matrices):
    """Computes the average distance from one matrix to a list of matrices.

    Parameters
    ==========
    matrices : numpy.ndarray, shape (n_matrices, n_features, n_features)
        The input matrices.

    Returns
    =======
    variance : float
        The average squared distance from each matrix to the others.
    """
    distances = [np.linalg.norm(matrix - m) for m in matrices]
    return np.mean(distances)


def _compute_geo_variance(matrix, matrices):
    """Computes the average squared geometric distance from one matrix to a
    list of matrices.

    Parameters
    ==========
    matrices : numpy.ndarray, shape (n_matrices, n_features, n_features)
        The input matrices.

    Returns
    =======
    variance : float
        The average distance from each matrix to the others.
    """
    vals, vecs = linalg.eigh(matrix)
    matrix_inv_sqrt = _form_symmetric(np.sqrt, 1. / vals, vecs)
    whitened_matrices = [_map_eigenvalues(
        np.log, matrix_inv_sqrt.dot(m).dot(matrix_inv_sqrt)) for m in matrices]
    distances = [np.linalg.norm(w) ** 2 for w in whitened_matrices]
    return np.mean(distances)


def _compute_geo_std(matrix, matrices):
    """Computes the average geometric distance from one matrix to a list of
    matrices.

    Parameters
    ==========
    matrices : numpy.ndarray, shape (n_matrices, n_features, n_features)
        The input matrices.

    Returns
    =======
    variance : float
        The average distance from each matrix to the others.
    """
    vals, vecs = linalg.eigh(matrix)
    matrix_inv_sqrt = _form_symmetric(np.sqrt, 1. / vals, vecs)
    whitened_matrices = [_map_eigenvalues(
        np.log, matrix_inv_sqrt.dot(m).dot(matrix_inv_sqrt)) for m in matrices]
    distances = [np.linalg.norm(w) ** 2 for w in whitened_matrices]
    return np.mean(distances)


def compute_spreading(matrices):
    """Computes the average squared distance from each matrix to the others.

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


def compute_std_spreading(matrices):
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
        variances.append(_compute_std(matrix, matrices[indices, ...]))
    return variances


def compute_geo_spreading(matrices):
    """Computes the average squared geometric distance from each matrix to the
    others.

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
        variances.append(_compute_geo_variance(matrix, matrices[indices, ...]))
    return variances


def compute_geo_std_spreading(matrices):
    """Computes the average squared geometric distance from each matrix to the
    others.

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
        variances.append(_compute_geo_std(matrix, matrices[indices, ...]))
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


def assert_is_outlier(outlier, inliers):
    """Asserts if a given outlier is greater than a sequence of inliers and
    prints the number of failing indices.

    Parameters
    ----------
    outlier : numpy.array
        Square matrix.

    inliers : array like
        Sequence of square matrices.
    """
    eigenvalues_outlier = np.linalg.eigvalsh(outlier)
    failing_indices = []
    for index, inlier in enumerate(inliers):
        eigenvalues_inlier = np.linalg.eigvalsh(inlier)
        try:
            np.testing.assert_array_less(eigenvalues_inlier,
                                         eigenvalues_outlier)
        except AssertionError:
            failing_indices.append(index)

    if failing_indices:
        raise AssertionError('outlier not greater than inliers {0}'.format(
            failing_indices))
