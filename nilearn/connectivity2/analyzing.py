from sklearn.covariance import LedoitWolf
from nilearn.connectivity import CovEmbedding, vec_to_sym


def compute_connectivity(subjects, measures=['covariance', 'precision',
                                             'tangent', 'correlation',
                                             'partial correlation'],
                         cov_estimator=LedoitWolf()):
    """ Computes the individual and group connectivity matrices for given
    subjects."""
    all_matrices = []
    mean_matrices = []
    for measure in measures:
        estimator = {'cov_estimator': cov_estimator,
                     'kind': measure}
        cov_embedding = CovEmbedding(**estimator)
        matrices = vec_to_sym(
            cov_embedding.fit_transform(subjects))
        all_matrices.append(matrices)
        if measure == 'tangent':
            mean = cov_embedding.mean_cov_
        else:
            mean = matrices.mean(axis=0)
        mean_matrices.append(mean)
    return all_matrices, mean_matrices

