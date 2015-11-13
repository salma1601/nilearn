"""This example plots the individual connectivity matrices and graphs for the
good and bad subjects identified by sorting w.r.t. different criteria
- the average squared euclidean distance to the other correlations, partial
correlations and covariances
- the average euclidean distance to the other correlations, partial
correlations and covariances
- the average squared euclidean distance to the other covariances
- the average squared geometric distance to the other covariances
- the maximal covariance eigenvalues
- the covariance determinant
"""
import numpy as np
import matplotlib.pylab as plt

from nilearn.connectivity2.analyzing import assert_is_outlier
from funtk.connectivity.matrix_stats import plot_matrices

import dataset_loader

# Specify the networks
WMN = ['IPL', 'LMFG_peak1',
       'RCPL_peak1', 'LCPL_peak3', 'LT']
AN = ['vIPS.cluster001', 'vIPS.cluster002',
      'pIPS.cluster001', 'pIPS.cluster002',
      'MT.cluster001', 'MT.cluster002',
      'FEF.cluster002', 'FEF.cluster001']
DMN = ['RTPJ', 'RDLPFC', 'AG.cluster001', 'AG.cluster002',
       'SFG.cluster001', 'SFG.cluster002',
       'PCC', 'MPFC', 'FP']
networks = [('WMN', WMN), ('AN', AN), ('DMN', DMN)]

# Specify the location of the  CONN project
conn_folders = np.genfromtxt(
    '/home/sb238920/CODE/anonymisation/conn_projects_paths.txt', dtype=str)
conn_folder_filt = conn_folders[0]
conn_folder_no_filt = conn_folders[1]

condition = 'ReSt1_Placebo'
dataset = dataset_loader.load_conn(conn_folder_no_filt, conditions=[condition],
                                   standardize=False,
                                   networks=networks)
subjects = dataset.time_series[condition]

# Estimate connectivity matrices
from sklearn.covariance import EmpiricalCovariance
import nilearn.connectivity
measures = ["covariance", "robust dispersion", 'partial correlation',
            'correlation', 'precision']

# Compute connectivity
mean_connectivity = {}
subjects_connectivity = {}
for measure in measures:
    cov_embedding = nilearn.connectivity.ConnectivityMeasure(
        kind=measure, cov_estimator=EmpiricalCovariance())
    subjects_connectivity[measure] = cov_embedding.fit_transform(subjects)
    if measure == 'robust dispersion':
        mean_connectivity[measure] = cov_embedding.robust_mean_
    else:
        mean_connectivity[measure] = \
            subjects_connectivity[measure].mean(axis=0)

# Sorting the subjects
from nilearn.connectivity2 import analyzing
indices = {}
features = {}
for measure in ['correlation', 'partial correlation']:
    features['euclidean ' + measure] = analyzing.compute_std_spreading(
        subjects_connectivity[measure])

for measure in ['correlation', 'partial correlation']:
    features['euclidean squared ' + measure] = analyzing.compute_spreading(
        subjects_connectivity[measure])

features['geometric squared'] = analyzing.compute_geo_spreading(
    subjects_connectivity['covariance'])
features['geometric'] = analyzing.compute_geo_std_spreading(
    subjects_connectivity['covariance'])
max_eigenvalues = [np.linalg.eigvalsh(subject_connectivity).max() for
                   subject_connectivity in subjects_connectivity['covariance']]
features['maximal eigenvalue'] = max_eigenvalues
determinant = [np.linalg.eigvalsh(subject_connectivity).prod() for
               subject_connectivity in subjects_connectivity['covariance']]
features['determinant'] = determinant

# Plot correlations and partial correlations matrices for inliers and outliers
zero_diag = True
n_inliers = 5
n_outliers = 2
n_subjects = len(subjects)
for criteria in features.keys():
    print(criteria)
    indices = np.argsort(features[criteria])
    inliers_indices = indices[:n_inliers]
    outliers_indices = indices[n_subjects - n_outliers:]
    inliers = {}
    outliers = {}
    for measure in ['correlation', 'partial correlation']:
        inliers[measure] = subjects_connectivity[measure][inliers_indices]
        titles = ['subject {}'.format(index + 1) for index in inliers_indices]
        plot_matrices(inliers[measure], zero_diag=zero_diag, titles=titles,
                      n_rows=2)
        outliers[measure] = subjects_connectivity[measure][outliers_indices]
        titles = ['subject {}'.format(index + 1) for index in outliers_indices]
        plot_matrices(outliers[measure], zero_diag=zero_diag, titles=titles)

    # Plot correlations and partial correlations connectivity graphs for
    # inliers and outliers
    from nilearn import plotting
    labels, region_coords = zip(*dataset.rois)
    node_color = ['g' if label in DMN else 'k' if label in WMN else 'm' for
                  label in labels]
    display_mode = 'z'
    edge_threshold = '95%'
    for measure in ['correlation', 'partial correlation']:
        for n, inlier in enumerate(inliers[measure]):
            plotting.plot_connectome(
                inlier,
                region_coords,
                node_color=node_color,
                edge_threshold=edge_threshold,
                display_mode=display_mode,
                title='inlier {0}, {1}'.format(n, measure))
            plt.savefig('/home/sb238920/CODE/salma/figures/inliers_{0}_{1}_rs1'
                        '.pdf'.format(measure, n))
        for n, outlier in enumerate(outliers[measure]):
            plotting.plot_connectome(
                inlier,
                region_coords,
                node_color=node_color,
                edge_threshold=edge_threshold,
                display_mode=display_mode,
                title='outlier {0}, {1}'.format(n, measure))
            plt.savefig('/home/sb238920/CODE/salma/figures/outliers_{0}_{1}_rs1'
                        '.pdf'.format(measure, n))
    plt.show()

# Scatter plot some features
plt.figure()
plt.scatter(features['geometric'],
            features['euclidean correlation'])
plt.xlabel('geometric')
plt.ylabel('euclidean correlation')
plt.figure()
plt.scatter(features['geometric'],
            features['euclidean partial correlation'])
plt.xlabel('geometric')
plt.ylabel('euclidean partial correlation')
plt.figure()
plt.scatter(features['euclidean correlation'],
            features['euclidean partial correlation'])
plt.xlabel('euclidean correlation')
plt.ylabel('euclidean partial correlation')
plt.show()