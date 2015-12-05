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

# Load preprocessed abide timeseries extracted from harvard oxford atlas
from nilearn import datasets
abide = datasets.fetch_abide_pcp(derivatives=['rois_ho'], DX_GROUP=2)
subjects_unscaled = abide.rois_ho

# Standardize the signals
scaling_type = 'normalized'
from nilearn import signal
if scaling_type == 'normalized':
    subjects = []
    for subject in subjects_unscaled:
        subjects.append(signal._standardize(subject))
else:
    subjects = subjects_unscaled

# Estimate connectivity matrices
from sklearn.covariance import LedoitWolf
import nilearn.connectivity
measures = ["correlation", "partial correlation", "robust dispersion"]
subjects_connectivity = {}
mean_connectivity = {}
from joblib import Memory
mem = Memory('/home/sb238920/CODE/Parietal/nilearn/nilearn_cache/abide')
for measure in measures:
    cov_embedding = nilearn.connectivity.ConnectivityMeasure(
        kind=measure, cov_estimator=LedoitWolf())
    subjects_connectivity[measure] = cov_embedding.fit_transform(subjects,
                                                                 mem=mem)
    if measure == "robust dispersion":
        mean_connectivity[measure] = cov_embedding.robust_mean_
    else:
        mean_connectivity[measure] = np.mean(subjects_connectivity[measure],
                                             axis=0)

# Sorting the subjects
from nilearn.connectivity2 import analyzing
indices = {}
features = {}
for measure in ['correlation', 'partial correlation']:
    features['euclidean ' + measure] = mem.cache(
        analyzing.compute_std_spreading)(subjects_connectivity[measure])
    features['euclidean squared ' + measure] = mem.cache(
        analyzing.compute_spreading)(subjects_connectivity[measure])

features['geometric squared'] = mem.cache(analyzing.compute_geo_spreading)(
    subjects_connectivity['covariance'])
features['geometric'] = mem.cache(analyzing.compute_geo_std_spreading)(
    subjects_connectivity['covariance'])
max_eigenvalues = [np.linalg.eigvalsh(subject_connectivity).max() for
                   subject_connectivity in subjects_connectivity['covariance']]
features['maximal eigenvalue'] = max_eigenvalues
features['determinant'] = [np.linalg.eigvalsh(subject_connectivity).prod() for
    subject_connectivity in subjects_connectivity['covariance']]

# Plot correlations and partial correlations matrices for inliers and outliers
zero_diag = True
n_inliers = 3
n_outliers = 1
n_subjects = len(subjects)
for criteria in ['geometric', 'euclidean correlation',
                                  'euclidean partial correlation']:
    print(criteria)
    indices = np.argsort(features[criteria])
    inliers_indices = indices[:n_inliers]
    outliers_indices = indices[n_subjects - n_outliers:]
    inliers = {}
    outliers = {}
    for measure in ['correlation', 'partial correlation']:
        inliers[measure] = subjects_connectivity[measure][inliers_indices]
        outliers[measure] = subjects_connectivity[measure][outliers_indices]
        titles = ['subject {}'.format(index) for index in
                  np.hstack((outliers_indices, inliers_indices))]
        xlabels = [np.round(features[criteria][index], 2) for index in
                   np.hstack((outliers_indices, inliers_indices))]
        cb_labels = [measure for index in
                     np.hstack((outliers_indices, inliers_indices))]
        figure = plt.figure(figsize=(10, 5))
        plot_matrices(np.vstack((outliers[measure], inliers[measure])),
                      figure=figure,
                      zero_diag=zero_diag,
                      titles=titles, n_rows=1,
                      xlabels=xlabels,
                      cb_labels=cb_labels)
        distance_type = criteria
        for old in ['squared', 'partial correlation', 'correlation']:
            distance_type = distance_type.replace(old, '')
        matrix_type = criteria
        for old in ['squared', 'geometric', 'euclidean']:
            matrix_type = matrix_type.replace(old, '')
        if 'geometric' in criteria:
            suptitle = 'outlier according to ' + distance_type +\
                       'distance'
        else:
            suptitle = 'outlier according to ' + distance_type +\
                       'distance between' + matrix_type
        figure.suptitle(suptitle,
                        fontsize=14, fontweight='bold')
        measure_name = measure.replace('partial correlation', 'partial')
        criteria_name = criteria
        for old, new in zip(['euclidean', 'geometric', 'squared',
                             'partial correlation', 'correlation', ' '],
                            ['euc', 'geo', 'sq', 'part', 'corr', '_']):
            criteria_name = criteria_name.replace(old, new)
        plt.savefig('/home/sb238920/CODE/salma/figures/outliers_{0}_{1}_rs1'
                    '.pdf'.format(measure_name, criteria_name))

# Plot correlations and partial correlations connectivity graphs for
# inliers and outliers
if False:
    from nilearn import plotting
    labels, region_coords = zip(*dataset.rois)
    node_color = ['g' if label in DMN else 'k' if label in WMN else 'm' for
                  label in labels]
    display_mode = 'z'
    edge_threshold = '90%'
    for criteria in []:
        indices = np.argsort(features[criteria])
        inliers_indices = indices[:n_inliers]
        outliers_indices = indices[n_subjects - n_outliers:]
        inliers = {}
        outliers = {}
        for measure in ['correlation', 'partial correlation', 'covariance']:
            inliers[measure] = subjects_connectivity[measure][inliers_indices]
            outliers[measure] = subjects_connectivity[measure][outliers_indices]
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
                    outlier,
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

#Try relate some motion parameters or variance to eigenvalues
if False:
    mvt_parameters = np.asarray(dataset.motion[condition])
    displacements = np.diff(dataset.motion[condition])
    norm_translation = np.linalg.norm(mvt_parameters[..., :3], axis=-1)
    norm_translation_disp = np.linalg.norm(displacements[..., :3], axis=-1)
    norm_rotation = np.linalg.norm(mvt_parameters[..., 3:], axis=-1)
    norm_rotation_disp = np.linalg.norm(displacements[..., 3:], axis=-1)
    motion = np.max(norm_translation_disp, axis=1)
    motion = np.median(norm_translation, axis=1)
    labels = ['sub ' + str(n) for n in range(40)]
    
    for n_plot, feature_name in enumerate(['geometric', 'euclidean correlation',
                                      'euclidean partial correlation']):
        plt.subplot(3, 1, n_plot + 1)
        plt.plot(motion, features[feature_name], '.')
        plt.ylabel(feature_name)
        for label, x, y in zip(labels, motion, features[feature_name]):
            p = plt.annotate(
                label, 
                xy = (x, y), xytext = (-20, 20),
                textcoords = 'offset points', ha = 'right', va = 'bottom',
                bbox = dict(boxstyle = 'round,pad=0.1', fc = 'yellow', alpha = 0.5),
                arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
    
    plt.xlabel('motion')
    plt.show()