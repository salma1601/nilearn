"""This example plots
- bars depicting the difference in distance to the amean and gmean w.r.t.
the maximal eigenvalue
- scatter of eucledian and geometric spreading and max/min eigenvalues
"""
import numpy as np
import matplotlib.pylab as plt

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
measures = ["covariance", "robust dispersion"]

# Compute mean connectivity for low moving subjects
from joblib import Memory
mem = Memory('/home/sb238920/CODE/Parietal/nilearn/nilearn_cache/abide')
mean_connectivity = {}
subjects_connectivity = {}
for measure in measures + ['correlation']:
    # TODO: cache cov_embedding
    cov_embedding = nilearn.connectivity.ConnectivityMeasure(
        kind=measure, cov_estimator=LedoitWolf())
    subjects_connectivity[measure] = cov_embedding.fit_transform(subjects,
                                                                 mem=mem)
    if measure == 'robust dispersion':
        mean_connectivity[measure] = cov_embedding.robust_mean_
    else:
        mean_connectivity[measure] = \
            subjects_connectivity[measure].mean(axis=0)

# Compute the distances to means
from nilearn.connectivity2 import analyzing
standard_distances = {}
robust_distances = {}
for distance_type in ['euclidean', 'geometric']:
    standard_distances[distance_type] = [mem.cache(analyzing._compute_distance)(
        mean_connectivity['covariance'], subject_connectivity,
        distance_type=distance_type)
        for subject_connectivity in subjects_connectivity['covariance']]
    robust_distances[distance_type] = [mem.cache(analyzing._compute_distance)(
        mean_connectivity['robust dispersion'], subject_connectivity,
        distance_type=distance_type)
        for subject_connectivity in subjects_connectivity['covariance']]

# Sort distances
feature = {}
feature_name = {'euclidean': 'maximal eigenvalue',
                'geometric': 'determinant'}
feature['euclidean'] = [np.linalg.eigvalsh(subject_connectivity).max() for
                        subject_connectivity in
                        subjects_connectivity['covariance']]
feature['geometric'] = [(np.linalg.eigvalsh(subject_connectivity) * 1.).prod()
                        for subject_connectivity in
                        subjects_connectivity['covariance']]
for distance_type in ['euclidean', 'geometric']:
    indices = np.argsort(feature[distance_type])
    standard_distances[distance_type] = np.array(
        standard_distances[distance_type])[indices]
    robust_distances[distance_type] = np.array(
        robust_distances[distance_type])[indices]

# Define the outliers : far away from the others
from nilearn.connectivity2 import analyzing
spreading_euc = mem.cache(analyzing.compute_spreading)(
    subjects_connectivity['correlation'])
spreading_geo = mem.cache(analyzing.compute_geo_spreading)(
    subjects_connectivity['correlation'])

# Plot the bars of distances
for distance_type in ['euclidean', 'geometric']:
    plt.figure(figsize=(5, 4.5))
    indices = np.argsort(feature[distance_type])
    print distance_type, indices
    for distances, color, label in zip([standard_distances[distance_type],
                                        robust_distances[distance_type]],
                                       'br',
                                       ['arithmetic mean', 'geometric mean']):
        plt.bar(np.arange(len(subjects)),
                distances,
                width=.7,
                label=label,
                edgecolor=color,
                color=color,
                alpha=.3)
    plt.xlabel('subject rank, sorted by covariance {}'
               .format(feature_name[distance_type]))
    plt.xlim(0, len(subjects))
    axes = plt.gca()
    axes.yaxis.tick_right()
    plt.ylabel('{} distance to means'.format(distance_type))
    plt.legend(loc='upper center')
    figure_name = 'abide_{0}_{1}_distance_bars_rs1.pdf'.format(scaling_type,
                                                               distance_type)
    plt.savefig('/home/sb238920/CODE/salma/figures/' + figure_name)
plt.show()
