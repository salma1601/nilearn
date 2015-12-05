"""This example plots the evolution of the distance from the mean covariance
and gmean to the least moving subject(s).
"""
import numpy as np
import matplotlib.pylab as plt

from nilearn.connectivity2 import analyzing

from funtk.connectivity import matrix_stats
# Load preprocessed abide timeseries extracted from harvard oxford atlas
from nilearn import datasets
abide = datasets.fetch_abide_pcp(DX_GROUP=2, derivatives=['rois_ho'])
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

# Sort subjects by maximal eigenvalue / noise
import nilearn.connectivity
cov_embedding = nilearn.connectivity.ConnectivityMeasure(kind='covariance')
subjects_covariance = cov_embedding.fit_transform(subjects)
max_eigenvalues = [np.linalg.eigvalsh(subject_connectivity).max() for
                   subject_connectivity in subjects_covariance]
indices_eig = np.argsort(max_eigenvalues)
subjects = np.array(subjects)[indices_eig]
n_subjects = len(subjects)
n_inliers = n_subjects / 2
max_outliers = n_subjects - n_inliers
low_motion_subjects = subjects[:n_inliers]
high_motion_subjects = subjects[n_inliers:]

# Estimate evolution of connectivity matrices
from sklearn.covariance import LedoitWolf
standard_measure = 'correlation'
measures = ["robust dispersion", standard_measure]

from joblib import Memory
mem = Memory('/home/sb238920/CODE/Parietal/nilearn/nilearn_cache/abide')

# Compute mean connectivity for low moving subjects
cov_embedding = nilearn.connectivity.ConnectivityMeasure(
    kind=standard_measure, cov_estimator=LedoitWolf())
subjects_connectivity = cov_embedding.fit_transform(low_motion_subjects,
                                                    mem=mem)
mean_connectivity_low_subjects = subjects_connectivity.mean(axis=0)

# Compute errors in mean connectivities
average_connectivity_errors = {}
std_connectivity_errors = {}
connectivity_errors = {}

max_combinations = 10
step = 20
from sklearn import cross_validation
for measure in measures:
    average_connectivity_errors[measure] = []
    std_connectivity_errors[measure] = []
for n_outliers in range(0, max_outliers + 1, step):
    print('{} outliers'.format(n_outliers))
    if n_outliers == 0:
        outliers_combinations = [(np.arange(max_outliers), np.array([]))]
    elif n_outliers < max_outliers:
        outliers_combinations = cross_validation.ShuffleSplit(
            max_outliers, n_iter=max_combinations, test_size=n_outliers,
            random_state=0)
    else:
        outliers_combinations = [((), np.arange(max_outliers))]
    for measure in measures:
        connectivity_errors[measure] = []

    # Add random combinations of n_outliers high moving subjects
    for n, (_, outliers_indices) in enumerate(outliers_combinations):
        if np.array(outliers_indices).shape != (0,):
            outliers = high_motion_subjects[outliers_indices]
            subjects_to_plot = np.concatenate(
                (low_motion_subjects, np.array(outliers)), axis=0)
        else:
            subjects_to_plot = low_motion_subjects

        # Compute mean connectivity
        for measure in measures:
            cov_embedding = nilearn.connectivity.ConnectivityMeasure(
                kind=measure, cov_estimator=LedoitWolf())
            subjects_connectivity = cov_embedding.fit_transform(
                subjects_to_plot, mem=mem)
            if measure == 'robust dispersion':
                if standard_measure == 'correlation':
                    mean_connectivity = matrix_stats.cov_to_corr(
                        cov_embedding.robust_mean_)
                else:
                    mean_connectivity = cov_embedding.robust_mean_
            else:
                mean_connectivity = subjects_connectivity.mean(axis=0)

            connectivity_errors[measure].append(np.linalg.norm(
                mean_connectivity_low_subjects - mean_connectivity))

    # Compute the average error for all combinations
    for measure in measures:
        average_connectivity_errors[measure].append(
            np.mean(connectivity_errors[measure]))
        std_connectivity_errors[measure].append(
            np.std(connectivity_errors[measure]))

for measure in measures:
    average_connectivity_errors[measure] = \
        np.array(average_connectivity_errors[measure])
    std_connectivity_errors[measure] = \
        np.array(std_connectivity_errors[measure])

# Plot the errors
figure = plt.figure(figsize=(5, 4.5))
for measure, color in zip(measures, ['red', 'blue']):
    if measure == standard_measure:
        label = 'arithmetic mean'
    elif measure == 'robust dispersion':
        if standard_measure == 'correlation':
            label = 'normalized geometric mean'
        else:
            label = 'geometric mean'
    plt.plot(np.arange(0, max_outliers + 1, step)[1:],
             average_connectivity_errors[measure][1:],
             label=label, color=color)
    axes = plt.gca()
    lower_bound = average_connectivity_errors[measure] -\
        std_connectivity_errors[measure]
    upper_bound = average_connectivity_errors[measure] +\
        std_connectivity_errors[measure]
    axes.fill_between(np.arange(0, max_outliers + 1, step)[1:],
                      lower_bound[1:], upper_bound[1:],
                      facecolor=color, alpha=0.2)
plt.rc('text', usetex=True)
plt.xlabel('number of noisy subjects used')
plt.xlim(step, max_outliers - step)
plt.xticks(range(0, max_outliers + 1, step)[1:])
figure.suptitle('impact of noisy subjects on mean ' + standard_measure,
                fontweight='bold', fontsize=14)
axes = plt.gca()
axes.yaxis.tick_right()
plt.ylabel('euclidean distance between mean of all subjects and\narithmetic '
           'mean of non-noisy subjects')
plt.legend(loc='lower right')
if standard_measure == "correlation":
    figure_name = 'abide_{}_euclidean_corr_curves2.pdf'.format(scaling_type)
else:
    figure_name = 'abide_{}_euclidean_curves2.pdf'.format(scaling_type)
plt.savefig('/home/sb238920/CODE/salma/figures/' + figure_name)
plt.show()
