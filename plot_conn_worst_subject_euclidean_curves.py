"""This example plots the evolution of the distance from the mean covariance
and gmean to the half best subjects starting from the worst subjects and adding the good subjects
incrementally.
"""
import numpy as np
import matplotlib.pylab as plt

from nilearn.connectivity2 import analyzing

from funtk.connectivity import matrix_stats
import dataset_loader


# Specify the networks
WMN = ['IPL', 'LMFG_peak1',
       'RCPL_peak1', 'LCPL_peak3', 'LT']
AN = ['vIPS.cluster001', 'vIPS.cluster002',
      'pIPS.cluster001', 'pIPS.cluster002',
      'MT.cluster001', 'MT.cluster002',
      'FEF.cluster001', 'FEF.cluster002']
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

# Sort subjects by maximal eigenvalue / noise
import nilearn.connectivity
standard_measure = 'correlation'
cov_embedding = nilearn.connectivity.ConnectivityMeasure(kind='covariance')
subjects_covariance = cov_embedding.fit_transform(subjects)
max_eigenvalues = [np.linalg.eigvalsh(subject_connectivity).max() for
                   subject_connectivity in subjects_covariance]
indices_eig = np.argsort(max_eigenvalues)
subjects = np.array(subjects)[indices_eig]
n_subjects = len(subjects)
n_inliers = n_subjects / 2
max_outliers = n_subjects - n_inliers
max_inliers = n_inliers
low_motion_subjects = subjects[:n_inliers]
high_motion_subjects = subjects[n_inliers:]


# Estimate evolution of connectivity matrices
from sklearn.covariance import EmpiricalCovariance
measures = ["robust dispersion", standard_measure]

# Compute mean connectivity for low moving subjects
cov_embedding = nilearn.connectivity.ConnectivityMeasure(
    kind=standard_measure, cov_estimator=EmpiricalCovariance())
subjects_connectivity = cov_embedding.fit_transform(low_motion_subjects)
mean_connectivity_low_subjects = subjects_connectivity.mean(axis=0)

# Compute errors in mean connectivities
average_connectivity_errors = {}
std_connectivity_errors = {}
connectivity_errors = {}

max_combinations = 30

from sklearn import cross_validation
for measure in measures:
    average_connectivity_errors[measure] = []
    std_connectivity_errors[measure] = []

for n_inliers in range(max_inliers + 1):
    print('{} inliers'.format(n_inliers))
    if n_inliers == 0:
        inliers_combinations = [(np.arange(max_inliers), np.array([]))]
    elif n_inliers < max_inliers:
        inliers_combinations = cross_validation.ShuffleSplit(
            max_inliers, n_iter=max_combinations, test_size=n_inliers,
            random_state=0)
    else:
        inliers_combinations = [((), np.arange(max_inliers))]

    for measure in measures:
        connectivity_errors[measure] = []

    # Incrementally add high moving subjects
    for n, (_, inliers_indices) in enumerate(inliers_combinations):
        if np.array(inliers_indices).shape != (0,):
            inliers = low_motion_subjects[inliers_indices]
            subjects_to_plot = np.concatenate(
                (high_motion_subjects, np.array(inliers)), axis=0)
        else:
            subjects_to_plot = high_motion_subjects

        # Compute mean connectivity
        for measure in measures:
            cov_embedding = nilearn.connectivity.ConnectivityMeasure(
                kind=measure, cov_estimator=EmpiricalCovariance())
            subjects_connectivity = cov_embedding.fit_transform(
                subjects_to_plot)
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

    # Compute the average error for a given number of outliers
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
plt.figure(figsize=(5, 4.5))
for measure, color in zip(measures, ['red', 'blue']):
    if measure == standard_measure:
        label = 'arithmetic mean'
    elif measure == 'robust dispersion':
        if standard_measure == 'correlation':
            label = 'corr(geometric mean)'
        else:
            label = 'geometric mean'
    plt.plot(np.arange(max_inliers + 1),
             average_connectivity_errors[measure],
             label=label, color=color)
    axes = plt.gca()
    lower_bound = average_connectivity_errors[measure] -\
        std_connectivity_errors[measure]
    upper_bound = average_connectivity_errors[measure] +\
        std_connectivity_errors[measure]
    axes.fill_between(np.arange(max_inliers + 1), lower_bound, upper_bound,
                      facecolor=color, alpha=0.2)
plt.rc('text', usetex=True)
plt.xlabel('number of non-noisy subjects used')
axes = plt.gca()
axes.yaxis.tick_right()
plt.ylabel('euclidean distance between mean of all subjects and\narithmetic '
           'mean of non-noisy subjects')
plt.legend(loc='upper right')
if standard_measure == "correlation":
    plt.savefig('/home/sb238920/CODE/salma/figures/euclidean_corr_inverted_curves.pdf')
else:
    plt.savefig('/home/sb238920/CODE/salma/figures/euclidean_inverted_curves.pdf')
plt.show()
