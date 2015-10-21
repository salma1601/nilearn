"""This example plots the evolution of the distance from the mean covariance
and gmean to the least moving subject(s).
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

# Sort subjects by determinant
import nilearn.connectivity
cov_embedding = nilearn.connectivity.ConnectivityMeasure(kind='covariance')
subjects_covariance = cov_embedding.fit_transform(subjects)
determinant = [(np.linalg.eigvalsh(subject_connectivity) * 10.).prod()
               for subject_connectivity in
               subjects_covariance]
indices_eig = np.argsort(determinant)
subjects = np.array(subjects)[indices_eig]
n_subjects = len(subjects)
n_inliers = n_subjects / 2
max_outliers = n_subjects - n_inliers
low_motion_subjects = subjects[:n_inliers]
high_motion_subjects = subjects[n_inliers:]

# Estimate evolution of connectivity matrices
from sklearn.covariance import EmpiricalCovariance
measures = ["robust dispersion", "covariance"]

# Compute mean connectivity for low moving subjects
cov_embedding = nilearn.connectivity.ConnectivityMeasure(
    kind='covariance', cov_estimator=EmpiricalCovariance())
subjects_connectivity = cov_embedding.fit_transform(low_motion_subjects)
mean_connectivity_low_subjects = subjects_connectivity.mean(axis=0)

# Compute errors in mean connectivities
connectivity_errors = {}
average_connectivity_errors = {}
std_connectivity_errors = {}
for measure in measures:
    average_connectivity_errors[measure] = []
    std_connectivity_errors[measure] = []
from sklearn import cross_validation
max_combinations = 30
for n_outliers in range(max_outliers + 1):
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
                kind=measure, cov_estimator=EmpiricalCovariance())
            subjects_connectivity = cov_embedding.fit_transform(
                subjects_to_plot)
            if measure == 'robust dispersion':
                mean_connectivity = cov_embedding.robust_mean_#matrix_stats.cov_to_corr(
            else:
                mean_connectivity = subjects_connectivity.mean(axis=0)

            connectivity_errors[measure].append(analyzing._compute_distance(
                mean_connectivity_low_subjects, mean_connectivity,
                distance_type='geometric'))

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
plt.figure(figsize=(5, 4))
for measure, color in zip(measures, ['red', 'blue']):
    if measure == 'covariance':
        label = 'arithmetic mean'
    elif measure == 'robust dispersion':
        label = 'geometric mean'
    plt.plot(np.arange(max_outliers + 1),
             average_connectivity_errors[measure],
             label=label, color=color)
    axes = plt.gca()
    lower_bound = average_connectivity_errors[measure] -\
        std_connectivity_errors[measure]
    upper_bound = average_connectivity_errors[measure] +\
        std_connectivity_errors[measure]
    axes.fill_between(np.arange(max_outliers + 1), lower_bound, upper_bound,
                      facecolor=color, alpha=0.2)
plt.rc('text', usetex=True)
plt.xlabel('number of noisy subjects used')
axes = plt.gca()
axes.yaxis.tick_right()
plt.ylabel('geometric distance between mean of all subjects and\narithmetic '
           'mean of non-noisy subjects')
plt.legend(loc='lower right')
plt.savefig('/home/sb238920/CODE/salma/figures/geometric_curves.pdf')
plt.show()
