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

# Compute median translation
displacement = np.diff(dataset.motion[condition], axis=1)
norm_displacement = np.linalg.norm(displacement[..., :3], axis=-1)
motion = np.max(norm_displacement, axis=1)

# Estimate evolution of connectivity matrices
from sklearn.covariance import EmpiricalCovariance
standard_measure = 'covariance'
measures = ["robust dispersion", standard_measure]

# Compute errors in mean connectivities
average_connectivity_errors = {}
std_connectivity_errors = {}
connectivity_errors = {}

max_combinations = 30
import nilearn
from sklearn import cross_validation
for measure in measures:
    average_connectivity_errors[measure] = []
    std_connectivity_errors[measure] = []

max_subjects = len(subjects)
min_subjects = 10
subjects = np.array(subjects)
for n_subjects in range(min_subjects, max_subjects + 1, 2):
    print('{} subjects'.format(n_subjects))
    # Select n_subjects random subjects
    if n_subjects < max_subjects:
        subjects_combinations = cross_validation.ShuffleSplit(
            max_subjects, n_iter=max_combinations, test_size=n_subjects,
            random_state=1)
    else:
        subjects_combinations = [((), np.arange(max_subjects))]

    for measure in measures:
        connectivity_errors[measure] = []

    # Iterate over all combinations of subjects
    for n, (_, subjects_indices) in enumerate(subjects_combinations):
        used_subjects = subjects[subjects_indices]
        # Sort subjects
        cov_embedding = nilearn.connectivity.ConnectivityMeasure(
            kind='covariance')
        subjects_covariance = cov_embedding.fit_transform(used_subjects)
        max_eigenvalues = [np.linalg.eigvalsh(subject_connectivity).max() for
                           subject_connectivity in subjects_covariance]
        indices_eig = np.argsort(max_eigenvalues)
        used_subjects = used_subjects[indices_eig]
        low_motion_subjects = used_subjects[:n_subjects / 2]
        high_motion_subjects = used_subjects[n_subjects / 2:]

       # Compute mean connectivity for low moving subjects
        cov_embedding = nilearn.connectivity.ConnectivityMeasure(
            kind=standard_measure, cov_estimator=EmpiricalCovariance())
        subjects_connectivity = cov_embedding.fit_transform(low_motion_subjects)
        mean_connectivity_low_subjects = subjects_connectivity.mean(axis=0)

        # Compute mean connectivity
        for measure in measures:
            cov_embedding = nilearn.connectivity.ConnectivityMeasure(
                kind=measure, cov_estimator=EmpiricalCovariance())
            subjects_connectivity = cov_embedding.fit_transform(used_subjects)
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
plt.figure(figsize=(5, 4.5))
for measure, color in zip(measures, ['red', 'blue']):
    if measure == standard_measure:
        label = 'arithmetic mean'
    elif measure == 'robust dispersion':
        if standard_measure == 'correlation':
            label = 'corr(geometric mean)'
        else:
            label = 'geometric mean'
    plt.plot(np.arange(min_subjects, max_subjects + 1, 2)[:-1],
             average_connectivity_errors[measure][:-1],
             label=label, color=color)
    axes = plt.gca()
    lower_bound = average_connectivity_errors[measure] -\
        std_connectivity_errors[measure]
    upper_bound = average_connectivity_errors[measure] +\
        std_connectivity_errors[measure]
    axes.fill_between(np.arange(min_subjects, max_subjects + 1, 2)[:-1],
                      lower_bound[:-1], upper_bound[:-1],
                      facecolor=color, alpha=0.2)
plt.rc('text', usetex=True)
plt.xlim([min_subjects, max_subjects - 2])
plt.xlabel('total number of subjects used')
axes = plt.gca()
axes.yaxis.tick_right()
plt.ylabel('euclidean distance between mean of all subjects and\narithmetic '
           'mean of non-noisy subjects')
plt.legend(loc='upper right')
if standard_measure == "correlation":
    plt.savefig('/home/sb238920/CODE/salma/figures/euclidean_corr_curves_variable.pdf')
else:
    plt.savefig('/home/sb238920/CODE/salma/figures/euclidean_curves_variable.pdf')
plt.show()
