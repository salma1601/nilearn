"""This example plots the evolution of the distance from the mean covariance
and gmean starting from the worse subjects and adding the good subjects
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

# Compute median translation
displacement = np.diff(dataset.motion[condition], axis=1)
norm_displacement = np.linalg.norm(displacement[..., :3], axis=-1)
motion = np.max(norm_displacement, axis=1)

# Sort subjects by maximal eigenvalue / noise
indices = np.argsort(motion)
import nilearn.connectivity
cov_embedding = nilearn.connectivity.ConnectivityMeasure(kind='correlation')
subjects_covariance = cov_embedding.fit_transform(subjects)
max_eigenvalues = [np.linalg.eigvalsh(subject_connectivity).max() for
                   subject_connectivity in subjects_covariance]
indices_eig = np.argsort(max_eigenvalues)
subjects = np.array(subjects)[indices_eig]
n_subjects = len(subjects)
n_inliers = n_subjects / 2
max_outliers = n_subjects - n_inliers
low_motion_subjects = subjects[:n_inliers][::-1]  # start by lowest moving
high_motion_subjects = subjects[n_inliers:]

# Estimate evolution of connectivity matrices
from sklearn.covariance import EmpiricalCovariance
measures = ["robust dispersion", "correlation"]

# Compute mean connectivity for low moving subjects
cov_embedding = nilearn.connectivity.ConnectivityMeasure(
    kind='correlation', cov_estimator=EmpiricalCovariance())
subjects_connectivity = cov_embedding.fit_transform(low_motion_subjects)
mean_connectivity_low_subjects = subjects_connectivity.mean(axis=0)

# Compute errors in mean connectivities
average_connectivity_errors = {}
std_connectivity_errors = {}
connectivity_errors = {}
for measure in measures:
    connectivity_errors[measure] = []
    average_connectivity_errors[measure] = []
    std_connectivity_errors[measure] = []
rand_gen = np.random.RandomState(seed=0)
max_combinations = 10
from sklearn.utils import shuffle
for n_outliers in range(max_outliers):
    print('{} inliers'.format(n_outliers))
    inliers_combinations = [shuffle(
        low_motion_subjects, random_state=1)[:n_outliers] for k in
        range(max_combinations)]

    # Incrementally add high moving subjects
    for n, inliers in enumerate(inliers_combinations):
        if n in indices:
            if np.array(inliers).shape != (0,):
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
                    mean_connectivity = matrix_stats.cov_to_corr(
                        cov_embedding.robust_mean_)
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

# Plot the errors
for measure in measures:
    plt.plot(average_connectivity_errors[measure], label=measure)

plt.rc('text', usetex=True)
plt.xlabel('number of non-noisy matrices used')
plt.ylabel(r'$\|corr(mean) - amean\, of\, non-noisy\, correlations|$')
plt.legend(loc='lower left')
plt.show()
