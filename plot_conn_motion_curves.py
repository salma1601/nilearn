"""This example:
- splits the dataset into high moving and low moving groups of subjects
- runs two-sample Student's t-tests betweens the groups
- plots the connectivity matrices and graphs
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

condition = 'ReSt2_Placebo'
dataset = dataset_loader.load_conn(conn_folder_filt, conditions=[condition],
                                   standardize=False,
                                   networks=networks)
subjects = dataset.time_series[condition]

# Compute median translation
displacement = np.diff(dataset.motion[condition], axis=1)
median_abs_displacement = np.median(np.abs(displacement), axis=1)

# Sort subjects by maximal eigenvalue / noise
indices = np.argsort(median_abs_displacement[:, :3])
n_subjects = len(subjects)
max_outliers = 19 * n_subjects / 20
low_motion_subjects = subjects[:n_subjects - max_outliers]
high_motion_subjects = subjects[n_subjects - max_outliers:][::-1]  # start by highest moving

# Estimate evolution of connectivity matrices
from sklearn.covariance import EmpiricalCovariance
import nilearn.connectivity
measures = ["robust dispersion", "correlation", "partial correlation",
            "covariance", "precision"]
measures = ["robust dispersion", "correlation"]

# Compute mean connectivity for low moving subjects
cov_embedding = nilearn.connectivity.ConnectivityMeasure(
    kind='correlation', cov_estimator=EmpiricalCovariance())
subjects_connectivity = cov_embedding.fit_transform(low_motion_subjects)
mean_connectivity_low_subjects = subjects_connectivity.mean(axis=0)

# Compute errors in mean connectivities
mean_connectivity_error = {}
for measure in measures:
    mean_connectivities = []
    for n_outliers in range(max_outliers):
        # Incrementally add high moving subjects
        subjects_to_plot = low_motion_subjects +\
                           high_motion_subjects[:n_outliers]

        # Compute mean connectivity
        cov_embedding = nilearn.connectivity.ConnectivityMeasure(
            kind=measure, cov_estimator=EmpiricalCovariance())
        subjects_connectivity = cov_embedding.fit_transform(subjects_to_plot)
        if measure == 'robust dispersion':
            mean_connectivities.append(
                matrix_stats.cov_to_corr(cov_embedding.robust_mean_))
        else:
            mean_connectivities.append(subjects_connectivity.mean(axis=0))

        # Compute error in mean
        mean_connectivity_error[measure] = [
            np.linalg.norm(mean_connectivity_low_subjects - mean_connectivity)
            for mean_connectivity in mean_connectivities]

# Plot the errors
for measure in measures:
    plt.plot(mean_connectivity_error[measure], label=measure)

plt.xlabel('outliers fraction')
plt.legend()
plt.show()
