"""This example plots the evolution of the distance from the mean covariance
and gmean to the least moving subject(s).
"""
import numpy as np
import matplotlib.pylab as plt

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

condition = 'Nbac2_Placebo'
dataset = dataset_loader.load_conn(conn_folder_no_filt, conditions=[condition],
                                   standardize=False,
                                   networks=networks)
subjects = dataset.time_series[condition]

# Compute median translation
displacement = np.diff(dataset.motion[condition], axis=1)
displacement[..., 3:] *= 100.  # rescale rotations
median_abs_displacement = np.median(np.abs(displacement), axis=1)

# Sort subjects by maximal eigenvalue / noise
motion = np.linalg.norm(median_abs_displacement[:, :], axis=-1)
indices = np.argsort(motion)
motion = motion[indices]
subjects = np.array(subjects)[indices]
n_subjects = len(subjects)

# Estimate connectivity matrices
from sklearn.covariance import EmpiricalCovariance
import nilearn.connectivity
measures = ["covariance", "robust dispersion"]

# Compute mean connectivity for low moving subjects
mean_connectivity = {}
subjects_connectivity = {}
#matrix_stats.cov_to_corr
for measure in measures + ['partial correlation']:
    cov_embedding = nilearn.connectivity.ConnectivityMeasure(
        kind=measure, cov_estimator=EmpiricalCovariance())
    subjects_connectivity[measure] = cov_embedding.fit_transform(subjects)
    if measure == 'robust dispersion':
        mean_connectivity[measure] = cov_embedding.robust_mean_
    else:
        mean_connectivity[measure] = \
            subjects_connectivity[measure].mean(axis=0)


# Compute the distances to means
distances = {}
for measure in measures:
    distances[measure] = [
        np.linalg.norm(subject_connectivity - mean_connectivity[measure])
        for subject_connectivity in subjects_connectivity['covariance']]

# Relate motion and eigenvalues
max_eigenvalues = [np.linalg.eigvalsh(subject_connectivity).max() for
                   subject_connectivity in subjects_connectivity['covariance']]
min_eigenvalues = [np.linalg.eigvalsh(subject_connectivity).min() for
                   subject_connectivity in subjects_connectivity['covariance']]

# Define the outliers : far away from the others
from nilearn.connectivity2 import analyzing
spreading = analyzing.compute_spreading(subjects_connectivity['covariance'])

# Relate maximal eigenvalues and spreading
plt.figure()
plt.scatter(max_eigenvalues, spreading)
plt.xlabel('maximal eigenvalue')
plt.ylabel('spreading')

# Relate minimal eigenvalues and spreading
plt.figure()
plt.scatter(min_eigenvalues, spreading)
plt.xlabel('minimal eigenvalue')
plt.ylabel('spreading')

# Plot the bars of distances
plt.figure()
indices_dist = np.argsort(max_eigenvalues)
for measure, color in zip(measures, 'rg'):
    plt.bar(np.arange(n_subjects), np.array(distances[measure])[indices_dist],
            width=.3,
            label=measure, color=color, alpha=.5)
plt.legend(loc='upper left')

# Plot individual connectivities
from funtk.connectivity.matrix_stats import cov_to_corr
from nilearn import plotting

labels, region_coords = zip(*dataset.rois)
for n in range(3):
    plotting.plot_connectome(
        cov_to_corr(subjects_connectivity['covariance'][indices_dist, ...][n]),
        region_coords,
        edge_threshold=.22,
        title='typical correlations {}'.format(n))

    plotting.plot_connectome(
        cov_to_corr(subjects_connectivity['covariance'][indices_dist, ...][39 - n]),
        region_coords,
        edge_threshold=.22,
        title='typical correlations {}'.format(n))

    plotting.plot_connectome(
        subjects_connectivity['partial correlation'][indices_dist, ...][n],
        region_coords,
        edge_threshold=.2,
        title='typical partials {}'.format(n))

    plotting.plot_connectome(
        cov_to_corr(subjects_connectivity['partial correlation'][indices_dist, ...][39 - n]),
        region_coords,
        edge_threshold=.2,
        title='outliers partials {}'.format(n))

plt.show()
