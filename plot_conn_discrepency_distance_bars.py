"""This example plots
- bars depicting the difference in distance to the amean and gmean w.r.t.
the maximal eigenvalue
- scatter of eucledian and geometric spreading and max/min eigenvalues
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
measures = ["covariance", "robust dispersion"]

# Compute mean connectivity for low moving subjects
mean_connectivity = {}
subjects_connectivity = {}
for measure in measures + ['correlation', 'partial correlation', 'precision']:
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

# Compute average correlation and partial correlation variance
from nilearn.connectivity2 import analyzing
corr_variances = analyzing.compute_spreading(
    subjects_connectivity['correlation'])
part_variances = analyzing.compute_spreading(
    subjects_connectivity['partial correlation'])
cov_variances = analyzing.compute_spreading(
    subjects_connectivity['covariance'])
prec_variances = analyzing.compute_spreading(
    subjects_connectivity['precision'])
cov_geo_variances = analyzing.compute_geo_spreading(
    subjects_connectivity['covariance'])
geo_distances = analyzing.compute_geo_spreading(np.concatenate(
    (mean_connectivity['robust dispersion'][np.newaxis],
     subjects_connectivity['covariance']),
    axis=0))[1:]
eig = [np.linalg.eigvalsh(cov).max() for cov in
       subjects_connectivity['covariance']]
cond = [np.linalg.cond(cov).max() for cov in
       subjects_connectivity['covariance']]

# Relate average correlation and partials distance
plt.figure(figsize=(5, 4.5))
plt.scatter(corr_variances, part_variances)
plt.xlabel('average distance to correlations')
plt.ylabel('average distance to partials')

# Sort fraction
fraction = np.array(cov_variances) / np.array(prec_variances)
discrepency = np.abs(fraction - np.mean(fraction))
indices = np.argsort(discrepency)

# Plot the bars of distances
plt.figure(figsize=(5, 4.5))
for measure, color in zip(measures, 'br'):
    if measure == 'covariance':
        label = 'arithmetic mean'
    elif measure == 'robust dispersion':
        label = 'geometric mean'
    plt.bar(np.arange(len(subjects)),
            np.array(distances[measure])[np.argsort(fraction)],
            width=.7,
            label=label,
            color=color,
            alpha=.3)
plt.xlabel('subject rank, sorted by discrepency')
plt.ylabel('eucledian distance to means')
plt.legend(loc='upper left')
plt.show()
