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

condition = 'Nbac3_Placebo'
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
for measure in measures + ['correlation']:
    cov_embedding = nilearn.connectivity.ConnectivityMeasure(
        kind=measure, cov_estimator=EmpiricalCovariance())
    subjects_connectivity[measure] = cov_embedding.fit_transform(subjects)
    if measure == 'robust dispersion':
        mean_connectivity[measure] = cov_embedding.robust_mean_
    else:
        mean_connectivity[measure] = \
            subjects_connectivity[measure].mean(axis=0)

# Compute the distances to means
from nilearn.connectivity2 import analyzing
distances = {}
distance_type = 'geometric'
for measure in measures:
    distances[measure] = [analyzing._compute_distance(mean_connectivity[measure],subject_connectivity,
         distance_type='geometric')
        for subject_connectivity in subjects_connectivity['covariance']]

# Relate motion and eigenvalues
max_eigenvalues = [np.linalg.eigvalsh(subject_connectivity).prod() for
                   subject_connectivity in subjects_connectivity['covariance']]
min_eigenvalues = [np.linalg.eigvalsh(subject_connectivity).min() for
                   subject_connectivity in subjects_connectivity['covariance']]

# Define the outliers : far away from the others
from nilearn.connectivity2 import analyzing
spreading_euc = analyzing.compute_spreading(
    subjects_connectivity['correlation'])
spreading_geo = analyzing.compute_geo_spreading(
    subjects_connectivity['correlation'])

for spreading, ylabel in zip([spreading_euc, spreading_geo],
                             ['eucledian', 'geometric']):
    # Relate maximal eigenvalues and spreading
    plt.figure(figsize=(5, 4.5))
    plt.scatter(max_eigenvalues, spreading)
    plt.xlabel('maximal covariance eigenvalue')
    plt.ylabel('average {} distance'.format(ylabel))

    # Relate minimal eigenvalues and spreading
    plt.figure()
    plt.scatter(min_eigenvalues, spreading)
    plt.xlabel('minimal covariance eigenvalue')
    plt.ylabel('correlation {} spreading'.format(ylabel))

# Plot the bars of distances
plt.figure()
indices_eig_max = np.argsort(max_eigenvalues)
indices_geo = np.argsort(spreading_geo)
plt.figure(figsize=(5, 4.5))
for measure, color in zip(measures, 'br'):
    if measure == 'covariance':
        label = 'arithmetic mean'
    elif measure == 'robust dispersion':
        label = 'geometric mean'
    plt.bar(np.arange(len(subjects)),
            np.array(distances[measure])[indices_eig_max],
            width=.7,
            label=label,
            color=color,
            alpha=.3)
plt.xlabel('subject rank, sorted by maximal eigenvalues')
axes = plt.gca()
axes.yaxis.tick_right()
plt.ylabel('euclidean distance to means')
plt.legend(loc='upper left')
plt.savefig('/home/sb238920/CODE/salma/figures/distance_bars_rs1.pdf')
plt.show()
