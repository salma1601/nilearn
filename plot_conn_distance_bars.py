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
from sklearn.covariance import EmpiricalCovariance, LedoitWolf
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
standard_distances = {}
robust_distances = {}
for distance_type in ['euclidean', 'geometric']:
    standard_distances[distance_type] = [analyzing._compute_distance(
        mean_connectivity['covariance'], subject_connectivity,
        distance_type=distance_type)
        for subject_connectivity in subjects_connectivity['covariance']]
    robust_distances[distance_type] = [analyzing._compute_distance(
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
feature['geometric'] = [(np.linalg.eigvalsh(subject_connectivity)).prod()
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
spreading_euc = analyzing.compute_spreading(
    subjects_connectivity['correlation'])
spreading_geo = analyzing.compute_geo_spreading(
    subjects_connectivity['correlation'])

displacement = np.diff(dataset.motion[condition], axis=1)
norm_displacement = np.linalg.norm(displacement[..., :3], axis=-1)
motion = np.max(norm_displacement, axis=1)
for spreading, distance_type in zip([spreading_euc, spreading_geo],
                                    ['euclidean', 'geometric']):
    # Relate feature and average squared distance to the rest of subjects
    plt.figure(figsize=(5, 4.5))
    plt.scatter(np.log10(feature[distance_type]), motion)  #spreading
    plt.xlabel('covariance {}'.format(feature_name[distance_type]))
    plt.ylabel('average {} distance'.format(distance_type))


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
                color=color,
                alpha=.3)
    plt.xlabel('subject rank, sorted by covariance {}'
               .format(feature_name[distance_type]))
    axes = plt.gca()
    axes.yaxis.tick_right()
    plt.ylabel('{} distance to means'.format(distance_type))
    plt.legend(loc='lower right')
    plt.savefig('/home/sb238920/CODE/salma/figures/{}_distance_bars_rs1.pdf'
                .format(distance_type))
plt.show()
