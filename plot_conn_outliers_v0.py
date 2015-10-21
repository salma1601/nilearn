"""This example plots the evolution of the distance from the mean covariance
and gmean to the least moving subject(s).
"""
import numpy as np
import matplotlib.pylab as plt

from nilearn.connectivity2.analyzing import assert_is_outlier
from funtk.connectivity.matrix_stats import plot_matrices

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
measures = ["covariance", "robust dispersion", 'partial correlation',
            'correlation', 'precision']

# Compute connectivity
mean_connectivity = {}
subjects_connectivity = {}
for measure in measures:
    cov_embedding = nilearn.connectivity.ConnectivityMeasure(
        kind=measure, cov_estimator=EmpiricalCovariance())
    subjects_connectivity[measure] = cov_embedding.fit_transform(subjects)
    if measure == 'robust dispersion':
        mean_connectivity[measure] = cov_embedding.robust_mean_
    else:
        mean_connectivity[measure] = \
            subjects_connectivity[measure].mean(axis=0)

# Relate motion and eigenvalues
max_eigenvalues = [np.linalg.eigvalsh(subject_connectivity).max() for
                   subject_connectivity in subjects_connectivity['covariance']]
min_eigenvalues = [np.linalg.eigvalsh(subject_connectivity).min() for
                   subject_connectivity in subjects_connectivity['covariance']]
indices_eig = np.argsort(max_eigenvalues)
indices_eig_min = np.argsort(min_eigenvalues)

# Define the outliers : far away from the others
from nilearn.connectivity2 import analyzing
spreading_cov = analyzing.compute_spreading(subjects_connectivity['correlation'])
indices_cov = np.argsort(spreading_cov)
spreading_prec = analyzing.compute_spreading(subjects_connectivity['partial correlation'])
indices_prec = np.argsort(spreading_prec)

# Relate covariance maximal eigenvalues and spreading
plt.figure()
plt.scatter(max_eigenvalues, spreading_cov)
plt.xlabel('maximal eigenvalue')
plt.ylabel('spreading')

# Relate covariance minimal eigenvalues and spreading
plt.figure()
plt.scatter(min_eigenvalues, spreading_prec)
plt.xlabel('minimal eigenvalue')
plt.ylabel('spreading')

# Plot individual connectivities
from nilearn import plotting

edge_threshold = 0.35
labels, region_coords = zip(*dataset.rois)
node_color = ['g' if label in DMN else 'm' for label in labels]
display_mode = 'z'

# Plot mean correlation
plotting.plot_connectome(
    mean_connectivity['correlation'],
    region_coords,
    node_color=node_color,
    edge_threshold=edge_threshold,
    display_mode=display_mode, title='mean')
#plt.savefig('/home/sb238920/CODE/salma/figures/mean_corr_rs1.pdf')
spreading_geo = analyzing.compute_geo_spreading(subjects_connectivity['covariance'])
indices_geo = np.argsort(spreading_geo)


# Plot individual correlation
edge_threshold = .7
n_inliers = 0
measure = 'correlation'
# Typical subjects for all the measures
typical_indices = []
for n in indices_eig[:20]:
    if (n in indices_cov[:20]) and (n in indices_geo[:20]):
        typical_indices.append(n)

zero_diag = True
plot_matrices(subjects_connectivity[measure][indices_eig, ...][:6],
              zero_diag=zero_diag,
              titles=['typ {0}'.format('eig') for n in range(6)])
plot_matrices(subjects_connectivity[measure][indices_geo, ...][:6],
              zero_diag=zero_diag,
              titles=['typ {0}'.format('geo') for n in range(6)])
plot_matrices(subjects_connectivity[measure][indices_cov, ...][:6],
              zero_diag=zero_diag,
              titles=['typ {0}'.format('corr') for n in range(6)])

# The outliers
for indices, title in zip([indices_eig, indices_geo, indices_cov],
                          ['eig', 'geo', 'cov']):
    plot_matrices(subjects_connectivity[measure][indices, ...][34:39],
                  titles=['out {0}'.format(title) for n in range(6)],
                  zero_diag=zero_diag)

for n in range(n_inliers):
    plotting.plot_connectome(
        subjects_connectivity['correlation'][indices_eig, ...][n],
        region_coords,
        node_color=node_color,
        edge_threshold=edge_threshold,
        display_mode=display_mode, title='typ')
#    plt.savefig('/home/sb238920/CODE/salma/figures/typical_corr_{}_rs1.pdf'.format(n))

for n in range(1):
#    plot_matrix(subjects_connectivity['correlation'][indices_eig, ...][39 - n])
    assert_is_outlier(subjects_connectivity['covariance'][indices_eig, ...][39 - n],
                      subjects_connectivity['covariance'][indices_eig,...][:n_inliers])    
    plotting.plot_connectome(
        subjects_connectivity['correlation'][indices_eig, ...][39 - n],
        region_coords,
        node_color=node_color,
        edge_threshold=edge_threshold,
        display_mode=display_mode, title='out')
#    plt.savefig('/home/sb238920/CODE/salma/figures/outlier_corr_{}_rs1.pdf'.format(n))
plt.show()
########################################
# covariance
#######################################
edge_threshold = '0%'
labels, region_coords = zip(*dataset.rois)
node_color = ['g' if label in DMN else 'm' for label in labels]
display_mode = 'z'

# Plot mean covariance
plotting.plot_connectome(
    mean_connectivity['covariance'],
    region_coords,
    node_color=node_color,
    edge_threshold=edge_threshold,
    display_mode=display_mode)
plt.savefig('/home/sb238920/CODE/salma/figures/mean_cov_rs1.pdf')

# Plot individual covariance
for n in range(6):
    plotting.plot_connectome(
        subjects_connectivity['covariance'][indices_eig, ...][n],
        region_coords,
        node_color=node_color,
        edge_threshold=edge_threshold,
        display_mode=display_mode)
    plt.savefig('/home/sb238920/CODE/salma/figures/typical_cov_{}_rs1.pdf'.format(n))

for n in range(3):
    plotting.plot_connectome(
        subjects_connectivity['covariance'][indices_eig, ...][39 - n],
        region_coords,
        node_color=node_color,
        edge_threshold=edge_threshold,
        display_mode=display_mode)
    plt.savefig('/home/sb238920/CODE/salma/figures/outlier_cov_{}_rs1.pdf'.format(n))
plt.show()
#######################################
if False:
    edge_threshold = .25
    for n in range(6):
        plotting.plot_connectome(
            subjects_connectivity['partial correlation'][indices_eig_min, ...][n],
            region_coords,
            node_color=node_color,
            edge_threshold=edge_threshold,
            display_mode=display_mode,
            title='typical partials, subject {}'.format(indices_eig_min[n]))
        plt.savefig('/home/sb238920/CODE/salma/figures/typical_part_{}_rs1.pdf'.format(n))
    
    for n in range(3):
        plotting.plot_connectome(
            subjects_connectivity['partial correlation'][indices_eig_min, ...][39 - n],
            region_coords,
            node_color=node_color,
            edge_threshold=edge_threshold,
            display_mode=display_mode,
            title='outliers partials, subject {}'.format(indices_eig_min[39 - n]))
        plt.savefig('/home/sb238920/CODE/salma/figures/outlier_part_{}_rs1.pdf'.format(n))
    plt.show()

for n in range(0):
    plotting.plot_connectome(
        subjects_connectivity['correlation'][indices_eig, ...][n],
        region_coords,
        edge_threshold=.2,
        title='typical corr max eig {}'.format(n))
    plotting.plot_connectome(
        subjects_connectivity['correlation'][indices_eig, ...][39 - n],
        region_coords,
        edge_threshold=.2,
        title='outliers corr max eig {}'.format(n))
    plotting.plot_connectome(
        subjects_connectivity['partial correlation'][indices_eig, ...][n],
        region_coords,
        edge_threshold=.2,
        title='typical part max eig {}'.format(n))

    plotting.plot_connectome(
        subjects_connectivity['partial correlation'][indices_eig, ...][39 - n],
        region_coords,
        edge_threshold=.2,
        title='outliers part max eigen {}'.format(n))
plt.show()

# Compare typical covariances to gmean and amean
edge_threshold = .025
for n in range(0):
    plotting.plot_connectome(
        subjects_connectivity['covariance'][indices_eig, ...][n],
        region_coords,
        edge_threshold=edge_threshold,
        display_mode=display_mode,
        title='typical covariance {}'.format(n))
for measure in ['robust dispersion', 'covariance']:
    plotting.plot_connectome(
        mean_connectivity[measure],
        region_coords,
        edge_threshold=edge_threshold,
        display_mode=display_mode)
plt.show()

