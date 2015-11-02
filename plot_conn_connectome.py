"""Plots mean connectivity matrices and graphs for different measures.
"""
import numpy as np
import matplotlib.pylab as plt

#from nilearn.connectivity.embedding import cov_to_corr, prec_to_partial
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

conditions = ['ReSt1_Placebo']
dataset = dataset_loader.load_conn(conn_folder_no_filt, conditions=conditions,
                                   standardize=False,
                                   networks=networks)

# Estimate connectivity matrices
from sklearn.covariance import EmpiricalCovariance
import nilearn.connectivity
measures = ["correlation", "partial correlation", "robust dispersion"]
subjects = np.array(dataset.time_series[conditions[0]])
subjects_connectivity = {}
mean_connectivity = {}
n_subjects = 40

for measure in measures:
    cov_embedding = nilearn.connectivity.ConnectivityMeasure(
        kind=measure, cov_estimator=EmpiricalCovariance())
    subjects_connectivity[measure] = cov_embedding.fit_transform(subjects)
    if measure == "robust dispersion":
        mean_connectivity[measure] = cov_embedding.robust_mean_
    else:
        mean_connectivity[measure] = np.mean(subjects_connectivity[measure],
                                             axis=0)

# Plot the mean difference in connectivity
import nilearn.plotting
labels, region_coords = zip(*dataset.rois)

node_color = ['g' if label in DMN else 'k' if label in WMN else 'm' for label
              in labels]
edge_threshold = '95%'
for measure in measures:
    if measure == 'robust dispersion':
        title = 'geometric covariances mean'
    else:
        title = 'mean ' + measure
    nilearn.plotting.plot_connectome(mean_connectivity[measure], region_coords,
                                     edge_threshold=edge_threshold,
                                     node_color=node_color,
                                     title=title)
    name = measure.replace(' ', '_')
    thereshold = edge_threshold.replace('%', '_percent')
    if measure == 'robust dispersion':
        name = 'geometric'
    plt.savefig('/home/sb238920/CODE/salma/figures/mean_{0}_th_{1}.pdf'
                .format(name, thereshold))

from funtk.connectivity import matrix_stats
titles = ['geometric covariances mean' if measure == 'robust dispersion'
          else measure for measure in measures]
matrix_stats.plot_matrices(
    [mean_connectivity[measure] for measure in measures],
    titles=titles, zero_diag=True)

nilearn.plotting.plot_connectome(
    matrix_stats.cov_to_corr(mean_connectivity['robust dispersion']) -
    mean_connectivity['partial correlation'],
    region_coords,
    edge_threshold='99%',
    node_color=node_color,
    title='mean difference %s' % measure)
plt.show()