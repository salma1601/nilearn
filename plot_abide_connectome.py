"""Plots mean connectivity matrices and graphs for different measures.
"""
import numpy as np
import matplotlib.pylab as plt

# Load preprocessed abide timeseries extracted from harvard oxford atlas
from nilearn import datasets
abide = datasets.fetch_abide_pcp(derivatives=['rois_ho'], DX_GROUP=2)
subjects_unscaled = abide.rois_ho

# Standardize the signals
scaling_type = 'normalized'
from nilearn import signal
if scaling_type == 'normalized':
    subjects = []
    for subject in subjects_unscaled:
        subjects.append(signal._standardize(subject))
else:
    subjects = subjects_unscaled

# Estimate connectivity matrices
from sklearn.covariance import LedoitWolf
import nilearn.connectivity
measures = ["correlation", "partial correlation", "robust dispersion"]
subjects_connectivity = {}
mean_connectivity = {}
from joblib import Memory
mem = Memory('/home/sb238920/CODE/Parietal/nilearn/nilearn_cache/abide')
for measure in measures:
    cov_embedding = nilearn.connectivity.ConnectivityMeasure(
        kind=measure, cov_estimator=LedoitWolf())
    subjects_connectivity[measure] = cov_embedding.fit_transform(subjects,
                                                                 mem=mem)
    if measure == "robust dispersion":
        mean_connectivity[measure] = cov_embedding.robust_mean_
    else:
        mean_connectivity[measure] = np.mean(subjects_connectivity[measure],
                                             axis=0)

# Plot the mean difference in connectivity
if False:
    import nilearn.plotting
    regions_labels, region_coords = zip(*dataset.rois)
    
    edge_threshold = '90%'
    for measure in measures:
        if measure == 'robust dispersion':
            title = 'geometric'
        else:
            title = measure
        nilearn.plotting.plot_connectome(mean_connectivity[measure], region_coords,
                                         edge_threshold=edge_threshold,
                                         node_color=node_color,
                                         display_mode='z',
                                         title=title)
        name = measure.replace(' ', '_')
        thereshold = edge_threshold.replace('%', '_percent')
        if measure == 'robust dispersion':
            name = 'geometric'
        figure_name = 'abide_{0}_mean_{1}_th_{2}_rs1.pdf'.format(scaling_type,
                                                                 name,
                                                                 thereshold)
        plt.savefig('/home/sb238920/CODE/salma/figures/' + figure_name)

# Plot mean connectivity matrices
if False:
    if networks is None:
        n_regions_per_ntwk = [9, 5, 7, 4, 10, 7, 2]
    else:
        n_regions_per_ntwk = [len(regions) for name, regions in networks]
    tick_labels = []
    for label in regions_labels:
        if label == 'LMFG_peak1':
            tick_labels.append('L MFG')
        elif label == 'RCPL_peak1':
            tick_labels.append('R CPL')
        elif label == 'LCPL_peak3':
            tick_labels.append('L CPL')
        elif label == 'FEF.cluster001':
            tick_labels.append('R FEF')
        elif label == 'FEF.cluster002':
            tick_labels.append('L FEF')
        elif 'cluster001' in label:
            tick_labels.append('L ' + label.replace('.cluster001', ''))
        elif 'cluster002' in label:
            tick_labels.append('R ' + label.replace('.cluster002', ''))
        else:
            tick_labels.append(label)

from funtk.connectivity import matrix_stats
titles = ['geometric' if measure == 'robust dispersion'
          else measure for measure in measures]
matrix_stats.plot_matrices(
    [mean_connectivity[measure] for measure in measures],
    titles=titles,
#    tick_labels=tick_labels,
#    lines=np.cumsum(n_regions_per_ntwk)[:-1],
    zero_diag=True, font_size=6)
figure_name = 'abide_{0}_mean_matrices_rs1.pdf'.format(scaling_type)
filename = '/home/sb238920/CODE/salma/figures/' + figure_name
from matplotlib import pylab
import os
pylab.savefig(filename)
os.system("pdfcrop %s %s" % (filename, filename))

plt.show()