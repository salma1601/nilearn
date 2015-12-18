"""Compares the connectivity matrices of the two subsets.
"""
import numpy as np
import matplotlib.pylab as plt

from funtk.connectivity import matrix_stats

import dataset_loader

# Load CPAC atlas
cpac_atlas = dataset_loader.load_ho_cpac(
    '/home/sb238920/CODE/salma/abide_data')
region_labels, region_coords = cpac_atlas.labels, cpac_atlas.coords

# Load preprocessed abide timeseries extracted from harvard oxford atlas
from nilearn import datasets
time_series = {}
time_series['control'] = datasets.fetch_abide_pcp(derivatives=['rois_ho'],
                                                  DX_GROUP=2).rois_ho
time_series['autist'] = datasets.fetch_abide_pcp(derivatives=['rois_ho'],
                                                 DX_GROUP=1).rois_ho
conditions = ['control', 'autist']


# Estimate connectivity matrices
import nilearn.connectivity
n_controls = len(time_series['control'])
n_autists = len(time_series['autist'])
mean_matrices = []
all_matrices = []
measures = ["correlation", "partial correlation", "robust dispersion"]
subjects_unscaled = [subj for condition in conditions for subj in
                     time_series[condition]]
# Standardize the signals
scaling_type = 'unnormalized'
from nilearn import signal
if scaling_type == 'normalized':
    subjects = []
    for subject in subjects_unscaled:
        subjects.append(signal._standardize(subject))
else:
    subjects = subjects_unscaled

# Estimate connectivity
subjects = np.array(subjects)
subjects_connectivity = {}
mean_connectivity = {}
from joblib import Memory
mem = Memory('/home/sb238920/CODE/Parietal/nilearn/nilearn_cache/abide')
from sklearn.covariance import LedoitWolf
for measure in measures:
    cov_embedding = nilearn.connectivity.ConnectivityMeasure(
        kind=measure, cov_estimator=LedoitWolf())
    subjects_connectivity[measure] = cov_embedding.fit_transform(subjects,
                                                                 mem=mem)
    # Compute the mean connectivity across all subjects
    if measure == 'robust dispersion':
        mean_connectivity[measure] = cov_embedding.robust_mean_
    else:
        mean_connectivity[measure] = \
            subjects_connectivity[measure].mean(axis=0)

# Statitistical tests between the conditions
from sklearn.covariance import EmpiricalCovariance
from nilearn.connectivity.connectivity_matrices import sym_to_vec
from nilearn import plotting
matrices_to_plot = {}
comparison_pvals = {}
pval_to_plot = {}
for measure in measures:  # + ['mix']
    print measure
    if measure in measures:
        matrices = subjects_connectivity[measure]
    else:
        corrs = subjects_connectivity['correlation']
        partials = subjects_connectivity['partial correlation']
        matrices = 0.573497578635 * (corrs - partials) ** 2 +\
            0.471676957796 * (corrs - partials) + partials

    baseline = matrices[:n_controls]
    follow_up = matrices[n_controls:]
    if measure in ['correlation', 'partial correlation', 'mix']:
        # Z-Fisher transform if needed
        baseline_n = matrix_stats.corr_to_Z(baseline)
        follow_up_n = matrix_stats.corr_to_Z(follow_up)
        conjunction = True
#        baseline_n[np.isnan(baseline_n)] = 1.
#        follow_up_n[np.isnan(follow_up_n)] = 1.
        n_tests = baseline.shape[-1]
        threshold = .05 / (n_tests * (n_tests - 1.) / 2.)
    elif measure == 'robust dispersion':
        baseline_n = baseline
        follow_up_n = follow_up
        conjunction = False
        n_tests = baseline.shape[-1]
        threshold = .05 / (n_tests * (n_tests + 1.) / 2.)
#        cov_estimator = LedoitWolf(store_precision=False)
#        vec = sym_to_vec(matrices)
#        cov_estimator.fit(vec)
#        whitener = nilearn.connectivity.connectivity_matrices._map_eigenvalues(
#            lambda x: 1. / np.sqrt(x), cov_estimator.covariance_)
#        baseline_n = np.array([whitener.dot(sym_to_vec(matrix)).dot(whitener)
#                               for matrix in baseline])
#        follow_up_n = np.array([whitener.dot(sym_to_vec(matrix)).dot(whitener)
#                               for matrix in follow_up])
#        matrix_stats.fill_nd_diagonal(baseline_n, np.nan)
#        matrix_stats.fill_nd_diagonal(follow_up_n, np.nan)
    threshold = 0.05
    paired = False
    corrected = True
    conjunction = False
    comparison_pvals[measure] = matrix_stats.get_pvalues(
        baseline_n, follow_up_n, axis=0, paired=paired, threshold=threshold,
        corrected=corrected, conjunction=conjunction)
    pval_b, pval_f, pval_diff = comparison_pvals[measure]
    mask_b, mask_f, mask_diff = pval_b < threshold, pval_f < threshold,\
        pval_diff < threshold

    if measure == 'robust dispersion':
        cov_embed_baseline = nilearn.connectivity.ConnectivityMeasure(
            kind='robust dispersion', cov_estimator=LedoitWolf())
        cov_embed_follow_up = nilearn.connectivity.ConnectivityMeasure(
            kind='robust dispersion', cov_estimator=LedoitWolf())
        cov_embed_baseline.fit_transform(subjects[:n_controls, ...])
        cov_embed_follow_up.fit_transform(subjects[n_controls:, ...])
        effects = [cov_embed_baseline.robust_mean_,
                   cov_embed_follow_up.robust_mean_,
                   follow_up.mean(axis=0) - baseline.mean(axis=0)]
        mask_b = np.ones(shape=cov_embed_baseline.robust_mean_.shape,
                         dtype=bool)
        mask_f = np.ones(shape=cov_embed_follow_up.robust_mean_.shape,
                         dtype=bool)
    else:
        effects = [baseline.mean(axis=0),
                   follow_up.mean(axis=0),
                   follow_up.mean(axis=0) - baseline.mean(axis=0)]

    matrices_to_plot[measure] = []
    for mean_matrix, mask in zip(effects,
                                 [mask_b, mask_f, mask_diff]):
        mean_matrix[np.logical_not(mask)] = 0.
        matrices_to_plot[measure].append(mean_matrix)
    print 'plotting matrices'
    n_regions_per_ntwk = [6, 5, 5, 6, 12, 6, 6]
    lines=np.cumsum(n_regions_per_ntwk)[:-1]
    matrix_stats.plot_matrices(matrices_to_plot[measure],
                               titles=[measure + ' ' + conditions[0],
                                       measure + ' ' + conditions[1],
                                       conditions[1] + ' - ' + conditions[0]],
                               tick_labels=[],
                               lines=lines,
                               zero_diag=True, font_size=8)

    # Plot the mean difference in connectivity
    print 'plotting connectome'
    node_color = ['k' if label == 'Right Thalamus' else 'y' if label ==
                  'Insular Cortex, left part' else 'g' if 'Caudate' in label
                  else 'r' for label in region_labels]
    # TODO: debug
    pval_to_plot[measure] = - np.log10(pval_diff) *\
        np.sign(matrices_to_plot[measure][2])
    pval_to_plot[measure][np.logical_not(mask_diff)] = 0.
    pval_to_plot[measure] = (pval_to_plot[measure] +
                             pval_to_plot[measure].T) / 2  # force symetry
#    lower_triangular_adjacency_matrix = np.tril(symmetric_mean_matrix, k=-1)
#    non_zero_indices = lower_triangular_adjacency_matrix.nonzero()
#    edge_color = np.array(['r' for label in region_labels])
#    line_colors = [edge_color[list(index)]
#                   for index in zip(*non_zero_indices)]
    plotting.plot_connectome(pval_to_plot[measure],
                             region_coords,
                             edge_threshold='0%',
                             node_color=node_color,
                             title='mean %s difference' % measure)
plt.show()

# Print significant regions labels
pairs = {}
for measure in measures:
    print measure
    significant_indices = np.where(matrices_to_plot[measure][2] != 0.)
    pairs[measure] = ['(' + region_labels[idx] + '|' + region_labels[idy] + ')'
        for (idx, idy) in zip(*np.triu_indices(matrices.shape[-1], k=1))
        if matrices_to_plot[measure][2][idx, idy] != 0]
    plt.figure()
    plt.plot(significant_indices[0], significant_indices[1], '.')
    plt.xticks(range(111))
    plt.title(measure)
plt.show()
################
# Overlay graphs
################
# diplay1: Which edges are captured by robust dispersion
# display2: Are there new edges in robust dispersion
for measure in ['correlation', 'partial correlation', 'robust dispersion']:
    if measure != 'robust dispersion':
        intersection = np.logical_and(
            pval_to_plot['robust dispersion'] != 0,
            pval_to_plot[measure] != 0)
    else:
        intersection1 = np.logical_and(
            pval_to_plot['robust dispersion'] != 0,
            pval_to_plot['correlation'] != 0)
        intersection2 = np.logical_and(
            pval_to_plot['robust dispersion'] != 0,
            pval_to_plot['partial correlation'] != 0)
        intersection3 = np.logical_and(
            pval_to_plot['robust dispersion'] != 0,
            pval_to_plot['partial correlation'] != 0,
            pval_to_plot['correlation'] != 0)
    display = plotting.plot_connectome(pval_to_plot[measure],
                                       region_coords,
                                       edge_threshold='0%',
                                       node_color=node_color,
                                       title='mean %s difference' % measure)
    if measure != 'robust dispersion':
        matrix = pval_to_plot[measure].copy()
        matrix[np.logical_not(intersection)] = 0.
        display.add_graph(matrix,
                          region_coords,
                          edge_threshold='0%',
                          node_color=node_color,
                          edge_kwargs={'color': 'm'})
    else:
        for intersection, color in zip([intersection1, intersection2,
                                        intersection3], 'myk'):
            matrix = pval_to_plot[measure].copy()
            matrix[np.logical_not(intersection)] = 0.
            display.add_graph(matrix,
                              region_coords,
                              edge_threshold='0%',
                              node_color=node_color,
                              edge_kwargs={'color': color})
    display.savefig('/tmp/{}_intersection.pdf'.format(measure))
    display.close()
plotting.show()
