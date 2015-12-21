"""Compares the connectivity matrices of the two subsets.
"""
import numpy as np
import matplotlib.pylab as plt


from funtk.connectivity import matrix_stats
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

conditions = ['ReSt1_Placebo', 'Nbac3_Placebo']
dataset = dataset_loader.load_conn(conn_folder_no_filt, conditions=conditions,
                                   standardize=False,
                                   networks=networks)
# Rename the ROIs
regions_labels = zip(*dataset.rois)[0]
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

# Estimate connectivity matrices
from sklearn.covariance import EmpiricalCovariance
import nilearn.connectivity
measures = ["robust dispersion", "correlation", "partial correlation"]
subjects = np.array(dataset.time_series[conditions[0]] +
                    dataset.time_series[conditions[1]])
subjects_connectivity = {}
n_subjects = 40

for measure in measures:
    cov_embedding = nilearn.connectivity.ConnectivityMeasure(
        kind=measure, cov_estimator=EmpiricalCovariance())
    subjects_connectivity[measure] = cov_embedding.fit_transform(subjects)

# Statitistical tests between the conditions
from scipy.linalg import logm
matrices_to_plot = {}
comparison_pvals = {}
pval_to_plot = {}
for measure in measures:
    matrices = subjects_connectivity[measure]
    baseline = matrices[:n_subjects]
    follow_up = matrices[n_subjects:]
    n_tests = baseline.shape[-1]
    if measure in ['correlation', 'partial correlation']:
        # Z-Fisher transform if needed
        baseline_n = matrix_stats.corr_to_Z(baseline)
        follow_up_n = matrix_stats.corr_to_Z(follow_up)
#        baseline_n[np.isnan(baseline_n)] = 1.
#        follow_up_n[np.isnan(follow_up_n)] = 1.
        conjunction = True
        threshold = .01 / (n_tests * (n_tests - 1.) / 2.)
    else:
        baseline_n = baseline
        follow_up_n = follow_up
        conjunction = False
        threshold = .05 / (n_tests * (n_tests + 1.) / 2.)
    corrected = True
    paired = True
    threshold = 0.05
    conjunction = False
    comparison_pvals[measure] = matrix_stats.get_pvalues(
        baseline_n, follow_up_n, axis=0, paired=paired, threshold=threshold,
        corrected=corrected, conjunction=conjunction)
    pval_b, pval_f, pval_diff = comparison_pvals[measure]
    mask_b, mask_f, mask_diff = pval_b < threshold, pval_f < threshold,\
        pval_diff < threshold

    if measure == 'robust dispersion':
        cov_embed_baseline = nilearn.connectivity.ConnectivityMeasure(
            kind='robust dispersion', cov_estimator=EmpiricalCovariance())
        cov_embed_follow_up = nilearn.connectivity.ConnectivityMeasure(
            kind='robust dispersion', cov_estimator=EmpiricalCovariance())
        cov_embed_baseline.fit_transform(subjects[:n_subjects, ...])
        cov_embed_follow_up.fit_transform(subjects[n_subjects:, ...])
        effects = [cov_embed_baseline.robust_mean_,
                   cov_embed_follow_up.robust_mean_,
                   (cov_embed_follow_up.robust_mean_ - 
                    cov_embed_baseline.robust_mean_)]
        if False:
            baseline_logs = np.array([logm(matrix) for matrix in baseline_n])
            follow_up_logs = np.array([logm(matrix) for matrix in follow_up_n])
            mask_b, mask_f, _ = matrix_stats.significance_masks(
                baseline_logs, follow_up_logs, axis=0, paired=True,
                threshold=.05, corrected=True, conjunction=False)
            mask_diff = mask_diff * (mask_b + mask_f)
        else:
            mask_b = np.ones(shape=cov_embed_baseline.robust_mean_.shape,
                             dtype=bool)
            mask_f = np.ones(shape=cov_embed_follow_up.robust_mean_.shape,
                             dtype=bool)
    else:
        effects = [baseline.mean(axis=0),
                   follow_up.mean(axis=0),
                   (follow_up - baseline).mean(axis=0)]

    matrices = []
    for mean_matrix, mask in zip(effects,
                                 [mask_b, mask_f, mask_diff]):
        mean_matrix[np.logical_not(mask)] = 0.
        matrices.append(mean_matrix)
    if networks is None:
        n_regions_per_ntwk = [9, 5, 7, 4, 10, 7, 2]
    else:
        n_regions_per_ntwk = [len(regions) for name, regions in networks]
    
    matrix_stats.plot_matrices(matrices,
                               titles=[measure + ' ' + conditions[0],
                                       measure + ' ' + conditions[1],
                                       conditions[1] + ' - ' + conditions[0]],
                               tick_labels=tick_labels,
                               lines=np.cumsum(n_regions_per_ntwk)[:-1],
                               zero_diag=True, font_size=8)

    pval_to_plot[measure] = - np.log10(pval_diff) #*\
#        np.sign(matrices_to_plot[measure][2])
    pval_to_plot[measure][np.logical_not(mask_diff)] = 0.
    pval_to_plot[measure] = np.minimum(pval_to_plot[measure],
                                       pval_to_plot[measure].T)
    pval_to_plot[measure] = (pval_to_plot[measure] +
                             pval_to_plot[measure].T) / 2  # force symetry
    # Plot the mean difference in connectivity
    import nilearn.plotting
    labels, region_coords = zip(*dataset.rois)
    node_color = ['g' if label in DMN else 'y' if label in WMN else 'm' for
                  label in labels]
    nilearn.plotting.plot_connectome(pval_to_plot[measure], region_coords,
                                     edge_threshold='0%',
                                     node_color=node_color,
                                     title='mean %s difference' % measure)
plt.show()
################
# Overlay graphs
################
# diplay1: Which edges are captured by robust dispersion
# display2: Are there new edges in robust dispersion
from nilearn import plotting
for measure in ['correlation', 'partial correlation', 'robust dispersion']:
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
                                       title='%s difference pvalues' % measure)
    for intersection, color in zip([intersection1, intersection2,
                                    intersection3], 'mgk'):
        matrix = pval_to_plot[measure].copy()
        matrix[np.logical_not(intersection)] = 0.
        display.add_graph(matrix,
                          region_coords,
                          edge_threshold='0%',
                          node_color=node_color,
                          edge_kwargs={'color': color})
    display.savefig('/tmp/{0}-vs-{1}_{2}_intersection.png'.format(
        conditions[0], conditions[1], measure))
    display.close()
plotting.show()
