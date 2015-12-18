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
                                   standardize=True,
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
for measure in measures:
    matrices = subjects_connectivity[measure]
    baseline = matrices[:n_subjects]
    follow_up = matrices[n_subjects:]
    if measure in ['correlation', 'partial correlation']:
        # Z-Fisher transform if needed
        baseline_n = matrix_stats.corr_to_Z(baseline)
        follow_up_n = matrix_stats.corr_to_Z(follow_up)
#        baseline_n[np.isnan(baseline_n)] = 1.
#        follow_up_n[np.isnan(follow_up_n)] = 1.
        conjunction = True
        n_tests = baseline.shape[-1]
        threshold = .01 / (n_tests * (n_tests - 1.) / 2.)
    else:
        baseline_n = baseline
        follow_up_n = follow_up
        conjunction = False
        threshold = .05 / (n_tests * (n_tests + 1.) / 2.)

    mask_b, mask_f, mask_diff = matrix_stats.significance_masks(
        baseline_n, follow_up_n, axis=0, paired=True, threshold=threshold,
        corrected=False, conjunction=False)
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

    # Plot the mean difference in connectivity
    symmetric_mean_matrix = (mean_matrix + mean_matrix.T) / 2.  # force symetry
    import nilearn.plotting
    labels, region_coords = zip(*dataset.rois)
    node_color = ['g' if label in DMN else 'y' if label in WMN else 'm' for
                  label in labels]
    nilearn.plotting.plot_connectome(symmetric_mean_matrix, region_coords,
                                     edge_threshold='0%',
                                     node_color=node_color,
                                     title='mean %s difference' % measure)
plt.show()
