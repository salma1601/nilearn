"""Splits the dataset into high moving and low moving subjects and compares the
connectivity matrices of the two subsets.
"""
import numpy as np
import matplotlib.pylab as plt

from nilearn.connectivity2 import analyzing
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

condition = 'ReSt1_Placebo'
dataset = dataset_loader.load_conn(conn_folder_no_filt, conditions=[condition],
                                   standardize=False,
                                   networks=networks)

# Estimate connectivity matrices
from sklearn.covariance import EmpiricalCovariance
import nilearn.connectivity
measures = ["robust dispersion", "correlation", "partial correlation"]

# Split the subjects with respect to median relative motion
displacement = np.diff(dataset.motion[condition], axis=1)
if False:
    median_abs_displacement = np.median(np.abs(displacement), axis=1)
    low_motion_indices, high_motion_indices = analyzing.split_sample(
        median_abs_displacement[:, :3], 'norm 2')
norm_displacement = np.linalg.norm(displacement[..., :3], axis=-1)
motion = np.max(norm_displacement, axis=1)
median_motion = np.median(motion)
low_motion_indices = np.where(motion < median_motion)
high_motion_indices = np.where(motion >= median_motion)
timeseries = np.array(dataset.time_series[condition])
subjects = np.concatenate((timeseries[low_motion_indices],
                           timeseries[high_motion_indices]), axis=0)

subjects_connectivity = {}
n_subjects = 20
for measure in measures:
    cov_embedding = nilearn.connectivity.ConnectivityMeasure(
        kind=measure, cov_estimator=EmpiricalCovariance())
    subjects_connectivity[measure] = cov_embedding.fit_transform(subjects)

# Statitistical tests between included highly moving or not
for measure in ['partial correlation',
                'robust dispersion', 'correlation']:
    matrices = subjects_connectivity[measure]
    baseline = matrices[:n_subjects]
    follow_up = matrices[n_subjects:]
    if measure in ['correlation', 'partial correlation']:
        # Z-Fisher transform if needed
        baseline_n = matrix_stats.corr_to_Z(baseline)
        follow_up_n = matrix_stats.corr_to_Z(follow_up)
        baseline_n[np.isnan(baseline_n)] = 1.
        follow_up_n[np.isnan(follow_up_n)] = 1.
    else:
        baseline_n = baseline
        follow_up_n = follow_up

    mask_b, mask_f, mask_diff = matrix_stats.significance_masks(
        baseline_n, follow_up_n, axis=0, paired=False, threshold=.05,
        corrected=True, conjunction=False)
    if measure == 'robust dispersion':
        cov_embed_baseline = nilearn.connectivity.ConnectivityMeasure(
            kind='robust dispersion', cov_estimator=EmpiricalCovariance())
        cov_embed_follow_up = nilearn.connectivity.ConnectivityMeasure(
            kind='robust dispersion', cov_estimator=EmpiricalCovariance())
        cov_embed_baseline.fit_transform(subjects[:n_subjects, ...])
        cov_embed_follow_up.fit_transform(subjects[n_subjects:, ...])
        effects = [cov_embed_baseline.robust_mean_,
                   cov_embed_follow_up.robust_mean_,
                   cov_embed_follow_up.robust_mean_ -
                   cov_embed_baseline.robust_mean_]
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
    if measure == 'robust dispersion':
        cb_label = 'geometric covariance'
        cb_label_difference = 'deviation'
    else:
        cb_label = measure
        cb_label_difference = 'difference'
    figure = plt.figure(figsize=(14, 7))
    matrix_stats.plot_matrices(matrices,
                           titles=['low motion',
                                   'high motion',
                                   'high - low'],                          
                           tick_labels=tick_labels,
                           lines=np.cumsum(n_regions_per_ntwk)[:-1],
                           zero_diag=True, font_size=6,
                           cb_labels=[cb_label, cb_label, cb_label_difference])
    import os
    from matplotlib import pylab
    if measure == 'robust dispersion':
        name = 'geometric'
    elif measure == 'partial correlation':
        name = 'partial'
    else:
        name = measure
    filename = os.path.join('/home/sb238920/CODE/salma/figures',
                            'motion_impact_{}_matrix_rs1.pdf'.format(name))
#    pylab.savefig(filename)
    os.system("pdfcrop %s %s" % (filename, filename))

    # Plot the mean difference in connectivity
    symmetric_mean_matrix = (mean_matrix + mean_matrix.T) / 2.  # force symetry
    import nilearn.plotting
    labels, region_coords = zip(*dataset.rois)
    node_color = ['g' if label in DMN else 'k' if label in WMN else 'm' for
                  label in labels]
    nilearn.plotting.plot_connectome(symmetric_mean_matrix, region_coords,
                                     edge_threshold=0,
                                     node_color=node_color,
                                     title='mean %s difference' % measure)
    filename = os.path.join('/home/sb238920/CODE/salma/figures',
                            'motion_impact_{}_graph_rs1.pdf'.format(name))
#    pylab.savefig(filename)
plt.show()
