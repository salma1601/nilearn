"""Compares the connectivity matrices of the two subsets.
"""
import numpy as np
import matplotlib.pylab as plt


from funtk.connectivity import matrix_stats
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
measures = ["correlation", "robust dispersion", "partial correlation"]
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

subjects = np.array(subjects)
subjects_connectivity = {}
mean_connectivity = {}
from joblib import Memory
mem = Memory('/home/sb238920/CODE/Parietal/nilearn/nilearn_cache/abide')
from sklearn.covariance import LedoitWolf
for measure in ['correlation', 'partial correlation', 'robust dispersion']:
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
for measure in measures:
    matrices = subjects_connectivity[measure]
    baseline = matrices[:n_controls]
    follow_up = matrices[n_controls:]
    if measure in ['correlation', 'partial correlation']:
        # Z-Fisher transform if needed
        baseline_n = matrix_stats.corr_to_Z(baseline)
        follow_up_n = matrix_stats.corr_to_Z(follow_up)
#        baseline_n[np.isnan(baseline_n)] = 1.
#        follow_up_n[np.isnan(follow_up_n)] = 1.
    else:
        baseline_n = baseline
        follow_up_n = follow_up
        matrix_stats.fill_nd_diagonal(baseline_n, np.nan)
        matrix_stats.fill_nd_diagonal(follow_up_n, np.nan)

    mask_b, mask_f, mask_diff = matrix_stats.significance_masks(
        baseline_n, follow_up_n, axis=0, paired=False, threshold=.05,
        corrected=True)
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

    matrices = []
    for mean_matrix, mask in zip(effects,
                                 [mask_b, mask_f, mask_diff]):
        mean_matrix[np.logical_not(mask)] = 0.
        matrices.append(mean_matrix)
    matrix_stats.plot_matrices(matrices,
                               titles=[measure + ' ' + conditions[0],
                                       measure + ' ' + conditions[1],
                                       conditions[0] + ' - ' + conditions[1]],
                               tick_labels=[],
                               lines=[],
                               zero_diag=True, font_size=8)
plt.show()
