"""Splits the dataset into high moving and low moving subjects and compares the
connectivity matrices of the two subsets.
"""
import numpy as np
import matplotlib.pylab as plt

from nilearn.connectivity2.collecting import single_glob
from nilearn.connectivity2 import analyzing

from funtk.connectivity import matrix_stats
import dataset_loader


# Specify the networks
WMN = [' L IPL', ' L MFG', ' L CPL', ' R CPL1', ' L Th']
AN = [' L vIPS', ' R vIPS', ' R TPJ', ' R DLPFC', ' L pIPS', ' R pIPS',
      ' L MT', ' R MT', ' L FEF', ' R FEF']
DMN = [' L AG', ' R AG', ' L SFG', ' R SFG', ' PCC', ' MPFC', ' FP']
networks = [('WMN', WMN), ('AN', AN), ('DMN', DMN)]
networks = None

# Specify the location of the files and their patterns
anonymisation_file = '/home/sb238920/CODE/anonymisation/nilearn_paths.txt'
condition = 'rs1'
[timeseries_folder, motion_folder, ids_file] = np.genfromtxt(
    anonymisation_file, dtype=str, skip_header=True, delimiter=" ")
ids = np.genfromtxt(ids_file, dtype=str, delimiter=':', usecols=(0, 1))
timeseries_pattern = [(sub_id, session_id) for [sub_id, session_id] in ids]
import re
motion_patterns = [(sub_id.lower(), re.search(r'\d+', session_id).group())
                   for [sub_id, session_id] in ids]

# Deal with single number acquisitions
import os
for n, (sub_id, session_number) in enumerate(motion_patterns):
    motion_path = os.path.join(motion_folder, 'rp_ars1_' + sub_id +
                               '*acq' + session_number + '*.txt')
    try:
        _ = single_glob(motion_path)
    except:
        motion_patterns[n] = (sub_id, session_number[0])

dataset = dataset_loader.load_nilearn(
    timeseries_folder, motion_folder, timeseries_pattern, motion_patterns,
    conditions=[condition], standardize=False, networks=networks)

# Estimate connectivity matrices
from sklearn.covariance import EmpiricalCovariance
import nilearn.connectivity
measures = ["tangent", "correlation", "partial correlation", "covariance",
            "precision"]

# Split the subjects with respect to median relative motion
displacement = np.diff(dataset.motion[condition], axis=1)
median_abs_displacement = np.median(np.abs(displacement), axis=1)
low_motion_indices, high_motion_indices = analyzing.split_sample(
    median_abs_displacement[:, :3], 'norm 2')
timeseries = np.array(dataset.time_series[condition])
subjects = np.concatenate((timeseries[low_motion_indices],
                           timeseries[high_motion_indices]), axis=0)

subjects_connectivity = {}
n_subjects = 20
for measure in measures:
    cov_embedding = nilearn.connectivity.CovEmbedding(
        kind=measure, cov_estimator=EmpiricalCovariance())
    subjects_connectivity[measure] = nilearn.connectivity.vec_to_sym(
        cov_embedding.fit_transform(subjects))

# Statitistical tests between included highly moving or not
for measure in measures:
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
        corrected=False)
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
    matrix_stats.plot_matrices(matrices,
                               titles=[measure + ' accepted',
                                       condition + ' rejected',
                                       'rejected - accepted'],
                               tick_labels=regions_labels,
                               lines=np.cumsum(n_regions_per_ntwk)[:-1],
                               zero_diag=False)

plt.title(measure)
plt.show()
