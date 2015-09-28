import numpy as np
import matplotlib.pylab as plt

from nilearn.connectivity2.collecting import single_glob
from funtk.connectivity import matrix_stats
import dataset_loader


# Specify the networks
WMN = [' L IPL', ' L MFG', ' L CPL', ' R CPL1', ' L Th']
AN = [' L vIPS', ' R vIPS', ' R TPJ', ' R DLPFC', ' L pIPS', ' R pIPS',
      ' L MT', ' R MT', ' L FEF', ' R FEF']
DMN = [' L AG', ' R AG', ' L SFG', ' R SFG', ' PCC', ' MPFC', ' FP']
networks = [('WMN', WMN), ('AN', AN), ('DMN', DMN)]
#networks = None
networks = [('WMN', WMN), ('AN', AN)]

# Specify the location of the files and their patterns
anonymisation_files = ['/home/sb238920/CODE/anonymisation/nilearn_paths.txt',
                       '/home/sb238920/CODE/anonymisation/nilearn_backuped_paths.txt']
datasets = []
condition = 'rs2'
data_labels = ['accepted', 'rejected']
datasets = {}
subjects_ids = {}
for anonymisation_file, label in zip(anonymisation_files, data_labels):
    [timeseries_folder, motion_folder, ids_file] = np.genfromtxt(
        anonymisation_file, dtype=str, skip_header=True, delimiter=" ")
    ids = np.genfromtxt(ids_file, dtype=str, delimiter=':', usecols=(0, 1))
    timeseries_pattern = [(sub_id, session_id) for [sub_id, session_id] in ids]
    subjects_ids[label] = zip(*timeseries_pattern)[0]
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

    datasets[label] = dataset_loader.load_nilearn(
        timeseries_folder, motion_folder, timeseries_pattern, motion_patterns,
        conditions=[condition], standardize=False, networks=networks)

# Estimate connectivity matrices
from sklearn.covariance import EmpiricalCovariance
import nilearn.connectivity
n_subjects = 40
mean_matrices = []
all_matrices = []
measures = ["tangent", "correlation", "partial correlation", "covariance",
            "precision"]
base_subjects = [subj_ts for subj_ts, sub_id in
    zip(datasets['accepted'].time_series[condition], subjects_ids['accepted'])
    if sub_id not in subjects_ids['rejected']]
subjects = [subj for subj in datasets['accepted'].time_series[condition]] +\
    base_subjects + [subj for subj in datasets['rejected'].time_series[condition]]

subjects_connectivity = {}
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
        baseline_n, follow_up_n, axis=0, paired=True, threshold=.05,
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
    regions_labels = zip(*datasets['accepted'].rois)[0]
    matrix_stats.plot_matrices(matrices,
                               titles=[measure + ' accepted',
                                       condition + ' rejected',
                                       'rejected - accepted'],
                               tick_labels=regions_labels,
                               lines=np.cumsum(n_regions_per_ntwk)[:-1],
                               zero_diag=False)

plt.title(measure)
plt.show()
