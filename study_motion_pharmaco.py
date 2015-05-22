import os

import nibabel
import numpy as np
import matplotlib.pylab as plt
from scipy.io import loadmat
from scipy.stats import pearsonr

import nilearn.image
import nilearn.input_data
import nilearn.signal
from pyhrf.retreat import locator
from sklearn.covariance import EmpiricalCovariance, LedoitWolf, MinCovDet
from nilearn import connectivity
from funtk.connectivity import matrix_stats
from nilearn.connectivity.embedding import prec_to_partial
from nilearn.connectivity.collecting import (single_glob, group_left_right,
                                             crescendo_replacement)
from nilearn.connectivity.analyzing import compute_connectivity


def revert_acquisition(name):
    """Reverts acqusition numbers"""
    number = name[-1]
    if number == '1':
        number = '2'
    else:
        number = '1'
    return name[:-1] + number


#####################################################
# Define the ROIs
all_names = []
all_seeds = []
n_regions = []
group = True
for get_networks in [locator.meta_motor, locator.meta_non_cortical,
                     locator.func_working_memory, locator.meta_visual,
                     locator.meta_attention, locator.meta_default_mode,
                     locator.meta_saliency]:
    names, seeds = get_networks()
    if group:
        grouped_names = group_left_right(names, np.empty((1, len(names))))[0]
        n_regions.append(len(grouped_names))
    else:
        n_regions.append(len(names))
    all_names += names
    all_seeds += seeds

###############################################################
# Compare rest to task for low motion
###############################################################
acq_filename = '/home/sb238920/nilearn_data/servier/sessions.txt'
save_folder = 'optimal_hv_confounds'
# Get folders for Placebo acquisitions
subjects_acqs = np.genfromtxt(acq_filename, delimiter=':',
                              dtype='str', usecols=(0, 1))
data_path = '/home/sb238920/nilearn_data/servier'
prefixes = ['rs1', 'nback_2', 'nback_3', 'rs2']
colors = ['r.', 'g.', 'b.', 'm.']
subjects = []
for prefix in prefixes:
    accepted_acqs = [os.path.join(
        data_path,  'low_motion/spheres', save_folder,
        subject_acq[0] + '_' + subject_acq[1] + '_' + prefix + '.npy') for
        subject_acq in subjects_acqs]
    subjects += [np.load(accepted_acq) for accepted_acq in accepted_acqs]

###############################################################
# Quantify the difference between connectivity across measures
###############################################################
measures = ['covariance', 'precision', 'tangent', 'correlation',
            'partial correlation']
estimators = [('ledoit', LedoitWolf()), ('emp', EmpiricalCovariance()),
              ('mcd', MinCovDet())]
n_estimator = 0
n_subjects = len(subjects)
for prefix, color in zip(prefixes, colors):
    accepted_acqs = [os.path.join(
        data_path,  'low_motion/spheres', save_folder,
        subject_acq[0] + '_' + subject_acq[1] + '_' + prefix + '.npy') for
        subject_acq in subjects_acqs]
    subjects0 = [np.load(accepted_acq) for accepted_acq in accepted_acqs]
    subjects = []
    for subject in subjects0:
        subjects.append(group_left_right(all_names, subject)[1])

    all_matrices, mean_matrices = compute_connectivity(subjects)
    plt.plot((mean_matrices[3] - matrix_stats.cov_to_corr(
              mean_matrices[2])).ravel(),
             (mean_matrices[3] - mean_matrices[4]).ravel(), color,
             label=prefix)
    plt.xlabel('corrs - corr(gmean)')
    plt.ylabel('corrs - partials')
    plt.legend()
    plt.title('differences between connectivity measures over regions, '
              '{}'.format(estimators[n_estimator][0]))
    r, pval = pearsonr((mean_matrices[3] -
                        matrix_stats.cov_to_corr(mean_matrices[2])).ravel(),
                       (mean_matrices[3] - mean_matrices[4]).ravel())
    print('pearson corr = {}, pval = {}'.format(r, pval))
plt.show()

#########################################################
# Between conditions comparisons
#########################################################
# Compute connectivities
print("-- Measuring connecivity ...")
mean_scores = []
std_scores = []
prefixes_couples = [('rs1', 'nback_2'), ('rs1', 'nback_3')]
for prefix1, prefix2 in prefixes_couples:
    accepted_acqs1 = [os.path.join(
        data_path,  'low_motion/spheres', save_folder,
        subject_acq[0] + '_' + subject_acq[1] + '_' + prefix1 + '.npy') for
        subject_acq in subjects_acqs]
    accepted_acqs2 = [os.path.join(
        data_path,  'low_motion/spheres', save_folder,
        subject_acq[0] + '_' + subject_acq[1] + '_' + prefix2 + '.npy') for
        subject_acq in subjects_acqs]
    raw_acqs1 = [os.path.join(
        data_path,  'low_motion/spheres/raw',
        subject_acq[0] + '_' + subject_acq[1] + '_' + prefix1 + '.npy') for
        subject_acq in subjects_acqs]
    raw_acqs2 = [os.path.join(
        data_path,  'low_motion/spheres/raw',
        subject_acq[0] + '_' + subject_acq[1] + '_' + prefix2 + '.npy') for
        subject_acq in subjects_acqs]
    subjects0 = [np.load(accepted_acq) for accepted_acq in accepted_acqs1] +\
                [np.load(accepted_acq) for accepted_acq in accepted_acqs2]
    raw_subjects = [np.load(accepted_acq) for accepted_acq in raw_acqs1] +\
                   [np.load(accepted_acq) for accepted_acq in raw_acqs2]
    subjects = []
    for subject, raw_subject in zip(subjects0, raw_subjects):
#        subject -= np.mean(subject, axis=0)  # PSC unit
        subject /= np.mean(raw_subject, axis=0)  # PSC unit
#        subject -= np.mean(raw_subject, axis=0)  # PSC unit
        if group:
            regions = range(33)
        else:
            regions = range(44)
#        regions = [12, 13, 14, 15, 17, 20, 23, 24, 25, 21, 22] + range(26, 31)
#        n_regions = [5, 6, 5]
        if group:
            new_subject = group_left_right(all_names, subject)[1][:, regions]
        else:
            new_subject = subject
        subjects.append(new_subject)

    if group:
        grouped_names = np.array(group_left_right(all_names, subject)[0])
    else:
        grouped_names = np.array(all_names)
    grouped_names = grouped_names[regions]
    n_subjects = len(subjects)
    all_matrices, mean_matrices = compute_connectivity(subjects)

    # Statistical t-tests
    for measure, matrices in zip(measures, all_matrices):
        baseline = matrices[:n_subjects / 2]
        follow_up = matrices[n_subjects / 2:]
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
            corrected=True)
        effects = [baseline.mean(axis=0),
                         follow_up.mean(axis=0),
                         (follow_up - baseline).mean(axis=0)]
        matrices = []
        for mean_matrix, mask in zip(effects,
                                     [mask_b, mask_f, mask_diff]):
            mean_matrix[np.logical_not(mask)] = 0.
            matrices.append(mean_matrix)
        matrix_stats.plot_matrices(matrices,
                                   titles=[measure + ' ' + prefix1,
                                           prefix2,
                                           prefix2 + '-' + prefix1],
                                   tick_labels=grouped_names,
                                   lines=np.cumsum(n_regions)[:-1],
                                   zero_diag=False)
    plt.show()

    # Classify conditions
    from sklearn.svm import LinearSVC
    from sklearn.cross_validation import StratifiedShuffleSplit
    for measure, coefs in zip(measures, all_matrices):
        y = np.hstack((np.zeros(n_subjects / 2), np.ones(n_subjects / 2)))
        # regions of servier report
        regions = range(coefs.shape[-1])
        X = coefs[:, regions, :]
        X = coefs[:, :, regions]
        X = connectivity.sym_to_vec(X)
        print('---Shuffle split, this takes time ...')
        skf = StratifiedShuffleSplit(y, n_iter=10, test_size=0.33)
        clf_svc = LinearSVC(random_state=0)  # TODO: multiclass?
        cv_scores = []
        for train_index, test_index in skf:
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf_svc.fit(X_train, y_train)
            cv_scores.append(clf_svc.score(X_test, y_test))
        mean_scores.append(np.mean(cv_scores) * 100)
        std_scores.append(np.std(cv_scores) * 100)

mean_scores = np.array(mean_scores)
std_scores = np.array(std_scores)
plt.figure()
colors = ['k', 'c', 'g', 'b', 'r']
for n, color in enumerate(colors):
    indices = [n, n + 5]
    plt.bar([n, n + 6], mean_scores[indices],
            color=color, yerr=std_scores[indices], align="center")

# TODO : automatic xticks and xticklabels
plt.xticks([2, 8], [prefixes_couples[0][1] + ' vs ' + prefixes_couples[0][0],
                    prefixes_couples[1][1] + ' vs ' + prefixes_couples[1][0]],
           fontsize='17')
plt.xlim([-1, 11])
plt.ylim([50, 100])
plt.ylabel('Classification accuracy (%)', fontsize='17')
plt.legend(measures, fontsize='17')
plt.show()

########################################################
# Compare high motion to low motion
########################################################
# Load high and low moving subjects
print('---Load highly moving subjects ...')
for prefix in ['rs1', 'rs2']:
    high_moving = []
    low_moving = []
    high_paths = []
    low_paths = []
    accepted_acqs = [os.path.join(
        data_path,  'low_motion/spheres', save_folder,
        subject_acq[0] + '_' + subject_acq[1] + '_' + prefix + '.npy') for
        subject_acq in subjects_acqs]
    rejected_acqs = [os.path.join(
        data_path, 'high_motion/spheres', save_folder, subject_acq[0] + '_' +
        revert_acquisition(subject_acq[1]) + '_' + prefix + '.npy') for
        subject_acq in subjects_acqs]
    low_moving += [path1 for path1, path2 in zip(accepted_acqs, rejected_acqs)
                   if os.path.isfile(path2)]
    high_moving += [path for path in rejected_acqs if os.path.isfile(path)]

    subjects = [np.load(path) for path in low_moving] +\
               [np.load(path) for path in high_moving]

    # Compute mean motion for each subject
    all_low_paths = [path for path in accepted_acqs]
    replacement_tuples = []
    for low_path, high_path in zip(low_moving, high_moving):
        paths = [low_path, high_path]
        mean_translations = []
        mean_rotations = []
        for path in paths:
            filename = os.path.basename(path)
    # TODO: use regexp and form paths once for all
            nip = filename[:8].lower()
            n_acq = filename[20:22]
            motion_filename = single_glob(os.path.join(
                data_path, 'motion_params',
                'rp_a' + prefix + '*' + nip + '*acq' + n_acq + '*.txt'))
            motion_confounds = np.genfromtxt(motion_filename)
            relative_translation = (motion_confounds[:, :3]).copy()
            relative_translation[1:] -= motion_confounds[:, :3][:-1]
            mean_translation = np.linalg.norm(
                relative_translation, axis=1).mean()
            mean_translations.append(mean_translation)
            relative_rotation = (motion_confounds[:, 3:]).copy()
            relative_rotation[1:] -= motion_confounds[:, 3:][:-1]
            mean_rotation = np.linalg.norm(
                relative_rotation, axis=1).mean()
            mean_rotations.append(mean_rotation)
        # Consider as noisy subjects only those that have increased mean
        # translation in the backup session
        print mean_translations    
        if mean_translations[1] - mean_translations[0] > 0.01:
            replacement_tuples.append((low_path, high_path))
        elif mean_translations[0] - mean_translations[1] > 0.01:
            all_low_paths = crescendo_replacement(all_low_paths,
                                                  [(low_path, high_path)])[0]
            replacement_tuples.append((high_path, low_path))

    cresendo_noisy_paths = crescendo_replacement(all_low_paths,
                                                 replacement_tuples)
    cresendo_noisy_paths = [all_low_paths] + cresendo_noisy_paths
# TODO: compare the connectivity with outliers: the norm, and the significant
# connections (hope is adds a small distance connection/ remove high distance)
    subjects = [np.load(path) for path in all_low_paths]
    _, mean_matrices = compute_connectivity(subjects)
    noisy_means = []
    for noisy_paths in cresendo_noisy_paths:
        noisy_subjects = [np.load(path) for path in noisy_paths]
        noisy_means.append(compute_connectivity(noisy_subjects)[1])

    cov_err = []
    robust_cov_err = []
    corr_err = []
    robust_corr_err = []
    prec_err = []
    robust_prec_err = []
    part_err = []
    robust_part_err = []

    def distance(x, y):
        return np.linalg.norm(x - y)

    for mean_matrices2 in noisy_means:
        # Compute the difference between contaminated and non contaminated data
        robust_prec = np.linalg.inv(mean_matrices[2])
        robust_part = prec_to_partial(robust_prec)
        robust_corr = matrix_stats.cov_to_corr(mean_matrices[2])
        cov_err.append(distance(
            mean_matrices[0], mean_matrices2[0]) / np.linalg.norm(mean_matrices[0]))
        robust_cov_err.append(distance(mean_matrices[2],
                                       mean_matrices2[2]) / np.linalg.norm(mean_matrices[2]))
        prec_err.append(distance(mean_matrices[1], mean_matrices2[1]) /
            np.linalg.norm(mean_matrices[1]))
        robust_prec_err.append(distance(robust_prec, np.linalg.inv(
            mean_matrices2[2])) / np.linalg.norm(robust_prec))
        corr_err.append(distance(mean_matrices[3], mean_matrices2[3]) /
            np.linalg.norm(mean_matrices[3]))
        robust_corr_err.append(distance(robust_corr, matrix_stats.cov_to_corr(
            mean_matrices2[2])) / np.linalg.norm(robust_corr))
        part_err.append(distance(mean_matrices[4], mean_matrices2[4]) /
            np.linalg.norm(mean_matrices[4]))
        robust_part_err.append(distance(robust_part, prec_to_partial(
            np.linalg.inv(mean_matrices2[2]))) / np.linalg.norm(robust_part))

    n_noisy = len(noisy_means)
    plt.figure()
    colors = ['r', 'b', 'g', 'k']
    ls = ['-', '--']
    percent_outliers = [100. * n_outliers / n_subjects for n_outliers in
                        range(n_noisy)]
    percent_str = [str(percent) + '%' for percent in percent_outliers]
    lineObjects = []
    for n, error in enumerate([cov_err, robust_cov_err,
                               prec_err, robust_prec_err,
                               corr_err, robust_corr_err,
                               part_err, robust_part_err]):
        lineObjects += plt.plot(percent_outliers, error, colors[n / 2],
                                linestyle=ls[n % 2])
    plt.legend(iter(lineObjects), ('mean covariances', 'gmean',
                                   'mean precisions', r'gmean$ ^{-1}$',
                                   'mean correlations', 'corr(gmean)',
                                   'mean partial correlations', 'partial(gmean)'),
               loc=0)
    plt.title('robustness')
    plt.xlabel('percentage of outliers')
    plt.xticks(percent_outliers, percent_str, size=8)
    plt.ylabel('relative error')

    # correlation, partials corr(gmean) and partial(gmean)
    plt.figure(figsize=(8, 6))
    colors = ['g', 'k']
    ls = ['-', '--']
    percent_outliers = [100. * n_outliers / n_subjects for n_outliers in
                        range(n_noisy)]
    percent_str = [str(percent) + '%' for percent in percent_outliers]
    lineObjects = []
    for n, error in enumerate([corr_err, robust_corr_err,
                               part_err, robust_part_err]):
        lineObjects += plt.plot(percent_outliers, error, colors[n / 2],
                                linestyle=ls[n % 2])
    plt.legend(iter(lineObjects), ('mean correlations', 'corr(gmean)',
                                   'mean partial correlations',
                                   'partial(gmean)'),
               loc=0)
    plt.xlabel('percentage of outliers')
    plt.xticks(percent_outliers, percent_str, size=8)
    plt.ylabel('relative error')
    fig_title = 'froe_relative_err_corr_part_tan' + estimators[n_estimator][0]

    # correlation, partials and corr(gmean)
    plt.figure(figsize=(8, 6))
    colors = ['g', 'k', 'r']
    ls = ['-', '--']
    percent_outliers = [100. * n_outliers / n_subjects for n_outliers in
                        range(n_noisy)]
    percent_str = [str(percent) + '%' for percent in percent_outliers]
    lineObjects = []
    for n, error in enumerate([corr_err, part_err, robust_corr_err]):
        lineObjects += plt.plot(percent_outliers, error, colors[n])
    plt.legend(iter(lineObjects), ('mean correlations',
                                   'mean partial correlations', 'corr(gmean)'),
               loc=0)
    #plt.title('robustness')
    plt.xlabel('percentage of outliers')
    plt.xticks(percent_outliers, percent_str, size=8)
    plt.ylabel('relative error')
    plt.show()


# Link eigenvalues of raw connectivity to motion (max, mean)

# Link eigenvalues of preprocessed connectivity to motion (max, mean) -> motion
# still impacts connectivity even after preprocs

stop
# Compute the displacement of each node and the resulting covariance

########################################################
# Compare high motion to low motion
########################################################
# Load high and low moving subjects
for prefix in ['rs1', 'rs2']:
    high_moving = []
    low_moving = []
    accepted_acqs = [os.path.join(
        data_path,  'low_motion/spheres', save_folder,
        subject_acq[0] + '_' + subject_acq[1] + '_' + prefix + '.npy') for
        subject_acq in subjects_acqs]
    rejected_acqs = [os.path.join(
        data_path, 'high_motion/spheres', save_folder, subject_acq[0] + '_' +
        revert_acquisition(subject_acq[1]) + '_' + prefix + '.npy') for
        subject_acq in subjects_acqs]
    low_moving += [path1 for path1, path2 in zip(accepted_acqs, rejected_acqs)
                   if os.path.isfile(path2)]
    high_moving += [path for path in rejected_acqs if os.path.isfile(path)]

    subjects = [np.load(path) for path in low_moving] +\
               [np.load(path) for path in high_moving]

    # Compute mean motion for each subject
    names = ['low', 'high']
    for name, paths in zip(names, [low_moving, high_moving]):
        mean_motion = []
        for path in paths:
            filename = os.path.basename(path)
    # TODO: use regexp and form paths once for all
            nip = filename[:8].lower()
            n_acq = filename[20:22]
            motion_filename = single_glob(os.path.join(
                data_path, 'motion_params',
                'rp_a' + prefix + '*' + nip + '*acq' + n_acq + '*.txt'))
            motion_confounds = np.genfromtxt(motion_filename)[:, :3]
            relative_motion = motion_confounds.copy()
            relative_motion[1:] -= motion_confounds[:-1]
            mean_motion.append(np.linalg.norm(relative_motion, axis=1).mean())
        if name == 'high':
            high_motion = np.array(mean_motion)
        else:
            low_motion = np.array(mean_motion)

    # Compute connectivity
    all_matrices = []
    mean_matrices = []
    for measure in measures:
        estimator = {'cov_estimator': estimators[n_estimator][1],
                     'kind': measure}
        cov_embedding = connectivity.CovEmbedding(**estimator)
        matrices = connectivity.vec_to_sym(
            cov_embedding.fit_transform(subjects))
        all_matrices.append(matrices)
        if measure == 'tangent':
            mean = cov_embedding.mean_cov_
        else:
            mean = matrices.mean(axis=0)
        mean_matrices.append(mean)

    n_subjects = len(subjects)
    for measure, matrices in zip(measures, all_matrices):
        baseline = matrices[:n_subjects / 2]
        follow_up = matrices[n_subjects / 2:]
        if measure != 'tangent':
            baseline = matrix_stats.corr_to_Z(baseline)
            follow_up = matrix_stats.corr_to_Z(follow_up)

        signif_b, signif_f, signif_diff = matrix_stats.compare(
            baseline, follow_up, 0, True, threshold=.05, corrected=True)
        matrices = [signif_b, signif_f, signif_diff]
        matrix_stats.plot_matrices(
            matrices, titles=[measure + ' ' + prefix + ' ' + names[0],
                              names[1], names[1] + '-' + names[0]],
            tick_labels=all_names, lines=np.cumsum(n_regions)[:-1])
plt.show()

# Compare the error in norm when using highly moving subjects











print("-- Measuring connecivity ...")
measures = ['tangent', 'correlation', 'partial correlation']

estimators = [('ledoit', LedoitWolf()), ('emp', EmpiricalCovariance()),
              ('mcd', MinCovDet())]
n_estimator = 0
all_matrices = []
mean_matrices = []
for measure in measures:
    estimator = {'cov_estimator': estimators[n_estimator][1],
                 'kind': measure}
    cov_embedding = connectivity.CovEmbedding(**estimator)
    matrices = connectivity.vec_to_sym(
        cov_embedding.fit_transform(subjects))
    all_matrices.append(matrices)
    if measure == 'tangent':
        mean = cov_embedding.mean_cov_
    else:
        mean = matrices.mean(axis=0)
    mean_matrices.append(mean)


for measure, matrices in zip(measures, all_matrices):
    baseline = matrices[:40]
    follow_up = matrices[40:]
    if measure != 'tangent':
        baseline = matrix_stats.corr_to_Z(baseline)
        follow_up = matrix_stats.corr_to_Z(follow_up)

    signif_b, signif_f, signif_diff = matrix_stats.compare(baseline, follow_up, 0, True,
                                       threshold=.05, corrected=True)
    matrices = [signif_b, signif_f, signif_diff]
    matrix_stats.plot_matrices(matrices,
                               titles=[measure + ' ' + prefixes[0],
                                       prefixes[1],
                                       prefixes[1] + '-' + prefixes[0]],
                               tick_labels=all_names,
                               lines = np.cumsum(n_regions)[:-1])
plt.show()



radius = 6.
raw_subjects = []
subjects = []
acq_filename = '/neurospin/servier2/salma/nilearn/sessions.txt'

# Get folders for Placebo acquisitions
subjects_acqs = np.genfromtxt(acq_filename, delimiter=':',
                              dtype='str', usecols=(0, 1))
data_path = '/neurospin/servier2/salma'
accepted_acqs = [os.path.join(
    data_path, subject_acq[0], 'fMRI', subject_acq[1]) for subject_acq in
    subjects_acqs]
data_path = '/neurospin/servier2/study'
rejected_acqs = [os.path.join(data_path, subject_acq[0], 'fMRI',
                              revert_acquisition(subject_acq[1]))
                 for subject_acq in subjects_acqs]
rejected_acqs = [folder for folder in rejected_acqs if os.path.isdir(folder)]
# TODO: spatially preprocess AP130327_acquisition11
rejected_acqs.remove('/neurospin/servier2/study/AP130327/fMRI/acquisition11')

# Get the path for the CONN project
conn_dir = os.path.join('/neurospin/servier2/salma/subject1to40',
    'conn_servier2_1to40sub_RS1-Nback2-Nback3-RS2_Pl-D_1_1_1')
raw_dir = os.path.join(conn_dir, 'data')
prefixes = ['rs1', 'nback_2', 'nback_3', 'rs2']

# Define directory to save output signals
out_folder = 'low_motion'
if out_folder == 'low_motion':
    folders = accepted_acqs
else:
    folders = rejected_acqs

subjects = []
nips = []
sessions = []
n_session = 3
prefix = prefixes[n_session]
for n, folder in enumerate(folders):
    func_filename = single_glob(os.path.join(folder,
                                             'swa' + prefix + '*.nii'))
    motion_filename = single_glob(os.path.join(folder,
                                               'rp_a' + prefix + '*.txt'))
    anat_folder = folder.replace('fMRI', 't1mri')
    gm_filename = single_glob(os.path.join(anat_folder, 'mwc1*.nii'))
    binary_gm_filename = folder.split('/salma/')[1]
    binary_gm_filename = os.path.join('/neurospin/servier2/salma',
                                      binary_gm_filename)
    mask_name = 'bin_' + os.path.basename(gm_filename)
    binary_gm_filename = os.path.join(os.path.dirname(binary_gm_filename),
                                      mask_name)
    if mask_name in ['bin_mwc1anat_cp110075_20130819_acq11_04.nii',
                     'bin_mwc1anat_mp130349_20130917_acq11_04.nii',
                     'bin_mwc1anat_hs120456_20121126_acq11_04.nii']:
        threshold = 0.3
    else:
        threshold = 0.4
    binarize(gm_filename, binary_gm_filename, threshold=threshold)
    if out_folder == 'low_motion':
# TODO: generate reg_task*.mat for high moving sessions
        task_filename = single_glob(os.path.join(folder,
                                                 'regTask_' + prefix + '*.mat'))
        task_ts = loadmat(task_filename)['reg']
    print("Processing file %s" % func_filename)

    print("-- Computing confounds ...")
    hv_confounds = mem.cache(nilearn.image.high_variance_confounds)(
        func_filename)

    motion_confounds = np.genfromtxt(motion_filename)
    relative_motion = motion_confounds.copy()
    relative_motion[1:] -= motion_confounds[:-1]
    diff_motion_confounds = []
    for motion in motion_confounds:
        confounds_data_dt = motion
        conv = np.convolve(motion, np.array([1., 0., -1.]) / 2,
                           'valid')
        confounds_data_dt[1:-1] = conv
        diff_motion_confounds.append(confounds_data_dt)

    # TODO: Add aCompCor confounds
    # TODO compute compCor confounds for highly moving subjects
    if out_folder == 'low_motion':
        subject_inpath = os.path.join(
            raw_dir, 'ROI_Subject{0:03d}_Session{1:03d}.mat'.format(
            n + 1, n_session + 1))
        mat_dict = loadmat(subject_inpath)
        conn_signals = _convert_matlab(mat_dict['data'], 'cell')
        gm_mean = conn_signals[0]
        wm_pca = conn_signals[1]
        wm_mean = wm_pca[:, 0]
        csf_pca = conn_signals[2]
        csf_mean = csf_pca[:, 0]

    gm_mean = hv_confounds[:, 0]
    wm_mean = hv_confounds[:, 1]
    csf_mean = hv_confounds[:, 2]

#    compcor_confound = [confounds[name] for name in confounds.dtype.names if
#                         ('compcor' in name)]
    all_motion_confounds = np.hstack((motion_confounds,
                                      diff_motion_confounds))
    my_confounds = np.hstack((motion_confounds,
                              diff_motion_confounds))
    my_confounds = np.hstack((my_confounds, hv_confounds))

    # TODO: include mean intensities confounds
    if False:
        # Satterthwaite preprocessings
        anat_folder = folder.replace('fMRI', 't1mri')
        wm_filename = single_glob(os.path.join(anat_folder, 'mwc2*.nii'))
        csf_filename = single_glob(os.path.join(anat_folder, 'mwc3*.nii'))
        wm_masker = nilearn.input_data.NiftiMapsMasker(
            wm_filename, memory=mem, memory_level=1, verbose=2)
        csf_masker = nilearn.input_data.NiftiMapsMasker(
            csf_filename, memory=mem, memory_level=1, verbose=2)
        wm_mean = wm_masker.fit_transform(func_filename)
        csf_mean = csf_masker.fit_transform(func_filename)
        import nibabel
        global_data = nibabel.load(func_filename).get_data()
        global_mean = np.zeros(global_data.shape[-1])
        for t in range(global_data.shape[-1]):
            global_mean[t] = global_data[global_data[..., t] > 0, t].mean()
        satter_masker = nilearn.input_data.NiftiMapsMasker(
            atlas['maps'], resampling_target="maps", detrend=True,
            low_pass=.1, high_pass=.01, t_r=1, standardize=False,
            memory=mem, memory_level=1, verbose=2)
        satter_confounds = np.hstack((motion_confounds, csf_mean, wm_mean,
                                      global_mean))

    low_pass = .08
    high_pass = .009

    print("-- Computing region time series ...")
    spheres_masker = nilearn.input_data.NiftiSpheresMasker(
        all_seeds, radius, mask_img=binary_gm_filename, smoothing_fwhm=None, detrend=False, low_pass=None,
        high_pass=None, t_r=1, standardize=False, memory=mem, memory_level=1,
        verbose=2)
    region_raw_ts = spheres_masker.fit_transform(func_filename)
    region_ts = nilearn.signal.clean(region_raw_ts, detrend=True,
                                     low_pass=low_pass, high_pass=high_pass,
                                     t_r=1.,
                                     standardize=True,
                                     confounds=[relative_motion, gm_mean,
                                                wm_mean, csf_mean])
    region_no_motion_ts = nilearn.signal.clean(region_raw_ts, detrend=True,
                                               low_pass=None,
                                               high_pass=None, t_r=1.,
                                               standardize=False,
                                               confounds=all_motion_confounds)

    subjects.append(region_ts)
    nip = folder.split('/fMRI/')[0]
    nip = os.path.basename(nip)
    nips.append(nip)
    session = os.path.basename(folder)
    sessions.append(session)
    np.save(os.path.join(
        '/neurospin/servier2/salma/nilearn_outputs', out_folder, 'spheres',
        'tmp', nip + '_' + session + '_' + prefix),
        region_ts)
    np.save(os.path.join(
        '/neurospin/servier2/salma/nilearn_outputs', out_folder, 'spheres',
        'raw', nip + '_' + session + '_' + prefix),
        region_raw_ts)
    np.save(os.path.join(
        '/neurospin/servier2/salma/nilearn_outputs', out_folder, 'spheres',
        'no_motion', nip + '_' + session + '_' + prefix),
        region_no_motion_ts)


##########################################################
# Means w.r.t. distances/connectivity
##########################################################
# Load the signals for the resting state sessions
all_subjects = []
for nip, session in zip(nips, sessions):
    all_subjects.append(np.load(os.path.join(
        '/neurospin/servier2/salma/nilearn_outputs', 'low_motion', 'spheres',
        'tmp', nip + '_' + revert_acquisition(session) + '_' + 'rs1.npy')))
for nip, session in zip(nips, sessions):
    all_subjects.append(np.load(os.path.join(
        '/neurospin/servier2/salma/nilearn_outputs', 'high_motion', 'spheres',
        'tmp', nip + '_' + session + '_' + 'rs1.npy')))

n_subjects = len(all_subjects)

# Compute connectivity coefficients for each subject
print("-- Measuring connecivity ...")
measures = ['covariance', 'precision', 'tangent', 'correlation',
            'partial correlation']

from sklearn.covariance import EmpiricalCovariance, LedoitWolf, MinCovDet
estimators = [('ledoit', LedoitWolf()), ('emp', EmpiricalCovariance()),
              ('mcd', MinCovDet())]
n_estimator = 0
all_matrices = []
mean_matrices = []
from nilearn import connectivity
for subjects in [all_subjects[:n_subjects / 2],
                 all_subjects[n_subjects / 2:]]:
    for measure in measures:
        estimator = {'cov_estimator': estimators[n_estimator][1],
                     'kind': measure}
        cov_embedding = connectivity.CovEmbedding(**estimator)
        matrices = connectivity.vec_to_sym(
            cov_embedding.fit_transform(subjects))
        all_matrices.append(matrices)
        if measure == 'tangent':
            mean = cov_embedding.mean_cov_
        else:
            mean = matrices.mean(axis=0)
        mean_matrices.append(mean)
coords = all_seeds
coords = np.array([list(coord) for coord in coords])
distance_matrix = coords - coords[:, np.newaxis]
dist = np.linalg.norm(distance_matrix, axis=-1)
plt.subplot(411)
plt.scatter((mean_matrices[3][dist < 96]).flatten(),
            (dist[dist < 96]).flatten(), c='r')
plt.scatter((cov_to_corr(mean_matrices[2])[dist < 96]).flatten(),
            (dist[dist < 96]).flatten(), c='g')
plt.subplot(412)
plt.scatter((mean_matrices[3][dist > 96]).flatten(),
            (dist[dist > 96]).flatten(), c='r')
plt.scatter((cov_to_corr(mean_matrices[2])[dist > 96]).flatten(),
            (dist[dist > 96]).flatten(), c='g')
plt.subplot(413)
plt.scatter((mean_matrices[3]).flatten(),
            (mean_matrices[8]).flatten(), c='r')
plt.scatter((cov_to_corr(mean_matrices[2])).flatten(),
            (cov_to_corr(mean_matrices[7])).flatten(), c='g')
plt.plot(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
plt.xlabel('conn')
plt.ylabel('dist')
plt.show()


# Correlation between motion and connectivity, relationship with distance
#########################################################################
# Load the signals for the resting state sessions

subjects = []
for nip, session in zip(nips, sessions):
    subjects.append(np.load(os.path.join(
        '/home/sb238920/nilearn_data/servier', out_folder, 'spheres',
        'tmp', nip + '_' + session + '_' + 'rs1.npy')))
for nip, session in zip(nips, sessions):
    subjects.append(np.load(os.path.join(
        '/neurospin/servier2/salma/nilearn_outputs', out_folder, 'spheres',
        'tmp', nip + '_' + session + '_' + 'rs1.npy')))

n_subjects = len(subjects)
# Compute connectivity coefficients for each subject
print("-- Measuring connecivity ...")
measures = ['covariance', 'precision', 'tangent', 'correlation',
            'partial correlation']

from sklearn.covariance import EmpiricalCovariance, LedoitWolf, MinCovDet
estimators = [('ledoit', LedoitWolf()), ('emp', EmpiricalCovariance()),
              ('mcd', MinCovDet())]
n_estimator = 1
all_matrices = []
mean_matrices = []
from nilearn import connectivity
for measure in measures:
    estimator = {'cov_estimator': estimators[n_estimator][1],
                 'kind': measure}
    cov_embedding = connectivity.CovEmbedding(**estimator)
    matrices = connectivity.vec_to_sym(
        cov_embedding.fit_transform(subjects))
    all_matrices.append(matrices)
    if measure == 'tangent':
        mean = cov_embedding.mean_cov_
    else:
        mean = matrices.mean(axis=0)
    mean_matrices.append(mean)

# Compute mean motion for each subject
mean_motion = []
for prefix in ['rs1', 'rs1']:
    for folder in folders:
        motion_filename = single_glob(os.path.join(folder,
                                                   'rp_a' + prefix + '*.txt'))
        motion_confounds = np.genfromtxt(motion_filename)[:, :3]
        relative_motion = motion_confounds.copy()
        relative_motion[1:] -= motion_confounds[:-1]
        mean_motion.append(np.linalg.norm(relative_motion, axis=1).mean())

# Compute Euclidean distances between nodes in mm
coords = all_seeds
coords = np.array([list(coord) for coord in coords])
distance_matrix = coords - coords[:, np.newaxis]
distance_matrix = np.linalg.norm(distance_matrix, axis=-1)

# Compute pearson correlation between motion and connectivity
correlation = np.zeros(distance_matrix.shape)
all_indices = np.triu_indices(distance_matrix.shape[0], 1)
x_indices = []
y_indices = []
for indices in zip(*all_indices):
    conn = []
    for n in range(n_subjects):
        conn.append(all_matrices[3][n][indices])
    if np.mean(conn) > -1:
        correlation[indices] = pearsonr(mean_motion, conn)[0]
        x_indices.append(indices[0])
        y_indices.append(indices[1])

new_indices = (x_indices, y_indices)

# Scatter plot
dist = distance_matrix[new_indices]
corr = correlation[new_indices]
plt.scatter(dist, corr)
plt.xlabel('euclidean distance (mm)')
plt.ylabel('correlation of motion and connectivity')
r, p = pearsonr(dist, corr)
print('Pearson correlation is {0} with pval {1}'.format(r, p))
t = np.polyfit(dist, corr, 1, full=True)
xp = np.linspace(dist.min(), dist.max(), 100)
p1 = np.poly1d(np.polyfit(dist, corr, 1))
plt.plot(xp, p1(xp))
print('corr = {0} (dist - {1})'.format(t[0][1], - t[0][0] / t[0][1]))
plt.show()

plot_matrix(all_matrices[3][40:].mean(axis=0) -
            all_matrices[3][:40].mean(axis=0))
plt.show()

