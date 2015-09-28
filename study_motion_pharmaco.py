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
from nilearn.connectivity2.collecting import (single_glob, group_left_right,
                                              combination_replacement)
from nilearn.connectivity2.analyzing import compute_connectivity


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
all_coords = []
n_regions_per_ntwk = []
group = True  # Group or not left and right ROIs
biyu_networks = [locator.meta_motor, locator.meta_non_cortical,
                 locator.func_working_memory, locator.meta_visual,
                 locator.meta_attention, locator.meta_default_mode,
                 locator.meta_saliency]
servier_networks = [locator.func_working_memory, locator.meta_attention,
                    locator.meta_default_mode]
for get_networks in servier_networks:
    names, coords = get_networks()
    if group:
        grouped_names = group_left_right(names, np.empty((1, len(names))))[0]
        n_regions_per_ntwk.append(len(grouped_names))
    else:
        n_regions_per_ntwk.append(len(names))
    all_names += names
    all_coords += coords

print all_names
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
measures = ['tangent', 'correlation',
            'partial correlation', 'precision', 'covariance']
estimators = {'ledoit': LedoitWolf(), 'emp': EmpiricalCovariance(),
              'mcd': MinCovDet()}
estimator_name = 'emp'
cov_estimator = estimators[estimator_name]
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

    all_matrices, mean_matrices = compute_connectivity(
        subjects, cov_estimator=cov_estimator)
    plt.plot((mean_matrices['correlation'] - matrix_stats.cov_to_corr(
              mean_matrices['tangent'])).ravel(),
             (mean_matrices['correlation'] -
                 mean_matrices['partial correlation']).ravel(), color,
             label=prefix)
    plt.xlabel('corrs - corr(gmean)')
    plt.ylabel('corrs - partials')
    plt.legend()
    plt.title('differences between connectivity measures over regions, '
              '{}'.format(estimator_name))
    r, pval = pearsonr((mean_matrices['correlation'] -
                        matrix_stats.cov_to_corr(
                            mean_matrices['tangent'])).ravel(),
                       (mean_matrices['correlation'] -
                        mean_matrices['partial correlation']).ravel())
    print('pearson corr = {}, pval = {}'.format(r, pval))
plt.show()
#########################################################
# Between conditions comparisons
#########################################################
# Compute connectivities
print("-- Measuring connecivity ...")
mean_scores = []
std_scores = []
prefixes_couples = [('nback_3', 'rs1'), ('nback_2', 'rs2'), ('rs1', 'rs2')]
from sklearn.cross_validation import StratifiedShuffleSplit
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

        if group:
            regions = range(len(set(
                [name.replace('L ', 'R ') for name in all_names])))
        else:
            regions = range(len(all_names))
#        regions = [12, 13, 14, 15, 17, 20, 23, 24, 25, 21, 22] + range(26, 31)
#        n_regions_per_ntwk = [5, 6, 5]
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
    n_subjects = len(subjects) / 2
    all_matrices, mean_matrices = compute_connectivity(
        subjects, cov_estimator=cov_estimator)

    # Statistical t-tests
    for measure in measures:
        matrices = all_matrices[measure]
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
                                   lines=np.cumsum(n_regions_per_ntwk)[:-1],
                                   zero_diag=False)
    plt.show()

    # Use connectivity coefficients to classify different conditions
    from sklearn.svm import LinearSVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import RidgeClassifier
    from sklearn.cross_validation import cross_val_score
    classifiers = [LinearSVC(), KNeighborsClassifier(n_neighbors=1,
                   weights = 'distance'),
                   LogisticRegression(), GaussianNB(), RidgeClassifier()]
    classifier_names = ['SVM', 'KNN', 'logistic', 'GNB', 'ridge']
    classes = np.hstack((np.zeros(n_subjects), np.ones(n_subjects)))
    cv = StratifiedShuffleSplit(classes, n_iter=100, test_size=0.33)
    scores = {}
    for measure in measures:
        scores[measure] = {}
        print('---------- %20s ----------' % measure)
        for classifier, classifier_name in zip(classifiers, classifier_names):
            coefs = all_matrices[measure]
            # regions of servier report
            regions = range(coefs.shape[-1])
            X = coefs[:, regions, :]
            X = coefs[:, :, regions]
            coefs_vec = connectivity.sym_to_vec(X)
            cv_scores = cross_val_score(
                classifier, coefs_vec, classes, cv=cv, scoring='accuracy')
            scores[measure][classifier_name] = cv_scores
            print(' %14s score: %1.2f +- %1.2f' % (classifier_name,
                  cv_scores.mean(), cv_scores.std()))

    # Display the classification scores
    plt.figure()
    tick_position = np.arange(len(classifiers))
    plt.xticks(tick_position + 0.35, classifier_names)
    for color, measure in zip('rgbyk', measures):
        score_means = [scores[measure][classifier_name].mean() for
                       classifier_name in classifier_names]
        score_stds = [scores[measure][classifier_name].std() for
                       classifier_name in classifier_names]
        plt.bar(tick_position, score_means, yerr=score_stds,
                label=measure, color=color,
                width=.2)
        tick_position = tick_position + .15
    plt.ylabel('Classification accuracy')
    plt.legend(measures, loc='upper left')
    plt.ylim([0., 1.])
    plt.title(prefix2 + ' vs ' + prefix1)
    plt.show()


# Classify conditions
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
y = np.hstack((np.zeros(n_subjects), np.ones(n_subjects)))
print('---Shuffle split, this takes time ...')
skf = StratifiedShuffleSplit(y, n_iter=100, test_size=0.33)
for measure in measures:
    coefs = all_matrices[measure]
    # regions of servier report
    regions = range(coefs.shape[-1])
    X = coefs[:, regions, :]
    X = coefs[:, :, regions]
    X = connectivity.sym_to_vec(X)
    clf_svc = GaussianNB()  # TODO: multiclass?
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
plt.ylim([0, 100])
plt.ylabel('Classification accuracy (%)', fontsize='17')
plt.legend(measures, fontsize='17')
plt.show()

###################################################################

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
            #eg: int(re.search(r'\d+', string1).group())
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
            all_low_paths = combination_replacement(
                all_low_paths, [(low_path, high_path)])[1][0]
            replacement_tuples.append((high_path, low_path))

    combination_noisy_paths = combination_replacement(all_low_paths,
                                                      replacement_tuples)
    for cresendo_noisy_paths in combination_noisy_paths.values():
        cresendo_noisy_paths = [all_low_paths] + cresendo_noisy_paths

# TODO: compare the connectivity with outliers: the norm, and the significant
# connections (hope is adds a small distance connection/ remove high distance)
    subjects = [np.load(path) for path in all_low_paths]
    _, mean_matrices = compute_connectivity(subjects,
                                            cov_estimator=cov_estimator)
    # TODO: update using combination_noisy_paths
    dict_noisy_means = {}
    for n_high, cresendo_noisy_paths in combination_noisy_paths.items():
        noisy_means = []
        for noisy_paths in cresendo_noisy_paths:
            noisy_subjects = [np.load(path) for path in noisy_paths]
            noisy_means.append(compute_connectivity(
                noisy_subjects, cov_estimator=cov_estimator)[1])
        dict_noisy_means[n_high] = noisy_means

    def distance(x, y):
        return np.linalg.norm(x - y)

    # TODO: factorize this code
    mean_cov_err = {}
    mean_robust_cov_err = {}
    mean_corr_err = {}
    mean_robust_corr_err = {}
    mean_prec_err = {}
    mean_robust_prec_err = {}
    mean_part_err = {}
    mean_robust_part_err = {}
    mean_errors = [mean_cov_err, mean_robust_cov_err,
                                       mean_corr_err, mean_robust_corr_err,
                                       mean_prec_err, mean_robust_prec_err,
                                       mean_part_err, mean_robust_part_err]
    for n_outliers, noisy_means in dict_noisy_means.items():
        cov_err = []
        robust_cov_err = []
        corr_err = []
        robust_corr_err = []
        prec_err = []
        robust_prec_err = []
        part_err = []
        robust_part_err = []
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
        for errors, mean_error in zip([cov_err, robust_cov_err,
                                       corr_err, robust_corr_err,
                                       prec_err, robust_prec_err,
                                       part_err, robust_part_err],
                                       mean_errors):
            mean_error[n_outliers] = np.mean(errors)

    # TODO: use tuples instead of a list
    errors = []
    for mean_error in mean_errors:
        errors.append([mean_error[n] for n in range(len(mean_error))])

    n_noisy = len(mean_error)
    plt.figure()
    colors = ['r', 'b', 'g', 'k']
    ls = ['-', '--']
    percent_outliers = [100. * n_outliers / n_subjects for n_outliers in
                        range(n_noisy)]
    percent_str = [str(percent) + '%' for percent in percent_outliers]
    lineObjects = []
    for n, error in enumerate(errors):
        lineObjects += plt.plot(percent_outliers, error, colors[n / 2],
                                linestyle=ls[n % 2])
    plt.legend(iter(lineObjects), ('mean covariances', 'gmean',
                                   'mean correlations', 'corr(gmean)',
                                   'mean precisions', r'gmean$ ^{-1}$',
                                   'mean partial correlations',
                                   'partial(gmean)'),
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
    [mean_corr_err, mean_robust_corr_err, mean_part_err,
        mean_robust_part_err] = errors[2:4] + errors[6:8]
    for n, error in enumerate([mean_corr_err, mean_robust_corr_err,
                               mean_part_err, mean_robust_part_err]):
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
    plt.show()

    # correlation, partials and corr(gmean)
    plt.figure(figsize=(8, 6))
    colors = ['g', 'k', 'r']
    ls = ['-', '--']
    percent_outliers = [100. * n_outliers / n_subjects for n_outliers in
                        range(n_noisy)]
    percent_str = [str(percent) + '%' for percent in percent_outliers]
    lineObjects = []
    for n, error in enumerate([mean_corr_err, mean_part_err,
                               mean_robust_corr_err]):
        lineObjects += plt.plot(percent_outliers, error, colors[n])
    plt.legend(iter(lineObjects), ('mean correlations',
                                   'mean partial correlations', 'corr(gmean)'),
               loc=0)
    #plt.title('robustness')
    plt.xlabel('percentage of outliers')
    plt.xticks(percent_outliers, percent_str, size=8)
    plt.ylabel('relative error')
    plt.show()

