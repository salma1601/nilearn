import numpy as np
import matplotlib.pylab as plt

import dataset_loader
from nilearn.connectivity2.collecting import single_glob

# Specify the networks
WMN = [' L IPL', ' L MFG', ' L CPL', ' R CPL1', ' L Th']
AN = [' L vIPS', ' R vIPS', ' R TPJ', ' R DLPFC', ' L pIPS', ' R pIPS',
      ' L MT', ' R MT', ' L FEF', ' R FEF']
DMN = [' L AG', ' R AG', ' L SFG', ' R SFG', ' PCC', ' MPFC', ' FP']
networks = [('WMN', WMN), ('AN', AN), ('DMN', DMN)]

# Specify the location of the files and their patterns
anonymisation_file = '/home/sb238920/CODE/anonymisation/nilearn_paths.txt'
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
    motion_path = os.path.join(motion_folder, 'rp_ars1_' + sub_id + '*acq' +
                               session_number + '*.txt')
    try:
        _ = single_glob(motion_path)
    except:
        motion_patterns[n] = (sub_id, session_number[0])

conditions = ['rs1', 'nback_2']
dataset = dataset_loader.load_nilearn(timeseries_folder, motion_folder,
                                      timeseries_pattern, motion_patterns,
                                      conditions=conditions,
                                      standardize=False, networks=networks)


# Estimate connectivity matrices
from sklearn.covariance import EmpiricalCovariance
import nilearn.connectivity
n_subjects = 40
mean_matrices = []
all_matrices = []
measures = ["tangent", "correlation", "partial correlation", "covariance",
            "precision"]
subjects = [subj for condition in conditions for subj in
            dataset.time_series[condition]]
subjects_connectivity = {}
mean_connectivity = {}
for measure in measures:
    cov_embedding = nilearn.connectivity.CovEmbedding(
        kind=measure, cov_estimator=EmpiricalCovariance())
    subjects_connectivity[measure] = nilearn.connectivity.vec_to_sym(
        cov_embedding.fit_transform(subjects))
    # Compute the mean connectivity across all subjects
    if measure == 'tangent':
        mean_connectivity[measure] = cov_embedding.mean_cov_
    else:
        mean_connectivity[measure] = \
            subjects_connectivity[measure].mean(axis=0)

# Plot the mean connectivity
import nilearn.plotting
labels, region_coords = zip(*dataset.rois)
for measure in ['tangent', 'correlation', 'partial correlation']:
    nilearn.plotting.plot_connectome(mean_connectivity[measure], region_coords,
                                     edge_threshold='95%',
                                     title='mean %s' % measure)

wmn = range(4)
an = range(5, 9)
dmn = range(11, 16)

wmn_an = range(0, 8)
dmn_big = range(9, 16)
wmn_an_2b = [0, 1, 2, 3, 5, 6, 7, 8]

# Use connectivity coefficients to classify ADHD vs controls
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import RidgeClassifier
from sklearn.lda import LDA
from sklearn.cross_validation import StratifiedShuffleSplit, cross_val_score
classifiers = [LinearSVC(), KNeighborsClassifier(n_neighbors=1), LDA(),
               LogisticRegression(), GaussianNB(), RidgeClassifier()]
classifier_names = ['SVM', 'KNN', 'LDA', 'logistic', 'GNB', 'ridge']
classes = np.hstack((np.zeros(n_subjects), np.ones(n_subjects)))
cv = StratifiedShuffleSplit(classes, n_iter=1000, test_size=0.33)
scores = {}
for measure in measures:
    scores[measure] = {}
    print('---------- %20s ----------' % measure)
    for classifier, classifier_name in zip(classifiers, classifier_names):
        coefs_vec = nilearn.connectivity.embedding.sym_to_vec(
            subjects_connectivity[measure])
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
    plt.bar(tick_position, score_means, yerr=score_stds, label=measure,
            color=color, width=.2)
    tick_position = tick_position + .15
plt.ylabel('Classification accuracy')
plt.legend(measures, loc='upper left')
plt.ylim([0., 1.2])
plt.title(conditions[0] + ' vs ' + conditions[1])
plt.show()
