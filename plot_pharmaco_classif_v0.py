import sys

import numpy as np
import matplotlib.pylab as plt


code_path = np.genfromtxt('/home/sb238920/CODE/anonymisation/code_path.txt',
                          dtype=str)
sys.path.append(str(code_path))
from my_conn import MyConn


# Specify the location of the  CONN project
conn_folders = np.genfromtxt(
    '/home/sb238920/CODE/anonymisation/conn_projects_paths.txt', dtype=str)
conn_folder_filt = conn_folders[0]
conn_folder_no_filt = conn_folders[1]
mc_filt = MyConn('from_conn', conn_folder_filt)
mc_no_filt = MyConn('from_conn', conn_folder_filt)

# Specify the conditions
condition_baseline = 'Nbac2_Placebo'
condition_follow_up = 'ReSt2_Placebo'

 # Specify the ROIs and their networks. Here: biyu's order
WMN = ['IPL', 'LMFG_peak1',  # 'RMFG_peak2' removed because of bad results
       'RCPL_peak1', 'LCPL_peak3', 'LT']
AN = ['vIPS_big', 'pIPS_big', 'MT_big', 'FEF_big', 'RTPJ', 'RDLPFC']
DMN = ['AG_big', 'SFG_big', 'PCC', 'MPFC', 'FP']
template_ntwk = [('WMN', WMN), ('AN', AN), ('DMN', DMN)]

# Whether to standardize or not the timeseries
standardize = False

# Collect the timeseries
for mc in [mc_no_filt, mc_filt]:
    mc.setup()
    mc.analysis(template_ntwk, standardize, 'correlations')
    if 'Nbac' in condition_baseline:
        signals_baseline = mc_no_filt.runs_[condition_baseline]
    else:
        signals_baseline = mc.runs_[condition_baseline]

    if 'Nbac' in condition_follow_up:
        signals_follow_up = mc_no_filt.runs_[condition_follow_up]
    else:
        signals_follow_up = mc.runs_[condition_follow_up]

n_subjects = 40
from sklearn.covariance import EmpiricalCovariance
mean_matrices = []
all_matrices = []
measures = ["tangent", "correlation", "partial correlation", "covariance",
            "precision"]

subjects = [subj for subj in signals_baseline] + \
           [subj for subj in signals_follow_up]

# Estimate connectivity matrices
import nilearn.connectivity
subjects_connectivity = {}
mean_connectivity = {}
for measure in measures:
    cov_embedding = nilearn.connectivity.CovEmbedding(
        measure=measure, cov_estimator=EmpiricalCovariance())
    subjects_connectivity[measure] = nilearn.connectivity.vec_to_sym(
        cov_embedding.fit_transform(subjects))
    # Compute the mean connectivity across all subjects
    if measure == 'tangent':
        mean_connectivity[measure] = cov_embedding.tangent_mean_
    else:
        mean_connectivity[measure] = \
            subjects_connectivity[measure].mean(axis=0)

# Plot the mean connectivity
if False:
    import numpy as np
    import nilearn.plotting
    labels = np.recfromcsv(atlas.labels)
    region_coords = np.vstack((labels['x'], labels['y'], labels['z'])).T
    for measure in ['tangent', 'correlation', 'partial correlation']:
        nilearn.plotting.plot_connectome(mean_connectivity[measure], region_coords,
                                         edge_threshold='98%',
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
plt.title(condition_follow_up + ' vs ' + condition_baseline)
plt.show()
