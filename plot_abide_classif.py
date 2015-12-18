import numpy as np
import matplotlib.pylab as plt


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
measures = ["correlation", "partial correlation"]
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

subjects_connectivity = {}
mean_connectivity = {}
from joblib import Memory
mem = Memory('/home/sb238920/CODE/Parietal/nilearn/nilearn_cache/abide')
from sklearn.covariance import LedoitWolf
for measure in ['correlation', 'partial correlation']:
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

# Plot the mean connectivity
import nilearn.plotting
if False:
    labels, region_coords = zip(*dataset.rois)
    for measure in ['robust dispersion', 'correlation', 'partial correlation']:
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
classifiers = [LinearSVC(random_state=1), LDA(),
               LogisticRegression(random_state=1),
               GaussianNB(), RidgeClassifier()]
classifier_names = ['SVM', 'LDA', 'logistic', 'GNB', 'ridge']
# TODO: stratify also w.r.t. site
classes = np.hstack((np.zeros(n_controls), np.ones(n_autists)))
n_iter = 10
cv = StratifiedShuffleSplit(classes, n_iter=n_iter, test_size=0.3,
                            random_state=1)
scores = {}
corrs = subjects_connectivity['correlation']
partials = subjects_connectivity['partial correlation']
subjects_connectivity['mix'] = 0.573497578635 * (corrs - partials) ** 2 +\
    0.471676957796 * (corrs - partials) + partials
for measure in measures + ['mix']:
    if measure == 'robust dispersion':
        mem2 = Memory(None)
    else:
        mem2 = mem
    scores[measure] = {}
    print('---------- %20s ----------' % measure)
    for classifier, classifier_name in zip(classifiers, classifier_names):
        coefs_vec = nilearn.connectivity.connectivity_matrices.sym_to_vec(
            subjects_connectivity[measure])
        cv_scores = mem2.cache(cross_val_score)(
            classifier, coefs_vec, classes, cv=cv, scoring='accuracy')
        scores[measure][classifier_name] = cv_scores
        print(' %14s score: %1.2f +- %1.2f' % (classifier_name,
              cv_scores.mean(), cv_scores.std()))

# Display the classification scores
plt.figure(figsize=(5, 4))
tick_position = np.arange(len(classifiers))
plt.xticks(tick_position + 0.25, classifier_names)
labels = []
for color, measure in zip('rgbyk', measures + ['mix']):
    score_means = [scores[measure][classifier_name].mean() * 100. for
                   classifier_name in classifier_names]
    score_stds = [scores[measure][classifier_name].std() * 100. for
                  classifier_name in classifier_names]
    if measure == 'robust dispersion':
        label = 'deviation from gmean'
    else:
        label = measure
    plt.bar(tick_position, score_means, yerr=score_stds, label=label,
            color=color, width=.2)
    labels.append(label)
    tick_position = tick_position + .15
plt.ylabel('Classification accuracy')
plt.legend(labels, loc='upper left')
plt.ylim([50, 85])
ax = plt.gca()
ax.yaxis.tick_right()
plt.yticks([50, 60, 70, 80], ['50%', '60%', '70%', '80%'])
plt.grid()
plt.title(conditions[0] + ' vs ' + conditions[1])
#plt.savefig('/home/sb238920/CODE/salma/figures/abide_{0}_classif_{1}iter.pdf'
#            .format(scaling_type, n_iter))
plt.show()
