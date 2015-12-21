"""This example plots the individual connectivity matrices and graphs for the
good and bad subjects identified by sorting w.r.t. different criteria
- the average squared euclidean distance to the other correlations, partial
correlations and covariances
- the average euclidean distance to the other correlations, partial
correlations and covariances
- the average squared euclidean distance to the other covariances
- the average squared geometric distance to the other covariances
- the maximal covariance eigenvalues
- the covariance determinant
"""
import numpy as np
import matplotlib.pylab as plt

# Load preprocessed abide timeseries extracted from harvard oxford atlas
from nilearn import datasets
abide = datasets.fetch_abide_pcp(derivatives=['rois_ho'], DX_GROUP=2)
subjects_unscaled = abide.rois_ho

# Standardize the signals
scaling_type = 'unnormalized'
from nilearn import signal
if scaling_type == 'normalized':
    subjects = []
    for subject in subjects_unscaled:
        subjects.append(signal._standardize(subject))
else:
    subjects = subjects_unscaled

# Estimate connectivity matrices
from sklearn.covariance import LedoitWolf
import nilearn.connectivity
measures = ["correlation", "partial correlation", "robust dispersion",
            "covariance"]
subjects_connectivity = {}
mean_connectivity = {}
from joblib import Memory
mem = Memory('/home/sb238920/CODE/Parietal/nilearn/nilearn_cache/abide')
for measure in measures:
    cov_embedding = nilearn.connectivity.ConnectivityMeasure(
        kind=measure, cov_estimator=LedoitWolf())
    subjects_connectivity[measure] = cov_embedding.fit_transform(subjects,
                                                                 mem=mem)
    if measure == "robust dispersion":
        mean_connectivity[measure] = cov_embedding.robust_mean_
    else:
        mean_connectivity[measure] = np.mean(subjects_connectivity[measure],
                                             axis=0)

# Compute average distance from each subject to the others
from nilearn.connectivity2 import analyzing
indices = {}
features = {}
for measure in ['correlation', 'partial correlation']:
    features['euclidean ' + measure] = mem.cache(
        analyzing.compute_std_spreading)(subjects_connectivity[measure])
    features['euclidean squared ' + measure] = mem.cache(
        analyzing.compute_spreading)(subjects_connectivity[measure])

features['geometric squared'] = mem.cache(analyzing.compute_geo_spreading)(
    subjects_connectivity['covariance'])
features['geometric'] = mem.cache(analyzing.compute_geo_std_spreading)(
    subjects_connectivity['covariance'])
max_eigenvalues = [np.linalg.eigvalsh(subject_connectivity).max() for
                   subject_connectivity in subjects_connectivity['covariance']]
features['maximal eigenvalue'] = max_eigenvalues
features['determinant'] = [np.linalg.eigvalsh(subject_connectivity).prod() for
    subject_connectivity in subjects_connectivity['covariance']]

# Sort the subjects
n_subjects = len(subjects)
sorted_indices = {}
for criteria in ['geometric', 'euclidean correlation',
                 'euclidean partial correlation', 'determinant',
                 'maximal eigenvalue']:
    print(criteria)
    sorted_indices[criteria] = np.argsort(features[criteria])

# Load quality assessment score

# Use only temporal criteria for functional data
# 93: func_perc_fd, 92: func_num_fd, 91: func_mean_fd, 90: func_quality,
# 89: func_outlier, 88: func_dvars
quality_names = {85: 'entropy',
                 86: 'Foreground to Background Energy Ratio',
                 87: 'func FWHM',
                 88: 'DVARS',
                 89: 'outlier voxels',
                 90: 'mean distance to\nmedian volume',
                 91: 'mean framewise\ndisplacement',
                 92: 'number fd',
                 93: 'percent FD',
                 94: 'func GSR'}
higher_is_noiser = {85: False, 86: False, 87: False,
                    88: True, 89: True, 90: True, 91: True, 92: True, 93: True,
                    94: False}
lower_is_noiser = {85: True, 86: True, 87: True,
                   88: False, 89: False, 90: False, 91: False, 92: False,
                   93: False, 94: True}
quality = {}
for quality_id in range(85, 94):
    quality[quality_id] = [abide.phenotypic[subject_id][quality_id] for
                           subject_id in range(n_subjects)]

# Get the worst subjects (outside the interquartile range for all criteria)
outliers = {}
centered_quality = {}
sorting = {}
for quality_id in range(85, 94):
    sorting[quality_id] = np.argsort(quality[quality_id])
    q75, q25 = np.percentile(quality[quality_id], [75, 25])
    q50 = np.median(quality[quality_id])
    iqr = q75 - q25
    outliers[quality_id] = [(value > q75 + 1.5 * iqr)
                            for value in quality[quality_id]]
    centered_quality[quality_id] = np.array([value for value in
                                             quality[quality_id]])
score = np.sum([outliers[quality_id] for quality_id in range(88, 94)], axis=0)
worst_subjects = np.where(score >= 3)[0]
mean_sorting = np.mean([sorting[quality_id] for quality_id in range(88, 89)],
                       axis=0, dtype=int)

# Check the rank of worst subjects
for criteria in ['geometric', 'euclidean correlation',
                 'euclidean partial correlation', 'determinant',
                 'maximal eigenvalue']:
    print criteria
    print [np.where(sorted_indices[criteria] == subject_idx) for subject_idx
           in worst_subjects]

# Check that low eigenvalue subjects are good subjects (inside the
# interquartile range for all the criteria)
for criteria in ['geometric', 'euclidean correlation',
                 'euclidean partial correlation', 'determinant',
                 'maximal eigenvalue']:
    print criteria
    print score[sorted_indices[criteria][450:]]

for criteria in ['geometric', 'euclidean correlation', 
                 'maximal eigenvalue', 'determinant']:
    plt.figure()
    plt.bar(np.arange(len(subjects)),
            score[sorted_indices[criteria]],  #centered_quality[93][sorted_indices[criteria]],
            width=.7,
            color='r',
            edgecolor='r')
    plt.xlabel('subject rank, sorted by {}'
               .format(criteria))
    plt.xlim(-10, len(subjects) + 10)
    axes = plt.gca()
    axes.yaxis.tick_right()
    plt.ylabel('score')
plt.show()


def modified_z_score(points):
    if len(points.shape) == 1:
        points = points[:, None]

    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)
    return 0.6745 * diff / med_abs_deviation


def is_outlier(points, threshold=3.5):
    """
    Returns a boolean array with True if points are outliers and False
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        threshold : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points.shape) == 1:
        points = points[:, None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > threshold


# Scatter plot geometric and euclidean distances and highlight outliers
figure = plt.figure(figsize=(5, 5.2))
x = np.array(features['geometric'])
y = np.array(features['euclidean correlation'])
z = np.array(features['euclidean partial correlation'])
plt.scatter(x, y, c='k', edgecolor='k', marker='x')
x_min, x_max = x.min() - .5, x.max() + .5
y_min, y_max = y.min() - .5, y.max() + .5
#x = np.arange(1,5,0.01)
yarr = np.vstack((np.arange(1,5,0.01),))

#plt.imshow(yarr, extent=(x_min,x_max, y_min,y_max), cmap=plt.cm.hot, alpha=.5,
#           interpolation='none')
#plt.imshow(yarr.T, extent=(x_min,x_max, y_min,y_max), cmap=plt.cm.hot_r, alpha=.5,
#           interpolation='none')
#z_min, z_max = z.min() - .5, z.max() + .5
h = 0.02
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
#cm = plt.cm.RdBu
#plt.contourf(xx, yy, modified_z_score(np.dstack((xx, yy))), cmap=cm, alpha=.8)
#plt.contourf(xx, yy, modified_z_score(np.dstack((yy, xx))), cmap=cm, alpha=.8)
for quality_id, color in zip(range(88, 92) + [93], 'cgbrmyrbg'):
    q75, q25 = np.percentile(quality[quality_id], [75, 25])
    q95, q5 = np.percentile(quality[quality_id], [95, 5])
    iqr = q75 - q25
    if quality_id == 88:  # DVARS distribution not very dispersed
        frac = 3.
    else:
        frac = 3.

    if quality_id == 93:  # percent FD has big values
        size_fractor = 5.
    else:
        size_fractor = 50.
    
    if higher_is_noiser[quality_id]:
        indices = np.where(quality[quality_id] > q75 + frac * iqr)
        size = size_fractor * np.array(quality[quality_id]) / np.median(quality[quality_id])
    elif lower_is_noiser[quality_id]:
        indices = np.where(quality[quality_id] < q5)
        size = 50. * np.median(quality[quality_id]) / np.array(quality[quality_id])
    else:
        indices = np.where(np.logical_or(quality[quality_id] > q75 + 1.5 * iqr,
                                         quality[quality_id] < q25 - 1.5 * iqr))
    plt.scatter(x[indices], y[indices], s=5, marker='o',
                c=color, edgecolor=color,
                label=quality_names[quality_id])
    plt.scatter(x[indices], y[indices], s=size[indices], marker='o',
                edgecolor=color,
                facecolor='none')
    plt.scatter(x[indices], y[indices], marker='x',
                c='k', edgecolor='k')

# Plot thresholds
#for percentile in [75, 90]:
#    plt.axhline(np.percentile(y, percentile))#, xmin=x_min, xmax=x_max, color='k')
#    plt.axvline(np.percentile(x, percentile))#, ymin=y_min, ymax=y_max)
threshold = 3.5
#size = 500. * modified_z_score(x[:, np.newaxis])
#plt.scatter(x[is_outlier(x, threshold=threshold)].ravel(),
#            y[is_outlier(x, threshold=threshold)].ravel(), marker='o', s=size,
#            facecolors='none')
#size = 500. * modified_z_score(y[:, np.newaxis])
#plt.scatter(x[is_outlier(y, threshold=threshold)].ravel(),
#            y[is_outlier(y, threshold=threshold)].ravel(), marker='<', s=size,
#            facecolors='none')
plt.xlabel('geometric covariance distance')
plt.ylabel('euclidean correlation distance')
plt.xlim(14, 35)
plt.ylim(17, 45)
figure.suptitle('outliers and average distance to the other subjects')
plt.legend(fontsize='12')
plt.savefig('/home/sb238920/CODE/salma/figures/outliers_abide.pdf')
plt.show()

#Try relate some motion parameters or variance to eigenvalues
if False:
    for criteria in ['geometric', 'euclidean correlation']:
        for quality_id, color in zip(range(88, 94), 'rgbkyc'):
            plt.figure()
            plt.scatter(quality[quality_id], features[criteria], c=color)
            plt.ylabel(criteria)

    mvt_parameters = np.asarray(dataset.motion[condition])
    displacements = np.diff(dataset.motion[condition])
    norm_translation = np.linalg.norm(mvt_parameters[..., :3], axis=-1)
    norm_translation_disp = np.linalg.norm(displacements[..., :3], axis=-1)
    norm_rotation = np.linalg.norm(mvt_parameters[..., 3:], axis=-1)
    norm_rotation_disp = np.linalg.norm(displacements[..., 3:], axis=-1)
    motion = np.max(norm_translation_disp, axis=1)
    motion = np.median(norm_translation, axis=1)
    labels = ['sub ' + str(n) for n in range(40)]
    
    for n_plot, feature_name in enumerate(['geometric', 'euclidean correlation',
                                      'euclidean partial correlation']):
        plt.subplot(3, 1, n_plot + 1)
        plt.plot(motion, features[feature_name], '.')
        plt.ylabel(feature_name)
        for label, x, y in zip(labels, motion, features[feature_name]):
            p = plt.annotate(
                label, 
                xy = (x, y), xytext = (-20, 20),
                textcoords = 'offset points', ha = 'right', va = 'bottom',
                bbox = dict(boxstyle = 'round,pad=0.1', fc = 'yellow', alpha = 0.5),
                arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
    
    plt.xlabel('motion')
    plt.show()