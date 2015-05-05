"""
Comparing different connectivity measures
=========================================

This example shows how to extract signals from regions defined by an atlas,
and to estimate different connectivity measures based on these signals.
"""
overwrite = False
max_outliers = 20

import os

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from funtk.connectivity import matrix_stats


# Copied from matplotlib 1.2.0 for matplotlib 0.99 compatibility.
_bwr_data = ((0.0, 0.0, 1.0), (1.0, 1.0, 1.0), (1.0, 0.0, 0.0))
plt.cm.register_cmap(cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
    "bwr", _bwr_data))


def corr_to_Z(corr, tol=1e-7):
    """Applies Z-Fisher transform. """
    Z = corr.copy()  # avoid side effects
    corr_is_one = 1.0 - abs(corr) < tol
    Z[corr_is_one] = np.inf * np.sign(Z[corr_is_one])
    Z[np.logical_not(corr_is_one)] = \
        np.arctanh(corr[np.logical_not(corr_is_one)])
    return Z


def plot_matrix(mean_conn, title="connectivity", ticks=[], tick_labels=[],
                xlabel="", ylabel=""):
    """Plot connectivity matrix, for a given measure. """

    mean_conn = mean_conn.copy()
    vmax = np.abs(mean_conn).max()
    if vmax <= 2e-16:
        vmax = 0.1

    # Display connectivity matrix
    plt.figure(figsize=(4, 3))
    plt.imshow(mean_conn, interpolation="nearest",
               vmin=-vmax, vmax=vmax, cmap=plt.cm.get_cmap("bwr"))
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=8)
    ax = plt.gca()
    ax.xaxis.set_ticks_position('bottom')
    plt.xticks(ticks, tick_labels, size=8, rotation=90)
    plt.xlabel(xlabel)
    plt.yticks(ticks, tick_labels, size=8)
    ax.yaxis.tick_left()
    plt.ylabel(ylabel)

    plt.title(title)


def plot_matrices(conns, title="connectivity", subtitles=[], ticks=[],
                  tick_labels=[]):
    """Plot connectivity matrices, for a given measure. """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    conns = conns.copy()
    n_subjects = conns.shape[0]
    if not subtitles:
        subtitles = ['subject ' + str(n_subject) for n_subject in
                     range(n_subjects)]

    ncols = 4
    n_subject = 0
    plt.figure(figsize=(10, 10))
    for conn in conns:
        size = conn.shape[0]
        conn[range(size), range(size)] = 0
        vmax = np.abs(conn).max()
        if vmax <= 2e-16:
            vmax = 0.1

        # Display connectivity matrices
        n_line = n_subject / ncols
        i = n_subject % ncols
        plt.subplot(n_subjects / ncols, ncols, ncols * n_line + i + 1)
        im = plt.imshow(conn, interpolation="nearest",
                        vmin=-vmax, vmax=vmax, cmap=plt.cm.get_cmap("bwr"))
        ax = plt.gca()
        ax.xaxis.set_ticks_position('top')
        plt.xticks(ticks, tick_labels, size=8, rotation=90)
        plt.yticks(ticks, tick_labels, size=8)
        ax.yaxis.tick_left()
        plt.xlabel(subtitles[n_subject])
        n_subject += 1

        # Customize the colorbars
        plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        int_abs_max = round(vmax, 0)
        r = 0
        while int_abs_max == 0:
            r = r + 2
            int_abs_max = round(vmax, r)
        cb_label_size = max(1, 8 - r)
        cbticks = [-vmax, 0, vmax]
        tickFormat = '%.1f'
        if r > 2:
            tickFormat = '%.1e'
        cb = plt.colorbar(im, cax=cax, ticks=cbticks, format=tickFormat)
        cb.ax.tick_params(labelsize=cb_label_size)

print("-- Fetching datasets ...")
import nilearn.datasets
atlas = nilearn.datasets.fetch_msdl_atlas()
dataset = nilearn.datasets.fetch_adhd()

import nilearn.image
import nilearn.input_data

import joblib

mem = joblib.Memory("/home/sb238920/CODE/Parietal/nilearn/nilearn_cache/adhd")


# Number of subjects to consider for connectivity computations
n_subjects = 40
subjects = []
adhds = []
controls = []
kki = []
neuro = []
nyu = []
ohsu = []
peking = []
adhd2 = []
adhd = dataset.phenotypic['adhd']
# Rorder the regions
# new order Aud, striate, DMN, VAN, DAN, Ant IPS, Cing insula, Occ Post, Motor,
# basal ...
# TODO : read from csv file
aud = [0, 1]
striate = [2]
dmn = [3, 4, 5, 6]
occ = [7]
motor = [8]
van = [9, 10, 11, 12, 14, 15, 16]
basal = [13]
dan = [17, 18]
vis = [19, 20, 21]
salience = [22, 23, 24]
temporal = [25, 26]
language = [27, 28, 29, 30, 31]
cerebellum = [32]
dpcc = [33]
cing = [34, 35, 36]
ips = [37, 38]
reorder = False
subjects3 = []
gms = []
sites = []
mean_motions = []
for subject_n in range(n_subjects):
    filename = dataset["func"][subject_n]
    print("Processing file %s" % filename)

    print("-- Computing confounds ...")
    confound_file = dataset["confounds"][subject_n]
    hv_confounds = mem.cache(nilearn.image.high_variance_confounds)(filename)

    # CSF, WM and global intensities
    means = []
    for col in [0, 3, 4]:
        confounds = np.genfromtxt(confound_file, delimiter='\t', names=True,
                                  usecols=(col))
        means.append(np.array(
            [float(conf[0]) for conf in confounds]))
    means = np.array(means).T

    # Rotation and translation
    motion_confounds = []
    for col in range(5, 11):
        confounds = np.genfromtxt(confound_file, delimiter='\t', names=True,
                                  usecols=(col))
        motion_confounds.append(np.array(
            [float(conf[0]) for conf in confounds]))
    motion_confounds = np.array(motion_confounds).T

    # Relative motion
    relative_motion = motion_confounds.copy()
    relative_motion[1:] -= motion_confounds[:-1]
    mean_motions.append(np.linalg.norm(relative_motion[3:], axis=1).mean())

    # All confounds except motion
    no_motion_confounds = []
    for col in range(0, 5) + range(11, 17):
        confounds = np.genfromtxt(confound_file, delimiter='\t', names=True,
                                  usecols=(col))
        no_motion_confounds.append(np.array(
            [float(conf[0]) for conf in confounds]))
    no_motion_confounds = np.array(no_motion_confounds).T
    print("-- Computing region signals ...")
    site = dataset.phenotypic['site'][subject_n][1:-1]
    if 'Peking' in site:
        site = 'Peking'
    sites.append(site)
    if site == 'NeuroImage':
        t_r = 1.96
    elif site in ['KKI', 'OHSU']:
        t_r = 2.5
    else:
        t_r = 2.
    low_pass = .08
    high_pass = .009
    masker = nilearn.input_data.NiftiMapsMasker(
        atlas["maps"], resampling_target="maps", detrend=True,
        low_pass=low_pass, high_pass=high_pass, t_r=t_r, standardize=True,
        memory=mem, memory_level=1, verbose=1)
    masker_no_hf_filt = nilearn.input_data.NiftiMapsMasker(
        atlas["maps"], resampling_target="maps", detrend=True,
        low_pass=None, high_pass=high_pass, t_r=t_r, standardize=False,
        memory=mem, memory_level=1, verbose=1)
    region_ts = masker.fit_transform(filename,
                                     confounds=[means, relative_motion])
    if reorder:
        new_order = aud + striate + dmn + van + dan + ips + cing + basal + occ\
            + motor + vis + salience + temporal + language + cerebellum + dpcc

    if subject_n > n_subjects - 1 - max_outliers:
        noisy_ts = masker.fit_transform(
            filename, confounds=[no_motion_confounds])
        region_ts2 = noisy_ts
    else:
        region_ts2 = region_ts

    if reorder:
        region_ts = region_ts[:, new_order]
        region_ts2 = region_ts2[:, new_order]

    subjects.append(region_ts)
    subjects3.append(region_ts2)

noisy_list = []
for n_outliers in range(max_outliers):  # TODO max_outliers + 1 ?
    noisy_list.append(subjects[:n_subjects - n_outliers] +
                      subjects3[n_subjects - n_outliers:])

n_subjects = len(subjects)
import nilearn.connectivity
print("-- Measuring connecivity ...")
all_matrices = []
mean_matrices = []
all_matrices2 = []
mean_matrices2 = []
measures = ['covariance', 'precision', 'tangent', 'correlation',
            'partial correlation']
from nilearn.connectivity.embedding import (map_sym, cov_to_corr,
                                            prec_to_partial)
from sklearn.covariance import EmpiricalCovariance, LedoitWolf, MinCovDet
estimators = [('ledoit', LedoitWolf()), ('emp', EmpiricalCovariance()),
              ('mcd', MinCovDet())]
n_estimator = 1

# Without outliers
for measure in measures:
    estimator = {'cov_estimator': estimators[n_estimator][1],
                 'kind': measure}
    cov_embedding = nilearn.connectivity.CovEmbedding(**estimator)
    matrices = nilearn.connectivity.vec_to_sym(
        cov_embedding.fit_transform(subjects))
    all_matrices.append(matrices)
    if measure == 'tangent':
        mean = cov_embedding.mean_cov_
    else:
        mean = matrices.mean(axis=0)
    mean_matrices.append(mean)

# With outliers
noisy_means = []
noisy_matrices = []
for noisy_subjects in noisy_list:
    all_matrices2 = []
    mean_matrices2 = []
    for measure in measures:
        estimator = {'cov_estimator': estimators[n_estimator][1],
                     'kind': measure}
        cov_embedding = nilearn.connectivity.CovEmbedding(**estimator)
        matrices2 = nilearn.connectivity.vec_to_sym(
            cov_embedding.fit_transform(noisy_subjects))
        all_matrices2.append(matrices2)
        if measure == 'tangent':
            mean2 = cov_embedding.mean_cov_
        else:
            mean2 = matrices2.mean(axis=0)
        mean_matrices2.append(mean2)
    noisy_matrices.append(all_matrices2)
    noisy_means.append(mean_matrices2)

# Without outliers
covariances = all_matrices[0]
precisions = all_matrices[1]
tangents = all_matrices[2]
correlations = all_matrices[3]
partials = all_matrices[4]
Z_correlations = corr_to_Z(correlations)
Z_partials = corr_to_Z(partials, tol=1e-9)
Z_correlations[np.isinf(Z_correlations)] = 1.
Z_partials[np.isinf(Z_partials)] = 1.

# Plot the matrix of connectivity and distance between nodes
msdl_filepath = os.path.join('/home/sb238920/nilearn_data/msdl_atlas/',
                             'MSDL_rois/msdl_rois_labels.csv')
coords = np.genfromtxt(msdl_filepath, names=True, delimiter=',',
                       usecols=(0, 1, 2))
coords = np.array([list(coord) for coord in coords])
distance_matrix = coords - coords[:, np.newaxis]
distance_matrix = np.linalg.norm(distance_matrix, axis=-1) / 1000

# Explore the effect of motion on high correlations
# TODO: keep only statistically significant differences
fig_difference = plt.figure()
ax_difference = fig_difference.add_subplot(111)
fig_compare = plt.figure()
ax_compare = fig_compare.add_subplot(111)
threshold = - 1 - 0.2  # threshold to keep high correlations
dist_th = 0.12
for n in range(-1, - max_outliers, -4):
    conn2 = all_matrices2[3][n]  # TODO: Z-Fisher transform
    conn = all_matrices[3][n]
    diff = all_matrices2[3][n] - all_matrices[3][n]

    # Significant long-range correlations
    long_range_matrix = distance_matrix.copy()
    long_range_matrix[long_range_matrix < dist_th] = 0
    long_range_matrix[conn < threshold] = 0
    tri_mask = np.tril(long_range_matrix) > 0
    long_range_matrix[tri_mask] = diff[tri_mask]
    plot_matrix(long_range_matrix, title='subject {}'.format(n_subjects + n))

#    ax_compare.scatter(distance_matrix.flatten(), conn.flatten(), c='g')
#    ax_compare.scatter(distance_matrix.flatten(), conn2.flatten(), c='r')
    ax_compare.scatter(conn[distance_matrix > dist_th].flatten(),
                       conn2[distance_matrix > dist_th].flatten(), c='b')
    ax_compare.scatter(conn[distance_matrix < dist_th / 3].flatten(),
                       conn2[distance_matrix < dist_th / 3].flatten(), c='r')
    ax_compare.plot(conn.flatten(), conn.flatten())

    # TODO: 2 clouds with different colors, x: conn, y: conn2
    ax_difference.scatter(distance_matrix[conn > threshold].flatten(),
                          conn[conn > threshold].flatten())
    xgrid = np.linspace(diff[conn > threshold].min(),
                        diff[conn > threshold].max())
    ax_difference.plot(xgrid, np.zeros(xgrid.size))

ax_compare.set_xlabel('conn^{preproc}')
ax_compare.set_ylabel('conn^{noisy}')
ax_difference.set_xlabel('distances')
ax_difference.set_ylabel(r'conn^{noisy} - conn^{preproc}')
plt.show()

# Correlation between motion and connectivity, relationship with distance
mean_motion = []
for n in range(n_subjects):
    confound_file = dataset["confounds"][n]
    motion_confounds = []
    for col in range(8, 11):
        confounds = np.genfromtxt(confound_file, delimiter='\t', names=True,
                                  usecols=(col))
        motion_confounds.append(np.array(
            [float(conf[0]) for conf in confounds]))
    motion_confounds = np.array(motion_confounds).T
    print motion_confounds.mean(axis=0)
    mean_motion.append(np.linalg.norm(motion_confounds, axis=1).mean())

distance_matrix = coords - coords[:, np.newaxis]
distance_matrix = np.linalg.norm(distance_matrix, axis=-1) / 100
from scipy.stats import pearsonr
correlation = np.zeros(distance_matrix.shape)
all_indices = np.triu_indices(distance_matrix.shape[0], 1)
for indices in zip(*all_indices):
    conn = []
    for n in range(n_subjects):
        conn.append(all_matrices[3][n][indices])
    if np.mean(conn) > -1:
        correlation[indices] = pearsonr(mean_motion, conn)[0]

#distance_matrix[distance_matrix > .96] = 0
#correlation[distance_matrix > .96] = 0
dist = distance_matrix[all_indices]
corr = correlation[all_indices]
#dist = dist[np.abs(corr > 0)]
#corr = corr[np.abs(corr > 0)]
plt.scatter(dist, corr)
plt.xlabel('euclidean distance (mm)')
plt.ylabel('correlation of motion and connectivity')
r, p = pearsonr(dist, corr)
print('Pearson correlation is {0} with pval {1}'.format(r, p))
plt.show()

# TODO: computations (model) tacking into account 2 different observations
# related to 2 different preprocs, keep in mind we don't have access to noise!

# Statistical tests between noisy and less noisy matrices
baseline = all_matrices[3][n_subjects - max_outliers:n_subjects]
follow_up = all_matrices2[3][n_subjects - max_outliers:n_subjects]
signif_b, signif_f, signif_diff = matrix_stats.compare(baseline, follow_up,
                                                       axis=0, paired=True,
                                                       corrected=False)
matrix_stats._plot_matrices([signif_b, signif_f, signif_diff],
                            titles=['baseline', 'follow-up', 'difference'])

long_range_matrix = distance_matrix.copy()
long_range_matrix[long_range_matrix < dist_th] = 0
tri_mask = np.tril(long_range_matrix) > 0
long_range_matrix[tri_mask] = signif_diff[tri_mask]
plot_matrix(long_range_matrix,
            title='distance and difference between preprocs')

plt.show()

# Correlation between motion and connectivity, relationship with distance
#########################################################################
# Compute Euclidean distances between nodes in mm
print(' Reproducing Satterwaith results')
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
    correlation[indices] = pearsonr(mean_motions, conn)[0]
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
print('corr = {0} (dist - {1})'.format(t[0][0], - t[0][1] / t[0][0]))
xp = np.linspace(dist.min(), dist.max(), 100)
p1 = np.poly1d(np.polyfit(dist, corr, 1))
plt.plot(xp, p1(xp))
plt.show()
