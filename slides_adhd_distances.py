"""
Comparing different connectivity measures
=========================================

This example shows how to extract signals from regions defined by an atlas,
and to estimate different connectivity measures based on these signals.
"""
overwrite = False
max_outliers = 0

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
# TODO : to remove the next imports
from matplotlib import pylab
import os

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

    # Put zeros on the diagonal, for graph clarity
#    size = mean_conn.shape[0]
#    mean_conn[range(size), range(size)] = 0
    vmax = np.abs(mean_conn).max()
    if vmax <= 2e-16:
        vmax = 0.1

    # Display connectivity matrix
    plt.figure()
    plt.imshow(mean_conn, interpolation="nearest",
              vmin=-vmax, vmax=vmax, cmap=plt.cm.get_cmap("bwr"))
    plt.colorbar()
    ax = plt.gca()
#    ax.xaxis.set_ticks_position('top')
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
        while  int_abs_max == 0:
            r = r + 2
            int_abs_max = round(vmax, r)
        cb_label_size = max(1, 8 - r)
        cbticks = [-vmax, 0, vmax]
        #    print '%f et %f pour %s' % (abs_max,int_abs_max, title)
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

mem = joblib.Memory("/home/salma/CODE/Parietal/nilearn/joblib/nilearn/adhd/filtering")


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
for subject_n in range(n_subjects):
    filename = dataset["func"][subject_n]
    print("Processing file %s" % filename)

    print("-- Computing confounds ...")
    confound_file = dataset["confounds"][subject_n]
    hv_confounds = mem.cache(nilearn.image.high_variance_confounds)(filename)

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
#    t_r = 2.5
    low_pass = .08
    high_pass = .009
    masker = nilearn.input_data.NiftiMapsMasker(
        atlas["maps"], resampling_target="maps", detrend=True,
        low_pass=low_pass, high_pass=high_pass, t_r=t_r, standardize=False,
        memory=mem, memory_level=1, verbose=1)
    region_ts = masker.fit_transform(filename,
                                     confounds=[hv_confounds, confound_file])
    if reorder:
        new_order = aud + striate + dmn + van + dan + ips + cing + basal + occ\
            + motor + vis + salience + temporal + language + cerebellum + dpcc
        region_ts = region_ts[:, new_order]

    if subject_n < max_outliers:
        region_ts2 = masker.fit_transform(filename,
                                          confounds=[hv_confounds])
        if reorder:
            region_ts2 = region_ts2[:, new_order]
    else:
        region_ts2 = region_ts

    subjects.append(region_ts)
    subjects3.append(region_ts2)
    adhd2.append(adhd[subject_n])
    if subject_n < 8:
        kki.append(region_ts)
    elif subject_n < 16:
        neuro.append(region_ts)
    elif subject_n < 24:
        nyu.append(region_ts)
    elif subject_n < 32:
        ohsu.append(region_ts)
        adhd2.pop(24)
    else:
        peking.append(region_ts)

subjects2 = kki + neuro + nyu + peking

subjects_neuro = [subject[:152] for subject in neuro] + \
    [subject for subject in neuro] + [subject[:176] for subject in neuro] + \
    [subject[:78] for subject in neuro] + [subject[:236] for subject in neuro]

subjects_tmp = [subject[:124] for subject in kki] + \
    [subject[:78] for subject in neuro] + [subject[:78] for subject in nyu] + \
    [subject[:78] for subject in ohsu] + [subject[:78] for subject in peking]

noisy_list = []
for n_outliers in range(max_outliers + 1):
    noisy_list.append(subjects3[:n_outliers] + subjects[n_outliers:])

import copy
rand_gen = np.random.RandomState(0)
outliers = []
n_outliers = 0
for k in range(n_outliers):
    subject = rand_gen.randn(200, 39)
    subject -= subject.mean(axis=0)
    outliers.append(subject)

subjects2 = copy.copy(subjects) + outliers
# Remove extreme subjects for precision measure
subjects2.pop(34)
# Remove extreme subjects for correlation measure
subjects2.pop(26)
subjects2.pop(21)
subjects2.pop(14)
n_subjects = len(subjects)
import nilearn.connectivity
print("-- Measuring connecivity ...")
all_matrices = []
mean_matrices = []
all_matrices2 = []
mean_matrices2 = []
measures = ['covariance', 'precision', 'tangent', 'correlation',
            'partial correlation']
from nilearn.connectivity import map_sym
from nilearn.connectivity.embedding import cov_to_corr, prec_to_partial
from sklearn.covariance import EmpiricalCovariance, ShrunkCovariance,\
    LedoitWolf, GraphLassoCV, MinCovDet
estimators = [('ledoit', LedoitWolf()), ('emp', EmpiricalCovariance()),
              ('mcd', MinCovDet())]
n_estimator = 0

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


def disp(A, B):
    A_sqrt_inv = map_sym(lambda x: 1. / np.sqrt(x), A)
    return map_sym(np.log, A_sqrt_inv.dot(B).dot(A_sqrt_inv))


# Robustness
def relative_max_norm(array, ref_array):
    """ Computes the relative max norm
    array : numpy.array
        the input array
    ref_array : numpy.array
        the reference array
    Returns
    norm : float
        the relative max norm
    """
    diff = np.abs(array - ref_array)
    idx = np.where(diff == np.amax(diff))
    quotient = np.amax(np.abs(ref_array[idx]))
    if quotient < 1e-7:
        quotient = np.amax(np.abs(array[idx]))

    return np.amax(diff) / quotient


for n_out, all_matrices2 in enumerate(noisy_matrices):
    covariances2 = all_matrices2[0]
    precisions2 = all_matrices2[1]
    tangents2 = all_matrices2[2]
    correlations2 = all_matrices2[3]
    partials2 = all_matrices2[4]
    Z_correlations2 = corr_to_Z(correlations2)
    Z_partials2 = corr_to_Z(partials2, tol=1e-9)
    Z_correlations2[np.isinf(Z_correlations2)] = 1.
    Z_partials2[np.isinf(Z_partials2)] = 1.

    # Plot the individual connectivty matrices
    for n_meas, measure in enumerate(measures):
        for n_site in range(5):
            subj_idx = range(n_site * 8, (n_site + 1) * 8)
            subtitles = ['subject ' + str(n_sub) for n_sub in subj_idx]
            fig_title = 'indiv_' + measure + '_' + sites[n_site * 8] + '_' +\
                estimators[n_estimator][0]
            filename = os.path.join(
                '/home/salma/slides/Parietal2/Images/robustness',
                fig_title + ".pdf")
            if not os.path.isfile(filename) or overwrite:
                plot_matrices(all_matrices2[n_meas][subj_idx, ...],
                              title=measure, subtitles=subtitles)
                pylab.savefig(filename)
                os.system("pdfcrop %s %s" % (filename, filename))

    #############################################
    # Compute the distances between the matrices
    #############################################
    mat_dist = np.zeros((5, len(subjects), len(subjects)))
    conns_all = [covariances2, precisions2, covariances2, correlations2,
                 partials2]
#    conns_all = [np.vstack((conns, mean_matrices2[n][np.newaxis, ...])) for
#        (n, conns) in enumerate(conns_all)]
    for n, conns in enumerate(conns_all):
        for sub_n in range(len(subjects)):
            if n == 2:
                mat_dist[n, sub_n] = [np.linalg.norm(disp(conns[sub_n], conn))\
                    for conn in conns]
            else:
                mat_dist[n, sub_n] = [np.linalg.norm(conns[sub_n] - conn) \
                    for conn in conns]
        percent = 95
        mask = mat_dist[n] < np.percentile(mat_dist[n], percent)
#        mat_dist[n][mask] = 0.
        if n_out < 3:
            plot_matrix(mat_dist[n], measures[n], ticks=range(0, 41),
                        tick_labels=[str(tick) for tick in range(0, 41)])
            fig_title = str(n_out) + 'outliers_' + str(percent) + '_percent' +\
                measures[n] + '_' + estimators[n_estimator][0]
            filename = os.path.join(
                '/home/salma/slides/Parietal2/Images/robustness',
                fig_title + ".pdf")
            if not os.path.isfile(filename) or overwrite:
                pylab.savefig(filename)
                os.system("pdfcrop %s %s" % (filename, filename))

plt.show()

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
    robust_corr = cov_to_corr(mean_matrices[2])
    cov_err.append(np.linalg.norm(mean_matrices[0] - mean_matrices2[0]) /\
        np.linalg.norm(mean_matrices[0]))
    robust_cov_err.append(np.linalg.norm(mean_matrices[2] -
        mean_matrices2[2]) / np.linalg.norm(mean_matrices[2]))
    prec_err.append(np.linalg.norm(mean_matrices[1] - mean_matrices2[1]) /\
        np.linalg.norm(mean_matrices[1]))
    robust_prec_err.append(np.linalg.norm(robust_prec - np.linalg.inv(
        mean_matrices2[2])) / np.linalg.norm(robust_prec))
    corr_err.append(np.linalg.norm(mean_matrices[3] - mean_matrices2[3]) /\
        np.linalg.norm(mean_matrices[3]))
    robust_corr_err.append(np.linalg.norm(robust_corr - cov_to_corr(
        mean_matrices2[2])) / np.linalg.norm(robust_corr))
    part_err.append(np.linalg.norm(mean_matrices[4] - mean_matrices2[4]) /\
        np.linalg.norm(mean_matrices[4]))
    robust_part_err.append(np.linalg.norm(robust_part - prec_to_partial(
        np.linalg.inv(mean_matrices2[2]))) / np.linalg.norm(robust_part))

plt.figure()
colors = ['r', 'b', 'g', 'k']
ls = ['-', '--']
percent_outliers = [100. * n_outliers / n_subjects for n_outliers in
    range(max_outliers)]
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
plt.ylabel('relative error in Froebenus norm')
fig_title = 'foeb_relative_err_' + estimators[n_estimator][0]
filename = os.path.join(
    '/home/salma/slides/Parietal2/Images/robustness',
    fig_title + ".pdf")
if not os.path.isfile(filename) or overwrite:
    pylab.savefig(filename)
    os.system("pdfcrop %s %s" % (filename, filename))

from scipy.stats import pearsonr

for (n1, n2) in [(2, 3), (2, 4), (3, 4)]:
    plt.figure()
    x = np.triu(mat_dist[n1]).ravel()
    y = np.triu(mat_dist[n2]).ravel()
    x = x[x > 1e-7]
    y = y[y > 1e-7]
    plt.scatter(x, y)
    plt.xlabel(measures[n1])
    plt.ylabel(measures[n2])
#    plt.title('distances scatter for ' + name)
    # Plot line of best fit if significative Pearson correlation
    rho, p_val = pearsonr(x, y)
    if p_val < 0.01:
        plt.text(.5, .5, str(p_val))
        fit = np.polyfit(x, y, 1)
        fit_fn = np.poly1d(fit)
        plt.plot(x, fit_fn(x), linestyle='-', c='red')

plt.show()