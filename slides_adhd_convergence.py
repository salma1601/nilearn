"""
Comparing different connectivity measures
=========================================

This example shows how to extract signals from regions defined by an atlas,
and to estimate different connectivity measures based on these signals.
"""
overwrite = True
min_included = 5
max_included = 40

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

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


def plot_matrices(conns, title="connectivity", ticks=[], tick_labels=[]):
    """Plot connectivity matrices, for a given measure. """

    conns = conns.copy()

    # Put zeros on the diagonal, for graph clarity.
    n_subjects = conns.shape[0]
    ncols = 4
    n_subject = 0
    plt.figure()
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
        plt.imshow(conn, interpolation="nearest",
                  vmin=-vmax, vmax=vmax, cmap=plt.cm.get_cmap("bwr"))
        plt.colorbar()
        ax = plt.gca()
        ax.xaxis.set_ticks_position('top')
        plt.xticks(ticks, tick_labels, size=8, rotation=90)
        plt.yticks(ticks, tick_labels, size=8)
        ax.yaxis.tick_left()
        plt.xlabel('subject ' + str(n_subject))
        n_subject += 1

    plt.title(title)

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
kki = []
neuro = []
nyu = []
ohsu = []
peking = []

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
reorder = True
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
    if subject_n < 8:
        kki.append(region_ts)
    elif subject_n < 16:
        neuro.append(region_ts)
    elif subject_n < 24:
        nyu.append(region_ts)
    elif subject_n < 32:
        ohsu.append(region_ts)
    else:
        peking.append(region_ts)

    subjects.append(region_ts)

subsignals_list1 = []
subsignals_list2 = []
subsignals_list3 = []
from sklearn.utils import resample
n_samples_all = []
for signals in [neuro + kki, neuro + kki + ohsu,
                neuro + kki + ohsu + peking, subjects]:
    subsignals1 = []
    subsignals2 = []
    subsignals3 = []
    n_samples_hetero = []
    for n_included in range(min_included, max_included + 1):
        n_samples = min(n_included, len(signals))
        n_samples_hetero.append(n_samples)
        signals_shuff = resample(signals, replace=False, n_samples=n_samples,
                            random_state=0)
        subsignals1.append(signals_shuff)
        signals_shuff = resample(signals, replace=False, n_samples=n_samples,
                            random_state=1)
        subsignals2.append(signals_shuff)
        signals_shuff = resample(signals, replace=False, n_samples=n_samples,
                            random_state=2)
        subsignals3.append(signals_shuff)
    n_samples_all.append(n_samples_hetero)
    subsignals_list1.append(subsignals1)
    subsignals_list2.append(subsignals2)
    subsignals_list3.append(subsignals3)

import nilearn.connectivity
print("-- Measuring connecivity ...")

#from sklearn.covariance import LedoitWolf  # ShrunkCovariance
from nilearn.connectivity import map_sym
from nilearn.connectivity.embedding import cov_to_corr, prec_to_partial
from sklearn.covariance import EmpiricalCovariance, ShrunkCovariance,\
    LedoitWolf, GraphLassoCV, MinCovDet
estimators = [('ledoit', LedoitWolf()), ('emp', EmpiricalCovariance()),
              ('mcd', MinCovDet())]
estimators = [('emp', EmpiricalCovariance())]
colors = ['r', 'g', 'b', 'm']

for (name, cov_estimator) in estimators:
    print(name)
    lineObjects = []
    n_all1 = []
    for n_hetero, subsignals in enumerate(subsignals_list1):
        n_iterations1 = []
        for subjects in subsignals:
            estimator = {'cov_estimator': cov_estimator, 'kind': 'covariance'}
            cov_embedding = nilearn.connectivity.CovEmbedding(**estimator)
            matrices = nilearn.connectivity.vec_to_sym(
                cov_embedding.fit_transform(subjects))
            grad_norm = nilearn.connectivity.embedding.grad_geometric_mean(
                matrices)
            n_iterations1.append(len(grad_norm))
        n_all1.append(n_iterations1)
    print n_all1
    print "------------"
    n_all2 = []
    for n_hetero, subsignals in enumerate(subsignals_list2):
        n_iterations2 = []
        for subjects in subsignals:
            estimator = {'cov_estimator': cov_estimator, 'kind': 'covariance'}
            cov_embedding = nilearn.connectivity.CovEmbedding(**estimator)
            matrices = nilearn.connectivity.vec_to_sym(
                cov_embedding.fit_transform(subjects))
            grad_norm = nilearn.connectivity.embedding.grad_geometric_mean(
                matrices)
            n_iterations2.append(len(grad_norm))
        n_all2.append(n_iterations2)
    print n_all2
    print'-------------'
    n_all = np.mean([n_all1, n_all2], axis=0)
    lineObjects = [plt.plot(n_samples_all[n],
                            n_iterations, colors[n])[0]
                            for n, n_iterations in enumerate(n_all)]
    plt.legend(iter(lineObjects), ('2 sites', '3 sites', '4 sites', '5 sites'),
               loc=0)
    plt.title('convergence for ' + name + ' estimator')
    plt.xlabel('number of input matrices')
    plt.xticks(range(min_included, max_included + 1, 5))
    plt.ylabel('number of iterations')
    axes = plt.gca()
    ymin, ymax = axes.get_ylim()
    axes.set_ylim([ymin - 1, ymax + 1])
    fig_title = 'iters_number_' + name
    filename = os.path.join(
        '/home/salma/slides/Parietal2/Images/convergence',
        fig_title + ".pdf")
    if not os.path.isfile(filename) or overwrite:
        pylab.savefig(filename)
        os.system("pdfcrop %s %s" % (filename, filename))
plt.show()