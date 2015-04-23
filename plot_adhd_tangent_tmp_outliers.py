"""
Comparing different connectivity measures
=========================================

This example shows how to extract signals from regions defined by an atlas,
and to estimate different connectivity measures based on these signals.
"""
plot_conn = False
scatter_conn = False
plot_dist_mean = False
predict_sites = False
plot_relative_dist = False
overwrite = False
max_outliers = 5

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
        plt.title('subject ' + str(n_subject))
        n_subject += 1

    plt.title(title)


def scatterplot_matrix(coefs1, coefs2, coefs_ref, names,
                       title1='measure 1', title2='measure 2',
                       title_ref='reference measure'):
    """Plot a scatterplot matrix of subplots. Each connectivity coefficient is
    scatter plotted for two given measures against a reference measure. The
    line of best fit is plotted for significantly correlated coefficients. """
    n_subjects = coefs1.shape[0]
    n_coefs = coefs1[0].size
    fig, axes = plt.subplots(nrows=n_coefs, ncols=n_coefs, figsize=(8, 8))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    for ax in axes.flat:
        # Hide all ticks and labels
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        # Set up ticks only on one side for the "edge" subplots
        if ax.is_first_col():
            ax.yaxis.set_ticks_position('left')
        if ax.is_last_col():
            ax.yaxis.set_ticks_position('right')
        if ax.is_first_row():
            ax.xaxis.set_ticks_position('top')
        if ax.is_last_row():
            ax.xaxis.set_ticks_position('bottom')

    # Plot the coefs
    from scipy.stats.stats import pearsonr
    all_coefs = [coefs1, coefs2, coefs_ref, coefs_ref]
    colors = ['b', 'g', 'r', 'r']
    plots = []
    for n, coefs in enumerate(all_coefs):
        coefs = coefs.reshape(n_subjects, -1)
        for x, y in zip(*np.triu_indices_from(axes, k=1)):
            indices = [(x, y), (y, x), (x, y), (y, x)]
            id_x, id_y = indices[n]
            plot = axes[id_x, id_y].scatter(coefs[:, id_x], coefs[:, id_y],
                                            c=colors[n])
            # Plot line of best fit if significative Pearson correlation
            rho, p_val = pearsonr(coefs[:, id_x], coefs[:, id_y])
            if p_val < 0.05 / coefs.shape[-1]:
                fit = np.polyfit(coefs[:, id_x], coefs[:, id_y], 1)
                fit_fn = np.poly1d(fit)
                axes[id_x, id_y].plot(coefs[:, id_x], fit_fn(coefs[:, id_x]),
                                      linestyle='-', c=colors[n])
        plots.append(plot)

    plt.figlegend(plots, [title1, title2, title_ref],
                  loc='lower center', mode='expand', ncol=3, borderaxespad=0.2)

    # Label the diagonal subplots
    for i, label in enumerate(names):
        axes[i, i].annotate(label, (0.5, 0.5), xycoords='axes fraction',
                            ha='center', va='center')

    # Turn on the proper x or y axes ticks
    import itertools
    for i, j in zip(range(n_coefs), itertools.cycle((-1, 0))):
        axes[j, i].xaxis.set_visible(True)
        axes[i, j].yaxis.set_visible(True)
        xlabels = axes[j, i].get_xticklabels()
        plt.setp(xlabels, rotation=90)
    plt.tight_layout(pad=4.5, w_pad=0., h_pad=0.)
    fig.suptitle("connectivity scatter plots")

print("-- Fetching datasets ...")
import nilearn.datasets
atlas = nilearn.datasets.fetch_msdl_atlas()
dataset = nilearn.datasets.fetch_adhd()

import nilearn.image
import nilearn.input_data

import joblib
mem = joblib.Memory(".")

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
reorder = True
subjects3 = []
gms = []
import nibabel
from nilearn.masking import compute_epi_mask, apply_mask
for subject_n in range(n_subjects):
    filename = dataset["func"][subject_n]
    img = nibabel.load(filename)
    mask_img = compute_epi_mask(img)
    masked_data = apply_mask(img, mask_img)
    gms.append(np.mean(masked_data))
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
    if subject_n < max_outliers:
        region_ts2 = masker.fit_transform(filename,
                                          confounds=[hv_confounds])
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
for n_outliers in range(max_outliers):
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
#from sklearn.covariance import LedoitWolf  # ShrunkCovariance
from nilearn.connectivity import map_sym
from nilearn.connectivity.embedding import cov_to_corr, prec_to_partial
from sklearn.covariance import EmpiricalCovariance, ShrunkCovariance,\
    LedoitWolf, GraphLassoCV, MinCovDet
estimators = [('ledoit', LedoitWolf()), ('emp', EmpiricalCovariance)]
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

#############################################
# Plotting connectivity matrices
#############################################
print("-- Displaying results")
regions = ['L DMN', 'med DMN', 'front DMN', 'R DMN']
titles = [kind + ' mean'for kind in measures]
if plot_conn:
    for matrix, title in zip(mean_matrices, titles):
        plot_matrix(matrix, title=title, ticks=range(3, 7),
                    tick_labels=regions)

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

#############################################
# Scatter plotting connectivity coefs
#############################################
# Scatter plot connectivity coefficients between some regions of the Default
# Mode Network
if scatter_conn:
    scatterplot_matrix(Z_correlations[:, 3:5, 5:7],
                       Z_partials[:, 3:5, 5:7],
                       tangents[:, 3:5, 5:7],
                       names=['L DMN/\nfront DMN', 'L DMN/\nR DMN',
                              'med DMN/\nfront DMN', 'med DMN/\nR DMN'],
                       title1='correlation\n(Z-transformed)',
                       title2='partial correlation\n(Z-transformed)',
                       title_ref='tangent')
    plt.draw()

# Compute distance of the outliers to the set
dist = np.zeros((5, len(subjects)))
y = []
for sub_n in range(len(subjects)):
    y.append(subjects[sub_n].shape[0])
    for n, conns in enumerate([covariances, precisions, tangents,
                               correlations, partials]):
        if n == 2:
            dist[n, sub_n] = np.linalg.norm(tangents[sub_n])
        else:
            dist[n, sub_n] = np.linalg.norm(conns[sub_n] - np.mean(
                conns, axis=0))

#############################################
# Plotting distances to mean
#############################################
# Try also the distance to the median
# Positive results without removing the mean
# Removing the mean : tangent 60% one vs rest
if plot_dist_mean:
    plt.figure()
    lineObjects = plt.plot(np.arange(0, sub_n + 1), dist.T - np.ones(
        (n_subjects, 5)) * dist.mean(axis=1))
    plt.legend(iter(lineObjects), ('cov', 'prec', 'tang', 'corr', 'part'),
               loc=0)
    plt.title('distances to mean')
    plt.xlabel('subject id')


# Predict site
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.cross_validation import permutation_test_score, cross_val_score
from sklearn.preprocessing.label import LabelBinarizer
from sklearn import lda

if predict_sites:
    sites = np.repeat(np.arange(n_subjects / 8), 8)
    for dst, coefs, measure in zip([d for d in dist],
                                   [covariances, precisions, tangents,
                                    Z_correlations, Z_partials],
                                   measures):
        skf = StratifiedKFold(sites, 3)
        cv_scores_log = []
        cv_scores_ovr = []
        X = dst.copy()
        X = X[:, np.newaxis]
        y = sites
        print '=========================================='
        print measure
    #    clf_log = OneVsOneClassifier(LinearSVC(random_state=0))
    #    score = np.mean(cross_val_score(clf_log, X, y, cv=skf))
    #    score, null_scores, p_val = permutation_test_score(clf_log, X, y,
    #                                                       cv=skf,
    #   scoring='accuracy')
    #    print 'Variance score: ', score, ', null score is',
    #    np.mean(null_scores), 'pval = ', p_val
        clf_svc_ovr = LinearSVC(random_state=0, multi_class='ovr')
        score, null_scores, p_val = permutation_test_score(clf_svc_ovr, X, y,
                                                           cv=skf,
                                                           scoring='accuracy')
        print 'Variance score one vs rest: ', score, ', null score is',\
            np.mean(null_scores), 'pval = ', p_val
        np.mean(null_scores), '=', score
        # Predict ADHD
        u = [[5 * adhd_id, site_id] for (adhd_id, site_id) in zip(adhd, sites)]
        y = LabelBinarizer().fit_transform(u)
        y = adhd
        # DMN including RDLPFC, RTPJ and dorsal PCC
        extended_dmn = [3, 4, 5, 6, 9, 30, 33]
        regions = dmn + van + dan
        X = coefs[:, regions, :]
        X = coefs[:, :, regions]
        r = X.copy()
    #    X = nilearn.connectivity.embedding.sym_to_vec(X)
        X = np.zeros((n_subjects, len(regions)))
        for k, coef in enumerate(r):
            X[k] = np.diag(coef)
    #    X = np.hstack((X, dst[:, np.newaxis]))
        clf_lda = OneVsRestClassifier(lda.LDA())
        score, null_scores, p_val = permutation_test_score(clf_lda, X, y, cv=skf,
                                                           scoring='accuracy')
    #    print 'score ADHD: ', score, ', null score is', np.mean(null_scores), 'pval = ', p_val
    #    clf_log = OneVsOneClassifier(LinearSVC(random_state=0))
    #    score, null_scores, p_val = permutation_test_score(clf_log, X, y, cv=skf,
    #                                                       scoring='accuracy')
        print 'score ADHD: ', score, ', null score is', np.mean(null_scores), 'pval = ', p_val
        clf_svc_ovr = LinearSVC(random_state=0, multi_class='ovr')
        score, null_scores, p_val = permutation_test_score(clf_svc_ovr, X, y, cv=skf,
                                                           scoring='accuracy')
        print 'score ADHD: ', score, ', null score is', np.mean(null_scores), 'pval = ', p_val
        print '=========================================='

if plot_dist_mean:
    x = np.vstack((dist[2], dist[3], dist[4]))
    plt.figure()
    lineObjects = plt.plot(np.arange(0, sub_n + 1), x.T)
    plt.legend(iter(lineObjects), ('tang', 'Z-corr', 'Z-part'), loc=0)
    plt.title('distances to mean')
    plt.xlabel('subject id')
    plt.show()
    
    #for n, conns in enumerate([covariances, precisions, covariances,
    #                               Z_correlations, Z_partials]):
    #    eig_mins = [np.amin(np.linalg.eigvalsh(conn)) for conn in conns]
    #    eig_min = np.amin(np.linalg.eigvalsh(mean_matrices[n]))
    #    eig_maxs = [np.amax(np.linalg.eigvalsh(conn)) for conn in conns]
    #    eig_max = np.amax(np.linalg.eigvalsh(mean_matrices[n]))
    #    plt.figure()
    #    plt.scatter(eig_mins, eig_maxs)
    #    plt.scatter(eig_min, eig_max, color='r')
    #    plt.title(measures[n])
    #    plt.xlabel('eig min')
    #plt.show()
    
    n_samples = [subject.shape[0] for subject in subjects]
    plt.figure()
    colors = ['r', 'b', 'g']
    lineObjects = []
    for n, d in enumerate(x):
        lineObjects.append(plt.scatter(n_samples, d, color=colors[n]))
    plt.legend(iter(lineObjects), ('tang', 'corr', 'part'), loc=0)
    plt.title('distances to mean w.r.t. number of samples')
    plt.xlabel('n_samples')
    plt.ylabel('distance')
    plt.show()

    n_samples = [subjects[n].shape[0] for n in range(0, 40, 8)]
    plt.figure()
    lineObjects = []
    for n, d in enumerate(x):
        means = [d[8 * n_site: 8 * (n_site + 1)].mean() for n_site in range(5)]
        mins = [d[8 * n_site: 8 * (n_site + 1)].min() for n_site in range(5)]
        maxs = [d[8 * n_site: 8 * (n_site + 1)].max() for n_site in range(5)]
        lineObjects.append(plt.errorbar(n_samples, means, yerr=means,
                                        fmt='o', color=colors[n]))
    plt.legend(iter(lineObjects), ('tang', 'corr', 'part'), loc=0)
    plt.title('distances to mean w.r.t. number of samples')
    plt.xlabel('n_samples')
    plt.ylabel('distance')
    plt.draw()


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


for all_matrices2 in noisy_matrices:
    covariances2 = all_matrices2[0]
    precisions2 = all_matrices2[1]
    tangents2 = all_matrices2[2]
    correlations2 = all_matrices2[3]
    partials2 = all_matrices2[4]
    Z_correlations2 = corr_to_Z(correlations2)
    Z_partials2 = corr_to_Z(partials2, tol=1e-9)
    Z_correlations2[np.isinf(Z_correlations2)] = 1.
    Z_partials2[np.isinf(Z_partials2)] = 1.

    #############################################
    # Compute the distances between the matrices
    #############################################
    mat_dist = np.zeros((5, len(subjects) + 1, len(subjects) + 1))
    conns_all = [covariances2, precisions2, covariances2, correlations2,
                 partials2]
    conns_all = [np.vstack((conns, mean_matrices2[n][np.newaxis, ...])) for
        (n, conns) in enumerate(conns_all)]
    for n, conns in enumerate(conns_all):
        for sub_n in range(len(subjects) + 1):
            if n == 2:
                mat_dist[n, sub_n] = [np.linalg.norm(disp(conns[sub_n], conn))\
                    for conn in conns]
            else:
                mat_dist[n, sub_n] = [np.linalg.norm(conns[sub_n] - conn) \
                    for conn in conns]
        mask = mat_dist[n] < np.percentile(mat_dist[n], 90)
        mat_dist[n][mask] = 0.
        plot_matrix(mat_dist[n], measures[n], ticks=range(0, 41),
                    tick_labels=[str(tick) for tick in range(0, 41)])
        fig_title = measures[n] + '_' + estimators[n_estimator][0]
        filename = os.path.join(
            '/home/salma/slides/Parietal2/Images/robustness',
            fig_title + ".pdf")
        if not os.path.isfile(filename) or overwrite:
            pylab.savefig(filename)
            os.system("pdfcrop %s %s" % (filename, filename))

    # Compute the relative distances between the matrices
    if plot_relative_dist:
        rel_mat_dist = np.zeros((5, len(subjects), len(subjects)))
        for n, conns in enumerate([covariances, precisions, covariances,
                                   correlations, partials]):
            for sub_n in range(len(subjects)):
                if n == 2:
                    rel_mat_dist[n, sub_n] = [np.linalg.norm(disp(conns[sub_n],
                                 conn)) / np.linalg.norm(conns[sub_n]) for conn
                                 in conns]
                else:
                    rel_mat_dist[n, sub_n] = [np.linalg.norm(conns[sub_n] -
                        conn) / np.linalg.norm(conns[sub_n]) for conn in conns]
            plot_matrix(rel_mat_dist[n], measures[n])
        plt.show()
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
    # Compute the difference between contaminated and non data
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
    
    
    for n, conn_mean in enumerate(mean_matrices):
    #    plot_matrix(mean_matrices2[n], title=measures[n])
        print '=============================='
        print measures[n], 'norm diff', np.linalg.norm(
            conn_mean - mean_matrices2[n]), \
            ', norm diff after cov_to_corr', np.linalg.norm(
            cov_to_corr(conn_mean) - cov_to_corr(mean_matrices2[n])), \
            ', relative norm diff after cov_to_corr', np.linalg.norm(
            cov_to_corr(conn_mean) - cov_to_corr(mean_matrices2[n])) / \
            np.linalg.norm(cov_to_corr(conn_mean)), \
            ', relative norm diff', np.linalg.norm(
            conn_mean - mean_matrices2[n]) / np.linalg.norm(conn_mean), \
            ', disp', np.linalg.norm(disp(
            conn_mean, mean_matrices2[n])), \
            ', relative disp', np.linalg.norm(disp(
            conn_mean, mean_matrices2[n])) / np.linalg.norm(conn_mean), \
            ', max difference after cov_to_corr', np.amax(np.abs(
            cov_to_corr(conn_mean) - cov_to_corr(mean_matrices2[n]))), \
            ', max difference after prec_to_partial', np.amax(np.abs(
            prec_to_partial(np.linalg.inv(conn_mean)) - prec_to_partial(
            np.linalg.inv(mean_matrices2[n]))))

plt.figure()
colors = ['r', 'b', 'g', 'k']
ls = ['-', '--']
percent_outliers = [100. * n_outliers / n_subjects for n_outliers in
    range(max_outliers)]
percent_str = [str(percent) + '%' for percent in percent_outliers]
#for n, error in enumerate([cov_err, robust_cov_err]):
lineObjects = plt.plot(percent_outliers, cov_err, 'r-',
                       percent_outliers, robust_cov_err, 'b--')
#                            linestyle=ls[n % 2]))
lineObjects2 = []
for n, error in enumerate([cov_err, robust_cov_err,
                           prec_err, robust_prec_err,
                           corr_err, robust_corr_err,
                           part_err, robust_part_err]):
    lineObjects2 += plt.plot(percent_outliers, error, colors[n / 2],
                            linestyle=ls[n % 2])
plt.legend(iter(lineObjects2), ('mean covariances', 'gmean',
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
plt.show()