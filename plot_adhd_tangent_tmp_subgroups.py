"""
Comparing different connectivity measures
=========================================

This example shows how to extract signals from regions defined by an atlas,
and to estimate different connectivity measures based on these signals.
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

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
    size = mean_conn.shape[0]
    mean_conn[range(size), range(size)] = 0
    vmax = np.abs(mean_conn).max()
    if vmax <= 2e-16:
        vmax = 0.1

    # Display connectivity matrix
    plt.figure()
    plt.imshow(mean_conn, interpolation="nearest",
              vmin=-vmax, vmax=vmax, cmap=plt.cm.get_cmap("bwr"))
    plt.colorbar()
    ax = plt.gca()
    ax.xaxis.set_ticks_position('top')
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
mem = joblib.Memory("/home/salma/CODE/Parietal/nilearn/joblib/nilearn/"\
    "adhd/filtering")

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
adhd = dataset.phenotypic['adhd']
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
    t_r = 2.5
    low_pass = .08
    high_pass = .009
    masker = nilearn.input_data.NiftiMapsMasker(
        atlas["maps"], resampling_target="maps", detrend=True,
        low_pass=low_pass, high_pass=high_pass, t_r=t_r, standardize=False,
        memory=mem, memory_level=1, verbose=1)
    region_ts = masker.fit_transform(filename,
                                     confounds=[hv_confounds, confound_file])
    subjects.append(region_ts)
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

import nilearn.connectivity
print("-- Measuring connecivity ...")
all_matrices = []
mean_matrices = []
all_matrices2 = []
measures = ['covariance', 'precision', 'tangent', 'correlation',
            'partial correlation']
from sklearn.covariance import LedoitWolf  # ShrunkCovariance
from nilearn.connectivity import map_sym

local_gmeans = []
for measure in measures:
    estimator = {'kind': measure}  #, 'cov_estimator': LedoitWolf()
    cov_embedding = nilearn.connectivity.CovEmbedding(**estimator)
    matrices = nilearn.connectivity.vec_to_sym(
        cov_embedding.fit_transform(subjects))
    all_matrices.append(matrices)
    if measure == 'tangent':
        mean = cov_embedding.mean_cov_
    else:
        mean = matrices.mean(axis=0)
    mean_matrices.append(mean)
    matrices2 = np.zeros(matrices.shape)
    for n_site, subjects_site in enumerate([kki, neuro, nyu, ohsu, peking]):
        cov_embedding = nilearn.connectivity.CovEmbedding(**estimator)
        matrices_site = nilearn.connectivity.vec_to_sym(
            cov_embedding.fit_transform(subjects_site))
        if measure == 'tangent':
            local_mean = cov_embedding.mean_cov_
            local_gmeans.append(local_mean)
#            whitening = map_sym(lambda x: 1. / np.sqrt(x), local_mean)
#            matrices2[8 * n_site:8 * (n_site + 1), ...] = np.array(
#                [map_sym(np.log, whitening.dot(map_sym(np.exp, matrix_site)
#                ).dot(whitening)) for matrix_site in matrices_site])
            matrices2[8 * n_site:8 * (n_site + 1), ...] = matrices_site
        else:
            local_mean = matrices_site.mean(axis=0)
            matrices2[8 * n_site:8 * (n_site + 1), ...] = matrices_site - \
                local_mean
        site = dataset.phenotypic['site'][8 * n_site][1:-1]
#        plot_matrix(local_mean, title=measure + ' ' + site)
#        plt.show()
    all_matrices2.append(matrices2)

print("-- Displaying results")
regions = ['L DMN', 'med DMN', 'front DMN', 'R DMN']
titles = [kind + ' mean'for kind in measures]
for matrix, title in zip(mean_matrices, titles):
    plot_matrix(matrix, title=title, ticks=range(3, 7), tick_labels=regions)

covariances = all_matrices[0]  # corr_to_Z(all_matrices[0])
precisions = all_matrices[1]  # corr_to_Z(all_matrices[1])
tangents = all_matrices[2]
Z_correlations = corr_to_Z(all_matrices[3])
Z_partials = corr_to_Z(all_matrices2[4], tol=1e-9)
Z_correlations[np.isinf(Z_correlations)] = 1.
Z_partials[np.isinf(Z_partials)] = 1.

covariances2 = all_matrices2[0]  # corr_to_Z(all_matrices[0])
precisions2 = all_matrices2[1]  # corr_to_Z(all_matrices[1])
tangents2 = all_matrices2[2]
Z_correlations2 = corr_to_Z(all_matrices2[3])
Z_partials2 = corr_to_Z(all_matrices2[4], tol=1e-9)
Z_correlations2[np.isinf(Z_correlations2)] = 1.
Z_partials2[np.isinf(Z_partials2)] = 1.

# Scatter plot connectivity coefficients between some regions of the Default
# Mode Network
scatterplot_matrix(Z_correlations[:, 3:5, 5:7],
                   Z_partials[:, 3:5, 5:7],
                   tangents[:, 3:5, 5:7],
                   names=['L DMN/\nfront DMN', 'L DMN/\nR DMN',
                          'med DMN/\nfront DMN', 'med DMN/\nR DMN'],
                   title1='correlation\n(Z-transformed)',
                   title2='partial correlation\n(Z-transformed)',
                   title_ref='tangent')
plt.show()

# Compute distance of the outliers to the set
dist = np.zeros((5, len(subjects)))
y = []
for sub_n in range(len(subjects)):
    y.append(subjects[sub_n].shape[0])
    dist[0, sub_n] = np.linalg.norm(covariances[sub_n])
    dist[1, sub_n] = np.linalg.norm(precisions[sub_n])
    dist[2, sub_n] = np.linalg.norm(tangents[sub_n])
    dist[3, sub_n] = np.linalg.norm(Z_correlations[sub_n])
    dist[4, sub_n] = np.linalg.norm(Z_partials[sub_n])

# Positive results without removing the mean
# Removing the mean : tangent 60% one vs rest
plt.figure()
lineObjects = plt.plot(np.arange(0, sub_n + 1), dist.T - np.ones((40, 5)) * dist.mean(axis=1))
plt.legend(iter(lineObjects),('cov', 'prec', 'tang', 'Z-corr', 'Z-part'), loc=0)
plt.title('distances to mean')
plt.xlabel('subject id')
plt.show()

covariances = all_matrices2[0]  # corr_to_Z(all_matrices[0])
precisions = all_matrices2[1]  # corr_to_Z(all_matrices[1])
tangents = all_matrices2[2]
correlations = all_matrices2[3]
partials = all_matrices2[4]
Z_correlations = corr_to_Z(all_matrices2[3])
Z_partials = corr_to_Z(all_matrices2[4], tol=1e-9)
Z_correlations[np.isinf(Z_correlations)] = 1.
Z_partials[np.isinf(Z_partials)] = 1.

# Predict site
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.cross_validation import permutation_test_score, cross_val_score
from sklearn.preprocessing.label import LabelBinarizer
from sklearn import lda
sites = np.repeat(np.arange(5), 8)
for dst, coefs, measure in zip([d for d in dist],
                               [covariances2, precisions2, tangents2, Z_correlations2, Z_partials2],
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
#    score, null_scores, p_val = permutation_test_score(clf_log, X, y, cv=skf,
#                                                       scoring='accuracy')
#    print 'Variance score: ', score, ', null score is', np.mean(null_scores), 'pval = ', p_val
    clf_svc_ovr = LinearSVC(random_state=0, multi_class='ovr')
    score, null_scores, p_val = permutation_test_score(clf_svc_ovr, X, y, cv=skf,
                                                       scoring='accuracy')
    print 'Variance score one vs rest: ', score, ', null score is', np.mean(null_scores), 'pval = ', p_val
    np.mean(null_scores), '=', score
    # Predict ADHD
    u = [[5 * adhd_id, site_id] for (adhd_id, site_id) in zip(adhd, sites)]
    y = LabelBinarizer().fit_transform(u)
    y = adhd
    skf = StratifiedKFold(adhd)
    striate = [2]
    dmn = [3, 4, 5, 6]
    extended_dmn = [3, 4, 5, 6, 9, 30, 33]  # including RDLPFC, RTPJ and dorsal PCC
    van = [9, 10, 11, 12, 14, 15, 16]
    dan = [17, 18]
    basal = [15]
    salience = [22, 23, 24]
    temporal = [25, 26]
    cerebellum = [32]
    dpcc = [33]
    insula = [34, 35, 36]
    regions = dmn
    X = coefs[:, regions, :]
    X = coefs[:, :, regions]
    X = nilearn.connectivity.embedding.sym_to_vec(X)
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
    score, null_scores, p_val = permutation_test_score(clf_svc_ovr, X, y, cv=3,
                                                       scoring='accuracy')
    print 'score ADHD: ', score, ', null score is', np.mean(null_scores), 'pval = ', p_val
    print '=========================================='

dist[1, :] = 0
plt.figure()
lineObjects = plt.plot(np.arange(0, sub_n + 1), dist.T)
plt.legend(iter(lineObjects),('cov', 'prec', 'tang', 'Z-corr', 'Z-part'), loc=0)
plt.title('distances to mean')
plt.xlabel('subject id')
plt.show()

from nilearn.connectivity.embedding import cov_to_corr
for n_site in range(5):
    site = dataset.phenotypic['site'][8 * n_site][1:-1]
    plot_matrix(cov_to_corr(local_gmeans[n_site]), title=site)
plt.show()