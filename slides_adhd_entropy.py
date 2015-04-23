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

    subjects.append(region_ts)


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


#############################################
# Plotting connectivity matrices
#############################################
print("-- Displaying results")
regions = ['L DMN', 'med DMN', 'front DMN', 'R DMN']
titles = [kind + ' mean'for kind in measures]

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

#############################
# Computing neural complexity
#############################
def compute_complexity(mat):
    """ Computes the neural complexity of a matrix"""
    import itertools
    from math import log

    n = len(mat)
    indices = range(n)
    complexity = 0.
    n_complexity = log(np.linalg.det(mat))
    for k in range(1, len(indices)):
        # take k regions out of indices, 1 <= k <= n - 1
        k_complexity = []
        for subset in itertools.combinations(indices, k):
            extracted_mat = mat[subset, :]
            extracted_mat = extracted_mat[:, subset]
            k_complexity.append(log(np.linalg.det(extracted_mat)))
        complexity += np.mean(k_complexity, axis=0) - n_complexity * k / n
    complexity *= .5
    return complexity

ntwks = dmn, van, language, van + dan
for ntwk in []:
    complexities = []
    for covariance in covariances:
        cov_extracted = covariance[ntwk, :]
        cov_extracted = cov_extracted[:, ntwk]
        complexities.append(compute_complexity(cov_extracted))
    mean_complexity = np.mean(complexities, axis=0)
    print "===========complexity errors ================"
    cov_extracted = mean_matrices[0][ntwk, :]
    cov_extracted = cov_extracted[:, ntwk]
    print np.abs(compute_complexity(cov_extracted) -\
        mean_complexity) / np.abs(mean_complexity)
    cov_extracted = mean_matrices[2][ntwk, :]
    cov_extracted = cov_extracted[:, ntwk]
    print np.abs(compute_complexity(cov_extracted) -\
        mean_complexity) / np.abs(mean_complexity)


#################################
# Computing entropies
#########################
van_dan = [9, 10, 11, 12, 14, 15, 16]
ntwks = [dmn, van, language, van + dan, van_dan]
n_regions = len(mean_matrices[0])
covs = [cov for cov in covariances]
covs = covs + [mean_matrices[0]] + [mean_matrices[2]]
for ntwk in [dmn]:
    entropies = []
    cond_entropies = []
    segregations = []
    cond_segregations = []
    complexities = []
    comp_ntwk = language  # [k for k in range(n_regions) if not k in ntwk]
    for n_subject, covariance in enumerate(covs):
        cov_extracted = covariance[ntwk, :]
        cov_extracted = cov_extracted[:, ntwk]
        # Complexity
        complexities.append(compute_complexity(cov_extracted))

        # Marginal entropies and segregations
        entropy = np.log(np.linalg.det(cov_extracted))
        entropies.append(entropy)
        cov_extracted = covariance[comp_ntwk, :]
        cov_extracted = cov_extracted[:, comp_ntwk]
        comp_entropy = np.log(np.linalg.det(cov_extracted))
        cov_extracted = covariance[ntwk + comp_ntwk, :]
        cov_extracted = cov_extracted[:, ntwk + comp_ntwk]
        global_entropy = np.log(np.linalg.det(cov_extracted))
        segregations.append(entropy + comp_entropy - global_entropy)

        # Conditional entropies and segregations
        precision = np.linalg.inv(covariance)
        prec_extracted = precision[ntwk, :]
        prec_extracted = prec_extracted[:, ntwk]
        entropy = np.log(np.linalg.det(prec_extracted))
        cond_entropies.append(entropy)
        prec_extracted = precision[comp_ntwk, :]
        prec_extracted = prec_extracted[:, comp_ntwk]
        comp_entropy = np.log(np.linalg.det(prec_extracted))
        prec_extracted = precision[ntwk + comp_ntwk, :]
        prec_extracted = prec_extracted[:, ntwk + comp_ntwk]
        global_entropy = np.log(np.linalg.det(prec_extracted))
        cond_segregations.append(entropy + comp_entropy - global_entropy)

    mean_complexity = np.mean(complexities[:n_subjects], axis=0)
    mean_entropy = np.mean(entropies[:n_subjects], axis=0)
    mean_cond_entropy = np.mean(cond_entropies[:n_subjects], axis=0)
    mean_segregation = np.mean(segregations[:n_subjects], axis=0)
    mean_cond_segregation = np.mean(cond_segregations[:n_subjects],
                                    axis=0)
    plt.figure()
    lineObjects = plt.plot(0.5 * np.array(entropies[:n_subjects]), 'r',
              0.5 * entropies[n_subjects + 1] * np.ones(n_subjects), 'r--',
              - 0.5 * np.array(cond_entropies[:n_subjects]), 'm',
              - 0.5 * cond_entropies[n_subjects + 1] * np.ones(n_subjects), 'm--')
    plt.legend(iter(lineObjects), ('individual entropies',
               'entropy of gmean',
               'individual conditional entropies',
               'conditional entropy of gmean'),
               loc=0)
#    plt.title('integrations and conditional integrations')
    plt.xlabel('subject id')
    plt.ylabel('entropies')
#    lineObjects.set_ylim([-18, -10])
    filename = os.path.join(
        '/home/salma/slides/Parietal2014/Images/entropy',
        "entropies.pdf")
    if not os.path.isfile(filename) or True:
        pylab.savefig(filename)
        os.system("pdfcrop %s %s" % (filename, filename))
    plt.figure()
    lineObjects = plt.plot(- np.array(segregations[:n_subjects]), 'r',
             - segregations[n_subjects] * np.ones(n_subjects), 'r',
             - segregations[n_subjects + 1] * np.ones(n_subjects), 'r--',
             - np.array(cond_segregations[:n_subjects]), 'm',
             - cond_segregations[n_subjects] * np.ones(n_subjects), 'm',
             - cond_segregations[n_subjects + 1] * np.ones(n_subjects), 'm--')
    plt.legend(iter(lineObjects), ('individual segregations',
               'segregation of amean', 'segregation of gmean',
               'individual conditional segregations',
               'conditional segregation of amean',
               'conditional segregation of gmean'),
               loc=0)
    plt.title('segregations and conditional segregations')
    plt.xlabel('subject id')
    plt.ylabel('segregations')

    plt.figure()
    plt.boxplot([- np.array(entropies[:n_subjects]),
                 - np.array(cond_entropies[:n_subjects])])
    lineObjects = plt.plot(1., - entropies[n_subjects + 1], color='g') + \
        plt.plot(2., - cond_entropies[n_subjects + 1], 'o', color='g')
    plt.xticks(np.arange(2) + 1., ['integrations', 'conditional integrations'],
               size=8)
    plt.legend(iter(lineObjects), ('integration of gmean',
               'conditional integration of gmean'))
    
plt.show()

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


