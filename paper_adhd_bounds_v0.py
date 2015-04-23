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
mem = joblib.Memory("/home/sb238920/CODE/Parietal/nilearn/joblib/nilearn/adhd/filtering")

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
estimators = [('ledoit', LedoitWolf()), ('emp', EmpiricalCovariance())]
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
larger_excep = [(8, 35), (19, 32)]  # mean(part) < mean(corr) < corr(gmean)
lower_excep = [(18, 36), (19, 25)]  # mean(part) > mean(corr) > corr(gmean)
lower = [(0, 1), (3, 4), (3, 5), (4, 5), (5, 6), (37, 38)]  # mean(part) < corr(gmean) < mean(corr)
larger = [(6, 35), (4, 35), (3, 35)]  # mean(corr) < corr(gmean) < mean(part)
# Find regions
n_regions = mean_matrices[0].shape[-1]
for x, y in zip(*np.triu_indices(n_regions, k=1)):
    if mean_matrices[3][x, y] > mean_matrices[4][x, y] + 0.2:
        larger.append((x, y))
    if mean_matrices[4][x, y] > mean_matrices[3][x, y] + 0.2:
        lower.append((x, y))

lower = []
larger = []
percent = .90
prop = percent * n_subjects
for x, y in zip(*np.triu_indices(n_regions, k=1)):
    if np.sum(all_matrices[3][:, x, y] > all_matrices[4][:, x, y] + .1) > prop:
        larger.append((x, y))
    if np.sum(all_matrices[4][:, x, y] > all_matrices[3][:, x, y] + .05) > prop:
        lower.append((x, y))
larger = larger[:3]
for regions in larger:
    coefs = []
    mean_coef = []
    for mean, matrices in zip([cov_to_corr(mean_matrices[2]),
                               mean_matrices[4]],
                              [all_matrices[3], all_matrices[4]]):
        coefs.append(matrices[:, regions[0], regions[1]])
        amean = mean_matrices[3][regions]
        mean_coef.append(mean[regions])
    plt.figure()
    plt.boxplot(coefs, whis=np.inf)
    lineObjects = plt.plot(1., mean_coef[0], '^', color='r') + \
        plt.plot(1., amean, '*', color='b')+ \
        plt.plot(2., mean_coef[1], 'o', color='g')
    plt.xticks(np.arange(2) + 1., ['correlations', 'partial correlations'],
               size=8)
    plt.legend(iter(lineObjects), ('corr(gmean)', 'corrs mean',
               'partials mean'))
    plt.title('regions {0}, {1}'.format(regions[0], regions[1]))
        
plt.show()


def g_larger_a(th):
    return (cov_to_corr(mean_matrices[2]) - mean_matrices[3]) > th


def a_larger_p(th):
    return (mean_matrices[3] - mean_matrices[4]) > th


def g_lower_a(th):
    return (-cov_to_corr(mean_matrices[2]) + mean_matrices[3]) > th


def a_lower_p(th):
    return (-mean_matrices[3] + mean_matrices[4]) > th


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


from nilearn.connectivity.tests.test_embedding_tmp2 import sample_wishart,\
    sample_spd_normal
from nilearn.connectivity.embedding import geometric_mean


def synthetic_data_wishart(sigma, n_samples=201, n_subjects=40):
    for n in range(0, 40):
        rand_gen = np.random.RandomState(0)
        spds = []
        for k in range(n_subjects):
#            spd = random_wishart(sigma, dof=n_samples, rand_gen=rand_gen)
            spd = sample_wishart(sigma, dof=n_samples, rand_gen=rand_gen)
#            if cov_to_corr(spd)[0, 1] - prec_to_partial(np.linalg.inv(spd))[0, 1] > 0.2 \
#                and (cov_to_corr(spd)[0, 1] - prec_to_partial(np.linalg.inv(spd))[0, 1] < 0.5):
            spds.append(spd)
        if not spds:
            print 'no'

    return spds


def synthetic_data_manifold(mean, cov=None, n_subjects=40, dispersion=0.01):
    rand_gen = np.random.RandomState(0)  # TODO: variable random state?
    spds = []
    if False:
        for n in range(0, n_subjects):
            for k in range(n_subjects):
                spd = sample_spd_normal(mean, cov=cov, rand_gen=rand_gen)
                spds.append(spd)
    from nilearn.connectivity.embedding import map_sym, vec_to_sym
    mean_sqrt = map_sym(np.sqrt, mean)
    p = mean.shape[0] * (mean.shape[0] + 1) / 2
    if cov is None:
        cov =  np.eye(p) * dispersion
    for n in range(n_subjects):
        tangent = rand_gen.multivariate_normal(np.zeros(cov.shape[0]), cov)
        tangent = vec_to_sym(tangent, isometry=False)
        tangent_exp = map_sym(np.exp, tangent)
        spds.append(mean_sqrt.dot(tangent_exp).dot(mean_sqrt))
    return spds


subject_n = 0
n_samples = np.mean([subject.shape[0] for subject in subjects])
n_samples = int(n_samples)
spds1 = synthetic_data_wishart(mean_matrices[0] / n_samples,
                              n_samples=n_samples, n_subjects=n_subjects)
spds2 = synthetic_data_manifold(mean_matrices[2],
                               n_subjects=40)
for spds, distribution in zip([spds1, spds2], ['wishart, dof={}'.format(
                                    n_samples), 'gaussian manifold']):
    geo = geometric_mean(spds)
    corrs = [cov_to_corr(spd) for spd in spds]
    partials = [prec_to_partial(np.linalg.inv(spd)) for spd in spds]
    plot_matrix(geo, "gmean, " + distribution)
    plot_matrix(mean_matrices[2], "gmean, data")
    plot_matrix(np.mean(spds, axis=0), "amean, " + distribution)
    plot_matrix(mean_matrices[0], "amean, data")
    plot_matrix(np.mean(spds, axis=0) - geo, "amean - geo, " + distribution)
    plt.show()
    #plot_matrix(cov_to_corr(geo), "corr(gmean), whishart")
    #plot_matrix(np.mean(corrs, axis=0), "amean of corrs, whishart")
    #plot_matrix(np.mean(partials, axis=0), "amean of partials, whishart")
    plot_matrix(np.mean(corrs, axis=0) - cov_to_corr(geo),
                "mean of corrs - corr(gmean), " + distribution)
    plot_matrix(mean_matrices[3] - cov_to_corr(mean_matrices[2]),
                "mean of corrs - corr(gmean), data")
    plot_matrix(np.mean(corrs, axis=0) - np.mean(partials, axis=0),
                "mean of corrs - mean of partials, " + distribution)
    plot_matrix(mean_matrices[3] - np.mean(partials, axis=0),
                "mean of corrs - mean of partial corrs, data")
    plot_matrix(cov_to_corr(geo) - mean_matrices[4],
                "corr(gmean) - mean of partial corrs, " + distribution)
    plot_matrix(cov_to_corr(mean_matrices[2]) - mean_matrices[4],
                "corr(gmean) - mean of partial corrs, data")
    plt.show()

for spds, distribution, color in zip([spds1, spds2],
                                     ['wishart', 'gaussian manifold'],
                                     ['b', 'r']):
    geo = geometric_mean(spds)
    corrs = [cov_to_corr(spd) for spd in spds]
    partials = [prec_to_partial(np.linalg.inv(spd)) for spd in spds]
    dist_plot, = plt.plot((np.mean(corrs, axis=0) - cov_to_corr(geo)).ravel(),
                          (np.mean(corrs, axis=0) - np.mean(
                           partials, axis=0)).ravel(),
                          color + '.', label=distribution + ' distribution')
adhd_plot, = plt.plot((mean_matrices[3] - cov_to_corr(
                      mean_matrices[2])).ravel(),
                      (mean_matrices[3] - mean_matrices[4]).ravel(), 'g.',
                      label='ADHD dataset')
plt.xlabel('corrs - corr(gmean)')
plt.ylabel('corrs - partials')
plt.legend()
plt.title('differences between connectivity measures over regions, {}'.format(
        estimators[n_estimator][0]))
from scipy.stats import pearsonr
r, pval = pearsonr((mean_matrices[3] - cov_to_corr(mean_matrices[2])).ravel(),
                   (mean_matrices[3] - mean_matrices[4]).ravel())
print('pearson corr = {}, pval = {}'.format(r, pval))
plt.show()

# Change the number of  dof in Wishart
adhd_plot, = plt.plot((mean_matrices[3]).ravel(),
                      (mean_matrices[4]).ravel(), 'g*',
                      label='ADHD dataset')
for n_samples in [176]:#range(70, 220, 50):
    spds = synthetic_data_wishart(mean_matrices[0] / n_samples,
                                  n_samples=n_samples, n_subjects=n_subjects)
#    plot_matrix(np.mean(spds, axis=0), "amean, " + distribution + str(n_samples))
    geo = geometric_mean(spds)
    corrs = [cov_to_corr(spd) for spd in spds]
    partials = [prec_to_partial(np.linalg.inv(spd)) for spd in spds]
    plt.plot((np.mean(corrs, axis=0)).ravel(),
             (np.mean(partials, axis=0)).ravel(), '.',
             label='wishart ' + str(n_samples) + ' dof')
#plot_matrix(mean_matrices[0], "amean, data")
plt.xlabel('corrs')
plt.ylabel('partials')
plt.legend()
plt.show()

# Change the dispersion value in Gaussian on manifold
adhd_plot, = plt.plot(np.triu(corr_to_Z(mean_matrices[3]), k=1).ravel(),
                      np.triu(corr_to_Z(mean_matrices[4]), k=1).ravel(), 'g*',
                      label='ADHD dataset')
for dispersion in [1, 0.1, 0.01]:
    spds = synthetic_data_manifold(mean_matrices[2], n_subjects=40,
                                   dispersion=dispersion)
#    plot_matrix(np.mean(spds, axis=0), "amean, " + distribution + str(n_samples))
    geo = geometric_mean(spds)
    corrs = [cov_to_corr(spd) for spd in spds]
    partials = [prec_to_partial(np.linalg.inv(spd)) for spd in spds]
    plt.plot(corr_to_Z((np.mean(corrs, axis=0)).ravel()),
             corr_to_Z(np.mean(partials, axis=0)).ravel(), '.',
             label='G manifold, dispersion {}'.format(dispersion))
#plot_matrix(mean_matrices[0], "amean, data")
plt.xlabel('corrs')
plt.ylabel('partials')
plt.legend()
