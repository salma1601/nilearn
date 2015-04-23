"""
Computation of covariance matrix between brain regions
======================================================

This example shows how to extract signals from regions defined by an atlas,
and to estimate a covariance matrix based on these signals.
"""
n_subjects = 5  # subjects to consider for group-sparse covariance (max: 40)
plotted_subject = 0  # subject index to plot

import numpy as np

import matplotlib.pyplot as plt

from nilearn import plotting, image
from nilearn.plotting import cm


def plot_matrices(cov, prec, title):
    """Plot covariance and precision matrices, for a given processing. """

    # Compute sparsity pattern
    sparsity = (prec == 0)

    prec = prec.copy()  # avoid side effects

    # Put zeros on the diagonal, for graph clarity.
    size = prec.shape[0]
    prec[range(size), range(size)] = 0
    span = max(abs(prec.min()), abs(prec.max()))

    # Display covariance matrix
    plt.figure()
    plt.imshow(cov, interpolation="nearest",
               vmin=-1, vmax=1, cmap=cm.bwr)
    plt.colorbar()
    plt.title("%s / covariance" % title)

    # Display sparsity pattern
    plt.figure()
    plt.imshow(sparsity, interpolation="nearest")
    plt.title("%s / sparsity" % title)

    # Display precision matrix
    plt.figure()
    plt.imshow(prec, interpolation="nearest",
               vmin=-span, vmax=span,
               cmap=cm.bwr)
    plt.colorbar()
    plt.title("%s / precision" % title)


def plot_histogram(region_ts, color='grey', label='raw signal'):
    """Plot historgram of empirical covariance coefficients, for a given
    processing"""
    from sklearn.covariance import EmpiricalCovariance
    cov_estimator = EmpiricalCovariance()
    cov_estimator.fit(region_ts)
    covariance = cov_estimator.covariance_
    n_regions = covariance.shape[0]
    bins = n_regions * (n_regions - 1) / 20
    plt.hist(covariance[np.triu_indices(n_regions, k=1)], bins=bins, normed=1,
             color=color, alpha=0.4, label=label)
    ax = plt.gca()
    ax.yaxis.set_label_position('right')
    plt.ylabel(label)


# Fetching datasets ###########################################################
print("-- Fetching datasets ...")
from nilearn import datasets
msdl_atlas_dataset = datasets.fetch_msdl_atlas()
adhd_dataset = datasets.fetch_adhd(n_subjects=n_subjects)

# Extracting region signals ###################################################
import nilearn.image
import nilearn.input_data
import nilearn.signal

from sklearn.externals.joblib import Memory
mem = Memory('nilearn_cache')

masker = nilearn.input_data.NiftiMapsMasker(
    msdl_atlas_dataset.maps, resampling_target="maps", detrend=False,
    low_pass=None, high_pass=None, t_r=2.5, standardize=True,
    memory=mem, memory_level=1, verbose=2)
masker.fit()

raw_subjects = []
subjects = []
func_filenames = adhd_dataset.func
confound_filenames = adhd_dataset.confounds
for n, (func_filename, confound_filename) in enumerate(zip(func_filenames,
                                            confound_filenames)):
    print("Processing file %s" % func_filename)

    print("-- Computing confounds ...")
    hv_confounds = mem.cache(nilearn.image.high_variance_confounds)(
        func_filename)

    print("-- Computing region signals ...")
    region_raw_ts = masker.transform(func_filename)
    # Preprocessing the signal: detrending, filtering and confounds removal
    # by linear regression
    region_ts = nilearn.signal.clean(region_raw_ts, detrend=True,
                                     low_pass=0.1, high_pass=0.01, t_r=2.5,
                                     standardize=True, confounds=[
                                     hv_confounds, confound_filename])
    if n == plotted_subject:
        # Plot the subject's motion confounds
        plt.figure(figsize=(3, 4))
        plt.subplot(121)
        confounds = np.genfromtxt(confound_filename, delimiter='\t',
                                  names=True)
        for name in confounds.dtype.names:
            if  'motion' in name:
                plt.plot(confounds[name], label=name.replace('motion', ''))
        plt.legend()

        # Plot the signal within a chosen region
        plotted_region = 1
        plt.subplot(222)
        plt.plot(region_raw_ts[:, plotted_region], '-')
        plt.title('subject {0}'.format(plotted_subject))
        plt.ylabel('raw signals')
        plt.subplot(224)
        plt.plot(region_ts[:, plotted_region], '-')
        plt.ylabel('preprocessed signals')
        plt.xlim([0, region_ts.shape[0]])

        # Plot the histogram of the covariance coefficients for the raw and
        # preprocessed signals. Preprocessing is expected to center and narrow
        # the distribution
        plt.figure()
        ax1 = plt.subplot(2, 1, 1)
        plot_histogram(region_raw_ts, color='blue', label='raw signal')
        ax2 = plt.subplot(2, 1, 2, sharex=ax1, sharey=ax1)
        plot_histogram(region_ts, color='grey', label="preprocessed signals")
        plt.title('Covariance coefficients distributions, subject {}'.format(
            plotted_subject))

        plt.show()

    subjects.append(region_ts)


# Computing group-sparse precision matrices ###################################
print("-- Computing group-sparse precision matrices ...")
from nilearn.group_sparse_covariance import GroupSparseCovarianceCV
gsc = GroupSparseCovarianceCV(verbose=2)
gsc.fit(subjects)

print("-- Computing graph-lasso precision matrices ...")
from sklearn import covariance
gl = covariance.GraphLassoCV(verbose=2)
gl.fit(subjects[plotted_subject])

# Displaying results ##########################################################
print("-- Displaying results")
atlas_imgs = image.iter_img(msdl_atlas_dataset.maps)
atlas_region_coords = [plotting.find_xyz_cut_coords(img) for img in atlas_imgs]

title = "Subject {0:d} GroupSparseCovariance $\\alpha={1:.2e}$".format(
    plotted_subject, gsc.alpha_)

plotting.plot_connectome(gsc.covariances_[..., plotted_subject],
                         atlas_region_coords, edge_threshold='90%',
                         title=title)
plot_matrices(gsc.covariances_[..., plotted_subject],
              gsc.precisions_[..., plotted_subject], title)

title = "Subject {0:d} GraphLasso $\\alpha={1:.2e}$".format(
    plotted_subject, gl.alpha_)

plotting.plot_connectome(gl.covariance_, atlas_region_coords,
                         edge_threshold='90%', title=title)
plot_matrices(gl.covariance_, gl.precision_, title)

plt.show()
