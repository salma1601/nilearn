"""
Computation of covariance matrix between brain regions
======================================================

This example shows how to extract signals from regions defined by an atlas,
and to estimate a covariance matrix based on these signals.
"""
n_subjects = 3  # subjects to consider for group-sparse covariance (max: 40)
plotted_subject = 2  # subject to plot

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
               vmin=-np.max(np.abs(cov)), vmax=np.max(np.abs(cov)), cmap=cm.bwr)
#               vmin=-1, vmax=1, cmap=cm.bwr)
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


def plot_ts(multiple_ts, region_names, preprocessings, title=''):
    """Plot time series within given regions for different preprocessing. """
    axes = None
    n_plots = len(multiple_ts)
    for i, preproc in enumerate(preprocessings):
        axes = plt.subplot(n_plots, 1, i + 1, sharex=axes)
        for region_name, ts in zip(region_names, multiple_ts[i].transpose()):
            print region_name, ts.std(), ts.mean()
            plt.plot(ts, label=region_name)
        plt.ylabel(preproc)
        plt.legend()
        if i == 0:
            plt.title(title)
    plt.xlabel('timestep')
    plt.title(title)


def plot_histograms(multiple_ts, colors=[], preprocessings=[], title=''):
    """Plot historgram of empirical covariance coefficients, for a given
    processing"""

    # Compute empirical covariance coefficients
    from sklearn.covariance import EmpiricalCovariance
    cov_estimator = EmpiricalCovariance()
    plt.figure()
    axes = None
    n_plots = len(multiple_ts)
    for i, preproc in enumerate(preprocessings):
        region_ts = multiple_ts[i].copy()

        # Standardize the signals to have coefficients in range [-1, 1]
 #       region_ts /= region_ts.std(axis=0)
        color = colors[i]
        cov_estimator.fit(region_ts)
        covariance = cov_estimator.covariance_

        # Plot the histogram
        axes = plt.subplot(n_plots, 1, i + 1)
        n_regions = covariance.shape[0]
        bins = n_regions * (n_regions - 1) / 20
        plt.hist(covariance[np.triu_indices(n_regions, k=1)], bins=bins,
                            normed=1, color=color, alpha=0.4)
        plt.ylabel(preproc)
        if i == 0:
            plt.title(title)
    plt.xlabel('covariance coefficients')
    for i, preproc in enumerate(preprocessings):
        region_ts = multiple_ts[i].copy()
        cov_estimator.fit(region_ts)
        covariance = cov_estimator.covariance_
        plot_matrices(covariance, np.linalg.inv(covariance), preproc)

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
    low_pass=None, high_pass=None, t_r=2.5, standardize=False,
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

    print("-- Computing region time series ...")
    region_raw_ts = masker.transform(func_filename)
    
    region_raw_ts1 = masker.transform(func_filename)
    region_raw_ts2 = masker.transform(func_filename)

    # Detrending, filtering and confounds removal by linear regression
    print '1'
    region_raw_ts = nilearn.signal.clean(region_raw_ts2, detrend=True,
                                     low_pass=None, high_pass=None, t_r=2.5,
                                     standardize=False)
    region_ts = nilearn.signal.clean(region_raw_ts2, detrend=True,
                                     low_pass=None, high_pass=None, t_r=2.5,
                                     standardize=False, confounds=[
                                     hv_confounds, confound_filename])
    from nilearn.signal_tmp import clean_psc
    print '2'
    region_ts = nilearn.signal.clean(region_raw_ts2, detrend=True,
                                     low_pass=None, high_pass=None, t_r=2.5,
                                     standardize=True)
    print '3'
    region_ts_psc = clean_psc(region_raw_ts2, detrend=True,
                                     low_pass=None, high_pass=None, t_r=2.5,
                                     standardize=False, psc=True)

    subjects.append(region_ts)
    if n == plotted_subject:
        plotted_regions = [17, 18, 19]  # regions to plot their time series
        preprocessings=['raw', 'standardized', 'psc']
        # Get the regions names
        labels = np.genfromtxt(msdl_atlas_dataset['labels'], delimiter=',',
                               names=True, dtype=None)
        region_names = labels['name'][plotted_regions]

        # Plot the time series
        multiple_ts = [ts[:, plotted_regions] for ts in [region_raw_ts,
                       region_ts, region_ts_psc]]
        plot_ts(multiple_ts,
                preprocessings=preprocessings,
                region_names=region_names,
                title='subject {}'.format(plotted_subject))

        # Plot the histogram of the covariance coefficients for the raw and
        # preprocessed signals. Preprocessing is expected to center and narrow
        # the distribution
        plot_histograms([region_raw_ts, region_ts, region_ts_psc],
                        preprocessings=preprocessings,
                        colors=['grey', 'blue', 'red'],
                        title='subject {}'.format(plotted_subject))
        plt.show()



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
