"""
Temporal preprocessing Computation of covariance matrix between brain regions
======================================================

This example shows how to extract signals from regions defined by an atlas and
to explore the effect of different temporal preprocessings.
"""
n_subjects = 10  # subjects to consider for group-sparse covariance (max: 40)
plotted_subjects = [5, 6]  # subjects indexes to plot

import numpy as np

import matplotlib.pyplot as plt
import nibabel

from nilearn import plotting, image
from nilearn.plotting import cm


def corr_to_Z(corr):
    """Applies Z-Fisher transform. """
    Z = corr.copy()  # avoid side effects
    corr_is_one = 1.0 - abs(corr) < 1e-15
    Z[corr_is_one] = np.inf * np.sign(Z[corr_is_one])
    Z[np.logical_not(corr_is_one)] = \
        np.arctanh(corr[np.logical_not(corr_is_one)])
    return Z


def fdr(p):
    """ FDR correction for multiple comparisons.

    Computes fdr corrected p-values from an array o of multiple-test false
    positive levels (uncorrected p-values) a set after removing nan values,
    following Benjamin & Hockenberg procedure.

    Parameters
    ==========
    p: numpy.array
        Uncorrected pvals.

    Returns
    =======
    pFDR: numpy.array
        Corrected pvals.
    """
    if p.ndim == 1:
        N1 = p.shape[0]
        q = np.nan+np.ones(p.shape)
        idx = p.argsort()
        sp = p[idx]
        N1 = np.sum(np.logical_not( np.isnan(p)))
        if N1 > 0:
            qt = np.minimum(1,N1*sp[0:N1]/(np.arange(N1)+1))
            min1 = np.inf
            for n in range(N1-1,-1,-1):
                min1 = min(min1,qt[n])
                q[idx[n]] = min1
    else:        
        q = np.array([fdr(p[j]) for j in range(p.shape[1])])
    return q


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
mem = Memory('nilearn_cache/preproc_adhd/')

masker = nilearn.input_data.NiftiMapsMasker(
    msdl_atlas_dataset.maps, resampling_target="maps", detrend=False,
    low_pass=None, high_pass=None, t_r=2.5, standardize=False,
    memory=mem, memory_level=1, verbose=2)
masker.fit()

raw_subjects = []
subjects = []
func_filenames = adhd_dataset.func
confound_filenames = adhd_dataset.confounds
is_adhd = []
for n, (func_filename, confound_filename) in enumerate(zip(func_filenames,
                                            confound_filenames)):
    print("Processing file %s" % func_filename)

    print("-- Computing confounds ...")
    hv_confounds = mem.cache(nilearn.image.high_variance_confounds)(
        func_filename)
    confounds = np.genfromtxt(confound_filename, delimiter='\t', names=True)
    motion_confounds = [confounds[name] for name in confounds.dtype.names if
        ('motion' in name)]
    diff_motion_confounds = []
    for motion in motion_confounds:
        confounds_data_dt = motion
        conv = np.convolve(motion, np.array([1., 0., -1.]) / 2,
                           'valid')
        confounds_data_dt[1:-1] = conv
        diff_motion_confounds.append(confounds_data_dt)
    compcor_confounds = [confounds[name] for name in confounds.dtype.names if
         ('compcor' in name)]
    my_confounds = motion_confounds + diff_motion_confounds + compcor_confounds
    my_confounds = np.hstack((np.transpose(my_confounds), hv_confounds))

    # Explore the correlation between the confonds and the voxel time series
    img = nibabel.load(func_filename)
    data = img.get_data()
    affine = img.get_affine()
    
    # Select 1000 random voxels
    # TODO fix the random state
    voxels_indices = np.random.randint(0, size=img.size + 1)
    voxel_ts = data[voxels_indices]
    voxel_ts /= voxel_ts.std
    confounds_ts

    low_pass = .08
    high_pass = .009

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
                                     
    my_region_ts = nilearn.signal.clean(region_raw_ts1, detrend=True,
                                     low_pass=low_pass, high_pass=high_pass, t_r=t_r,
                                     standardize=True, confounds=my_confounds)
                                     # TODO : improve with psc later on

    is_adhd.append(adhd_dataset.phenotypic['adhd'][n])
    subjects.append(my_region_ts)
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

n_subjects = len(subjects)
is_adhd = np.array(is_adhd)
# Statistical tests
from scipy.stats.stats import ttest_ind, ttest_1samp
from nilearn.connectivity import cov_to_corr
from sklearn.covariance import EmpiricalCovariance

cov_estimator = EmpiricalCovariance()
covariances = []
for region_ts in subjects:
    cov_estimator.fit(region_ts)
    covariances.append(cov_estimator.covariance_)
correlations = np.array([cov_to_corr(cov) for cov in covariances])
z_correlations = np.array([corr_to_Z(corr) for corr in correlations])

# TODO Center with respect to site mean
t, pval = ttest_1samp(z_correlations[is_adhd == 1], 0., axis=0)
plot_matrices(np.mean(correlations[is_adhd == 1], axis=0)  * (fdr(pval) < 0.05),
              - np.log10(pval) * (fdr(pval) < 0.05),title='adhd')
t, pval = ttest_1samp(z_correlations[is_adhd == 0], 0., axis=0)
plot_matrices(np.mean(correlations[is_adhd == 0], axis=0)  * (fdr(pval) < 0.05),
              - np.log10(pval) * (fdr(pval) < 0.05),title='controls')
#is_adhd = is_adhd[adhd_dataset.phenotypic['site'] == '"NeuroImage"']
t, pval = ttest_ind(z_correlations[is_adhd == 1], z_correlations[is_adhd == 0],
                    axis=0)

mean_difference = np.mean(correlations[is_adhd == 1], axis=0) - \
    np.mean(correlations[is_adhd == 0], axis=0)
plot_matrices(mean_difference * (pval < 0.05),
              - np.log10(pval) * (pval < 0.05),
              title='adhd vs controls')