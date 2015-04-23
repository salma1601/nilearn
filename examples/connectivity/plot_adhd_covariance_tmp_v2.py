"""
Computation of covariance matrix between brain regions
======================================================

This example shows how to extract signals from regions defined by an atlas,
and to estimate a covariance matrix based on these signals.
"""
n_subjects = 5  # subjects to consider for group-sparse covariance (max: 40)
plotted_subject = 4  # subject index to plot

import numpy as np

import matplotlib.pyplot as plt

from nilearn import plotting, image
from nilearn.plotting import cm
from sklearn.neighbors import KernelDensity
from scipy.stats.distributions import norm


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


def kernel_density(ax, x, title='pdf', bandwidth=0.2, **kwargs):
    """Kernel kensity estimation"""
    # The grid we'll use for plotting
    x_grid = np.linspace(-4.5, 3.5, 1000)
    pdf_true = (0.8 * norm(-1, 1).pdf(x_grid) +
                0.2 * norm(1, 0.3).pdf(x_grid))
    # Plot the three kernel density estimates
#    fig, ax = plt.subplots(1, 1, sharey=True,
#                           figsize=(13, 3))
#    fig.subplots_adjust(wspace=0)
    kde = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde.fit(x[:, np.newaxis])
    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde.score_samples(x_grid[:, np.newaxis])
    pdf = np.exp(log_pdf)
    ax.plot(x_grid, pdf, color='blue', alpha=0.5, lw=3)
    ax.fill(x_grid, pdf_true, ec='gray', fc='gray', alpha=0.4)
    ax.set_title(title)
    ax.set_xlim(-4.5, 3.5)


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
    region_raw_ts0 = masker.transform(func_filename)
    region_raw_ts1 = masker.transform(func_filename)
    # Preprocessing the signal: detrending, filtering and confounds removal
    # by linear regression
    region_ts = nilearn.signal.clean(region_raw_ts0, detrend=True,
                                     low_pass=None, high_pass=0.01, t_r=2.5,
                                     standardize=True, confounds=[
                                     hv_confounds, confound_filename])
    confounds = np.genfromtxt(confound_filename, delimiter='\t', names=True)
    motion_confounds = [confounds[name] for name in confounds.dtype.names if
        ('motion' in name)]
    diff_motion_confounds = []
    for motion in motion_confounds:
        confounds_data_dt = motion
        conv = np.convolve(motion, np.array([1.,0.,-1.])/2, 
                       'valid')
        confounds_data_dt[1:-1] = conv
        diff_motion_confounds.append(confounds_data_dt)
    compcor_confounds = [confounds[name] for name in confounds.dtype.names if
         ('compcor' in name)]
    my_confounds = motion_confounds + diff_motion_confounds + compcor_confounds
    my_confounds = np.hstack((np.transpose(my_confounds), hv_confounds))
    my_region_ts = nilearn.signal.clean(region_raw_ts0, detrend=True,
                                     low_pass=None, high_pass=0.01, t_r=2.5,
                                     standardize=True, confounds=my_confounds)
    region_raw_ts = nilearn.signal.clean(region_raw_ts1, detrend=False,
                                     low_pass=None, high_pass=None, t_r=2.5,
                                     standardize=True)
    if n == plotted_subject:
        # Plot the first signals
        plotted_regions = 1
        plt.subplot(222)
        plt.plot(region_raw_ts[:, :plotted_regions], '-')
        plt.title('subject {0}'.format(plotted_subject))
        plt.ylabel('raw signals')
        plt.subplot(224)
        plt.plot(region_ts[:, :plotted_regions], '-')
        plt.ylabel('preprocessed signals')
        # Plot the subject's high variance confounds
        plt.subplot(121)
        plt.plot(np.array(my_confounds[:, :6]))
        plt.ylabel('high variance confounds')
        my_subject = my_region_ts

    subjects.append(region_ts)
    raw_subjects.append(region_raw_ts)

plt.show()

from sklearn.covariance import EmpiricalCovariance
cov_estimator = EmpiricalCovariance()
cov_estimator.fit(raw_subjects[plotted_subject])
raw_cov = cov_estimator.covariance_
cov_estimator.fit(subjects[plotted_subject])
preproc_cov = cov_estimator.covariance_
cov_estimator.fit(my_subject)
my_cov = cov_estimator.covariance_
from scipy.interpolate import UnivariateSpline, InterpolatedUnivariateSpline

#These plots represent the (sample) distribution of 
#voxel-to-voxel measures before removal of potential
#confounding variables (labeled 'original'), and after
#removal of these effects (labeled 'without confound').
#Typically you would expect to see a somewhat biased
#(shifted to the right) and wide distribution of
#connectivity values before preprocessing (e.g.
#depending on the amount/strength of subject
#movement which introduces artifactual positive
#correlations between distant voxels), and a somewhat
#centered and narrower distribution after removal of
#confounding effects. If the 'without confound'
#distribution looks still biased you might want to
#increase the dimension of white/csf confounds, and/or
#explore additional potential sources of variability (e.g.
#potential time-series confounds, see the art_detect
#toolbox in http://www.nitrc.org/projects
#/artifact_detect/; this will create additional first-level
#covariates that you can enter in the conn toolbox to
#effectively remove a set of outlier scans from
#consideration). If the 'without confounds' distribution
#looks wider than the 'original' distribution, this could
#indicate too few degrees of freedom, you could try to
#remove some confounds (e.g. decrease the
#'derivatives order' to 0 of the realignment confound)
#to increase the dofs of the connectivity analyses


# Typically confounds introduce a positive bias in connectivity measures so the histogram
# of original connectivity values can appear 'shifted' to the right. After confound removal the
# distribution of connectivity values appears approximately centered.

#Typically you would expect to see a somewhat biased
#(shifted to the right) and wide distribution of
#connectivity values before preprocessing (e.g.
#depending on the amount/strength of subject
#movement which introduces artifactual positive
#correlations between distant voxels), and a somewhat
#centered and narrower distribution after removal of
#confounding effects
N = raw_cov.shape[0]
n = N * (N - 1) / 20
i = 0
colors = ['gray', 'blue', 'green']
fig = plt.figure()
fig.subplots_adjust(wspace=0)

# Plot the histogram of the connectivity coefficients for the raw and
# preprocessed signals. Preprocessing is expected to center and narrow the
# distribution.
for cov, label in zip([preproc_cov, my_cov], ['preprocesssed', 'mine']):
    coefs = cov[np.triu_indices(N, k=1)]
    # the histogram of the data with histtype='step'
    color = colors[i]
    i += 1
    plt.hist(coefs, bins=n, normed=1, color=color, alpha=0.4, label=label)
plt.legend(loc='upper right')
plt.title('Covariance coefficients distributions, subject {}'.format(
    plotted_subject))
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
