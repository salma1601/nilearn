import numpy as np
import matplotlib.pylab as plt
from scipy import stats, linalg
from scipy.signal import welch

from nilearn import input_data, datasets, signal
from nilearn.image import high_variance_confounds
from nilearn.signal import _mean_of_squares, _detrend, _standardize
from nilearn._utils import check_niimg_4d, as_ndarray
import nilearn


def get_series(imgs):
    imgs = check_niimg_4d(imgs)
    sigs = as_ndarray(imgs.get_data())
    del imgs  # help reduce memory consumption
    sigs = np.reshape(sigs, (-1, sigs.shape[-1])).T
    return sigs


def scaled_high_variance_confounds(series, n_confounds=5, percentile=2.,
                                detrend=True, scaling=True):
    """ Return confounds time series extracted from series with highest
        variance.

        Parameters
        ==========
        series: numpy.ndarray
            Timeseries. A timeseries is a column in the "series" array.
            shape (sample number, feature number)

        n_confounds: int, optional
            Number of confounds to return

        percentile: float, optional
            Highest-variance series percentile to keep before computing the
            singular value decomposition, 0. <= `percentile` <= 100.
            series.shape[0] * percentile / 100 must be greater than n_confounds

        detrend: bool, optional
            If True, detrend timeseries before processing.

        scaling: bool, optional
            If True, scaling is the mean across all voxels and timepoints.

        Returns
        =======
        v: numpy.ndarray
            highest variance confounds. Shape: (samples, n_confounds)

        Notes
        ======
        This method is related to what has been published in the literature
        as 'CompCor' (Behzadi NeuroImage 2007).

        The implemented algorithm does the following:

        - compute sum of squares for each time series (no mean removal)
        - keep a given percentile of series with highest variances (percentile)
        - compute an svd of the extracted series
        - return a given number (n_confounds) of series from the svd with
          highest singular values.

        See also
        ========
        nilearn.image.high_variance_confounds
    """
    # Print confounds selected if no scaling
    series_tmp = series.copy()
    if detrend:
        series_tmp = _detrend(series_tmp)
    var = _mean_of_squares(series_tmp)
    var_thr = stats.scoreatpercentile(var, 100. - percentile)
    print('No scaling, selected voxels are {}'.format(np.where(
                                                    series_tmp[:, var > var_thr])))

    if scaling:
        non_zero_mean = series.mean(axis=0) > np.finfo(np.float).eps * 10
        series[:, non_zero_mean] /= series[:, non_zero_mean].mean(axis=0) / 100

    if detrend:
        series = _detrend(series)  # copy

    # Retrieve the voxels|features with highest variance

    # Compute variance without mean removal.
    var = _mean_of_squares(series)

    var_thr = stats.scoreatpercentile(var, 100. - percentile)
    print('Scaling, selected voxels are {}'.format(np.where(
                                        series[:, var > var_thr])))
    series = series[:, var > var_thr]  # extract columns (i.e. features)
    # Return the singular vectors with largest singular values
    # We solve the symmetric eigenvalue problem here, increasing stability
    s, u = linalg.eigh(series.dot(series.T) / series.shape[0])
    ix_ = np.argsort(s)[::-1]
    u = u[:, ix_[:n_confounds]].copy()
    return u


dataset = datasets.fetch_adhd()
atlas = datasets.fetch_msdl_atlas()
# Selecting a low moving subject
mean_motion = []
for n_subject in range(16, 24):
    confound_filename = dataset.confounds[n_subject]
    confounds = np.genfromtxt(confound_filename, skip_header=1)
    motion = confounds[:, 5:11]
    relative_motion = motion.copy()
    relative_motion[1:] -= motion[:-1]
    mean_motion.append(np.linalg.norm(relative_motion) +
                       np.max(np.abs(motion)))

n_subject = 8 + np.argsort(mean_motion)[0]
print n_subject
print motion.shape
confound_filename = dataset.confounds[n_subject]
confounds = np.genfromtxt(confound_filename, skip_header=1)
motion = confounds[:, 5:11]
func_filename = dataset['func'][n_subject]
from joblib import Memory
mem = Memory('/tmp')
n_confounds = 5
confounds = mem.cache(nilearn.image.high_variance_confounds)(
                      func_filename, n_confounds=n_confounds)
voxels_ts = get_series(func_filename)
voxels_ts = mem.cache(signal.clean)(voxels_ts, detrend=False,
                                    standardize=False,
                                    confounds=motion)
scaled_confounds = scaled_high_variance_confounds(voxels_ts,
                                                  n_confounds=n_confounds,
                                                  detrend=True,
                                                  scaling=False)
masker = input_data.NiftiMapsMasker(atlas["maps"], resampling_target="maps")
masker.fit_transform(dataset['func'][0])
region_ts = masker.fit_transform(dataset['func'][n_subject])  # raw signal
region_ts /= np.mean(region_ts, axis=0) / 100
region_ts -= region_ts.mean(axis=0)
cleaned = signal.clean(region_ts, detrend=True, standardize=False,
                       confounds=[motion, confounds])
cleaned_scaling = signal.clean(region_ts, detrend=True,
                               standardize=False,
                               confounds=[motion, scaled_confounds])

# Plotting the extracted confounds
for n, (confound, scaled_confound) in enumerate(zip(
                                    confounds.T, scaled_confounds.T)):
        plt.subplot(n_confounds + 1, 1, n + 1)
        plt.plot(confound, label='no scaling')
        plt.plot(scaled_confound, label='scaling')
plt.legend()
plt.subplot(n_confounds + 1, 1, n + 2)
plt.plot(motion[:, 3:])
plt.show()

# Average Power spectra of confounds
# Sun FT, Miller LM, D'Esposito M. Measuring interregional functional
# connectivity using coherence and partial coherence analyses of fMRI data.
# Neuroimage 2004;21(2):647-58. [PubMed: 14980567]
for n, (confound, scaled_confound) in enumerate(zip(
                                    confounds.T, scaled_confounds.T)):
    plt.subplot(n_confounds + 1, 1, n + 1)
    fs, Pxx = welch(confound, window='hanning', nperseg=64, noverlap=32)
    plt.plot(fs, Pxx, label='no scaling')
    fs, Pxx = welch(scaled_confound, window='hanning', nperseg=64,
                          noverlap=32)
    plt.plot(fs, Pxx, label='scaling')
plt.show()


import nitime.algorithms as tsa
from nitime.viz import plot_spectral_estimate

all_confounds_ortho = []
for all_confounds in [confounds,
                      scaled_confounds]:
    Q = linalg.qr(all_confounds, mode="economic", pivoting=True)[0]
    all_confounds_ortho.append(Q.T)

for all_confounds in all_confounds_ortho:
    all_psd =[]
    all_freq = []
    for n, confound in enumerate(all_confounds):
#        plt.subplot(n_confounds + 1, 1, n + 1)
        welch_freqs, welch_psd = tsa.get_spectra(confound,
                                                 method=dict(this_method='welch',
                                                             NFFT=64,
                                                             n_overlap=32))
        # normalize by peak value
        # TODO: these signals are not orthogonal, so we will not get a neat
        # spectrum by averaging them
        welch_psd /= welch_psd.max()
        welch_freqs *= (np.pi / welch_freqs.max())
        welch_psd = welch_psd.squeeze()
        print welch_psd.max()
        all_psd.append(welch_psd)
        all_freq.append(welch_freqs)
#        plt.plot(welch_freqs, welch_psd)
#    plt.show()
    all_freq = np.array(all_freq)
    all_psd = np.array(all_psd)
    for freq in all_freq:
        np.testing.assert_allclose(freq, all_freq[0])
    
    plt.plot(freq, all_psd.mean(axis=0))
#    fig04 = plot_spectral_estimate(welch_freqs, welch_psd, elabels=("Welch",))
plt.show()

# Plotting the effect of confounds extracted on cleaning
regions = [6]
plt.subplot(3, 1, 1)
plt.plot(region_ts[:, regions], label='raw')
plt.plot(cleaned[:, regions], label='cleaned, no scaling')
plt.plot(cleaned_scaling[:, regions], label='cleaned, scaling')
plt.plot(np.zeros((region_ts.shape[0],)), '--')
plt.legend()
plt.ylabel('standardized signal')

plt.subplot(3, 1, 2)
plt.plot(confounds[:, :3])
plt.ylabel('top 3 confounds no sclaing')

plt.subplot(3, 1, 3)
plt.plot(motion)
plt.ylabel('motion')
plt.show()
