# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 20:01:00 2014

@author: salma
"""
# Standard library imports

# Related third party imports

# Local application/library specific imports
import nibabel
import numpy as np
import matplotlib.pylab as plt
from nilearn.masking import compute_epi_mask


gms = []
import glob
filenames = glob.glob('/home/salma/.local/share/nsap/NYU_TRT_session1a/*/'\
                      'func/swralfo.nii')
counfound_files = glob.glob('/home/salma/.local/share/nsap/'\
    'NYU_TRT_session1a/*/func/rp_alfo.txt')

for filename in filenames:
    img = nibabel.load(filename)
    data = img.get_data()
    mask_img = compute_epi_mask(img)  # brain mask image
    mask_data = mask_img.get_data()
    n_times = data.shape[-1]
    n_slices = data.shape[-2]
    x = mask_data.reshape(mask_data.shape + (1, )).repeat(n_times, 3)
    # A sum of mean timepoints intensities for each slice
    sum_slice = [np.sum(np.mean(x[..., n_slice, :] * data[:, :, n_slice, :],
                                 axis=2)) for n_slice in range(0, n_slices)]
    nvox_slice = [np.sum(x[..., n_slice, 0]) for n_slice in range(0, n_slices)]
    intensity = np.array(sum_slice) / np.array(nvox_slice)
    mean_intensity = intensity[np.isfinite(intensity)]
    # Grand mean scale: normalize over time and voxels
    scale = 100. / np.mean(mean_intensity)
    gms.append(scale)

import nilearn.image
import nilearn.input_data
print("-- Fetching datasets ...")
import nilearn.datasets
atlas = nilearn.datasets.fetch_msdl_atlas()

import joblib
mem = joblib.Memory("/home/salma/CODE/Parietal/nilearn/joblib/nilearn/nyu/filtering")
n_subjects = 13
reorder = False
subjects = []
for subject_n in range(n_subjects):
    filename = filenames[subject_n]
    print("Processing file %s" % filename)

    print("-- Computing confounds ...")
    confound_file = counfound_files[subject_n]
    hv_confounds = mem.cache(nilearn.image.high_variance_confounds)(filename)

    print("-- Computing region signals ...")
    t_r = 2.5
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
n_estimator = 1

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

def fdr(p):
    """ FDR correction for multiple comparisons.
    
    Computes fdr corrected p-values from an array o of multiple-test false 
    positive levels (uncorrected p-values) a set after removing nan values, 
    following Benjamin & Hockenberg procedure.
    
    Parameters
    ==========
    p: np.array
        uncorrected pvals
    
    Returns
    =======
    pFDR: np.array
        corrected pvals
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


psc_covs = [all_matrices[0][k] * (gms[k] ** 2) / np.mean(np.array(gms)**2)
    for k in range(n_subjects)]
psc_precs = [all_matrices[1][k] / (gms[k] ** 2) * np.mean(np.array(gms)**2)
    for k in range(n_subjects)]

from scipy import stats

t, pval = stats.ttest_1samp(all_matrices[0], 0., axis=0)
pval_corr = fdr(pval)
masked_matrices = [all_matrices[0][k] * (pval_corr < 0.05) for k in range(n_subjects)]
t, pval = stats.ttest_1samp(psc_covs, 0., axis=0)
pval_corr = fdr(pval)
masked_psc_matrices = psc_covs * (pval_corr < 0.05)
t, pval = stats.ttest_rel(all_matrices[0] , psc_covs, axis=0)
pval_corr = fdr(pval)
#plot_matrix(-np.log10(pval_corr) * (pval_corr < 0.05), title="- log10(pval)")
psc_mean_cov = np.mean(psc_covs, axis=0)
psc_mean_prec = np.mean(psc_precs, axis=0)
diff_cov = mean_matrices[0] - psc_mean_cov
diff_prec = mean_matrices[0] - psc_mean_prec
#diff2 = cov_to_corr(mean_matrices[0]) - cov_to_corr(psc_mean)
plt.hist(np.array(gms)**2)
plot_matrix(-diff_cov)
print np.linalg.norm(diff_cov)/np.linalg.norm(mean_matrices[0])
print np.linalg.norm(diff_prec)/np.linalg.norm(mean_matrices[1])
plt.show()