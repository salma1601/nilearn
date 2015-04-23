# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 09:13:31 2015

@author: sb238920
"""
# Standard library imports

# Related third party imports

# Local application/library specific imports
# Extracting region signals ###################################################
import nilearn.image
import nilearn.input_data
import nilearn.signal
from nilearn import datasets

atlas_img, labels = datasets.fetch_msdl()


from sklearn.externals.joblib import Memory
mem = Memory('/nfs/neurospin/servier2/salma/nilearn_cache')

masker = nilearn.input_data.NiftiLabelsMasker(
    labels_img=atlas_img, resampling_target="labels", detrend=False,
    low_pass=None, high_pass=None, t_r=1, standardize=False,
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
    print confound_filename
    hv_confounds = mem.cache(nilearn.image.high_variance_confounds)(
        func_filename)
    confounds = np.genfromtxt(confound_filename, delimiter='\t', names=True)
    motion_names = [name for name in confounds.dtype.names if 'motion' in name]
    motion_confounds = [confounds[name] for name in motion_names]
    diff_motion_confounds = []
    for motion in motion_confounds:
        confounds_data_dt = motion
        conv = np.convolve(motion, np.array([1., 0., -1.]) / 2,
                           'valid')
        confounds_data_dt[1:-1] = conv
        diff_motion_confounds.append(confounds_data_dt)
    diff_motion_names = ['diff_' + name for name in motion_names]
    compcor_names = [name for name in confounds.dtype.names if
        'compcor' in name]
    compcor_confounds = [confounds[name] for name in confounds.dtype.names if
                         ('compcor' in name)]
    my_confounds = motion_confounds + diff_motion_confounds + compcor_confounds
    my_confounds_names = motion_names + diff_motion_names + compcor_names +\
        ['hv_confound' + str(u) for u in range(hv_confounds.shape[-1])]
    my_confounds = np.hstack((np.transpose(my_confounds), hv_confounds))
    low_pass = .08
    high_pass = .009

    print("-- Computing region time series ...")
    region_raw_ts = masker.transform(func_filename)
    region_ts = nilearn.signal.clean(region_raw_ts, detrend=True,
                                     low_pass=low_pass, high_pass=high_pass,
                                     t_r=1.,
                                     standardize=True, confounds=[
                                     hv_confounds, confound_filename])
