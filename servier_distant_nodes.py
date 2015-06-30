# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 09:13:31 2015

@author: sb238920
"""
# Standard library imports
import os
import numpy as np
import glob
from scipy.io import loadmat

# Related third party imports

# Local application/library specific imports
# Extracting region signals ###################################################
import nilearn.image
import nilearn.input_data
import nilearn.signal
from nilearn import datasets


def _convert_matlab(var_in, var_type):
    if var_type == 'cellstr':
        var_in = var_in.transpose()
#        length = len(var_out)
#        var_out = [var_out[string][0][0][:] for string in range(length)]
        var_out = [string[0][0][:] for string in var_in]
    elif var_type == 'cell':           
        var_out = var_in[0]
    elif var_type == 'str':
        var_in = var_in.transpose()
#        var_out = [var_out[string][:] for string in range(len(var_out))][0]            
        var_out = [string[:] for string in var_in][0]            
    else:
        raise ValueError('{} unknown type'.format(var_type))
    return var_out


def single_glob(pattern):
    filenames = glob.glob(pattern)
    if not filenames:
        raise ValueError('No file for pattern {}'.format(pattern))
    if len(filenames) > 1:
        raise ValueError('Non unique file for pattern {}'.format(pattern))
    return filenames[0]


atlas= datasets.fetch_msdl_atlas()


from sklearn.externals.joblib import Memory
mem = Memory('/neurospin/servier2/salma/nilearn_cache')

masker = nilearn.input_data.NiftiMapsMasker(
    atlas['maps'], resampling_target="maps", detrend=False,
    low_pass=None, high_pass=None, t_r=1, standardize=False,
    memory=mem, memory_level=1, verbose=2)
masker.fit()

raw_subjects = []
subjects = []
sessions_filename = '/neurospin/servier2/salma/nilearn/sessions.txt'
sessions = np.genfromtxt(sessions_filename, delimiter=':', dtype='str',
                         usecols=(0, 1))
data_path = '/neurospin/servier2/salma'
folders = [os.path.join(data_path, session[0], 'fMRI', session[1]) for
           session in sessions]
for n, folder in enumerate(folders):
    func_filename = single_glob(os.path.join(folder, 'swars1*.nii'))
    motion_filename = single_glob(os.path.join(folder, 'rp_ars1*.txt'))
    task_filename = single_glob(os.path.join(folder, 'regTask_rs1*.mat'))
    task_ts = loadmat(task_filename)['reg']

    print("Processing file %s" % func_filename)

    print("-- Computing confounds ...")
    hv_confounds = mem.cache(nilearn.image.high_variance_confounds)(
        func_filename)

    motion_confounds = np.genfromtxt(motion_filename)
    diff_motion_confounds = []
    for motion in motion_confounds:
        confounds_data_dt = motion
        conv = np.convolve(motion, np.array([1., 0., -1.]) / 2,
                           'valid')
        confounds_data_dt[1:-1] = conv
        diff_motion_confounds.append(confounds_data_dt)

    # TODO: Add aCompCor confounds
#    compcor_confound = [confounds[name] for name in confounds.dtype.names if
#                         ('compcor' in name)]
    my_confounds = np.hstack((motion_confounds,
                              diff_motion_confounds))
    my_confounds = np.hstack((my_confounds, hv_confounds))
    low_pass = .08
    high_pass = .009

    print("-- Computing region time series ...")
    region_raw_ts = masker.transform(func_filename)
    region_ts = nilearn.signal.clean(region_raw_ts, detrend=True,
                                     low_pass=low_pass, high_pass=high_pass,
                                     t_r=1.,
                                     standardize=True, confounds=my_confounds)
    np.save(os.path.join('/neurospin/servier2/salma/nilearn_outputs',
                          sessions[n][0] + '_' + sessions[n][1] + '_rs1'),
            region_ts)
