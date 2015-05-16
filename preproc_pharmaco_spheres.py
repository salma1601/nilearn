import os
import glob

import nibabel
import numpy as np
import matplotlib.pylab as plt
from scipy.io import loadmat

import nilearn.image
import nilearn.input_data
import nilearn.signal


def _convert_matlab(var_in, var_type):
    if var_type == 'cellstr':
        var_in = var_in.transpose()
        var_out = [string[0][0][:] for string in var_in]
    elif var_type == 'cell':
        var_out = var_in[0]
    elif var_type == 'str':
        var_in = var_in.transpose()
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


def binarize(in_filename, out_filename, threshold=None):
    if not os.path.isfile(out_filename):
        img = nibabel.load(in_filename)
        data = img.get_data()
        data[np.isnan(data)] = 0
        if threshold is None:
            threshold = np.percentile(data, 90.)

        data[data <= threshold] = 0
        data[data > threshold] = 1
        img = nibabel.Nifti1Image(data, img.get_affine(), img.get_header())
        nibabel.save(img, out_filename)
    else:
        pass


def revert_acquisition(name):
    """Reverts acqusition numbers"""
    number = name[-1]
    if number == '1':
        number = '2'
    else:
        number = '1'
    return name[:-1] + number


from sklearn.externals.joblib import Memory
mem = Memory('/neurospin/servier2/salma/nilearn_cache')

# Define the coordinates from the litterature/functionals
from pyhrf.retreat import locator
all_names = []
all_seeds = []
for get_networks in [locator.meta_motor, locator.meta_non_cortical,
                     locator.func_working_memory, locator.meta_visual,
                     locator.meta_attention, locator.meta_default_mode,
                     locator.meta_saliency]:
    names, seeds = get_networks()
    all_names += names
    all_seeds += seeds
radius = 6.
raw_subjects = []
subjects = []
acq_filename = '/neurospin/servier2/salma/nilearn/sessions.txt'

# Get folders for Placebo acquisitions
subjects_acqs = np.genfromtxt(acq_filename, delimiter=':',
                              dtype='str', usecols=(0, 1))
data_path = '/neurospin/servier2/salma'
accepted_acqs = [os.path.join(
    data_path, subject_acq[0], 'fMRI', subject_acq[1]) for subject_acq in
    subjects_acqs]
data_path = '/neurospin/servier2/study'
rejected_acqs = [os.path.join(data_path, subject_acq[0], 'fMRI',
                              revert_acquisition(subject_acq[1]))
                 for subject_acq in subjects_acqs]
rejected_acqs = [folder for folder in rejected_acqs if os.path.isdir(folder)]
# TODO: spatially preprocess AP130327_acquisition11
rejected_acqs.remove('/neurospin/servier2/study/AP130327/fMRI/acquisition11')

# Get the path for the CONN project
conn_dir = os.path.join('/neurospin/servier2/salma/subject1to40',
                        'conn_servier2_1to40sub_RS1-Nback2-Nback3-RS2_Pl-D_1_1_1')
raw_dir = os.path.join(conn_dir, 'data')
prefixes = ['rs1', 'nback_2', 'nback_3', 'rs2']

# Define directory to save output signals
out_folder = 'high_motion'
if out_folder == 'low_motion':
    folders = accepted_acqs
else:
    folders = rejected_acqs

subjects = []
nips = []
sessions = []
n_session = 0
prefix = prefixes[n_session]
for n, folder in enumerate(folders):
    func_filename = single_glob(os.path.join(folder,
                                             'swa' + prefix + '*.nii'))
    motion_filename = single_glob(os.path.join(folder,
                                               'rp_a' + prefix + '*.txt'))
    anat_folder = folder.replace('fMRI', 't1mri')

    # Compute binary GM mask
    gm_filename = single_glob(os.path.join(anat_folder, 'mwc1*.nii'))
    if out_folder == 'low_motion':
        binary_gm_filename = folder.split('/salma/')[1]
    else:
        binary_gm_filename = folder.split('/study/')[1]

    binary_gm_filename = os.path.join('/neurospin/servier2/salma',
                                      binary_gm_filename)
    mask_name = 'bin_' + os.path.basename(gm_filename)
    binary_gm_filename = os.path.join(os.path.dirname(binary_gm_filename),
                                      mask_name)
    # Some regions do not intersect GM mask for high threshold
    if mask_name in ['bin_mwc1anat_cp110075_20130819_acq11_04.nii',
                     'bin_mwc1anat_mp130349_20130917_acq11_04.nii',
                     'bin_mwc1anat_hs120456_20121126_acq11_04.nii']:
        threshold = 0.3
    else:
        threshold = 0.4
    binarize(gm_filename, binary_gm_filename, threshold=threshold)

    if out_folder == 'low_motion':
    # TODO: generate reg_task*.mat for high moving sessions
        task_filename = single_glob(os.path.join(folder,
                                                 'regTask_' + prefix + '*.mat'))
        task_ts = loadmat(task_filename)['reg']
    print("Processing file %s" % func_filename)

    print("-- Computing confounds ...")
    # Motion and motion derivatives
    ###############################
    motion_confounds = np.genfromtxt(motion_filename)
    relative_motion = motion_confounds.copy()
    relative_motion[1:] -= motion_confounds[:-1]
    diff_motion_confounds = []
    for motion in motion_confounds:
        confounds_data_dt = motion
        conv = np.convolve(motion, np.array([1., 0., -1.]) / 2,
                           'valid')
        confounds_data_dt[1:-1] = conv
        diff_motion_confounds.append(confounds_data_dt)

    all_motion_confounds = np.hstack((motion_confounds,
                                      diff_motion_confounds))

    # PCA components from WM and CSF
    ################################
    hv_confounds = mem.cache(nilearn.image.high_variance_confounds)(
        func_filename, n_confounds=10)
    if out_folder == 'low_motion':
        # Compute binary WM and CSF masks
        binary_masks = []
        for pattern in ['mwc2*.nii', 'mwc3*.nii']:
            mask_filename = single_glob(os.path.join(anat_folder, pattern))
            if out_folder == 'low_motion':
                binary_mask_filename = folder.split('/salma/')[1]
            else:
                binary_mask_filename = folder.split('/study/')[1]
            binary_mask_filename = os.path.join('/neurospin/servier2/salma',
                                                binary_mask_filename)
            binary_mask_basename = 'bin_' + os.path.basename(mask_filename)
            binary_mask_filename = os.path.join(
                os.path.dirname(binary_mask_filename), binary_mask_basename)
            binarize(mask_filename, binary_mask_filename)
            binary_masks.append(binary_mask_filename)

        # Nilearn aCompCor confounds
        gm_mean = hv_confounds[:, 0]
        wm_mean = hv_confounds[:, 1]
        csf_mean = hv_confounds[:, 2]
        nilearn_pca = []
        for tissue_mask in binary_masks:
            niimg = nibabel.load(func_filename)
            mask_img = mem.cache(nilearn.image.resample_img)(
                tissue_mask, target_affine=niimg.get_affine(),
                target_shape=niimg.shape[:3],
                interpolation='nearest')
            tissue_confounds = mem.cache(nilearn.image.high_variance_confounds)(
                func_filename, n_confounds=5, percentile=100.,
                mask_img=mask_img, detrend=False)
            nilearn_pca.append(tissue_confounds)
           #from sklearn.decomposition import PCA
           #pca = PCA(n_components=5)
           #pca.fit(tissue_func.T)
           #tissue_confounds = pca.components_.T
        wm_pca = nilearn_pca[0]
        csf_pca = nilearn_pca[1]

        # CONN aCompCor computation
        # TODO compute compCor confounds for highly moving subjects
        subject_inpath = os.path.join(
            raw_dir, 'ROI_Subject{0:03d}_Session{1:03d}.mat'.format(
            n + 1, n_session + 1))
        mat_dict = loadmat(subject_inpath)
        conn_signals = _convert_matlab(mat_dict['data'], 'cell')
        gm_mean = conn_signals[0]
        wm_pca = conn_signals[1]
        wm_mean = wm_pca[:, 0]
        csf_pca = conn_signals[2]
        csf_mean = csf_pca[:, 0]
    else:
        wm_pca = hv_confounds[:, :5]
        csf_pca = hv_confounds[:, 5:]

    my_confounds = np.hstack((motion_confounds,
                              diff_motion_confounds))
    my_confounds = np.hstack((my_confounds, wm_pca))
    my_confounds = np.hstack((my_confounds, csf_pca))
    plt.plot(wm_pca)
    plt.plot(csf_pca)

#    compcor_confound = [confounds[name] for name in confounds.dtype.names if
#                         ('compcor' in name)]

    # Task confound
    ###############
    # TODO compute task regressors for highly moving subjects
    if out_folder == 'low_motion':
        my_confounds = np.hstack((my_confounds, task_ts))

    print(' regressing out {} confounds'.format(my_confounds.shape[-1]))
    low_pass = .08
    high_pass = .009

    print("-- Computing region time series ...")
    spheres_masker = nilearn.input_data.NiftiSpheresMasker(
        all_seeds, radius, mask_img=binary_gm_filename, smoothing_fwhm=None,
        detrend=False, low_pass=None,
        high_pass=None, t_r=1, standardize=False, memory=mem, memory_level=1,
        verbose=2)
    region_raw_ts = spheres_masker.fit_transform(func_filename)
    region_ts = nilearn.signal.clean(region_raw_ts, detrend=True,
                                     low_pass=low_pass, high_pass=high_pass,
                                     t_r=1.,
                                     standardize=True,
                                     confounds=[relative_motion, gm_mean,
                                                wm_mean, csf_mean])
    region_ts_optimal = nilearn.signal.clean(region_raw_ts, detrend=True,
                                             low_pass=low_pass,
                                             high_pass=high_pass, t_r=1.,
                                             standardize=False,
                                             confounds=my_confounds)
    region_no_motion_ts = nilearn.signal.clean(region_raw_ts, detrend=True,
                                               low_pass=None,
                                               high_pass=None, t_r=1.,
                                               standardize=False,
                                               confounds=all_motion_confounds)

    subjects.append(region_ts)
    nip = folder.split('/fMRI/')[0]
    nip = os.path.basename(nip)
    nips.append(nip)
    session = os.path.basename(folder)
    sessions.append(session)
    np.save(os.path.join(
        '/neurospin/servier2/salma/nilearn_outputs', out_folder, 'spheres',
        'tmp', nip + '_' + session + '_' + prefix),
        region_ts)
    np.save(os.path.join(
        '/neurospin/servier2/salma/nilearn_outputs', out_folder, 'spheres',
        'raw', nip + '_' + session + '_' + prefix),
        region_raw_ts)
    np.save(os.path.join(
        '/neurospin/servier2/salma/nilearn_outputs', out_folder, 'spheres',
        'no_motion', nip + '_' + session + '_' + prefix),
        region_no_motion_ts)
    np.save(os.path.join(
        '/neurospin/servier2/salma/nilearn_outputs', out_folder, 'spheres',
        'optimal', nip + '_' + session + '_' + prefix),
        region_ts_optimal)
