import os
import glob
import collections

import nibabel
import numpy as np
import matplotlib.pylab as plt
from scipy.io import loadmat
from scipy.stats import pearsonr
from mpl_toolkits.axes_grid1 import make_axes_locatable

import nilearn.image
import nilearn.input_data
import nilearn.signal
from nilearn import datasets


def cov_to_corr(cov):
    """Return correlation matrix for a given covariance matrix.

    Parameters
    ----------
    cov : 2D numpy.ndarray
        The input covariance matrix.

    Returns
    -------
    corr : 2D numpy.ndarray
        The ouput correlation matrix.
    """
    d = np.atleast_2d(1. / np.sqrt(np.diag(cov)))
    corr = cov * d * d.T
    return corr


def plot_matrix(matrix, zero_diag=True, figure=None, axes=None,
                title="connectivity", xlabel="", ylabel="",
                ticks=[], tick_labels=[]):
    """Plot connectivity matrix, for a given measure.

    Parameters
    ==========
    matrix : 2D numpy.ndarray
        Matrix to plot
    zero_diag : bool, optional
        If True, zero the matrix diagonal.
    figure : integer or matplotlib figure, optional
        Matplotlib figure used or its number. If None is given, a
        new figure is created.
    axes : matplotlib axes or 4 tuple of float: (xmin, ymin, width, height),
        optional
        The axes, or the coordinates, in matplotlib figure space,
        of the axes used to display the plot. If None, the complete
        figure is used.
    title : str, optional
        Figure title.
    xlabel : str, optional
        Figure xlabel.
    ylabel : str, optional
        Figure ylabel.
    ticks : list of float, optional
        Figure ticks.
    tick_labels : list of str, optional
        Figure tick labels.
    """
    matrix = matrix.copy()
    if matrix.ndim != 2:
        raise ValueError('expect a 2D array')

    # Put zeros on the diagonal, for graph clarity
    if zero_diag:
        size = matrix.shape[0]
        matrix[range(size), range(size)] = 0

    vmax = np.abs(matrix).max()
    if vmax <= 1e-7:
        vmax = 0.1

    # Display connectivity matrix
    if isinstance(axes, plt.Axes) and figure is None:
        figure = axes.figure

    if not isinstance(figure, plt.Figure):
        # Make sure that we have a figure
        figure = plt.figure(figure, figsize=[2.2, 2.6])

    if isinstance(axes, plt.Axes):
        assert axes.figure is figure, ("The axes passed are not "
                                       "in the figure")

    if axes is None:
        axes = [0., 0., 1., 1.]

    if isinstance(axes, collections.Sequence):
        axes = figure.add_axes(axes)

    im = plt.imshow(matrix, interpolation="nearest",
                    vmin=-vmax, vmax=vmax, cmap=plt.cm.get_cmap("bwr"))

    plt.xticks(ticks, tick_labels, size=8, rotation=90)
    plt.xlabel(xlabel)
    plt.yticks(ticks, tick_labels, size=8)
    axes.yaxis.tick_left()
    plt.ylabel(ylabel)
    plt.title(title)

    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(axes)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cb = plt.colorbar(im, cax=cax, ticks=[-vmax, 0., vmax], format='%.2g')
    cb.ax.tick_params(labelsize=8)


def _sqr_distance_matrix(array, affine, seed):
    # You need a lot of faith to believe in this code!
    seed = np.asarray(seed)
    # Create an array of shape (3, array.shape) containing the i, j, k indices
    indices = np.vstack((np.indices(array.shape), np.ones((1,) + array.shape)))
    # Transform the indices into original space
    indices = np.tensordot(affine, indices, axes=[[1], [0]])[:3]
    # Compute square distance to the seed
    indices = ((indices - seed[:, None, None, None]) ** 2).sum(axis=0)
    return indices


def _create_sphere(array, affine, seed, radius):
    dist = _sqr_distance_matrix(array, affine, seed)
    array[dist <= radius ** 2] = 1


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


def binarize(in_filename, out_filename, threshold=0.9):
    if not os.path.isfile(out_filename):
        img = nibabel.load(in_filename)
        data = img.get_data()
        data[np.isnan(data)] = 0
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

atlas = datasets.fetch_msdl_atlas()


from sklearn.externals.joblib import Memory
mem = Memory('/neurospin/servier2/salma/nilearn_cache')

# TODO: mask the atlas with individual GM masks
masker = nilearn.input_data.NiftiMapsMasker(
    atlas['maps'], resampling_target="maps", detrend=False,
    low_pass=None, high_pass=None, t_r=1, standardize=False,
    memory=mem, memory_level=1, verbose=2)

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
out_folder = 'low_motion'
if out_folder == 'low_motion':
    folders = accepted_acqs
else:
    folders = rejected_acqs

subjects = []
nips = []
sessions = []
n_session = 3
prefix = prefixes[n_session]
for n, folder in enumerate(folders):
    func_filename = single_glob(os.path.join(folder,
                                             'swa' + prefix + '*.nii'))
    motion_filename = single_glob(os.path.join(folder,
                                               'rp_a' + prefix + '*.txt'))
    anat_folder = folder.replace('fMRI', 't1mri')
    gm_filename = single_glob(os.path.join(anat_folder, 'mwc1*.nii'))
    binary_gm_filename = folder.split('/salma/')[1]
    binary_gm_filename = os.path.join('/neurospin/servier2/salma',
                                      binary_gm_filename)
    mask_name = 'bin_' + os.path.basename(gm_filename)
    binary_gm_filename = os.path.join(os.path.dirname(binary_gm_filename),
                                      mask_name)
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
    hv_confounds = mem.cache(nilearn.image.high_variance_confounds)(
        func_filename)

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

    # TODO: Add aCompCor confounds
    # TODO compute compCor confounds for highly moving subjects
    if out_folder == 'low_motion':
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

    gm_mean = hv_confounds[:, 0]
    wm_mean = hv_confounds[:, 1]
    csf_mean = hv_confounds[:, 2]

#    compcor_confound = [confounds[name] for name in confounds.dtype.names if
#                         ('compcor' in name)]
    all_motion_confounds = np.hstack((motion_confounds,
                                      diff_motion_confounds))
    my_confounds = np.hstack((motion_confounds,
                              diff_motion_confounds))
    my_confounds = np.hstack((my_confounds, hv_confounds))

    # TODO: include mean intensities confounds
    if False:
        # Satterthwaite preprocessings
        anat_folder = folder.replace('fMRI', 't1mri')
        wm_filename = single_glob(os.path.join(anat_folder, 'mwc2*.nii'))
        csf_filename = single_glob(os.path.join(anat_folder, 'mwc3*.nii'))
        wm_masker = nilearn.input_data.NiftiMapsMasker(
            wm_filename, memory=mem, memory_level=1, verbose=2)
        csf_masker = nilearn.input_data.NiftiMapsMasker(
            csf_filename, memory=mem, memory_level=1, verbose=2)
        wm_mean = wm_masker.fit_transform(func_filename)
        csf_mean = csf_masker.fit_transform(func_filename)
        global_data = nibabel.load(func_filename).get_data()
        global_mean = np.zeros(global_data.shape[-1])
        for t in range(global_data.shape[-1]):
            global_mean[t] = global_data[global_data[..., t] > 0, t].mean()
        satter_masker = nilearn.input_data.NiftiMapsMasker(
            atlas['maps'], resampling_target="maps", detrend=True,
            low_pass=.1, high_pass=.01, t_r=1, standardize=False,
            memory=mem, memory_level=1, verbose=2)
        satter_confounds = np.hstack((motion_confounds, csf_mean, wm_mean,
                                      global_mean))

    low_pass = .08
    high_pass = .009

    print("-- Computing region time series ...")
    spheres_masker = nilearn.input_data.NiftiSpheresMasker(
        all_seeds, radius, mask_img=binary_gm_filename, smoothing_fwhm=None, detrend=False, low_pass=None,
        high_pass=None, t_r=1, standardize=False, memory=mem, memory_level=1,
        verbose=2)
    region_raw_ts = spheres_masker.fit_transform(func_filename)
    region_ts = nilearn.signal.clean(region_raw_ts, detrend=True,
                                     low_pass=low_pass, high_pass=high_pass,
                                     t_r=1.,
                                     standardize=True,
                                     confounds=[relative_motion, gm_mean,
                                                wm_mean, csf_mean])
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


##########################################################
# Means w.r.t. distances/connectivity
##########################################################
# Load the signals for the resting state sessions
all_subjects = []
for nip, session in zip(nips, sessions):
    all_subjects.append(np.load(os.path.join(
        '/neurospin/servier2/salma/nilearn_outputs', 'low_motion', 'spheres',
        'tmp', nip + '_' + revert_acquisition(session) + '_' + 'rs1.npy')))
for nip, session in zip(nips, sessions):
    all_subjects.append(np.load(os.path.join(
        '/neurospin/servier2/salma/nilearn_outputs', 'high_motion', 'spheres',
        'tmp', nip + '_' + session + '_' + 'rs1.npy')))

n_subjects = len(all_subjects)

# Compute connectivity coefficients for each subject
print("-- Measuring connecivity ...")
measures = ['covariance', 'precision', 'tangent', 'correlation',
            'partial correlation']

from sklearn.covariance import EmpiricalCovariance, LedoitWolf, MinCovDet
estimators = [('ledoit', LedoitWolf()), ('emp', EmpiricalCovariance()),
              ('mcd', MinCovDet())]
n_estimator = 1
all_matrices = []
mean_matrices = []
from nilearn import connectivity
for subjects in [all_subjects[:n_subjects / 2],
                 all_subjects[n_subjects / 2:]]:
    for measure in measures:
        estimator = {'cov_estimator': estimators[n_estimator][1],
                     'kind': measure}
        cov_embedding = connectivity.CovEmbedding(**estimator)
        matrices = connectivity.vec_to_sym(
            cov_embedding.fit_transform(subjects))
        all_matrices.append(matrices)
        if measure == 'tangent':
            mean = cov_embedding.mean_cov_
        else:
            mean = matrices.mean(axis=0)
        mean_matrices.append(mean)
coords = all_seeds
coords = np.array([list(coord) for coord in coords])
distance_matrix = coords - coords[:, np.newaxis]
dist = np.linalg.norm(distance_matrix, axis=-1)
plt.subplot(411)
plt.scatter((mean_matrices[3][dist < 96]).flatten(),
            (dist[dist < 96]).flatten(), c='r')
plt.scatter((cov_to_corr(mean_matrices[2])[dist < 96]).flatten(),
            (dist[dist < 96]).flatten(), c='g')
plt.subplot(412)
plt.scatter((mean_matrices[3][dist > 96]).flatten(),
            (dist[dist > 96]).flatten(), c='r')
plt.scatter((cov_to_corr(mean_matrices[2])[dist > 96]).flatten(),
            (dist[dist > 96]).flatten(), c='g')
plt.subplot(413)
plt.scatter((mean_matrices[3]).flatten(),
            (mean_matrices[8]).flatten(), c='r')
plt.scatter((cov_to_corr(mean_matrices[2])).flatten(),
            (cov_to_corr(mean_matrices[7])).flatten(), c='g')
plt.plot(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
plt.xlabel('conn')
plt.ylabel('dist')
plt.show()


# Correlation between motion and connectivity, relationship with distance
#########################################################################
# Load the signals for the resting state sessions
subjects = []
for nip, session in zip(nips, sessions):
    subjects.append(np.load(os.path.join(
        '/neurospin/servier2/salma/nilearn_outputs', out_folder, 'spheres',
        'tmp', nip + '_' + session + '_' + 'rs1.npy')))
for nip, session in zip(nips, sessions):
    subjects.append(np.load(os.path.join(
        '/neurospin/servier2/salma/nilearn_outputs', out_folder, 'spheres',
        'tmp', nip + '_' + session + '_' + 'rs1.npy')))

n_subjects = len(subjects)
# Compute connectivity coefficients for each subject
print("-- Measuring connecivity ...")
measures = ['covariance', 'precision', 'tangent', 'correlation',
            'partial correlation']

from sklearn.covariance import EmpiricalCovariance, LedoitWolf, MinCovDet
estimators = [('ledoit', LedoitWolf()), ('emp', EmpiricalCovariance()),
              ('mcd', MinCovDet())]
n_estimator = 1
all_matrices = []
mean_matrices = []
from nilearn import connectivity
for measure in measures:
    estimator = {'cov_estimator': estimators[n_estimator][1],
                 'kind': measure}
    cov_embedding = connectivity.CovEmbedding(**estimator)
    matrices = connectivity.vec_to_sym(
        cov_embedding.fit_transform(subjects))
    all_matrices.append(matrices)
    if measure == 'tangent':
        mean = cov_embedding.mean_cov_
    else:
        mean = matrices.mean(axis=0)
    mean_matrices.append(mean)

# Compute mean motion for each subject
mean_motion = []
for prefix in ['rs1', 'rs1']:
    for folder in folders:
        motion_filename = single_glob(os.path.join(folder,
                                                   'rp_a' + prefix + '*.txt'))
        motion_confounds = np.genfromtxt(motion_filename)[:, :3]
        relative_motion = motion_confounds.copy()
        relative_motion[1:] -= motion_confounds[:-1]
        mean_motion.append(np.linalg.norm(relative_motion, axis=1).mean())

# Compute Euclidean distances between nodes in mm
coords = all_seeds
coords = np.array([list(coord) for coord in coords])
distance_matrix = coords - coords[:, np.newaxis]
distance_matrix = np.linalg.norm(distance_matrix, axis=-1)

# Compute pearson correlation between motion and connectivity
correlation = np.zeros(distance_matrix.shape)
all_indices = np.triu_indices(distance_matrix.shape[0], 1)
x_indices = []
y_indices = []
for indices in zip(*all_indices):
    conn = []
    for n in range(n_subjects):
        conn.append(all_matrices[3][n][indices])
    if np.mean(conn) > -1:
        correlation[indices] = pearsonr(mean_motion, conn)[0]
        x_indices.append(indices[0])
        y_indices.append(indices[1])

new_indices = (x_indices, y_indices)

# Scatter plot
dist = distance_matrix[new_indices]
corr = correlation[new_indices]
plt.scatter(dist, corr)
plt.xlabel('euclidean distance (mm)')
plt.ylabel('correlation of motion and connectivity')
r, p = pearsonr(dist, corr)
print('Pearson correlation is {0} with pval {1}'.format(r, p))
t = np.polyfit(dist, corr, 1, full=True)
xp = np.linspace(dist.min(), dist.max(), 100)
p1 = np.poly1d(np.polyfit(dist, corr, 1))
plt.plot(xp, p1(xp))
print('corr = {0} (dist - {1})'.format(t[0][1], - t[0][0] / t[0][1]))
plt.show()

plot_matrix(all_matrices[3][40:].mean(axis=0) -
            all_matrices[3][:40].mean(axis=0))
plt.show()

