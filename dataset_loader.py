import sys
import os

import numpy as np

from sklearn.datasets.base import Bunch
from nilearn.connectivity2.collecting import single_glob

from pyhrf.retreat import locator
code_path = np.genfromtxt('/home/sb238920/CODE/anonymisation/code_path.txt',
                          dtype=str)
sys.path.append(str(code_path))
from my_conn import MyConn


def load_conn(conn_folder, conditions=['ReSt1_Placebo'], standardize=False,
              networks=None, files_pattern='results/preprocessing/'
              'ROI_Subject%03d_Condition%03d.mat'):
    """Return preprocessed times series, motion parameters and regions labels
    and coordinates from a specified conn project.

    Parameters
    ----------
    conn_folder : str
        Path to existant folder output from CONN toolbox.

    conditions : list of str, optional
        List of conditions to load, default to 'ReSt1_Placebo'.

    standardize : bool, optional
        If True, regions timeseries are standardized (default to False).

    networks : list of tuples or None, optional
        Each tuple is a pair (str, list of str) giving network name and the
        associated ROIs names. Default to WMN, AN and DMN regions in biyu's
        order.  # TODO add reference

    Returns
    -------
    output : sklearn.datasets.base.Bunch
        Dictionary like object, the interest attributes are :
         - 'time_series': dict, keys are conditions, values are lists of
         numpy.ndarray giving the signals within the specified ROIs for each
         subject
         - 'motion': dict, keys are conditions, values are lists of
         numpy.ndarray giving the motion parameters for each subject
         - 'rois': list of pairs giving the labels and coordinates of each ROI
    """
    mc = MyConn('from_conn', conn_folder)

    # Collect the conditions and the covariates
    from scipy.io import loadmat
    conn_struct = loadmat(conn_folder + '.mat', squeeze_me=True,
                          struct_as_record=False)['CONN_x']
    conn_conditions = conn_struct.Setup.conditions.names
    covariates = conn_struct.Setup.l1covariates.files

    # Specify the ROIs and their networks. Here: biyu's order
    if networks is None:
        WMN = ['IPL',   # 'RMFG_peak2' removed because of bad results
               'LMFG_peak1', 'RCPL_peak1', 'LCPL_peak3', 'LT']
        AN = ['vIPS_big', 'pIPS_big', 'MT_big', 'FEF_big', 'RTPJ', 'RDLPFC']
        DMN = ['AG_big', 'SFG_big', 'PCC', 'MPFC', 'FP']
        networks = [('WMN', WMN), ('AN', AN), ('DMN', DMN)]

    # Get the ROIs labels and coordinates
    import glob
    preprocessing_file = glob.glob(os.path.join(
        conn_folder, 'results', 'preprocessing', 'ROI*.mat'))[0]
    roi_mat = loadmat(preprocessing_file, squeeze_me=True)
    conn_rois_labels = roi_mat['names']
    conn_rois_coords = roi_mat['xyz']
    rois_labels = [roi_label for network in networks for roi_label in
                   network[1]]
    rois_coords = [conn_rois_coords[np.where(conn_rois_labels == label)][0] for
                   label in rois_labels]
    if len(rois_labels) != len(rois_coords):
        print conn_rois_labels
        raise ValueError('Mismatch between ROIs labels and coordinates')

    # Run the pipeline
    mc.setup()
    mc.analysis(networks, standardize, files_pattern, 'correlations')
    time_series = {}
    motion = {}
    for condition in conditions:
        try:
            condition_id = np.where(conn_conditions == condition)[0][0]
        except IndexError:
            raise ValueError('no condition named {}'.format(condition))

        subjects = mc.runs_[condition]
        if standardize:
            subjects = [signal / signal.std(axis=1)[:, np.newaxis] for signal
                        in subjects]

        time_series[condition] = subjects
        n_subjects = len(time_series[condition])
        motion[condition] = [covariates[n][0][condition_id][2] for n in
                             range(n_subjects)]

    rois = zip(rois_labels, rois_coords)
    return Bunch(time_series=time_series, motion=motion, rois=rois)


def load_nilearn(timeseries_folder, motion_folder, timeseries_pattern,
                 motion_pattern, conditions=['rs1'], standardize=False,
                 networks=None):
    """Return preprocessed times series, motion parameters and regions labels
    and coordinates from nilearn_data.

    Parameters
    ----------
    timeseries_folder : str
        Path to existant folder with extracted timeseries for each subject.

    motion_folder : str
        Path to existant folder with motion parameters for each subject.

    timeseries_pattern : list of tuples (str, str)
        Patterns for timeseries files, consisting of subjects nips and
        acquisition numbers.

    motion_pattern : list of tuples (str, str)
        Patterns for motion files, consisting of subjects nips and
        acquisition numbers.

    conditions : list of str, optional
        List of conditions to load, default to 'rs1'.

    standardize : bool, optional
        If True, regions timeseries are standardized (default to False).

    networks : list of tuples or None, optional
        Each tuple is a pair (str, list of str) giving network name and the
        associated ROIs names. Default to motor, cortical, WMN, visual,
        attention, DMN and silency regions in biyu's
        order.  # TODO add reference

    Returns
    -------
    output : sklearn.datasets.base.Bunch
        Dictionary like object, the interest attributes are :
         - 'time_series': dict, keys are conditions, values are lists of
         numpy.ndarray giving the signals within the specified ROIs for each
         subject
         - 'motion': dict, keys are conditions, values are lists of
         numpy.ndarray giving the motion parameters for each subject
         - 'rois': list of pairs giving the labels and coordinates of each ROI
    """
    # Specify the ROIs
    from pyhrf.retreat import locator
    get_networks = [locator.meta_motor, locator.meta_non_cortical,
                    locator.func_working_memory, locator.meta_visual,
                    locator.meta_attention, locator.meta_default_mode,
                    locator.meta_saliency]
    nilearn_rois_labels = []
    nilearn_rois_coords = []
    for get_network in get_networks:
        network_rois_labels, network_rois_coords = get_network()
        nilearn_rois_labels += network_rois_labels
        nilearn_rois_coords += network_rois_coords

    if networks is None:
        rois_labels = nilearn_rois_labels
        rois_coords = nilearn_rois_coords
    else:
        rois_labels = [roi_label for network in networks for roi_label in
                       network[1] if roi_label in nilearn_rois_labels]
        rois_coords = [coord for (label, coord) in zip(nilearn_rois_labels,
                       nilearn_rois_coords) if label in rois_labels]
        rois_indices = [n for n in range(len(nilearn_rois_labels)) if
                        nilearn_rois_labels[n] in rois_labels]
    if len(rois_labels) != len(rois_coords):
        print nilearn_rois_labels
        raise ValueError('Mismatch between ROIs labels and coordinates')

    rois = zip(rois_labels, rois_coords)

    # Collect time series and motion parameters
    time_series = {}
    motion = {}
    for condition in conditions:
        subjects_paths = [os.path.join(timeseries_folder, subject_id + '_' +
                          acquisition_id + '_' + condition + '.npy') for
                          (subject_id, acquisition_id) in timeseries_pattern]
        subjects = [np.load(path) for path in subjects_paths]
        if standardize:
            subjects = [signal / signal.std(axis=1) for signal in subjects]

        if networks is not None:
            subjects = [signal[:, rois_indices] for signal in subjects]
        time_series[condition] = subjects
        motion_paths = [os.path.join(motion_folder, 'rp_a' + condition + '_' +
                        subject_id + '*acq' + acquisition_id + '*.txt') for
                        (subject_id, acquisition_id) in motion_pattern]
        motion[condition] = [np.genfromtxt(single_glob(path)) for path in
                             motion_paths]

    return Bunch(time_series=time_series, motion=motion, rois=rois)


def load_ho_cpac(data_dir):
    """Loads the laterized masked version of the Harvard Oxford atlas used in
    the ABIDE-CPAC pipeline.

    Parameters
    ==========
    data_dir: str
        Location of the atlas.

    Returns
    =======
    atlas: sklearn.datasets.base.Bunch
        dictionary-like object, keys are:

        - "maps": nibabel.Nifti1Image, 4D maps if a probabilistic atlas is
        requested and 3D labels if a maximum probabilistic atlas was
        requested.

        - "labels": string list, labels of the regions in the atlas.

        - "coords": list of 1D numpy.ndarray of floats, coordinates of the
        atlas regions centers.
    """
    atlas_filename = os.path.join(data_dir, 'ho_roi_atlas.nii.gz')
    if not os.path.isfile(atlas_filename):
        raise ValueError('{} do not contain image ho_roi_atlas.nii.gz'.format(
                         data_dir))
    labels_filename = os.path.join(data_dir, 'ho_roi_labels.txt')
    coords_filename = os.path.join(data_dir, 'ho_roi_coords.txt')
    if not os.path.isfile(atlas_filename):
        raise ValueError('{} do not contain image ho_roi_atlas.nii.gz'.format(
                         data_dir))
    if (not os.path.isfile(labels_filename)) or\
            (not os.path.isfile(coords_filename)):
        # Get labels and corrdinates of cortical regions
        cort_labels, cort_coords = locator.locate_labels_image(
            atlas_filename, ho_atlas_name='cort-maxprob-thr25-1mm',
            symmetric_split=True)

        # Get labels and corrdinates of subcortical regions
        sub_labels, sub_coords = locator.locate_labels_image(
            atlas_filename, ho_atlas_name='sub-maxprob-thr25-1mm',
            symmetric_split=False)

        labels, coords = zip(*[
            (cort_label, cort_coord) if cort_label != 'Background'
            else (sub_label, sub_coord) for
            (cort_label, cort_coord, sub_label, sub_coord) in
            zip(cort_labels, cort_coords, sub_labels, sub_coords)])
        np.savetxt(labels_filename, labels, fmt='%s')
        np.savetxt(coords_filename, coords, fmt='%.5f')
    else:
        labels = np.loadtxt(labels_filename, delimiter=';', dtype=str)
        coords = np.loadtxt(coords_filename)

    return Bunch(maps=atlas_filename, labels=labels, coords=coords)
