# -*- coding: utf-8 -*-
"""
Loader for ROIs time series and confounds
"""
# Standard library imports

# Related third party imports
import numpy as np
from scipy.io import loadmat
# Local application/library specific imports


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


def fetch_servier(conn_dir, output_dir, conditions=None):
    """Download and load a dataset analysed by CONN.

    Parameters
    ----------
    data_dir: string, optional
        Path of the data directory. Used to force data storage in a specified
        location. Default: None

    url: string, optional
        Override download URL. Used for test only (or if you setup a mirror of
        the data).

    Returns
    -------
    data: sklearn.datasets.base.Bunch
        Dictionary-like object, the interest attributes are :
         - 'func': Path to functional timeseries
         - 'rois': Names and coordinates of the ROIs
         - 'phenotypic': Explanations of preprocessing steps
         - 'confounds': CSV files containing the nuisance variables

    References
    ----------
    :Download:
        ftp://www.nitrc.org/fcon_1000/htdocs/indi/adhd200/sites/ADHD200_40sub_preprocessed.tgz

    """
    if output_dir:
        
    # Load and save a selction of CONN outputs as numpy arrays
    conn_x = loadmat(conn_dir + '.mat', struct_as_record=False,
                     squeeze_me=True)['CONN_x']
    n_subjects = conn_x.Setup.nsubjects
    preproc_dir = os.path.join(conn_dir, 'results', 'preprocessing')
    if conditions is None:
        conditions = conn_x.Setup.conditions.names
        conditions = conditions[:-1]  # remove empty condition
    subjects = []
    conn_signals[conn_rois.index(roi)][:, 0]
    for c, condition in enumerate(conditions):
        for n in range(n_subjects):
            subject_path = os.path.join(preproc_dir, 'ROI_Subject{1:03d}_'
                                        'Condition{1:03d}.mat'.format(n + 1,
                                                                      c + 1))
            mat_file = io.loadmat(subject_path)
            subject = _convert_matlab(mat_file['data'], 'cell')
            subject = np.array([sub for sub in subject])
            subjects.append(subject)
        np.save(subjects,
                os.path.join(output_dir, 'subjects_{0}'.format(condition)))
    rois = _convert_matlab(mat_file['names'], 'cellstr')
    np.save(rois, os.path.join(output_dir, 'rois'))

    phenotypic['despiking'] = conn_x.Preproc.despiking
    phenotypic['detrending'] = conn_x.Preproc.detrending
    phenotypic['filter'] = conn_x.Preproc.filter

    confound_names = conn_x.Preproc.confounds.names
    confound_dims = conn_x.Preproc.confounds.dimensions
    confound_derivs = conn_x.Preproc.confounds.deriv
    phenotypic['confounds'] = [name + '_comp' + dim + '_deriv' + deriv for name
                               in confound_names for dim in confounds_dims for
                               deriv in confound_derivs]

    # TODO: get confound value from preproc folder (effect of blabla)
    
    if url is None:
        url = 'http://connectir.projects.nitrc.org'
    f1 = url + '/adhd40_p1.tar.gz'
    f2 = url + '/adhd40_p2.tar.gz'
    f3 = url + '/adhd40_p3.tar.gz'
    f4 = url + '/adhd40_p4.tar.gz'
    f1_opts = {'uncompress': True}
    f2_opts = {'uncompress': True}
    f3_opts = {'uncompress': True}
    f4_opts = {'uncompress': True}

    fname = '%s_rest_tshift_RPI_voreg_mni.nii.gz'
    rname = '%s_regressors.csv'

    # Subjects ID per file
    sub1 = ['3902469', '7774305', '3699991']
    sub2 = ['2014113', '4275075', '1019436', '3154996', '3884955', '0027034',
            '4134561', '0027018', '6115230', '0027037', '8409791', '0027011']
    sub3 = ['3007585', '8697774', '9750701', '0010064', '0021019', '0010042',
            '0010128', '2497695', '4164316', '1552181', '4046678', '0023012']
    sub4 = ['1679142', '1206380', '0023008', '4016887', '1418396', '2950754',
            '3994098', '3520880', '1517058', '9744150', '1562298', '3205761',
            '3624598']
    subs = sub1 + sub2 + sub3 + sub4

    subjects_funcs = \
        [(os.path.join('data', i, fname % i), f1, f1_opts) for i in sub1] + \
        [(os.path.join('data', i, fname % i), f2, f2_opts) for i in sub2] + \
        [(os.path.join('data', i, fname % i), f3, f3_opts) for i in sub3] + \
        [(os.path.join('data', i, fname % i), f4, f4_opts) for i in sub4]

    subjects_confounds = \
        [(os.path.join('data', i, rname % i), f1, f1_opts) for i in sub1] + \
        [(os.path.join('data', i, rname % i), f2, f2_opts) for i in sub2] + \
        [(os.path.join('data', i, rname % i), f3, f3_opts) for i in sub3] + \
        [(os.path.join('data', i, rname % i), f4, f4_opts) for i in sub4]

    phenotypic = [('ADHD200_40subs_motion_parameters_and_phenotypics.csv', f1,
        f1_opts)]

    max_subjects = len(subjects_funcs)
    # Check arguments
    if n_subjects is None:
        n_subjects = max_subjects
    if n_subjects > max_subjects:
        warnings.warn('Warning: there are only %d subjects' % max_subjects)
        n_subjects = max_subjects

    subs = subs[:n_subjects]
    subjects_funcs = subjects_funcs[:n_subjects]
    subjects_confounds = subjects_confounds[:n_subjects]

    dataset_name = 'adhd'
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)
    subjects_funcs = _fetch_files(data_dir, subjects_funcs, resume=resume,
                                  verbose=verbose)
    subjects_confounds = _fetch_files(data_dir, subjects_confounds,
            resume=resume, verbose=verbose)
    phenotypic = _fetch_files(data_dir, phenotypic, resume=resume,
                              verbose=verbose)[0]

    fdescr = _get_dataset_descr(dataset_name)

    # Load phenotypic data
    phenotypic = np.genfromtxt(phenotypic, names=True, delimiter=',',
                               dtype=None)
    # Keep phenotypic information for selected subjects
    isubs = np.asarray(subs, dtype=int)
    phenotypic = phenotypic[[np.where(phenotypic['Subject'] == i)[0][0]
                             for i in isubs]]

    return Bunch(func=subjects_funcs, confounds=subjects_confounds,
                 phenotypic=phenotypic, description=fdescr)

    
def load_adhd():