import glob


import numpy as np
import itertools


def single_glob(pattern):
    filenames = glob.glob(pattern)
    if not filenames:
        raise ValueError('No file for pattern {}'.format(pattern))
    if len(filenames) > 1:
        raise ValueError('Non unique file for pattern {}'.format(pattern))
    return filenames[0]


def crescendo_replacement(initial_list, replacement_tuples):
    """Returns a list of lists obtained by replacing gradually the specified
    elements.

    Parameters
    ==========
    initial_list: list
        The list ou update.

    replacement_tuples: list of tuples.
        Each tuple is a pair of an element of the list and its replacing
        element.

    Returns
    =======
    updates: list of lists
        The updates lists.
    """
    updates = []
    new_list = initial_list
    for old, new in replacement_tuples:
        new_list = [element if element != old else new for element in new_list]
        updates.append(new_list)

    return updates


def combination_replacement(initial_list, replacement_tuples):
    """Returns a list of lists obtained by replacing gradually the specified
    elements.

    Parameters
    ==========
    initial_list: list
        The list ou update.

    replacement_tuples: list of tuples.
        Each tuple is a pair of an element of the list and its replacing
        element.

    Returns
    =======
    updates: dictionary
        Keys are the number of replaced items, values are the updates lists.
    """
    updates = {}
    for n_updated in range(len(replacement_tuples) + 1):
        combinations = itertools.combinations(replacement_tuples, n_updated)
        updates[n_updated] = []
        for combination in combinations:
            new_list = list(initial_list)  # copy to avoid side effects           
#            olds, news = zip(*combination)
            # TODO: remove for loop
            for old, new in combination:
                new_list[new_list.index(old)] = new
            updates[n_updated].append(new_list)
    return updates


def group_left_right(names, signals):
    """Groups signals for left and right ROIs.

    Parameters
    ==========
    names : list of str, length n_rois
        ROIs names, assumed to start with 'R ' and 'L ' for right and left
        regions.

    signals : numpy.ndarray, shape (n_samples, n_rois)
        Signals within each ROI

    Returns
    =======
    names_grouped : list of str, length n_rois_grouped
        ROIs names, without prefixes 'R ' and 'L '.

    signals_grouped : numpy.ndarray, shape (n_samples, n_rois_grouped)
        Signals within grouped ROIs. For lateralized ROIs, this is the average
        signal for the left and right ROIs.
    """
    names_grouped = []
    for name in names:
        if 'R ' in name:
            names_grouped.append(name.replace('R ', ''))
        elif 'L ' in name:
            names_grouped.append(name.replace('L ', ''))
        else:
            names_grouped.append(name)

    signals_grouped = signals.copy()
    names_grouped = np.array(names_grouped)
    for name in names_grouped:
        if np.sum(names_grouped == name) == 2:
            for idx in np.where(names_grouped == name)[0]:
                signals_grouped[:, idx] = np.mean(
                    signals[:, names_grouped == name], axis=1)

    _, indices = np.unique(names_grouped, return_index=True)
    names_grouped = names_grouped[np.sort(indices)]
    signals_grouped = signals_grouped[:, np.sort(indices)]
    return list(names_grouped), signals_grouped


from scipy.io import loadmat


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
        pass
        
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
    pass