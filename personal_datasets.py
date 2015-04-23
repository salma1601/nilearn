import os
import warnings

import numpy as np
from sklearn.datasets.base import Bunch


def fetch_adhd_neuroimage(n_subjects=None, data_dir=None):
    """Load the ADHD resting-state dataset for the NeuroImage site.

    Parameters
    ----------
    n_subjects: int, optional
        The number of subjects to load. If None is given, all the
        40 subjects are used.

    data_dir: string
        Path of the data directory.

    Returns
    -------
    data: sklearn.datasets.base.Bunch
        Dictionary-like object, the interest attributes are :
         - 'func': string list. Paths to functional images
         - 'parameters': string list. Parameters of preprocessing steps

    References
    ----------
    :Download:
        http://www.nitrc.org/frs/downloadlink.php/3260
    """
    # Subjects ID per file
    subs = ['1017176', '1125505', '1208586', '1312097', '1411495', '1438162',
            '1538046', '1585708', '1588809', '2029723', '2074737', '2352986',
            '2419464', '2574674', '2671604', '2756846', '2876903', '2961243',
            '3007585', '3048588', '3082137', '3108222', '3190461', '3304956',
            '3322144', '3449233', '3515506', '3566449', '3808273', '3858891',
            '3888614', '3941358', '3959823', '3980079', '4020830', '4134561',
            '4239636', '4285031', '4919979', '5045355', '6115230', '7339173',
            '7446626', '7504392', '8387093', '8409791', '8991934', '9956994']

    subjects_funcs = [os.path.join(
        data_dir, 'sfnwmrdasubject{}_session_1_rest_1.nii.gz'.format(i))
        for i in subs]

    phenotypic = os.path.join(data_dir, 'NeuroIMAGE_phenotypic.csv')

    max_subjects = len(subjects_funcs)
    # Check arguments
    if n_subjects is None:
        n_subjects = max_subjects
    if n_subjects > max_subjects:
        warnings.warn('Warning: there are only %d subjects' % max_subjects)
        n_subjects = max_subjects

    subs = subs[:n_subjects]
    subjects_funcs = subjects_funcs[:n_subjects]

    # Load phenotypic data
    phenotypic = np.genfromtxt(phenotypic, names=True, delimiter=',',
                               dtype=None)
    # Keep phenotypic information for selected subjects
    isubs = np.asarray(subs, dtype=int)
    print phenotypic
    phenotypic = phenotypic[[np.where(phenotypic['ScanDir ID'] == i)[0][0]
                             for i in isubs]]

    return Bunch(func=subjects_funcs, phenotypic=phenotypic)