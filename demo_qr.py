import nilearn.datasets
import nilearn.testing
import nilearn.region

import utils


def benchmark():
    """ """
    n_regions = 1500

    print("Loading data ...")
    adhd = nilearn.datasets.fetch_adhd()
    filename = adhd["func"][0]
    img = nilearn.utils.check_niimg(filename)
    shape = img.shape[:3]
    affine = img.get_affine()
    _ = img.get_data()  # Preload data
    print("Generating regions ...")
    regions = nilearn.testing.generate_labeled_regions_large(shape, n_regions,
                                                          affine=affine)
    signals, labels = utils.timeit(profile(nilearn.region.img_to_signals_labels)
                                   )(img, regions)
    img_r = utils.timeit(profile(nilearn.region.signals_to_img_labels)
                         )(signals, regions, order='C')


if __name__ == "__main__":
    benchmark()


import timeit
import time

import numpy as np
import scipy
from scipy.linalg import lstsq, qr
from nilearn.signal import _standardize
from nilearn.tests.test_signal import generate_signals


t = timeit.Timer("foo(num1, num2)", "from myfile import foo")
t.timeit(5)

setup = """
 # some pre-defined constants
A = 1
B = 2

# function that does something critical
def foo(num1, num2):
    # do something

# main program.... do something to A and B
for i in range(20):
    # do something to A and B
    # and update A and B during each iteration
"""

t = timeit.Timer("foo(num1, num2)", setup)
t.timeit(5)


signals = []
to_regress = []
length = 400
for n_features in range(15, 100):
    signal, noise, confounds = generate_signals(n_features=n_features,
                                                n_confounds=25,
                                                length=length,
                                                same_variance=False)
    signals.append(signal + noise)
    to_regress.append(confounds)


def qr_manual(to_clean, confounds):
    Q, R = qr(confounds, mode='economic')
    non_null_diag = np.abs(np.diag(R)) > np.finfo(np.float).eps * 100.
    if np.all(non_null_diag):
        cleaned = to_clean - Q.dot(Q.T).dot(to_clean)
    elif np.any(non_null_diag):
        R = R[:, non_null_diag]
        confounds = confounds[:, non_null_diag]
        inv = scipy.linalg.inv(np.dot(R.T, R))
        cleaned = to_clean - confounds.dot(inv).dot(confounds.T).dot(to_clean)
    print('{0} confounds, rank {1}, precision = {1}'.format(confounds.shape[0],
          np.linalg.matrix_rank(confounds),
          np.linalg.norm(confounds.T.dot(cleaned))))



def qr_pivot(to_clean, confounds):
    Q, R, _ = qr(confounds, mode='economic', pivoting=True)
    Q_full = Q[:, np.abs(np.diag(R)) > np.finfo(np.float).eps * 100]
    cleaned = to_clean - np.dot(Q_full, np.dot(Q_full.T, to_clean))
    return cleaned


def np_pinv(to_clean, confounds):
    cleaned = to_clean - confounds.dot(np.linalg.pinv(confounds).dot(to_clean))
    return cleaned


def np_lstsq(to_clean, confounds):
    cleaned = to_clean - confounds.dot(np.linalg.lstsq(confounds, to_clean)[0])
    return cleaned


rand_gen = np.random.RandomState(0)
    for rank in ['full', 'non-full']:
        for to_clean, confounds in zip(signals, to_regress):
            # Generate full rank confounds
            confounds_full = np.hstack((confounds, rand_gen.randn(length, 1)))

            # Generate non-full rank confounds
            if rank == 'non-full':
                confounds_tmp = np.hstack((np.ones((length, 1)),
                                           confounds[:, :2]))
                confounds = np.hstack((confounds[:, 2:], confounds_tmp))

            # Standardize the confounds, as done in signal.clean
            confounds = _standardize(confounds, detrend=False, normalize=True)

            # Clean
            cleaned = regress(to_clean, confounds)
            cleaned = to_clean
        print('  {0}: time = {1}, precision = {2}, {3} rank'.format(name,
              elapsed, np.linalg.norm(confounds.T.dot(cleaned)), rank))

