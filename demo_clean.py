# -*- coding: utf-8 -*-a
"""
Created on Sun Apr 26 08:06:56 2015

@author: sb238920
"""
import numpy as np
from scipy.linalg import lstsq, qr
from nilearn.signal import clean, _standardize
from nilearn.tests.test_signal import generate_signals

# Generate a normalized signal
length = 30
signal, noise, confounds = generate_signals(n_features=1, n_confounds=5,
                                            length=length)
to_clean = _standardize(signal + noise, detrend=False, normalize=True)

# Generate full rank confounds, with a confound non present in the signal
confounds_full = np.hstack((confounds, np.random.rand(length, 1)))

# Generate non-full rank confounds
confounds_twice = np.hstack((confounds,
                             confounds[:, -1][:, np.newaxis]))
confounds_cst = np.hstack((np.ones((length, 1)), confounds[:, :2]))
confounds_cst = np.hstack((confounds[:, 2:], confounds_cst))
confounds_cst = np.hstack((confounds_cst,
                           confounds_cst[:, -1][:, np.newaxis]))


# Compare with scipy residuals
for confounds in [confounds_full, confounds_cst, confounds_twice]:
    # Standardize the confounds, as done in signal.clean
    confounds = _standardize(confounds, detrend=False, normalize=True)
    cleaned_signal = clean(to_clean, confounds=confounds, detrend=False,
                           standardize=False)
    beta = lstsq(confounds, to_clean)[0]
    residual = to_clean - confounds.dot(beta)
    print('{0} confounds, rank is {1}, mismatch is {2}'.format(
        confounds.shape[-1],
        np.linalg.matrix_rank(confounds),
        np.max(np.abs(cleaned_signal - residual))))

    # Automatic pivoting
    Q, R, _ = qr(confounds, mode='economic', pivoting=True)
    Q_full = Q[:, np.abs(np.diag(R)) > np.finfo(np.float).eps * 100]
    cleaned = to_clean - np.dot(Q_full, np.dot(Q_full.T, to_clean))
#    np.testing.assert_allclose(cleaned, residual)

    # Manual pivoting
    Q, R = qr(confounds, mode='economic')
    confounds_full = confounds[:, np.abs(np.diag(R)) > np.finfo(np.float).eps * 100]
    Q, R = qr(confounds_full, mode='economic')
    cleaned = to_clean - np.dot(Q_full, np.dot(Q_full.T, to_clean))
#    np.testing.assert_allclose(cleaned, residual)
