# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 18:43:42 2015

@author: sb238920
"""
# Standard library imports

# Related third party imports

# Local application/library specific imports

import sys
import os

import numpy as np
import matplotlib.pylab as plt

from nilearn.connectivity import CovEmbedding, vec_to_sym
from nilearn.connectivity.embedding import cov_to_corr, prec_to_partial
from sklearn.covariance import EmpiricalCovariance, ShrunkCovariance,\
    LedoitWolf, GraphLassoCV, MinCovDet
sys.path.append('/home/sb238920/CODE/servier2')
from my_conn import MyConn
from comparing import plot_histograms


def conn_load(conn_folder, condition, template_ntwk=None):
    """Return the lists of rois and time series across subjects for a given
    condition.
    """
    if template_ntwk is None:
         # biyu's order
        AN = ['vIPS_big', 'pIPS_big', 'MT_big', 'FEF_big', 'RTPJ', 'RDLPFC']
        DMN = ['AG_big', 'SFG_big', 'PCC', 'MPFC', 'FP']
        WMN = ['IPL', 'LMFG_peak1', 'RCPL_peak1', 'LCPL_peak3', 'LT']
        template_ntwk = [('WMN', WMN), ('AN', AN), ('DMN', DMN)]

    mc = MyConn('from_conn', conn_folder)
    mc.setup()
    mc.preproc(template_ntwk)
    rois = [roi for (ntwk, names) in template_ntwk for roi in names]
    return rois, mc.runs_[condition]


def compute_connectivity(subjects, estimator=EmpiricalCovariance(),
                         measure='correlation'):
    """Return individual connectivity matrices and the group mean matrix.

    Parameters
    ==========
    measure : str
        One of "covariance", "precision", "tangent", "correlation",
        "partial correlation"
    """
    cov_estimator = {'cov_estimator': estimator, 'kind': measure}
    cov_embedding = CovEmbedding(**cov_estimator)
    matrices = vec_to_sym(cov_embedding.fit_transform(subjects))
    if measure == 'tangent':
        mean = cov_embedding.mean_cov_
    else:
        mean = matrices.mean(axis=0)
    return matrices, mean


def split_motion(motions, percent=50, ord=None):
    """Classifies motion as high and low.
    Parameters
    ==========
    motions : array like, shape (n_subjects, n_samples)
        Motion values for each subject.
    percent : float, optional
        Percent of high motion
    ord :  {non-zero int, inf, -inf, ‘fro’}, optional
       Order of the norm, passed to np.linalg.norm().

    Returns
    =======
    labels : array of int
        Label of each motion as low (1) or high (0)
    """
    norms = np.linalg.norm(motions, ord=ord, axis=-1)
    n_subjects = len(motions)
    labels = np.zeros((n_subjects,))
    labels[norms.argsort()[: percent * n_subjects / 100]] = 1
    return labels

# TODO: plot connectivity matrix with distances matrix
if __name__ == '__main__':
    data_path = '/volatile/new/salma'
    conn_folders = [os.path.join(data_path, folder, 'conn_study') for folder
                    in ['subject1to40_noDespike', 'subject1to40_noFilt']]
    preprocs = ['no Despike', 'no Filt']
    multiple_coefs = []
    preprocessings = []
    for conn_folder, preproc in zip(conn_folders, preprocs):
        subjects_file = os.path.join(conn_folder, 'signals_ReSt1_Drug.npy')
        if not os.path.isfile(subjects_file):
            _, subjects_rest = conn_load(conn_folder, 'ReSt1_Drug')
            np.save(subjects_file, subjects_rest)
        else:
            subjects_rest = np.load(subjects_file)
        plt.figure()
        plt.plot(subjects_rest[0])
        plt.title(preproc)
        matrices, mean = compute_connectivity(subjects_rest,
                                              measure='correlation')
        n_regions = matrices.shape[-1]
        coefs = [matrix[np.triu_indices(n_regions, k=1)] for matrix in
                 matrices]
        multiple_coefs.append((np.array(coefs)[:, 0]).flatten())
        preprocessings.append(preproc + 'subj 0')
#        multiple_coefs.append((np.array(coefs)[1]).flatten())
#        preprocessings.append(preproc + 'subj 1')
#        multiple_coefs.append((mean[np.triu_indices(n_regions, k=1)][0]).flatten())
#        preprocessings.append(preproc + 'mean')
    plt.show()
#    plot_histograms(multiple_coefs, preprocessings, colors=[], title='',
#                    xlabel='')
    print mean[np.triu_indices(n_regions, k=1)][0]
    subjects_placebo = np.load(os.path.join(conn_folder, 'signals_ReSt1_Placebo.npy'))
    subjects_drug = np.load(os.path.join(conn_folder, 'signals_ReSt1_Drug.npy'))
    from scipy.stats import pearsonr
    for n, (pl, dr) in enumerate(zip(subjects_placebo, subjects_drug)):
        print '----------'
        for r, (ts1, ts2) in enumerate(zip(pl.transpose(), dr.transpose())):
            if pearsonr(ts1, ts2)[0] > 0.2:
                print pearsonr(ts1, ts2), r
                pass
                plt.subplot(2, 1, 1)
                plt.plot(ts1)
                plt.subplot(2, 1, 2)
                plt.plot(ts2)
                plt.title('sub {} '.format(n))
    plt.show()
    
    # TODO: compare within subject across ROIs, within ROI across subjects
    plt.subplot(6, 1, 1)
    plt.plot(subjects_placebo[s1][:, 0])
    plt.ylabel('sub 40, region {}')
    plt.subplot(6, 1, 2)
    plt.plot(dr[:, 0])
    plt.subplot(6, 1, 3)
    plt.plot(pl[:, 1])
    plt.subplot(6, 1, 4)
    plt.plot(dr[:, 1])
    plt.subplot(6, 1, 5)
    plt.plot(subjects_placebo[0][:, 0])
    plt.subplot(6, 1, 6)
    plt.plot(subjects_drug[0][:, 0])
    plt.show()