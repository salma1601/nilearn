# -*- coding: utf-8 -*-
"""
Test on servier dataset
"""
import sys
import os

import numpy as np
import matplotlib.pylab as plt
from matplotlib import pylab

from nilearn.connectivity import CovEmbedding, vec_to_sym
from nilearn.connectivity.embedding import map_sym
sys.path.append('/home/salma/CODE/servier2')
from my_conn import MyConn

def plot_matrix(mean_conn, title="connectivity", ticks=[], tick_labels=[],
                xlabel="", ylabel=""):
    """Plot connectivity matrix, for a given measure. """

    mean_conn = mean_conn.copy()

    # Put zeros on the diagonal, for graph clarity
    size = mean_conn.shape[0]
    mean_conn[range(size), range(size)] = 0
    vmax = np.abs(mean_conn).max()
    if vmax <= 2e-16:
        vmax = 0.1

    # Display connectivity matrix
    plt.figure()
    plt.imshow(mean_conn, interpolation="nearest",
              vmin=-vmax, vmax=vmax, cmap=plt.cm.get_cmap("bwr"))
    plt.colorbar()
    ax = plt.gca()
#    ax.xaxis.set_ticks_position('top')
    plt.xticks(ticks, tick_labels, size=8, rotation=90)
    plt.xlabel(xlabel)
    plt.yticks(ticks, tick_labels, size=8)
    ax.yaxis.tick_left()
    plt.ylabel(ylabel)

    plt.title(title)


overwrite = True
 # biyu's order
AN = ['vIPS_big', 'pIPS_big', 'MT_big', 'FEF_big', 'RTPJ', 'RDLPFC']
DMN = ['AG_big', 'SFG_big', 'PCC', 'MPFC', 'FP']
final = 1
if final:
#        WMN = ['IPL','MFG_peak1_peak2','LMFG_peak1','RMFG_peak2',
#              'CPL_peak1_peak3','RCPL_peak1','RCPL_peak2','LCPL_peak3','LT']
    #WMN = ['IPL','LMFG_peak1','CPL_peak1_peak3','LT']
    WMN = ['IPL', 'LMFG_peak1',  # 'RMFG_peak2' removed because of bad results
           'RCPL_peak1', 'LCPL_peak3', 'LT']
else:
    WMN = ['RT', 'LT', 'PL.cluster002', 'LFG',
           'MFG.cluster002', 'LPC.cluster001',
           'SPL.cluster002']  # main peaks in SPM.mat

template_ntwk = [('WMN', WMN), ('AN', AN), ('DMN', DMN)]
conn_folder = "/volatile/new/salma/subject1to40/conn_servier2_1to40sub_RS1-Nback2-Nback3-RS2_Pl-D_1_1_1"
mc = MyConn('from_conn', conn_folder)
mc.setup()
#fake_mc = MyConn(setup='test')
standardize = True
mc.analysis(template_ntwk, standardize, 'correlations', 'partial correlations')
#            'segregations', 'precisions')  # TODO include variablity

rs1_pl = "ReSt1_Placebo"
nb2_pl = "Nbac2_Placebo"
nb3_pl = "Nbac3_Placebo"
rs2_pl = "ReSt2_Placebo"
rs1_d = "ReSt1_Drug"
nb2_d = "Nbac2_Drug"
nb3_d = "Nbac3_Drug"
rs2_d = "ReSt2_Drug"
conditions = [(rs1_pl, nb2_pl), (rs1_pl, nb3_pl), (rs2_pl, nb2_pl),
              (rs2_pl, nb3_pl), (rs1_pl, rs1_d), (nb2_pl, nb2_d),
              (nb2_pl, nb2_d), (rs2_pl, rs2_d)]
conditions = [(rs1_pl, nb2_pl)]
# Between subjects covariates
n_subjects_A1 = 21
n_subjects_A2 = 19
n_subjects = 40
group_all = np.ones((1, n_subjects))
group_A1 = np.hstack(
(np.ones((1, n_subjects_A1)), np.zeros((1, n_subjects_A2))))
group_A2 = np.hstack(
(np.zeros((1, n_subjects_A1)), np.ones((1, n_subjects_A2))))
groups = {'All': group_all,
          'A1': group_A1,
          'A2': group_A2}

mean_matrices = []
all_matrices = []
measures = ["covariance", "precision", "tangent", "correlation",
             "partial correlation"]
from sklearn.covariance import EmpiricalCovariance, ShrunkCovariance,\
    LedoitWolf, GraphLassoCV, MinCovDet
estimators = [('ledoit', LedoitWolf()), ('emp', EmpiricalCovariance()),
              ('mcd', MinCovDet())]
n_estimator = 1
for kind in measures:
    cov_estimator = {'kind': kind,
                  'cov_estimator': estimators[n_estimator][1]}
#    for condition1, condition2 in conditions:
    condition1 = 'ReSt1_Placebo'
    condition2 = 'Nbac2_Placebo'
    print(condition1, condition2)
    signals1 = mc.runs_[condition1]
    signals2 = mc.runs_[condition2]
    subjects = []
    for subj1, subj2 in zip(signals1, signals2):
        subjects.append(subj1)
        subjects.append(subj2)
    subjects = signals1 + signals2  # Comment if checkboard output matrices
    cov_embedding = CovEmbedding(**cov_estimator)
    X = cov_embedding.fit_transform(subjects)
    print cov_estimator['kind']
    #fre_mean = cov_embedding.mean_cov_
    n_subjects = X.shape[0]  # / 2
    matrices = vec_to_sym(X)
    all_matrices.append(matrices)
    if kind == 'tangent':
        mean2 = cov_embedding.mean_cov_
    else:
        mean2 = matrices.mean(axis=0)
    mean_matrices.append(mean2)



#############################################
# Compute the distances between the matrices
#############################################
def disp(A, B):
    A_sqrt_inv = map_sym(lambda x: 1. / np.sqrt(x), A)
    return map_sym(np.log, A_sqrt_inv.dot(B).dot(A_sqrt_inv))


mat_dist = np.zeros((5, len(subjects), len(subjects)))
conns_all = all_matrices
conns_all[2] = conns_all[0]
#    conns_all = [np.vstack((conns, mean_matrices2[n][np.newaxis, ...])) for
#        (n, conns) in enumerate(conns_all)]
task_rest_dist = []
relative = False
normalized = False
for n, conns in enumerate(conns_all):
    norms = [np.linalg.norm(conn) for conn in conns]
    for sub_n in range(len(subjects)):
        if relative:
            if n == 2:
                mat_dist[n, sub_n] = [
#                    np.linalg.norm(disp(conns[sub_n], conn))\
#                    / np.sqrt(np.linalg.norm(conns[sub_n]) * np.linalg.norm(conn))
                     np.linalg.norm(disp(conns[sub_n] / np.linalg.norm(conns[sub_n]),
                                         conn / np.linalg.norm(conn)))
                    for conn in conns]
            else:
                mat_dist[n, sub_n] = [np.linalg.norm(conns[sub_n] - conn) \
#                    / np.sqrt(np.linalg.norm(conns[sub_n]) * np.linalg.norm(conn))
                    / np.mean(conns)
                    for conn in conns]
        else:
            if n == 2:
#                mat_dist[n, sub_n] = [np.linalg.norm(disp(conns[sub_n], conn))
                mat_dist[n, sub_n] = [np.linalg.norm(disp(conns[sub_n] / \
                    np.linalg.norm(conns[sub_n]), conn / np.linalg.norm(conn)))
                    for conn in conns]
            else:
                mat_dist[n, sub_n] = [np.linalg.norm(conns[sub_n] - conn)
                    for conn in conns]
    task_rest = np.diag(mat_dist[n][n_subjects / 2:, :n_subjects / 2])
    if normalized:
        task_rest /= np.median(task_rest)
    task_rest_dist.append(task_rest)
    within_subjects_mask = np.zeros((n_subjects, n_subjects), dtype=bool)
    within_subjects_mask[n_subjects / 2:, :n_subjects / 2] = True
    np.fill_diagonal(within_subjects_mask[n_subjects / 2:, :n_subjects / 2],
                     False)
    within_subjects_mask[:n_subjects / 2:, n_subjects / 2:] = True
    np.fill_diagonal(within_subjects_mask[:n_subjects / 2, n_subjects / 2:],
                     False)
    mat_dist[n][within_subjects_mask] = 0.
    percent = 75.
    mask = mat_dist[n] < np.percentile(mat_dist[n], percent)
    mat_dist[n][mask] = 0.
#    plot_matrix(mat_dist[n], title=measures[n], ticks=range(0, 80),
#                tick_labels=[str(tick) for tick in range(0, 80)])
    fig_title = 'conds_' + str(int(percent)) + measures[n] + '_' + \
        estimators[n_estimator][0]
    if relative:
        suffix = '_relative'
    else:
        suffix = ''
    filename = os.path.join(
        '/home/salma/slides/Parietal2/Images/sensitivity',
        fig_title + suffix + "2.pdf")
    if not os.path.isfile(filename) or overwrite:
        plot_matrix(mat_dist[n], title=measures[n], ticks=[20, 60],
                    tick_labels=['rest 1', 'task 2B'])
        pylab.savefig(filename)
        os.system("pdfcrop %s %s" % (filename, filename))
if relative:
    tick_labels = [['tangent'],
                   ['covariances',
                   'precisions',
                   'correlations',
                   'partial correlations']]
    datas = [task_rest_dist[2], task_rest_dist[:2] + task_rest_dist[3:]]
    names = ['tangent', 'all except tangent']

else:
    tick_labels = [['covariances'],
                   ['precisions'],
                   ['tangent',
                    'correlations',
                    'partial correlations']]
    datas = [task_rest_dist[0], task_rest_dist[1], task_rest_dist[2:]]
    names = ['cov', 'prec', 'tangent']
if normalized:
    tick_labels = [['covariances',
                   'precisions',
                   'tangent',
                   'correlations',
                   'partial correlations']]
    datas = [task_rest_dist]
    names = ['all']

for data, tick_label, name in zip(datas, tick_labels, names):
    plt.figure()
    plt.boxplot(data)
    plt.xticks(np.arange(len(tick_label)) + .5, tick_label, size=8,
               rotation=45)
    plt.title('distances between connectivity matrices at rest and task')
    if normalized:
        suffix_n = '_normalized'
    else:
        suffix_n = ''
    filename = os.path.join(
        '/home/salma/slides/Parietal2/Images/sensitivity',
        name + '_2b-vs-rs1' + suffix_n + '_boxplots' + suffix + '2' +\
        estimators[n_estimator][0] + '.pdf')
    if not os.path.isfile(filename) or overwrite:
        pylab.savefig(filename)
        os.system("pdfcrop %s %s" % (filename, filename))
plt.show()