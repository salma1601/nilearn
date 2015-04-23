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


overwrite = False
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

measures = ["covariance", "precision", "tangent", "correlation",
             "partial correlation"]
from sklearn.covariance import EmpiricalCovariance, ShrunkCovariance,\
    LedoitWolf, GraphLassoCV, MinCovDet
estimators = [('ledoit', LedoitWolf()), ('emp', EmpiricalCovariance()),
              ('mcd', MinCovDet())]
n_estimator = 0
conditions = [rs1_pl, nb2_pl, rs1_d, nb2_d]
conditions_couples = [(rs1_pl, nb2_pl), (rs1_pl, rs2_pl)]


def disp(A, B):
    A_sqrt_inv = map_sym(lambda x: 1. / np.sqrt(x), A)
    return map_sym(np.log, A_sqrt_inv.dot(B).dot(A_sqrt_inv))

flattened_dist = [[], [], [], [], []]
for n_cond, (condition1, condition2) in enumerate(conditions_couples):
    mean_matrices = []
    all_matrices = []
    for kind in measures:
        cov_estimator = {'kind': kind,
                      'cov_estimator': estimators[n_estimator][1]}
        subjects = mc.runs_[condition1] + mc.runs_[condition2]
        cov_embedding = CovEmbedding(**cov_estimator)
        X = cov_embedding.fit_transform(subjects)
        n_subjects = X.shape[0]
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
    
    n_conditions = len(conditions_couples)
    mat_dist = np.zeros((5, len(subjects), len(subjects)))
    conns_all = all_matrices
    conns_all[2] = conns_all[0]
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
                        / np.mean(conns) for conn in conns]
            else:
                if n == 2:
                    mat_dist[n, sub_n] = [np.linalg.norm(disp(conns[sub_n], conn))
    #                mat_dist[n, sub_n] = [np.linalg.norm(disp(conns[sub_n] / \
    #                    np.linalg.norm(conns[sub_n]), conn / np.linalg.norm(conn)))
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
        percent = 90
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
            fig_title + suffix + str(percent) + ".pdf")
        if not os.path.isfile(filename) or overwrite:
            plot_matrix(mat_dist[n], title=measures[n], ticks=[20, 60],
                        tick_labels=['rest 1', 'task 2B'])
            pylab.savefig(filename)
            os.system("pdfcrop %s %s" % (filename, filename))
        flattened_dist[n].append(task_rest)


tick_labels = ['rest1 vs 2B task',
               'rest1 vs rest2']

for n, measure in enumerate(measures):
    datas = flattened_dist[n]
    plt.figure()
    plt.boxplot(datas) #, whis=np.inf from minimum to maximum, no outlier
    plt.xticks(np.arange(len(tick_labels)) + 1., tick_labels, size=8)
    plt.title('within subjects distances between ' + measure + ' matrices')
    if normalized:
        suffix_n = '_normalized'
    else:
        suffix_n = ''
    filename = os.path.join(
        '/home/salma/slides/Parietal2/Images/sensitivity',
        '_2b-rs1-rs2' + suffix_n + '_boxplots' + suffix + '2' +\
        estimators[n_estimator][0] + '.pdf')
    if not os.path.isfile(filename) or overwrite:
        pylab.savefig(filename)
        os.system("pdfcrop %s %s" % (filename, filename))
plt.show()

from scipy.stats import pearsonr

titles = ['Euclidean btw. covariances', 'Euclidean btw. precisions',
          'Riemannian btw. covariances', 'Euclidean btw. correlations',
          'Euclidean btw. partial correlations']
for (n1, n2) in [(2, 4), (2, 3), (3, 4)]:
    for cond, name in enumerate(tick_labels):
        name = name.replace(' ','-')
        print n1, n2, cond
        plt.figure(figsize=(4, 3))
        plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
        x = flattened_dist[n1][cond]
        y = flattened_dist[n2][cond]
        plt.scatter(x, y)
        plt.xlabel(titles[n1])
        plt.ylabel(titles[n2])
#        plt.title('distances scatter for ' + name)
        # Plot line of best fit if significative Pearson correlation
        rho, p_val = pearsonr(x, y)
        if p_val < 0.01:
            print p_val, name, measures[n1], measures[n2]
            plt.text(.5, .5, str(p_val))
            fit = np.polyfit(x, y, 1)
            fit_fn = np.poly1d(fit)
            plt.plot(x, fit_fn(x), linestyle='-', c='red')
        filename = os.path.join(
            '/home/salma/slides/NiConnect/Images/distances',
            'scatter_' + measures[n1][:4] + '_' + measures[n2][:4] + '_' + name\
            + ".pdf")
        pylab.savefig(filename)
        os.system("pdfcrop %s %s" % (filename, filename))
plt.show()


# Boxplots for rest1 vs rest2 condition
# tangent, corr, part
tick_labels = ['Riemanian btw.\ncovariances', 
               'Euclidean btw.\ncorrelations',
               'Euclidean btw.\npartial correlations']
datas = [flattened_dist[2][1], flattened_dist[3][1], flattened_dist[4][1]]
fig = plt.figure(figsize=(4, 3.2))
ax = fig.add_subplot(111)
#plt.subplots_adjust(hspace=0.,wspace=0.5)
ax.boxplot(datas)  # , whis=np.inf from minimum to maximum, no outlier
#ax.set_aspect(.8)
ax.set_xticks(np.arange(len(tick_labels)) + 1.)
ax.set_xticklabels(tick_labels, size=9)
#plt.title('within subjects distances')
filename = os.path.join(
    '/home/salma/slides/NiConnect/Images/sensitivity',
    'rs1-rs2' + '_boxplots' + suffix + '2' +\
    estimators[n_estimator][0] + '.pdf')
pylab.savefig(filename)
os.system("pdfcrop %s %s" % (filename, filename))

# cov and prec
all_tick_labels = [['Euclidean btw.\ncovariances'],
                   ['Euclidean btw.\nprecisions']]
all_datas = [[flattened_dist[0][1]], [flattened_dist[1][1]]]
for datas, tick_labels in zip(all_datas, all_tick_labels):
    fig = plt.figure(figsize=(3, 4.5))  #(4.5, 5.5) (3.2, 4.26) (3.5, 4.5)
#    plt.subplots_adjust(hspace=6.8,wspace=0.5)
    plt.subplots_adjust(left=0.19, right=0.9, top=0.9, bottom=0.15)
    ax = fig.add_subplot(111)
    ax.boxplot(datas)  #, whis=np.inf from minimum to maximum, no outlier 
    #, positions=[.5]
#    plt.xticks(np.arange(len(tick_labels)) + 1., tick_labels, size=12)
    ax.set_xticks(np.arange(len(tick_labels)) + 1.)
    ax.set_xticklabels(tick_labels)
#    ax.set_xlim([.9, 1.1])
    ax.tick_params(axis='both', which='major', labelsize=15)
    filename = os.path.join(
        '/home/salma/slides/NiConnect/Images/sensitivity',
        'rs1-rs2' + '_boxplots' + suffix +'_' + tick_labels[0][-10:-6]+\
        estimators[n_estimator][0] + '.pdf')
    pylab.savefig(filename)
    os.system("pdfcrop %s %s" % (filename, filename))

plt.show()