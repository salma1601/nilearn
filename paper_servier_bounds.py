# -*- coding: utf-8 -*-
"""
Test on servier dataset
"""
import sys
import os

import numpy as np
import matplotlib.pylab as plt

from nilearn.connectivity import CovEmbedding, vec_to_sym
from nilearn.connectivity.embedding import cov_to_corr
sys.path.append('/home/sb238920/CODE/servier2')
from my_conn import MyConn
#from compute_precision import plot_matrix
def corr_to_Z(corr, tol=1e-7):
    """Applies Z-Fisher transform. """
    Z = corr.copy()  # avoid side effects
    corr_is_one = 1.0 - abs(corr) < tol
    Z[corr_is_one] = np.inf * np.sign(Z[corr_is_one])
    Z[np.logical_not(corr_is_one)] = \
        np.arctanh(corr[np.logical_not(corr_is_one)])
    return Z


def plot_matrix(mean_conn, title="connectivity", ticks=[], tick_labels=[],
                xlabel="", ylabel=""):
    """Plot connectivity matrix, for a given measure. """

    mean_conn = mean_conn.copy()

    # Put zeros on the diagonal, for graph clarity
#    size = mean_conn.shape[0]
#    mean_conn[range(size), range(size)] = 0
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
from sklearn.covariance import EmpiricalCovariance
mean_matrices = []
all_matrices = []
for kind in ["covariance", "precision", "tangent", "correlation",
             "partial correlation"]:
    estimators = {'kind': kind, 'cov_estimator': EmpiricalCovariance()}
#    for condition1, condition2 in conditions:
    condition1 = 'ReSt1_Placebo'
    condition2 = 'Nbac2_Placebo'
    print(condition1, condition2)
    signals1 = mc.runs_[condition1]
    signals2 = mc.runs_[condition2]
    signals_list = [subj for subj in signals1]# + [subj for subj in signals2]
    cov_embedding = CovEmbedding(**estimators)
    X = cov_embedding.fit_transform(signals_list)
    print estimators['kind']
    #fre_mean = cov_embedding.mean_cov_
    nsubjects = X.shape[0]# / 2
    matrices = vec_to_sym(X)
    all_matrices.append(matrices)
    if kind == 'tangent':
        mean2 = cov_embedding.mean_cov_
    else:
        mean2 = matrices.mean(axis=0)
    mean_matrices.append(mean2)

# Find regions
larger = []
lower = []
n_regions = mean_matrices[0].shape[-1]
for x, y in zip(*np.triu_indices(n_regions, k=1)):
    if mean_matrices[3][x, y] > mean_matrices[4][x, y] + 0.2:
        larger.append((x, y))
    if mean_matrices[4][x, y] > mean_matrices[3][x, y] + 0.2:
        lower.append((x, y))

lower = []
larger = []
percent = .90
prop = percent * n_subjects
for x, y in zip(*np.triu_indices(n_regions, k=1)):
    if np.sum(all_matrices[3][:, x, y] > all_matrices[4][:, x, y] + .1) > prop:
        larger.append((x, y))
    if np.sum(all_matrices[4][:, x, y] > all_matrices[3][:, x, y] + .05) > prop:
        lower.append((x, y))

for regions in larger:
    coefs = []
    mean_coef = []
    for mean, matrices in zip([cov_to_corr(mean_matrices[2]),
                               mean_matrices[4]],
                              [all_matrices[3], all_matrices[4]]):
        coefs.append(matrices[:, regions[0], regions[1]])
        amean = mean_matrices[3][regions]
        mean_coef.append(mean[regions])
    plt.figure()
    plt.boxplot(coefs, whis=np.inf)
    lineObjects = plt.plot(1., mean_coef[0], '^', color='r') + \
        plt.plot(1., amean, '*', color='b')+ \
        plt.plot(2., mean_coef[1], 'o', color='g')
    plt.xticks(np.arange(2) + 1., ['correlations', 'partial correlations'],
               size=8)
    plt.legend(iter(lineObjects), ('corr(gmean)', 'corrs mean',
               'partials mean'))
    plt.title('regions {0}, {1}'.format(regions[0], regions[1]))
        
plt.show()

# Compare correlations - partial correlation to correlations - gmean
from nilearn.connectivity.tests.test_embedding_tmp2 import sample_wishart,\
    sample_spd_normal
from nilearn.connectivity.embedding import geometric_mean, cov_to_corr, prec_to_partial


def synthetic_data_wishart(sigma, n_samples=201, n_subjects=40):
    for n in range(0, 40):
        rand_gen = np.random.RandomState(0)
        spds = []
        for k in range(n_subjects):
#            spd = random_wishart(sigma, dof=n_samples, rand_gen=rand_gen)
            spd = sample_wishart(sigma, dof=n_samples, rand_gen=rand_gen)
#            if cov_to_corr(spd)[0, 1] - prec_to_partial(np.linalg.inv(spd))[0, 1] > 0.2 \
#                and (cov_to_corr(spd)[0, 1] - prec_to_partial(np.linalg.inv(spd))[0, 1] < 0.5):
            spds.append(spd)
        if not spds:
            print 'no'

    return spds


def synthetic_data_manifold(mean, cov=None, n_subjects=40):
    rand_gen = np.random.RandomState(0)  # TODO: variable random state?
    spds = []
    if False:
        for n in range(0, 40):
            for k in range(n_subjects):
                spd = sample_spd_normal(mean, cov=cov, rand_gen=rand_gen)
                spds.append(spd)
    from nilearn.connectivity.embedding import map_sym, vec_to_sym
    mean_sqrt = map_sym(np.sqrt, mean)
    p = mean.shape[0] * (mean.shape[0] + 1) / 2
    if cov is None:
        cov = np.eye(p)
    for n in range(n_subjects):
        tangent = rand_gen.multivariate_normal(np.zeros(cov.shape[0]), cov)
        tangent = vec_to_sym(tangent)
        tangent_exp = map_sym(np.exp, tangent)
        spds.append(mean_sqrt.dot(tangent_exp).dot(mean_sqrt))
    return spds


subject_n = 0
n_samples = np.mean([subject.shape[0] for subject in signals_list])
n_samples = int(n_samples)
n_subjects *= 2
spds = synthetic_data_wishart(mean_matrices[0] / n_samples,
                              n_samples=n_samples, n_subjects=n_subjects)
#spds = synthetic_data_manifold(mean_matrices[2],
#                               n_subjects=2)
geo = geometric_mean(spds)
corrs = [cov_to_corr(spd) for spd in spds]
partials = [prec_to_partial(np.linalg.inv(spd)) for spd in spds]
plot_matrix(geo, "gmean, whishart")
plot_matrix(mean_matrices[2], "gmean, data")
plot_matrix(np.mean(spds, axis=0), "amean, whishart")
plot_matrix(np.mean(spds, axis=0) - geo, "amean - geo, whishart")
plot_matrix(mean_matrices[0], "amean, data")
plt.show()
#plot_matrix(cov_to_corr(geo), "corr(gmean), whishart")
#plot_matrix(np.mean(corrs, axis=0), "amean of corrs, whishart")
#plot_matrix(np.mean(partials, axis=0), "amean of partials, whishart")
plot_matrix(np.mean(corrs, axis=0) - cov_to_corr(geo),
            "mean of corrs - corr(gmean), whishart")
plot_matrix(mean_matrices[3] - cov_to_corr(mean_matrices[2]),
            "mean of corrs - corr(gmean), data")
plot_matrix(np.mean(corrs, axis=0) - np.mean(partials, axis=0),
            "mean of corrs - mean of partials, whishart")
plot_matrix(mean_matrices[3] - np.mean(partials, axis=0),
            "mean of corrs - mean of partial corrs, data")
plot_matrix(cov_to_corr(geo) - mean_matrices[4],
            "corr(gmean) - mean of partial corrs, wishart")
plot_matrix(cov_to_corr(mean_matrices[2]) - mean_matrices[4],
            "corr(gmean) - mean of partial corrs, data")
plt.show()
ser, = plt.plot((mean_matrices[3] - cov_to_corr(mean_matrices[2])).ravel(),
                (mean_matrices[3] - mean_matrices[4]).ravel(), 'g.',
                label='servier dataset')
plt.xlabel('corrs - corr(gmean)')
plt.ylabel('corrs - partials')
wish, = plt.plot((np.mean(corrs, axis=0) - cov_to_corr(geo)).ravel(),
                 (np.mean(corrs, axis=0) - np.mean(partials, axis=0)).ravel(),
                 'r.', label='wishart distribution')
plt.legend()
plt.title('differences between connectivity measures over regions')
from scipy.stats import pearsonr
r, pval = pearsonr((mean_matrices[3] - cov_to_corr(mean_matrices[2])).ravel(),
                   (mean_matrices[3] - mean_matrices[4]).ravel())
print('pearson corr = {}, pval = {}'.format(r, pval))
plt.show()

ser_plot, = plt.plot(np.triu(corr_to_Z(mean_matrices[3]), k=1).ravel(),
                      np.triu(corr_to_Z(mean_matrices[4]), k=1).ravel(), 'g*',
                      label='servier dataset')