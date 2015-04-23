"""
Test on servier dataset
"""
import sys
import os

import numpy as np
import matplotlib.pylab as plt

from nilearn.connectivity import CovEmbedding, vec_to_sym
from nilearn.connectivity.embedding import sym_to_vec
sys.path.append('/home/salma/CODE/servier2')
from my_conn import MyConn
from compute_precision import plot_matrix


def corr_to_Z(corr, tol=1e-7):
    """Applies Z-Fisher transform. """
    Z = corr.copy()  # avoid side effects
    corr_is_one = 1.0 - abs(corr) < tol
    Z[corr_is_one] = np.inf * np.sign(Z[corr_is_one])
    Z[np.logical_not(corr_is_one)] = \
        np.arctanh(corr[np.logical_not(corr_is_one)])
    return Z


 # biyu's order
AN = ['vIPS', 'pIPS', 'MT', 'FEF', 'RTPJ', 'RDLPFC']
DMN = ['AG', 'SFG', 'PCC', 'MPFC', 'FP']
Retino = ['vRetino', 'dRetino']
h_visual = ['hv1', 'hv2']
p_visual = ['pv1', 'pv2']
recn = ['recn1', 'recn2', 'recn3', 'recn4', 'recn5', 'recn6']
visuo = ['visuo1', 'visuo2', 'visuo3', 'visuo4', 'visuo5', 'visuo6', 'visuo7',
         'visuo8', 'visuo11']
biyu = False
template_biyu = [('Retino', Retino), ('AN', AN), ('DMN', DMN)]
template_grecius = [('HVN', h_visual), ('PVN', p_visual),
                 ('RDLPFC-parietal', recn), ('IPS-FEF', visuo)]

if biyu:
    template_ntwk = template_biyu
else:
    template_ntwk = template_grecius

template_ntwk = template_biyu + template_grecius

conn_folder = "/media/Elements/conn_projects/ds107_outputs/sub1to49/conn_study"
mc = MyConn('from_conn', conn_folder)
mc.setup()
#fake_mc = MyConn(setup='test')
standardize = True
mc.analysis(template_ntwk, standardize, 'correlations', 'partial correlations')
#            'segregations', 'precisions')  # TODO include variablity

rs1_pl = "Words"
nb2_pl = "Objects"
nb3_pl = "Scrambled objects"
rs2_pl = "Consonant strings"
conditions = [(rs1_pl, nb2_pl), (rs1_pl, nb3_pl), (rs2_pl, nb2_pl),
              (rs2_pl, nb3_pl)]
conditions = [(rs1_pl, nb2_pl)]
n_subjects = 48
group_all = np.ones((1, n_subjects))
groups = {'All': group_all}
from sklearn.covariance import EmpiricalCovariance
mean_matrices = []
all_matrices = []
measures = ["covariance", "precision", "tangent", "correlation",
             "partial correlation"]
for kind in measures:
    estimators = {'kind': kind, 'cov_estimator': EmpiricalCovariance()}
#    for condition1, condition2 in conditions:
    condition1 = rs1_pl
    condition2 = nb3_pl
    print(condition1, condition2)
    signals1 = mc.runs_[condition1]
    signals2 = mc.runs_[condition2]
    signals_list = [subj for subj in signals1] + [subj for subj in signals2]
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

    X = [vec_to_sym(x) for x in X]
    X = np.asarray(X)
    mc.fc_[(condition1, estimators['kind'])] = X[:nsubjects, :]
    mc.fc_[(condition2, estimators['kind'])] = X[nsubjects:, :]
    mc.measure_names = mc.measure_names + (estimators['kind'], )
    paired_tests = [([condition1, condition2], ['All'])]
    if False:
        mc.results(paired_tests, corrected=True)  # add masking option

        # Plotting second level result, TODO: encapsulate in second_level_new
        test_name = "All " + condition1 + " vs All " + condition2
        measure_name = estimators['kind']
        comp = mc.results_[(test_name, measure_name)]
        th = 0.05
        comp.threshold(th)
        test_names = [test_name]
        measure_names = [measure_name]
        ticks = []
        tickLabels = []
        titles = [condition2, condition1, "difference"]
        ylabels = [kind, kind, ""]
        abs_maxs = [None, None, None]
        abs_mins = [None, None, None]
        symmetric_cbs = [True, True, True]
        fig = plt.figure(figsize=(10, 7))
        plt.subplots_adjust(hspace=0.8, wspace=0.5)
        signifs = [comp.signif_follow_up_,
                   comp.signif_baseline_,
                   comp.signif_difference_] #vec_to_sym(comp.signif_difference_)
        matr = zip(titles, signifs)
        for i, (name, this_matr) in enumerate(matr):
            plt.subplot(1, 3, i + 1)
            plot_matrix(this_matr, plt.gca(), title = name, ticks=ticks, tickLabels=tickLabels,
                        abs_max=abs_maxs[i], abs_min=abs_mins[i],
                        symmetric_cb=symmetric_cbs[i], ylabel = ylabels[i])
        plt.draw()  # Draws, but does not block
        # raw_input() # waiting for entry
        fig_title = test_name + "_" + measure_name
        fig_title = fig_title.replace(" ", "_")
        filename = fig_dir + fig_title + ".pdf"
        overwrite = True
    
    plt.show()
    
    
    #    mc.results_fig(th, fig_dir, "overwrite", test_names,
    #                 measure_names)
    #    mc.performances(perfs_file, test)
    #    overwrite = True
    #    mc.analysis_fig = (overwrite,'/home/salma/slides/servier2/Images')


covariances = all_matrices[0]
precisions = all_matrices[1]
tangents = all_matrices[2]
correlations = all_matrices[3]
partials = all_matrices[4]
Z_correlations = corr_to_Z(correlations)
Z_partials = corr_to_Z(partials, tol=1e-9)
Z_correlations[np.isinf(Z_correlations)] = 1.
Z_partials[np.isinf(Z_partials)] = 1.

wmn = range(4)
an = range(5, 9)
dmn = range(11, 16)

wmn_an = range(0,8)
dmn_big = range(9, 16)
wmn_an_2b = [0,1,2,3,5,6,7,8]
# Predict site
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.cross_validation import permutation_test_score, cross_val_score
from sklearn.preprocessing.label import LabelBinarizer
from sklearn import lda
sites = np.repeat(np.arange(5), 8)
for coefs, measure in zip([covariances, precisions, tangents, correlations,
                           partials], measures):
    # Predict task vs rest
    y = np.hstack((np.zeros(n_subjects), np.ones(n_subjects)))
    skf = StratifiedKFold(y, 3)
    regions = range(32)
    X = coefs[:, regions, :]
    X = coefs[:, :, regions]
    X = sym_to_vec(X)
#    X = np.hstack((X, dst[:, np.newaxis]))
    clf_lda = OneVsRestClassifier(lda.LDA())
    score, null_scores, p_val = permutation_test_score(clf_lda, X, y, cv=skf,
                                                       scoring='accuracy')
#    print 'score ADHD: ', score, ', null score is', np.mean(null_scores), 'pval = ', p_val
#    clf_log = OneVsOneClassifier(LinearSVC(random_state=0))
#    score, null_scores, p_val = permutation_test_score(clf_log, X, y, cv=skf,
#                                                       scoring='accuracy')
    print '=========================================='
    print measure
    print 'score lda: ', score, ', null score is', np.mean(null_scores), 'pval = ', p_val
    clf_svc_ovr = LinearSVC(random_state=0, multi_class='ovr')
    score, null_scores, p_val = permutation_test_score(clf_svc_ovr, X, y, cv=skf,
                                                       scoring='accuracy')
    print 'score linear svc: ', score, ', null score is', np.mean(null_scores), 'pval = ', p_val
    clf_svc_ovr = LinearSVC(random_state=0, multi_class='ovr')
    score, null_scores, p_val = permutation_test_score(clf_svc_ovr, X, y, cv=3,
                                                       scoring='accuracy')
    print 'score linear svc, cv=3: ', score, ', null score is', np.mean(null_scores), 'pval = ', p_val
