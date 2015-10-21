import sys

import numpy as np
import matplotlib.pylab as plt


code_path = np.genfromtxt('/home/sb238920/CODE/anonymisation/code_path.txt',
                          dtype=str)
sys.path.append(str(code_path))
from my_conn import MyConn


# Specify the location of the  CONN project
conn_folders = np.genfromtxt(
    '/home/sb238920/CODE/anonymisation/conn_projects_paths.txt', dtype=str)
conn_folder_filt = conn_folders[0]
conn_folder_no_filt = conn_folders[1]
mc_filt = MyConn('from_conn', conn_folder_filt)
mc_no_filt = MyConn('from_conn', conn_folder_filt)

# Specify the conditions
condition_baseline = 'ReSt1_Placebo'
condition_follow_up = 'ReSt2_Placebo'

 # Specify the ROIs and their networks. Here: biyu's order
WMN = ['IPL', 'LMFG_peak1',  # 'RMFG_peak2' removed because of bad results
       'RCPL_peak1', 'LCPL_peak3', 'LT']
AN = ['vIPS_big', 'pIPS_big', 'MT_big', 'FEF_big', 'RTPJ', 'RDLPFC']
DMN = ['AG_big', 'SFG_big', 'PCC', 'MPFC', 'FP']
template_ntwk = [('WMN', WMN), ('AN', AN), ('DMN', DMN)]

# Whether to standardize or not the timeseries
standardize = False

# Collect the timeseries
for mc in [mc_no_filt, mc_filt]:
    mc.setup()
    mc.analysis(template_ntwk, standardize, 'correlations')
    if 'Nbac' in condition_baseline:
        signals_baseline = mc_no_filt.runs_[condition_baseline]
    else:
        signals_baseline = mc.runs_[condition_baseline]

    if 'Nbac' in condition_follow_up:
        signals_follow_up = mc_no_filt.runs_[condition_follow_up]
    else:
        signals_follow_up = mc.runs_[condition_follow_up]

n_subjects = 40
from sklearn.covariance import EmpiricalCovariance
mean_matrices = []
all_matrices = []
measures = ["tangent", "correlation", "partial correlation", "covariance",
            "precision"]

subjects = [subj for subj in signals_baseline] + \
           [subj for subj in signals_follow_up]

# Estimate connectivity matrices
import nilearn.connectivity
subjects_connectivity = {}
mean_connectivity = {}
for measure in measures:
    cov_embedding = nilearn.connectivity.CovEmbedding(
        measure=measure, cov_estimator=EmpiricalCovariance())
    subjects_connectivity[measure] = nilearn.connectivity.vec_to_sym(
        cov_embedding.fit_transform(subjects))
    # Compute the mean connectivity across all subjects
    if measure == 'tangent':
        mean_connectivity[measure] = cov_embedding.tangent_mean_
    else:
        mean_connectivity[measure] = \
            subjects_connectivity[measure].mean(axis=0)

# Principal componenets analysis
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
components = {}
for measure in measures:
    components[measure] = []
    matrices = [mean_connectivity[measure]] + \
        subjects_connectivity['covariance']
    vec = nilearn.connectivity.embedding.sym_to_vec(np.array(matrices))
    pca.fit(vec.T)
    components[measure] = pca.components_

# Scatter plot the distances
plt.plot(components['covariance'][0, 1:], components['covariance'][1, 1:],
         '.b')
plt.plot(components['covariance'][0, 0], components['covariance'][1, 0],
         'or')
plt.plot(components['tangent'][0, 1:], components['tangent'][1, 1:],
         '.m')
plt.plot(components['tangent'][0, 0], components['tangent'][1, 0],
         'og')
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.show()
