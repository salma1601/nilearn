import numpy as np
import matplotlib.pylab as plt

import dataset_loader

# Specify the networks
WMN = ['IPL', 'LMFG_peak1',
       'RCPL_peak1', 'LCPL_peak3', 'LT']
AN = ['vIPS.cluster001', 'vIPS.cluster002',
      'pIPS.cluster001', 'pIPS.cluster002',
      'MT.cluster001', 'MT.cluster002',
      'FEF.cluster001', 'FEF.cluster002',
      'RTPJ', 'RDLPFC']
DMN = ['AG.cluster001', 'AG.cluster002',
       'SFG.cluster001', 'SFG.cluster002',
       'PCC', 'MPFC', 'FP']
networks = [('WMN', WMN), ('AN', AN), ('DMN', DMN)]

# Specify the location of the  CONN project
conn_folders = np.genfromtxt(
    '/home/sb238920/CODE/anonymisation/conn_projects_paths.txt', dtype=str)
conn_folder_filt = conn_folders[0]
conn_folder_no_filt = conn_folders[1]

conditions = ['ReSt1_Placebo', 'ReSt2_Placebo']
for conn_folder, preproc in zip(conn_folders, ['filtering', 'no filtering']):
    dataset = dataset_loader.load_conn(conn_folder, conditions=conditions,
                                       standardize=False,
                                       networks=networks)

    # Estimate connectivity matrices
    from sklearn.covariance import EmpiricalCovariance
    import nilearn.connectivity
    n_subjects = 40
    mean_matrices = []
    all_matrices = []
    measures = ["tangent", "correlation", "partial correlation", "covariance",
                "precision"]
    subjects = [subj for condition in conditions for subj in
                dataset.time_series[condition]]
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

    # Plot the mean connectivity
    import nilearn.plotting
    labels, region_coords = zip(*dataset.rois)
    for measure in ["tangent", "correlation", "partial correlation"]:
        nilearn.plotting.plot_connectome(
            mean_connectivity[measure], region_coords, edge_threshold='95%',
            title='mean {0}, {1}'.format(measure, preproc))

plt.show()
