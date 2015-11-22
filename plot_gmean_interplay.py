"""This example plots the relationship between 
correlation - partial correlation and corr(gmean) - partial correlation
"""
import matplotlib.pylab as plt


# Specify the networks
WMN = ['IPL', 'LMFG_peak1',
       'RCPL_peak1', 'LCPL_peak3', 'LT']
AN = ['vIPS.cluster001', 'vIPS.cluster002',
      'pIPS.cluster001', 'pIPS.cluster002',
      'MT.cluster001', 'MT.cluster002',
      'FEF.cluster002', 'FEF.cluster001']
DMN = ['RTPJ', 'RDLPFC', 'AG.cluster001', 'AG.cluster002',
       'SFG.cluster001', 'SFG.cluster002',
       'PCC', 'MPFC', 'FP']
networks = [('WMN', WMN), ('AN', AN), ('DMN', DMN)]

# Specify the location of the  CONN project
import numpy as np
import dataset_loader
conn_folders = np.genfromtxt(
    '/home/sb238920/CODE/anonymisation/conn_projects_paths.txt', dtype=str)
conn_folder_filt = conn_folders[0]
conn_folder_no_filt = conn_folders[1]

condition = 'ReSt1_Placebo'
conditions = ['ReSt1_Placebo']
dataset = dataset_loader.load_conn(conn_folder_no_filt, conditions=conditions,
                                   standardize=False,
                                   networks=networks)
# Estimate connectivity matrices
import nilearn.connectivity
from funtk.connectivity.matrix_stats import cov_to_corr
measures = ['robust dispersion', 'covariance', 'correlation', 'partial correlation']
figure = plt.figure(figsize=(5, 4.5))
mean_normalized_matrix = {}
mean_connectivity_vector = {}
for measure in measures:
    subjects = dataset.time_series[condition]
    cov_embedding = nilearn.connectivity.ConnectivityMeasure(
        kind=measure)
    connectivities = cov_embedding.fit_transform(subjects)
    if measure == 'robust dispersion':
        mean_matrix = cov_embedding.robust_mean_
    else:
        mean_matrix = connectivities.mean(axis=0)
    mean_normalized_matrix[measure] = mean_matrix
    mean_connectivity_vector[measure] = mean_matrix[
        np.triu(np.ones(mean_matrix.shape), 1).astype(np.bool)]

# Scatter plots
for measure, color, label in zip(['covariance', 'robust dispersion'],
                                 ['r', 'b'],
                                 ['arithmetic mean', 'geometric mean']):
    x = mean_connectivity_vector['correlation'] - \
        mean_connectivity_vector['partial correlation']
    y = cov_to_corr(mean_normalized_matrix[measure])[
            np.triu(np.ones(mean_matrix.shape), 1).astype(np.bool)] -\
        mean_connectivity_vector['partial correlation']
    plt.scatter(x, y, c=color, label=label)
    a_fit, b_fit = np.polyfit(x, y ,1)
    print measure, a_fit, b_fit
    #plt.plot(x, a_fit * x + b_fit)
ax = plt.gca()
ax.yaxis.tick_right()
plt.xlabel('mean correlations - mean partial correlations')
plt.ylabel('corr(mean) - mean partial correlations')
plt.legend(loc='lower right')
plt.savefig('/home/sb238920/CODE/salma/figures/gmean_interplay_scatter.pdf')


# Other scatters
plt.figure(figsize=(5, 4.5))
for measure, color, label in zip(['covariance', 'robust dispersion'],
                                 ['r', 'b'],
                                 ['arithmetic mean', 'geometric mean']):
    x2 = mean_connectivity_vector['correlation'] - \
    mean_connectivity_vector['partial correlation']
    y2 = mean_connectivity_vector['correlation'] -\
        cov_to_corr(mean_normalized_matrix[measure])[
            np.triu(np.ones(mean_matrix.shape), 1).astype(np.bool)]       
    plt.scatter(x2, y2, c=color, label=label)
    a_fit, b_fit = np.polyfit(x2, y2 ,1)
    print measure, a_fit, b_fit
ax = plt.gca()
ax.yaxis.tick_right()
plt.legend(loc='lower left')
plt.xlabel('mean correlations - mean partial correlations')
plt.ylabel('mean correlations - corr(mean)')
plt.savefig('/home/sb238920/CODE/salma/figures/gmean_interplay_scatter2.pdf')


plt.figure(figsize=(5, 4))
x = mean_connectivity_vector['correlation'] - mean_connectivity_vector['partial correlation']
y = mean_connectivity_vector['covariance'] - mean_connectivity_vector['robust dispersion']
plt.scatter(x, y)
ax = plt.gca()
ax.yaxis.tick_right()
plt.xlabel('mean correlations - mean partial correlations')
plt.ylabel('mean covariances - geometric mean')
plt.savefig('/home/sb238920/CODE/salma/figures/gmean_amean_scatters.pdf')




# Matrices plot
from funtk.connectivity.matrix_stats import plot_matrices
plot_matrices([mean_normalized_matrix['correlation'] - \
   cov_to_corr( mean_normalized_matrix['robust dispersion']),
   -mean_normalized_matrix['partial correlation'] + \
   cov_to_corr( mean_normalized_matrix['robust dispersion']),
    mean_normalized_matrix['correlation'] - \
    mean_normalized_matrix['partial correlation']],
    titles=[r'$\overline{\mathrm{correlation}} - \mathrm{corr(gmean)}$',
            r'$\mathrm{corr(gmean)} - \overline{\mathrm{partial}}$',
            r'$\overline{\mathrm{correlation}} - \overline{\mathrm{partial}}$'])

plot_matrices([mean_normalized_matrix['correlation'] - \
   cov_to_corr( mean_normalized_matrix['covariance']),
   -mean_normalized_matrix['partial correlation'] + \
   cov_to_corr(mean_normalized_matrix['covariance']),
    mean_normalized_matrix['correlation'] - \
    mean_normalized_matrix['partial correlation']],
    titles=[r'$\overline{\mathrm{correlation}} - \mathrm{corr(amean)}$',
            r'$\mathrm{corr(amean)} - \overline{\mathrm{partial}}$',
            r'$\overline{\mathrm{correlation}} - \overline{\mathrm{partial}}$'])


plot_matrices([mean_normalized_matrix['covariance'],
               mean_normalized_matrix['robust dispersion'],
               mean_normalized_matrix['covariance'] -
               mean_normalized_matrix['robust dispersion']],
               titles=['arithmetic',
                       'geometric',
                       'arithmetic - geometric'],
               font_size=8)

plot_matrices([cov_to_corr(mean_normalized_matrix['covariance']),
              cov_to_corr(mean_normalized_matrix['robust dispersion']),
              cov_to_corr(mean_normalized_matrix['covariance']) -
              cov_to_corr(mean_normalized_matrix['robust dispersion'])],
               titles=['corr(amean)',
                       'corr(gmean)',
                       'corr(amean) - corr(gmean)'],
               font_size=8)

plt.show()