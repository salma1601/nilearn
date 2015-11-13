"""This example plots scatters of between conditions distances
- the geometric  between covariances
- the euclidean distances between correlations and partial correlations
"""
import matplotlib.pylab as plt


# function for setting the colors of the box plots pairs
def set_box_colors(boxplot, colors=['blue']):
    n_boxplots = len(boxplot['boxes'])
    if len(colors) != n_boxplots:
        raise ValueError('expected a list of {0} colros, given {1}'.format(
            n_boxplots, len(colors)))
    for boxplot_id, color in enumerate(colors):
        plt.setp(boxplot['boxes'][boxplot_id], color=color)
        plt.setp(boxplot['medians'][boxplot_id], color=color)
        caps_ids = range(boxplot_id * 2, boxplot_id * 2 + 2)
        for caps_id in caps_ids:
            plt.setp(boxplot['caps'][caps_id], color=color)
            plt.setp(boxplot['whiskers'][caps_id], color=color)
            plt.setp(boxplot['fliers'][caps_id], color=color)

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

conditions = ['ReSt1_Placebo', 'Nbac2_Placebo', 'Nbac3_Placebo',
              'ReSt2_Placebo']
dataset = dataset_loader.load_conn(conn_folder_no_filt, conditions=conditions,
                                   standardize=False,
                                   networks=networks)
# Estimate connectivity matrices
import nilearn.connectivity
covariances = {}
correlations = {}
partials = {}
# TODO: factorize w.r.t. measures
for condition in conditions:
    subjects = dataset.time_series[condition]
    cov_embedding = nilearn.connectivity.ConnectivityMeasure(
        kind='covariance')
    covariances[condition] = cov_embedding.fit_transform(subjects)
    cov_embedding = nilearn.connectivity.ConnectivityMeasure(
        kind='correlation')
    correlations[condition] = cov_embedding.fit_transform(subjects)
    cov_embedding = nilearn.connectivity.ConnectivityMeasure(
        kind='partial correlation')
    partials[condition] = cov_embedding.fit_transform(subjects)

from nilearn.connectivity2 import analyzing

# Compute between conditions distances
from itertools import combinations
conditions_pairs = list(combinations(conditions, 2))
intra_subjects_gdistances = {}
intra_subjects_edistances = {}
intra_subjects_corr_distances = {}
intra_subjects_part_distances = {}
for (condition1, condition2) in conditions_pairs:
    intra_subjects_gdistances[(condition1, condition2)] = [
        analyzing._compute_distance(conn1, conn2, distance_type='geometric')
        for (conn1, conn2) in
        zip(covariances[condition1], covariances[condition2])]
    intra_subjects_edistances[(condition1, condition2)] = [
        analyzing._compute_distance(conn1, conn2, distance_type='euclidean')
        for (conn1, conn2) in
        zip(covariances[condition1], covariances[condition2])]
    intra_subjects_corr_distances[(condition1, condition2)] = [
        analyzing._compute_distance(conn1, conn2, distance_type='euclidean')
        for (conn1, conn2) in
        zip(correlations[condition1], correlations[condition2])]
    intra_subjects_part_distances[(condition1, condition2)] = [
        analyzing._compute_distance(conn1, conn2, distance_type='euclidean')
        for (conn1, conn2) in
        zip(partials[condition1], partials[condition2])]

# Scatter plot intra-subject distances
# TODO: improve with seaborn
from scipy.stats import pearsonr
distances_pairs = list(combinations([intra_subjects_corr_distances,
                                     intra_subjects_part_distances,
                                     intra_subjects_gdistances], 2))
labels_pairs = list(combinations([
    'euclidean distance between correlations',
    'euclidean distance between partial correlations',
    'geometric distance between covariances'], 2))

import pandas
for (distances_x, distances_y), (x_label, y_label) in zip(distances_pairs,
                                                          labels_pairs):
    plt.figure(figsize=(5, 4))
    x = np.ravel([distances_x[cond_pair] for cond_pair in conditions_pairs])
    y = np.ravel([distances_y[cond_pair] for cond_pair in conditions_pairs])
    plt.scatter(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # Plot line of best fit if significative Pearson correlation
    rho, p_val = pearsonr(x, y)
    if p_val < -0.05 / 3:
        print rho, p_val, x_label, y_label
        fit = np.polyfit(x, y, 1)
        fit_fn = np.poly1d(fit)
        plt.plot(x, fit_fn(x), linestyle='-')
    figure_name = x_label + '_' + y_label
    figure_name = figure_name.replace('euclidean distance between ', '')
    figure_name = figure_name.replace('geometric distance between ', '')
    figure_name = figure_name.replace(' ', '_')
    plt.savefig('/home/sb238920/CODE/salma/figures/scatter_' + figure_name + '.pdf')
if False:
    import seaborn as sns
    sns.set(style="ticks", context="talk")
    data = pandas.DataFrame({x_label:x, y_label:y})
    g = sns.lmplot(x=x_label, y=y_label, data=data, lowess=True)
    plt.savefig('/home/sb238920/CODE/salma/figures/scatter_' + figure_name + '_sns.pdf')

plt.show()
