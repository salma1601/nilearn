"""This example plots boxplots of between conditions and between subjects
distances for
- the geometric and euclidean distances between covariances
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

conditions = ['ReSt1_Placebo', 'Nbac2_Placebo']
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

# Compute between subjects distances for each condition
from nilearn.connectivity2 import analyzing
inter_subjects_gdistances = {}
inter_subjects_edistances = {}
inter_subjects_corr_distances = {}
inter_subjects_part_distances = {}
for condition in conditions:
    inter_subjects_gdistances[condition] = []
    inter_subjects_edistances[condition] = []
    inter_subjects_corr_distances[condition] = []
    inter_subjects_part_distances[condition] = []
    for connectivity_id, connectivity in enumerate(
            covariances[condition]):
        inter_subjects_gdistances[condition] += [analyzing._compute_distance(
            conn, connectivity, distance_type='geometric') for conn in
            covariances[condition][connectivity_id + 1:]]
        inter_subjects_edistances[condition] += [analyzing._compute_distance(
            conn, connectivity, distance_type='euclidean') for conn in
            covariances[condition][connectivity_id + 1:]]
        inter_subjects_corr_distances[condition] += [analyzing._compute_distance(
            conn, connectivity, distance_type='euclidean') for conn in
            correlations[condition][connectivity_id + 1:]]
        inter_subjects_part_distances[condition] += [analyzing._compute_distance(
            conn, connectivity, distance_type='euclidean') for conn in
            partials[condition][connectivity_id + 1:]]

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

# Plot the boxplots
fig = plt.figure(figsize=(5, 5.5))
ax = plt.axes()
plt.hold(True)
colors = ['blue', 'red', 'green']
n_boxes = len(colors)

# geometric
distances = [inter_subjects_gdistances[cond] for cond in conditions] +\
    [intra_subjects_gdistances[cond_pair] for cond_pair in conditions_pairs]

bp = plt.boxplot(distances,
                 positions=range(1, n_boxes + 1),
                 widths=0.6)
set_box_colors(bp, colors)

# correlations
distances = [inter_subjects_corr_distances[cond] for cond in conditions] +\
    [intra_subjects_corr_distances[cond_pair] for cond_pair in conditions_pairs]
bp = plt.boxplot(distances,
                 positions=range(n_boxes + 2, 2 * n_boxes + 2),
                 widths=0.6)
set_box_colors(bp, colors)

# partial correlations
distances = [inter_subjects_part_distances[cond] for cond in conditions] +\
    [intra_subjects_part_distances[cond_pair] for cond_pair in conditions_pairs]
bp = plt.boxplot(distances,
                 positions=range(2 * n_boxes + 3, 3 * n_boxes + 3),
                 widths=0.6)
set_box_colors(bp, colors)

# set axes limits and labels
plt.xlim(0, 3 * (n_boxes + 1))
plt.ylim(1, 8)
ax.set_xticklabels(['geometric\nbetween\ncovariances',
                    'euclidean\n between \ncorrelations',
                    'euclidean\n between \npartial correlations'])
ax.set_xticks([.5 * (1 + n_boxes), .5 * (3 * n_boxes + 3),
               .5 * (5 * n_boxes + 5)])
# draw temporary red and blue lines and use them to create a legend
hB, = plt.plot([1, 1], 'b-')
hR, = plt.plot([1, 1], 'r-')
hG, = plt.plot([1, 1], 'g-')
conditions_names = [cond.replace('_Placebo', '') for cond in conditions]
plt.legend([hB, hR, hG], conditions_names + ['between\nconditions'])
hB.set_visible(False)
hR.set_visible(False)
hG.set_visible(False)
plt.savefig('/home/sb238920/CODE/salma/figures/boxplots.pdf')

# Plot the euclidean distance between covariances boxplots
fig = plt.figure(figsize=(5, 5.5))
ax = plt.axes()
plt.hold(True)
colors = ['blue', 'red', 'green']
n_boxes = len(colors)

distances = [inter_subjects_edistances[cond] for cond in conditions] +\
    [intra_subjects_edistances[cond_pair] for cond_pair in conditions_pairs]

bp = plt.boxplot(distances,
                 positions=range(1, n_boxes + 1),
                 widths=0.6)
set_box_colors(bp, colors)


# set axes limits and labels
plt.xlim(0, n_boxes + 1)
ax.set_xticklabels(['euclidean\nbetween\ncovariances'])
ax.set_xticks([.5 * (1 + n_boxes)])

# draw temporary red and blue lines and use them to create a legend
hB, = plt.plot([1, 1], 'b-')
hR, = plt.plot([1, 1], 'r-')
hG, = plt.plot([1, 1], 'g-')
conditions_names = [cond.replace('_Placebo', '') for cond in conditions]
plt.legend([hB, hR, hG], conditions_names + ['between\nconditions'])
hB.set_visible(False)
hR.set_visible(False)
hG.set_visible(False)
plt.savefig('/home/sb238920/CODE/salma/figures/boxplots_euclidean_covariances.pdf')
plt.show()
