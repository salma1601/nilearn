"""This example plots boxplots of between conditions and between subjects
distances for
- the geometric and euclidean distances between covariances
- the euclidean distances between correlations and partial correlations
"""
import bredala
bredala.USE_PROFILER = True
#bredala.register('nilearn.connectivity',
#                 names=['ConnectivityMeasure.fit_transform'])
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

conditions = ['ReSt1_Placebo', 'ReSt2_Placebo']
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
inter_subjects_corr_gdistances = {}
inter_subjects_part_distances = {}
labels = {}
my_parts = {}
my_correlations = {}
colors_list = 'rgbkycm'
from itertools import cycle
for condition in conditions:
    distances = analyzing.compute_pairwise_distances(
        covariances[condition], distance_type='euclidean')
    inter_subjects_edistances[condition] = distances[
        np.triu(np.ones((40, 40)), 1).astype(np.bool)]
    distances = analyzing.compute_pairwise_distances(
        covariances[condition], distance_type='geometric')
    inter_subjects_gdistances[condition] = distances[
        np.triu(np.ones((40, 40)), 1).astype(np.bool)]
    distances = analyzing.compute_pairwise_distances(
        correlations[condition], distance_type='euclidean')
    inter_subjects_corr_distances[condition] = distances[
        np.triu(np.ones((40, 40)), 1).astype(np.bool)]
    distances = analyzing.compute_pairwise_distances(
        correlations[condition], distance_type='geometric')
    inter_subjects_corr_gdistances[condition] = distances[
        np.triu(np.ones((40, 40)), 1).astype(np.bool)]
    distances = analyzing.compute_pairwise_distances(
        partials[condition], distance_type='euclidean')
    inter_subjects_part_distances[condition] = distances[
        np.triu(np.ones((40, 40)), 1).astype(np.bool)]    
    labels = []
    colors_scatter = []
    cycle_color = cycle(colors_list)
    # TODO: horrible, use np.triu_indices
    for connectivity_id, connectivity in enumerate(
            covariances[condition]):
        u = cycle_color.next()
        labels += [str(connectivity_id) + ',' + str(connectivity_id + 1 + n)
                   for n, conn in enumerate(partials[condition][connectivity_id + 1:])]
        colors_scatter += [u for conn in 
                   partials[condition][connectivity_id + 1:]]

# Compute between conditions distances
from itertools import combinations
conditions_pairs = list(combinations(conditions, 2))
intra_subjects_gdistances = {}
intra_subjects_edistances = {}
intra_subjects_corr_distances = {}
intra_subjects_corr_gdistances = {}
intra_subjects_part_distances = {}
for (condition1, condition2) in conditions_pairs:
    intra_subjects_edistances[(condition1, condition2)] = [
        analyzing._compute_distance(conn1, conn2, distance_type='euclidean')
        for (conn1, conn2) in
        zip(covariances[condition1], covariances[condition2])]
    intra_subjects_gdistances[(condition1, condition2)] = [
        analyzing._compute_distance(conn1, conn2, distance_type='geometric')
        for (conn1, conn2) in
        zip(covariances[condition1], covariances[condition2])]
    intra_subjects_corr_distances[(condition1, condition2)] = [
        analyzing._compute_distance(conn1, conn2, distance_type='euclidean')
        for (conn1, conn2) in
        zip(correlations[condition1], correlations[condition2])]
    intra_subjects_corr_gdistances[(condition1, condition2)] = [
        analyzing._compute_distance(conn1, conn2, distance_type='geometric')
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
medians = []
boxplots_positions = []
sym = '+'
# euclidean
distances = [inter_subjects_edistances[cond] for cond in conditions] +\
    [intra_subjects_edistances[cond_pair] for cond_pair in conditions_pairs]
medians += [np.median(distance) for distance in distances]
positions = range(1, n_boxes + 1)
boxplots_positions += positions
bp = plt.boxplot(distances,
                 positions=positions,
                 widths=0.6,
                 sym=sym)
set_box_colors(bp, colors)

# geometric
distances = [inter_subjects_gdistances[cond] for cond in conditions] +\
    [intra_subjects_gdistances[cond_pair] for cond_pair in conditions_pairs]
medians += [np.median(distance) for distance in distances]
positions = range(n_boxes + 3, 2 * n_boxes + 3)
boxplots_positions += positions
bp = plt.boxplot(distances,
                 positions=positions,
                 widths=0.6,
                 sym=sym)
set_box_colors(bp, colors)

# correlations
distances = [inter_subjects_corr_distances[cond] for cond in conditions] +\
    [intra_subjects_corr_distances[cond_pair] for cond_pair in conditions_pairs]
medians += [np.median(distance) for distance in distances]
positions = range(2 * n_boxes + 5, 3 * n_boxes + 5)
boxplots_positions += positions
bp = plt.boxplot(distances,
                 positions=positions,
                 widths=0.6,
                 sym=sym)
set_box_colors(bp, colors)

# geometric between correlations
distances = [inter_subjects_corr_gdistances[cond] for cond in conditions] +\
    [intra_subjects_corr_gdistances[cond_pair] for cond_pair in conditions_pairs]
medians += [np.median(distance) for distance in distances]
positions = range(3 * n_boxes + 7, 4 * n_boxes + 7)
boxplots_positions += positions
bp = plt.boxplot(distances,
                 positions=positions,
                 widths=0.6,
                 sym=sym)
set_box_colors(bp, colors)

# Due to the Y-axis scale being different across samples, it can be
# hard to compare differences in medians across the samples. Add upper
# X-axis tick labels with the sample medians to aid in comparison
# (just use two decimal places of precision)
total_n_boxes = n_boxes * 4
top = 13
upperLabels = [str(np.round(s, 2)) for s in medians]
from itertools import cycle
boxColors = cycle(['b', 'r', 'g'])
weights = ['bold', 'semibold']
ypositions = cycle([.95 * top, .9 * top, .95 * top])
for tick in range(total_n_boxes):
    k = tick % 2
    ax.text(boxplots_positions[tick], .95 * top, upperLabels[tick],
            horizontalalignment='center', size=9, weight=weights[k],
            color=boxColors.next(), rotation=45)

# set axes limits and labels
plt.xlim(0, 4 * (n_boxes + 1) + 4)
plt.ylim(0, top)
ax.set_xticklabels(['', 'euclidean\nbetween\ncovariances', '',
                    '', 'geometric\nbetween\ncovariances', '',
                    '', 'euclidean\n between \ncorrelations', '',
                    '', 'geometric\n between \ncorrelations', ''])
#ax.set_xticks([.5 * (1 + n_boxes), .5 * (3 * n_boxes + 5),
#               .5 * (5 * n_boxes + 9), .5 * (7 * n_boxes + 13)])
ax.set_xticks(boxplots_positions)

# draw temporary red and blue lines and use them to create a legend
hB, = plt.plot([1, 1], 'b-')
hR, = plt.plot([1, 1], 'r-')
hG, = plt.plot([1, 1], 'g-')
conditions_names = [cond.replace('_Placebo', '') for cond in conditions]
plt.legend([hB, hR, hG], conditions_names + ['between\nconditions'],
           loc='center left')
hB.set_visible(False)
hR.set_visible(False)
hG.set_visible(False)
plt.savefig('/home/sb238920/CODE/salma/figures/cov_corr_4_boxplots.pdf')

# Plot the euclidean distance between partial boxplots
fig = plt.figure(figsize=(5, 5.5))
ax = plt.axes()
plt.hold(True)
colors = ['blue', 'red', 'green']
n_boxes = len(colors)

distances = [inter_subjects_part_distances[cond] for cond in conditions] +\
    [intra_subjects_part_distances[cond_pair] for cond_pair in conditions_pairs]

bp = plt.boxplot(distances,
                 positions=range(1, n_boxes + 1),
                 widths=0.6)
set_box_colors(bp, colors)


# set axes limits and labels
plt.xlim(0, n_boxes + 1)
ax.set_xticklabels(['euclidean\nbetween\npartial correlations'])
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
plt.savefig('/home/sb238920/CODE/salma/figures/boxplots_euclidean_partials.pdf')
plt.show()

# Scatter plot inter-subject distances
condition = conditions[0]
plt.figure()
from itertools import cycle
#colors = cycle(['r', 'g', 'b', 'm', 'y'])
plt.scatter(inter_subjects_corr_distances[condition],
            inter_subjects_part_distances[condition])
if False:
    for label, x, y in zip(labels, inter_subjects_corr_distances[condition],
                           inter_subjects_part_distances[condition]):
        plt.annotate(
            label, 
            xy = (x, y), xytext = (-20, 20),
            textcoords = 'offset points', ha = 'right', va = 'bottom',
            bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
    for color, x, y in zip(colors_scatter,
                           inter_subjects_corr_distances[condition],
                           inter_subjects_part_distances[condition]):
        plt.plot(x, y, 'o' + color)
plt.xlabel('euclidean distance between correlations')
plt.ylabel('euclidean distance between partial correlations')
plt.savefig('/home/sb238920/CODE/salma/figures/scatter_euclidean_rs1.pdf')
plt.figure()
plt.scatter(inter_subjects_corr_distances[condition],
            inter_subjects_gdistances[condition])
plt.xlabel('euclidean distance between correlations')
plt.ylabel('geometric distance between covariances')
plt.savefig('/home/sb238920/CODE/salma/figures/scatter_geo_corr_rs1.pdf')
plt.figure()
plt.scatter(inter_subjects_part_distances[condition],
            inter_subjects_gdistances[condition])
plt.xlabel('euclidean distance between partial correlations')
plt.ylabel('geometric distance between covariances')
plt.savefig('/home/sb238920/CODE/salma/figures/scatter_geo_part_rs1.pdf')
plt.show()