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

conditions = ['ReSt1_Placebo', 'Nbac2_Placebo', 'Nbac3_Placebo',
              'ReSt2_Placebo']
dataset = dataset_loader.load_conn(conn_folder_no_filt, conditions=conditions,
                                   standardize=False,
                                   networks=networks)
# Estimate connectivity matrices
import nilearn.connectivity
connectivities = {}
measures = ['covariance', 'correlation', 'partial correlation']
# TODO: factorize w.r.t. measures
for condition in conditions:
    connectivities[condition] = {}
    for measure in measures:
        subjects = dataset.time_series[condition]
        cov_embedding = nilearn.connectivity.ConnectivityMeasure(
            kind=measure)
        connectivities[condition][measure] = cov_embedding.fit_transform(
            subjects)

# Compute between subjects distances for each condition
from nilearn.connectivity2 import analyzing
inter_subjects_gdistances = {}
inter_subjects_edistances = {}
for condition in conditions:
    inter_subjects_edistances[condition] = {}
    inter_subjects_gdistances[condition] = {}
    for measure in measures:
        edistances = analyzing.compute_pairwise_distances(
            connectivities[condition][measure], distance_type='euclidean')
        inter_subjects_edistances[condition][measure] = edistances[
            np.triu(np.ones((40, 40)), 1).astype(np.bool)]
        if measure != 'partial correlation':  # partial is not spd
            gdistances = analyzing.compute_pairwise_distances(
                connectivities[condition][measure], distance_type='geometric')
            inter_subjects_gdistances[condition][measure] = gdistances[
                np.triu(np.ones((40, 40)), 1).astype(np.bool)]

# Compute between conditions distances
from itertools import combinations
conditions_pairs = list(combinations(conditions, 2))
intra_subjects_gdistances = {}
intra_subjects_edistances = {}
for (condition1, condition2) in conditions_pairs:
    intra_subjects_gdistances[(condition1, condition2)] = {}
    intra_subjects_edistances[(condition1, condition2)] = {}
    for measure in measures:
        intra_subjects_edistances[(condition1, condition2)][measure] = [
            analyzing._compute_distance(c1, c2, distance_type='euclidean')
            for (c1, c2) in zip(connectivities[condition1][measure],
                                connectivities[condition2][measure])]
        if measure != 'partial correlation':  # partial is not spd
            intra_subjects_gdistances[(condition1, condition2)][measure] = [
                analyzing._compute_distance(c1, c2, distance_type='geometric')
                for (c1, c2) in zip(connectivities[condition1][measure],
                                    connectivities[condition2][measure])]

# Plot the boxplots
fig = plt.figure(figsize=(5, 5.5))
ax1 = plt.axes()
plt.hold(True)
# To lighten boxplots, plot only one intrasubject condition
pair_to_plot = ('ReSt1_Placebo', 'ReSt2_Placebo')
conds_to_plot = ['ReSt1_Placebo', 'Nbac3_Placebo', 'ReSt2_Placebo']
n_boxes = len(conds_to_plot) + 1
colors = ['blue', 'red', 'green', 'magenta', 'cyan'][:n_boxes]
n_spaces = 7
sym = '+'
widths = .6
start_position = 1
xticks = []
xticks_labels = []
data = []
all_positions = []
for measure in ['covariance', 'correlation']:
    edistances_to_plot = [
        inter_subjects_edistances[cond][measure] for cond in conds_to_plot] +\
        [intra_subjects_edistances[pair_to_plot][measure]]
    gdistances_to_plot = [
        inter_subjects_gdistances[cond][measure] for cond in conds_to_plot] +\
        [intra_subjects_gdistances[pair_to_plot][measure]]
    if measure == 'covariance':
        all_distances = [edistances_to_plot, gdistances_to_plot]
    else:
        all_distances = [edistances_to_plot]
    for (distance_type, distances_to_plot) in zip(['Euclidean', 'geometric'],
                                                  all_distances):
        positions = range(start_position, start_position + n_boxes)
        all_positions += positions
        xticks.append(.5 * positions[0] + .5 * positions[-1])
        xticks_labels.append(distance_type + '\nbetween\n' + measure + 's')
        # Plot euclidian between covariances on a seperate axis
        # TODO: same axis but zoom
        if measure == 'covariance' and distance_type == 'Euclidean':
            ax = ax1  #.twinx()
        else:
            ax = ax1
        data += distances_to_plot
        bp = ax.boxplot(distances_to_plot,
                        positions=positions,
                        widths=widths,
                        sym=sym)
        set_box_colors(bp, colors)
        start_position += n_boxes + n_spaces
# set axes limits and labels
plt.xlim(0, start_position - n_spaces)

ax.set_xticklabels(xticks_labels)
ax.set_xticks(xticks)
ax.yaxis.tick_right()

conditions_names = []
for condition in conds_to_plot:
    condition_name = condition.replace('_Placebo', '')
    condition_name = condition_name.replace('ReSt', 'rest ')
    condition_name = condition_name.replace('Nbac2', '2-back')
    condition_name = condition_name.replace('Nbac3', '3-back')
    conditions_names.append(condition_name)
markers = [color[0] + '-' for color in colors]
# draw temporary red and blue lines and use them to create a legend
lines = [plt.plot([1, 1], marker, label=label)[0] for marker, label in 
         zip(markers, conditions_names + ['between\nconditions'])]
# TODO: split legend
#plt.legend(lines, conditions_names + ['between\nconditions'],
#           loc='lower right')
p5, = plt.plot([0], marker='None',
           linestyle='None', label='dummy-tophead')
p7, = plt.plot([0],  marker='None',
           linestyle='None', label='dummy-empty')

leg3 = plt.legend([p5, lines[-1], p7, p5] + [p5] + lines[:-1],
              ['intra-subjects', 'between rest 1\nand rest 2', '', ''] + ['inter-subjects'] + conditions_names,
              loc='lower right', ncol=2, prop={'size':10}) # Two columns, vertical group labels

#leg4 = plt.legend([p5, p7, p5, p7, lines[-1]] + lines[:-1],
#              ['intra-subject', '', 'inter-subjects', ''] + ['between\nconditions'] + conditions_names,
#              loc=2, ncol=2) # Two columns, horizontal group labels

#plt.gca().add_artist(leg3)
for line in lines:
    line.set_visible(False)


# Create the zoomed axes
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
zoom = 1
#axins = zoomed_inset_axes(ax1, zoom, loc=6) # zoom = 3, location = 1 (upper right)
axins = inset_axes(ax1, 1., zoom, loc=6) # zoom = 3, location = 1 (upper right)
all_positions = np.array(all_positions, dtype=float)
#all_positions[1] = 1 + 1. /float(zoom)
#all_positions[2] = 2 + 1./ float(zoom)
bp = axins.boxplot(data, widths=widths, sym=sym,
                   positions=all_positions)
set_box_colors(bp, colors + colors + colors)

# sub region of the original image
x1, x2, y1, y2 = 0.5, 4.5, .1, .75
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
plt.xticks(visible=False)
plt.yticks(visible=True)
axins.yaxis.tick_right()

# draw a bbox of the region of the inset axes in the parent axes and
# connecting lines between the bbox and the inset axes area
mark_inset(ax, axins, loc1=4, loc2=4, fc="none", ec="0.9")

plt.savefig('/home/sb238920/CODE/salma/figures/cov_corr_{}_boxplots.pdf'
            .format(n_boxes))

# Scatter plots
alpha = .5
plt.subplots(1, 2, sharey=True, figsize=(5, 4))
for color, condition, condition_name in zip(colors, conds_to_plot,
                                            conditions_names):
    plt.scatter(inter_subjects_edistances[condition]['correlation'],
                inter_subjects_edistances[condition]['partial correlation'],
                c=color, label=condition_name, alpha=alpha)
plt.scatter(intra_subjects_edistances[pair_to_plot]['correlation'],
            intra_subjects_edistances[pair_to_plot]['partial correlation'],
            c=colors[-1], label='between conditions', alpha=alpha)
plt.xlabel('euclidean distance between correlations')
plt.ylabel('euclidean distance between partial correlations')
plt.legend()
plt.savefig('/home/sb238920/CODE/salma/figures/scatter_euclidean_{}conds.'
            'pdf'.format(n_boxes - 1))
plt.figure(figsize=(5, 4))
for color, condition, condition_name in zip(colors, conds_to_plot,
                                            conditions_names):
    plt.scatter(inter_subjects_edistances[condition]['correlation'],
                inter_subjects_gdistances[condition]['covariance'],
                c=color, label=condition_name, alpha=alpha)
plt.scatter(intra_subjects_edistances[pair_to_plot]['correlation'],
            intra_subjects_edistances[pair_to_plot]['covariance'],
            c=colors[-1], label='between conditions', alpha=alpha)
plt.xlabel('euclidean distance between correlations')
plt.ylabel('geometric distance between covariances')
plt.legend()
plt.savefig('/home/sb238920/CODE/salma/figures/scatter_geo_corr_{}conds.'
            'pdf'.format(n_boxes - 1))
plt.figure(figsize=(5, 4))
for color, condition, condition_name in zip(colors, conds_to_plot,
                                            conditions_names):
    plt.scatter(inter_subjects_edistances[condition]['partial correlation'],
                inter_subjects_gdistances[condition]['covariance'],
                c=color, label=condition_name, alpha=alpha)
plt.scatter(intra_subjects_edistances[pair_to_plot]['partial correlation'],
            intra_subjects_edistances[pair_to_plot]['covariance'],
            c=colors[-1], label='between conditions', alpha=alpha)
plt.xlabel('euclidean distance between partial correlations')
plt.ylabel('geometric distance between covariances')
plt.legend()
plt.savefig('/home/sb238920/CODE/salma/figures/scatter_geo_part_{}conds.'
            'pdf'.format(n_boxes - 1))
plt.show()