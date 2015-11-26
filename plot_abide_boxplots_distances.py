"""This example plots boxplots of between conditions and between subjects
distances for
- the geometric and euclidean distances between covariances
- the euclidean distances between correlations and partial correlations
"""
import numpy as np
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

# Load preprocessed abide timeseries extracted from harvard oxford atlas
from nilearn import datasets
time_series = {}
time_series['control'] = datasets.fetch_abide_pcp(derivatives=['rois_ho'],
                                                  DX_GROUP=2).rois_ho
time_series['autist'] = datasets.fetch_abide_pcp(derivatives=['rois_ho'],
                                                 DX_GROUP=1).rois_ho
conditions = ['control', 'autist']

# Estimate connectivity matrices
import nilearn.connectivity
from sklearn.covariance import LedoitWolf
connectivities = {}
measures = ['covariance', 'correlation', 'partial correlation']
# TODO: factorize w.r.t. measures
from joblib import Memory
mem = Memory('/home/sb238920/CODE/Parietal/nilearn/nilearn_cache/abide')
for condition in conditions:
    connectivities[condition] = {}
    for measure in measures:
        subjects = time_series[condition]
        cov_embedding = nilearn.connectivity.ConnectivityMeasure(
            kind=measure, cov_estimator=LedoitWolf())
        connectivities[condition][measure] = cov_embedding.fit_transform(
            subjects, mem=mem)

# Compute between subjects distances for each condition
from nilearn.connectivity2 import analyzing
inter_subjects_gdistances = {}
inter_subjects_edistances = {}
for condition in conditions:
    inter_subjects_edistances[condition] = {}
    inter_subjects_gdistances[condition] = {}
    n_subjects = len(connectivities[condition][measure])
    for measure in measures:
        edistances = mem.cache(analyzing.compute_pairwise_distances)(
            connectivities[condition][measure], distance_type='euclidean')
        inter_subjects_edistances[condition][measure] = edistances[
            np.triu(np.ones((n_subjects, n_subjects)), 1).astype(np.bool)]
        if measure != 'partial correlation':  # partial is not spd
            gdistances = mem.cache(analyzing.compute_pairwise_distances)(
                connectivities[condition][measure], distance_type='geometric')
            inter_subjects_gdistances[condition][measure] = gdistances[
                np.triu(np.ones((n_subjects, n_subjects)), 1).astype(np.bool)]

# Compute between conditions distances
from itertools import combinations
conditions_pairs = list(combinations(conditions, 2))
intra_subjects_gdistances = {}
intra_subjects_edistances = {}
for (condition1, condition2) in conditions_pairs:
    intra_subjects_gdistances[(condition1, condition2)] = {}
    intra_subjects_edistances[(condition1, condition2)] = {}
    for measure in measures:
        print measure, 'euclidean distances'
        intra_subjects_edistances[(condition1, condition2)][measure] = [
            mem.cache(analyzing._compute_distance)(c1, c2, distance_type='euclidean')
            for c1 in connectivities[condition1][measure] for c2 in
            connectivities[condition2][measure]]
        if measure != 'partial correlation':  # partial is not spd
            print measure, 'geometric distances'
            intra_subjects_gdistances[(condition1, condition2)][measure] = [
                mem.cache(analyzing._compute_distance)(c1, c2, distance_type='geometric')
                for c1 in connectivities[condition1][measure] for c2 in
                connectivities[condition2][measure]]

# Plot the boxplots
fig = plt.figure(figsize=(5.2, 5.5))
ax1 = plt.axes()
plt.hold(True)
# To lighten boxplots, plot only one intrasubject condition
pair_to_plot = ('control', 'autist')
conds_to_plot = ['control', 'autist']
n_boxes = len(conds_to_plot) + 1
colors = ['blue', 'red', 'green', 'magenta', 'cyan'][:n_boxes]
n_spaces = 4
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
    all_distances = [edistances_to_plot, gdistances_to_plot]
    for (distance_type, distances_to_plot) in zip(['Euclidean', 'geometric'],
                                                  all_distances):
        positions = range(start_position, start_position + n_boxes)
        all_positions += positions
        xticks.append(.5 * positions[0] + .5 * positions[-1])
        xticks_labels.append(distance_type + '\nbetween\n' + measure + 's')
        # Plot euclidian between covariances on a seperate axis
        # TODO: same axis but zoom
        data += distances_to_plot
        if measure == 'covariance' and distance_type == 'Euclidean':
            import matplotlib.ticker as mtick
            ax = ax1.twinx()
            ax.yaxis.tick_left()
            ax.minorticks_on()
#            ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
            ax.set_xlabel('distances between covariances', x=1, ha='right')
            offset = ax.get_xaxis().get_offset_text()
            ax.set_xlabel('{0} {1}'.format(ax.get_xlabel(), offset.get_text()))
            offset.set_visible(False)
            ax.set_ylim(0, 1400000)
        else:
            ax = ax1
            ax.yaxis.tick_right()
        bp = ax.boxplot(distances_to_plot,
                        positions=positions,
                        widths=widths,
                        sym=sym)
        set_box_colors(bp, colors)
        start_position += n_boxes + n_spaces

# set axes limits and labels
plt.xlim(0, start_position - n_spaces)
ax.set_ylim(0, 80)
ax.set_xticklabels(xticks_labels)
ax.set_xticks(xticks)
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
p5, = plt.plot([0], marker='None', linestyle='None', label='dummy-tophead')
p7, = plt.plot([0],  marker='None', linestyle='None', label='dummy-empty')
plt.legend([p5] + lines[:-1] + [p7, lines[-1]],
           ['intra-classes'] + conditions_names +
           ['inter-classes', 'between controls\nand autists'],
           loc='upper center',
           ncol=2, prop={'size': 12})  # vertical group labels
for line in lines:
    line.set_visible(False)

fig.suptitle('Pairwise distances between subjects')
plt.savefig('/home/sb238920/CODE/salma/figures/abide_cov_corr_{}_boxplots.pdf'
            .format(n_boxes))

# Scatter plots
import prettyplotlib as ppl

# This is "import matplotlib.pyplot as plt" from the prettyplotlib library
import matplotlib.pyplot as plt
alpha = .5
figure, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(5, 8))
for color, condition, condition_name in zip(colors, conds_to_plot,
                                            conditions_names):
    ppl.scatter(ax1,
                inter_subjects_gdistances[condition]['covariance'],
                inter_subjects_edistances[condition]['correlation'],
                color=color)
ppl.scatter(ax1, intra_subjects_gdistances[pair_to_plot]['covariance'],
            intra_subjects_edistances[pair_to_plot]['correlation'],
            color=colors[-1])
p7, = ppl.plot([0],  marker='None', linestyle='None', label='inter-subjects')

for n_tasks_to_plot in range(len(conditions_names) - 2):
    ppl.plot([0],  marker='None', linestyle='None', label=' ')

for color, condition, condition_name in zip(colors, conds_to_plot,
                                            conditions_names):
    ppl.scatter(ax2,
                inter_subjects_gdistances[condition]['covariance'],
                inter_subjects_edistances[condition]['partial correlation'],
                color=color, label=condition_name)
p5, = ppl.plot([0], marker='None', linestyle='None', label='intra-subjects')
ppl.scatter(ax2,
            intra_subjects_gdistances[pair_to_plot]['covariance'],
            intra_subjects_edistances[pair_to_plot]['partial correlation'],
            color=colors[-1], label='between rest 1\nand rest 2')
ax1.set_ylabel('euclidean distance between correlations')
ax2.set_ylabel('euclidean distance between partial correlations')
ax2.set_xlabel('geometric distance between covariances')

# Create grouped legend
handles, labels = ax2.get_legend_handles_labels()


def sorting_function(tup):
    label, _ = tup
    if label == 'inter-subjects':
        order = 'a'
    elif label == 'rest 1':
        order = 'b'
    elif label == '2-back':
        order = 'c'
    elif label == '3-back':
        order = 'd'
    elif label == 'rest 2':
        order = 'e'
    elif label == 'intra-subjects':
        order = 'f'
    elif label == 'between rest 1\nand rest 2':
        order = 'g'
    elif label == ' ':
        order = 'h'
    else:
        order = label[0]
    return order

labels, handles = zip(*sorted(zip(labels, handles),
                              key=lambda t: sorting_function(t)))
plt.legend(handles, labels, loc='lower center',
           ncol=2, prop={'size': 10})  # vertical group labels
plt.xlim(10, 25)
plt.savefig('/home/sb238920/CODE/salma/figures/abide_scatter_geo_{}conds.'
            'pdf'.format(n_boxes - 1))
figure, ax = plt.subplots(1, 1, figsize=(5, 4))
scatters = []
for color, condition, condition_name in zip(colors, conds_to_plot,
                                            conditions_names):
    scatters.append(plt.scatter(
        inter_subjects_edistances[condition]['correlation'],
        inter_subjects_edistances[condition]['partial correlation'],
        c=color, label=condition_name, alpha=alpha))
scatters.append(plt.scatter(intra_subjects_edistances[pair_to_plot]['correlation'],
            intra_subjects_edistances[pair_to_plot]['partial correlation'],
            c=colors[-1], label='between conditions', alpha=alpha))
plt.xlabel('euclidean distance between correlations')
plt.ylabel('euclidean distance between partial correlations')
# Create grouped legend
p5, = plt.plot([0], marker='None', linestyle='None', label='dummy-tophead')
p7, = plt.plot([0],  marker='None', linestyle='None', label='dummy-empty')
plt.legend([p5, scatters[-1], p7, p5, p5] + scatters[:-1],
           ['intra-subjects', 'between rest 1\nand rest 2', '', '',
            'inter-subjects'] + conditions_names,
           loc='lower right',
           ncol=2, prop={'size': 10})  # vertical group labels
plt.savefig('/home/sb238920/CODE/salma/figures/abide_scatter_corr_part_{}conds.'
            'pdf'.format(n_boxes - 1))
plt.show()