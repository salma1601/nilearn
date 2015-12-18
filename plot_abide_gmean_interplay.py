"""This example plots the relationship between
correlation - partial correlation and corr(gmean) - partial correlation
"""
import matplotlib.pylab as plt
import numpy as np

# Load preprocessed abide timeseries extracted from harvard oxford atlas
from nilearn import datasets
abide = datasets.fetch_abide_pcp(derivatives=['rois_ho'], DX_GROUP=2)
subjects_unscaled = abide.rois_ho

# Standardize the signals
scaling_type = 'unnormalized'
from nilearn import signal
if scaling_type == 'normalized':
    subjects = []
    for subject in subjects_unscaled:
        subjects.append(signal._standardize(subject))
else:
    subjects = subjects_unscaled

# Estimate connectivity matrices
from sklearn.covariance import LedoitWolf
import nilearn.connectivity
from funtk.connectivity.matrix_stats import cov_to_corr
measures = ['robust dispersion', 'covariance', 'correlation',
            'partial correlation']
figure = plt.figure(figsize=(5, 4.5))
mean_normalized_matrix = {}
mean_connectivity_vector = {}
from joblib import Memory
mem = Memory('/home/sb238920/CODE/Parietal/nilearn/nilearn_cache/abide')
for measure in measures:
    cov_embedding = nilearn.connectivity.ConnectivityMeasure(
        kind=measure, cov_estimator=LedoitWolf())
    connectivities = cov_embedding.fit_transform(subjects, mem=mem)
    if measure == 'robust dispersion':
        mean_matrix = cov_embedding.robust_mean_
    else:
        mean_matrix = connectivities.mean(axis=0)
    mean_normalized_matrix[measure] = mean_matrix
    mean_connectivity_vector[measure] = mean_matrix[
        np.triu(np.ones(mean_matrix.shape), 1).astype(np.bool)]

# Scatter plots
import prettyplotlib as ppl
# This is "import matplotlib.pyplot as plt" from the prettyplotlib library
import matplotlib.pyplot as plt
alpha = .5
figure, ax = plt.subplots(1, 1, figsize=(5, 4.5))
for measure, color, label in zip(['robust dispersion'],
                                 ['b'],
                                 ['geometric mean']):
    x = mean_connectivity_vector['correlation'] - \
        mean_connectivity_vector['partial correlation']
    y = cov_to_corr(mean_normalized_matrix[measure])[
        np.triu(np.ones(mean_matrix.shape), 1).astype(np.bool)] -\
        mean_connectivity_vector['partial correlation']
    ppl.scatter(ax, x, y, color=color, label=label, alpha=alpha)
    q_fit, a_fit, b_fit = np.polyfit(x, y, 2)
    xmin, xmax = ax.get_xlim()
    x_plot = np.linspace(xmin, xmax, 10)
    ppl.plot(ax, x_plot, q_fit * x_plot ** 2 + a_fit * x_plot + b_fit,
             color='k')
    print measure, q_fit, a_fit, b_fit
ax = plt.gca()
xmin, xmax = ax.get_xlim()
plt.plot(np.linspace(xmin, xmax), np.linspace(xmin, xmax), 'k')
plt.xlabel('mean correlations - mean partial correlations')
plt.ylabel('corr(mean) - mean partial correlations')
ppl.plot(ax, (xmax - .002) * np.ones(10), np.linspace(xmin, xmax, 10), 'k')
ppl.plot(ax, np.linspace(xmin, xmax, 10), (xmax - .001) * np.ones(10), 'k')
ax.yaxis.tick_right()
plt.xlim(xmin, xmax)
plt.ylim(xmin, xmax)
plt.legend(loc='lower right')
figure_name = 'abide_{0}_gmean_interplay_scatter.pdf'.format(scaling_type)
#plt.savefig('/home/sb238920/CODE/salma/figures/' + figure_name)

# Other scatters
figure, ax = plt.subplots(1, 1, figsize=(5, 4.5))
for measure, color, label in zip(['robust dispersion'],
                                 ['b'],
                                 ['geometric mean']):
    x2 = mean_connectivity_vector['correlation'] - \
        mean_connectivity_vector['partial correlation']
    y2 = mean_connectivity_vector['correlation'] -\
        cov_to_corr(mean_normalized_matrix[measure])[
            np.triu(np.ones(mean_matrix.shape), 1).astype(np.bool)]
    ppl.scatter(ax, x2, y2, color=color, label=label, alpha=alpha)
    q_fit, a_fit, b_fit = np.polyfit(x2, y2, 2)
    x_plot = np.linspace(xmin, xmax, 10)
    ppl.plot(ax, x_plot, q_fit * x_plot ** 2 + a_fit * x_plot + b_fit,
             color='k')
#    ax.annotate('{0:.2f} x + {1:.2g}'.format(a_fit, b_fit), xytext=(.6, .1),
#                xy=(.6, .1))
    print measure, q_fit, a_fit, b_fit
ax = plt.gca()
xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
plt.xlabel('mean correlations - mean partial correlations')
plt.ylabel('mean correlations - normalized gmean')
ppl.plot(ax, (xmax - .002) * np.ones(10), np.linspace(ymin, ymax, 10), 'k')
ppl.plot(ax, np.linspace(xmin, xmax, 10), (ymax - .0001) * np.ones(10), 'k')
ax.yaxis.tick_right()
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
figure_name = 'abide_{}_gmean_interplay_scatter2.pdf'.format(scaling_type)
#plt.savefig('/home/sb238920/CODE/salma/figures/' + figure_name)

plt.figure(figsize=(5, 4))
x = mean_connectivity_vector['correlation'] - \
    mean_connectivity_vector['partial correlation']
y = mean_connectivity_vector['covariance'] - \
    mean_connectivity_vector['robust dispersion']
plt.scatter(x, y)
ax = plt.gca()
ax.yaxis.tick_right()
plt.xlabel('mean correlations - mean partial correlations')
plt.ylabel('mean covariances - geometric mean')
figure_name = 'abide_{}_gmean_amean_scatters.pdf'.format(scaling_type)
#plt.savefig('/home/sb238920/CODE/salma/figures/')

# Matrices plot
from funtk.connectivity.matrix_stats import plot_matrices
plot_matrices(
    [mean_normalized_matrix['correlation'] -
     cov_to_corr(mean_normalized_matrix['robust dispersion']),
     -mean_normalized_matrix['partial correlation'] +
     cov_to_corr(mean_normalized_matrix['robust dispersion']),
     mean_normalized_matrix['correlation'] -
     mean_normalized_matrix['partial correlation']],
    titles=
    [r'$\overline{\mathrm{correlation}} - \mathrm{corr(gmean)}$',
     r'$\mathrm{corr(gmean)} - \overline{\mathrm{partial}}$',
     r'$\overline{\mathrm{correlation}} - \overline{\mathrm{partial}}$'])
plot_matrices(
    [mean_normalized_matrix['correlation'] -
     cov_to_corr(mean_normalized_matrix['covariance']),
     -mean_normalized_matrix['partial correlation'] +
     cov_to_corr(mean_normalized_matrix['covariance']),
     mean_normalized_matrix['correlation'] -
     mean_normalized_matrix['partial correlation']],
    titles=
    [r'$\overline{\mathrm{correlation}} - \mathrm{corr(amean)}$',
     r'$\mathrm{corr(amean)} - \overline{\mathrm{partial}}$',
     r'$\overline{\mathrm{correlation}} - \overline{\mathrm{partial}}$'])
plot_matrices(
    [mean_normalized_matrix['covariance'],
     mean_normalized_matrix['robust dispersion'],
     mean_normalized_matrix['covariance'] -
     mean_normalized_matrix['robust dispersion']],
    titles=['arithmetic', 'geometric', 'arithmetic - geometric'],
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
