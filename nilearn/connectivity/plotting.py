import numpy as np
import matplotlib.pylab as plt


def plot_histograms(multiple_coefs, preprocessings, colors=[], title='',
                    xlabel=''):
    """Row subplot of multiple historgrams"""
    n_plots = len(multiple_coefs)
    axes = None
    if len(colors) < n_plots:
        colors = ['b' for n in range(n_plots)]

    for i, preproc in enumerate(preprocessings):
        coefs = multiple_coefs[i].copy()
        if colors:
            color = colors[i]

        # Plot the histogram
        axes = plt.subplot(n_plots, 1, i + 1, sharex=axes, sharey=axes)
        bins = coefs.shape[0] / 10
        plt.hist(coefs, bins=bins, normed=1, color=color, alpha=0.4)
        plt.ylabel(preproc)
        if i == 0:
            plt.title(title)

    plt.xlabel(xlabel)
    plt.show()


def plot_matrix(mean_conn, title="connectivity", ticks=[], tick_labels=[],
                xlabel="", ylabel="", zero_diag=True):
    """Plot connectivity matrix, for a given measure. """

    mean_conn = mean_conn.copy()

    # Put zeros on the diagonal, for graph clarity
    if zero_diag:
        size = mean_conn.shape[0]
        mean_conn[range(size), range(size)] = 0

    vmax = np.abs(mean_conn).max()
    if vmax <= 2e-16:
        vmax = 0.1

    # Display connectivity matrix
    plt.figure()
    plt.imshow(mean_conn, interpolation="nearest",
               vmin=-vmax, vmax=vmax, cmap=plt.cm.get_cmap("bwr"))
    plt.colorbar()
    ax = plt.gca()
#    ax.xaxis.set_ticks_position('top')
    plt.xticks(ticks, tick_labels, size=8, rotation=90)
    plt.xlabel(xlabel)
    plt.yticks(ticks, tick_labels, size=8)
    ax.yaxis.tick_left()
    plt.ylabel(ylabel)
    plt.title(title)


# Build a classification task using 3 informative features
X, y = make_classification(n_samples=1000,
                           n_features=10,
                           n_informative=3,
                           n_redundant=0,
                           n_repeated=0,
                           n_classes=2,
                           random_state=0,
                           shuffle=False)


# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(10):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(10), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(10), indices)
plt.xlim([-1, 10])
plt.show()