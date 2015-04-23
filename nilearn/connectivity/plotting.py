import numpy as np
import matplotlib.pylab as plt


def plot_histograms(multiple_coefs, preprocessings, colors=[], title='',
                    xlabel=''):
    """Plot historgram of coefficients, for a given processing"""
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