import numpy as np
from scipy import stats
import matplotlib.pylab as plt


# Generate positive numbers from the normal distribution
n_subjects = 40
subjects = np.random.randn(n_subjects)
subjects += np.abs(subjects.min()) + 1.
sorted_subjects = np.sort(subjects)

# Compute the distances to means
amean = np.mean(sorted_subjects)
gmean = stats.gmean(sorted_subjects)
amean_distances = [abs(subject - amean) for subject in sorted_subjects]
gmean_distances = [abs(subject - gmean) for subject in sorted_subjects]

# Plot the bars of distances
plt.bar(np.arange(n_subjects), amean_distances, width=.3, label='amean',
        color='r', alpha=.5)
plt.bar(np.arange(n_subjects), gmean_distances, width=.3, label='gmean',
        color='g', alpha=.5)
plt.legend()
plt.show()
