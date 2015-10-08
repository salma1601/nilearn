import numpy as np
from scipy import stats
import matplotlib.pylab as plt

from nilearn.connectivity2 import analyzing

# Generate positive numbers from the normal distribution
n_subjects = 40
max_outliers = n_subjects / 2
subjects = np.random.randn(n_subjects)
subjects += np.abs(subjects.min()) + 1.
sorted_subjects = np.sort(subjects)
low_subjects = sorted_subjects[:max_outliers]  # stable subjects
high_subjects = sorted_subjects[max_outliers:][::-1]  # outliers, from the worst

# Compute the means
ameans = [np.mean(np.hstack((low_subjects, high_subjects[:n_outliers]))) for
          n_outliers in range(max_outliers)]
gmeans = [stats.gmean(np.hstack((low_subjects, high_subjects[:n_outliers])))
          for n_outliers in range(max_outliers)]

# Compute the errors in means
ameans_error = [amean - ameans[0] for amean in ameans]
gmeans_error = [gmean - gmeans[0] for gmean in gmeans]

# Plot the errors
plt.plot(ameans_error, label='amean')
plt.plot(gmeans_error, label='gmean')
plt.show()
