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
high_subjects = sorted_subjects[max_outliers:]  # outliers

# Compute the means
amean_no_outlier = low_subjects.mean()
ameans_errors = {}
gmeans_errors = {}
average_ameans_error = []
for n_outlier in max_outliers:
    subjects_combinations = [np.hstack((low_subjects, outlier)) for
                             outlier in high_subjects]
    ameans = [subjects_combination.mean() for subjects_combination in
              subjects_combinations]
    ameans_errors[n_outlier] = [amean - amean_no_outlier for amean in ameans]
    average_ameans_error.append(np.mean(
        [amean - amean_no_outlier for amean in ameans]))

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
