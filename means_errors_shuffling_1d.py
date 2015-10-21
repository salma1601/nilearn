import numpy as np
from scipy import stats
import matplotlib.pylab as plt

from nilearn.connectivity2 import analyzing

# Generate positive numbers from the normal distribution
n_subjects = 40
subjects = np.random.randn(n_subjects)
subjects += np.abs(subjects.min()) + 1.
sorted_subjects = np.sort(subjects)
n_subjects = len(subjects)
n_inliers = n_subjects / 2
max_outliers = n_subjects - n_inliers
low_subjects = sorted_subjects[:n_inliers]  # stable subjects
high_subjects = sorted_subjects[n_inliers:]  # outliers

# Compute the means
amean_no_outlier = low_subjects.mean()
average_ameans_error = []
std_ameans_error = []
average_gmeans_error = []
std_gmeans_error = []
max_combinations = 300
import itertools
rand_gen = np.random.RandomState(seed=0)
for n_outlier in range(max_outliers):
    # Get tuples of outliers of length n_outlier
    outliers_combinations = itertools.combinations(high_subjects, n_outlier)
    subjects_combinations = [np.hstack((low_subjects, np.array((outliers))))
                             for outliers in outliers_combinations]
    # Randomly select a given number of combinations
    indices = rand_gen.randint(0, len(subjects_combinations), max_combinations)
    print(n_outlier, len(subjects_combinations), len(indices))
    subjects_combinations = np.array(subjects_combinations)[indices]
    ameans = [subjects_combination.mean() for subjects_combination in
              subjects_combinations]
    ameans_errors = [amean - amean_no_outlier for amean in ameans]
    average_ameans_error.append(np.mean(ameans_errors))
    std_ameans_error.append(np.std(ameans_errors))

    gmeans = [stats.gmean(subjects_combination) for subjects_combination in
              subjects_combinations]
    gmeans_errors = [gmean - amean_no_outlier for gmean in gmeans]
    average_gmeans_error.append(np.mean(gmeans_errors))
    std_gmeans_error.append(np.std(gmeans_errors))

# Plot the errors
plt.plot(average_ameans_error, label='amean')
plt.plot(average_gmeans_error, label='gmean')
plt.xlabel('number of outliers')
plt.title('difference between average and stable average')
plt.legend()
plt.show()
