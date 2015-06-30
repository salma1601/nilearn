##########################################################
# Means w.r.t. distances/connectivity
##########################################################
# Load the signals for the resting state sessions
all_subjects = []
for nip, session in zip(nips, sessions):
    all_subjects.append(np.load(os.path.join(
        '/neurospin/servier2/salma/nilearn_outputs', 'low_motion', 'spheres',
        'optimal', nip + '_' + revert_acquisition(session) + '_' + 'rs1.npy')))
for nip, session in zip(nips, sessions):
    all_subjects.append(np.load(os.path.join(
        '/neurospin/servier2/salma/nilearn_outputs', 'high_motion', 'spheres',
        'optimal', nip + '_' + session + '_' + 'rs1.npy')))

n_subjects = len(all_subjects)

# Compute connectivity coefficients for each subject
print("-- Measuring connecivity ...")
measures = ['covariance', 'precision', 'tangent', 'correlation',
            'partial correlation']

from sklearn.covariance import EmpiricalCovariance, LedoitWolf, MinCovDet
estimators = [('ledoit', LedoitWolf()), ('emp', EmpiricalCovariance()),
              ('mcd', MinCovDet())]
n_estimator = 1
all_matrices = []
mean_matrices = []
from nilearn import connectivity
for subjects in [all_subjects[:n_subjects / 2],
                 all_subjects[n_subjects / 2:]]:
    for measure in measures:
        estimator = {'cov_estimator': estimators[n_estimator][1],
                     'kind': measure}
        cov_embedding = connectivity.CovEmbedding(**estimator)
        matrices = connectivity.vec_to_sym(
            cov_embedding.fit_transform(subjects))
        all_matrices.append(matrices)
        if measure == 'tangent':
            mean = cov_embedding.mean_cov_
        else:
            mean = matrices.mean(axis=0)
        mean_matrices.append(mean)
coords = all_seeds
coords = np.array([list(coord) for coord in coords])
distance_matrix = coords - coords[:, np.newaxis]
dist = np.linalg.norm(distance_matrix, axis=-1)
plt.subplot(411)
plt.scatter((mean_matrices[3][dist < 96]).flatten(),
            (dist[dist < 96]).flatten(), c='r')
plt.scatter((cov_to_corr(mean_matrices[2])[dist < 96]).flatten(),
            (dist[dist < 96]).flatten(), c='g')
plt.subplot(412)
plt.scatter((mean_matrices[3][dist > 96]).flatten(),
            (dist[dist > 96]).flatten(), c='r')
plt.scatter((cov_to_corr(mean_matrices[2])[dist > 96]).flatten(),
            (dist[dist > 96]).flatten(), c='g')
plt.subplot(413)
plt.scatter((mean_matrices[3]).flatten(),
            (mean_matrices[8]).flatten(), c='r')
plt.scatter((cov_to_corr(mean_matrices[2])).flatten(),
            (cov_to_corr(mean_matrices[7])).flatten(), c='g')
plt.plot(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
plt.xlabel('conn')
plt.ylabel('dist')
plt.show()


# Correlation between motion and connectivity, relationship with distance
#########################################################################
# Load the signals for the resting state sessions
subjects = []
for nip, session in zip(nips, sessions):
    subjects.append(np.load(os.path.join(
        '/neurospin/servier2/salma/nilearn_outputs', out_folder, 'spheres',
        'optimal', nip + '_' + session + '_' + 'rs1.npy')))
for nip, session in zip(nips, sessions):
    subjects.append(np.load(os.path.join(
        '/neurospin/servier2/salma/nilearn_outputs', out_folder, 'spheres',
        'optimal', nip + '_' + session + '_' + 'rs1.npy')))

n_subjects = len(subjects)
# Compute connectivity coefficients for each subject
print("-- Measuring connecivity ...")
measures = ['covariance', 'precision', 'tangent', 'correlation',
            'partial correlation']

from sklearn.covariance import EmpiricalCovariance, LedoitWolf, MinCovDet
estimators = [('ledoit', LedoitWolf()), ('emp', EmpiricalCovariance()),
              ('mcd', MinCovDet())]
n_estimator = 1
all_matrices = []
mean_matrices = []
from nilearn import connectivity
for measure in measures:
    estimator = {'cov_estimator': estimators[n_estimator][1],
                 'kind': measure}
    cov_embedding = connectivity.CovEmbedding(**estimator)
    matrices = connectivity.vec_to_sym(
        cov_embedding.fit_transform(subjects))
    all_matrices.append(matrices)
    if measure == 'tangent':
        mean = cov_embedding.mean_cov_
    else:
        mean = matrices.mean(axis=0)
    mean_matrices.append(mean)

# Compute mean motion for each subject
mean_motion = []
for prefix in ['rs1', 'rs1']:
    for folder in folders:
        motion_filename = single_glob(os.path.join(folder,
                                                   'rp_a' + prefix + '*.txt'))
        motion_confounds = np.genfromtxt(motion_filename)[:, :3]
        relative_motion = motion_confounds.copy()
        relative_motion[1:] -= motion_confounds[:-1]
        mean_motion.append(np.linalg.norm(relative_motion, axis=1).mean())

# Compute Euclidean distances between nodes in mm
coords = all_seeds
coords = np.array([list(coord) for coord in coords])
distance_matrix = coords - coords[:, np.newaxis]
distance_matrix = np.linalg.norm(distance_matrix, axis=-1)

# Compute pearson correlation between motion and connectivity
correlation = np.zeros(distance_matrix.shape)
all_indices = np.triu_indices(distance_matrix.shape[0], 1)
x_indices = []
y_indices = []
for indices in zip(*all_indices):
    conn = []
    for n in range(n_subjects):
        conn.append(all_matrices[3][n][indices])
    if np.mean(conn) > -1:
        correlation[indices] = pearsonr(mean_motion, conn)[0]
        x_indices.append(indices[0])
        y_indices.append(indices[1])

new_indices = (x_indices, y_indices)

# Scatter plot
dist = distance_matrix[new_indices]
corr = correlation[new_indices]
plt.scatter(dist, corr)
plt.xlabel('euclidean distance (mm)')
plt.ylabel('correlation of motion and connectivity')
r, p = pearsonr(dist, corr)
print('Pearson correlation is {0} with pval {1}'.format(r, p))
t = np.polyfit(dist, corr, 1, full=True)
xp = np.linspace(dist.min(), dist.max(), 100)
p1 = np.poly1d(np.polyfit(dist, corr, 1))
plt.plot(xp, p1(xp))
print('corr = {0} (dist - {1})'.format(t[0][1], - t[0][0] / t[0][1]))
plt.show()

plot_matrix(all_matrices[3][40:].mean(axis=0) -
            all_matrices[3][:40].mean(axis=0))
plt.show()

