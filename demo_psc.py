# -*- coding: utf-8 -*-from nilearn import (input_data, datasets, signal)
import numpy as np
import matplotlib.pylab as plt
import nilearn.signal_tmp as my_signal
from nilearn import input_data, datasets, signal


dataset = datasets.fetch_adhd(n_subjects=1)
atlas = datasets.fetch_msdl_atlas()
confound_filename = dataset.confounds[0]
motion = []
for col in range(2):
    confounds = np.genfromtxt(confound_filename, delimiter='\t', names=True,
                              usecols=(col))
    motion.append(np.array([float(conf[0]) for conf in confounds]))
motion = np.array(motion).transpose()
motion_standardized = signal._standardize(motion[:, np.newaxis],
                                          normalize=True, detrend=False)
motion_psc = motion
masker = input_data.NiftiMapsMasker(atlas["maps"], resampling_target="maps",
                                    standardize=False, detrend=False)
masker.fit_transform(dataset['func'][0])
region_ts = masker.fit_transform(dataset['func'][0])  # raw signal
low_pass = None
high_pass = None
confounds = motion
cleaned = signal.clean(region_ts, detrend=True, standardize=False,
                       confounds=confounds, high_pass=high_pass,
                       low_pass=low_pass)
assume_confounds_centered = (np.mean(motion) < 1e-7)
psc_cleaned = my_signal.clean_psc(region_ts.copy(), detrend=False, psc=True,
                                  confounds=confounds,
                                  assume_confounds_centered=assume_confounds_centered,
                                  high_pass=high_pass, low_pass=low_pass)

raw_cleaned = my_signal.clean_psc(region_ts.copy(), detrend=False, psc=False,
                                  confounds=confounds,
                                  assume_confounds_centered=assume_confounds_centered,
                                  high_pass=high_pass, low_pass=low_pass)

regions = [12, 13]
plt.subplot(5, 1, 1)
plt.plot(region_ts[:, 12], label='region 12')
plt.plot(region_ts[:, 13], label='region 13')
plt.legend()
plt.ylabel('raw')
plt.subplot(5, 1, 2)
psc = 100. * region_ts / region_ts.mean(axis=0)
plt.plot(psc[:, regions])
plt.ylabel('PSC')
plt.subplot(5, 1, 3)
plt.plot(cleaned[:, regions])
plt.ylabel('signal.clean')
plt.subplot(5, 1, 4)
plt.plot(psc_cleaned[:, regions])
plt.ylabel('cleaned PSC')
plt.subplot(5, 1, 5)
plt.plot(raw_cleaned[:, regions])
plt.ylabel('cleaned raw')
plt.show()

plt.subplot(3, 1, 1)
plt.plot(cleaned[:, :5])
plt.ylabel('signal.clean')
plt.subplot(3, 1, 2)
plt.plot(psc_cleaned[:, :5])
plt.ylabel('cleaned PSC')
plt.subplot(3, 1, 3)
plt.plot(raw_cleaned[:, :5])
plt.ylabel('cleaned raw')
plt.show()
