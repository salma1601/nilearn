# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 15:14:17 2015

@author: sb238920
"""
# Standard library imports

# Related third party imports

# Local application/library specific imports
import numpy as np
import matplotlib.pylab as plt


a = 0.4
b = 0.4
t = np.arange(0., 2.0, 0.05)
s = a * t + b  # a + b < 1
alpha = (1 - b) / a
ax = plt.plot(t, s)
t = np.arange(0, alpha, 0.01)
plt.plot(t, np.ones(t.shape), 'r--')
t = np.arange(0, 1, 0.01)
plt.plot(alpha * np.ones(t.shape), t, 'r--')
plt.xticks([0, 1, (1 - b) / a],
           ['0', '1',
            r'$\frac{\sqrt{obs} - \sqrt{neuro}}{\sqrt{noise}}$'],
           fontsize=16)
plt.yticks([0, 1, b],
           ['0', '1',
            r'$\frac{\sqrt{neuro}}{\sqrt{obs}}$'],
           fontsize=16)
#plt.yticks([0, 1], ['0', '1'], fontsize=16)
plt.xlabel(
    r'$\frac{{corr}^{noise}}{{corr}^{neuro}}$', fontsize=20)
plt.ylabel(
    r'$\frac{{corr}^{obs}}{{corr}^{neuro}}$', fontsize=20, rotation=0)
ax = plt.gca()
ax.xaxis.set_label_coords(1.05, -0.025)
ax.yaxis.set_label_coords(-0.1, 0.95)
plt.show()