.. for doctests to run, we need to define variables that are define in
   the literal includes
    >>> import numpy as np
    >>> from sklearn import datasets
    >>> iris = datasets.load_iris()
    >>> fmri_masked  = iris.data
    >>> target = iris.target
    >>> session = np.ones_like(target)
    >>> n_samples = len(target)

.. _decoding_tutorial:

=====================
A decoding tutorial
=====================

This page is a decoding tutorial articulated on the analysis of the Haxby
2001 dataset. It shows how to predict from brain activity images the
stimuli that the subject is viewing.


.. contents:: **Contents**
    :local:
    :depth: 1


The isochromats summation model
================================

Reference to Yarnick.

.. figure:: ../images/initial_spins.png
   :target: ../spoiling.html
   :scale: 30
   :align: left

   Magnetization is the average of the magnetizations of N spins completeley relaxed.
   Spins have uniformly distributed phases \psi_1=0, \psi_2=2pi/N, ..., \psi_N=2 pi.

.. figure:: ../images/first_rf.png
   :target: ../spoiling.html
   :scale: 30
   :align: left

   First RF pulse with phase 0° and flip angle 20°: magnetization of each spin is rotated around the z-axis by 20°.

.. figure:: ../images/precession.png
   :target: ../spoling.html
   :scale: 30
   :align: left

   Precession in the transverse plane of each spin by its proper phase angle.

.. figure:: ../images/relaxation.png
   :target: ../spoiling.html
   :scale: 35
   :align: left

   Longitudinal and transverse relaxation with the time constants T1 and T2 during delay TR.


.. figure:: ../images/second_spins.png
   :target: ../spoiling.html
   :scale: 30
   :align: left

   Second RF pulse with phase \Delta \Phi and same flip angle.


.. figure:: ../images/second_spins.png
   :target: ../spoiling.html
   :scale: 30
   :align: left

   The procdeure is repeated with varying pulse phases. The phase difference between subsequent RF pulses increases linearly by a constant phase increment \Delta \Phi.

.. figure:: ../images/transverse_signals.png
   :target: ../spoiling.html
   :scale: 30
   :align: left


   The procedure stops when a steady state is reached. Note the gap to the steady-state signal of an ideal spoiling.

.. figure:: ../images/increments_on_transverse.png
   :target: ../spoiling.html
   :scale: 30
   :align: left

   The gap depends on the chosen phase increment \Delta \Phi.


Getting the real T1
===================

