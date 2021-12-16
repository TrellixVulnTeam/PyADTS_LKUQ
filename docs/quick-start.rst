***********
Quick Start
***********

``PyADTS`` provides a "full-stack" framework to build time-series anomaly detection workflows, such as .

=================
Loading Datasets
=================

``PyADTS`` offers various datasets

.. code-block:: python

    from pyadts.datasets import NABDataset

    data = NABDataset(root='data/nab', subset='realAWSCloudwatch', download=True)


==============
Preprocessing
==============

.. code-block:: python

    from pyadts.preprocessing import train_test_split


=================
Model Definition
=================

=========
Training
=========

===========
Evaluation
===========

====================
Ensemble (optional)
====================

=============================
Score Calibration (optional)
=============================
